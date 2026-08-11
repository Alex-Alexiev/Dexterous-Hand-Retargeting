import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dex_retargeting.robot_wrapper import RobotWrapper
from retargeting.custom_hand_retargeting_fn import retarget_hand_to_robot_joints
from utils.dataset.dataset import YCB_CLASSES
from utils.dataset.grasp_phase_utils import compute_grasp_phase_and_object_selection
from utils.manopth.geometry import compute_hand_geometry, compute_joint_trajectory_with_frames
from utils.manopth.mano_layer import MANOLayer
from utils.system.path_utils import resolve_repo_path
from utils.robot.mano_root_alignment import (
    compute_aligned_robot_root_pose,
    compute_hand_root_frame,
    compute_mano_frame_rotations,
    compute_mapped_mano_points,
    load_mano_link_mapping,
)
from utils.robot.urdf_utils import build_temp_urdf, resolve_visual_urdf
from utils.viser.scene_utils import make_transform, quat_xyzw_to_wxyz


@dataclass(frozen=True)
class RetargetedGraspDatasetConfig:
    dataset_name: str = "retargeted_grasps_test"
    dexycb_dir: str = "assets/datasets/dexycb"
    output_dir: str = "assets/dataset/retargeted_grasps_test"
    retarget_config_path: Optional[str] = None
    robot_name: str = "robotis_5f_hand"
    hand_type: str = "right"
    urdf_path: Optional[str] = None
    mapping_path: Optional[str] = None
    trajectory_fraction: float = 1.0
    max_trajectories: int = 20
    seed: int = 0
    trajectory_sample_spacing_m: float = 0.01
    inflection_stride: int = 4
    approach_offset_m: float = 0.05
    lift_offset_m: float = 0.05
    refine_window_m: float = 0.10
    refine_resample_spacing_m: Optional[float] = None


@dataclass(frozen=True)
class RetargetingResources:
    urdf_path: Path
    mapping_path: Path
    temp_urdf_path: Path
    robot_joint_names: Tuple[str, ...]
    optimizer_robot_model: RobotWrapper
    mapping_robot_model: RobotWrapper
    link_indices_by_mano: np.ndarray
    offsets_by_mano: np.ndarray


_PHASE_TO_INDEX = {
    "approach": 0,
    "grasp": 1,
    "lift": 2,
}


def _rotation_matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_generation_config(config_path: str, repo_root: Path) -> RetargetedGraspDatasetConfig:
    path = resolve_repo_path(config_path, repo_root)
    with path.expanduser().open("r") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict config in {path}, got {type(payload)}.")
    return RetargetedGraspDatasetConfig(**payload)


def _resolve_optional_config_path(
    path_value: Optional[str],
    *,
    base_dir: Path,
    repo_root: Path,
) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        base_dir / path,
        repo_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve()


def _load_retarget_config_paths(
    retarget_config_path: str,
    *,
    repo_root: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    config_file = resolve_repo_path(retarget_config_path, repo_root).expanduser().resolve()
    with config_file.open("r") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        return None, None

    custom_cfg = payload.get("custom_hand_retargeting")
    if not isinstance(custom_cfg, dict):
        return None, None

    urdf_path = _resolve_optional_config_path(
        custom_cfg.get("urdf_path"),
        base_dir=config_file.parent,
        repo_root=repo_root,
    )
    if urdf_path is None:
        nested = custom_cfg.get("retargeting")
        if isinstance(nested, dict):
            urdf_path = _resolve_optional_config_path(
                nested.get("urdf_path"),
                base_dir=config_file.parent,
                repo_root=repo_root,
            )

    mapping_path = _resolve_optional_config_path(
        custom_cfg.get("mapping_path"),
        base_dir=config_file.parent,
        repo_root=repo_root,
    )
    return urdf_path, mapping_path


def resolve_generation_paths(
    config: RetargetedGraspDatasetConfig,
    repo_root: Path,
) -> Tuple[Path, Path, Path, Path]:
    dexycb_dir = resolve_repo_path(config.dexycb_dir, repo_root)
    output_dir = resolve_repo_path(config.output_dir, repo_root)
    retarget_urdf_path = None
    retarget_mapping_path = None
    if config.retarget_config_path is not None:
        retarget_urdf_path, retarget_mapping_path = _load_retarget_config_paths(
            config.retarget_config_path,
            repo_root=repo_root,
        )

    default_urdf_path = (
        repo_root / "assets" / "robots" / config.robot_name / f"{config.robot_name}.urdf"
    )
    default_mapping_path = (
        repo_root
        / "assets"
        / "robots"
        / config.robot_name
        / f"robotis_5f_{config.hand_type}_mano_link_mapping.json"
    )
    urdf_path = (
        resolve_repo_path(config.urdf_path, repo_root)
        if config.urdf_path is not None
        else (retarget_urdf_path if retarget_urdf_path is not None else default_urdf_path)
    )
    mapping_path = (
        resolve_repo_path(config.mapping_path, repo_root)
        if config.mapping_path is not None
        else (
            retarget_mapping_path
            if retarget_mapping_path is not None
            else default_mapping_path
        )
    )
    return dexycb_dir, output_dir, urdf_path, mapping_path


def build_retargeting_resources(
    urdf_path: Path,
    mapping_path: Path,
) -> RetargetingResources:
    visual_urdf_path = resolve_visual_urdf(urdf_path)
    temp_urdf_path = build_temp_urdf(visual_urdf_path, add_dummy_free_joints=False)
    optimizer_robot_model = RobotWrapper(str(temp_urdf_path))
    mapping_robot_model = RobotWrapper(str(temp_urdf_path))
    link_indices_by_mano, offsets_by_mano = load_mano_link_mapping(
        mapping_path, mapping_robot_model
    )
    return RetargetingResources(
        urdf_path=urdf_path,
        mapping_path=mapping_path,
        temp_urdf_path=temp_urdf_path,
        robot_joint_names=tuple(optimizer_robot_model.dof_joint_names),
        optimizer_robot_model=optimizer_robot_model,
        mapping_robot_model=mapping_robot_model,
        link_indices_by_mano=link_indices_by_mano.astype(np.int32),
        offsets_by_mano=offsets_by_mano.astype(np.float32),
    )


def select_trajectory_indices(
    num_trajectories: int,
    config: RetargetedGraspDatasetConfig,
) -> np.ndarray:
    if num_trajectories <= 0:
        return np.zeros((0,), dtype=np.int32)
    fraction_count = int(np.ceil(float(config.trajectory_fraction) * num_trajectories))
    target_count = max(1, min(num_trajectories, fraction_count, int(config.max_trajectories)))
    rng = np.random.default_rng(config.seed)
    return rng.permutation(num_trajectories)[:target_count].astype(np.int32)


def _object_pose_world(
    object_pose_frame: np.ndarray,
    *,
    object_index: int,
    camera_transform: np.ndarray,
) -> np.ndarray:
    return (
        camera_transform
        @ make_transform(
            object_pose_frame[object_index, 4:],
            quat_xyzw_to_wxyz(object_pose_frame[object_index, :4]),
        )
    ).astype(np.float32)


def _robot_mano_root_pose(
    *,
    hand_joints_world: np.ndarray,
    robot_qpos: np.ndarray,
    resources: RetargetingResources,
) -> Tuple[np.ndarray, np.ndarray]:
    robot_root_pos_world, robot_root_rot_world = compute_aligned_robot_root_pose(
        hand_joints_world=hand_joints_world,
        robot_model=resources.optimizer_robot_model,
        robot_qpos=robot_qpos,
        link_indices_by_mano=resources.link_indices_by_mano,
        offsets_by_mano=resources.offsets_by_mano,
        hand_to_robot_translation=np.zeros((3,), dtype=np.float32),
    )
    resources.optimizer_robot_model.compute_forward_kinematics(robot_qpos.astype(np.float64))
    mapped_points_local, valid_mask = compute_mapped_mano_points(
        resources.optimizer_robot_model,
        resources.link_indices_by_mano,
        resources.offsets_by_mano,
    )
    mapped_rotations_local = compute_mano_frame_rotations(mapped_points_local, valid_mask)
    robot_mano0_pos_world = (
        robot_root_pos_world.astype(np.float64)
        + robot_root_rot_world.astype(np.float64) @ mapped_points_local[0]
    )
    robot_mano0_rot_world = (
        robot_root_rot_world.astype(np.float64) @ mapped_rotations_local[0]
    )
    return (
        robot_mano0_pos_world.astype(np.float32),
        robot_mano0_rot_world.astype(np.float32),
    )


def build_retargeted_phase_datapoints(
    *,
    sample: Dict[str, Any],
    data_id: int,
    config: RetargetedGraspDatasetConfig,
    resources: RetargetingResources,
) -> List[Dict[str, Any]]:
    mano_layer = MANOLayer(config.hand_type, np.asarray(sample["hand_shape"], dtype=np.float32))
    camera_transform = np.linalg.inv(np.asarray(sample["extrinsics"], dtype=np.float32)).astype(
        np.float32
    )
    joint_trajectory, joint_frame_ids = compute_joint_trajectory_with_frames(
        np.asarray(sample["hand_pose"], dtype=np.float32),
        mano_layer,
        camera_transform,
    )
    if joint_trajectory.shape[0] < 3:
        raise ValueError("Need at least 3 valid MANO frames to build retargeted datapoints.")

    wrist_trajectory = joint_trajectory[:, 0, :].astype(np.float32)
    selection = compute_grasp_phase_and_object_selection(
        wrist_trajectory,
        hand_joint_trajectory=joint_trajectory,
        object_pose_seq=np.asarray(sample["object_pose"], dtype=np.float32),
        object_mesh_files=sample["object_mesh_file"],
        camera_transform=camera_transform,
        frame_coords=joint_frame_ids.astype(np.float32),
        resample_spacing_m=config.trajectory_sample_spacing_m,
        inflection_stride=config.inflection_stride,
        approach_offset_m=config.approach_offset_m,
        lift_offset_m=config.lift_offset_m,
        refine_window_m=config.refine_window_m,
        refine_resample_spacing_m=config.refine_resample_spacing_m,
    )

    phase_frames = selection.phase_diagnostics.phase_frames
    phase_items = [
        ("approach", phase_frames.approach_frame),
        ("grasp", phase_frames.grasp_frame),
        ("lift", phase_frames.lift_frame),
    ]
    selected_object_index = int(selection.grasped_object.object_index)
    selected_object_id = int(sample["ycb_ids"][selected_object_index])
    selected_object_name = YCB_CLASSES.get(selected_object_id, f"obj_{selected_object_id}")
    selected_object_mesh_file = str(sample["object_mesh_file"][selected_object_index])

    datapoints: List[Dict[str, Any]] = []
    for phase_name, frame_id in phase_items:
        hand_pose_frame = np.asarray(sample["hand_pose"][frame_id], dtype=np.float32)
        hand_vertices, hand_joints = compute_hand_geometry(
            hand_pose_frame,
            mano_layer,
            camera_transform,
        )
        if hand_vertices is None or hand_joints is None:
            raise ValueError(f"Phase frame {frame_id} for {phase_name} is invalid.")

        human_root_translation, human_root_rotation = compute_hand_root_frame(hand_joints)
        robot_qpos = retarget_hand_to_robot_joints(
            hand_keypoints=hand_joints.astype(np.float32),
            hand_joint_values=hand_pose_frame[:, :48].astype(np.float32),
            robot_joint_names=list(resources.robot_joint_names),
            robot_model=resources.optimizer_robot_model,
            mapping_link_indices=resources.link_indices_by_mano,
            mapping_offsets=resources.offsets_by_mano,
        ).astype(np.float32)
        robot_root_translation, robot_root_rotation = _robot_mano_root_pose(
            hand_joints_world=hand_joints,
            robot_qpos=robot_qpos,
            resources=resources,
        )
        object_transform = _object_pose_world(
            np.asarray(sample["object_pose"][frame_id], dtype=np.float32),
            object_index=selected_object_index,
            camera_transform=camera_transform,
        )

        datapoints.append(
            {
                "trajectory_id": int(data_id),
                "capture_name": str(sample["capture_name"]),
                "frame_id": int(frame_id),
                "phase": phase_name,
                "phase_index": int(_PHASE_TO_INDEX[phase_name]),
                "human_mano": hand_pose_frame[0].astype(np.float32),
                "human_shape": np.asarray(sample["hand_shape"], dtype=np.float32),
                "camera_transform": camera_transform.astype(np.float32),
                "human_mano_root_translation": np.asarray(
                    human_root_translation, dtype=np.float32
                ),
                "human_mano_root_orientation_wxyz": _rotation_matrix_to_quat_wxyz(
                    human_root_rotation
                ),
                "retargeted_qpos": robot_qpos.astype(np.float32),
                "retargeted_mano_root_translation": robot_root_translation.astype(np.float32),
                "retargeted_mano_root_orientation_wxyz": _rotation_matrix_to_quat_wxyz(
                    robot_root_rotation
                ),
                "object_identifier": int(selected_object_id),
                "object_name": selected_object_name,
                "object_mesh_file": selected_object_mesh_file,
                "object_translation": object_transform[:3, 3].astype(np.float32),
                "object_orientation_wxyz": _rotation_matrix_to_quat_wxyz(
                    object_transform[:3, :3]
                ),
            }
        )
    return datapoints


def save_retargeted_grasp_dataset(
    output_dir: Path,
    *,
    config: RetargetedGraspDatasetConfig,
    resources: RetargetingResources,
    datapoints: Sequence[Dict[str, Any]],
    skipped_trajectories: Sequence[Dict[str, Any]],
    num_selected_trajectories: int,
    num_processed_trajectories: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "data.npz"
    arrays = {
        key: np.asarray([datapoint[key] for datapoint in datapoints])
        for key in datapoints[0].keys()
    } if len(datapoints) > 0 else {}
    np.savez_compressed(data_file, **arrays)

    metadata = {
        "dataset_name": config.dataset_name,
        "num_selected_trajectories": int(num_selected_trajectories),
        "num_processed_trajectories": int(num_processed_trajectories),
        "num_datapoints": int(len(datapoints)),
        "robot_name": config.robot_name,
        "hand_type": config.hand_type,
        "retarget_config_path": config.retarget_config_path,
        "robot_joint_names": list(resources.robot_joint_names),
        "urdf_path": str(resources.urdf_path),
        "mapping_path": str(resources.mapping_path),
        "skipped_trajectories": list(skipped_trajectories),
        "field_conventions": {
            "human_mano": "MANO pose+translation frame data with shape (51,)",
            "human_mano_root_orientation_wxyz": "Quaternion in wxyz order",
            "retargeted_mano_root_orientation_wxyz": "Quaternion in wxyz order",
            "object_orientation_wxyz": "Quaternion in wxyz order",
        },
    }
    metadata_path = output_dir / "metadata.json"
    _ensure_parent(metadata_path)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    config_path = output_dir / "config_resolved.yaml"
    _ensure_parent(config_path)
    with config_path.open("w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
    return data_file


def load_retargeted_grasp_dataset(dataset_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    with np.load(dataset_dir / "data.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    with (dataset_dir / "metadata.json").open("r") as f:
        metadata = json.load(f)
    return arrays, metadata
