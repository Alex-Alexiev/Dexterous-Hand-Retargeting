import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tyro
import yaml

from retargeting.custom_hand_retargeting_fn import retarget_hand_to_robot_joints
from utils.dataset.dataset import DexYCBVideoDataset
from utils.manopth.compat import apply_legacy_numpy_aliases
from utils.manopth.geometry import compute_hand_geometry
from utils.robot.urdf_utils import build_temp_urdf, resolve_visual_urdf
from utils.viser.custom_hand_retargeting_viser import CustomRetargetingViserViewer
from dex_retargeting.constants import ROBOT_NAME_MAP, RobotName
from dex_retargeting.robot_wrapper import RobotWrapper
from utils.manopth.mano_layer import MANOLayer

apply_legacy_numpy_aliases()


def _resolve_optional_path(
    path_value: Optional[str], *, base_dir: Path, repo_root: Path
) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    candidates = [
        (base_dir / path),
        (repo_root / path),
        (Path.cwd() / path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Default to config-relative for unknown/non-existent paths.
    return str((base_dir / path).resolve())


def _load_custom_config_options(config_path: Optional[str]) -> Dict[str, Any]:
    result = {
        "urdf_path": None,
        "add_dummy_free_joints": False,
        "mapping_load_path": None,
        "mapping_output_path": None,
        "default_reference_pose_path": None,
    }
    if config_path is None:
        return result

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        return result

    try:
        with path.open("r") as f:
            payload = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        return result

    if not isinstance(payload, dict):
        return result

    custom_cfg = payload.get("custom_hand_retargeting")
    if not isinstance(custom_cfg, dict):
        return result

    repo_root = Path(__file__).resolve().parent.parent.parent
    urdf_path = _resolve_optional_path(
        custom_cfg.get("urdf_path"), base_dir=path.parent, repo_root=repo_root
    )
    add_dummy_free_joints = custom_cfg.get("add_dummy_free_joints", False)
    # Backward-compatible fallback: allow urdf_path under custom_hand_retargeting.retargeting
    # while avoiding any retargeting type system.
    if urdf_path is None:
        nested = custom_cfg.get("retargeting")
        if isinstance(nested, dict):
            urdf_path = _resolve_optional_path(
                nested.get("urdf_path"), base_dir=path.parent, repo_root=repo_root
            )

    mapping_load = _resolve_optional_path(
        custom_cfg.get("mapping_path"), base_dir=path.parent, repo_root=repo_root
    )
    mapping_output = _resolve_optional_path(
        custom_cfg.get("mapping_output_path"), base_dir=path.parent, repo_root=repo_root
    )
    if mapping_output is None:
        mapping_output = mapping_load

    # Per-hand default/reference pose JSON (save target for the tuner; may not exist yet).
    reference_pose_path = _resolve_optional_path(
        custom_cfg.get("default_reference_pose_path"),
        base_dir=path.parent,
        repo_root=repo_root,
    )

    result["urdf_path"] = urdf_path
    result["add_dummy_free_joints"] = bool(add_dummy_free_joints)
    result["mapping_load_path"] = mapping_load
    result["mapping_output_path"] = mapping_output
    result["default_reference_pose_path"] = reference_pose_path
    return result


class CustomHandRetargetingApp:
    def __init__(
        self,
        *,
        dexycb_dir: str,
        robot: RobotName,
        hand_type: str,
        data_id: int,
        frame_id: int,
        host: str,
        port: int,
        config: Optional[str],
        robot_x_offset: float,
        mapping_load_path: Optional[str],
        mapping_output_path: Optional[str],
    ):
        self.dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type)
        if len(self.dataset) == 0:
            raise ValueError("DexYCBVideoDataset is empty with the current filters.")

        self.hand_type = hand_type
        repo_root = Path(__file__).resolve().parent.parent.parent
        custom_cfg = _load_custom_config_options(config)
        default_urdf = (
            repo_root
            / "assets"
            / "robots"
            / "hands"
            / ROBOT_NAME_MAP[robot]
            / f"{ROBOT_NAME_MAP[robot]}.urdf"
        )
        urdf_path = Path(custom_cfg["urdf_path"]) if custom_cfg["urdf_path"] else default_urdf
        if not urdf_path.is_file():
            raise ValueError(
                f"URDF does not exist: {urdf_path}. "
                "Set custom_hand_retargeting.urdf_path in retarget_config_path."
            )

        visual_urdf = resolve_visual_urdf(urdf_path)
        temp_urdf = build_temp_urdf(
            visual_urdf, add_dummy_free_joints=bool(custom_cfg["add_dummy_free_joints"])
        )
        # Use a dedicated model for optimization to avoid concurrent Pinocchio access
        # between solver callbacks and viewer updates.
        self.optimizer_robot_model = RobotWrapper(str(temp_urdf))
        self.robot_model = RobotWrapper(str(temp_urdf))
        self.robot_joint_names = list(self.optimizer_robot_model.dof_joint_names)
        self.viewer = CustomRetargetingViserViewer(
            urdf_path=str(temp_urdf),
            input_joint_names=self.robot_joint_names,
            robot_model=self.robot_model,
            host=host,
            port=port,
            robot_x_offset=robot_x_offset,
            mapping_load_path=mapping_load_path,
            mapping_output_path=mapping_output_path,
            robot_name_for_default=robot.name,
            hand_type_for_default=hand_type,
            reference_pose_path=custom_cfg["default_reference_pose_path"],
        )
        self.viewer.set_debug_update_callback(lambda: self._refresh(force=True))
        self.mapping_load_path = mapping_load_path

        self.current_data_id = int(np.clip(data_id, 0, len(self.dataset) - 1))
        self.current_frame_id = max(0, int(frame_id))
        self.sample = None
        self.hand_pose_seq = None
        self.object_pose_seq = None
        self.mano_layer = None
        self.mano_faces = None
        self.camera_transform = None
        self.num_frames = 0
        self.valid_frame_indices = np.array([], dtype=np.int32)
        self._last_state = None
        self._is_updating_gui = False

        self.max_dataset_frames = self._get_max_frame_count()
        self._setup_gui()
        self._load_trajectory(self.current_data_id)
        self._refresh()

    def _get_max_frame_count(self) -> int:
        max_frames = 1
        for pose in self.dataset._capture_pose.values():
            max_frames = max(max_frames, int(pose["pose_m"].shape[0]))
        return max_frames

    def _compute_wrist_trajectory(self) -> np.ndarray:
        points = []
        for i in range(self.num_frames):
            vertices, joints = compute_hand_geometry(
                self.hand_pose_seq[i], self.mano_layer, self.camera_transform
            )
            if vertices is None or joints is None:
                continue
            points.append(joints[0])
        if len(points) == 0:
            return np.zeros((1, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _compute_valid_frame_indices(self) -> np.ndarray:
        return np.where(np.abs(self.hand_pose_seq).sum(axis=(1, 2)) > 1e-5)[0].astype(np.int32)

    def _load_trajectory(self, data_id: int) -> None:
        self.current_data_id = int(np.clip(data_id, 0, len(self.dataset) - 1))
        self.sample = self.dataset[self.current_data_id]
        self.hand_pose_seq = self.sample["hand_pose"]
        self.object_pose_seq = self.sample["object_pose"]
        self.num_frames = int(self.hand_pose_seq.shape[0])
        self.valid_frame_indices = self._compute_valid_frame_indices()
        if self.valid_frame_indices.size == 0:
            raise ValueError(
                f"Trajectory {self.current_data_id} has no valid MANO frames."
            )
        self.current_frame_id = int(np.clip(self.current_frame_id, 0, self.num_frames - 1))
        if np.abs(self.hand_pose_seq[self.current_frame_id]).sum() <= 1e-5:
            self.current_frame_id = int(self.valid_frame_indices[0])

        self.mano_layer = MANOLayer(self.hand_type, self.sample["hand_shape"].astype(np.float32))
        self.mano_faces = self.mano_layer.f.cpu().numpy().astype(np.uint32)
        self.camera_transform = np.linalg.inv(self.sample["extrinsics"]).astype(np.float32)
        wrist_traj = self._compute_wrist_trajectory()

        self.viewer.load_trajectory(
            camera_transform=self.camera_transform,
            hand_faces=self.mano_faces,
            ycb_ids=self.sample["ycb_ids"],
            object_mesh_files=self.sample["object_mesh_file"],
            wrist_trajectory_points=wrist_traj,
        )

    def _setup_gui(self) -> None:
        gui = self.viewer.server.gui
        with gui.add_folder("Dataset"):
            self.trajectory_slider = gui.add_slider(
                "trajectory_id",
                min=0,
                max=max(0, len(self.dataset) - 1),
                step=1,
                initial_value=self.current_data_id,
            )
            self.frame_slider = gui.add_slider(
                "frame_id",
                min=0,
                max=max(0, self.max_dataset_frames - 1),
                step=1,
                initial_value=self.current_frame_id,
            )

        @self.trajectory_slider.on_update
        def _(_event):
            if self._is_updating_gui:
                return
            self._refresh()

        @self.frame_slider.on_update
        def _(_event):
            if self._is_updating_gui:
                return
            self._refresh()

    def _refresh(self, force: bool = False) -> None:
        state = (int(self.trajectory_slider.value), int(self.frame_slider.value))
        if not force and state == self._last_state:
            return
        self._last_state = state

        requested_data_id, requested_frame_id = state
        if requested_data_id != self.current_data_id:
            self._load_trajectory(requested_data_id)

        self.current_frame_id = int(np.clip(requested_frame_id, 0, self.num_frames - 1))
        if np.abs(self.hand_pose_seq[self.current_frame_id]).sum() <= 1e-5:
            nearest_valid = self.valid_frame_indices[
                np.argmin(np.abs(self.valid_frame_indices - self.current_frame_id))
            ]
            self.current_frame_id = int(nearest_valid)
        if self.current_frame_id != requested_frame_id:
            self._is_updating_gui = True
            self.frame_slider.value = self.current_frame_id
            self._is_updating_gui = False

        hand_pose_frame = self.hand_pose_seq[self.current_frame_id]
        object_pose_frame = self.object_pose_seq[self.current_frame_id]
        vertices, joints = compute_hand_geometry(
            hand_pose_frame, self.mano_layer, self.camera_transform
        )
        if vertices is None or joints is None:
            return

        hand_keypoints = joints.astype(np.float32)
        hand_joint_values = hand_pose_frame[:, :48].astype(np.float32)
        self.viewer.set_hand_joints_for_root_alignment(hand_keypoints)
        hand_to_robot_translation, hand_to_robot_rotation = (
            self.viewer.get_hand_to_robot_transform()
        )
        debug_link_index, debug_target_rotation_local = (
            self.viewer.get_single_frame_orientation_target()
        )
        reference_pose, reg_weight = self.viewer.get_reference_pose_and_weight()
        if self.viewer.is_retarget_enabled():
            mapping_link_indices, mapping_offsets = (
                self.viewer.get_current_mapping_for_optimizer()
            )
            robot_qpos = retarget_hand_to_robot_joints(
                hand_keypoints=hand_keypoints,
                hand_joint_values=hand_joint_values,
                robot_joint_names=self.robot_joint_names,
                robot_model=self.optimizer_robot_model,
                mapping_path=self.mapping_load_path,
                mapping_link_indices=mapping_link_indices,
                mapping_offsets=mapping_offsets,
                hand_to_robot_translation=hand_to_robot_translation,
                hand_to_robot_rotation=hand_to_robot_rotation,
                debug_link_index=debug_link_index,
                debug_target_rotation_local=debug_target_rotation_local,
                reference_pose=reference_pose,
                reg_weight=reg_weight,
            )
        else:
            # Default-pose mode: park the green hand at the reference pose and skip the
            # (expensive) retarget solve entirely, so aligning link frames stays responsive.
            robot_qpos = np.asarray(reference_pose, dtype=np.float32)

        self.viewer.update_frame(
            hand_vertices=vertices,
            hand_joints=joints,
            object_pose_frame=object_pose_frame,
            robot_qpos=robot_qpos,
        )


def main(
    dexycb_dir: str,
    robot: RobotName = RobotName.robotis_5f,
    hand_type: str = "right",
    data_id: int = 0,
    frame_id: int = 0,
    host: str = "0.0.0.0",
    port: int = 8080,
    config: Optional[str] = None,
    robot_x_offset: float = 0.24,
    mapping_load_path: Optional[str] = None,
    mapping_output_path: Optional[str] = None,
):
    """
    Interactive custom retargeting viewer.

    Loads one DexYCB trajectory, visualizes hand mesh + MANO skeleton,
    object meshes, robot hand/skeleton, and MANO-point to robot-link mapping
    controls. Retargeting uses a simplified single-frame orientation debug
    optimizer driven by GUI sliders.

    If config contains a `custom_hand_retargeting` block:
      custom_hand_retargeting.urdf_path
      custom_hand_retargeting.add_dummy_free_joints
      custom_hand_retargeting.mapping_path
      custom_hand_retargeting.mapping_output_path
    CLI mapping args override config values when provided.
    """
    custom_cfg = _load_custom_config_options(config)
    resolved_mapping_load = (
        mapping_load_path if mapping_load_path is not None else custom_cfg["mapping_load_path"]
    )
    resolved_mapping_output = (
        mapping_output_path
        if mapping_output_path is not None
        else custom_cfg["mapping_output_path"]
    )
    _ = CustomHandRetargetingApp(
        dexycb_dir=dexycb_dir,
        robot=robot,
        hand_type=hand_type,
        data_id=data_id,
        frame_id=frame_id,
        host=host,
        port=port,
        config=config,
        robot_x_offset=robot_x_offset,
        mapping_load_path=resolved_mapping_load,
        mapping_output_path=resolved_mapping_output,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
