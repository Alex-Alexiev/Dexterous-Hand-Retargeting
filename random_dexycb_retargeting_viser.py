import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from pytransform3d import rotations

from dex_retargeting.robot_wrapper import RobotWrapper
from retargeting.custom_hand_retargeting_fn import retarget_hand_to_robot_joints
from utils.dataset.dataset import DexYCBVideoDataset, YCB_CLASSES
from utils.dataset.grasp_phase_utils import (
    compute_grasp_phase_and_object_selection,
    plot_grasp_phase_diagnostics,
)
from utils.dataset.sample_utils import iter_random_valid_trajectories
from utils.manopth.geometry import (
    compute_hand_geometry,
    compute_joint_trajectory_with_frames,
)
from utils.system.path_utils import resolve_repo_path
from utils.robot.mano_root_alignment import (
    compute_aligned_robot_root_pose,
    load_mano_link_mapping,
)
from utils.robot.urdf_utils import build_temp_urdf, resolve_visual_urdf
from utils.viser.scene_utils import (
    HAND_MESH_COLOR,
    HAND_SKELETON_EDGES,
    OBJECT_COLORS,
    load_obj_tri_mesh,
    make_transform,
    quat_xyzw_to_wxyz,
    segments_from_edges,
    segments_from_polyline,
    trajectory_velocity_and_turn_vectors,
    vector_segments_from_origins,
)
from utils.viser.viser_urdf_viewer import ViserHandViewer

from utils.manopth.mano_layer import MANOLayer


def _make_default_phase_plot_path(repo_root: Path, capture_name: str, data_id: int) -> Path:
    safe_capture_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in capture_name
    )
    return (
        repo_root
        / "debug"
        / "grasp_phase_plots"
        / f"{data_id:04d}_{safe_capture_name}_grasp_phase_debug.png"
    )


def _offset_vertices(vertices: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return vertices.astype(np.float32) + translation.reshape(1, 3).astype(np.float32)


def _render_phase(
    *,
    server,
    phase_name: str,
    phase_frame: int,
    phase_offset: np.ndarray,
    sample,
    mano_layer: MANOLayer,
    camera_transform: np.ndarray,
    hand_faces: np.ndarray,
    temp_urdf_path: Path,
    robot_x_offset: float,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
    mesh_cache,
    wrist_trajectory_points: np.ndarray,
    velocity_vector_segments: np.ndarray,
    turn_vector_segments: np.ndarray,
    selected_object_index: int,
):
    hand_pose_frame = sample["hand_pose"][phase_frame]
    object_pose_frame = sample["object_pose"][phase_frame]
    hand_vertices, hand_joints = compute_hand_geometry(
        hand_pose_frame, mano_layer, camera_transform
    )
    if hand_vertices is None or hand_joints is None:
        raise ValueError(f"Selected phase frame {phase_frame} is invalid.")

    optimizer_robot_model = RobotWrapper(str(temp_urdf_path))
    robot_joint_names = list(optimizer_robot_model.dof_joint_names)
    robot_qpos = retarget_hand_to_robot_joints(
        hand_keypoints=hand_joints.astype(np.float32),
        hand_joint_values=hand_pose_frame[:, :48].astype(np.float32),
        robot_joint_names=robot_joint_names,
        robot_model=optimizer_robot_model,
        mapping_link_indices=link_indices_by_mano,
        mapping_offsets=offsets_by_mano,
    )

    robot_root = server.scene.add_frame(f"/{phase_name}/robot", show_axes=False)
    robot_root.position = (
        phase_offset + np.array([robot_x_offset, 0.0, 0.0], dtype=np.float32)
    ).astype(np.float32)
    robot_viewer = ViserHandViewer(
        urdf_path=str(temp_urdf_path),
        input_joint_names=robot_joint_names,
        server=server,
        root_node_name=f"/{phase_name}/robot",
        mesh_color_override=(0.55, 0.90, 0.55, 0.45),
    )
    robot_viewer.set_qpos(robot_qpos)

    hand_vertices_world = _offset_vertices(hand_vertices, phase_offset)
    hand_joints_world = _offset_vertices(hand_joints, phase_offset)
    robot_root_pos_world, robot_root_rot_world = compute_aligned_robot_root_pose(
        hand_joints_world=hand_joints_world,
        robot_model=optimizer_robot_model,
        robot_qpos=robot_qpos,
        link_indices_by_mano=link_indices_by_mano,
        offsets_by_mano=offsets_by_mano,
        hand_to_robot_translation=np.asarray([robot_x_offset, 0.0, 0.0], dtype=np.float32),
    )
    robot_root.position = robot_root_pos_world
    robot_root.wxyz = rotations.quaternion_from_matrix(robot_root_rot_world).astype(np.float32)

    server.scene.add_mesh_simple(
        f"/{phase_name}/hand/mesh",
        vertices=hand_vertices_world,
        faces=hand_faces,
        color=HAND_MESH_COLOR,
        opacity=0.45,
    )
    server.scene.add_line_segments(
        f"/{phase_name}/hand/skeleton_lines",
        points=segments_from_edges(hand_joints_world, HAND_SKELETON_EDGES),
        colors=(60, 120, 255),
        line_width=3.0,
    )
    server.scene.add_point_cloud(
        f"/{phase_name}/hand/skeleton_points",
        points=hand_joints_world,
        colors=(60, 120, 255),
        point_size=0.006,
        point_shape="circle",
    )
    server.scene.add_line_segments(
        f"/{phase_name}/hand/wrist_trajectory",
        points=segments_from_polyline(_offset_vertices(wrist_trajectory_points, phase_offset)),
        colors=(250, 220, 80),
        line_width=2.0,
    )
    server.scene.add_line_segments(
        f"/{phase_name}/hand/velocity_vectors",
        points=velocity_vector_segments + phase_offset.reshape(1, 1, 3).astype(np.float32),
        colors=(40, 200, 120),
        line_width=2.0,
    )
    server.scene.add_line_segments(
        f"/{phase_name}/hand/turn_vectors",
        points=turn_vector_segments + phase_offset.reshape(1, 1, 3).astype(np.float32),
        colors=(255, 80, 80),
        line_width=2.0,
    )

    object_id = int(selected_object_index)
    ycb_id = sample["ycb_ids"][object_id]
    mesh_path = Path(sample["object_mesh_file"][object_id])
    if mesh_path not in mesh_cache:
        mesh_cache[mesh_path] = load_obj_tri_mesh(mesh_path)
    vertices, faces = mesh_cache[mesh_path]
    handle = server.scene.add_mesh_simple(
        f"/{phase_name}/objects/{object_id}_{YCB_CLASSES.get(int(ycb_id), f'obj_{int(ycb_id)}')}",
        vertices=vertices,
        faces=faces,
        color=OBJECT_COLORS[object_id % len(OBJECT_COLORS)],
        opacity=0.60,
    )
    object_tf = camera_transform @ make_transform(
        object_pose_frame[object_id, 4:],
        quat_xyzw_to_wxyz(object_pose_frame[object_id, :4]),
    )
    handle.position = (object_tf[:3, 3].astype(np.float32) + phase_offset).astype(np.float32)
    handle.wxyz = rotations.quaternion_from_matrix(object_tf[:3, :3]).astype(np.float32)

    return robot_viewer


def main(
    dexycb_dir: str,
    hand_type: str = "right",
    urdf_path: Optional[str] = None,
    mapping_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    robot_x_offset: float = 0.24,
    phase_spacing: float = 0.42,
    trajectory_sample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
    trajectory_vector_scale: float = 0.03,
    trajectory_vector_stride: int = 5,
    phase_plot_path: Optional[str] = None,
    show_phase_plot: bool = False,
    seed: Optional[int] = None,
):
    """
    Load one random DexYCB trajectory, detect approach/grasp/lift phases, and
    visualize the three phase frames side by side in viser.
    """
    try:
        import viser
    except ImportError as exc:
        raise ImportError("This script requires: pip install viser") from exc

    repo_root = Path(__file__).resolve().parent
    default_urdf_path = repo_root / "assets" / "robots" / "robotis_5f_hand" / "robotis_5f_hand.urdf"
    default_mapping_path = (
        repo_root
        / "assets"
        / "robots"
        / "robotis_5f_hand"
        / f"robotis_5f_{hand_type}_mano_link_mapping.json"
    )

    resolved_urdf_path = resolve_repo_path(urdf_path, repo_root) if urdf_path else default_urdf_path
    resolved_mapping_path = (
        resolve_repo_path(mapping_path, repo_root)
        if mapping_path
        else (default_mapping_path if default_mapping_path.is_file() else None)
    )
    if not resolved_urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {resolved_urdf_path}")
    if resolved_mapping_path is None or not resolved_mapping_path.is_file():
        raise ValueError(f"Mapping does not exist: {resolved_mapping_path}")

    dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type)
    if len(dataset) == 0:
        raise ValueError("DexYCBVideoDataset is empty with the current settings.")

    rng = np.random.default_rng(seed)
    data_id = None
    sample = None
    mano_layer = None
    hand_faces = None
    camera_transform = None
    wrist_trajectory_points = None
    wrist_frame_ids = None
    trajectory_selection = None
    selected_object_index = None
    for candidate_data_id, candidate_sample in iter_random_valid_trajectories(dataset, rng):
        candidate_mano_layer = MANOLayer(
            hand_type, candidate_sample["hand_shape"].astype(np.float32)
        )
        candidate_camera_transform = np.linalg.inv(candidate_sample["extrinsics"]).astype(
            np.float32
        )
        candidate_joint_trajectory, candidate_joint_frame_ids = compute_joint_trajectory_with_frames(
            candidate_sample["hand_pose"],
            candidate_mano_layer,
            candidate_camera_transform,
        )
        candidate_wrist_points = candidate_joint_trajectory[:, 0, :].astype(np.float32)
        if candidate_wrist_points.shape[0] < 3:
            continue
        try:
            candidate_trajectory_selection = compute_grasp_phase_and_object_selection(
                candidate_wrist_points,
                hand_joint_trajectory=candidate_joint_trajectory,
                object_pose_seq=candidate_sample["object_pose"],
                object_mesh_files=candidate_sample["object_mesh_file"],
                camera_transform=candidate_camera_transform,
                frame_coords=candidate_joint_frame_ids.astype(np.float32),
                resample_spacing_m=trajectory_sample_spacing_m,
                inflection_stride=inflection_stride,
                approach_offset_m=approach_offset_m,
                lift_offset_m=lift_offset_m,
                refine_window_m=refine_window_m,
                refine_resample_spacing_m=refine_resample_spacing_m,
            )
        except ValueError:
            continue

        data_id = candidate_data_id
        sample = candidate_sample
        mano_layer = candidate_mano_layer
        hand_faces = mano_layer.f.cpu().numpy().astype(np.uint32)
        camera_transform = candidate_camera_transform
        wrist_trajectory_points = candidate_wrist_points
        wrist_frame_ids = candidate_joint_frame_ids.astype(np.float32)
        trajectory_selection = candidate_trajectory_selection
        selected_object_index = trajectory_selection.grasped_object.object_index
        break

    if sample is None or trajectory_selection is None or selected_object_index is None:
        raise ValueError("Could not find a DexYCB trajectory with detectable grasp phases.")

    phase_frames = trajectory_selection.phase_diagnostics.phase_frames
    selected_object_name = YCB_CLASSES.get(
        int(sample["ycb_ids"][selected_object_index]),
        f"obj_{int(sample['ycb_ids'][selected_object_index])}",
    )

    resolved_phase_plot_path = (
        resolve_repo_path(phase_plot_path, repo_root)
        if phase_plot_path is not None
        else _make_default_phase_plot_path(repo_root, str(sample["capture_name"]), data_id)
    )
    if resolved_phase_plot_path is not None or show_phase_plot:
        _, plotted_phase_frames = plot_grasp_phase_diagnostics(
            wrist_trajectory_points,
            frame_coords=wrist_frame_ids,
            resample_spacing_m=trajectory_sample_spacing_m,
            inflection_stride=inflection_stride,
            approach_offset_m=approach_offset_m,
            lift_offset_m=lift_offset_m,
            refine_window_m=refine_window_m,
            refine_resample_spacing_m=refine_resample_spacing_m,
            output_path=str(resolved_phase_plot_path),
            show=show_phase_plot,
            title=f"{sample['capture_name']} grasp-phase diagnostics",
        )
        phase_frames = plotted_phase_frames
    (
        resampled_wrist_trajectory_points,
        wrist_velocity_vectors,
        wrist_turn_vectors,
    ) = trajectory_velocity_and_turn_vectors(
        wrist_trajectory_points,
        resample_spacing_m=trajectory_sample_spacing_m,
        inflection_stride=inflection_stride,
    )
    velocity_vector_segments = vector_segments_from_origins(
        resampled_wrist_trajectory_points,
        wrist_velocity_vectors,
        scale=trajectory_vector_scale,
        stride=trajectory_vector_stride,
    )
    turn_vector_segments = vector_segments_from_origins(
        resampled_wrist_trajectory_points,
        wrist_turn_vectors,
        scale=trajectory_vector_scale,
        stride=trajectory_vector_stride,
    )

    visual_urdf_path = resolve_visual_urdf(resolved_urdf_path)
    temp_urdf_path = build_temp_urdf(visual_urdf_path, add_dummy_free_joints=False)
    mapping_robot_model = RobotWrapper(str(temp_urdf_path))
    link_indices_by_mano, offsets_by_mano = load_mano_link_mapping(
        resolved_mapping_path, mapping_robot_model
    )

    server = viser.ViserServer(host=host, port=port)
    mesh_cache = {}
    robot_viewers = []

    phase_sequence = [
        ("approach", phase_frames.approach_frame, np.array([-phase_spacing, 0.0, 0.0], dtype=np.float32)),
        ("grasp", phase_frames.grasp_frame, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        ("lift", phase_frames.lift_frame, np.array([phase_spacing, 0.0, 0.0], dtype=np.float32)),
    ]
    for phase_name, phase_frame, phase_offset in phase_sequence:
        robot_viewers.append(
            _render_phase(
                server=server,
                phase_name=phase_name,
                phase_frame=phase_frame,
                phase_offset=phase_offset,
                sample=sample,
                mano_layer=mano_layer,
                camera_transform=camera_transform,
                hand_faces=hand_faces,
                temp_urdf_path=temp_urdf_path,
                robot_x_offset=robot_x_offset,
                link_indices_by_mano=link_indices_by_mano,
                offsets_by_mano=offsets_by_mano,
                mesh_cache=mesh_cache,
                wrist_trajectory_points=wrist_trajectory_points,
                velocity_vector_segments=velocity_vector_segments,
                turn_vector_segments=turn_vector_segments,
                selected_object_index=selected_object_index,
            )
        )

    with server.gui.add_folder("Selected Phases"):
        server.gui.add_text(
            "capture_name",
            initial_value=str(sample["capture_name"]),
            disabled=True,
        )
        server.gui.add_text(
            "approach_frame",
            initial_value=str(phase_frames.approach_frame),
            disabled=True,
        )
        server.gui.add_text(
            "grasp_frame",
            initial_value=str(phase_frames.grasp_frame),
            disabled=True,
        )
        server.gui.add_text(
            "lift_frame",
            initial_value=str(phase_frames.lift_frame),
            disabled=True,
        )
        server.gui.add_text(
            "grasped_object",
            initial_value=selected_object_name,
            disabled=True,
        )
        server.gui.add_text(
            "object_frame",
            initial_value=str(trajectory_selection.grasped_object.reference_frame),
            disabled=True,
        )
        server.gui.add_text(
            "mean_fingertip_distance_m",
            initial_value=f"{trajectory_selection.grasped_object.mean_fingertip_distance_m:.4f}",
            disabled=True,
        )

    print(
        f"Loaded DexYCB trajectory {data_id} on {host}:{port} "
        f"(capture={sample['capture_name']}, "
        f"approach={phase_frames.approach_frame}, "
        f"grasp={phase_frames.grasp_frame}, "
        f"lift={phase_frames.lift_frame}, "
        f"object={selected_object_name}"
        + (
            f", phase_plot={resolved_phase_plot_path}"
            if resolved_phase_plot_path is not None
            else ""
        )
        + ")."
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
