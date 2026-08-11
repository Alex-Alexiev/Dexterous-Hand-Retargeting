import time
from pathlib import Path
from typing import Dict

import numpy as np
import tyro
from pytransform3d import rotations

from dex_retargeting.robot_wrapper import RobotWrapper
from utils.dataset.retargeted_grasp_dataset import load_retargeted_grasp_dataset
from utils.manopth.geometry import compute_hand_geometry
from utils.manopth.mano_layer import MANOLayer
from utils.system.path_utils import resolve_repo_path
from utils.robot.mano_root_alignment import (
    compute_mano_frame_rotations,
    compute_mapped_mano_points,
    load_mano_link_mapping,
)
from utils.robot.urdf_utils import build_temp_urdf, resolve_visual_urdf
from utils.viser.scene_utils import (
    HAND_MESH_COLOR,
    HAND_SKELETON_EDGES,
    OBJECT_COLORS,
    load_obj_tri_mesh,
    segments_from_edges,
)
from utils.viser.viser_urdf_viewer import ViserHandViewer


def _row(payload: Dict[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    return {key: value[index] for key, value in payload.items()}


def _offset_points(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points.astype(np.float32) + offset.reshape(1, 3).astype(np.float32)


def _robot_root_pose_from_target_mano_root(
    *,
    qpos: np.ndarray,
    target_mano0_translation: np.ndarray,
    target_mano0_orientation_wxyz: np.ndarray,
    robot_model: RobotWrapper,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
):
    robot_model.compute_forward_kinematics(qpos.astype(np.float64))
    mapped_points_local, valid_mask = compute_mapped_mano_points(
        robot_model,
        link_indices_by_mano,
        offsets_by_mano,
    )
    mapped_rotations_local = compute_mano_frame_rotations(mapped_points_local, valid_mask)
    target_rotation = rotations.matrix_from_quaternion(
        np.asarray(target_mano0_orientation_wxyz, dtype=np.float64)
    )
    robot_root_rotation = target_rotation @ mapped_rotations_local[0].T
    robot_root_translation = (
        np.asarray(target_mano0_translation, dtype=np.float64)
        - robot_root_rotation @ mapped_points_local[0]
    )
    return robot_root_translation.astype(np.float32), robot_root_rotation.astype(np.float32)


def _cell_offset(index: int, *, rows: int, cols: int, spacing_x: float, spacing_y: float) -> np.ndarray:
    row = index // cols
    col = index % cols
    centered_x = (col - 0.5 * (cols - 1)) * spacing_x
    centered_y = (0.5 * (rows - 1) - row) * spacing_y
    return np.asarray([centered_x, centered_y, 0.0], dtype=np.float32)


def _render_datapoint_cell(
    *,
    server,
    datapoint: Dict[str, np.ndarray],
    dataset_index: int,
    cell_index: int,
    cell_offset: np.ndarray,
    hand_type: str,
    temp_urdf_path: Path,
    robot_joint_names,
    robot_model: RobotWrapper,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
    mesh_cache: Dict[Path, tuple],
):
    prefix = f"/cell_{cell_index:02d}"
    mano_layer = MANOLayer(hand_type, np.asarray(datapoint["human_shape"], dtype=np.float32))
    hand_faces = mano_layer.f.cpu().numpy().astype(np.uint32)
    hand_pose_frame = np.asarray(datapoint["human_mano"], dtype=np.float32)[None, :]
    camera_transform = np.asarray(datapoint["camera_transform"], dtype=np.float32)
    hand_vertices, hand_joints = compute_hand_geometry(hand_pose_frame, mano_layer, camera_transform)
    if hand_vertices is None or hand_joints is None:
        raise ValueError(f"Datapoint {dataset_index} contains an invalid MANO frame.")

    human_mano_root_translation = np.asarray(
        datapoint["human_mano_root_translation"], dtype=np.float32
    )
    anchor_offset = cell_offset - human_mano_root_translation
    hand_vertices_world = _offset_points(hand_vertices, anchor_offset)
    hand_joints_world = _offset_points(hand_joints, anchor_offset)
    server.scene.add_mesh_simple(
        f"{prefix}/human/mesh",
        vertices=hand_vertices_world,
        faces=hand_faces,
        color=HAND_MESH_COLOR,
        opacity=0.45,
    )
    server.scene.add_line_segments(
        f"{prefix}/human/skeleton_lines",
        points=segments_from_edges(hand_joints_world, HAND_SKELETON_EDGES),
        colors=(60, 120, 255),
        line_width=3.0,
    )
    server.scene.add_point_cloud(
        f"{prefix}/human/skeleton_points",
        points=hand_joints_world,
        colors=(60, 120, 255),
        point_size=0.005,
        point_shape="circle",
    )

    qpos = np.asarray(datapoint["retargeted_qpos"], dtype=np.float32)
    robot_root_translation, robot_root_rotation = _robot_root_pose_from_target_mano_root(
        qpos=qpos,
        target_mano0_translation=cell_offset,
        target_mano0_orientation_wxyz=np.asarray(
            datapoint["human_mano_root_orientation_wxyz"], dtype=np.float32
        ),
        robot_model=robot_model,
        link_indices_by_mano=link_indices_by_mano,
        offsets_by_mano=offsets_by_mano,
    )
    robot_root = server.scene.add_frame(f"{prefix}/robot", show_axes=False)
    robot_root.position = robot_root_translation
    robot_root.wxyz = rotations.quaternion_from_matrix(
        robot_root_rotation.astype(np.float64)
    ).astype(np.float32)
    robot_viewer = ViserHandViewer(
        urdf_path=str(temp_urdf_path),
        input_joint_names=robot_joint_names,
        server=server,
        root_node_name=f"{prefix}/robot",
        mesh_color_override=(0.55, 0.90, 0.55, 0.45),
    )
    robot_viewer.set_qpos(qpos)

    object_mesh_path = Path(str(datapoint["object_mesh_file"]))
    if object_mesh_path not in mesh_cache:
        mesh_cache[object_mesh_path] = load_obj_tri_mesh(object_mesh_path)
    mesh_vertices, mesh_faces = mesh_cache[object_mesh_path]
    object_handle = server.scene.add_mesh_simple(
        f"{prefix}/object",
        vertices=mesh_vertices,
        faces=mesh_faces,
        color=OBJECT_COLORS[cell_index % len(OBJECT_COLORS)],
        opacity=0.60,
    )
    object_handle.position = (
        np.asarray(datapoint["object_translation"], dtype=np.float32) + anchor_offset
    )
    object_handle.wxyz = np.asarray(datapoint["object_orientation_wxyz"], dtype=np.float32)

    return robot_viewer


def main(
    dataset_dir: str,
    start_index: int = 0,
    rows: int = 10,
    cols: int = 10,
    spacing_x: float = 0.60,
    spacing_y: float = 0.60,
    host: str = "0.0.0.0",
    port: int = 8081,
):
    try:
        import viser
    except ImportError as exc:
        raise ImportError("This script requires: pip install viser") from exc

    repo_root = Path(__file__).resolve().parent
    resolved_dataset_dir = resolve_repo_path(dataset_dir, repo_root)
    payload, metadata = load_retargeted_grasp_dataset(resolved_dataset_dir)
    num_datapoints = int(payload["frame_id"].shape[0])
    if num_datapoints == 0:
        raise ValueError(f"No datapoints found in {resolved_dataset_dir}.")

    rows = max(1, int(rows))
    cols = max(1, int(cols))
    grid_size = rows * cols
    start_index = int(np.clip(start_index, 0, num_datapoints - 1))
    end_index = min(num_datapoints, start_index + grid_size)
    selected_indices = list(range(start_index, end_index))

    hand_type = str(metadata["hand_type"])
    urdf_path = Path(metadata["urdf_path"])
    mapping_path = Path(metadata["mapping_path"])
    visual_urdf_path = resolve_visual_urdf(urdf_path)
    temp_urdf_path = build_temp_urdf(visual_urdf_path, add_dummy_free_joints=False)

    robot_model = RobotWrapper(str(temp_urdf_path))
    link_indices_by_mano, offsets_by_mano = load_mano_link_mapping(mapping_path, robot_model)
    robot_joint_names = list(metadata["robot_joint_names"])

    server = viser.ViserServer(host=host, port=port)
    mesh_cache: Dict[Path, tuple] = {}
    robot_viewers = []
    for cell_index, dataset_index in enumerate(selected_indices):
        datapoint = _row(payload, dataset_index)
        robot_viewers.append(
            _render_datapoint_cell(
                server=server,
                datapoint=datapoint,
                dataset_index=dataset_index,
                cell_index=cell_index,
                cell_offset=_cell_offset(
                    cell_index,
                    rows=rows,
                    cols=cols,
                    spacing_x=spacing_x,
                    spacing_y=spacing_y,
                ),
                hand_type=hand_type,
                temp_urdf_path=temp_urdf_path,
                robot_joint_names=robot_joint_names,
                robot_model=robot_model,
                link_indices_by_mano=link_indices_by_mano,
                offsets_by_mano=offsets_by_mano,
                mesh_cache=mesh_cache,
            )
        )

    with server.gui.add_folder("Dataset Grid"):
        server.gui.add_text("dataset_name", initial_value=str(metadata["dataset_name"]), disabled=True)
        server.gui.add_text("start_index", initial_value=str(start_index), disabled=True)
        server.gui.add_text("end_index", initial_value=str(end_index - 1), disabled=True)
        server.gui.add_text("grid_shape", initial_value=f"{rows}x{cols}", disabled=True)
        server.gui.add_text("num_rendered", initial_value=str(len(selected_indices)), disabled=True)

    print(
        f"Loaded {len(selected_indices)} datapoints on {host}:{port} "
        f"(dataset={metadata['dataset_name']}, start_index={start_index}, "
        f"grid={rows}x{cols})."
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
