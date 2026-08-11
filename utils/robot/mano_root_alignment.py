import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from utils.robot.robot_wrapper import RobotWrapper

_NUM_MANO_POINTS = 21
_MANO_PARENT_IDX = np.asarray(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)
_MANO_CHILD_IDX = np.asarray(
    [9, 2, 3, 4, -1, 6, 7, 8, -1, 10, 11, 12, -1, 14, 15, 16, -1, 18, 19, 20, -1],
    dtype=np.int32,
)
_EPS = 1e-8


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec64 = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec64))
    if norm < _EPS:
        return np.zeros((3,), dtype=np.float64)
    return vec64 / norm


def rotation_from_x_prev_z(x_vec: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
    x_axis = _normalize(x_vec)
    if float(np.linalg.norm(x_axis)) < _EPS:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    z_axis_prev = _normalize(z_prev)
    if float(np.linalg.norm(z_axis_prev)) < _EPS:
        z_axis_prev = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    y_axis = np.cross(x_axis, z_axis_prev)
    if float(np.linalg.norm(y_axis)) < _EPS:
        alt_z = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, alt_z))) > 0.9:
            alt_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(x_axis, alt_z)
    y_axis = _normalize(y_axis)

    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, z_axis_prev)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def compute_mano_frame_rotations(points: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    rots = np.tile(np.eye(3, dtype=np.float64), (_NUM_MANO_POINTS, 1, 1))
    if points.shape != (_NUM_MANO_POINTS, 3):
        return rots

    if valid_mask is None:
        valid_mask = np.ones((_NUM_MANO_POINTS,), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not bool(valid_mask[0]):
        return rots

    root_forward = points[9] - points[0]
    root_lateral = points[13] - points[5]
    if float(np.linalg.norm(root_forward)) < _EPS:
        root_forward = points[1] - points[0]
    if float(np.linalg.norm(root_lateral)) < _EPS:
        root_lateral = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    palm_z = np.cross(root_forward, root_lateral)
    if float(np.linalg.norm(palm_z)) < _EPS:
        palm_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rots[0] = rotation_from_x_prev_z(root_forward, palm_z)

    for idx in range(1, _NUM_MANO_POINTS):
        parent = int(_MANO_PARENT_IDX[idx])
        if parent < 0 or not valid_mask[idx] or not valid_mask[parent]:
            rots[idx] = rots[parent] if parent >= 0 else np.eye(3, dtype=np.float64)
            continue

        child = int(_MANO_CHILD_IDX[idx])
        if child >= 0 and valid_mask[child]:
            forward = points[child] - points[idx]
        else:
            forward = points[idx] - points[parent]
        rots[idx] = rotation_from_x_prev_z(forward, rots[parent, :, 2])
    return rots


def load_mano_link_mapping(
    mapping_path: Path, robot_model: RobotWrapper
) -> Tuple[np.ndarray, np.ndarray]:
    link_indices_by_mano = np.full((_NUM_MANO_POINTS,), -1, dtype=np.int32)
    offsets_by_mano = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float64)

    with Path(mapping_path).expanduser().open("r") as f:
        payload = json.load(f)

    assignments = payload.get("mano_link_assignments", [])
    link_name_to_frame = {name: i for i, name in enumerate(robot_model.link_names)}
    for entry in assignments:
        point_idx = int(entry["mano_point_index"])
        link_name = str(entry["link_name"])
        offset = np.asarray(entry.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        frame_idx = link_name_to_frame.get(link_name, -1)
        if 0 <= point_idx < _NUM_MANO_POINTS and frame_idx >= 0 and offset.shape == (3,):
            link_indices_by_mano[point_idx] = frame_idx
            offsets_by_mano[point_idx] = offset

    return link_indices_by_mano, offsets_by_mano


def compute_mapped_mano_points(
    robot_model: RobotWrapper,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float64)
    valid_mask = link_indices_by_mano >= 0
    for point_idx in range(_NUM_MANO_POINTS):
        frame_idx = int(link_indices_by_mano[point_idx])
        if frame_idx < 0:
            continue
        pose = robot_model.get_link_pose(frame_idx).astype(np.float64)
        points[point_idx] = pose[:3, :3] @ offsets_by_mano[point_idx] + pose[:3, 3]
    return points, valid_mask


def compute_hand_root_frame(hand_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hand_joints = np.asarray(hand_joints, dtype=np.float64)
    root_forward = hand_joints[9] - hand_joints[0]
    root_lateral = hand_joints[13] - hand_joints[5]
    if float(np.linalg.norm(root_forward)) < _EPS:
        root_forward = hand_joints[1] - hand_joints[0]
    if float(np.linalg.norm(root_lateral)) < _EPS:
        root_lateral = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    palm_z = np.cross(root_forward, root_lateral)
    if float(np.linalg.norm(palm_z)) < _EPS:
        palm_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    root_rot = rotation_from_x_prev_z(root_forward, palm_z)
    root_pos = hand_joints[0]
    return root_pos, root_rot


def compute_aligned_robot_root_pose(
    *,
    hand_joints_world: np.ndarray,
    robot_model: RobotWrapper,
    robot_qpos: np.ndarray,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
    hand_to_robot_translation: Optional[np.ndarray] = None,
    hand_to_robot_rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    root_pos_h, root_rot_h = compute_hand_root_frame(hand_joints_world)
    trans_hr = np.zeros((3,), dtype=np.float64)
    rot_hr = np.eye(3, dtype=np.float64)
    if hand_to_robot_translation is not None:
        trans_hr = np.asarray(hand_to_robot_translation, dtype=np.float64)
    if hand_to_robot_rotation is not None:
        rot_hr = np.asarray(hand_to_robot_rotation, dtype=np.float64)

    target_mano0_pos_world = root_pos_h + root_rot_h @ trans_hr
    target_mano0_rot_world = root_rot_h @ rot_hr

    robot_model.compute_forward_kinematics(np.asarray(robot_qpos, dtype=np.float64))
    mapping_points_local, valid_mask = compute_mapped_mano_points(
        robot_model, link_indices_by_mano, offsets_by_mano
    )
    mapping_rots_local = compute_mano_frame_rotations(mapping_points_local, valid_mask)
    mano0_pos_local = mapping_points_local[0]
    mano0_rot_local = mapping_rots_local[0]

    robot_root_rot_world = target_mano0_rot_world @ mano0_rot_local.T
    robot_root_pos_world = target_mano0_pos_world - robot_root_rot_world @ mano0_pos_local
    return robot_root_pos_world.astype(np.float32), robot_root_rot_world.astype(np.float32)
