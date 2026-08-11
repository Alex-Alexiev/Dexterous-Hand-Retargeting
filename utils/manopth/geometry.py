from typing import Optional, Tuple

import numpy as np
import torch

from utils.manopth.mano_layer import MANOLayer


def compute_hand_geometry(
    hand_pose_frame: np.ndarray,
    mano_layer: MANOLayer,
    camera_transform: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if np.abs(hand_pose_frame).sum() < 1e-5:
        return None, None

    pose = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    translation = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    vertices, joints = mano_layer(pose, translation)
    vertices = vertices.detach().cpu().numpy()[0]
    joints = joints.detach().cpu().numpy()[0]
    vertices = vertices @ camera_transform[:3, :3].T + camera_transform[:3, 3]
    joints = joints @ camera_transform[:3, :3].T + camera_transform[:3, 3]
    return np.ascontiguousarray(vertices), np.ascontiguousarray(joints)


def compute_wrist_trajectory(
    hand_pose_seq: np.ndarray,
    mano_layer: MANOLayer,
    camera_transform: np.ndarray,
) -> np.ndarray:
    points, _ = compute_wrist_trajectory_with_frames(
        hand_pose_seq, mano_layer, camera_transform
    )
    return points


def compute_joint_trajectory_with_frames(
    hand_pose_seq: np.ndarray,
    mano_layer: MANOLayer,
    camera_transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    joint_frames = []
    frame_ids = []
    for frame_id, hand_pose_frame in enumerate(hand_pose_seq):
        _, joints = compute_hand_geometry(hand_pose_frame, mano_layer, camera_transform)
        if joints is None:
            continue
        joint_frames.append(joints)
        frame_ids.append(frame_id)

    if not joint_frames:
        return (
            np.zeros((1, 21, 3), dtype=np.float32),
            np.zeros((1,), dtype=np.int32),
        )
    return (
        np.asarray(joint_frames, dtype=np.float32),
        np.asarray(frame_ids, dtype=np.int32),
    )


def compute_wrist_trajectory_with_frames(
    hand_pose_seq: np.ndarray,
    mano_layer: MANOLayer,
    camera_transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    joint_frames, frame_ids = compute_joint_trajectory_with_frames(
        hand_pose_seq, mano_layer, camera_transform
    )
    if joint_frames.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1,), dtype=np.int32)
    return (
        joint_frames[:, 0, :].astype(np.float32),
        frame_ids.astype(np.int32),
    )
