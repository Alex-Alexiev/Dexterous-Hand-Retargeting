from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from pytransform3d import rotations
from utils.dataset.trajectory_utils import (
    nth_vector_difference,
    normalized_consecutive_vectors,
    resample_trajectory_by_distance,
)

HAND_SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
HAND_MESH_COLOR = (160, 175, 255)
OBJECT_COLORS = [
    (242, 139, 130),
    (251, 188, 5),
    (52, 168, 83),
    (66, 133, 244),
    (171, 71, 188),
]


def load_obj_tri_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with mesh_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                face_indices = []
                for token in line.split()[1:]:
                    idx = int(token.split("/")[0])
                    face_indices.append(idx - 1 if idx > 0 else len(vertices) + idx)
                for i in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Failed to parse triangle mesh from {mesh_path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[3:4], quat_xyzw[:3]])


def make_transform(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotations.matrix_from_quaternion(quat_wxyz).astype(np.float32)
    transform[:3, 3] = position.astype(np.float32)
    return transform


def segments_from_edges(points: np.ndarray, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(edges) == 0:
        return np.zeros((1, 2, 3), dtype=np.float32)
    return np.asarray([[points[i], points[j]] for i, j in edges], dtype=np.float32)


def segments_from_polyline(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros((1, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


def trajectory_velocity_and_turn_vectors(
    points: np.ndarray,
    *,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        zeros = np.zeros_like(points, dtype=np.float32)
        return points.astype(np.float32), zeros, zeros

    resampled_points, _ = resample_trajectory_by_distance(
        points,
        spacing=resample_spacing_m,
    )
    velocity_unit = normalized_consecutive_vectors(resampled_points)
    turn_vectors = nth_vector_difference(velocity_unit, inflection_stride)
    return (
        resampled_points.astype(np.float32),
        velocity_unit.astype(np.float32),
        turn_vectors.astype(np.float32),
    )


def vector_segments_from_origins(
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    scale: float = 0.03,
    stride: int = 1,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    if (
        origins.ndim != 2
        or vectors.ndim != 2
        or origins.shape != vectors.shape
        or origins.shape[0] == 0
    ):
        return np.zeros((1, 2, 3), dtype=np.float32)

    stride = max(1, int(stride))
    sampled_origins = origins[::stride]
    sampled_vectors = vectors[::stride]
    endpoints = sampled_origins + float(scale) * sampled_vectors
    return np.stack([sampled_origins, endpoints], axis=1).astype(np.float32)
