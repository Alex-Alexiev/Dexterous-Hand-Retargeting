from typing import Optional, Tuple

import numpy as np

_EPS = 1e-8


def cumulative_path_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1).astype(np.float32)
    return np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )


def resample_trajectory_by_distance(
    points: np.ndarray,
    *,
    sample_coords: Optional[np.ndarray] = None,
    spacing: Optional[float] = None,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected points shape (T, D), got {points.shape}")
    if points.shape[0] == 0:
        empty_coords = np.zeros((0,), dtype=np.float32)
        return points.astype(np.float32), empty_coords

    if sample_coords is None:
        sample_coords = np.arange(points.shape[0], dtype=np.float32)
    else:
        sample_coords = np.asarray(sample_coords, dtype=np.float32)
        if sample_coords.shape != (points.shape[0],):
            raise ValueError(
                "Expected sample_coords to have shape "
                f"({points.shape[0]},), got {sample_coords.shape}"
            )

    if spacing is not None and spacing <= 0.0:
        raise ValueError(f"Expected positive spacing, got {spacing}")

    if num_samples is None and spacing is None:
        num_samples = points.shape[0]
    if num_samples is not None:
        num_samples = max(1, int(num_samples))
    if points.shape[0] == 1 or num_samples == 1:
        return points[[0]].astype(np.float32), np.asarray([sample_coords[0]], dtype=np.float32)

    cumulative_distance = cumulative_path_length(points)
    keep_mask = np.concatenate(
        [np.asarray([True]), np.diff(cumulative_distance) > _EPS]
    )
    kept_points = points[keep_mask]
    kept_coords = sample_coords[keep_mask]
    kept_distance = cumulative_distance[keep_mask]

    if kept_points.shape[0] == 1 or float(kept_distance[-1]) <= _EPS:
        repeat_count = num_samples if num_samples is not None else 1
        repeated_points = np.repeat(kept_points[:1], repeat_count, axis=0)
        repeated_coords = np.repeat(kept_coords[:1], repeat_count, axis=0)
        return repeated_points.astype(np.float32), repeated_coords.astype(np.float32)

    if spacing is not None:
        target_distance = np.arange(
            0.0,
            float(kept_distance[-1]) + 0.5 * float(spacing),
            float(spacing),
            dtype=np.float32,
        )
        if target_distance.shape[0] == 0 or abs(float(target_distance[-1]) - float(kept_distance[-1])) > _EPS:
            target_distance = np.concatenate(
                [target_distance, np.asarray([float(kept_distance[-1])], dtype=np.float32)]
            )
    else:
        target_distance = np.linspace(
            0.0,
            float(kept_distance[-1]),
            num_samples,
            dtype=np.float32,
        )
    resampled_points = np.stack(
        [
            np.interp(target_distance, kept_distance, kept_points[:, dim])
            for dim in range(kept_points.shape[1])
        ],
        axis=1,
    ).astype(np.float32)
    resampled_coords = np.interp(target_distance, kept_distance, kept_coords).astype(np.float32)
    return resampled_points, resampled_coords


def normalized_consecutive_vectors(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.asarray(points, dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros_like(points, dtype=np.float32)

    diffs = np.diff(points, axis=0).astype(np.float32)
    diff_norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    unit_diffs = diffs / np.maximum(diff_norms, _EPS)
    return np.concatenate([unit_diffs, unit_diffs[-1:]], axis=0).astype(np.float32)


def nth_vector_difference(vectors: np.ndarray, stride: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.asarray(vectors, dtype=np.float32)

    stride = max(1, int(stride))
    if vectors.shape[0] <= stride:
        return np.zeros_like(vectors, dtype=np.float32)

    differences = np.zeros_like(vectors, dtype=np.float32)
    left_offset = max(1, stride // 2)
    right_offset = max(1, stride - left_offset)
    for idx in range(vectors.shape[0]):
        prev_idx = max(0, idx - left_offset)
        next_idx = min(vectors.shape[0] - 1, idx + right_offset)
        differences[idx] = vectors[next_idx] - vectors[prev_idx]
    return differences.astype(np.float32)
