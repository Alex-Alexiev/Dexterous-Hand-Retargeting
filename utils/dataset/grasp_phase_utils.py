from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from utils.dataset.trajectory_utils import (
    cumulative_path_length,
    nth_vector_difference,
    normalized_consecutive_vectors,
    resample_trajectory_by_distance,
)

_MANO_FINGERTIP_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int32)


@dataclass(frozen=True)
class GraspPhaseFrames:
    approach_frame: int
    grasp_frame: int
    lift_frame: int


@dataclass(frozen=True)
class GraspPhaseDiagnostics:
    resampled_points: np.ndarray
    resampled_frame_coords: np.ndarray
    resampled_arc_lengths_m: np.ndarray
    derivative_vectors: np.ndarray
    derivative_magnitude: np.ndarray
    turn_vectors: np.ndarray
    turn_magnitude: np.ndarray
    approach_index: int
    grasp_index: int
    lift_index: int
    phase_frames: GraspPhaseFrames


@dataclass(frozen=True)
class GraspedObjectSelection:
    object_index: int
    reference_frame: int
    fingertip_distances_m: np.ndarray
    mean_fingertip_distance_m: float


@dataclass(frozen=True)
class GraspTrajectorySelection:
    phase_diagnostics: GraspPhaseDiagnostics
    grasped_object: GraspedObjectSelection


def _validate_trajectory_inputs(
    trajectory_points: np.ndarray,
    frame_coords: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(trajectory_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected trajectory_points shape (T, 3), got {points.shape}."
        )
    if points.shape[0] < 3:
        raise ValueError(
            "Need at least 3 wrist-trajectory points to detect grasp phases, "
            f"got {points.shape[0]}."
        )

    if frame_coords is None:
        coords = np.arange(points.shape[0], dtype=np.float32)
    else:
        coords = np.asarray(frame_coords, dtype=np.float32)
        if coords.shape != (points.shape[0],):
            raise ValueError(
                "Expected frame_coords to have shape "
                f"({points.shape[0]},), got {coords.shape}."
            )
    return points, coords


def _nearest_resampled_position(values: np.ndarray, target_value: float) -> int:
    return int(np.argmin(np.abs(values.astype(np.float32) - float(target_value))))


def _nearest_original_frame_id(frame_coords: np.ndarray, frame_coord: float) -> int:
    nearest_idx = int(
        np.argmin(np.abs(frame_coords.astype(np.float32) - float(frame_coord)))
    )
    return int(np.round(frame_coords[nearest_idx]))


def _select_phase_arc_lengths(
    grasp_arc_length_m: float,
    start_arc_length_m: float,
    end_arc_length_m: float,
    approach_offset_m: float,
    lift_offset_m: float,
) -> Tuple[float, float]:
    approach_arc_length_m = max(
        start_arc_length_m,
        grasp_arc_length_m - float(approach_offset_m),
    )
    lift_arc_length_m = min(
        end_arc_length_m,
        grasp_arc_length_m + float(lift_offset_m),
    )
    return float(approach_arc_length_m), float(lift_arc_length_m)


def _middle_arc_length_mask(arc_lengths_m: np.ndarray) -> np.ndarray:
    if arc_lengths_m.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    total_arc_length_m = float(arc_lengths_m[-1] - arc_lengths_m[0])
    start_arc_length_m = float(arc_lengths_m[0]) + 0.10 * total_arc_length_m
    end_arc_length_m = float(arc_lengths_m[0]) + 0.90 * total_arc_length_m
    return (
        (arc_lengths_m >= start_arc_length_m) & (arc_lengths_m <= end_arc_length_m)
    ).astype(bool)


def _interior_candidate_indices(
    arc_lengths_m: np.ndarray,
    *,
    local_center_m: Optional[float] = None,
    local_window_m: Optional[float] = None,
) -> np.ndarray:
    if arc_lengths_m.shape[0] < 3:
        return np.zeros((0,), dtype=np.int32)

    candidate_mask = np.zeros((arc_lengths_m.shape[0],), dtype=bool)
    candidate_mask[1:-1] = True
    candidate_mask &= _middle_arc_length_mask(arc_lengths_m)
    if local_center_m is not None and local_window_m is not None:
        candidate_mask &= (
            np.abs(arc_lengths_m.astype(np.float32) - float(local_center_m))
            <= float(local_window_m)
        )

    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0 and local_center_m is None:
        return np.arange(1, arc_lengths_m.shape[0] - 1, dtype=np.int32)
    if candidate_indices.size == 0 and local_center_m is not None:
        nearest_idx = int(
            np.argmin(np.abs(arc_lengths_m.astype(np.float32) - float(local_center_m)))
        )
        nearest_idx = int(np.clip(nearest_idx, 1, arc_lengths_m.shape[0] - 2))
        return np.asarray([nearest_idx], dtype=np.int32)
    return candidate_indices.astype(np.int32)


def _validate_joint_trajectory_inputs(
    hand_joint_trajectory: np.ndarray,
    frame_coords: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(hand_joint_trajectory, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(
            f"Expected hand_joint_trajectory shape (T, J, 3), got {joints.shape}."
        )
    if joints.shape[0] < 1 or joints.shape[1] <= int(_MANO_FINGERTIP_INDICES.max()):
        raise ValueError(
            "Need a valid MANO joint trajectory with fingertip joints to select a grasped object."
        )

    if frame_coords is None:
        coords = np.arange(joints.shape[0], dtype=np.float32)
    else:
        coords = np.asarray(frame_coords, dtype=np.float32)
        if coords.shape != (joints.shape[0],):
            raise ValueError(
                "Expected frame_coords to have shape "
                f"({joints.shape[0]},), got {coords.shape}."
            )
    return joints, coords


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        points.astype(np.float32) @ transform[:3, :3].T.astype(np.float32)
        + transform[:3, 3].astype(np.float32)
    ).astype(np.float32)


def _load_obj_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    if len(vertices) == 0:
        raise ValueError(f"Failed to parse vertices from {mesh_path}")
    return np.asarray(vertices, dtype=np.float32)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return np.concatenate([quat_xyzw[3:4], quat_xyzw[:3]]).astype(np.float32)


def _rotation_matrix_from_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / q_norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _make_transform(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rotation_matrix_from_quaternion_wxyz(
        _quat_xyzw_to_wxyz(quat_xyzw)
    )
    transform[:3, 3] = np.asarray(position, dtype=np.float32)
    return transform


def select_grasped_object(
    hand_joint_trajectory: np.ndarray,
    *,
    object_pose_seq: np.ndarray,
    object_mesh_files,
    camera_transform: np.ndarray,
    frame_coords: Optional[np.ndarray] = None,
    mesh_cache: Optional[dict] = None,
) -> GraspedObjectSelection:
    joint_frames, coords = _validate_joint_trajectory_inputs(hand_joint_trajectory, frame_coords)
    object_pose_seq = np.asarray(object_pose_seq, dtype=np.float32)
    if object_pose_seq.ndim != 3 or object_pose_seq.shape[2] < 7:
        raise ValueError(
            f"Expected object_pose_seq shape (T, O, 7+), got {object_pose_seq.shape}."
        )
    if object_pose_seq.shape[1] != len(object_mesh_files):
        raise ValueError(
            "object_pose_seq/object_mesh_files length mismatch: "
            f"{object_pose_seq.shape[1]} vs {len(object_mesh_files)}."
        )
    if object_pose_seq.shape[1] == 0:
        raise ValueError("Need at least one object in the scene to select a grasp target.")

    reference_frame = int(np.clip(np.round(coords[-1]), 0, object_pose_seq.shape[0] - 1))
    fingertip_points = joint_frames[-1, _MANO_FINGERTIP_INDICES, :].astype(np.float32)
    object_pose_frame = object_pose_seq[reference_frame]
    cache = {} if mesh_cache is None else mesh_cache

    best_object_index = 0
    best_distances = None
    best_score = float("inf")
    for object_index, mesh_file in enumerate(object_mesh_files):
        mesh_path = Path(mesh_file)
        if mesh_path not in cache:
            cache[mesh_path] = _load_obj_vertices(mesh_path)
        vertices = cache[mesh_path]
        object_transform = camera_transform @ _make_transform(
            object_pose_frame[object_index, 4:],
            object_pose_frame[object_index, :4],
        )
        object_vertices_world = _transform_points(vertices, object_transform)
        fingertip_to_vertex = (
            fingertip_points[:, None, :] - object_vertices_world[None, :, :]
        ).astype(np.float32)
        fingertip_distances = np.linalg.norm(fingertip_to_vertex, axis=2).min(axis=1)
        score = float(np.mean(fingertip_distances))
        if score < best_score:
            best_score = score
            best_object_index = object_index
            best_distances = fingertip_distances.astype(np.float32)

    return GraspedObjectSelection(
        object_index=int(best_object_index),
        reference_frame=reference_frame,
        fingertip_distances_m=best_distances,
        mean_fingertip_distance_m=float(best_score),
    )


def compute_grasp_phase_diagnostics(
    trajectory_points: np.ndarray,
    *,
    frame_coords: Optional[np.ndarray] = None,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
) -> GraspPhaseDiagnostics:
    points, coords = _validate_trajectory_inputs(trajectory_points, frame_coords)
    if resample_spacing_m <= 0.0:
        raise ValueError(f"Expected positive resample_spacing_m, got {resample_spacing_m}.")
    if refine_window_m <= 0.0:
        raise ValueError(f"Expected positive refine_window_m, got {refine_window_m}.")

    refined_spacing_m = (
        0.5 * float(resample_spacing_m)
        if refine_resample_spacing_m is None
        else float(refine_resample_spacing_m)
    )
    if refined_spacing_m <= 0.0:
        raise ValueError(
            f"Expected positive refine_resample_spacing_m, got {refined_spacing_m}."
        )

    coarse_points, coarse_frame_coords = resample_trajectory_by_distance(
        points,
        sample_coords=coords,
        spacing=resample_spacing_m,
    )
    if coarse_points.shape[0] < 3:
        raise ValueError(
            "Distance-resampled wrist trajectory is too short to detect grasp phases, "
            f"got {coarse_points.shape[0]} samples."
        )

    coarse_derivative_vectors = normalized_consecutive_vectors(coarse_points)
    coarse_turn_vectors = nth_vector_difference(coarse_derivative_vectors, inflection_stride)
    coarse_turn_magnitude = np.linalg.norm(coarse_turn_vectors, axis=1).astype(np.float32)

    if coarse_turn_magnitude.shape[0] < 3:
        raise ValueError(
            "Need at least 3 resampled trajectory points to localize an inflection point."
        )

    coarse_arc_lengths_m = cumulative_path_length(coarse_points)
    coarse_candidate_indices = _interior_candidate_indices(coarse_arc_lengths_m)
    coarse_grasp_index = int(
        coarse_candidate_indices[
            np.argmax(coarse_turn_magnitude[coarse_candidate_indices])
        ]
    )
    coarse_grasp_arc_length_m = float(coarse_arc_lengths_m[coarse_grasp_index])

    resampled_points, resampled_frame_coords = resample_trajectory_by_distance(
        points,
        sample_coords=coords,
        spacing=refined_spacing_m,
    )
    if resampled_points.shape[0] < 3:
        raise ValueError(
            "Refined wrist trajectory is too short to detect grasp phases, "
            f"got {resampled_points.shape[0]} samples."
        )

    derivative_vectors = normalized_consecutive_vectors(resampled_points)
    derivative_magnitude = np.linalg.norm(derivative_vectors, axis=1).astype(np.float32)
    turn_vectors = nth_vector_difference(derivative_vectors, inflection_stride)
    turn_magnitude = np.linalg.norm(turn_vectors, axis=1).astype(np.float32)

    if turn_magnitude.shape[0] < 3:
        raise ValueError(
            "Need at least 3 refined resampled trajectory points to localize an inflection point."
        )

    resampled_arc_lengths_m = cumulative_path_length(resampled_points)
    candidate_indices = _interior_candidate_indices(
        resampled_arc_lengths_m,
        local_center_m=coarse_grasp_arc_length_m,
        local_window_m=refine_window_m,
    )
    grasp_index = int(candidate_indices[np.argmax(turn_magnitude[candidate_indices])])

    approach_arc_length_m, lift_arc_length_m = _select_phase_arc_lengths(
        float(resampled_arc_lengths_m[grasp_index]),
        float(resampled_arc_lengths_m[0]),
        float(resampled_arc_lengths_m[-1]),
        approach_offset_m=approach_offset_m,
        lift_offset_m=lift_offset_m,
    )
    approach_index = _nearest_resampled_position(
        resampled_arc_lengths_m[: grasp_index + 1], approach_arc_length_m
    )
    lift_index = grasp_index + _nearest_resampled_position(
        resampled_arc_lengths_m[grasp_index:], lift_arc_length_m
    )

    phase_frames = GraspPhaseFrames(
        approach_frame=_nearest_original_frame_id(
            coords, float(resampled_frame_coords[approach_index])
        ),
        grasp_frame=_nearest_original_frame_id(
            coords, float(resampled_frame_coords[grasp_index])
        ),
        lift_frame=_nearest_original_frame_id(
            coords, float(resampled_frame_coords[lift_index])
        ),
    )
    return GraspPhaseDiagnostics(
        resampled_points=resampled_points.astype(np.float32),
        resampled_frame_coords=resampled_frame_coords.astype(np.float32),
        resampled_arc_lengths_m=resampled_arc_lengths_m.astype(np.float32),
        derivative_vectors=derivative_vectors.astype(np.float32),
        derivative_magnitude=derivative_magnitude.astype(np.float32),
        turn_vectors=turn_vectors.astype(np.float32),
        turn_magnitude=turn_magnitude.astype(np.float32),
        approach_index=approach_index,
        grasp_index=grasp_index,
        lift_index=lift_index,
        phase_frames=phase_frames,
    )


def compute_grasp_phase_and_object_selection(
    trajectory_points: np.ndarray,
    *,
    hand_joint_trajectory: np.ndarray,
    object_pose_seq: np.ndarray,
    object_mesh_files,
    camera_transform: np.ndarray,
    frame_coords: Optional[np.ndarray] = None,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
    mesh_cache: Optional[dict] = None,
) -> GraspTrajectorySelection:
    phase_diagnostics = compute_grasp_phase_diagnostics(
        trajectory_points,
        frame_coords=frame_coords,
        resample_spacing_m=resample_spacing_m,
        inflection_stride=inflection_stride,
        approach_offset_m=approach_offset_m,
        lift_offset_m=lift_offset_m,
        refine_window_m=refine_window_m,
        refine_resample_spacing_m=refine_resample_spacing_m,
    )
    grasped_object = select_grasped_object(
        hand_joint_trajectory,
        object_pose_seq=object_pose_seq,
        object_mesh_files=object_mesh_files,
        camera_transform=camera_transform,
        frame_coords=frame_coords,
        mesh_cache=mesh_cache,
    )
    return GraspTrajectorySelection(
        phase_diagnostics=phase_diagnostics,
        grasped_object=grasped_object,
    )


def detect_grasp_phase_frames(
    trajectory_points: np.ndarray,
    *,
    frame_coords: Optional[np.ndarray] = None,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
) -> GraspPhaseFrames:
    return compute_grasp_phase_diagnostics(
        trajectory_points,
        frame_coords=frame_coords,
        resample_spacing_m=resample_spacing_m,
        inflection_stride=inflection_stride,
        approach_offset_m=approach_offset_m,
        lift_offset_m=lift_offset_m,
        refine_window_m=refine_window_m,
        refine_resample_spacing_m=refine_resample_spacing_m,
    ).phase_frames


def detect_grasp_phase_arrays(
    trajectory_points: np.ndarray,
    *,
    frame_coords: Optional[np.ndarray] = None,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = compute_grasp_phase_diagnostics(
        trajectory_points,
        frame_coords=frame_coords,
        resample_spacing_m=resample_spacing_m,
        inflection_stride=inflection_stride,
        approach_offset_m=approach_offset_m,
        lift_offset_m=lift_offset_m,
        refine_window_m=refine_window_m,
        refine_resample_spacing_m=refine_resample_spacing_m,
    )
    return (
        diagnostics.resampled_arc_lengths_m,
        diagnostics.derivative_magnitude,
        diagnostics.turn_magnitude,
    )


def plot_grasp_phase_diagnostics(
    trajectory_points: np.ndarray,
    *,
    frame_coords: Optional[np.ndarray] = None,
    resample_spacing_m: float = 0.01,
    inflection_stride: int = 4,
    approach_offset_m: float = 0.05,
    lift_offset_m: float = 0.05,
    refine_window_m: float = 0.10,
    refine_resample_spacing_m: Optional[float] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
):
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting grasp diagnostics requires: pip install matplotlib") from exc

    diagnostics = compute_grasp_phase_diagnostics(
        trajectory_points,
        frame_coords=frame_coords,
        resample_spacing_m=resample_spacing_m,
        inflection_stride=inflection_stride,
        approach_offset_m=approach_offset_m,
        lift_offset_m=lift_offset_m,
        refine_window_m=refine_window_m,
        refine_resample_spacing_m=refine_resample_spacing_m,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = diagnostics.resampled_arc_lengths_m

    axes[0].plot(x, diagnostics.derivative_magnitude, color="#1f77b4", linewidth=2.0)
    axes[0].set_ylabel("|dp / ds|")
    axes[0].set_title(title or "Grasp Phase Diagnostics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, diagnostics.turn_magnitude, color="#d62728", linewidth=2.0)
    axes[1].set_ylabel("|delta(dp / ds)|")
    axes[1].set_xlabel("Arc Length (m)")
    axes[1].grid(True, alpha=0.3)

    phase_markers = [
        ("approach", diagnostics.approach_index, "#2ca02c"),
        ("grasp", diagnostics.grasp_index, "#ff7f0e"),
        ("lift", diagnostics.lift_index, "#9467bd"),
    ]
    for axis in axes:
        for label, index, color in phase_markers:
            axis.axvline(x[index], color=color, linestyle="--", linewidth=1.5, label=label)
        axis.set_xlim(float(x[0]), float(x[-1]))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:3], labels[:3], loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, diagnostics.phase_frames
