import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dex_retargeting.robot_wrapper import RobotWrapper

_NUM_MANO_POINTS = 21
_MANO_PARENT_IDX = np.asarray(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)
_MANO_CHILD_IDX = np.asarray(
    [9, 2, 3, 4, -1, 6, 7, 8, -1, 10, 11, 12, -1, 14, 15, 16, -1, 18, 19, 20, -1],
    dtype=np.int32,
)
_EPS = 1e-9
_DEFAULT_REFERENCE_POSE = np.asarray(
    [
        -1.0,
        -1.5,
        0.5,
        0.1,
        -0.3,
        1.2,
        0.7,
        0.5,
        0.0,
        1.2,
        0.8,
        0.5,
        0.4,
        1.2,
        0.9,
        0.5,
        0.6,
        1.2,
        1.0,
        0.5,
    ],
    dtype=np.float64,
)


def get_default_reference_pose() -> np.ndarray:
    """Hardcoded fallback reference pose (indexed like the non-locked robot joints)."""
    return _DEFAULT_REFERENCE_POSE.copy()


def load_reference_pose(
    path: Optional[str], expected_dim: Optional[int] = None
) -> Tuple[np.ndarray, Optional[float]]:
    """Load the optimizer's default/reference pose (+ optional reg weight) from a JSON file.

    Falls back to ``_DEFAULT_REFERENCE_POSE`` (reg_weight None) when ``path`` is None, missing,
    or unreadable. When ``expected_dim`` is given, the pose is padded/truncated to that length so
    it always matches the robot's DoF ordering.
    """
    pose = _DEFAULT_REFERENCE_POSE.copy()
    reg_weight: Optional[float] = None
    if path is not None:
        p = Path(path).expanduser()
        if p.is_file():
            try:
                with p.open("r") as f:
                    payload = json.load(f)
                raw = payload.get("reference_pose")
                if raw is not None:
                    pose = np.asarray(raw, dtype=np.float64).reshape(-1)
                if payload.get("reg_weight") is not None:
                    reg_weight = float(payload["reg_weight"])
            except Exception:
                pose = _DEFAULT_REFERENCE_POSE.copy()
                reg_weight = None
    if expected_dim is not None and pose.shape[0] != expected_dim:
        fitted = np.zeros((int(expected_dim),), dtype=np.float64)
        m = min(int(expected_dim), pose.shape[0])
        fitted[:m] = pose[:m]
        pose = fitted
    return pose, reg_weight


def save_reference_pose(
    path: str,
    pose: np.ndarray,
    reg_weight: Optional[float] = None,
    joint_names: Optional[Sequence[str]] = None,
) -> str:
    """Write the default/reference pose (+ reg weight, joint names) to ``path`` as JSON.

    Returns the resolved absolute path written.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_pose": [
            float(v) for v in np.asarray(pose, dtype=np.float64).reshape(-1)
        ],
    }
    if reg_weight is not None:
        payload["reg_weight"] = float(reg_weight)
    if joint_names is not None:
        payload["joint_names"] = [str(n) for n in joint_names]
    with p.open("w") as f:
        json.dump(payload, f, indent=2)
    return str(p.resolve())


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec64 = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec64))
    if norm < _EPS:
        return np.zeros((3,), dtype=np.float64)
    return vec64 / norm


def _rotation_from_x_prev_z(x_vec: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
    x_axis = _normalize(x_vec)
    if float(np.linalg.norm(x_axis)) < _EPS:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    z_axis_prev = _normalize(z_prev)
    if float(np.linalg.norm(z_axis_prev)) < _EPS:
        z_axis_prev = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    # Rule requested by user: y = x × z_prev
    y_axis = np.cross(x_axis, z_axis_prev)
    if float(np.linalg.norm(y_axis)) < _EPS:
        alt_z = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, alt_z))) > 0.9:
            alt_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(x_axis, alt_z)
    y_axis = _normalize(y_axis)

    # Right-handed frame.
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, z_axis_prev)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    return np.stack([x_axis, y_axis, z_axis], axis=1)


@dataclass
class _MappingCache:
    path: str
    link_indices_by_mano: np.ndarray  # (21,) with -1 for unmapped
    offsets_by_mano: np.ndarray  # (21, 3) in mapped link local frame


_MAPPING_CACHE: Dict[str, _MappingCache] = {}
_LAST_QPOS_CACHE: Dict[Tuple[int, Tuple[str, ...], str], np.ndarray] = {}
_ROOT_REF_TF_CACHE: Dict[Tuple[int, str], np.ndarray] = {}
_DEFAULT_HR_TRANSLATION = np.asarray([0.05, 0.0, 0.0], dtype=np.float64)
_OPTIMIZE_LOCK = threading.Lock()


def _extract_dummy_joint_indices(robot_joint_names: Sequence[str]) -> Dict[str, int]:
    """Find dummy translation/rotation joint indices in the active q ordering."""
    idx_map: Dict[str, int] = {}
    for idx, name in enumerate(robot_joint_names):
        lower_name = name.lower()
        if "dummy_x_translation_joint" in lower_name:
            idx_map["tx"] = idx
        elif "dummy_y_translation_joint" in lower_name:
            idx_map["ty"] = idx
        elif "dummy_z_translation_joint" in lower_name:
            idx_map["tz"] = idx
        elif "dummy_x_rotation_joint" in lower_name:
            idx_map["rx"] = idx
        elif "dummy_y_rotation_joint" in lower_name:
            idx_map["ry"] = idx
        elif "dummy_z_rotation_joint" in lower_name:
            idx_map["rz"] = idx
    return idx_map


def _build_reference_q(
    robot_joint_names: Sequence[str], locked_dummy_indices: np.ndarray,
    reference_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build full-q reference using the (optionally overridden) N-DoF target pose.

    ``reference_pose`` is indexed in the same non-locked joint order as
    ``_DEFAULT_REFERENCE_POSE``; when None the module default is used.
    """
    dof = len(robot_joint_names)
    q_ref = np.zeros((dof,), dtype=np.float64)
    locked = set(int(i) for i in locked_dummy_indices.tolist())
    src = (
        _DEFAULT_REFERENCE_POSE if reference_pose is None
        else np.asarray(reference_pose, dtype=np.float64).reshape(-1)
    )

    src_idx = 0
    for dst_idx in range(dof):
        if dst_idx in locked:
            q_ref[dst_idx] = 0.0
            continue
        if src_idx < src.shape[0]:
            q_ref[dst_idx] = src[src_idx]
            src_idx += 1
        else:
            q_ref[dst_idx] = 0.0
    return q_ref


def _single_link_orientation_residual(
    q_active: np.ndarray,
    *,
    q_base: np.ndarray,
    active_indices: np.ndarray,
    robot_model: RobotWrapper,
    link_index: int,
    target_rotation: np.ndarray,
    q_ref_active: np.ndarray,
    sqrt_reg_weight: float,
) -> np.ndarray:
    q = q_base.copy()
    q[active_indices] = q_active
    robot_model.compute_forward_kinematics(q)
    rot_cur = robot_model.get_link_pose(link_index)[:3, :3].astype(np.float64)
    rot_err = target_rotation @ rot_cur.T
    rot_residual = Rotation.from_matrix(rot_err).as_rotvec()
    reg_residual = sqrt_reg_weight * (q_active - q_ref_active)
    return np.concatenate([rot_residual, reg_residual], axis=0)


def _compute_mano_frame_rotations(points: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rots = np.tile(np.eye(3, dtype=np.float64), (_NUM_MANO_POINTS, 1, 1))
    if points.shape != (_NUM_MANO_POINTS, 3):
        return rots

    has_root = bool(valid_mask[0])
    if not has_root:
        return rots

    def _vec_or_default(i: int, j: int, default: np.ndarray) -> np.ndarray:
        if valid_mask[i] and valid_mask[j]:
            v = points[j] - points[i]
            if float(np.linalg.norm(v)) >= _EPS:
                return v
        return default

    root_forward = _vec_or_default(0, 9, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    if float(np.linalg.norm(root_forward)) < _EPS:
        root_forward = _vec_or_default(0, 1, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    root_lateral = _vec_or_default(5, 13, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    palm_z = np.cross(root_forward, root_lateral)
    if float(np.linalg.norm(palm_z)) < _EPS:
        palm_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rots[0] = _rotation_from_x_prev_z(root_forward, palm_z)

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
        prev_z = rots[parent, :, 2]
        rots[idx] = _rotation_from_x_prev_z(forward, prev_z)
    return rots


def _load_mapping(
    mapping_path: str, robot_model: RobotWrapper
) -> _MappingCache:
    resolved = str(Path(mapping_path).expanduser().resolve())
    cached = _MAPPING_CACHE.get(resolved)
    if cached is not None:
        return cached

    link_indices_by_mano = np.full((_NUM_MANO_POINTS,), -1, dtype=np.int32)
    offsets_by_mano = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float64)
    mapping_file = Path(resolved)
    if mapping_file.is_file():
        try:
            with mapping_file.open("r") as f:
                payload = json.load(f)
            assignments = payload.get("mano_link_assignments", [])
        except Exception:
            assignments = []

        link_name_to_frame = {name: i for i, name in enumerate(robot_model.link_names)}
        for entry in assignments:
            try:
                point_idx = int(entry["mano_point_index"])
                link_name = str(entry["link_name"])
                offset = np.asarray(entry.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
            except Exception:
                continue
            if point_idx < 0 or point_idx >= _NUM_MANO_POINTS:
                continue
            frame_idx = link_name_to_frame.get(link_name, -1)
            if frame_idx < 0 or offset.shape != (3,):
                continue
            link_indices_by_mano[point_idx] = frame_idx
            offsets_by_mano[point_idx] = offset

    mapping = _MappingCache(
        path=resolved,
        link_indices_by_mano=link_indices_by_mano,
        offsets_by_mano=offsets_by_mano,
    )
    _MAPPING_CACHE[resolved] = mapping
    return mapping


def _compute_robot_mano_points(
    robot_model: RobotWrapper,
    link_indices_by_mano: np.ndarray,
    offsets_by_mano: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float64)
    valid = link_indices_by_mano >= 0
    for i in range(_NUM_MANO_POINTS):
        frame_idx = int(link_indices_by_mano[i])
        if frame_idx < 0:
            continue
        pose = robot_model.get_link_pose(frame_idx).astype(np.float64)
        points[i] = pose[:3, :3] @ offsets_by_mano[i] + pose[:3, 3]
    return points, valid


def _collect_mapped_edges(valid_mask: np.ndarray) -> np.ndarray:
    edges: List[int] = []
    for idx in range(1, _NUM_MANO_POINTS):
        parent = int(_MANO_PARENT_IDX[idx])
        if parent < 0:
            continue
        if valid_mask[idx] and valid_mask[parent]:
            edges.append(idx)
    return np.asarray(edges, dtype=np.int32)


def _collect_active_indices_for_frames(
    robot_model: RobotWrapper,
    q_ref: np.ndarray,
    frame_indices: np.ndarray,
    locked_indices: np.ndarray,
) -> np.ndarray:
    active_mask = np.zeros((q_ref.shape[0],), dtype=bool)
    robot_model.compute_forward_kinematics(q_ref)
    for frame_idx in np.unique(frame_indices):
        if int(frame_idx) < 0:
            continue
        jac = robot_model.compute_single_link_local_jacobian(q_ref, int(frame_idx)).astype(np.float64)
        ang_jac = jac[3:6, :]
        active_mask |= np.linalg.norm(ang_jac, axis=0) > 1e-10
    if locked_indices.size > 0:
        active_mask[locked_indices] = False
    return np.where(active_mask)[0].astype(np.int32)


def _mano_relative_orientation_residual(
    q_active: np.ndarray,
    *,
    q_base: np.ndarray,
    active_indices: np.ndarray,
    robot_model: RobotWrapper,
    mapped_link_indices: np.ndarray,
    mapped_offsets: np.ndarray,
    mapped_edge_indices: np.ndarray,
    hand_frame_rots: np.ndarray,
    q_ref_active: np.ndarray,
    sqrt_reg_weight: float,
) -> np.ndarray:
    q_full = q_base.copy()
    q_full[active_indices] = q_active
    robot_model.compute_forward_kinematics(q_full)

    robot_points, robot_valid = _compute_robot_mano_points(
        robot_model, mapped_link_indices, mapped_offsets
    )
    robot_rots = _compute_mano_frame_rotations(robot_points, robot_valid)

    residuals: List[np.ndarray] = []
    for idx in mapped_edge_indices:
        parent = int(_MANO_PARENT_IDX[idx])
        delta_h = hand_frame_rots[parent].T @ hand_frame_rots[idx]
        delta_r = robot_rots[parent].T @ robot_rots[idx]
        rot_err = delta_h @ delta_r.T
        residuals.append(Rotation.from_matrix(rot_err).as_rotvec())

    residuals.append(sqrt_reg_weight * (q_active - q_ref_active))
    return np.concatenate(residuals, axis=0)


def _solve_single_link_debug(
    *,
    robot_joint_names: Sequence[str],
    robot_model: RobotWrapper,
    link_idx: int,
    target_rot: np.ndarray,
) -> np.ndarray:
    dof = len(robot_joint_names)
    cache_key = (id(robot_model), tuple(robot_joint_names), f"single_frame_scipy_{link_idx}")
    dummy_idx_map = _extract_dummy_joint_indices(robot_joint_names)
    locked_dummy_indices = np.asarray(sorted(dummy_idx_map.values()), dtype=np.int32)
    q_ref_full = _build_reference_q(robot_joint_names, locked_dummy_indices)
    q_init = q_ref_full.copy()

    robot_model.compute_forward_kinematics(q_init)
    jac = robot_model.compute_single_link_local_jacobian(q_init, link_idx).astype(np.float64)
    ang_jac = jac[3:6, :]
    influential = np.linalg.norm(ang_jac, axis=0) > 1e-10
    active_mask = influential.copy()
    if locked_dummy_indices.size > 0:
        active_mask[locked_dummy_indices] = False
    active_indices = np.where(active_mask)[0].astype(np.int32)
    if active_indices.size == 0:
        _LAST_QPOS_CACHE[cache_key] = q_init.copy()
        return q_init.astype(np.float32)

    reg_weight = 1e-4
    sqrt_reg_weight = float(np.sqrt(reg_weight))
    q_active_init = q_init[active_indices].copy()
    q_ref_active = q_ref_full[active_indices].copy()

    joint_limits = robot_model.joint_limits.astype(np.float64)
    lbx = joint_limits[active_indices, 0].copy()
    ubx = joint_limits[active_indices, 1].copy()
    bad_bounds = lbx >= ubx
    if np.any(bad_bounds):
        lbx[bad_bounds] = -np.inf
        ubx[bad_bounds] = np.inf

    with _OPTIMIZE_LOCK:
        try:
            result = least_squares(
                _single_link_orientation_residual,
                x0=q_active_init,
                bounds=(lbx, ubx),
                method="trf",
                max_nfev=40,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                kwargs={
                    "q_base": q_init,
                    "active_indices": active_indices,
                    "robot_model": robot_model,
                    "link_index": link_idx,
                    "target_rotation": target_rot,
                    "q_ref_active": q_ref_active,
                    "sqrt_reg_weight": sqrt_reg_weight,
                },
            )
            q_opt = q_init.copy()
            q_opt[active_indices] = np.asarray(result.x, dtype=np.float64).reshape(
                active_indices.shape[0]
            )
            if locked_dummy_indices.size > 0:
                q_opt[locked_dummy_indices] = q_init[locked_dummy_indices]
        except Exception:
            q_opt = q_init.copy()

    _LAST_QPOS_CACHE[cache_key] = q_opt.copy()
    return q_opt.astype(np.float32)



def retarget_hand_to_robot_joints(
    hand_keypoints: np.ndarray,
    hand_joint_values: np.ndarray,
    robot_joint_names: Sequence[str],
    robot_model: Optional[RobotWrapper] = None,
    mapping_path: Optional[str] = None,
    mapping_link_indices: Optional[np.ndarray] = None,
    mapping_offsets: Optional[np.ndarray] = None,
    hand_to_robot_translation: Optional[np.ndarray] = None,
    hand_to_robot_rotation: Optional[np.ndarray] = None,
    debug_link_index: Optional[int] = None,
    debug_target_rotation_local: Optional[np.ndarray] = None,
    reference_pose: Optional[np.ndarray] = None,
    reg_weight: Optional[float] = None,
) -> np.ndarray:
    """Retarget robot joints from MANO orientation targets.

    Args:
        hand_keypoints: Human hand keypoint positions, shape (21, 3).
        hand_joint_values: Human hand joint/pose values, e.g. MANO pose params.
        robot_joint_names: Target robot joint ordering.
        robot_model: Pinocchio-backed robot wrapper for FK/Jacobians.
        mapping_path: JSON mapping from MANO points to robot links + offsets.
        mapping_link_indices: Optional live mapping frame indices, shape (21,).
        mapping_offsets: Optional live mapping offsets in link frames, shape (21, 3).
        hand_to_robot_translation: Unused (base pose handled in viewer/root frame).
        hand_to_robot_rotation: Unused (base pose handled in viewer/root frame).
        debug_link_index: Fallback debug link index when mapping is unavailable.
        debug_target_rotation_local: Fallback desired link rotation for debug mode.

    Returns:
        Robot qpos in the same order as ``robot_joint_names``.
    """
    dof = len(robot_joint_names)
    _ = hand_joint_values
    _ = hand_to_robot_translation
    _ = hand_to_robot_rotation

    if robot_model is None:
        return np.zeros((dof,), dtype=np.float32)

    live_mapping_ready = (
        mapping_link_indices is not None
        and mapping_offsets is not None
        and np.asarray(mapping_link_indices).shape == (_NUM_MANO_POINTS,)
        and np.asarray(mapping_offsets).shape == (_NUM_MANO_POINTS, 3)
    )

    # Primary path: mapped MANO-relative orientation optimization.
    if live_mapping_ready or (mapping_path is not None and Path(mapping_path).expanduser().is_file()):
        hand_points = np.asarray(hand_keypoints, dtype=np.float64)
        if hand_points.shape != (_NUM_MANO_POINTS, 3):
            return np.zeros((dof,), dtype=np.float32)

        if live_mapping_ready:
            link_indices_by_mano = np.asarray(mapping_link_indices, dtype=np.int32).copy()
            offsets_by_mano = np.asarray(mapping_offsets, dtype=np.float64).copy()
            mapping_key = "live_mapping"
        else:
            mapping = _load_mapping(mapping_path, robot_model)
            link_indices_by_mano = mapping.link_indices_by_mano
            offsets_by_mano = mapping.offsets_by_mano
            mapping_key = mapping.path

        mapped_valid = link_indices_by_mano >= 0
        mapped_edges = _collect_mapped_edges(mapped_valid)
        if mapped_edges.size == 0:
            return np.zeros((dof,), dtype=np.float32)

        hand_valid = np.ones((_NUM_MANO_POINTS,), dtype=bool)
        hand_rots = _compute_mano_frame_rotations(hand_points, hand_valid)

        cache_key = (
            id(robot_model),
            tuple(robot_joint_names),
            f"mano_relative_{mapping_key}",
        )
        dummy_idx_map = _extract_dummy_joint_indices(robot_joint_names)
        locked_dummy_indices = np.asarray(sorted(dummy_idx_map.values()), dtype=np.int32)
        q_ref_full = _build_reference_q(
            robot_joint_names, locked_dummy_indices, reference_pose
        )
        q_init = q_ref_full.copy()

        mapped_frame_indices = link_indices_by_mano[mapped_valid].astype(np.int32)
        active_indices = _collect_active_indices_for_frames(
            robot_model,
            q_init,
            mapped_frame_indices,
            locked_dummy_indices,
        )
        if active_indices.size == 0:
            _LAST_QPOS_CACHE[cache_key] = q_init.copy()
            return q_init.astype(np.float32)

        reg_weight_val = 1e-3 if reg_weight is None else float(reg_weight)
        sqrt_reg_weight = float(np.sqrt(max(reg_weight_val, 0.0)))
        q_active_init = q_init[active_indices].copy()
        q_ref_active = q_ref_full[active_indices].copy()

        joint_limits = robot_model.joint_limits.astype(np.float64)
        lbx = joint_limits[active_indices, 0].copy()
        ubx = joint_limits[active_indices, 1].copy()
        bad_bounds = lbx >= ubx
        if np.any(bad_bounds):
            lbx[bad_bounds] = -np.inf
            ubx[bad_bounds] = np.inf

        # Keep the warm-start strictly inside the bounds. A reference/default pose value can sit
        # exactly on a joint limit (or, after a float32 round-trip, a hair past it), which makes
        # least_squares reject x0 with "Initial guess is outside of provided bounds" -> the solve
        # is skipped in the except below and EVERY frame collapses onto the reference pose.
        q_active_init = np.clip(q_active_init, lbx, ubx)

        with _OPTIMIZE_LOCK:
            try:
                result = least_squares(
                    _mano_relative_orientation_residual,
                    x0=q_active_init,
                    bounds=(lbx, ubx),
                    method="trf",
                    max_nfev=80,
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                    kwargs={
                        "q_base": q_init,
                        "active_indices": active_indices,
                        "robot_model": robot_model,
                        "mapped_link_indices": link_indices_by_mano,
                        "mapped_offsets": offsets_by_mano,
                        "mapped_edge_indices": mapped_edges,
                        "hand_frame_rots": hand_rots,
                        "q_ref_active": q_ref_active,
                        "sqrt_reg_weight": sqrt_reg_weight,
                    },
                )
                q_opt = q_init.copy()
                q_opt[active_indices] = np.asarray(result.x, dtype=np.float64).reshape(
                    active_indices.shape[0]
                )
                if locked_dummy_indices.size > 0:
                    q_opt[locked_dummy_indices] = q_init[locked_dummy_indices]
            except Exception as _exc:
                import os as _os
                if _os.environ.get("RETARGET_DEBUG"):
                    print(f"[RT] least_squares EXC: {type(_exc).__name__}: {_exc}", flush=True)
                q_opt = q_init.copy()

        _LAST_QPOS_CACHE[cache_key] = q_opt.copy()
        return q_opt.astype(np.float32)

    # Fallback: single-link orientation debug mode.
    if debug_link_index is None or debug_target_rotation_local is None:
        return np.zeros((dof,), dtype=np.float32)
    link_idx = int(debug_link_index)
    if link_idx < 0 or link_idx >= len(robot_model.link_names):
        return np.zeros((dof,), dtype=np.float32)
    target_rot = np.asarray(debug_target_rotation_local, dtype=np.float64)
    if target_rot.shape != (3, 3):
        return np.zeros((dof,), dtype=np.float32)
    return _solve_single_link_debug(
        robot_joint_names=robot_joint_names,
        robot_model=robot_model,
        link_idx=link_idx,
        target_rot=target_rot,
    )
