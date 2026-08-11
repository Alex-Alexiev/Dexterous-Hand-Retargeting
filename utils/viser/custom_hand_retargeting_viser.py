import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pytransform3d import rotations
from scipy.spatial.transform import Rotation as ScipyRotation

from utils.dataset.dataset import YCB_CLASSES
from utils.robot.urdf_utils import build_temp_urdf, resolve_visual_urdf
from utils.viser.scene_utils import (
    HAND_MESH_COLOR as _HAND_MESH_COLOR,
    HAND_SKELETON_EDGES as _HAND_SKELETON_EDGES,
    OBJECT_COLORS as _OBJECT_COLORS,
    load_obj_tri_mesh,
    make_transform,
    quat_xyzw_to_wxyz,
    segments_from_edges,
    segments_from_polyline,
)
from utils.viser.viser_urdf_viewer import ViserHandViewer

_NUM_MANO_POINTS = 21
_MANO_PARENT_IDX = np.asarray(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)
_MANO_CHILD_IDX = np.asarray(
    [9, 2, 3, 4, -1, 6, 7, 8, -1, 10, 11, 12, -1, 14, 15, 16, -1, 18, 19, 20, -1],
    dtype=np.int32,
)


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec64 = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec64))
    if norm < eps:
        return np.zeros((3,), dtype=np.float64)
    return vec64 / norm


def _quat_from_forward_prev_z(
    forward: np.ndarray, prev_z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x_axis = _normalize(forward)
    if float(np.linalg.norm(x_axis)) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    z_prev = _normalize(prev_z)
    if float(np.linalg.norm(z_prev)) < 1e-8:
        z_prev = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Requested rule: y = x cross z_prev
    y_axis = np.cross(x_axis, z_prev)
    if float(np.linalg.norm(y_axis)) < 1e-8:
        alt_z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, alt_z))) > 0.9:
            alt_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(x_axis, alt_z)
    y_axis = _normalize(y_axis)

    # Rebuild z with a right-handed basis: z = x cross y.
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, z_prev)) < 0.0:
        # Flip y/z together to keep right-handedness while matching prev_z direction.
        y_axis = -y_axis
        z_axis = -z_axis

    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat_wxyz = rotations.quaternion_from_matrix(rot_mat).astype(np.float32)
    return quat_wxyz, rot_mat


class CustomRetargetingViserViewer:
    def __init__(
        self,
        *,
        urdf_path: str,
        input_joint_names: Sequence[str],
        robot_model,
        host: str = "0.0.0.0",
        port: int = 8080,
        robot_x_offset: float = 0.24,
        mapping_load_path: Optional[str] = None,
        mapping_output_path: Optional[str] = None,
        robot_name_for_default: str = "robot",
        hand_type_for_default: str = "right",
    ):
        try:
            import viser
        except ImportError as exc:
            raise ImportError("This visualization requires: pip install viser") from exc

        self.server = viser.ViserServer(host=host, port=port)
        self.robot_model = robot_model
        self.robot_x_offset = float(robot_x_offset)

        self.mesh_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
        self.camera_transform: Optional[np.ndarray] = None
        self.hand_faces: Optional[np.ndarray] = None
        self.object_handles: List = []
        self.object_mesh_paths: List[Path] = []
        self.wrist_traj_handle = None

        self.hand_mesh = None
        self.hand_skeleton = None
        self.hand_points = None
        self.hand_joint_frames: List = []
        self.robot_joint_frames: List = []
        self.last_hand_joints: Optional[np.ndarray] = None
        self.last_robot_qpos = np.zeros((len(input_joint_names),), dtype=np.float32)

        self.mapping_skeleton = None
        self.mapping_points = None
        self.mapping_selected_point = None
        self._mapping_syncing_gui = False
        self._debug_syncing_gui = False
        self._debug_update_callback = None

        self.robot_root_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self.robot_root_frame.position = np.array([self.robot_x_offset, 0.0, 0.0], dtype=np.float32)

        self.robot_viewer = ViserHandViewer(
            urdf_path=urdf_path,
            input_joint_names=input_joint_names,
            server=self.server,
            root_node_name="/robot",
            mesh_color_override=(0.55, 0.90, 0.55, 0.45),
        )

        default_mapping_path = (
            Path(__file__).resolve().parent
            / "saved_alignments"
            / f"{robot_name_for_default}_{hand_type_for_default}_mano_link_mapping.json"
        )
        self.mapping_load_path = (
            Path(mapping_load_path).expanduser().resolve()
            if mapping_load_path is not None
            else default_mapping_path
        )
        self.mapping_output_path = (
            Path(mapping_output_path).expanduser().resolve()
            if mapping_output_path is not None
            else self.mapping_load_path
        )
        self.assignable_link_names, self.assignable_link_indices = self._build_assignable_links()
        if len(self.assignable_link_names) == 0:
            raise ValueError("No assignable robot links found for mapping.")
        palm_idx = self._find_palm_link_idx()
        self.point_to_link_idx = np.full((_NUM_MANO_POINTS,), palm_idx, dtype=np.int32)
        self.point_offsets = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float32)
        self.point_to_link_idx[0] = palm_idx
        self._load_mapping_if_exists()

        self._setup_mapping_gui()
        self._setup_hand_to_robot_alignment_gui()
        self._setup_single_frame_debug_gui()
        self._sync_mapping_gui_from_selection()
        self.robot_model.compute_forward_kinematics(self.last_robot_qpos)
        self._refresh_mapping_visualization(recompute_fk=False)

        self.server.scene.world_axes.visible = True
        self.server.scene.world_axes.axes_length = 0.10
        self.server.scene.world_axes.axes_radius = 0.003
        self.server.scene.world_axes.origin_radius = 0.006
        self.server.scene.set_up_direction("+z")

    def _setup_hand_to_robot_alignment_gui(self) -> None:
        with self.server.gui.add_folder("Hand->Robot Frame (H->R)"):
            self.hr_tx_slider = self.server.gui.add_slider(
                "hr_tx_m",
                min=-0.30,
                max=0.30,
                step=0.001,
                initial_value=0.05,
            )
            self.hr_ty_slider = self.server.gui.add_slider(
                "hr_ty_m",
                min=-0.30,
                max=0.30,
                step=0.001,
                initial_value=0.0,
            )
            self.hr_tz_slider = self.server.gui.add_slider(
                "hr_tz_m",
                min=-0.30,
                max=0.30,
                step=0.001,
                initial_value=0.0,
            )
            self.hr_rx_slider = self.server.gui.add_slider(
                "hr_rx_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=0.0,
            )
            self.hr_ry_slider = self.server.gui.add_slider(
                "hr_ry_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=0.0,
            )
            self.hr_rz_slider = self.server.gui.add_slider(
                "hr_rz_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=0.0,
            )

        @self.hr_tx_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

        @self.hr_ty_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

        @self.hr_tz_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

        @self.hr_rx_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

        @self.hr_ry_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

        @self.hr_rz_slider.on_update
        def _(_event):
            self._update_robot_root_from_last_hand()

    def get_hand_to_robot_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        translation = np.asarray(
            [self.hr_tx_slider.value, self.hr_ty_slider.value, self.hr_tz_slider.value],
            dtype=np.float64,
        )
        euler_xyz_deg = np.asarray(
            [self.hr_rx_slider.value, self.hr_ry_slider.value, self.hr_rz_slider.value],
            dtype=np.float64,
        )
        rotation_hr = ScipyRotation.from_euler(
            "XYZ", np.deg2rad(euler_xyz_deg), degrees=False
        ).as_matrix()
        return translation, rotation_hr.astype(np.float64)

    def _setup_single_frame_debug_gui(self) -> None:
        default_debug_link = self._find_default_debug_link_idx()
        self.robot_model.compute_forward_kinematics(self.last_robot_qpos)
        initial_frame_idx = int(self.assignable_link_indices[default_debug_link])
        initial_rot = self.robot_model.get_link_pose(initial_frame_idx)[:3, :3].astype(np.float64)
        initial_euler_deg = ScipyRotation.from_matrix(initial_rot).as_euler(
            "XYZ", degrees=True
        )
        with self.server.gui.add_folder("Single-Frame Debug"):
            self.debug_link_slider = self.server.gui.add_slider(
                "debug_link_index",
                min=0,
                max=len(self.assignable_link_names) - 1,
                step=1,
                initial_value=default_debug_link,
            )
            self.debug_link_name_text = self.server.gui.add_text(
                "debug_link_name",
                initial_value=self.assignable_link_names[int(self.debug_link_slider.value)],
                disabled=True,
            )
            self.debug_rx_slider = self.server.gui.add_slider(
                "debug_rx_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_euler_deg[0]),
            )
            self.debug_ry_slider = self.server.gui.add_slider(
                "debug_ry_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_euler_deg[1]),
            )
            self.debug_rz_slider = self.server.gui.add_slider(
                "debug_rz_deg",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_euler_deg[2]),
            )

        @self.debug_link_slider.on_update
        def _(_event):
            self.debug_link_name_text.value = self.assignable_link_names[
                int(self.debug_link_slider.value)
            ]
            self._sync_debug_target_to_selected_link()
            self._refresh_mapping_visualization(recompute_fk=False)
            self._notify_debug_update()

        @self.debug_rx_slider.on_update
        def _(_event):
            if self._debug_syncing_gui:
                return
            self._refresh_mapping_visualization(recompute_fk=False)
            self._notify_debug_update()

        @self.debug_ry_slider.on_update
        def _(_event):
            if self._debug_syncing_gui:
                return
            self._refresh_mapping_visualization(recompute_fk=False)
            self._notify_debug_update()

        @self.debug_rz_slider.on_update
        def _(_event):
            if self._debug_syncing_gui:
                return
            self._refresh_mapping_visualization(recompute_fk=False)
            self._notify_debug_update()

    def _sync_debug_target_to_selected_link(self) -> None:
        link_sel_idx = int(self.debug_link_slider.value)
        frame_idx = int(self.assignable_link_indices[link_sel_idx])
        self.robot_model.compute_forward_kinematics(self.last_robot_qpos)
        link_rot = self.robot_model.get_link_pose(frame_idx)[:3, :3].astype(np.float64)
        euler_deg = ScipyRotation.from_matrix(link_rot).as_euler("XYZ", degrees=True)
        self._debug_syncing_gui = True
        self.debug_rx_slider.value = float(euler_deg[0])
        self.debug_ry_slider.value = float(euler_deg[1])
        self.debug_rz_slider.value = float(euler_deg[2])
        self._debug_syncing_gui = False

    def set_debug_update_callback(self, callback) -> None:
        self._debug_update_callback = callback

    def _notify_debug_update(self) -> None:
        if self._debug_update_callback is None:
            return
        self._debug_update_callback()

    def get_single_frame_orientation_target(self) -> Tuple[int, np.ndarray]:
        link_sel_idx = int(self.debug_link_slider.value)
        frame_idx = int(self.assignable_link_indices[link_sel_idx])
        target_rot_local = ScipyRotation.from_euler(
            "XYZ",
            [self.debug_rx_slider.value, self.debug_ry_slider.value, self.debug_rz_slider.value],
            degrees=True,
        ).as_matrix()
        return frame_idx, target_rot_local.astype(np.float64)

    def get_current_mapping_for_optimizer(self) -> Tuple[np.ndarray, np.ndarray]:
        link_indices = np.asarray(
            [
                self.assignable_link_indices[int(link_sel_idx)]
                for link_sel_idx in self.point_to_link_idx
            ],
            dtype=np.int32,
        )
        offsets = self.point_offsets.astype(np.float64).copy()
        return link_indices, offsets

    def set_hand_joints_for_root_alignment(self, hand_joints: np.ndarray) -> None:
        self.last_hand_joints = hand_joints.astype(np.float32).copy()
        self._apply_hand_to_robot_root_pose(hand_joints)

    def _compute_hand_root_frame(self, hand_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        root_forward = hand_joints[9] - hand_joints[0]
        root_lateral = hand_joints[13] - hand_joints[5]
        if float(np.linalg.norm(root_forward)) < 1e-8:
            root_forward = hand_joints[1] - hand_joints[0]
        if float(np.linalg.norm(root_lateral)) < 1e-8:
            root_lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        palm_z = np.cross(root_forward, root_lateral).astype(np.float64)
        if float(np.linalg.norm(palm_z)) < 1e-8:
            palm_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _, root_rot = _quat_from_forward_prev_z(root_forward, palm_z)
        root_pos = np.asarray(hand_joints[0], dtype=np.float64)
        return root_pos, root_rot

    def _apply_hand_to_robot_root_pose(self, hand_joints: np.ndarray) -> None:
        root_pos_h, root_rot_h = self._compute_hand_root_frame(hand_joints)
        trans_hr, rot_hr = self.get_hand_to_robot_transform()
        target_mano0_pos_world = root_pos_h + root_rot_h @ trans_hr
        target_mano0_rot_world = root_rot_h @ rot_hr

        # Compensate URDF-root -> mapped MANO-0 frame transform so that:
        #   T_world_robotRoot * T_robotRoot_mano0 == T_world_mano0_target
        self.robot_model.compute_forward_kinematics(self.last_robot_qpos)
        mapping_points_local = self._compute_mapping_points()
        mapping_quats_local = self._compute_skeleton_joint_frame_quaternions(mapping_points_local)
        mano0_pos_local = mapping_points_local[0].astype(np.float64)
        mano0_rot_local = rotations.matrix_from_quaternion(
            mapping_quats_local[0].astype(np.float64)
        ).astype(np.float64)

        robot_root_rot_world = target_mano0_rot_world @ mano0_rot_local.T
        robot_root_pos_world = target_mano0_pos_world - robot_root_rot_world @ mano0_pos_local

        self.robot_root_frame.position = robot_root_pos_world.astype(np.float32)
        self.robot_root_frame.wxyz = rotations.quaternion_from_matrix(
            robot_root_rot_world
        ).astype(np.float32)

    def _update_robot_root_from_last_hand(self) -> None:
        if self.last_hand_joints is None:
            return
        self._apply_hand_to_robot_root_pose(self.last_hand_joints)
        self._refresh_mapping_visualization(recompute_fk=False)

    def _robot_local_to_world(self, points_local: np.ndarray) -> np.ndarray:
        rot = rotations.matrix_from_quaternion(
            np.asarray(self.robot_root_frame.wxyz, dtype=np.float64)
        )
        trans = np.asarray(self.robot_root_frame.position, dtype=np.float64)
        flat = points_local.reshape(-1, 3).astype(np.float64)
        flat_world = flat @ rot.T + trans
        return flat_world.reshape(points_local.shape).astype(np.float32)

    def _compute_skeleton_joint_frame_quaternions(self, joint_points: np.ndarray) -> np.ndarray:
        quats = np.zeros((_NUM_MANO_POINTS, 4), dtype=np.float32)
        if joint_points.shape != (_NUM_MANO_POINTS, 3):
            quats[:, 0] = 1.0
            return quats
        rot_mats = np.tile(np.eye(3, dtype=np.float64), (_NUM_MANO_POINTS, 1, 1))

        root_forward = joint_points[9] - joint_points[0]
        root_lateral = joint_points[13] - joint_points[5]
        if float(np.linalg.norm(root_forward)) < 1e-8:
            root_forward = joint_points[1] - joint_points[0]
        if float(np.linalg.norm(root_lateral)) < 1e-8:
            root_lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        palm_z = np.cross(root_forward, root_lateral).astype(np.float64)
        if float(np.linalg.norm(palm_z)) < 1e-8:
            palm_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        root_quat, root_rot = _quat_from_forward_prev_z(root_forward, palm_z)
        quats[0] = root_quat
        rot_mats[0] = root_rot

        for idx in range(1, _NUM_MANO_POINTS):
            parent_idx = int(_MANO_PARENT_IDX[idx])
            child_idx = int(_MANO_CHILD_IDX[idx])
            if child_idx >= 0:
                forward = joint_points[child_idx] - joint_points[idx]
            else:
                forward = joint_points[idx] - joint_points[parent_idx]
            prev_z = rot_mats[parent_idx, :, 2]
            quat, rot = _quat_from_forward_prev_z(forward, prev_z)
            quats[idx] = quat
            rot_mats[idx] = rot
        return quats

    def _update_mano_joint_frames(self, hand_joints: np.ndarray) -> None:
        joint_quats = self._compute_skeleton_joint_frame_quaternions(hand_joints)
        if len(self.hand_joint_frames) == 0:
            for idx in range(_NUM_MANO_POINTS):
                handle = self.server.scene.add_frame(
                    f"/hand/joint_frames/{idx}",
                    position=hand_joints[idx],
                    wxyz=joint_quats[idx],
                    axes_length=0.015,
                    axes_radius=0.0018,
                    origin_radius=0.0036,
                )
                self.hand_joint_frames.append(handle)
            return

        for idx, handle in enumerate(self.hand_joint_frames):
            handle.position = hand_joints[idx]
            handle.wxyz = joint_quats[idx]

    def _update_robot_joint_frames(self, robot_joint_points: np.ndarray) -> None:
        joint_quats = self._compute_skeleton_joint_frame_quaternions(robot_joint_points)
        if len(self.robot_joint_frames) == 0:
            for idx in range(_NUM_MANO_POINTS):
                handle = self.server.scene.add_frame(
                    f"/robot/joint_frames/{idx}",
                    position=robot_joint_points[idx],
                    wxyz=joint_quats[idx],
                    axes_length=0.015,
                    axes_radius=0.0018,
                    origin_radius=0.0036,
                )
                self.robot_joint_frames.append(handle)
            return

        for idx, handle in enumerate(self.robot_joint_frames):
            handle.position = robot_joint_points[idx]
            handle.wxyz = joint_quats[idx]

    def _build_assignable_links(self) -> Tuple[List[str], List[int]]:
        names: List[str] = []
        indices: List[int] = []
        for frame_idx, name in enumerate(self.robot_model.link_names):
            lname = name.lower()
            if lname in {"universe", "world"}:
                continue
            if "dummy" in lname:
                continue
            if "joint" in lname:
                continue
            if lname.endswith("_fixed"):
                continue
            names.append(name)
            indices.append(frame_idx)
        return names, indices

    def _find_palm_link_idx(self) -> int:
        for i, name in enumerate(self.assignable_link_names):
            if "palm" in name.lower():
                return i
        return 0

    def _find_default_debug_link_idx(self) -> int:
        """Pick a non-palm link by default so orientation debug affects finger joints."""
        palm_idx = self._find_palm_link_idx()
        preferred_mano_points = [8, 12, 16, 20, 4, 7, 11, 15, 19, 3]
        for point_idx in preferred_mano_points:
            link_idx = int(self.point_to_link_idx[point_idx])
            if 0 <= link_idx < len(self.assignable_link_names) and link_idx != palm_idx:
                return link_idx

        for idx, name in enumerate(self.assignable_link_names):
            if "palm" not in name.lower():
                return idx
        return palm_idx

    def _compute_mapping_points(self) -> np.ndarray:
        points = np.zeros((_NUM_MANO_POINTS, 3), dtype=np.float32)
        for point_idx in range(_NUM_MANO_POINTS):
            link_sel_idx = int(self.point_to_link_idx[point_idx])
            frame_idx = self.assignable_link_indices[link_sel_idx]
            pose = self.robot_model.get_link_pose(frame_idx)
            offset = self.point_offsets[point_idx]
            points[point_idx] = (pose[:3, :3] @ offset) + pose[:3, 3]
        return points

    def _mapping_payload(self) -> dict:
        entries = []
        for point_idx in range(_NUM_MANO_POINTS):
            link_idx = int(self.point_to_link_idx[point_idx])
            entries.append(
                {
                    "mano_point_index": int(point_idx),
                    "link_name": self.assignable_link_names[link_idx],
                    "offset_xyz": [float(v) for v in self.point_offsets[point_idx]],
                }
            )
        return {"mano_link_assignments": entries}

    def _reset_mapping_to_default(self) -> None:
        palm_idx = self._find_palm_link_idx()
        self.point_to_link_idx[:] = palm_idx
        self.point_offsets[:] = 0.0
        self.point_to_link_idx[0] = palm_idx

    def _load_mapping_if_exists(self) -> None:
        if not self.mapping_load_path.is_file():
            print(f"Mapping file not found: {self.mapping_load_path}")
            return
        try:
            with self.mapping_load_path.open("r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Failed to read mapping file {self.mapping_load_path}: {exc}")
            return

        assignments = payload.get("mano_link_assignments", [])
        if not isinstance(assignments, list):
            print(f"Invalid mapping format in {self.mapping_load_path}")
            return

        link_name_to_idx = {name: i for i, name in enumerate(self.assignable_link_names)}
        num_loaded = 0
        for entry in assignments:
            try:
                point_idx = int(entry["mano_point_index"])
                link_name = str(entry["link_name"])
                offset_xyz = np.asarray(entry.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
            except Exception:
                continue
            if point_idx < 0 or point_idx >= _NUM_MANO_POINTS:
                continue
            if link_name not in link_name_to_idx:
                continue
            if offset_xyz.shape != (3,):
                continue
            self.point_to_link_idx[point_idx] = link_name_to_idx[link_name]
            self.point_offsets[point_idx] = offset_xyz
            num_loaded += 1

        if num_loaded > 0:
            print(f"Loaded {num_loaded} MANO link mappings from {self.mapping_load_path}")

    def _reload_mapping_from_file(self) -> None:
        self._reset_mapping_to_default()
        self._load_mapping_if_exists()
        self._sync_mapping_gui_from_selection()
        self._refresh_mapping_visualization(recompute_fk=True)
        self._notify_debug_update()

    def _print_and_save_mapping(self) -> None:
        payload = self._mapping_payload()
        payload_text = json.dumps(payload, indent=2)
        print("\n=== MANO LINK MAPPING START ===")
        print(payload_text)
        print("=== MANO LINK MAPPING END ===\n")
        self.mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.mapping_output_path.open("w") as f:
            f.write(payload_text + "\n")
        print(f"Saved mapping to {self.mapping_output_path}")

    def _refresh_mapping_visualization(self, recompute_fk: bool = True) -> None:
        if recompute_fk:
            self.robot_model.compute_forward_kinematics(self.last_robot_qpos)

        mapping_points = self._compute_mapping_points()
        mapping_segments = segments_from_edges(mapping_points, _HAND_SKELETON_EDGES)
        mapping_points_world = self._robot_local_to_world(mapping_points)
        mapping_segments_world = self._robot_local_to_world(mapping_segments)
        selected_idx = int(self.mapping_point_slider.value)
        selected_point = mapping_points_world[selected_idx][None, :]

        if self.mapping_skeleton is None:
            self.mapping_skeleton = self.server.scene.add_line_segments(
                "/mapping/skeleton_lines",
                points=mapping_segments_world,
                colors=(255, 145, 70),
                line_width=3.0,
            )
            self.mapping_points = self.server.scene.add_point_cloud(
                "/mapping/skeleton_points",
                points=mapping_points_world,
                colors=(255, 145, 70),
                point_size=0.007,
                point_shape="circle",
            )
            self.mapping_selected_point = self.server.scene.add_point_cloud(
                "/mapping/selected_point",
                points=selected_point,
                colors=(255, 30, 30),
                point_size=0.014,
                point_shape="circle",
            )
        else:
            self.mapping_skeleton.points = mapping_segments_world
            self.mapping_points.points = mapping_points_world
            self.mapping_selected_point.points = selected_point
        self._update_robot_joint_frames(mapping_points)

    def _sync_mapping_gui_from_selection(self) -> None:
        self._mapping_syncing_gui = True
        point_idx = int(self.mapping_point_slider.value)
        self.mapping_link_slider.value = int(self.point_to_link_idx[point_idx])
        self.mapping_offset_x_slider.value = float(self.point_offsets[point_idx, 0])
        self.mapping_offset_y_slider.value = float(self.point_offsets[point_idx, 1])
        self.mapping_offset_z_slider.value = float(self.point_offsets[point_idx, 2])
        self.mapping_link_name_text.value = self.assignable_link_names[int(self.mapping_link_slider.value)]
        self._mapping_syncing_gui = False

    def _apply_mapping_gui_to_selection(self) -> None:
        point_idx = int(self.mapping_point_slider.value)
        self.point_to_link_idx[point_idx] = int(self.mapping_link_slider.value)
        self.point_offsets[point_idx, 0] = float(self.mapping_offset_x_slider.value)
        self.point_offsets[point_idx, 1] = float(self.mapping_offset_y_slider.value)
        self.point_offsets[point_idx, 2] = float(self.mapping_offset_z_slider.value)
        self.mapping_link_name_text.value = self.assignable_link_names[int(self.mapping_link_slider.value)]

    def _setup_mapping_gui(self) -> None:
        with self.server.gui.add_folder("MANO Link Mapping"):
            self.mapping_point_slider = self.server.gui.add_slider(
                "mano_point_index",
                min=0,
                max=_NUM_MANO_POINTS - 1,
                step=1,
                initial_value=0,
            )
            self.mapping_link_slider = self.server.gui.add_slider(
                "assigned_link_index",
                min=0,
                max=len(self.assignable_link_names) - 1,
                step=1,
                initial_value=self._find_palm_link_idx(),
            )
            self.mapping_offset_x_slider = self.server.gui.add_slider(
                "offset_x",
                min=-0.10,
                max=0.10,
                step=0.001,
                initial_value=0.0,
            )
            self.mapping_offset_y_slider = self.server.gui.add_slider(
                "offset_y",
                min=-0.10,
                max=0.10,
                step=0.001,
                initial_value=0.0,
            )
            self.mapping_offset_z_slider = self.server.gui.add_slider(
                "offset_z",
                min=-0.10,
                max=0.10,
                step=0.001,
                initial_value=0.0,
            )
            self.mapping_link_name_text = self.server.gui.add_text(
                "assigned_link_name",
                initial_value=self.assignable_link_names[self._find_palm_link_idx()],
                disabled=True,
            )
            self.mapping_print_button = self.server.gui.add_button("print_mapping")
            self.mapping_reload_button = self.server.gui.add_button("reload_mapping_from_file")

        @self.mapping_point_slider.on_update
        def _(_event):
            if self._mapping_syncing_gui:
                return
            self._sync_mapping_gui_from_selection()
            self._refresh_mapping_visualization(recompute_fk=True)
            self._notify_debug_update()

        @self.mapping_link_slider.on_update
        def _(_event):
            if self._mapping_syncing_gui:
                return
            self._apply_mapping_gui_to_selection()
            self._refresh_mapping_visualization(recompute_fk=True)
            self._notify_debug_update()

        @self.mapping_offset_x_slider.on_update
        def _(_event):
            if self._mapping_syncing_gui:
                return
            self._apply_mapping_gui_to_selection()
            self._refresh_mapping_visualization(recompute_fk=True)
            self._notify_debug_update()

        @self.mapping_offset_y_slider.on_update
        def _(_event):
            if self._mapping_syncing_gui:
                return
            self._apply_mapping_gui_to_selection()
            self._refresh_mapping_visualization(recompute_fk=True)
            self._notify_debug_update()

        @self.mapping_offset_z_slider.on_update
        def _(_event):
            if self._mapping_syncing_gui:
                return
            self._apply_mapping_gui_to_selection()
            self._refresh_mapping_visualization(recompute_fk=True)
            self._notify_debug_update()

        @self.mapping_print_button.on_click
        def _(_event):
            self._print_and_save_mapping()

        @self.mapping_reload_button.on_click
        def _(_event):
            self._reload_mapping_from_file()

    def load_trajectory(
        self,
        *,
        camera_transform: np.ndarray,
        hand_faces: np.ndarray,
        ycb_ids: Sequence[int],
        object_mesh_files: Sequence[str],
        wrist_trajectory_points: np.ndarray,
    ) -> None:
        self.camera_transform = camera_transform.astype(np.float32)
        self.hand_faces = hand_faces.astype(np.uint32)

        for handle in self.object_handles:
            handle.remove()
        self.object_handles = []
        self.object_mesh_paths = [Path(p) for p in object_mesh_files]

        for i, (ycb_id, mesh_path) in enumerate(zip(ycb_ids, self.object_mesh_paths)):
            if mesh_path not in self.mesh_cache:
                self.mesh_cache[mesh_path] = load_obj_tri_mesh(mesh_path)
            vertices, faces = self.mesh_cache[mesh_path]
            obj_name = YCB_CLASSES.get(int(ycb_id), f"obj_{int(ycb_id)}")
            handle = self.server.scene.add_mesh_simple(
                f"/objects/{i}_{obj_name}",
                vertices=vertices,
                faces=faces,
                color=_OBJECT_COLORS[i % len(_OBJECT_COLORS)],
                opacity=0.60,
            )
            self.object_handles.append(handle)

        wrist_segments = segments_from_polyline(wrist_trajectory_points.astype(np.float32))
        if self.wrist_traj_handle is None:
            self.wrist_traj_handle = self.server.scene.add_line_segments(
                "/hand/wrist_trajectory",
                points=wrist_segments,
                colors=(250, 220, 80),
                line_width=2.0,
            )
        else:
            self.wrist_traj_handle.points = wrist_segments

    def _update_objects(self, object_pose_frame: np.ndarray) -> None:
        for obj_idx, pose in enumerate(object_pose_frame):
            transform = make_transform(pose[4:], quat_xyzw_to_wxyz(pose[:4]))
            world_tf = self.camera_transform @ transform
            self.object_handles[obj_idx].position = world_tf[:3, 3].astype(np.float32)
            self.object_handles[obj_idx].wxyz = rotations.quaternion_from_matrix(
                world_tf[:3, :3]
            ).astype(np.float32)

    def update_frame(
        self,
        *,
        hand_vertices: np.ndarray,
        hand_joints: np.ndarray,
        object_pose_frame: np.ndarray,
        robot_qpos: np.ndarray,
    ) -> None:
        hand_vertices = hand_vertices.astype(np.float32)
        hand_joints = hand_joints.astype(np.float32)
        self.last_hand_joints = hand_joints.copy()

        if self.hand_mesh is None:
            self.hand_mesh = self.server.scene.add_mesh_simple(
                "/hand/mesh",
                vertices=hand_vertices,
                faces=self.hand_faces,
                color=_HAND_MESH_COLOR,
                opacity=0.45,
            )
        else:
            self.hand_mesh.vertices = hand_vertices

        hand_segments = segments_from_edges(hand_joints, _HAND_SKELETON_EDGES)
        if self.hand_skeleton is None:
            self.hand_skeleton = self.server.scene.add_line_segments(
                "/hand/skeleton_lines",
                points=hand_segments,
                colors=(60, 120, 255),
                line_width=3.0,
            )
            self.hand_points = self.server.scene.add_point_cloud(
                "/hand/skeleton_points",
                points=hand_joints,
                colors=(60, 120, 255),
                point_size=0.006,
                point_shape="circle",
            )
        else:
            self.hand_skeleton.points = hand_segments
            self.hand_points.points = hand_joints
        self._update_mano_joint_frames(hand_joints)
        self._apply_hand_to_robot_root_pose(hand_joints)

        self._update_objects(object_pose_frame.astype(np.float32))

        robot_qpos = np.asarray(robot_qpos, dtype=np.float32)
        self.last_robot_qpos = robot_qpos.copy()
        self.robot_viewer.set_qpos(robot_qpos)

        self._refresh_mapping_visualization(recompute_fk=False)
