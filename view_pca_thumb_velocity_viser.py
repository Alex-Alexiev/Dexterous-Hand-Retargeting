import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils.system.path_utils import resolve_repo_path
from utils.training.pca import decode_pca, load_pca_model
from utils.viser.viser_urdf_viewer import ViserHandViewer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "results/retargeted_grasps/pca/qpos_pca_z5"

def _resolve_stats_path(model_dir: Path, manifest: dict, filename: str | None, key: str, default: str) -> Path:
    resolved_name = filename if filename is not None else manifest.get(key, default)
    return model_dir / resolved_name


class PCAThumbVelocityController:
    def __init__(
        self,
        *,
        pca_model: dict,
        latent_std: np.ndarray,
        robot_qpos_limits: np.ndarray,
        robot_model,
        palm_link_name: str,
        tip_link_name: str,
        ctrl_dt: float,
        fd_epsilon: float,
        latent_damping: float,
        latent_speed_limit: float,
        max_backtracking_steps: int,
    ):
        self.pca_model = pca_model
        self.latent_std = np.asarray(latent_std, dtype=np.float32)
        self.robot_qpos_limits = np.asarray(robot_qpos_limits, dtype=np.float32)
        self.robot_model = robot_model
        self.palm_link_idx = robot_model.get_link_index(palm_link_name)
        self.tip_link_idx = robot_model.get_link_index(tip_link_name)
        self.ctrl_dt = float(ctrl_dt)
        self.fd_epsilon = float(fd_epsilon)
        self.latent_damping = float(latent_damping)
        self.latent_speed_limit = float(latent_speed_limit)
        self.max_backtracking_steps = int(max_backtracking_steps)

        self.z_dim = int(pca_model["z_dim"])
        self.q_dim = int(pca_model["x_dim"])
        self.z_state = np.zeros((self.z_dim,), dtype=np.float32)
        self.q_state = decode_pca(self.pca_model, self.z_state).reshape(-1).astype(np.float32)
        self.prev_thumb_pos_palm = None
        self.actual_thumb_vel_palm = np.zeros((3,), dtype=np.float32)

    def reset(self, latent: np.ndarray | None = None) -> None:
        if latent is None:
            self.z_state = np.zeros((self.z_dim,), dtype=np.float32)
        else:
            latent = np.asarray(latent, dtype=np.float32).reshape(-1)
            if latent.shape[0] != self.z_dim:
                raise ValueError(f"Expected latent dim {self.z_dim}, got {latent.shape[0]}.")
            self.z_state = latent.copy()
        self.q_state = decode_pca(self.pca_model, self.z_state).reshape(-1).astype(np.float32)
        self.prev_thumb_pos_palm = None
        self.actual_thumb_vel_palm[:] = 0.0

    def _fk_link_poses(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.robot_model.compute_forward_kinematics(np.asarray(qpos, dtype=np.float64))
        palm_pose = self.robot_model.get_link_pose(self.palm_link_idx).astype(np.float32)
        tip_pose = self.robot_model.get_link_pose(self.tip_link_idx).astype(np.float32)
        return palm_pose, tip_pose

    def thumb_position_in_palm_frame(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        palm_pose, tip_pose = self._fk_link_poses(qpos)
        palm_rot = palm_pose[:3, :3]
        palm_pos = palm_pose[:3, 3]
        tip_pos = tip_pose[:3, 3]
        tip_pos_palm = palm_rot.T @ (tip_pos - palm_pos)
        return tip_pos_palm.astype(np.float32), palm_pos.astype(np.float32), palm_rot.astype(np.float32)

    def latent_jacobian_fd(self, z_curr: np.ndarray) -> np.ndarray:
        z_curr = np.asarray(z_curr, dtype=np.float32).reshape(-1)
        jac = np.zeros((3, self.z_dim), dtype=np.float32)
        for dim in range(self.z_dim):
            perturb = np.zeros((self.z_dim,), dtype=np.float32)
            perturb[dim] = self.fd_epsilon
            q_plus = decode_pca(self.pca_model, z_curr + perturb).reshape(-1).astype(np.float32)
            q_minus = decode_pca(self.pca_model, z_curr - perturb).reshape(-1).astype(np.float32)
            x_plus, _, _ = self.thumb_position_in_palm_frame(q_plus)
            x_minus, _, _ = self.thumb_position_in_palm_frame(q_minus)
            jac[:, dim] = (x_plus - x_minus) / (2.0 * self.fd_epsilon)
        return jac

    def solve_latent_velocity(self, desired_thumb_vel_palm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        desired_thumb_vel_palm = np.asarray(desired_thumb_vel_palm, dtype=np.float32).reshape(3)
        jac = self.latent_jacobian_fd(self.z_state)
        jtj = jac.T @ jac
        damping_eye = (self.latent_damping ** 2) * np.eye(self.z_dim, dtype=np.float32)
        rhs = jac.T @ desired_thumb_vel_palm
        z_dot = np.linalg.solve(jtj + damping_eye, rhs).astype(np.float32)

        if self.latent_speed_limit > 0.0:
            z_dot_norm = float(np.linalg.norm(z_dot))
            if z_dot_norm > self.latent_speed_limit:
                z_dot *= self.latent_speed_limit / max(z_dot_norm, 1.0e-8)
        return z_dot, jac

    def _qpos_within_limits(self, qpos: np.ndarray) -> bool:
        lower = self.robot_qpos_limits[:, 0]
        upper = self.robot_qpos_limits[:, 1]
        return bool(np.all(qpos >= lower) and np.all(qpos <= upper))

    def integrate_on_manifold(self, z_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step_scale = 1.0
        for _ in range(self.max_backtracking_steps + 1):
            z_next = self.z_state + (self.ctrl_dt * step_scale) * z_dot
            q_next = decode_pca(self.pca_model, z_next).reshape(-1).astype(np.float32)
            if self._qpos_within_limits(q_next):
                return z_next.astype(np.float32), q_next.astype(np.float32)
            step_scale *= 0.5
        return self.z_state.copy(), self.q_state.copy()

    def step(self, desired_thumb_vel_palm: np.ndarray) -> dict:
        z_dot, jac = self.solve_latent_velocity(desired_thumb_vel_palm)
        z_next, q_next = self.integrate_on_manifold(z_dot)
        thumb_pos_palm, palm_pos_world, palm_rot_world = self.thumb_position_in_palm_frame(q_next)

        if self.prev_thumb_pos_palm is None:
            actual_thumb_vel_palm = np.zeros((3,), dtype=np.float32)
        else:
            actual_thumb_vel_palm = ((thumb_pos_palm - self.prev_thumb_pos_palm) / self.ctrl_dt).astype(np.float32)
        self.prev_thumb_pos_palm = thumb_pos_palm.copy()

        self.z_state = z_next
        self.q_state = q_next
        self.actual_thumb_vel_palm = actual_thumb_vel_palm

        return {
            "z": self.z_state.copy(),
            "qpos": self.q_state.copy(),
            "z_dot": z_dot.astype(np.float32),
            "jacobian": jac.astype(np.float32),
            "thumb_pos_palm": thumb_pos_palm.astype(np.float32),
            "actual_thumb_vel_palm": actual_thumb_vel_palm.astype(np.float32),
            "palm_pos_world": palm_pos_world.astype(np.float32),
            "palm_rot_world": palm_rot_world.astype(np.float32),
        }


def _segment_from_origin_and_vector(origin: np.ndarray, vec: np.ndarray, scale: float) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    return np.stack([origin, origin + float(scale) * vec], axis=0).reshape(1, 2, 3).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--ctrl-dt", type=float, default=0.05)
    parser.add_argument("--fd-epsilon", type=float, default=0.01)
    parser.add_argument("--latent-damping", type=float, default=0.05)
    parser.add_argument("--latent-speed-limit", type=float, default=6.0)
    parser.add_argument("--arrow-scale", type=float, default=0.20)
    parser.add_argument("--velocity-slider-max", type=float, default=0.30)
    parser.add_argument("--tip-link", default="thumb_tip")
    parser.add_argument("--palm-link", default="palm")
    parser.add_argument("--max-backtracking-steps", type=int, default=12)
    args = parser.parse_args()

    try:
        import viser
    except ImportError as exc:
        raise ImportError("This script requires: pip install viser") from exc

    from utils.robot.robot_wrapper import RobotWrapper
    from utils.robot.urdf_utils import resolve_visual_urdf

    model_dir = resolve_repo_path(args.model_dir, REPO_ROOT)
    manifest_path = model_dir / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    latent_stats_path = _resolve_stats_path(
        model_dir,
        manifest,
        filename=None,
        key="latent_stats",
        default="latent_stats.npz",
    )
    pca_model_path = model_dir / manifest.get("pca_model", "pca_model.npz")
    metadata_path = model_dir / manifest.get("robot_metadata", "robot_metadata.json")

    with metadata_path.open("r") as f:
        model_metadata = json.load(f)
    with np.load(latent_stats_path, allow_pickle=False) as payload:
        latent_std = payload["std"].astype(np.float32)

    pca_model = load_pca_model(pca_model_path)
    robot_joint_names = list(model_metadata["robot_joint_names"])
    urdf_path = Path(model_metadata["urdf_path"])
    visual_urdf_path = resolve_visual_urdf(urdf_path)

    robot_model = RobotWrapper(str(urdf_path))
    robot_qpos_limits = robot_model.joint_limits.astype(np.float32)

    controller = PCAThumbVelocityController(
        pca_model=pca_model,
        latent_std=latent_std,
        robot_qpos_limits=robot_qpos_limits,
        robot_model=robot_model,
        palm_link_name=args.palm_link,
        tip_link_name=args.tip_link,
        ctrl_dt=args.ctrl_dt,
        fd_epsilon=args.fd_epsilon,
        latent_damping=args.latent_damping,
        latent_speed_limit=args.latent_speed_limit,
        max_backtracking_steps=args.max_backtracking_steps,
    )
    controller.reset()

    server = viser.ViserServer(host=args.host, port=args.port)
    robot_viewer = ViserHandViewer(
        urdf_path=str(visual_urdf_path),
        input_joint_names=robot_joint_names,
        server=server,
        root_node_name="/robot",
        mesh_color_override=(0.55, 0.90, 0.55, 0.88),
    )

    desired_handle = server.scene.add_line_segments(
        "/debug/desired_thumb_velocity",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=(225, 110, 70),
        line_width=4.0,
    )
    actual_handle = server.scene.add_line_segments(
        "/debug/actual_thumb_velocity",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=(70, 140, 235),
        line_width=4.0,
    )
    thumb_point = server.scene.add_point_cloud(
        "/debug/thumb_tip",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(30, 30, 30),
        point_size=0.010,
        point_shape="circle",
    )
    palm_frame = server.scene.add_frame("/debug/palm_frame", show_axes=True)

    with server.gui.add_folder("Thumb Velocity"):
        running_checkbox = server.gui.add_checkbox("run", initial_value=False)
        vx_slider = server.gui.add_slider(
            "vx_palm",
            min=-args.velocity_slider_max,
            max=args.velocity_slider_max,
            step=0.01,
            initial_value=0.0,
        )
        vy_slider = server.gui.add_slider(
            "vy_palm",
            min=-args.velocity_slider_max,
            max=args.velocity_slider_max,
            step=0.01,
            initial_value=0.0,
        )
        vz_slider = server.gui.add_slider(
            "vz_palm",
            min=-args.velocity_slider_max,
            max=args.velocity_slider_max,
            step=0.01,
            initial_value=0.0,
        )
        damping_slider = server.gui.add_slider(
            "latent_damping",
            min=0.0,
            max=0.5,
            step=0.005,
            initial_value=float(args.latent_damping),
        )
        speed_limit_slider = server.gui.add_slider(
            "latent_speed_limit",
            min=0.1,
            max=20.0,
            step=0.1,
            initial_value=float(args.latent_speed_limit),
        )
        reset_button = server.gui.add_button("reset_to_mean")

    with server.gui.add_folder("State"):
        thumb_pos_text = server.gui.add_text("thumb_pos_palm", initial_value="0, 0, 0", disabled=True)
        desired_vel_text = server.gui.add_text("desired_vel_palm", initial_value="0, 0, 0", disabled=True)
        actual_vel_text = server.gui.add_text("actual_vel_palm", initial_value="0, 0, 0", disabled=True)
        latent_norm_text = server.gui.add_text("latent_norm", initial_value="0.000", disabled=True)
        latent_speed_text = server.gui.add_text("latent_speed", initial_value="0.000", disabled=True)
        qpos_norm_text = server.gui.add_text("qpos_norm", initial_value="0.000", disabled=True)
        jacobian_rank_text = server.gui.add_text("jacobian_rank", initial_value="0", disabled=True)

    with server.gui.add_folder("Model"):
        server.gui.add_text("model_dir", initial_value=str(model_dir), disabled=True)
        server.gui.add_text("robot_name", initial_value=str(model_metadata["robot_name"]), disabled=True)
        server.gui.add_text("palm_link", initial_value=str(args.palm_link), disabled=True)
        server.gui.add_text("tip_link", initial_value=str(args.tip_link), disabled=True)
        server.gui.add_text("z_dim", initial_value=str(pca_model["z_dim"]), disabled=True)
        server.gui.add_text("qpos_dim", initial_value=str(pca_model["x_dim"]), disabled=True)

    @reset_button.on_click
    def _(_event):
        controller.reset()
        refresh_visuals(
            qpos=controller.q_state,
            thumb_pos_palm=controller.thumb_position_in_palm_frame(controller.q_state)[0],
            actual_thumb_vel_palm=np.zeros((3,), dtype=np.float32),
            z_dot=np.zeros((controller.z_dim,), dtype=np.float32),
            jacobian=np.zeros((3, controller.z_dim), dtype=np.float32),
        )

    def desired_thumb_velocity_palm() -> np.ndarray:
        return np.asarray([vx_slider.value, vy_slider.value, vz_slider.value], dtype=np.float32)

    def refresh_visuals(
        *,
        qpos: np.ndarray,
        thumb_pos_palm: np.ndarray,
        actual_thumb_vel_palm: np.ndarray,
        z_dot: np.ndarray,
        jacobian: np.ndarray,
    ) -> None:
        thumb_pos_palm = np.asarray(thumb_pos_palm, dtype=np.float32)
        qpos = np.asarray(qpos, dtype=np.float32)
        desired_vel_palm = desired_thumb_velocity_palm()
        thumb_pos_palm_now, palm_pos_world, palm_rot_world = controller.thumb_position_in_palm_frame(qpos)
        thumb_pos_world = palm_pos_world + palm_rot_world @ thumb_pos_palm_now
        desired_vel_world = palm_rot_world @ desired_vel_palm
        actual_vel_world = palm_rot_world @ np.asarray(actual_thumb_vel_palm, dtype=np.float32)

        robot_viewer.set_qpos(qpos)
        thumb_point.points = thumb_pos_world.reshape(1, 3).astype(np.float32)
        palm_frame.position = palm_pos_world.astype(np.float32)
        try:
            import viser.transforms as vtf

            palm_frame.wxyz = vtf.SO3.from_matrix(palm_rot_world.astype(np.float64)).wxyz.astype(np.float32)
        except Exception:
            pass
        desired_handle.points = _segment_from_origin_and_vector(thumb_pos_world, desired_vel_world, args.arrow_scale)
        actual_handle.points = _segment_from_origin_and_vector(thumb_pos_world, actual_vel_world, args.arrow_scale)

        thumb_pos_text.value = ", ".join(f"{value:+.3f}" for value in thumb_pos_palm_now.tolist())
        desired_vel_text.value = ", ".join(f"{value:+.3f}" for value in desired_vel_palm.tolist())
        actual_vel_text.value = ", ".join(f"{value:+.3f}" for value in np.asarray(actual_thumb_vel_palm, dtype=np.float32).tolist())
        latent_norm_text.value = f"{float(np.linalg.norm(controller.z_state)):.3f}"
        latent_speed_text.value = f"{float(np.linalg.norm(z_dot)):.3f}"
        qpos_norm_text.value = f"{float(np.linalg.norm(qpos)):.3f}"
        jacobian_rank_text.value = str(int(np.linalg.matrix_rank(jacobian)))

    refresh_visuals(
        qpos=controller.q_state,
        thumb_pos_palm=controller.thumb_position_in_palm_frame(controller.q_state)[0],
        actual_thumb_vel_palm=np.zeros((3,), dtype=np.float32),
        z_dot=np.zeros((controller.z_dim,), dtype=np.float32),
        jacobian=np.zeros((3, controller.z_dim), dtype=np.float32),
    )

    print(f"PCA thumb velocity viewer running on {args.host}:{args.port}")
    while True:
        controller.latent_damping = float(damping_slider.value)
        controller.latent_speed_limit = float(speed_limit_slider.value)

        if running_checkbox.value:
            state = controller.step(desired_thumb_velocity_palm())
            refresh_visuals(
                qpos=state["qpos"],
                thumb_pos_palm=state["thumb_pos_palm"],
                actual_thumb_vel_palm=state["actual_thumb_vel_palm"],
                z_dot=state["z_dot"],
                jacobian=state["jacobian"],
            )
        else:
            refresh_visuals(
                qpos=controller.q_state,
                thumb_pos_palm=controller.thumb_position_in_palm_frame(controller.q_state)[0],
                actual_thumb_vel_palm=controller.actual_thumb_vel_palm,
                z_dot=np.zeros((controller.z_dim,), dtype=np.float32),
                jacobian=np.zeros((3, controller.z_dim), dtype=np.float32),
            )
        time.sleep(args.ctrl_dt)


if __name__ == "__main__":
    main()
