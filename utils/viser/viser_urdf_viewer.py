from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class ViserHandViewer:
    def __init__(
        self,
        urdf_path: str,
        input_joint_names: Optional[Sequence[str]] = None,
        *,
        server=None,
        host: str = "0.0.0.0",
        port: int = 8080,
        root_node_name: str = "/hand",
        mesh_color_override: Optional[Tuple[float, float, float, float]] = None,
        collision_mesh_color_override: Optional[
            Tuple[float, float, float, float]
        ] = None,
        load_collision_meshes: bool = False,
    ):
        try:
            import viser
            from viser.extras import ViserUrdf
            import yourdfpy
        except ImportError as exc:
            raise ImportError(
                "Viser URDF viewer requires: pip install viser yourdfpy"
            ) from exc

        urdf_path = str(Path(urdf_path).absolute())
        urdf_model = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)

        self.server = server if server is not None else viser.ViserServer(host=host, port=port)
        self.robot = ViserUrdf(
            self.server,
            urdf_model,
            root_node_name=root_node_name,
            mesh_color_override=mesh_color_override,
            collision_mesh_color_override=collision_mesh_color_override,
            load_collision_meshes=load_collision_meshes,
        )

        self.urdf_joint_names = self._get_actuated_joint_names()
        self.input_joint_names = (
            list(input_joint_names)
            if input_joint_names is not None
            else list(self.urdf_joint_names)
        )

        missing = [name for name in self.urdf_joint_names if name not in self.input_joint_names]
        if missing:
            raise ValueError(
                "Input qpos joint names do not cover all URDF actuated joints. "
                f"Missing: {missing}"
            )

        self._input_idx_by_urdf_order = [
            self.input_joint_names.index(name) for name in self.urdf_joint_names
        ]
        self._cfg = np.zeros(len(self.urdf_joint_names), dtype=np.float32)
        self._apply_cfg(self._cfg)

    def _get_actuated_joint_names(self):
        if hasattr(self.robot, "get_actuated_joint_names"):
            return list(self.robot.get_actuated_joint_names())
        names = getattr(self.robot, "actuated_joint_names", None)
        if names is not None:
            return list(names)
        raise RuntimeError("Could not query actuated joint names from ViserUrdf.")

    def _apply_cfg(self, cfg: np.ndarray):
        if hasattr(self.robot, "update_cfg"):
            self.robot.update_cfg(cfg)
            return
        if hasattr(self.robot, "set_cfg"):
            self.robot.set_cfg(cfg)
            return
        raise RuntimeError("Could not apply qpos to ViserUrdf (no update_cfg/set_cfg).")

    def set_qpos(self, qpos: Sequence[float]):
        qpos_np = np.asarray(qpos, dtype=np.float32)
        if qpos_np.shape[0] != len(self.input_joint_names):
            raise ValueError(
                f"Expected qpos dim {len(self.input_joint_names)}, got {qpos_np.shape[0]}"
            )

        self._cfg[:] = qpos_np[self._input_idx_by_urdf_order]
        self._apply_cfg(self._cfg)
