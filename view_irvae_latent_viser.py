import argparse
import time
from pathlib import Path

import numpy as np

from utils.system.path_utils import resolve_repo_path
from utils.training import (
    build_irvae_model,
    decode_latent,
    load_config,
    load_dataset_metadata,
    load_model_checkpoint,
    resolve_device,
)
from utils.viser.viser_urdf_viewer import ViserHandViewer


REPO_ROOT = Path(__file__).resolve().parent
IRVAE_ROOT = REPO_ROOT / "IRVAE"
DEFAULT_CONFIG_PATH = "configs/train/irvae_z4.yaml"
DEFAULT_CHECKPOINT_PATH = "results/retargeted_grasps/irvae/qpos_z4/model_best.pkl"
DEFAULT_DATASET_DIR = "assets/datasets/retargeted_grasps_test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--slider-min", type=float, default=-2.0)
    parser.add_argument("--slider-max", type=float, default=2.0)
    parser.add_argument("--slider-step", type=float, default=0.01)
    args = parser.parse_args()

    try:
        import viser
    except ImportError as exc:
        raise ImportError("This script requires: pip install viser") from exc

    cfg = load_config(args.config, REPO_ROOT)
    z_dim = int(cfg.model.z_dim)

    device = resolve_device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint, REPO_ROOT)
    metadata = load_dataset_metadata(args.dataset_path, REPO_ROOT)

    model = build_irvae_model(cfg, IRVAE_ROOT).to(device)
    load_model_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()

    server = viser.ViserServer(host=args.host, port=args.port)
    robot_viewer = ViserHandViewer(
        urdf_path=metadata["urdf_path"],
        input_joint_names=metadata["robot_joint_names"],
        server=server,
        root_node_name="/robot",
        mesh_color_override=(0.55, 0.90, 0.55, 0.85),
    )

    with server.gui.add_folder("Latent Space"):
        latent_sliders = [
            server.gui.add_slider(
                f"z{index}",
                min=args.slider_min,
                max=args.slider_max,
                step=args.slider_step,
                initial_value=0.0,
            )
            for index in range(z_dim)
        ]
        qpos_norm_text = server.gui.add_text("qpos_norm", initial_value="0.000", disabled=True)

    with server.gui.add_folder("Model"):
        server.gui.add_text("checkpoint", initial_value=str(checkpoint_path), disabled=True)
        server.gui.add_text("robot_name", initial_value=str(metadata["robot_name"]), disabled=True)
        server.gui.add_text("z_dim", initial_value=str(z_dim), disabled=True)
        server.gui.add_text("qpos_dim", initial_value=str(cfg.model.x_dim), disabled=True)

    def refresh() -> None:
        latent = np.asarray([slider.value for slider in latent_sliders], dtype=np.float32)
        qpos = decode_latent(model, latent, device=device)
        robot_viewer.set_qpos(qpos)
        qpos_norm_text.value = f"{float(np.linalg.norm(qpos)):.3f}"

    for slider in latent_sliders:
        @slider.on_update
        def _(_event):
            refresh()

    refresh()
    print(f"Latent viewer running on {args.host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
