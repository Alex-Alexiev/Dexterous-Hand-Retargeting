import argparse
import time
from pathlib import Path

import numpy as np

from utils.system.path_utils import resolve_repo_path
from utils.training.data import load_dataset_metadata
from utils.training.pca import decode_pca, load_pca_model
from utils.viser.viser_urdf_viewer import ViserHandViewer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "results/retargeted_grasps/pca/default/pca_model.npz"
DEFAULT_DATASET_DIR = "assets/datasets/retargeted_grasps_test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--slider-scale", type=float, default=2.0)
    parser.add_argument("--slider-step", type=float, default=0.01)
    args = parser.parse_args()

    try:
        import viser
    except ImportError as exc:
        raise ImportError("This script requires: pip install viser") from exc

    model_path = resolve_repo_path(args.model, REPO_ROOT)
    metadata = load_dataset_metadata(args.dataset_path, REPO_ROOT)
    pca_model = load_pca_model(model_path)

    z_dim = int(pca_model["z_dim"])
    explained = np.asarray(pca_model["explained_variance_ratio"], dtype=np.float32)

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
                f"pc{index + 1}",
                min=-args.slider_scale,
                max=args.slider_scale,
                step=args.slider_step,
                initial_value=0.0,
            )
            for index in range(z_dim)
        ]
        qpos_norm_text = server.gui.add_text("qpos_norm", initial_value="0.000", disabled=True)

    with server.gui.add_folder("Model"):
        server.gui.add_text("model", initial_value=str(model_path), disabled=True)
        server.gui.add_text("robot_name", initial_value=str(metadata["robot_name"]), disabled=True)
        server.gui.add_text("z_dim", initial_value=str(z_dim), disabled=True)
        server.gui.add_text("qpos_dim", initial_value=str(pca_model["x_dim"]), disabled=True)
        server.gui.add_text(
            "explained_variance",
            initial_value=", ".join(f"{ratio:.3f}" for ratio in explained),
            disabled=True,
        )

    def refresh() -> None:
        latent = np.asarray([slider.value for slider in latent_sliders], dtype=np.float32)
        qpos = decode_pca(pca_model, latent).reshape(-1)
        robot_viewer.set_qpos(qpos)
        qpos_norm_text.value = f"{float(np.linalg.norm(qpos)):.3f}"

    for slider in latent_sliders:
        @slider.on_update
        def _(_event):
            refresh()

    refresh()
    print(f"PCA latent viewer running on {args.host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
