import argparse
import shutil
from pathlib import Path

import numpy as np

from utils.system.path_utils import resolve_repo_path
from utils.training import (
    build_irvae_model,
    collect_latent_codes,
    collect_reconstructions,
    compute_latent_statistics,
    compute_reconstruction_statistics,
    load_config,
    load_dataset_metadata,
    load_model_checkpoint,
    resolve_device,
    save_config,
    save_json,
    save_model_artifacts,
    save_scripted_artifacts,
)
from utils.training.data import load_robot_qpos


REPO_ROOT = Path(__file__).resolve().parent
IRVAE_ROOT = REPO_ROOT / "IRVAE"
DEFAULT_CONFIG_PATH = "configs/train/irvae_test.yaml"
DEFAULT_CHECKPOINT_PATH = "results/retargeted_grasps/irvae/qpos_z3_test/model_best.pkl"
DEFAULT_DATASET_DIR = "assets/datasets/retargeted_grasps_test"
DEFAULT_EXPORT_ROOT = "exports"


def default_export_dir(checkpoint_path: Path) -> Path:
    return REPO_ROOT / DEFAULT_EXPORT_ROOT / checkpoint_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config, REPO_ROOT)
    checkpoint_path = resolve_repo_path(args.checkpoint, REPO_ROOT)
    dataset_path = resolve_repo_path(args.dataset_path, REPO_ROOT)
    output_dir = (
        resolve_repo_path(args.output_dir, REPO_ROOT)
        if args.output_dir is not None
        else default_export_dir(checkpoint_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    metadata = load_dataset_metadata(str(dataset_path), REPO_ROOT)
    qpos = load_robot_qpos(str(dataset_path), REPO_ROOT)

    model = build_irvae_model(cfg, IRVAE_ROOT).to(device)
    load_model_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()

    latent_codes = collect_latent_codes(model, qpos, batch_size=args.batch_size, device=device)
    recon_qpos = collect_reconstructions(model, qpos, batch_size=args.batch_size, device=device)
    latent_stats = compute_latent_statistics(latent_codes)
    reconstruction_stats = compute_reconstruction_statistics(qpos.numpy(), recon_qpos)
    model_files = save_model_artifacts(cfg, model, output_dir)
    scripted_files = save_scripted_artifacts(model, output_dir)

    save_config(cfg, output_dir, "config.yaml")
    shutil.copy2(checkpoint_path, output_dir / "checkpoint_full.pt")
    save_json(metadata, output_dir / "robot_metadata.json")
    np.savez_compressed(output_dir / "latent_stats.npz", **latent_stats)
    np.savez_compressed(output_dir / "reconstruction_stats.npz", **reconstruction_stats)

    manifest = {
        "config": "config.yaml",
        "checkpoint_full": "checkpoint_full.pt",
        "robot_metadata": "robot_metadata.json",
        "latent_stats": "latent_stats.npz",
        "reconstruction_stats": "reconstruction_stats.npz",
        "model_files": model_files,
        "scripted_files": scripted_files,
        "dataset_path": str(dataset_path),
        "source_checkpoint": str(checkpoint_path),
        "z_dim": int(cfg.model.z_dim),
        "qpos_dim": int(cfg.model.x_dim),
        "robot_joint_names": metadata["robot_joint_names"],
        "num_samples": int(qpos.shape[0]),
    }
    save_json(manifest, output_dir / "manifest.json")

    print(f"Exported artifacts to {output_dir}")


if __name__ == "__main__":
    main()
