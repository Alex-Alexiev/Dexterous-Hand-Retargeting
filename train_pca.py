import argparse
from pathlib import Path

import numpy as np

from utils.system.path_utils import resolve_repo_path
from utils.training.config import load_config, make_logdir, save_config
from utils.training.data import load_robot_qpos, split_robot_qpos
from utils.training.export import (
    compute_latent_statistics,
    compute_reconstruction_statistics,
    save_json,
)
from utils.training.data import load_dataset_metadata
from utils.training.pca import decode_pca, encode_pca, fit_pca, save_pca_model, save_pca_torch_artifacts


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = "assets/datasets/retargeted_grasps_test"
DEFAULT_CONFIG_PATH = "configs/train/pca_test.yaml"


def _scalar(value: np.ndarray) -> float:
    return float(np.asarray(value).item())


def _save_stats(output_path: Path, stats: dict[str, np.ndarray]) -> dict[str, float]:
    np.savez_compressed(output_path, **stats)
    return {
        "mean_l2": _scalar(stats["mean_l2"]),
        "std_l2": _scalar(stats["std_l2"]),
        "p95_l2": _scalar(stats["p95_l2"]),
        "p99_l2": _scalar(stats["p99_l2"]),
        "mean_mse": _scalar(stats["mean_mse"]),
    }


def train(cfg: dict, *, run_name: str | None = None) -> None:
    train_cfg = cfg.data.training
    dataset_path = train_cfg.get("dataset_path", DEFAULT_DATASET_DIR)
    qpos = load_robot_qpos(dataset_path, REPO_ROOT)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to have shape [N, D], got {tuple(qpos.shape)}.")
    if int(cfg.model.x_dim) != int(qpos.shape[1]):
        raise ValueError(
            f"Config x_dim={cfg.model.x_dim} does not match dataset qpos dim={qpos.shape[1]}."
        )

    train_qpos, val_qpos = split_robot_qpos(
        qpos,
        split_ratio=train_cfg.get("split_ratio", 0.9),
        seed=train_cfg.get("seed", 0),
    )

    pca_model = fit_pca(
        train_qpos,
        z_dim=int(cfg.model.z_dim),
        center=bool(cfg.model.get("center", True)),
    )

    logdir = make_logdir(cfg, REPO_ROOT, run_name)
    save_config(cfg, logdir, cfg._config_path)
    save_pca_model(pca_model, logdir / "pca_model.npz")

    metadata = load_dataset_metadata(dataset_path, REPO_ROOT)
    save_json(metadata, logdir / "robot_metadata.json")

    train_latent = encode_pca(pca_model, train_qpos)
    val_latent = encode_pca(pca_model, val_qpos)
    full_latent = encode_pca(pca_model, qpos)

    train_recon = decode_pca(pca_model, train_latent)
    val_recon = decode_pca(pca_model, val_latent)
    full_recon = decode_pca(pca_model, full_latent)

    latent_stats = compute_latent_statistics(full_latent)
    np.savez_compressed(logdir / "latent_stats.npz", **latent_stats)
    np.savez_compressed(logdir / "train_latent_stats.npz", **compute_latent_statistics(train_latent))
    np.savez_compressed(logdir / "validation_latent_stats.npz", **compute_latent_statistics(val_latent))

    train_summary = _save_stats(
        logdir / "reconstruction_stats_training.npz",
        compute_reconstruction_statistics(np.asarray(train_qpos, dtype=np.float32), train_recon),
    )
    val_summary = _save_stats(
        logdir / "reconstruction_stats_validation.npz",
        compute_reconstruction_statistics(np.asarray(val_qpos, dtype=np.float32), val_recon),
    )
    full_summary = _save_stats(
        logdir / "reconstruction_stats.npz",
        compute_reconstruction_statistics(np.asarray(qpos, dtype=np.float32), full_recon),
    )

    model_files = {}
    scripted_files = {}
    try:
        model_files, scripted_files = save_pca_torch_artifacts(pca_model, logdir)
    except ImportError:
        pass

    resolved_dataset_path = resolve_repo_path(dataset_path, REPO_ROOT)
    summary = {
        "dataset_path": str(resolved_dataset_path),
        "num_samples": int(qpos.shape[0]),
        "train_samples": int(train_qpos.shape[0]),
        "validation_samples": int(val_qpos.shape[0]),
        "x_dim": int(cfg.model.x_dim),
        "z_dim": int(cfg.model.z_dim),
        "center": bool(cfg.model.get("center", True)),
        "explained_variance_ratio": np.asarray(pca_model["explained_variance_ratio"]).tolist(),
        "cumulative_explained_variance_ratio": np.asarray(
            pca_model["cumulative_explained_variance_ratio"]
        ).tolist(),
        "training_reconstruction": train_summary,
        "validation_reconstruction": val_summary,
        "full_reconstruction": full_summary,
    }
    save_json(summary, logdir / "summary.json")

    manifest = {
        "config": Path(cfg._config_path).name,
        "pca_model": "pca_model.npz",
        "robot_metadata": "robot_metadata.json",
        "summary": "summary.json",
        "latent_stats": "latent_stats.npz",
        "train_latent_stats": "train_latent_stats.npz",
        "validation_latent_stats": "validation_latent_stats.npz",
        "reconstruction_stats": "reconstruction_stats.npz",
        "reconstruction_stats_training": "reconstruction_stats_training.npz",
        "reconstruction_stats_validation": "reconstruction_stats_validation.npz",
        "model_files": model_files,
        "scripted_files": scripted_files,
        "dataset_path": summary["dataset_path"],
        "z_dim": int(cfg.model.z_dim),
        "qpos_dim": int(cfg.model.x_dim),
        "robot_joint_names": metadata["robot_joint_names"],
        "num_samples": int(qpos.shape[0]),
    }
    save_json(manifest, logdir / "manifest.json")

    print(f"Saved PCA experiment artifacts to {logdir}")
    print(
        "Explained variance ratio:",
        ", ".join(f"{ratio:.4f}" for ratio in pca_model["explained_variance_ratio"]),
    )
    print(
        f"Validation mean L2: {val_summary['mean_l2']:.6f} | "
        f"Validation mean MSE: {val_summary['mean_mse']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, REPO_ROOT)
    cfg._config_path = args.config
    train(cfg, run_name=args.run)


if __name__ == "__main__":
    main()
