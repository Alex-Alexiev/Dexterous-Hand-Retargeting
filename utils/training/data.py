from pathlib import Path
import json

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    DataLoader = None
    TensorDataset = None

from utils.system.path_utils import resolve_repo_path


def _require_torch() -> None:
    if torch is None:
        raise ImportError("This operation requires PyTorch to be installed.")


def load_robot_qpos(dataset_path: str, repo_root: Path):
    resolved_path = resolve_repo_path(dataset_path, repo_root)
    data_path = resolved_path / "data.npz" if resolved_path.is_dir() else resolved_path

    with np.load(data_path, allow_pickle=False) as data:
        if "retargeted_qpos" not in data:
            raise KeyError(f"'retargeted_qpos' not found in {data_path}.")
        qpos = data["retargeted_qpos"].astype(np.float32)

    if torch is None:
        return qpos
    return torch.from_numpy(qpos)


def load_dataset_metadata(dataset_path: str, repo_root: Path) -> dict:
    resolved_path = resolve_repo_path(dataset_path, repo_root)
    metadata_path = resolved_path / "metadata.json" if resolved_path.is_dir() else resolved_path
    if metadata_path.name != "metadata.json":
        metadata_path = metadata_path.parent / "metadata.json"
    with metadata_path.open("r") as f:
        return json.load(f)


def split_robot_qpos(
    qpos,
    *,
    split_ratio: float,
    seed: int,
):
    num_samples = qpos.shape[0]
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to create training and validation splits.")

    num_train = min(max(int(num_samples * split_ratio), 1), num_samples - 1)
    if torch is not None and torch.is_tensor(qpos):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_samples)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    return qpos[train_idx], qpos[val_idx]


def make_qpos_dataloader(
    qpos,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    _require_torch()
    labels = torch.zeros(len(qpos), dtype=torch.long)
    dataset = TensorDataset(qpos, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_qpos_dataloaders(cfg: dict, repo_root: Path, default_dataset_dir: str) -> dict[str, DataLoader]:
    _require_torch()
    train_cfg = cfg["data"]["training"]
    val_cfg = cfg["data"]["validation"]

    qpos = load_robot_qpos(train_cfg.get("dataset_path", default_dataset_dir), repo_root)
    train_qpos, val_qpos = split_robot_qpos(
        qpos,
        split_ratio=train_cfg.get("split_ratio", 0.9),
        seed=train_cfg.get("seed", 0),
    )

    return {
        "training": make_qpos_dataloader(
            train_qpos,
            batch_size=train_cfg["batch_size"],
            shuffle=train_cfg.get("shuffle", True),
            num_workers=train_cfg.get("n_workers", 0),
        ),
        "validation": make_qpos_dataloader(
            val_qpos,
            batch_size=val_cfg["batch_size"],
            shuffle=val_cfg.get("shuffle", False),
            num_workers=val_cfg.get("n_workers", 0),
        ),
    }
