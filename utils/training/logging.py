from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from utils.training.config import sanitize_config


class WandbWriter:
    def __init__(self, cfg: dict, logdir: Path, run_name: str | None):
        import wandb

        wandb_cfg = cfg.get("wandb", {})
        self._wandb = wandb
        self._run = wandb.init(
            project=wandb_cfg.get("project", "irvae"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            dir=str(logdir),
            config=sanitize_config(cfg),
            mode=wandb_cfg.get("mode"),
        )

    def add_scalar(self, tag: str, value, step: int) -> None:
        self._wandb.log({tag: float(value)}, step=step)

    def add_image(self, tag: str, image, step: int) -> None:
        image_np = (
            image.detach().cpu().numpy()
            if torch is not None and torch.is_tensor(image)
            else np.asarray(image)
        )
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
            image_np = np.moveaxis(image_np, 0, -1)
        self._wandb.log({tag: self._wandb.Image(image_np)}, step=step)

    def close(self) -> None:
        self._run.finish()


def make_writer(cfg: dict, logdir: Path, run_name: str | None):
    backend = cfg.get("logging", {}).get("backend", "wandb")
    if backend == "wandb":
        return WandbWriter(cfg, logdir, run_name)
    if backend == "tensorboard":
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(logdir=str(logdir))
    raise ValueError(f"Unsupported logging backend: {backend}")
