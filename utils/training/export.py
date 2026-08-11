import json
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from utils.training.config import sanitize_config


def collect_latent_codes(
    model,
    qpos,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    if torch is None:
        raise ImportError("collect_latent_codes requires PyTorch to be installed.")
    model.eval()
    latents = []
    with torch.no_grad():
        for start in range(0, len(qpos), batch_size):
            batch = qpos[start : start + batch_size].to(device)
            latents.append(model.encode(batch).detach().cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float32)


def collect_reconstructions(
    model,
    qpos,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    if torch is None:
        raise ImportError("collect_reconstructions requires PyTorch to be installed.")
    model.eval()
    reconstructions = []
    with torch.no_grad():
        for start in range(0, len(qpos), batch_size):
            batch = qpos[start : start + batch_size].to(device)
            z = model.encode(batch)
            recon = model.decode(z)
            reconstructions.append(recon.detach().cpu().numpy())
    return np.concatenate(reconstructions, axis=0).astype(np.float32)


def compute_latent_statistics(latent_codes: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "latent_codes": latent_codes.astype(np.float32),
        "mean": latent_codes.mean(axis=0).astype(np.float32),
        "std": latent_codes.std(axis=0).astype(np.float32),
        "min": latent_codes.min(axis=0).astype(np.float32),
        "max": latent_codes.max(axis=0).astype(np.float32),
        "p01": np.percentile(latent_codes, 1, axis=0).astype(np.float32),
        "p99": np.percentile(latent_codes, 99, axis=0).astype(np.float32),
    }


def compute_reconstruction_statistics(
    qpos: np.ndarray,
    recon_qpos: np.ndarray,
) -> dict[str, np.ndarray]:
    error = recon_qpos - qpos
    l2 = np.linalg.norm(error, axis=1).astype(np.float32)
    mse = np.mean(error**2, axis=1).astype(np.float32)
    return {
        "qpos": qpos.astype(np.float32),
        "reconstructed_qpos": recon_qpos.astype(np.float32),
        "error": error.astype(np.float32),
        "l2": l2,
        "mse": mse,
        "mean_l2": np.asarray(l2.mean(), dtype=np.float32),
        "std_l2": np.asarray(l2.std(), dtype=np.float32),
        "p95_l2": np.asarray(np.percentile(l2, 95), dtype=np.float32),
        "p99_l2": np.asarray(np.percentile(l2, 99), dtype=np.float32),
        "mean_mse": np.asarray(mse.mean(), dtype=np.float32),
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_model_artifacts(
    cfg: dict,
    model,
    output_dir: Path,
) -> dict[str, str]:
    if torch is None:
        raise ImportError("save_model_artifacts requires PyTorch to be installed.")
    model_cfg = sanitize_config(cfg)["model"]

    autoencoder_path = output_dir / "autoencoder.pt"
    encoder_path = output_dir / "encoder.pt"
    decoder_path = output_dir / "decoder.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg,
        },
        autoencoder_path,
    )
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "encoder_config": model_cfg["encoder"],
            "x_dim": int(model_cfg["x_dim"]),
            "z_dim": int(model_cfg["z_dim"]),
        },
        encoder_path,
    )
    torch.save(
        {
            "decoder_state_dict": model.decoder.state_dict(),
            "decoder_config": model_cfg["decoder"],
            "x_dim": int(model_cfg["x_dim"]),
            "z_dim": int(model_cfg["z_dim"]),
        },
        decoder_path,
    )

    return {
        "autoencoder": autoencoder_path.name,
        "encoder": encoder_path.name,
        "decoder": decoder_path.name,
    }


if torch is not None:
    class ScriptedEncoder(nn.Module):
        def __init__(self, encoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            half_chan = z.shape[1] // 2
            return z[:, :half_chan]


    class ScriptedDecoder(nn.Module):
        def __init__(self, decoder: torch.nn.Module):
            super().__init__()
            self.decoder = decoder

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)


    class ScriptedAutoencoder(nn.Module):
        def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        @torch.jit.export
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            half_chan = z.shape[1] // 2
            return z[:, :half_chan]

        @torch.jit.export
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        @torch.jit.export
        def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.reconstruct(x)


def save_scripted_artifacts(
    model,
    output_dir: Path,
) -> dict[str, str]:
    if torch is None:
        raise ImportError("save_scripted_artifacts requires PyTorch to be installed.")
    encoder_scripted_path = output_dir / "encoder_scripted.pt"
    decoder_scripted_path = output_dir / "decoder_scripted.pt"
    autoencoder_scripted_path = output_dir / "autoencoder_scripted.pt"

    encoder_scripted = torch.jit.script(ScriptedEncoder(model.encoder))
    decoder_scripted = torch.jit.script(ScriptedDecoder(model.decoder))
    autoencoder_scripted = torch.jit.script(ScriptedAutoencoder(model.encoder, model.decoder))

    encoder_scripted.save(str(encoder_scripted_path))
    decoder_scripted.save(str(decoder_scripted_path))
    autoencoder_scripted.save(str(autoencoder_scripted_path))

    return {
        "encoder_scripted": encoder_scripted_path.name,
        "decoder_scripted": decoder_scripted_path.name,
        "autoencoder_scripted": autoencoder_scripted_path.name,
    }
