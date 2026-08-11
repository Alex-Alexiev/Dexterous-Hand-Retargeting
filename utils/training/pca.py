from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def _as_numpy_2d(value) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def fit_pca(
    qpos,
    *,
    z_dim: int,
    center: bool = True,
) -> dict[str, Any]:
    samples = _as_numpy_2d(qpos).astype(np.float64, copy=False)
    num_samples, x_dim = samples.shape

    if num_samples < 2:
        raise ValueError("Need at least 2 samples to fit PCA.")
    if z_dim < 1 or z_dim > x_dim:
        raise ValueError(f"z_dim must be in [1, {x_dim}], got {z_dim}.")

    mean = samples.mean(axis=0) if center else np.zeros(x_dim, dtype=np.float64)
    centered = samples - mean[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    explained_variance_full = (singular_values**2) / (num_samples - 1)
    explained_variance = explained_variance_full[:z_dim]
    total_variance = float(explained_variance_full.sum())
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    return {
        "mean": mean.astype(np.float32),
        "components": vt[:z_dim].astype(np.float32),
        "singular_values": singular_values[:z_dim].astype(np.float32),
        "explained_variance": explained_variance.astype(np.float32),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float32),
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio).astype(np.float32),
        "center": np.asarray(center),
        "x_dim": np.asarray(x_dim, dtype=np.int64),
        "z_dim": np.asarray(z_dim, dtype=np.int64),
        "num_samples": np.asarray(num_samples, dtype=np.int64),
    }


def encode_pca(
    model: dict[str, Any],
    qpos,
) -> np.ndarray:
    samples = _as_numpy_2d(qpos)
    mean = np.asarray(model["mean"], dtype=np.float32)
    components = np.asarray(model["components"], dtype=np.float32)
    if bool(np.asarray(model.get("center", True)).item()):
        samples = samples - mean[None, :]
    return (samples @ components.T).astype(np.float32)


def decode_pca(
    model: dict[str, Any],
    latent,
) -> np.ndarray:
    latent_array = _as_numpy_2d(latent)
    mean = np.asarray(model["mean"], dtype=np.float32)
    components = np.asarray(model["components"], dtype=np.float32)
    qpos = latent_array @ components
    if bool(np.asarray(model.get("center", True)).item()):
        qpos = qpos + mean[None, :]
    return qpos.astype(np.float32)


def save_pca_model(model: dict[str, Any], output_path: Path) -> None:
    np.savez_compressed(output_path, **model)


def load_pca_model(model_path: str | Path) -> dict[str, Any]:
    with np.load(model_path, allow_pickle=False) as payload:
        model = {key: payload[key] for key in payload.files}
    model["center"] = bool(np.asarray(model.get("center", True)).item())
    model["x_dim"] = int(np.asarray(model["x_dim"]).item())
    model["z_dim"] = int(np.asarray(model["z_dim"]).item())
    model["num_samples"] = int(np.asarray(model["num_samples"]).item())
    return model


if torch is not None:
    class PCAEncoder(nn.Module):
        def __init__(self, mean: np.ndarray, components: np.ndarray, *, center: bool):
            super().__init__()
            self.center_inputs = center
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1))
            self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.center_inputs:
                x = x - self.mean
            return x @ self.components.t()


    class PCADecoder(nn.Module):
        def __init__(self, mean: np.ndarray, components: np.ndarray, *, center: bool):
            super().__init__()
            self.center_outputs = center
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1))
            self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = z @ self.components
            if self.center_outputs:
                x = x + self.mean
            return x


    class PCAAutoencoder(nn.Module):
        def __init__(self, mean: np.ndarray, components: np.ndarray, *, center: bool):
            super().__init__()
            self.encoder = PCAEncoder(mean, components, center=center)
            self.decoder = PCADecoder(mean, components, center=center)

        @torch.jit.export
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        @torch.jit.export
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        @torch.jit.export
        def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.reconstruct(x)


def save_pca_torch_artifacts(
    model: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    if torch is None:
        raise ImportError("save_pca_torch_artifacts requires PyTorch to be installed.")
    mean = np.asarray(model["mean"], dtype=np.float32)
    components = np.asarray(model["components"], dtype=np.float32)
    center = bool(model.get("center", True))
    x_dim = int(model["x_dim"])
    z_dim = int(model["z_dim"])

    encoder = PCAEncoder(mean, components, center=center)
    decoder = PCADecoder(mean, components, center=center)
    autoencoder = PCAAutoencoder(mean, components, center=center)

    encoder_path = output_dir / "encoder.pt"
    decoder_path = output_dir / "decoder.pt"
    autoencoder_path = output_dir / "autoencoder.pt"
    encoder_scripted_path = output_dir / "encoder_scripted.pt"
    decoder_scripted_path = output_dir / "decoder_scripted.pt"
    autoencoder_scripted_path = output_dir / "autoencoder_scripted.pt"

    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "x_dim": x_dim,
            "z_dim": z_dim,
            "center": center,
        },
        encoder_path,
    )
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "x_dim": x_dim,
            "z_dim": z_dim,
            "center": center,
        },
        decoder_path,
    )
    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "x_dim": x_dim,
            "z_dim": z_dim,
            "center": center,
        },
        autoencoder_path,
    )

    torch.jit.script(encoder).save(str(encoder_scripted_path))
    torch.jit.script(decoder).save(str(decoder_scripted_path))
    torch.jit.script(autoencoder).save(str(autoencoder_scripted_path))

    return (
        {
            "autoencoder": autoencoder_path.name,
            "encoder": encoder_path.name,
            "decoder": decoder_path.name,
        },
        {
            "encoder_scripted": encoder_scripted_path.name,
            "decoder_scripted": decoder_scripted_path.name,
            "autoencoder_scripted": autoencoder_scripted_path.name,
        },
    )
