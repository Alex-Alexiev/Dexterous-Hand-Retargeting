import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def prepare_irvae_imports(irvae_root: Path) -> None:
    root_str = str(irvae_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_irvae_module(irvae_root: Path, module_name: str, relative_path: str):
    module_path = irvae_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, geometry_lib):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.geometry_lib = geometry_lib

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        half_chan = z.shape[1] // 2
        return z[:, :half_chan]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample_latent(self, z: torch.Tensor) -> torch.Tensor:
        half_chan = z.shape[1] // 2
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sig) * eps

    def kl_loss(self, z: torch.Tensor) -> torch.Tensor:
        half_chan = z.shape[1] // 2
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x: torch.Tensor, optimizer, **kwargs) -> dict:
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        loss = (nll + kl_loss).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, x: torch.Tensor, **kwargs) -> dict:
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        loss = (nll + kl_loss).mean()
        return {"loss": loss.item()}

    def eval_step(self, dl, **kwargs) -> dict:
        device = kwargs["device"]
        scores = []
        for x, _ in dl:
            z = self.encode(x.to(device))
            metric = self.geometry_lib.get_pullbacked_Riemannian_metric(self.decode, z)
            scores.append(self.geometry_lib.get_flattening_scores(metric, mode="condition_number"))
        mean_condition_number = torch.cat(scores).mean()
        return {"MCN_": mean_condition_number.item()}

    def visualization_step(self, dl, **kwargs) -> dict:
        return {}


class IRVAEModel(VAEModel):
    def __init__(self, encoder, decoder, geometry_lib, *, iso_reg: float, metric: str):
        super().__init__(encoder, decoder, geometry_lib)
        self.iso_reg = iso_reg
        self.metric = metric

    def train_step(self, x: torch.Tensor, optimizer, **kwargs) -> dict:
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = self.geometry_lib.relaxed_distortion_measure(
            self.decode,
            z_sample,
            eta=0.2,
            metric=self.metric,
        )
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}


def build_irvae_model(cfg: dict, irvae_root: Path) -> torch.nn.Module:
    modules_lib = load_irvae_module(irvae_root, "irvae_modules", "models/modules.py")
    geometry_lib = load_irvae_module(irvae_root, "irvae_geometry", "geometry.py")
    model_cfg = cfg["model"]

    def build_net(in_dim: int, out_dim: int, net_cfg: dict):
        arch = net_cfg["arch"]
        kwargs = {key: value for key, value in net_cfg.items() if key != "arch"}
        if arch == "fc_vec":
            return modules_lib.FC_vec(in_chan=in_dim, out_chan=out_dim, **kwargs)
        if arch == "fc_image":
            return modules_lib.FC_image(in_chan=in_dim, out_chan=out_dim, **kwargs)
        if arch == "conv28":
            return modules_lib.ConvNet28(in_chan=in_dim, out_chan=out_dim, **kwargs)
        if arch == "dconv28":
            return modules_lib.DeConvNet28(in_chan=in_dim, out_chan=out_dim, **kwargs)
        raise ValueError(f"Unsupported network architecture: {arch}")

    encoder = build_net(model_cfg["x_dim"], model_cfg["z_dim"] * 2, model_cfg["encoder"])
    decoder_net = build_net(model_cfg["z_dim"], model_cfg["x_dim"], model_cfg["decoder"])
    decoder = modules_lib.IsotropicGaussian(decoder_net)

    if model_cfg["arch"] == "irvae":
        return IRVAEModel(
            encoder,
            decoder,
            geometry_lib,
            iso_reg=model_cfg.get("iso_reg", 1.0),
            metric=model_cfg.get("metric", "identity"),
        )
    if model_cfg["arch"] == "vae":
        return VAEModel(encoder, decoder, geometry_lib)
    raise ValueError(f"Unsupported model architecture: {model_cfg['arch']}")


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def decode_latent(
    model: torch.nn.Module,
    latent: torch.Tensor | np.ndarray | list[float],
    *,
    device: str | torch.device,
) -> np.ndarray:
    latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        qpos = model.decode(latent_tensor).squeeze(0).detach().cpu().numpy()
    return qpos.astype(np.float32)
