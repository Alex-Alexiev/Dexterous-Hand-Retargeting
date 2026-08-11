from pathlib import Path
from typing import Any

import yaml

from utils.system.path_utils import resolve_repo_path


class AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: to_attr_dict(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def to_plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def load_config(config_path: str, repo_root: Path) -> AttrDict:
    resolved_path = resolve_repo_path(config_path, repo_root)
    with resolved_path.open("r") as f:
        return to_attr_dict(yaml.safe_load(f))


def save_config(cfg: dict, logdir: Path, config_path: str) -> None:
    config_name = Path(config_path).name
    config_to_save = to_plain_data(
        {key: value for key, value in cfg.items() if key != "_config_path"}
    )
    with (logdir / config_name).open("w") as f:
        yaml.safe_dump(config_to_save, f, sort_keys=False)


def sanitize_config(cfg: dict) -> dict:
    return to_plain_data({key: value for key, value in cfg.items() if key != "_config_path"})


def make_logdir(cfg: dict, repo_root: Path, run_name: str | None) -> Path:
    base_logdir = resolve_repo_path(cfg["logdir"], repo_root)
    run_dir = run_name or "default"
    logdir = base_logdir / run_dir
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir


def resolve_device(device_arg: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if device_arg == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda:{device_arg}"
    return "cpu"
