import argparse
from pathlib import Path

from utils.training import (
    build_irvae_model,
    build_qpos_dataloaders,
    load_config,
    make_logdir,
    make_writer,
    prepare_irvae_imports,
    resolve_device,
    save_config,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parent
IRVAE_ROOT = REPO_ROOT / "IRVAE"
DEFAULT_DATASET_DIR = "assets/datasets/retargeted_grasps_test"
DEFAULT_CONFIG_PATH = "configs/train/irvae_test.yaml"


def train(cfg: dict, *, run_name: str | None = None) -> None:
    prepare_irvae_imports(IRVAE_ROOT)

    from IRVAE.optimizers import get_optimizer
    from IRVAE.trainers import get_logger, get_trainer

    set_seed(cfg.training.get("seed", 0))
    dataloaders = build_qpos_dataloaders(cfg, REPO_ROOT, DEFAULT_DATASET_DIR)
    model = build_irvae_model(cfg, IRVAE_ROOT).to(cfg.device)
    optimizer = get_optimizer(cfg.training.optimizer, model.parameters())
    trainer = get_trainer(optimizer, cfg)

    logdir = make_logdir(cfg, REPO_ROOT, run_name)
    writer = make_writer(cfg, logdir, run_name)
    save_config(cfg, logdir, cfg._config_path)

    logger = get_logger(cfg, writer)
    trainer.train(model, dataloaders, logger=logger, logdir=str(logdir))
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, REPO_ROOT)
    cfg._config_path = args.config
    cfg.device = resolve_device(args.device)
    train(cfg, run_name=args.run)


if __name__ == "__main__":
    main()
