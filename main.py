import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything

from configs.config import config_network, parse_arguments
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint


DEFAULT_CONFIG = "configs/tspn_suda_demo.yaml"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSPN demo trainer")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to the TSPN demo YAML config.")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --config. --config takes precedence.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override device.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument("--notes", type=str, default="", help="Optional run note printed to stdout.")
    return parser.parse_args()


def resolve_config_path(cli_args: argparse.Namespace) -> str:
    if cli_args.config != DEFAULT_CONFIG:
        return cli_args.config
    return cli_args.config_dir or cli_args.config


def apply_cli_overrides(args: argparse.Namespace, cli_args: argparse.Namespace) -> argparse.Namespace:
    if cli_args.device is not None:
        args.device = cli_args.device
    if cli_args.epochs is not None:
        args.num_epochs = cli_args.epochs
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.patience is not None:
        args.patience = cli_args.patience

    if str(args.device).lower() == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    args.gpus = 1 if str(args.device).lower() == "cpu" else int(getattr(args, "gpus", 1))
    args.num_workers = int(getattr(args, "num_workers", 0))
    args.pin_memory = bool(getattr(args, "pin_memory", False))
    args.log_parameters = bool(getattr(args, "log_parameters", False))
    args.pruning = getattr(args, "pruning", None)
    args.monitor = getattr(args, "monitor", "val_loss")
    args.patience = int(getattr(args, "patience", 20))
    return args


def build_tspn_model(configs: dict, args: argparse.Namespace) -> Basic_plmodel:
    if args.model != "TSPN":
        raise ValueError(f"TSPN demo only supports model='TSPN', got {args.model!r}.")

    signal_processing_modules, feature_extractor_modules = config_network(configs, args)
    network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules, args)
    model = Basic_plmodel(network, args)
    print(model.network)
    return model


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    cli_args = parse_cli_args()
    config_path = resolve_config_path(cli_args)
    configs, args, path, name = parse_arguments(config_path, 0)
    args = apply_cli_overrides(args, cli_args)

    seed_everything(args.seed)
    if cli_args.notes:
        print(f"Run notes: {cli_args.notes}")

    model = build_tspn_model(configs, args)
    trainer, train_dataloader, val_dataloader, test_dataloader = trainer_set(args, path)

    trainer.fit(model, train_dataloader, val_dataloader)
    model = load_best_model_checkpoint(model, trainer)
    result = trainer.test(model, test_dataloader)

    result_df = pd.DataFrame(result)
    result_path = Path(path) / "test_result.csv"
    os.makedirs(path, exist_ok=True)
    result_df.to_csv(result_path, index=False)

    print(f"Run name: {name}")
    print(f"Run directory: {path}")
    print(f"Test result: {result}")
    print(f"Saved test result: {result_path}")


if __name__ == "__main__":
    main()
