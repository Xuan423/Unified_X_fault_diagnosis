#!/usr/bin/env python3
"""TSPN demo analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config_network
from data.datasets import Default_dataset
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel


DEFAULT_CONFIG = PROJECT_ROOT / "configs/tspn_suda_demo.yaml"
RUN_ROOT = PROJECT_ROOT / "save/task_TSPN_SUDA_DEMO/model_TSPN"
REPORT_PATH = PROJECT_ROOT / "reports/demo_tspn_analysis_report.md"
FIGURE_DIR = PROJECT_ROOT / "reports/demo_tspn_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TSPN demo post-analysis.")
    parser.add_argument("--run-dir", type=Path, default=None, help="TSPN demo run directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TSPN demo config path.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Analysis device.")
    parser.add_argument("--batch-size", type=int, default=64, help="Analysis batch size.")
    return parser.parse_args()


def is_valid_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "test_result.csv").exists()
        and (path / "logs").is_dir()
        and any(path.glob("*.ckpt"))
    )


def find_latest_run() -> Path:
    if not RUN_ROOT.exists():
        raise FileNotFoundError(f"No demo run root found: {RUN_ROOT}. Run python main.py first.")

    candidates = [path for path in RUN_ROOT.glob("model_TSPNtime*") if is_valid_run_dir(path)]
    if not candidates:
        raise FileNotFoundError(
            f"No valid TSPN demo run found under {RUN_ROOT}. Run python main.py first or pass --run-dir."
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def select_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(path for path in run_dir.glob("*.ckpt") if path.name != "last.ckpt")
    if checkpoints:
        return max(checkpoints, key=lambda item: item.stat().st_mtime)
    last = run_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {run_dir}.")


def load_config(config_path: Path) -> tuple[dict, SimpleNamespace]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, SimpleNamespace(**config["args"])


def resolve_asset_path(data_dir: str, target: str, suffix: str) -> Path:
    base = Path(data_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / f"{target}_{suffix}.npy"


def build_network(config: dict, args: SimpleNamespace, checkpoint_path: Path, device: torch.device):
    args.device = device.type
    signal_processing_modules, feature_extractor_modules = config_network(config, args)
    network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules, args)
    lightning_wrapper = Basic_plmodel(network, args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lightning_wrapper.load_state_dict(checkpoint["state_dict"])
    network = lightning_wrapper.network.to(device)
    network.eval()
    return network


def load_metrics_csv(run_dir: Path) -> pd.DataFrame:
    metrics_files = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {run_dir / 'logs'}.")
    return pd.read_csv(max(metrics_files, key=lambda item: item.stat().st_mtime))


def plot_loss_curve(metrics: pd.DataFrame, output_path: Path) -> str | None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for column, label in [("train_loss_epoch", "train_loss"), ("val_loss", "val_loss")]:
        if column in metrics:
            series = metrics[["epoch", column]].dropna()
            if not series.empty:
                ax.plot(series["epoch"], series[column], marker="o", label=label)
                plotted = True
    if not plotted:
        plt.close(fig)
        return "metrics.csv did not contain train_loss_epoch or val_loss rows."

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("TSPN Demo Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return None


def predict_and_collect_features(network, dataset: Default_dataset, batch_size: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    rows: list[dict[str, float | int]] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        row_index = 0
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            stage = x
            for layer in network.signal_processing_layers:
                stage = layer(stage)
            feature_tensor = network.feature_extractor_layers(stage)
            logits = network.clf(feature_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

            features.append(feature_tensor.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())

            probs_np = probs.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            label_np = y.detach().cpu().numpy()
            for local_idx in range(len(label_np)):
                row = {
                    "row_index": row_index,
                    "true_label": int(label_np[local_idx]),
                    "predicted_label": int(pred_np[local_idx]),
                    "confidence": float(probs_np[local_idx, pred_np[local_idx]]),
                }
                for class_idx in range(probs_np.shape[1]):
                    row[f"prob_class_{class_idx}"] = float(probs_np[local_idx, class_idx])
                rows.append(row)
                row_index += 1

    return (
        pd.DataFrame(rows),
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(predictions, axis=0),
    )


def plot_confusion(labels: np.ndarray, preds: np.ndarray, num_classes: int, output_path: Path) -> None:
    matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("TSPN Demo Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_filter_summary(network, output_path: Path) -> None:
    layer_count = len(network.signal_processing_layers)
    fig, axes = plt.subplots(layer_count, 1, figsize=(7, max(2.2 * layer_count, 4)))
    if layer_count == 1:
        axes = [axes]

    for idx, (ax, layer) in enumerate(zip(axes, network.signal_processing_layers), start=1):
        alpha = getattr(layer, "alpha_main", None)
        if alpha is None or alpha.numel() == 0:
            weight = layer.weight_connection.weight.detach()
            alpha = torch.softmax(weight / getattr(layer, "temperature", 1.0), dim=1)
        alpha_np = alpha.detach().cpu().numpy()
        sns.heatmap(alpha_np, cmap="viridis", ax=ax, cbar=idx == layer_count)
        ax.set_title(f"Layer {idx} main connection alpha")
        ax.set_xlabel("Input channel")
        ax.set_ylabel("Output channel")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_summary(features: np.ndarray, network, output_path: Path) -> pd.DataFrame:
    feature_names = list(network.feature_extractor_layers.feature_extractor_modules.keys())
    channel_count = network.channel_for_feature
    reshaped = features.reshape(features.shape[0], len(feature_names), channel_count)
    values = np.mean(np.abs(reshaped), axis=(0, 2))
    summary = pd.DataFrame({"feature": feature_names, "mean_abs_value": values})

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(summary, x="feature", y="mean_abs_value", color="#4c78a8", ax=ax)
    ax.set_title("TSPN Demo Feature Summary")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean absolute value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def finite_metric_summary(test_result: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for column in test_result.columns:
        value = float(test_result[column].iloc[0])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite test metric {column}: {value}")
        summary[column] = value
    return summary


def write_report(
    *,
    run_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    raw_data: np.ndarray,
    raw_labels: np.ndarray,
    metrics: pd.DataFrame,
    test_result: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    generated: list[Path],
    notes: list[str],
) -> None:
    test_metrics = finite_metric_summary(test_result)
    best_val = metrics["val_loss"].dropna().min() if "val_loss" in metrics else np.nan
    final_train = metrics["train_loss_epoch"].dropna().iloc[-1] if "train_loss_epoch" in metrics and not metrics["train_loss_epoch"].dropna().empty else np.nan
    accuracy = float((prediction_summary["true_label"] == prediction_summary["predicted_label"]).mean())
    label_counts = pd.Series(raw_labels).value_counts().sort_index().to_dict()
    pred_counts = prediction_summary["predicted_label"].value_counts().sort_index().to_dict()

    lines = [
        "# TSPN Demo Analysis Report",
        "",
        "## Run Summary",
        f"- Run directory: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- Config: `{config_path.relative_to(PROJECT_ROOT)}`",
        f"- Checkpoint: `{checkpoint_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## Data Summary",
        f"- Raw data shape: `{tuple(raw_data.shape)}`",
        f"- Raw data dtype: `{raw_data.dtype}`; analysis dtype: `float32`",
        f"- Raw label shape: `{tuple(raw_labels.shape)}`",
        f"- Raw label dtype: `{raw_labels.dtype}`; analysis dtype: `int64`",
        f"- Label counts: `{label_counts}`",
        "",
        "## Training Metrics",
        f"- Best val_loss: `{best_val}`",
        f"- Final train_loss_epoch: `{final_train}`",
        "",
        "## Test Result",
        *[f"- {key}: `{value}`" for key, value in test_metrics.items()],
        "",
        "## Signal Processing Weights",
        "- See `reports/demo_tspn_figures/filter_summary.png` for signal-layer alpha heatmaps.",
        "",
        "## Feature Summary",
        f"- Top feature by mean absolute value: `{feature_summary.sort_values('mean_abs_value', ascending=False).iloc[0]['feature']}`",
        "",
        "## Prediction Summary",
        f"- Test split rows: `{len(prediction_summary)}`",
        f"- Prediction accuracy on analysis split: `{accuracy}`",
        f"- Predicted label counts: `{pred_counts}`",
        "",
        "## Generated Artifacts",
        *[f"- `{path.relative_to(PROJECT_ROOT)}`" for path in generated],
    ]

    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli_args = parse_args()
    run_dir = cli_args.run_dir or find_latest_run()
    run_dir = run_dir.resolve()
    if not is_valid_run_dir(run_dir):
        raise FileNotFoundError(f"Invalid TSPN demo run: {run_dir}")

    device = torch.device(cli_args.device if cli_args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = cli_args.config.resolve()
    config, args = load_config(config_path)
    args.device = device.type
    args.num_workers = 0
    args.pin_memory = False

    checkpoint_path = select_checkpoint(run_dir)
    data_path = resolve_asset_path(args.data_dir, args.target, "data")
    label_path = resolve_asset_path(args.data_dir, args.target, "label")
    raw_data = np.load(data_path)
    raw_labels = np.load(label_path)

    network = build_network(config, args, checkpoint_path, device)
    dataset = Default_dataset(args, flag="test")
    prediction_summary, features, labels, preds = predict_and_collect_features(
        network, dataset, cli_args.batch_size, device
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    notes: list[str] = []

    metrics = load_metrics_csv(run_dir)
    loss_path = FIGURE_DIR / "loss_curve.png"
    loss_note = plot_loss_curve(metrics, loss_path)
    if loss_note:
        notes.append(f"loss_curve.png not generated: {loss_note}")
    else:
        generated.append(loss_path)

    confusion_path = FIGURE_DIR / "confusion_matrix.png"
    plot_confusion(labels, preds, args.num_classes, confusion_path)
    generated.append(confusion_path)

    filter_path = FIGURE_DIR / "filter_summary.png"
    plot_filter_summary(network, filter_path)
    generated.append(filter_path)

    feature_path = FIGURE_DIR / "feature_summary.png"
    feature_summary = plot_feature_summary(features, network, feature_path)
    generated.append(feature_path)

    prediction_path = FIGURE_DIR / "prediction_summary.csv"
    prediction_summary.to_csv(prediction_path, index=False)
    generated.append(prediction_path)

    feature_csv_path = FIGURE_DIR / "feature_summary.csv"
    feature_summary.to_csv(feature_csv_path, index=False)
    generated.append(feature_csv_path)

    test_result = pd.read_csv(run_dir / "test_result.csv")
    write_report(
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        raw_data=raw_data,
        raw_labels=raw_labels,
        metrics=metrics,
        test_result=test_result,
        prediction_summary=prediction_summary,
        feature_summary=feature_summary,
        generated=generated,
        notes=notes,
    )

    print(f"Run directory: {run_dir}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
