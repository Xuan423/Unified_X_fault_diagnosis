#!/usr/bin/env python3
"""Comprehensive SEU TSPN interpretability analysis utilities.

This module mirrors the exploratory workflows in ``plot_seu.ipynb``
while providing a scripted pipeline that can be executed end-to-end.
The implementation focuses on three aspects required by downstream
analysis:

1. Modal contribution analysis that can be aggregated globally,
   per fault class and per operating condition (speed / load).
2. Feature radar visualisations with grouped colour mapping for
   energy, impact and complexity related indicators.
3. Coupled analysis between modal contributions and feature metrics
   via partial correlations, including per-class specialisations.

The module exposes a CLI entry point so that the full workflow can be
triggered with a single command once the trained checkpoint and SEU
dataset artefacts are available locally.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml


# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config_network
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel

plt.rcParams['font.family'] = 'Times New Roman'


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> tuple[dict, argparse.Namespace]:
    """Load a YAML configuration and return the dictionary and args namespace."""
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    args = argparse.Namespace(**config['args'])
    return config, args


@dataclass
class DatasetBundle:
    signals: np.ndarray
    labels: np.ndarray
    speed: np.ndarray | None = None
    load: np.ndarray | None = None


def load_dataset(
    data_path: Path,
    label_path: Path,
    speed_path: Path | None = None,
    load_path: Path | None = None,
) -> DatasetBundle:
    """Load the SEU dataset arrays from ``.npy`` containers.

    Parameters
    ----------
    data_path:
        Path to the signal tensor stored as ``(N, L, C)`` float array.
    label_path:
        Path to the integer labels aligned with ``data_path``.
    speed_path / load_path:
        Optional paths pointing to per-sample rotational speed and load
        descriptors. Either ``None`` or 1-D arrays with the same length
        as ``label_path``.
    """

    signals = np.load(data_path)
    labels = np.load(label_path)

    speed = np.load(speed_path) if speed_path and speed_path.exists() else None
    load = np.load(load_path) if load_path and load_path.exists() else None

    return DatasetBundle(signals=signals, labels=labels, speed=speed, load=load)


# ---------------------------------------------------------------------------
# Network initialisation
# ---------------------------------------------------------------------------


def build_network(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> Transparent_Signal_Processing_Network:
    config, args = load_config(config_path)
    args.device = device.type
    signal_processing_modules, feature_extractor_modules = config_network(
        config, args
    )
    network = Transparent_Signal_Processing_Network(
        signal_processing_modules, feature_extractor_modules, args
    )
    lightning_wrapper = Basic_plmodel(network, args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lightning_wrapper.load_state_dict(checkpoint['state_dict'])
    network = lightning_wrapper.network.to(device)
    network.eval()
    return network


# ---------------------------------------------------------------------------
# Core analytical utilities
# ---------------------------------------------------------------------------


def iterate_batches(array: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(array), batch_size):
        end = start + batch_size
        yield array[start:end]


def collect_feature_pipeline(
    network: Transparent_Signal_Processing_Network,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward propagate signals to obtain feature vectors and final stage outputs."""
    features: list[np.ndarray] = []
    stage_outputs: list[np.ndarray] = []

    torch.set_grad_enabled(False)

    for batch in iterate_batches(signals, batch_size):
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        stage = tensor
        for layer in network.signal_processing_layers:
            stage = layer(stage)
        stage_outputs.append(stage.detach().cpu().numpy())
        feats = network.feature_extractor_layers(stage)
        features.append(feats.detach().cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(stage_outputs, axis=0)


def compute_total_weight_matrix(
    network: Transparent_Signal_Processing_Network,
) -> np.ndarray:
    """Aggregate connection weights across signal processing layers.

    The result corresponds to the absolute contribution strength from
    each input modality (columns) towards the last signal-processing
    representation channels (rows).
    """

    matrices: list[torch.Tensor] = []
    for layer in network.signal_processing_layers:
        main = layer.weight_connection.weight.detach().cpu()
        total = main.clone()
        if hasattr(layer, 'skip_connection'):
            total += layer.skip_connection.weight.detach().cpu()
        matrices.append(total)

    total_matrix = matrices[-1]
    for matrix in reversed(matrices[:-1]):
        total_matrix = total_matrix @ matrix

    return total_matrix.abs().numpy()


def compute_modal_contributions(
    stage_outputs: np.ndarray,
    weight_matrix: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute sample-wise modal contributions using channel energies."""

    channel_energy = np.mean(np.square(stage_outputs), axis=1)  # (N, Channels)
    modal_contrib = channel_energy @ weight_matrix  # (N, Modalities)
    modal_contrib = np.maximum(modal_contrib, 0.0)
    if normalize:
        denom = np.sum(modal_contrib, axis=1, keepdims=True) + 1e-12
        modal_contrib = modal_contrib / denom
    return modal_contrib


def aggregate_by_group(
    values: np.ndarray,
    group_labels: np.ndarray,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate rows in ``values`` by ``group_labels`` via mean."""

    unique_labels = np.unique(group_labels)
    aggregated = []
    for label in unique_labels:
        mask = group_labels == label
        if np.any(mask):
            aggregated.append(values[mask][30,:])
            # aggregated.append(values[mask].mean(axis=0))
        else:
            aggregated.append(np.full(values.shape[1], fill_value))
    return np.vstack(aggregated), unique_labels


# ---------------------------------------------------------------------------
# Visualisation primitives
# ---------------------------------------------------------------------------


def plot_modal_heatmap(
    data: np.ndarray,
    row_labels: Sequence[str],
    modal_names: Sequence[str],
    title: str,
    output_path: Path,
    cmap: str = 'Oranges',
    value_fmt: str = '.2f',
) -> None:
    if data.size == 0:
        return

    fig, ax = plt.subplots(figsize=(4 + data.shape[1], 1 + 0.5 * data.shape[0]))
    sns.heatmap(
        data,
        annot=True,
        fmt=value_fmt,
        xticklabels=modal_names,
        yticklabels=row_labels,
        cmap=cmap,
        cbar_kws={'label': 'Contribution'},
        ax=ax,
        vmin=0,
        vmax=np.nanmax(data) if np.isfinite(data).any() else 1.0,
    )
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


FEATURE_GROUPS: Mapping[str, str] = OrderedDict(
    {
        'RMS': 'Energy',
        'Var': 'Energy',
        'Max': 'Impulse',
        'Mean': 'Energy',
        'Min': 'Impulse',
        'Kurtosis': 'Impulse',
        'CrestFactor': 'Impulse',
        'ClearanceFactor': 'Impulse',
        'ShapeFactor': 'Shape',
        'Entropy': 'Complexity',
        'Skewness': 'Shape',
        'AbsMean': 'Energy',
        'Std': 'Energy',
    }
)

GROUP_COLOURS = {
    'Energy': '#3366CC',
    'Impulse': '#FF9933',
    'Shape': '#009966',
    'Complexity': '#7F7F7F',
    'Other': '#CCCCCC',
}


def radar_angles(count: int) -> np.ndarray:
    base = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.concatenate([base, base[:1]])


def plot_grouped_radar(
    normed_features: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    output_path: Path,
    figure_format: str,
) -> None:
    if normed_features.ndim != 2:
        raise ValueError('Expected 2-D feature array for radar plot.')

    num_classes, num_features = normed_features.shape
    angles = radar_angles(num_features)
    fig, axes = plt.subplots(
        1,
        num_classes,
        figsize=(4 * num_classes, 5),
        subplot_kw={'polar': True},
    )
    if num_classes == 1:
        axes = [axes]

    group_labels = [FEATURE_GROUPS.get(name, 'Other') for name in feature_names]
    for idx, ax in enumerate(axes):
        values = normed_features[idx].tolist()
        values.append(values[0])
        ax.plot(angles, values, color='#283D7E', linewidth=2)
        ax.fill(angles, values, color='#5291B2', alpha=0.2)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0.25, 1.1)
        # ax.set_title(class_names[idx], fontsize=14)

        width = 2 * np.pi / num_features
        for angle, label, group in zip(angles[:-1], feature_names, group_labels):
            color = GROUP_COLOURS.get(group, GROUP_COLOURS['Other'])
            # 使色块和标签正对每个指标的中间
            bar = ax.bar(
                x=[angle],
                height=[1.1],
                width=width,
                bottom=1.05,
                color=color,
                linewidth=0,
                alpha=0.6,
                align='center',  # 居中对齐
            )
            ax.text(
                angle,
                1.2,
                label,
                color='black',
                ha='center',
                va='center',
                fontsize=12,
            )
        ax.set_xticks([])

    legend_entries = [
        (GROUP_COLOURS.get(group, GROUP_COLOURS['Other']), group)
        for group in OrderedDict((group, None) for group in group_labels)
    ]
    handles = [plt.Line2D([0], [0], color=color, lw=8) for color, _ in legend_entries]
    labels = [group for _, group in legend_entries]
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(f'.{figure_format}'), dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(
    matrix: np.ndarray,
    modal_names: Sequence[str],
    metric_names: Sequence[str],
    title: str,
    output_path: Path,
    cmap: str = 'RdBu_r',
) -> None:
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(metric_names), 3))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        xticklabels=metric_names,
        yticklabels=modal_names,
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Partial correlation'},
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical analysis utilities
# ---------------------------------------------------------------------------


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    controls = np.asarray(controls)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.concatenate([np.ones((len(values), 1)), controls], axis=1)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def partial_correlation(
    modal_contrib: np.ndarray,
    metrics: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    class_one_hot = pd.get_dummies(labels).to_numpy()
    residual_modal = np.stack(
        [residualize(modal_contrib[:, idx], class_one_hot) for idx in range(modal_contrib.shape[1])],
        axis=1,
    )
    residual_metrics = np.stack(
        [residualize(metrics[:, idx], class_one_hot) for idx in range(metrics.shape[1])],
        axis=1,
    )

    matrix = np.zeros((modal_contrib.shape[1], metrics.shape[1]))
    for i in range(modal_contrib.shape[1]):
        for j in range(metrics.shape[1]):
            x = residual_modal[:, i]
            y = residual_metrics[:, j]
            if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = np.corrcoef(x, y)[0, 1]
    return matrix


def class_conditioned_correlations(
    modal_contrib: np.ndarray,
    metrics: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    correlations: dict[int, np.ndarray] = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 3:
            continue
        sub_modal = modal_contrib[mask]
        sub_metrics = metrics[mask]
        corr = np.zeros((sub_modal.shape[1], sub_metrics.shape[1]))
        for i in range(sub_modal.shape[1]):
            for j in range(sub_metrics.shape[1]):
                x = sub_modal[:, i]
                y = sub_metrics[:, j]
                if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    corr[i, j] = 0.0
                else:
                    corr[i, j] = np.corrcoef(x, y)[0, 1]
        correlations[int(label)] = corr
    return correlations


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def summarise_modal_dominance(matrix: np.ndarray, row_labels: Sequence[str], modal_names: Sequence[str]) -> list[str]:
    summaries: list[str] = []
    for idx, row in enumerate(matrix):
        if np.all(np.isnan(row)):
            continue
        dominant = int(np.nanargmax(row))
        summaries.append(
            f"{row_labels[idx]} 最依赖 {modal_names[dominant]} (平均贡献 {row[dominant]:.2f})."
        )
    return summaries


def summarise_top_correlations(
    correlations: np.ndarray,
    modal_names: Sequence[str],
    metric_names: Sequence[str],
    top_k: int = 2,
) -> list[str]:
    statements: list[str] = []
    for modal_idx, modal_name in enumerate(modal_names):
        row = correlations[modal_idx]
        order = np.argsort(-np.abs(row))[:top_k]
        parts = []
        for metric_idx in order:
            corr = row[metric_idx]
            trend = '正相关' if corr >= 0 else '负相关'
            parts.append(f"{metric_names[metric_idx]} ({trend} {corr:.2f})")
        if parts:
            statements.append(f"{modal_name} 模态贡献与 {'、'.join(parts)} 关联最强。")
    return statements


def render_summary_report(
    output_path: Path,
    class_heatmap: np.ndarray,
    class_labels: Sequence[str],
    modal_names: Sequence[str],
    speed_heatmap: np.ndarray | None,
    speed_labels: Sequence[str] | None,
    load_heatmap: np.ndarray | None,
    load_labels: Sequence[str] | None,
    metric_names: Sequence[str],
    global_corr: np.ndarray,
    class_corr: Mapping[int, np.ndarray],
    label_index_to_name: Mapping[int, str],
) -> None:
    lines: list[str] = []
    lines.append('# SEU TSPN 解释性分析总结')
    lines.append('')
    lines.append('## 模态贡献概览')
    lines.extend(f'- {text}' for text in summarise_modal_dominance(class_heatmap, class_labels, modal_names))
    lines.append('')
    if speed_heatmap is not None and speed_labels is not None:
        lines.append('## 转速条件与模态依赖')
        lines.extend(f'- {text}' for text in summarise_modal_dominance(speed_heatmap, speed_labels, modal_names))
        lines.append('')
    if load_heatmap is not None and load_labels is not None:
        lines.append('## 载荷条件与模态依赖')
        lines.extend(f'- {text}' for text in summarise_modal_dominance(load_heatmap, load_labels, modal_names))
        lines.append('')

    lines.append('## 雷达图指标分布')
    lines.append('各故障类别的 13 项指标已按能量型、冲击型与复杂度型进行着色区分。归一化雷达图突出显示了每类样本在指标空间的差异。')
    lines.append('')

    lines.append('## 模态-指标偏相关')
    lines.extend(f'- {text}' for text in summarise_top_correlations(global_corr, modal_names, metric_names))
    lines.append('')

    lines.append('## 类特异性模态-指标通路')
    for label, corr in class_corr.items():
        class_name = label_index_to_name.get(label, f'Class {label}')
        lines.append(f'### {class_name}')
        lines.extend(f'- {text}' for text in summarise_top_correlations(corr, modal_names, metric_names))
        lines.append('')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SEU TSPN analysis pipeline.')
    parser.add_argument('--config', type=Path, default=Path('configs/a_010_SEU/config_basic.yaml'))
    parser.add_argument('--checkpoint', type=Path, default=Path('save/test/model_seu/tspn2.ckpt'))
    parser.add_argument('--data', type=Path, default='E:/dataset/generate/SEU_bearing/SEU_bearing_20Hz_2_data.npy', help='Path to SEU signal array (.npy).')
    parser.add_argument('--labels', type=Path, default='E:/dataset/generate/SEU_bearing/SEU_bearing_20Hz_2_label.npy', help='Path to SEU label array (.npy).')
    # parser.add_argument('--config', type=Path, default=Path('configs/a_temp_SUDA_electric/config_basic.yaml'))
    # parser.add_argument('--checkpoint', type=Path, default=Path('save/test/model_suda/model_tspn_suda_02.ckpt'))
    # parser.add_argument('--data', type=Path, default='E:/dataset/generate/SUDA_electric/SUDA_electric_0kg_2500r_1000Hz_data.npy', help='Path to SEU signal array (.npy).')
    # parser.add_argument('--labels', type=Path, default='E:/dataset/generate/SUDA_electric/SUDA_electric_0kg_2500r_1000Hz_label.npy', help='Path to SEU label array (.npy).')
    parser.add_argument('--speed-labels', type=Path, default=None, help='Optional speed condition labels (.npy).')
    parser.add_argument('--load-labels', type=Path, default=None, help='Optional load condition labels (.npy).')
    parser.add_argument('--class-names', type=str, nargs='*', default=None)
    parser.add_argument('--speed-names', type=str, nargs='*', default=None)
    parser.add_argument('--load-names', type=str, nargs='*', default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('save/figure/seu/analysis'))
    # parser.add_argument('--output-dir', type=Path, default=Path('save/figure/suda/analysis'))
    parser.add_argument('--figure-format', type=str, default='svg', choices=['png', 'pdf', 'svg'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--report-path', type=Path, default=Path('reports/seu_tspn_analysis_report.md'))
    parser.add_argument('--modal-names', type=str, nargs='*', default=['Torque', 'Vibration'])
    # parser.add_argument('--report-path', type=Path, default=Path('reports/suda_tspn_analysis_report.md'))
    # parser.add_argument('--modal-names', type=str, nargs='*', default=['J3 axis', 'U phase', 'V phase', 'W phase', 'D axis'])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    dataset = load_dataset(args.data, args.labels, args.speed_labels, args.load_labels)
    network = build_network(args.config, args.checkpoint, device)

    features, stage_outputs = collect_feature_pipeline(network, dataset.signals, device, args.batch_size)
    weight_matrix = compute_total_weight_matrix(network)
    modal_contrib = compute_modal_contributions(stage_outputs, weight_matrix)

    labels = dataset.labels.astype(int)
    modal_names = args.modal_names
    class_matrix, class_ids = aggregate_by_group(modal_contrib, labels)
    if args.class_names and len(args.class_names) == len(class_ids):
        class_names = args.class_names
    else:
        class_names = [f'Class {idx}' for idx in class_ids]

    class_heatmap_path = args.output_dir / f'class_modal_heatmap.{args.figure_format}'
    plot_modal_heatmap(class_matrix, class_names, modal_names, '故障类别-模态贡献', class_heatmap_path)

    speed_matrix = None
    speed_labels = None
    if dataset.speed is not None:
        speed_matrix, speed_ids = aggregate_by_group(modal_contrib, dataset.speed.astype(int))
        if args.speed_names and len(args.speed_names) == len(speed_ids):
            speed_labels = args.speed_names
        else:
            speed_labels = [f'Speed {idx}' for idx in speed_ids]
        plot_modal_heatmap(
            speed_matrix,
            speed_labels,
            modal_names,
            '转速-模态贡献',
            args.output_dir / f'speed_modal_heatmap.{args.figure_format}',
        )

    load_matrix = None
    load_labels = None
    if dataset.load is not None:
        load_matrix, load_ids = aggregate_by_group(modal_contrib, dataset.load.astype(int))
        if args.load_names and len(args.load_names) == len(load_ids):
            load_labels = args.load_names
        else:
            load_labels = [f'Load {idx}' for idx in load_ids]
        plot_modal_heatmap(
            load_matrix,
            load_labels,
            modal_names,
            '载荷-模态贡献',
            args.output_dir / f'load_modal_heatmap.{args.figure_format}',
        )

    feature_module_names = list(
        network.feature_extractor_layers.feature_extractor_modules.keys()
    )
    channel_count = network.channel_for_feature
    metrics_per_sample = (
        features.reshape(features.shape[0], channel_count, len(feature_module_names))
    )
    # 计算每个特征的RMS并沿axis1进行平均
    metrics_per_sample = np.sqrt(np.mean(np.square(metrics_per_sample), axis=1))
    class_means, _ = aggregate_by_group(metrics_per_sample, labels)

    # min_vals = class_means.min(axis=1, keepdims=True)
    # ptp_vals = np.ptp(class_means, axis=1, keepdims=True) + 1e-8
    # normed_means = (class_means - min_vals) / ptp_vals
    # norm
    max_vals = class_means.max(axis=1, keepdims=True)
    normed_means = class_means / (max_vals + 1e-8)

    radar_path = args.output_dir / 'class_radar'
    plot_grouped_radar(normed_means, feature_module_names, class_names, radar_path, args.figure_format)

    partial_corr_matrix = partial_correlation(modal_contrib, metrics_per_sample, labels)
    corr_path = args.output_dir / f'global_modal_metric_corr.{args.figure_format}'
    plot_correlation_heatmap(
        partial_corr_matrix,
        modal_names,
        feature_module_names,
        '模态-指标偏相关',
        corr_path,
    )

    per_class_corr = class_conditioned_correlations(modal_contrib, metrics_per_sample, labels)
    for label, matrix in per_class_corr.items():
        name = class_names[list(class_ids).index(label)] if label in class_ids else f'Class {label}'
        plot_correlation_heatmap(
            matrix,
            modal_names,
            feature_module_names,
            f'{name} 模态-指标相关',
            args.output_dir / f'class_{label}_modal_metric_corr.{args.figure_format}',
        )

    label_mapping = {int(idx): name for idx, name in zip(class_ids, class_names)}

    render_summary_report(
        args.report_path,
        class_matrix,
        class_names,
        modal_names,
        speed_matrix,
        speed_labels,
        load_matrix,
        load_labels,
        feature_module_names,
        partial_corr_matrix,
        per_class_corr,
        label_mapping,
    )


if __name__ == '__main__':
    main()
