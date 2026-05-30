# TSPN Demo

## Overview

This repository is a minimal TSPN-only demo for transparent signal processing fault diagnosis. It keeps one training path, one real SUDA `.npy` asset pair, and one post-training analysis command.

The default command trains TSPN for 20 epochs on CPU with PyTorch Lightning and writes a benchmark-style timestamped run directory.

## Environment

Create the minimal CPU environment:

```bash
conda env create -f environment.yml
conda activate UXFD
```

The validation environment used during development was:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python
```

The default environment intentionally excludes SwanLab, W&B, TensorBoard, Gradio, CUDA-specific packages, and comparison-model dependencies.

## Data

The demo uses real SUDA assets committed under `data/demo_tspn/`:

- `SUDA_electric_0kg_2500r_1000Hz_data.npy`
- `SUDA_electric_0kg_2500r_1000Hz_label.npy`

Expected raw shapes:

- data: `(204, 1024, 5)`, raw dtype `float64`, converted to `float32` at load time
- label: `(204,)`, raw dtype `int32`, converted to `int64` at load time

The dataset split is deterministic by label order:

- train: 60%
- validation: 10%
- test: 30%

## Train

Default run:

```bash
python main.py
```

Explicit equivalent:

```bash
python main.py \
  --config configs/tspn_suda_demo.yaml \
  --device cpu \
  --epochs 20 \
  --batch-size 32 \
  --patience 20
```

The default config uses:

- model: `TSPN`
- signal layers: four layers of `HT/WF/I`
- features: 13 statistical feature extractors
- batch size: `32`
- epochs: `20`
- early stopping patience: `20`
- data workers: `0`
- pin memory: `false`
- logger: Lightning `CSVLogger`

## Outputs

Training writes to:

```text
save/task_TSPN_SUDA_DEMO/model_TSPN/model_TSPNtime.../
```

Each valid run contains:

- `logs/version_*/metrics.csv`
- best checkpoint
- `last.ckpt`
- `test_result.csv`

`save/` is ignored by git because it is generated output.

## Analysis

Run analysis after training:

```bash
python post/demo_tspn_analysis.py
```

By default, the script chooses the latest valid run under `save/task_TSPN_SUDA_DEMO/model_TSPN/`. You can also pass a run explicitly:

```bash
python post/demo_tspn_analysis.py --run-dir save/task_TSPN_SUDA_DEMO/model_TSPN/<run_dir>
```

Analysis writes:

- `reports/demo_tspn_analysis_report.md`
- `reports/demo_tspn_figures/loss_curve.png`
- `reports/demo_tspn_figures/confusion_matrix.png`
- `reports/demo_tspn_figures/filter_summary.png`
- `reports/demo_tspn_figures/feature_summary.png`
- `reports/demo_tspn_figures/prediction_summary.csv`

The report includes run summary, data summary, training metrics, test result, signal-processing weights, feature summary, prediction summary, and generated artifacts.

## Project Structure

```text
configs/config.py                         # TSPN demo YAML builder
configs/tspn_suda_demo.yaml               # Default demo config
data/                                     # Minimal demo dataset and loaders
data/demo_tspn/                           # Real SUDA demo .npy assets
model/                                    # TSPN, signal operators, feature extractors
trainer/                                  # Lightning module, trainer setup, utilities
post/demo_tspn_analysis.py                # Scripted post-training analysis
post/plot_tspn_radar.py                   # Compatibility visual-analysis CLI
post/notebooks/                           # Demo-compatible notebook launchers
reports/                                  # Demo analysis report; figures are generated
script/demo.sh                            # Train plus analysis convenience command
```

## Scope

This demo does not preserve the old multi-model benchmark harness. Comparison models, generated benchmark configs, non-TSPN scripts, external logger setup, and historical reports were removed. Local planning and cleanup notes live under ignored `docs/`.

Accuracy is reported but no accuracy threshold is required for pass/fail. The acceptance gate is that training and analysis complete with finite loss and metrics on the bundled real SUDA data.
