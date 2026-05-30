#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_DIR="configs/b_pu"
TMP_CONFIG_DIR="configs/.generated_loo/b_pu"
DATASET_TASK="b_pu_generalization"
DOMAINS=(
  "PU_N09_M07_F10"
  "PU_N15_M01_F10"
  "PU_N15_M07_F04"
  "PU_N15_M07_F10"
)
CONFIGS=(
  "config_bsstn_gen.yaml"
  "config_convformernse_gen.yaml"
  "config_mwacnn_gen.yaml"
  "config_resnet_gen.yaml"
  "config_tfn_gen.yaml"
)
MODELS=(
  "BSSTN_Flex"
  "Convoformer_NSE"
  "Huan_net"
  "Resnet"
  "TFN_Morlet"
)

mkdir -p "$TMP_CONFIG_DIR" reports

"$PYTHON_BIN" - "$CONFIG_DIR" "$TMP_CONFIG_DIR" "${DOMAINS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
domains = sys.argv[3:]
source_list = [[domain for domain in domains if domain != target] for target in domains]
target_list = [[target] for target in domains]

output_dir.mkdir(parents=True, exist_ok=True)
for config_path in sorted(config_dir.glob("config_*_gen.yaml")):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    args = config.setdefault("args", {})
    args["source"] = source_list[0]
    args["target"] = target_list[0]
    args["source_list"] = source_list
    args["target_list"] = target_list
    output_path = output_dir / config_path.name
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"generated {output_path}")
PY

for config_name in "${CONFIGS[@]}"; do
  config_path="$TMP_CONFIG_DIR/$config_name"
  echo "Running $DATASET_TASK leave-one-out: $config_path"
  "$PYTHON_BIN" main_com.py --config_dir "$config_path"
done

"$PYTHON_BIN" script/collect_loo_acc.py \
  --save-root save \
  --dataset-task "$DATASET_TASK" \
  --output-prefix reports/b_pu_loo \
  --models "${MODELS[@]}" \
  --targets "${DOMAINS[@]}"
