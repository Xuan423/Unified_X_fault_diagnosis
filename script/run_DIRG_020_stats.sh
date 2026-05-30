#!/bin/bash
set -e

# Model stats for DIRG_020 (TSPN + comparison methods)
python script/benchmark_model_stats.py \
  --configs \
    configs/a_020_DIRG/config_TSPN_basic.yaml \
    configs/a_020_DIRG/config_TSPN_onlyI.yaml \
    configs/a_020_DIRG/config_TSPN_onlyentropy.yaml \
    configs/a_020_DIRG/config_Resnet.yaml \
    configs/a_020_DIRG/config_Sincnet.yaml \
    configs/a_020_DIRG/config_MWA_CNN.yaml \
    configs/a_020_DIRG/config_MCN.yaml \
    configs/a_020_DIRG/config_TFN.yaml \
    configs/a_020_DIRG/config_Convoformer_NSE.yaml \
    configs/a_020_DIRG/config_bsstn.yaml \
  --output reports/dirg_020_model_stats.csv \
  --latency-batch-sizes 64 
