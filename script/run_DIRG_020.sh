#!/bin/bash

# DIRG_020 basic: TSPN + ablations
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_basic.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_onlyI.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_onlyentropy.yaml

# DIRG_020 basic: comparison methods
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Resnet.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Sincnet.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_MWA_CNN.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_MCN.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TFN.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Convoformer_NSE.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_bsstn.yaml

# DIRG_020 gen: TSPN + ablations
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_onlyI_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TSPN_onlyentropy_gen.yaml

# DIRG_020 gen: comparison methods
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Resnet_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Sincnet_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_MWA_CNN_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_MCN_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_TFN_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_Convoformer_NSE_gen.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_020_DIRG/config_bsstn_gen.yaml
