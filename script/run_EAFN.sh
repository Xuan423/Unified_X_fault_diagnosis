#!/bin/bash

# HUST motor basic: TSPN + ablations
# python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_onlyI.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_onlyentropy.yaml

# HUST motor basic: comparison methods
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_Resnet_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_Sincnet_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_MWA_CNN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_MCN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_TFN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_bsstn.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/config_Convoformer_NSE_basic.yaml

# HUST motor gen: TSPN + ablations
python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_onlyI.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_onlyentropy.yaml

# HUST motor gen: comparison methods
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_Resnet_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_Sincnet_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_MWA_CNN_gen_basic.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_MCN_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_TFN_gen_basic.yaml
python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_bsstn_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_HUST_motor/gen/config_Convoformer_NSE_gen_basic.yaml

# # SUDA electric basic: comparison methods
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/config_TFN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/config_BSSTN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/config_Convoformer_NSE_basic.yaml

# # SUDA electric gen: comparison methods
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/gen/config_TFN_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/gen/config_BSSTN_gen_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_temp_SUDA_electric/gen/config_Convoformer_NSE_gen_basic.yaml

# # SEU basic: comparison methods
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_010_SEU/config_TFN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_010_SEU/config_BSSTN_basic.yaml
# python E:/PHM_bench/Unified_X_fault_diagnosis/main_com.py --config_dir E:/PHM_bench/Unified_X_fault_diagnosis/configs/a_010_SEU/config_Convoformer_NSE_basic.yaml
