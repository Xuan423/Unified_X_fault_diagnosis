from ast import arg
# from os import pread
import torch
from torch import nn
from collections import OrderedDict
import yaml
from types import SimpleNamespace
import os 
import time 
import re

# from model.Signal_processing import SignalProcessingBase,\
#         SignalProcessingModuleDict,\
#         FFTSignalProcessing,\
#         HilbertTransform,\
#         WaveFilters,\
#         Identity
from model.Signal_processing import *
from model.Signal_processing_2D import *
from model.Feature_extract import FeatureExtractionBase,\
        FeatureExtractionModuleDict,\
        MeanFeature,\
        StdFeature,\
        VarFeature,\
        EntropyFeature,\
        MaxFeature,\
        MinFeature,\
        AbsMeanFeature,\
        KurtosisFeature,\
        RMSFeature,\
        CrestFactorFeature ,\
        SkewnessFeature,\
        ClearanceFactorFeature,\
        ShapeFactorFeature
from model.Logic_inference import LogicInferenceBase,\
        ImplicationOperation,\
        EquivalenceOperation,\
        NegationOperation,\
        WeakConjunctionOperation,\
        WeakDisjunctionOperation,\
        StrongConjunctionOperation,\
        StrongDisjunctionOperation
ALL_SP = {
    'FFT': FFTSignalProcessing,
    'HT': HilbertTransform,
    'WF': WaveFilters,
    'I': Identity,
    'LNO': Laplace_neural_operator,
    'RWF':RickerWaveletFilter,
    'LWF':LaplaceWaveletFilter,
    'CWF':ChirpletWaveletFilter,
    'MWF':MorletWaveletFilter,
    
    'Morlet':Morlet, # 'Morlet':Morlet,
    'Laplace':Laplace,
    'Order1MAFilter':Order1MAFilter,
    'Order2MAFilter':Order2MAFilter,
    'Order1DFFilter':Order1DFFilter,
    'Order2DFFilter':Order2DFFilter,
    'Log':LogOperation,
    'Squ':SquOperation,
    'Sin':SinOperation,
    # 2arity
    'Add':AddOperation,
    'Mul':MulOperation,
    'Div':DivOperation,
    
    # 2D
    'SBCT':SBCT_NOP,
    'GF':GlobalFilterOperator,    
    
}
ALL_FE = {
    'Mean': MeanFeature,
    'Std': StdFeature,
    'Var': VarFeature,
    'Entropy': EntropyFeature,
    'Max': MaxFeature,
    'Min': MinFeature,
    'AbsMean': AbsMeanFeature,
    'Kurtosis': KurtosisFeature,
    'RMS': RMSFeature,
    'CrestFactor': CrestFactorFeature,
    'Skewness': SkewnessFeature,
    'ClearanceFactor': ClearanceFactorFeature,
    'ShapeFactor': ShapeFactorFeature,
}

ALL_LI = {
    'imp': ImplicationOperation,
    'equ': EquivalenceOperation,
    'neg': NegationOperation,
    'conj': WeakConjunctionOperation,
    'disj': WeakDisjunctionOperation,
    'sconj': StrongConjunctionOperation,
    'sdisj': StrongDisjunctionOperation,
}

def parse_arguments(config_dir,it):
    # 解析参数
    
    # args_dir = parser.parse_args()
    # 使用参数
    yaml_dir = config_dir # args_dir.config_dir
    # 读取YAML文件
    with open(yaml_dir, 'r') as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config['args'])

    # WSL/Linux兼容：将 Windows 盘符路径(如 E:/xxx 或 E:\\xxx)映射到 /mnt/e/xxx
    if hasattr(args, "data_dir") and isinstance(args.data_dir, str):
        args.data_dir = _maybe_map_windows_drive_path(args.data_dir)

    # WSL/CPU环境兼容：当配置写了cuda但实际不可用时，自动回退到cpu
    if hasattr(args, "device") and isinstance(args.device, str):
        device_lower = args.device.lower()
        if device_lower in ("cuda", "gpu") and not torch.cuda.is_available():
            args.device = "cpu"
            if hasattr(args, "gpus"):
                args.gpus = 1
    
    
    # dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%d-%H-%M-%S", time.localtime())
    name = f'model_{args.model}time{time_stamp}_dataset{args.dataset_task}_it{it}'

    print(f'Running experiment: {name}')
    
    # if args.debug != 'True':
    path = 'save/' + f'task_{args.dataset_task}/'+f'model_{args.model}/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    args.path = path
    return config,args,path,name


def _maybe_map_windows_drive_path(path: str) -> str:
    """
    在非Windows系统上，将形如 'E:/dataset/...' 或 'E:\\dataset\\...' 的路径
    尝试映射为 '/mnt/e/dataset/...'（仅当映射路径存在时才替换）。
    """
    if os.name == "nt":
        return path

    if not path:
        return path

    # 已经是Linux路径 /mnt/* 则不处理
    if path.startswith("/mnt/"):
        return path

    # 如果原路径本身存在，也不处理（比如用户自己挂载了同名目录）
    if os.path.exists(path):
        return path

    match = re.match(r"^([A-Za-z]):[\\\\/](.*)$", path)
    if not match:
        return path

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    mapped = f"/mnt/{drive}/{rest}"

    # 保持末尾的 / 语义（主要用于拼接文件名）
    if path.endswith(("/", "\\")) and not mapped.endswith("/"):
        mapped += "/"

    return mapped if os.path.exists(mapped) else path
#### 暂时无用，和parse功能一样 ################
def yaml_arguments(yaml_dir): # 暂时无用
    # 读取YAML文件
    with open(yaml_dir, 'r') as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config['args'])

    
    # dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    name = f'post_time{time_stamp}_lr{args.learning_rate}_epochs{args.num_epochs}_scale{args.scale}_l1norm{args.l1_norm}_dataset{args.dataset_task}_seed{args.seed}'

    print(f'Running experiment: {name}')
    path = 'save/' + f'model_{args.model}/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    
    return config,args,path
####################

def config_network(config,args):
    """
    input: config,args
    putput: signal_processing_modules,feature_extractor_modules
    function: 从配置文件中构建信号处理模块和特征提取模块。
    """
    
    signal_processing_modules = []
    for layer in config['signal_processing_configs'].values():
        signal_module = OrderedDict()
        for module_name in layer:
            
            module_class = ALL_SP[module_name]
            
            module_name = get_unique_module_name(signal_module.keys(), module_name)
            signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
        signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

    feature_extractor_modules = OrderedDict()
    for feature_name in config['feature_extractor_configs']:
        module_class = ALL_FE[feature_name]
        feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
    
    # TODO logic
    
    return signal_processing_modules,feature_extractor_modules

def get_unique_module_name(existing_names, module_name):
    """
    根据已存在的模块名列表，为新模块生成一个唯一的名称。
    
    :param existing_names: 已存在的模块名称的集合或列表。
    :param module_name: 要检查的模块名称。
    :return: 唯一的模块名称。
    """
    if module_name not in existing_names:
        # 如果模块名不存在，则直接返回
        return module_name
    else:
        # 如果模块名已存在，尝试添加序号直到找到一个唯一的名字
        index = 1
        unique_name = f"{module_name}_{index}"
        while unique_name in existing_names:
            index += 1
            unique_name = f"{module_name}_{index}"
        return unique_name
# logic
if __name__ == '__main__':
    pass
    # 使用示例
    # config_dir = 'configs/THU_006/config_TSPN.yaml'
    # model, test_data, test_labels, predictions = load_and_predict(config_dir,best_model_path='save/THU1+THU2剪枝/THU1_new/model-epoch=61-val_loss=0.0545.ckpt')
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import OrderedDict


    import yaml
    from types import SimpleNamespace

    config_dir = 'configs/config_basic.yaml' # 从工作目录开始的相对路径
    # 读取YAML文件
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config['args'])



    signal_processing_modules = []
    for layer in config['signal_processing_configs'].values():
        signal_module = OrderedDict()
        for module_name in layer:
            module_class = ALL_SP[module_name]
            signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
        signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

    feature_extractor_modules = OrderedDict()
    for feature_name in config['feature_extractor_configs']:
        module_class = ALL_FE[feature_name]
        feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
