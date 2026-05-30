from collections import OrderedDict
import os
import re
import time
from types import SimpleNamespace

import torch
import yaml

from model.Feature_extract import (
    AbsMeanFeature,
    ClearanceFactorFeature,
    CrestFactorFeature,
    EntropyFeature,
    FeatureExtractionModuleDict,
    KurtosisFeature,
    MaxFeature,
    MeanFeature,
    MinFeature,
    RMSFeature,
    ShapeFactorFeature,
    SkewnessFeature,
    StdFeature,
    VarFeature,
)
from model.Signal_processing import HilbertTransform, Identity, SignalProcessingModuleDict, WaveFilters


ALL_SP = {
    "HT": HilbertTransform,
    "WF": WaveFilters,
    "I": Identity,
}

ALL_FE = {
    "Mean": MeanFeature,
    "Std": StdFeature,
    "Var": VarFeature,
    "Entropy": EntropyFeature,
    "Max": MaxFeature,
    "Min": MinFeature,
    "AbsMean": AbsMeanFeature,
    "Kurtosis": KurtosisFeature,
    "RMS": RMSFeature,
    "CrestFactor": CrestFactorFeature,
    "Skewness": SkewnessFeature,
    "ClearanceFactor": ClearanceFactorFeature,
    "ShapeFactor": ShapeFactorFeature,
}


def parse_arguments(
    config_dir,
    it,
    run_tag=None,
    target_override=None,
    source_override=None,
    create_path=True,
    verbose=True,
):
    with open(config_dir, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    args = SimpleNamespace(**config["args"])
    if target_override is not None:
        args.target = target_override
    if source_override is not None:
        args.source = source_override

    if hasattr(args, "data_dir") and isinstance(args.data_dir, str):
        args.data_dir = _maybe_map_windows_drive_path(args.data_dir)

    if hasattr(args, "device") and isinstance(args.device, str):
        device_lower = args.device.lower()
        if device_lower in ("cuda", "gpu") and not torch.cuda.is_available():
            args.device = "cpu"
            if hasattr(args, "gpus"):
                args.gpus = 1

    time_stamp = time.strftime("%d-%H-%M-%S", time.localtime())
    tag_suffix = f"_{run_tag}" if run_tag else ""
    name = f"model_{args.model}time{time_stamp}_dataset{args.dataset_task}_it{it}{tag_suffix}"
    if verbose:
        print(f"Running experiment: {name}")

    path = "save/" + f"task_{args.dataset_task}/" + f"model_{args.model}/" + name
    if create_path and not os.path.exists(path):
        os.makedirs(path)
    args.path = path
    return config, args, path, name


def _maybe_map_windows_drive_path(path: str) -> str:
    if os.name == "nt" or not path or path.startswith("/mnt/") or os.path.exists(path):
        return path

    match = re.match(r"^([A-Za-z]):[\\\\/](.*)$", path)
    if not match:
        return path

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    mapped = f"/mnt/{drive}/{rest}"
    if path.endswith(("/", "\\")) and not mapped.endswith("/"):
        mapped += "/"
    return mapped if os.path.exists(mapped) else path


def yaml_arguments(yaml_dir):
    config, args, path, _ = parse_arguments(yaml_dir, 0)
    return config, args, path


def config_network(config, args):
    signal_processing_modules = []
    for layer in config["signal_processing_configs"].values():
        signal_module = OrderedDict()
        for module_name in layer:
            if module_name not in ALL_SP:
                raise KeyError(f"Unsupported signal processing module for TSPN demo: {module_name}")
            module_class = ALL_SP[module_name]
            unique_name = get_unique_module_name(signal_module.keys(), module_name)
            signal_module[unique_name] = module_class(args)
        signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

    feature_extractor_modules = OrderedDict()
    for feature_name in config["feature_extractor_configs"]:
        if feature_name not in ALL_FE:
            raise KeyError(f"Unsupported feature extractor for TSPN demo: {feature_name}")
        feature_extractor_modules[feature_name] = ALL_FE[feature_name]()

    return signal_processing_modules, feature_extractor_modules


def get_unique_module_name(existing_names, module_name):
    if module_name not in existing_names:
        return module_name
    index = 1
    unique_name = f"{module_name}_{index}"
    while unique_name in existing_names:
        index += 1
        unique_name = f"{module_name}_{index}"
    return unique_name
