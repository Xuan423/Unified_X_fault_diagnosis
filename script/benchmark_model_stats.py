#!/usr/bin/env python3
import argparse
import csv
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from torchinfo import summary as torchinfo_summary
except Exception:  # pragma: no cover - optional dependency
    torchinfo_summary = None

from configs.config import config_network, parse_arguments


TSPN_FAMILY = {"TSPN", "TKAN", "NNSPN", "TFON"}
IDENTITY_TOKENS = {"I", "IDENTITY"}


def _is_identity(name: str) -> bool:
    return str(name).upper() in IDENTITY_TOKENS


def _get_flag(cfg_args: Any, *names: str, default: Optional[bool] = None) -> Optional[bool]:
    for name in names:
        if hasattr(cfg_args, name):
            value = getattr(cfg_args, name)
            if value is not None:
                return bool(value)
    return default


def _format_na(value: Optional[float]) -> str:
    return "N/A" if value is None else str(value)


def _safe_json(value: Optional[Any]) -> str:
    if value is None:
        return "N/A"
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _find_data_label_paths(data_dir: str, target: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not data_dir or not target:
        return None, None, None
    candidates = [
        (f"{target}_data.npy", f"{target}_label.npy", "target_suffix"),
        (f"data_{target}.npy", f"label_{target}.npy", "data_prefix"),
    ]
    for data_name, label_name, tag in candidates:
        data_path = os.path.join(data_dir, data_name)
        label_path = os.path.join(data_dir, label_name)
        if os.path.exists(data_path) and os.path.exists(label_path):
            return data_path, label_path, tag
    return None, None, None


class NumpyCarrierDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def _load_dataset_with_fallback(
    cfg_args: Any, shuffle: bool, seed: Optional[int] = None
) -> Tuple[Optional[torch.utils.data.Dataset], List[str]]:
    notes: List[str] = []
    try:
        from data.data_provider import DATASET_TASK_CLASS
    except Exception as exc:
        notes.append(f"dataset_import_error={type(exc).__name__}")
        return None, notes

    dataset_class = DATASET_TASK_CLASS.get(getattr(cfg_args, "dataset_task", ""), None)
    if dataset_class is None:
        notes.append("dataset_task_not_supported")
        return None, notes

    dataset = None
    try:
        dataset = dataset_class(cfg_args, flag="train")
    except Exception as exc:
        notes.append(f"dataset_load_error={type(exc).__name__}")
        data_dir = getattr(cfg_args, "data_dir", "")
        target = getattr(cfg_args, "target", "")
        data_path, label_path, tag = _find_data_label_paths(data_dir, target)
        if data_path and label_path:
            try:
                data = np.load(data_path)
                labels = np.load(label_path)
                dataset = NumpyCarrierDataset(data, labels)
                notes.append(f"dataset_fallback={tag}")
            except Exception as exc2:
                notes.append(f"dataset_fallback_error={type(exc2).__name__}")
                return None, notes
        else:
            notes.append("dataset_fallback_missing")
            return None, notes

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    return dataset, notes


def resolve_device(args: Any, override: str) -> torch.device:
    if override and override != "auto":
        device = torch.device(override)
    else:
        device_str = getattr(args, "device", None)
        if device_str:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def build_model(configs: Dict[str, Any], args: Any) -> torch.nn.Module:
    if args.model in TSPN_FAMILY:
        from model.TSPN import Transparent_Signal_Processing_Network
        from model.TSPN_KAN import Transparent_Signal_Processing_KAN
        from model.NNSPN import NN_Signal_Processing_Network
        from model.TFON import Time_Frequency_Operator_Network

        signal_processing_modules, feature_extractor_modules = config_network(configs, args)
        model_dict = {
            "TSPN": Transparent_Signal_Processing_Network,
            "TKAN": Transparent_Signal_Processing_KAN,
            "NNSPN": NN_Signal_Processing_Network,
            "TFON": Time_Frequency_Operator_Network,
        }
        return model_dict[args.model](
            signal_processing_modules, feature_extractor_modules, args
        )

    if args.model == "Resnet":
        from model_collection.Resnet import BasicBlock, ResNet

        return ResNet(
            BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes
        )
    if args.model == "WKN_m":
        from model_collection.Resnet import BasicBlock
        from model_collection.WKN import WKN_m

        return WKN_m(
            BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes
        )
    if args.model == "Sinc_net_m":
        from model_collection.Resnet import BasicBlock
        from model_collection.Sincnet import Sinc_net_m

        return Sinc_net_m(
            BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes
        )
    if args.model == "Huan_net":
        from model_collection.MWA_CNN import Huan_net

        return Huan_net(input_size=args.in_channels, num_class=args.num_classes)
    if args.model == "TFN_Morlet":
        from model_collection.TFN.Models.TFN import TFN_Morlet

        return TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes)
    if args.model == "MCN_GFK":
        from model_collection.MCN.models import MultiChannel_MCN_GFK

        ff = np.arange(0, args.in_dim // 2 + 1) / args.in_dim // 2 + 1
        return MultiChannel_MCN_GFK(
            ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes
        )
    if args.model == "Convoformer_NSE":
        from model_collection.Convformer_NSE import convoformer_v1_small

        return convoformer_v1_small(
            in_channel=args.in_channels, out_channel=args.num_classes
        )
    if args.model == "BSSTN_Flex":
        from model_collection.bsstn_flex import BSSTNFlex

        return BSSTNFlex(num_classes=args.num_classes, max_sensors=args.in_channels)

    raise ValueError(f"Unsupported model: {args.model}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_macs(
    model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device
) -> Optional[float]:
    if torchinfo_summary is None:
        return None
    try:
        info = torchinfo_summary(
            model, input_data=input_tensor, device=device, verbose=0
        )
    except Exception:
        return None
    return info.total_mult_adds


def _pool_tensor(output: Any) -> Optional[torch.Tensor]:
    if not isinstance(output, torch.Tensor):
        return None
    tensor = output
    if torch.is_complex(tensor):
        tensor = tensor.abs()
    else:
        tensor = tensor.abs()
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1)


def _pool_channel_tensor(output: Any) -> Optional[torch.Tensor]:
    if not isinstance(output, torch.Tensor):
        return None
    tensor = output
    if torch.is_complex(tensor):
        tensor = tensor.abs()
    else:
        tensor = tensor.abs()
    if tensor.ndim < 2:
        return None
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    return tensor.mean(dim=2)


@dataclass
class CarrierRecord:
    name: str
    tensor: torch.Tensor
    mode: str


class CarrierCollector:
    def __init__(self) -> None:
        self.records: List[CarrierRecord] = []

    def add(self, name: str, tensor: Optional[torch.Tensor], mode: str) -> None:
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor):
            return
        self.records.append(CarrierRecord(name=name, tensor=tensor, mode=mode))

    def aggregate(self) -> List[CarrierRecord]:
        return self.records


def _make_raw_hook(name: str, collector: CarrierCollector, mode: str):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            collector.add(name, output, mode)
    return hook


def _make_channel_hook(base_name: str, collector: CarrierCollector):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            collector.add(base_name, output, "channel_first")
    return hook


def _make_mcn_hook(
    channel_idx: Optional[int], per_channel: bool, collector: CarrierCollector
):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            prefix = f"ch{channel_idx}" if per_channel and channel_idx is not None else "shared"
            collector.add(f"MFK:{prefix}", output, "channel_first")
    return hook


def _make_dwt_hook(stage_name: str, collector: CarrierCollector):
    def hook(_module, _inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            return
        low_band, high_bands = output
        collector.add(f"{stage_name}:low", low_band, "scalar")
        if isinstance(high_bands, (tuple, list)):
            for idx, band in enumerate(high_bands):
                collector.add(f"{stage_name}:high{idx}", band, "scalar")
    return hook


def _dwt_stage_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name in dir(model):
        if name.startswith("DWT"):
            names.append(name)
    def _key(n: str) -> Tuple[int, str]:
        suffix = n[3:]
        return (int(suffix) if suffix.isdigit() else 10**9, n)
    return sorted(names, key=_key)


def _count_mwa_subbands(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> Tuple[int, int, List[str]]:
    notes: List[str] = []

    def _run_count(local_model: torch.nn.Module, local_input: torch.Tensor) -> Tuple[int, int]:
        was_training = local_model.training
        local_model.eval()
        collector = CarrierCollector()
        handles = []
        for stage_name in _dwt_stage_names(local_model):
            module = getattr(local_model, stage_name, None)
            if module is not None:
                handles.append(module.register_forward_hook(_make_dwt_hook(stage_name, collector)))
        try:
            with torch.no_grad():
                _ = local_model(local_input)
        finally:
            if was_training:
                local_model.train()
            for handle in handles:
                handle.remove()
        carriers = collector.aggregate()
        stage_names = {record.name.split(":")[0] for record in carriers}
        return len(stage_names), len(carriers)

    try:
        return (*_run_count(model, input_tensor), notes)
    except Exception as exc:
        notes.append(f"dwt_count_primary_error={type(exc).__name__}:{exc}")

    try:
        cpu_model = copy.deepcopy(model).to("cpu")
        cpu_input = input_tensor.detach().to("cpu")
        stage_count, subband_count = _run_count(cpu_model, cpu_input)
        notes.append("dwt_count_fallback=cpu")
        return stage_count, subband_count, notes
    except Exception as exc:
        notes.append(f"dwt_count_fallback_error={type(exc).__name__}:{exc}")
        return None, None, notes


def _compute_icu_struct(
    configs: Dict[str, Any],
    cfg_args: Any,
    model: Optional[torch.nn.Module],
    input_tensor: torch.Tensor,
) -> Dict[str, Any]:
    model_name = getattr(cfg_args, "model", "")
    notes: List[str] = []
    capacity: Dict[str, Any] = {}

    if model_name in TSPN_FAMILY:
        sp_layers = configs.get("signal_processing_configs", {}) or {}
        fe_list = configs.get("feature_extractor_configs", []) or []

        non_identity = 0
        identity = 0
        total_sp = 0
        for layer in sp_layers.values():
            for module_name in layer:
                total_sp += 1
                if _is_identity(module_name):
                    identity += 1
                else:
                    non_identity += 1

        unique_features: List[str] = []
        for name in fe_list:
            if name not in unique_features:
                unique_features.append(name)

        icu_struct = non_identity + len(unique_features)
        capacity = {
            "out_channels": getattr(cfg_args, "out_channels", None),
            "scale": getattr(cfg_args, "scale", None),
            "num_layers": len(sp_layers),
            "sp_instances": total_sp,
            "sp_non_identity": non_identity,
            "feature_types_unique": len(unique_features),
        }
        if identity > 0:
            notes.append(f"identity_instances={identity}")
        return {
            "icu_struct": icu_struct,
            "capacity": capacity,
            "notes": notes,
            "model_type": "TSPN",
        }

    if model is None:
        return {
            "icu_struct": None,
            "capacity": None,
            "notes": ["no_model_instance"],
            "model_type": "None",
        }

    if model_name == "MCN_GFK":
        features = getattr(model, "features", None)
        num_mfks = None
        if isinstance(features, torch.nn.ModuleList) and len(features) > 0:
            num_mfks = getattr(features[0], "num", None)
        else:
            num_mfks = getattr(features, "num", None)
        per_channel_flag = _get_flag(cfg_args, "icu_mcn_per_channel", "icu_per_channel")
        per_channel = per_channel_flag
        if per_channel is None:
            per_channel = isinstance(features, torch.nn.ModuleList) and len(features) > 1
        in_channels = getattr(cfg_args, "in_channels", None)
        if num_mfks is None:
            notes.append("num_mfks_unavailable")
            return {"icu_struct": None, "capacity": None, "notes": notes, "model_type": "MCN"}
        icu_struct = int(num_mfks * in_channels) if per_channel else int(num_mfks)
        capacity = {
            "num_mfks": int(num_mfks),
            "in_channels": in_channels,
            "per_channel": per_channel,
        }
        if per_channel_flag is not None:
            notes.append(f"per_channel_flag={per_channel_flag}")
        return {"icu_struct": icu_struct, "capacity": capacity, "notes": notes, "model_type": "MCN"}

    funconv = getattr(model, "funconv", None)
    if funconv is not None and funconv.__class__.__name__.startswith("TFconv"):
        out_channels = getattr(funconv, "out_channels", None)
        in_channels = getattr(funconv, "in_channels", None)
        per_channel_flag = _get_flag(cfg_args, "icu_tfn_per_channel", "icu_per_channel")
        per_channel = bool(per_channel_flag) if per_channel_flag is not None else False
        if out_channels is None:
            notes.append("tfconv_out_channels_unavailable")
            return {"icu_struct": None, "capacity": None, "notes": notes, "model_type": "TFN"}
        icu_struct = int(out_channels * (in_channels if per_channel else 1))
        capacity = {
            "out_channels": out_channels,
            "in_channels": in_channels,
            "per_channel": per_channel,
        }
        if per_channel_flag is not None:
            notes.append(f"per_channel_flag={per_channel_flag}")
        return {"icu_struct": icu_struct, "capacity": capacity, "notes": notes, "model_type": "TFN"}

    conv1 = getattr(model, "conv1", None)
    if conv1 is not None and conv1.__class__.__name__.startswith("SincConv"):
        out_channels = getattr(conv1, "out_channels", None)
        in_channels = getattr(conv1, "in_channels", None)
        per_channel_flag = _get_flag(cfg_args, "icu_sinc_per_channel", "icu_per_channel")
        if per_channel_flag is None:
            per_channel = conv1.__class__.__name__ == "SincConv_multiple_channel"
        else:
            per_channel = bool(per_channel_flag)
        if out_channels is None:
            notes.append("sinc_out_channels_unavailable")
            return {"icu_struct": None, "capacity": None, "notes": notes, "model_type": "SincNet"}
        icu_struct = int(out_channels * (in_channels if per_channel else 1))
        capacity = {
            "out_channels": out_channels,
            "in_channels": in_channels,
            "per_channel": per_channel,
            "conv_type": conv1.__class__.__name__,
        }
        if per_channel_flag is not None:
            notes.append(f"per_channel_flag={per_channel_flag}")
        return {"icu_struct": icu_struct, "capacity": capacity, "notes": notes, "model_type": "SincNet"}

    if model_name == "Huan_net":
        try:
            stage_count, subband_count, mwa_notes = _count_mwa_subbands(model, input_tensor)
            notes.extend(mwa_notes)
        except Exception as exc:
            notes.append(f"dwt_count_error={type(exc).__name__}:{exc}")
            return {"icu_struct": None, "capacity": None, "notes": notes, "model_type": "MWA"}
        if stage_count is None or subband_count is None:
            notes.append("dwt_count_unavailable")
            return {"icu_struct": None, "capacity": None, "notes": notes, "model_type": "MWA"}
        numf = None
        try:
            if hasattr(model, "SConv1") and hasattr(model.SConv1, "conv"):
                numf = getattr(model.SConv1.conv[0], "out_channels", None)
        except Exception:
            numf = None
        icu_struct = int(subband_count)
        capacity = {
            "dwt_stages": stage_count,
            "subband_count": subband_count,
            "numf": numf,
        }
        return {"icu_struct": icu_struct, "capacity": capacity, "notes": notes, "model_type": "MWA"}

    return {
        "icu_struct": None,
        "capacity": None,
        "notes": ["no_declared_carriers"],
        "model_type": "None",
    }


def _register_carrier_hooks(
    model: torch.nn.Module,
    cfg_args: Any,
    model_type: str,
    collector: CarrierCollector,
) -> Tuple[List[Any], List[str]]:
    handles: List[Any] = []
    notes: List[str] = []

    if model_type == "TSPN":
        for layer_idx, layer in enumerate(getattr(model, "signal_processing_layers", [])):
            module_dict = getattr(layer, "signal_processing_modules", None)
            if module_dict is None:
                continue
            for key, module in module_dict.items():
                module_name = getattr(module, "name", key)
                if _is_identity(module_name):
                    continue
                carrier_name = f"SP:L{layer_idx}:{key}"
                handles.append(module.register_forward_hook(_make_raw_hook(carrier_name, collector, "channel_last")))
        fe_layer = getattr(model, "feature_extractor_layers", None)
        fe_modules = getattr(fe_layer, "feature_extractor_modules", None)
        if fe_modules:
            for key, module in fe_modules.items():
                feature_name = getattr(module, "name", key)
                carrier_name = f"FE:{feature_name}"
                handles.append(module.register_forward_hook(_make_raw_hook(carrier_name, collector, "channel_first")))
        return handles, notes

    if model_type == "MCN":
        features = getattr(model, "features", None)
        per_channel_flag = _get_flag(cfg_args, "icu_mcn_per_channel", "icu_per_channel")
        per_channel = per_channel_flag
        if per_channel is None:
            per_channel = isinstance(features, torch.nn.ModuleList) and len(features) > 1
        if isinstance(features, torch.nn.ModuleList):
            for idx, module in enumerate(features):
                handles.append(module.register_forward_hook(_make_mcn_hook(idx, per_channel, collector)))
        elif features is not None:
            handles.append(features.register_forward_hook(_make_mcn_hook(None, per_channel, collector)))
        return handles, notes

    if model_type == "TFN":
        per_channel = _get_flag(cfg_args, "icu_tfn_per_channel", "icu_per_channel", default=False)
        if per_channel:
            notes.append("per_channel_tfconv_not_separable")
            return handles, notes
        funconv = getattr(model, "funconv", None)
        if funconv is not None:
            handles.append(funconv.register_forward_hook(_make_raw_hook("TFconv", collector, "channel_first")))
        return handles, notes

    if model_type == "SincNet":
        per_channel = _get_flag(cfg_args, "icu_sinc_per_channel", "icu_per_channel", default=None)
        conv1 = getattr(model, "conv1", None)
        if per_channel is None and conv1 is not None:
            per_channel = conv1.__class__.__name__ == "SincConv_multiple_channel"
        if per_channel:
            notes.append("per_channel_sinc_not_separable")
            return handles, notes
        if conv1 is not None:
            handles.append(conv1.register_forward_hook(_make_raw_hook("Sinc", collector, "channel_first")))
        return handles, notes

    if model_type == "MWA":
        for stage_name in _dwt_stage_names(model):
            module = getattr(model, stage_name, None)
            if module is not None:
                handles.append(module.register_forward_hook(_make_dwt_hook(stage_name, collector)))
        return handles, notes

    notes.append("no_carrier_hooks")
    return handles, notes


def _compute_icu_eff(
    model: torch.nn.Module,
    cfg_args: Any,
    model_type: str,
    device: torch.device,
    icu_min_samples: int,
    icu_max_samples: int,
    icu_quantile: float,
    icu_topk: int,
) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, float]], List[str]]:
    notes: List[str] = []
    if model_type in ("None",):
        return None, None, None, ["no_declared_carriers"]

    if model_type == "SincNet":
        per_channel = _get_flag(cfg_args, "icu_sinc_per_channel", "icu_per_channel")
        if per_channel is None:
            conv1 = getattr(model, "conv1", None)
            if conv1 is not None and conv1.__class__.__name__ == "SincConv_multiple_channel":
                per_channel = True
        if per_channel:
            return None, None, None, ["per_channel_sinc_not_separable"]

    carrier_source = str(getattr(cfg_args, "icu_carrier_source", "activation")).lower()
    use_param_carriers = carrier_source in ("param", "parameter", "parameters")
    notes.append(f"carrier_source={carrier_source}")

    batch_size = getattr(cfg_args, "batch_size", 1)
    if batch_size <= 0:
        batch_size = 1

    seed = getattr(cfg_args, "seed", None)
    dataset, dataset_notes = _load_dataset_with_fallback(cfg_args, shuffle=True, seed=seed)
    notes.extend(dataset_notes)
    if dataset is None:
        return None, None, None, notes

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    total_samples = 0
    carrier_sums: Dict[str, float] = {}
    carrier_counts: Dict[str, int] = {}

    model.eval()
    with torch.enable_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch
            if not isinstance(x, torch.Tensor):
                continue
            x = x.to(device)
            x = x.detach().requires_grad_(True)
            batch_size = x.shape[0]

            collector = CarrierCollector()
            handles: List[Any] = []
            if not use_param_carriers:
                handles, hook_notes = _register_carrier_hooks(model, cfg_args, model_type, collector)
                notes.extend(hook_notes)

            model.zero_grad(set_to_none=True)
            try:
                logits = model(x)
            except Exception as exc:
                for handle in handles:
                    handle.remove()
                notes.append(f"forward_error={type(exc).__name__}")
                continue
            if logits.ndim != 2:
                for handle in handles:
                    handle.remove()
                notes.append("logits_shape_unexpected")
                continue
            pred = logits.argmax(dim=1)
            s = logits.gather(1, pred.view(-1, 1)).squeeze(1)
            s_sum = s.sum()

            for handle in handles:
                handle.remove()

            batch_used = False
            if not use_param_carriers:
                carriers = collector.aggregate()
                if not carriers:
                    notes.append("no_carriers_captured")
                    continue
                carrier_names = []
                carrier_modes = []
                carrier_tensors = []
                no_grad_count = 0
                for record in carriers:
                    tensor = record.tensor
                    if not tensor.requires_grad:
                        no_grad_count += 1
                        continue
                    carrier_names.append(record.name)
                    carrier_modes.append(record.mode)
                    carrier_tensors.append(tensor)

                if no_grad_count:
                    notes.append(f"no_grad_carriers={no_grad_count}")
                if not carrier_tensors:
                    notes.append("no_grad_carriers_all")
                    continue

                try:
                    grads = torch.autograd.grad(
                        s_sum, carrier_tensors, retain_graph=False, allow_unused=True
                    )
                except Exception as exc:
                    notes.append(f"grad_error={type(exc).__name__}:{exc}")
                    continue

                for name, mode, tensor, grad_i in zip(
                    carrier_names, carrier_modes, carrier_tensors, grads
                ):
                    if grad_i is None:
                        continue
                    u = torch.abs(tensor * grad_i)
                    if mode == "channel_last":
                        if u.ndim < 2:
                            continue
                        reduce_dims = tuple(range(1, u.ndim - 1))
                        u = u.mean(dim=reduce_dims)
                        for idx in range(u.shape[1]):
                            u_vals = u[:, idx]
                            mask = torch.isfinite(u_vals)
                            if not mask.any():
                                continue
                            u_vals = u_vals[mask]
                            name_key = f"{name}:ch{idx}"
                            carrier_sums[name_key] = carrier_sums.get(name_key, 0.0) + float(u_vals.sum().item())
                            carrier_counts[name_key] = carrier_counts.get(name_key, 0) + int(u_vals.numel())
                            batch_used = True
                    elif mode == "channel_first":
                        if u.ndim < 2:
                            continue
                        reduce_dims = tuple(range(2, u.ndim))
                        u = u.mean(dim=reduce_dims)
                        for idx in range(u.shape[1]):
                            u_vals = u[:, idx]
                            mask = torch.isfinite(u_vals)
                            if not mask.any():
                                continue
                            u_vals = u_vals[mask]
                            name_key = f"{name}:ch{idx}"
                            carrier_sums[name_key] = carrier_sums.get(name_key, 0.0) + float(u_vals.sum().item())
                            carrier_counts[name_key] = carrier_counts.get(name_key, 0) + int(u_vals.numel())
                            batch_used = True
                    else:
                        u = u.reshape(u.shape[0], -1).mean(dim=1)
                        mask = torch.isfinite(u)
                        if not mask.any():
                            continue
                        u = u[mask]
                        carrier_sums[name] = carrier_sums.get(name, 0.0) + float(u.sum().item())
                        carrier_counts[name] = carrier_counts.get(name, 0) + int(u.numel())
                        batch_used = True
            else:
                try:
                    if model_type == "MCN":
                        features = getattr(model, "features", None)
                        per_channel_flag = _get_flag(cfg_args, "icu_mcn_per_channel", "icu_per_channel")
                        per_channel = per_channel_flag
                        if per_channel is None:
                            per_channel = isinstance(features, torch.nn.ModuleList) and len(features) > 1

                        modules = (
                            list(features)
                            if isinstance(features, torch.nn.ModuleList)
                            else [features] if features is not None else []
                        )
                        if not modules:
                            notes.append("no_param_carriers")
                            continue
                        param_tensors: List[torch.Tensor] = []
                        for module in modules:
                            param_tensors.extend([module.fc, module.sigma])
                        grads = torch.autograd.grad(
                            s_sum, param_tensors, retain_graph=False, allow_unused=True
                        )
                        for idx, module in enumerate(modules):
                            grad_fc = grads[idx * 2]
                            grad_sigma = grads[idx * 2 + 1]
                            fc = module.fc
                            sigma = module.sigma
                            num = int(fc.shape[0])
                            prefix = f"ch{idx}" if per_channel else "shared"
                            for k in range(num):
                                vals = []
                                if grad_fc is not None:
                                    vals.append(torch.abs(fc[k] * grad_fc[k]).mean())
                                if grad_sigma is not None:
                                    vals.append(torch.abs(sigma[k] * grad_sigma[k]).mean())
                                if not vals:
                                    continue
                                u_val = torch.stack(vals).mean() / max(batch_size, 1)
                                name = f"MFK:{prefix}:{k}"
                                carrier_sums[name] = carrier_sums.get(name, 0.0) + float(u_val.item()) * batch_size
                                carrier_counts[name] = carrier_counts.get(name, 0) + batch_size
                                batch_used = True
                    elif model_type == "TFN":
                        funconv = getattr(model, "funconv", None)
                        if funconv is not None and hasattr(funconv, "superparams"):
                            params = funconv.superparams
                            if params is None:
                                notes.append("no_param_carriers")
                                continue
                            grads = torch.autograd.grad(
                                s_sum, [params], retain_graph=False, allow_unused=True
                            )
                            grad_params = grads[0]
                            if grad_params is not None:
                                per_channel = _get_flag(
                                    cfg_args, "icu_tfn_per_channel", "icu_per_channel", default=False
                                )
                                if per_channel:
                                    for i in range(params.shape[0]):
                                        for j in range(params.shape[1]):
                                            u_val = torch.abs(params[i, j] * grad_params[i, j]).mean()
                                            u_val = u_val / max(batch_size, 1)
                                            name = f"TFconv:{i}:{j}"
                                            carrier_sums[name] = carrier_sums.get(name, 0.0) + float(u_val.item()) * batch_size
                                            carrier_counts[name] = carrier_counts.get(name, 0) + batch_size
                                            batch_used = True
                                else:
                                    for i in range(params.shape[0]):
                                        u_val = torch.abs(params[i] * grad_params[i]).mean()
                                        u_val = u_val / max(batch_size, 1)
                                        name = f"TFconv:{i}"
                                        carrier_sums[name] = carrier_sums.get(name, 0.0) + float(u_val.item()) * batch_size
                                        carrier_counts[name] = carrier_counts.get(name, 0) + batch_size
                                        batch_used = True
                    elif model_type == "SincNet":
                        conv1 = getattr(model, "conv1", None)
                        if conv1 is not None and hasattr(conv1, "a_") and hasattr(conv1, "b_"):
                            grads = torch.autograd.grad(
                                s_sum, [conv1.a_, conv1.b_], retain_graph=False, allow_unused=True
                            )
                            grad_a, grad_b = grads
                            out_channels = int(conv1.a_.shape[0])
                            for i in range(out_channels):
                                vals = []
                                if grad_a is not None:
                                    vals.append(torch.abs(conv1.a_[i] * grad_a[i]).mean())
                                if grad_b is not None:
                                    vals.append(torch.abs(conv1.b_[i] * grad_b[i]).mean())
                                if not vals:
                                    continue
                                u_val = torch.stack(vals).mean() / max(batch_size, 1)
                                name = f"Sinc:{i}"
                                carrier_sums[name] = carrier_sums.get(name, 0.0) + float(u_val.item()) * batch_size
                                carrier_counts[name] = carrier_counts.get(name, 0) + batch_size
                                batch_used = True
                        else:
                            notes.append("no_param_carriers")
                except Exception as exc:
                    notes.append(f"grad_error={type(exc).__name__}")

            if batch_used:
                total_samples += batch_size
                if total_samples >= icu_max_samples:
                    break

    if total_samples < icu_min_samples:
        notes.append(f"insufficient_samples={total_samples}")
        return None, None, None, notes

    u_values: Dict[str, float] = {}
    for name in carrier_sums:
        count = carrier_counts.get(name, 0)
        if count <= 0:
            continue
        u_values[name] = carrier_sums[name] / count

    if not u_values:
        notes.append("no_valid_u_values")
        return None, None, None, notes

    u_tensor = torch.tensor(list(u_values.values()), dtype=torch.float32)
    tau = float(torch.quantile(u_tensor, icu_quantile).item())
    tau_alt = float(0.1 * u_tensor.max().item())

    icu_eff = int(sum(1 for val in u_values.values() if val > tau))

    topk = sorted(u_values.items(), key=lambda item: item[1], reverse=True)[:icu_topk]
    topk_dict = {name: float(val) for name, val in topk}

    notes.append(f"tau={tau:.6g}")
    notes.append(f"tau_alt={tau_alt:.6g}")
    notes.append(f"quantile={icu_quantile}")
    notes.append(f"samples={total_samples}")
    return icu_eff, tau, topk_dict, notes


def estimate_icu(
    configs: Dict[str, Any],
    cfg_args: Any,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    params: Optional[int],
    device: torch.device,
    icu_min_samples: int,
    icu_max_samples: int,
    icu_quantile: float,
    icu_topk: int,
) -> Dict[str, Any]:
    struct_info = _compute_icu_struct(configs, cfg_args, model, input_tensor)
    icu_struct = struct_info["icu_struct"]
    capacity = struct_info["capacity"]
    notes = list(struct_info["notes"])
    model_type = struct_info["model_type"]

    icu_eff = None
    icu_topk_dict = None
    if icu_struct is not None:
        icu_eff, _tau, icu_topk_dict, eff_notes = _compute_icu_eff(
            model=model,
            cfg_args=cfg_args,
            model_type=model_type,
            device=device,
            icu_min_samples=icu_min_samples,
            icu_max_samples=icu_max_samples,
            icu_quantile=icu_quantile,
            icu_topk=icu_topk,
        )
        notes.extend(eff_notes)

    icu_density = None
    if icu_eff is not None and params:
        icu_density = float(icu_eff / (params / 1e6))

    return {
        "icu_struct": icu_struct,
        "icu_eff": icu_eff,
        "icu_density": icu_density,
        "icu_capacity": capacity,
        "icu_notes": notes,
        "icu_topk": icu_topk_dict,
    }


def _pool_carrier_tensor(
    tensor: torch.Tensor, mode: str, pool: str
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if torch.is_complex(tensor):
        tensor = torch.abs(tensor)
    else:
        tensor = tensor.abs()

    if tensor.ndim == 1:
        pooled = tensor.unsqueeze(1)
    elif tensor.ndim == 2:
        pooled = tensor
    else:
        if mode == "channel_last":
            if tensor.ndim < 3:
                return None
            reduce_dims = tuple(range(1, tensor.ndim - 1))
            if pool == "rms":
                pooled = torch.sqrt(torch.mean(tensor.pow(2), dim=reduce_dims))
            else:
                pooled = torch.mean(tensor, dim=reduce_dims)
        elif mode == "channel_first":
            if tensor.ndim < 3:
                return None
            reduce_dims = tuple(range(2, tensor.ndim))
            if pool == "rms":
                pooled = torch.sqrt(torch.mean(tensor.pow(2), dim=reduce_dims))
            else:
                pooled = torch.mean(tensor, dim=reduce_dims)
        else:
            reduce_dims = tuple(range(1, tensor.ndim))
            if pool == "rms":
                pooled = torch.sqrt(torch.mean(tensor.pow(2), dim=reduce_dims))
            else:
                pooled = torch.mean(tensor, dim=reduce_dims)
            pooled = pooled.unsqueeze(1)

    return pooled


def _forward_with_carriers(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg_args: Any,
    model_type: str,
    pool: str,
    enable_grad: bool,
) -> Tuple[torch.Tensor, OrderedDict, List[str]]:
    collector = CarrierCollector()
    handles, notes = _register_carrier_hooks(model, cfg_args, model_type, collector)
    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with context:
        logits = model(x)
    for handle in handles:
        handle.remove()

    carriers: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for record in collector.aggregate():
        pooled = _pool_carrier_tensor(record.tensor, record.mode, pool)
        if pooled is None:
            continue
        carriers[record.name] = pooled
    return logits, carriers, notes


def estimate_neuron_ig(
    model: torch.nn.Module,
    cfg_args: Any,
    model_type: str,
    device: torch.device,
    steps: int,
    baseline_mode: str,
    pool: str,
    use_label: bool,
    max_samples: int,
    baseline_samples: int,
    topk: int,
) -> Dict[str, Any]:
    notes: List[str] = []
    if model_type in ("None",):
        return {
            "status": "N/A",
            "notes": ["no_declared_carriers"],
        }
    if steps <= 0:
        return {"status": "disabled", "notes": ["steps<=0"]}

    seed = getattr(cfg_args, "seed", None)
    dataset, dataset_notes = _load_dataset_with_fallback(cfg_args, shuffle=False, seed=seed)
    notes.extend(dataset_notes)
    if dataset is None:
        return {"status": "N/A", "notes": notes}
    if any(note.startswith("dataset_fallback=") for note in dataset_notes):
        notes.append("nig_skipped_fallback_dataset")
        return {"status": "N/A", "notes": notes}

    batch_size = getattr(cfg_args, "batch_size", 1)
    if batch_size <= 0:
        batch_size = 1

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    in_dim = int(getattr(cfg_args, "in_dim", 0))
    in_channels = int(getattr(cfg_args, "in_channels", 0))
    if in_dim <= 0 or in_channels <= 0:
        return {"status": "N/A", "notes": notes + ["invalid_input_shape"]}

    baseline_mode = str(baseline_mode).lower()
    pool = str(pool).lower()
    notes.append(f"baseline={baseline_mode}")
    notes.append(f"pool={pool}")
    notes.append(f"steps={steps}")

    baseline = torch.zeros((1, in_dim, in_channels), dtype=torch.float32)
    if baseline_mode == "mean":
        sum_x = torch.zeros((1, in_dim, in_channels), dtype=torch.float32)
        count = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(sum_x.dtype)
            sum_x += x.sum(dim=0, keepdim=True)
            count += x.shape[0]
            if count >= baseline_samples:
                break
        if count > 0:
            baseline = sum_x / float(count)
            notes.append(f"baseline_samples={count}")
        else:
            notes.append("baseline_samples=0")

    baseline = baseline.to(device)

    total_samples = 0
    nig_sums: Dict[str, float] = {}
    nig_counts: Dict[str, int] = {}
    completeness_errors: List[float] = []

    model.eval()
    with torch.enable_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None
            if not isinstance(x, torch.Tensor):
                continue
            x = x.to(device)
            batch_size = x.shape[0]
            x0 = baseline.expand(batch_size, -1, -1)

            try:
                logits_x = model(x)
            except Exception as exc:
                notes.append(f"forward_error={type(exc).__name__}")
                continue
            if logits_x.ndim != 2:
                notes.append("logits_shape_unexpected")
                continue
            if use_label and y is not None:
                target_idx = y.to(device).long()
            else:
                target_idx = logits_x.argmax(dim=1)

            logits_x0, carriers_x0, hook_notes = _forward_with_carriers(
                model, x0, cfg_args, model_type, pool, enable_grad=False
            )
            notes.extend(hook_notes)
            if logits_x0.ndim != 2:
                notes.append("logits_shape_unexpected_baseline")
                continue

            if not carriers_x0:
                notes.append("no_carriers_captured")
                continue

            f_x0 = logits_x0.gather(1, target_idx.view(-1, 1)).squeeze(1).detach()
            c_prev = {name: tensor.detach() for name, tensor in carriers_x0.items()}
            nig_acc: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(tensor) for name, tensor in carriers_x0.items()
            }

            f_x = None
            for step in range(1, steps + 1):
                alpha = float(step) / float(steps)
                x_i = x0 + alpha * (x - x0)
                x_i = x_i.detach().requires_grad_(True)
                logits_i, carriers_i, hook_notes = _forward_with_carriers(
                    model, x_i, cfg_args, model_type, pool, enable_grad=True
                )
                notes.extend(hook_notes)
                if logits_i.ndim != 2:
                    notes.append("logits_shape_unexpected_path")
                    break
                f_i = logits_i.gather(1, target_idx.view(-1, 1)).squeeze(1)
                if step == steps:
                    f_x = f_i.detach()

                if not carriers_i:
                    notes.append("no_carriers_captured")
                    break

                carrier_names = []
                carrier_tensors = []
                no_grad_count = 0
                for name, tensor in carriers_i.items():
                    if not tensor.requires_grad:
                        no_grad_count += 1
                        continue
                    carrier_names.append(name)
                    carrier_tensors.append(tensor)
                if no_grad_count:
                    notes.append(f"nig_no_grad_carriers={no_grad_count}")
                if not carrier_tensors:
                    notes.append("nig_no_grad_carriers_all")
                    break
                grads = torch.autograd.grad(
                    f_i.sum(), carrier_tensors, retain_graph=False, allow_unused=True
                )
                for name, c_i, grad_i in zip(carrier_names, carrier_tensors, grads):
                    if grad_i is None:
                        continue
                    if name not in c_prev:
                        continue
                    delta = c_i - c_prev[name]
                    nig_acc[name] = nig_acc[name] + grad_i * delta
                    c_prev[name] = c_i.detach()

            if f_x is None:
                continue

            total_nig = torch.zeros_like(f_x)
            for name, nig_tensor in nig_acc.items():
                total_nig = total_nig + nig_tensor.sum(dim=1)
                abs_nig = nig_tensor.abs()
                for idx in range(abs_nig.shape[1]):
                    key = f"{name}:n{idx}"
                    vals = abs_nig[:, idx]
                    mask = torch.isfinite(vals)
                    if not mask.any():
                        continue
                    vals = vals[mask]
                    nig_sums[key] = nig_sums.get(key, 0.0) + float(vals.sum().item())
                    nig_counts[key] = nig_counts.get(key, 0) + int(vals.numel())

            delta_f = f_x - f_x0
            err = torch.abs(total_nig - delta_f)
            err = err[torch.isfinite(err)]
            completeness_errors.extend(err.detach().cpu().tolist())

            total_samples += batch_size
            if total_samples >= max_samples:
                break

    if total_samples == 0 or not nig_sums:
        notes.append("no_valid_nig")
        return {"status": "N/A", "notes": notes}

    nig_scores = {
        name: nig_sums[name] / nig_counts[name]
        for name in nig_sums
        if nig_counts[name] > 0
    }
    topk_items = sorted(nig_scores.items(), key=lambda item: item[1], reverse=True)[:topk]
    topk_dict = {name: float(score) for name, score in topk_items}

    completeness_mean = None
    completeness_p90 = None
    if completeness_errors:
        err_tensor = torch.tensor(completeness_errors, dtype=torch.float32)
        completeness_mean = float(err_tensor.mean().item())
        completeness_p90 = float(torch.quantile(err_tensor, 0.9).item())

    return {
        "status": "ok",
        "notes": notes,
        "topk": topk_dict,
        "completeness_mean": completeness_mean,
        "completeness_p90": completeness_p90,
        "samples": total_samples,
    }


def measure_latency_ms(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Optional[float]:
    model.eval()
    with torch.no_grad():
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
                for _ in range(warmup):
                    _ = model(input_tensor)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    _ = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                return elapsed_ms / max(iters, 1)

            for _ in range(warmup):
                _ = model(input_tensor)
            start_t = time.perf_counter()
            for _ in range(iters):
                _ = model(input_tensor)
            end_t = time.perf_counter()
            return (end_t - start_t) * 1000.0 / max(iters, 1)
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute params, FLOPs (MACs), and latency for model configs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config paths, e.g. configs/a_020_DIRG/config_TSPN_basic.yaml",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV output path (e.g. reports/dirg_020_model_stats.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for FLOPs/latency estimation.",
    )
    parser.add_argument(
        "--latency-batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Additional batch sizes for latency only (e.g. 64 128).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before latency timing.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="Timed iterations for latency measurement.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Override device (auto|cpu|cuda|cuda:0).",
    )
    parser.add_argument(
        "--icu-max-samples",
        type=int,
        default=512,
        help="Max samples for ICU_eff computation.",
    )
    parser.add_argument(
        "--icu-min-samples",
        type=int,
        default=500,
        help="Minimum samples required for ICU_eff; otherwise N/A.",
    )
    parser.add_argument(
        "--icu-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold for ICU_eff (tau=quantile(u,p)).",
    )
    parser.add_argument(
        "--icu-topk",
        type=int,
        default=5,
        help="Top-k carriers to record in ICU notes.",
    )
    parser.add_argument(
        "--nig-steps",
        type=int,
        default=32,
        help="Neuron-IG interpolation steps (n).",
    )
    parser.add_argument(
        "--nig-baseline",
        default="zero",
        help="Baseline for Neuron-IG: zero|mean.",
    )
    parser.add_argument(
        "--nig-baseline-samples",
        type=int,
        default=256,
        help="Samples used to estimate mean baseline.",
    )
    parser.add_argument(
        "--nig-pool",
        default="mean_abs",
        help="Carrier pooling: mean_abs|rms.",
    )
    parser.add_argument(
        "--nig-use-label",
        action="store_true",
        help="Use true labels for Neuron-IG target logit instead of argmax.",
    )
    parser.add_argument(
        "--nig-max-samples",
        type=int,
        default=256,
        help="Max samples for Neuron-IG.",
    )
    parser.add_argument(
        "--nig-topk",
        type=int,
        default=5,
        help="Top-k Neuron-IG carriers to record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict[str, Any]] = []

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    extra_latency_sizes = [int(x) for x in (args.latency_batch_sizes or []) if int(x) > 0]
    seen_sizes = set()
    extra_latency_sizes = [
        size for size in extra_latency_sizes if not (size in seen_sizes or seen_sizes.add(size))
    ]
    extra_latency_sizes = [size for size in extra_latency_sizes if size != args.batch_size]

    for config_path in args.configs:
        record: Dict[str, Any] = {
            "config": config_path,
            "model": "",
            "dataset_task": "",
            "target": "",
            "in_dim": "",
            "in_channels": "",
            "num_classes": "",
            "params": "",
            "trainable_params": "",
            "icu_struct": "",
            "icu_eff": "",
            "icu_density": "",
            "icu_capacity": "",
            "icu_notes": "",
            "icu_topk": "",
            "nig_status": "",
            "nig_steps": "",
            "nig_baseline": "",
            "nig_pool": "",
            "nig_completeness_mean": "",
            "nig_completeness_p90": "",
            "nig_topk": "",
            "nig_notes": "",
            "macs": "",
            "flops": "",
            "latency_ms": "",
            "device": "",
            "batch_size": args.batch_size,
            "status": "ok",
            "error": "",
        }
        for size in extra_latency_sizes:
            record[f"latency_ms_bs{size}"] = ""
        try:
            configs, cfg_args, _, _ = parse_arguments(
                config_path, 0, create_path=False, verbose=False
            )
            device = resolve_device(cfg_args, args.device)
            cfg_args.device = str(device)

            record["model"] = getattr(cfg_args, "model", "")
            record["dataset_task"] = getattr(cfg_args, "dataset_task", "")
            record["target"] = getattr(cfg_args, "target", "")
            record["in_dim"] = getattr(cfg_args, "in_dim", "")
            record["in_channels"] = getattr(cfg_args, "in_channels", "")
            record["num_classes"] = getattr(cfg_args, "num_classes", "")
            record["device"] = str(device)

            model = build_model(configs, cfg_args)
            model = model.to(device)

            input_tensor = torch.randn(
                args.batch_size,
                int(cfg_args.in_dim),
                int(cfg_args.in_channels),
                device=device,
            )

            total_params, trainable_params = count_parameters(model)
            record["params"] = total_params
            record["trainable_params"] = trainable_params

            struct_info = _compute_icu_struct(configs, cfg_args, model, input_tensor)
            model_type = struct_info.get("model_type", "None")

            icu_info = estimate_icu(
                configs=configs,
                cfg_args=cfg_args,
                model=model,
                input_tensor=input_tensor,
                params=total_params,
                device=device,
                icu_min_samples=args.icu_min_samples,
                icu_max_samples=args.icu_max_samples,
                icu_quantile=args.icu_quantile,
                icu_topk=args.icu_topk,
            )
            record["icu_struct"] = _format_na(icu_info.get("icu_struct"))
            record["icu_eff"] = _format_na(icu_info.get("icu_eff"))
            record["icu_density"] = _format_na(icu_info.get("icu_density"))
            record["icu_capacity"] = _safe_json(icu_info.get("icu_capacity"))
            record["icu_notes"] = _safe_json(icu_info.get("icu_notes"))
            record["icu_topk"] = _safe_json(icu_info.get("icu_topk"))

            nig_steps = int(getattr(cfg_args, "nig_steps", args.nig_steps))
            nig_baseline = str(getattr(cfg_args, "nig_baseline", args.nig_baseline))
            nig_pool = str(getattr(cfg_args, "nig_pool", args.nig_pool))
            nig_use_label = bool(getattr(cfg_args, "nig_use_label", args.nig_use_label))
            nig_max_samples = int(getattr(cfg_args, "nig_max_samples", args.nig_max_samples))
            nig_baseline_samples = int(
                getattr(cfg_args, "nig_baseline_samples", args.nig_baseline_samples)
            )
            nig_topk = int(getattr(cfg_args, "nig_topk", args.nig_topk))

            nig_info = estimate_neuron_ig(
                model=model,
                cfg_args=cfg_args,
                model_type=model_type,
                device=device,
                steps=nig_steps,
                baseline_mode=nig_baseline,
                pool=nig_pool,
                use_label=nig_use_label,
                max_samples=nig_max_samples,
                baseline_samples=nig_baseline_samples,
                topk=nig_topk,
            )
            record["nig_status"] = nig_info.get("status", "N/A")
            record["nig_steps"] = nig_steps
            record["nig_baseline"] = nig_baseline
            record["nig_pool"] = nig_pool
            record["nig_completeness_mean"] = _format_na(nig_info.get("completeness_mean"))
            record["nig_completeness_p90"] = _format_na(nig_info.get("completeness_p90"))
            record["nig_topk"] = _safe_json(nig_info.get("topk"))
            record["nig_notes"] = _safe_json(nig_info.get("notes"))

            macs = estimate_macs(model, input_tensor, device)
            record["macs"] = macs if macs is not None else ""
            record["flops"] = (macs * 2) if macs is not None else ""

            latency = measure_latency_ms(
                model, input_tensor, device, args.warmup, args.iters
            )
            record["latency_ms"] = latency if latency is not None else ""

            for size in extra_latency_sizes:
                latency_input = torch.randn(
                    size,
                    int(cfg_args.in_dim),
                    int(cfg_args.in_channels),
                    device=device,
                )
                latency_extra = measure_latency_ms(
                    model, latency_input, device, args.warmup, args.iters
                )
                record[f"latency_ms_bs{size}"] = (
                    latency_extra if latency_extra is not None else ""
                )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        results.append(record)
        status = record["status"]
        print(f"[{status}] {record['config']} -> {record['model']}")

    if args.output:
        output_path = args.output
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fieldnames = list(results[0].keys()) if results else []
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
