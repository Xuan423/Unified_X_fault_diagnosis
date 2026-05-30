#!/usr/bin/env python3
import argparse
import csv
import os
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
if os.name != "nt":
    release = platform.release().lower()
    if "microsoft" in release or os.environ.get("WSL_INTEROP"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configs.config import config_network
from trainer.trainer_basic import Basic_plmodel
from data.data_provider import DATASET_TASK_CLASS
from data import datasets as datasets_module

TSPN_FAMILY = {"TSPN", "TKAN", "NNSPN", "TFON"}


def normalize_path(path_str: str) -> Path:
    path_str = os.path.expanduser(path_str)
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", path_str)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    return Path(path_str)


def ensure_trailing_sep(path: str) -> str:
    if not path:
        return path
    if path.endswith(("/", "\\")):
        return path
    return path + "/"


def parse_hparams(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_run_dirs(save_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    if not save_root.exists():
        return run_dirs
    hparams = save_root / "logs" / "version_0" / "hparams.yaml"
    if hparams.exists():
        return [save_root]

    direct_runs = []
    for item in sorted(save_root.iterdir()):
        if not item.is_dir():
            continue
        hparams = item / "logs" / "version_0" / "hparams.yaml"
        if hparams.exists():
            direct_runs.append(item)
    if direct_runs:
        return direct_runs
    for model_dir in sorted(save_root.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("model_"):
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            hparams = run_dir / "logs" / "version_0" / "hparams.yaml"
            if hparams.exists():
                run_dirs.append(run_dir)
    return run_dirs


def parse_seed_idx(name: str) -> Optional[int]:
    match = re.search(r"_it(\d+)", name)
    return int(match.group(1)) if match else None


def parse_target_from_name(name: str) -> Optional[str]:
    match = re.search(r"_target([^_]+.*)$", name)
    return match.group(1) if match else None


def find_best_checkpoint(run_dir: Path) -> Tuple[Optional[Path], Optional[float]]:
    ckpts = list(run_dir.glob("*.ckpt"))
    if not ckpts:
        return None, None
    best_ckpt = None
    best_loss = None
    for ckpt in ckpts:
        match = re.search(r"val_loss=([0-9.]+)", ckpt.name)
        loss = float(match.group(1)) if match else None
        if loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_ckpt = ckpt
    if best_ckpt is not None:
        return best_ckpt, best_loss
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0], None


def read_reported_test_acc(run_dir: Path) -> Optional[float]:
    result_path = run_dir / "test_result.csv"
    if not result_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(result_path)
        if "test_acc" not in df.columns or df.empty:
            return None
        value = float(df["test_acc"].iloc[0])
        return value
    except Exception:
        return None


def load_config(path: Path) -> Dict[str, Any]:
    path = normalize_path(str(path))
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_tspn_configs(config_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    index: Dict[Tuple[str, str], List[Path]] = {}
    for path in config_root.rglob("*.yaml"):
        try:
            config = load_config(path)
        except Exception:
            continue
        if not isinstance(config, dict):
            continue
        args = config.get("args", {})
        model = args.get("model")
        dataset_task = args.get("dataset_task")
        if not model or not dataset_task:
            continue
        if model not in TSPN_FAMILY:
            continue
        key = (model, dataset_task)
        index.setdefault(key, []).append(path)
    return index


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
        return model_dict[args.model](signal_processing_modules, feature_extractor_modules, args)

    if args.model == "Resnet":
        from model_collection.Resnet import BasicBlock, ResNet
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes)
    if args.model == "WKN_m":
        from model_collection.Resnet import BasicBlock
        from model_collection.WKN import WKN_m
        return WKN_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes)
    if args.model == "Sinc_net_m":
        from model_collection.Resnet import BasicBlock
        from model_collection.Sincnet import Sinc_net_m
        return Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes)
    if args.model == "Huan_net":
        from model_collection.MWA_CNN import Huan_net
        return Huan_net(input_size=args.in_channels, num_class=args.num_classes)
    if args.model == "TFN_Morlet":
        from model_collection.TFN.Models.TFN import TFN_Morlet
        return TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes)
    if args.model == "MCN_GFK":
        from model_collection.MCN.models import MultiChannel_MCN_GFK
        ff = np.arange(0, args.in_dim // 2 + 1) / args.in_dim // 2 + 1
        return MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes)
    if args.model == "Convoformer_NSE":
        from model_collection.Convformer_NSE import convoformer_v1_small
        return convoformer_v1_small(in_channel=args.in_channels, out_channel=args.num_classes)
    if args.model == "BSSTN_Flex":
        from model_collection.bsstn_flex import BSSTNFlex
        return BSSTNFlex(num_classes=args.num_classes, max_sensors=args.in_channels)

    raise ValueError(f"Unsupported model: {args.model}")


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    idx = torch.arange(labels.numel(), device=labels.device)
    nll = -torch.log(probs[idx, labels]).mean()
    return float(nll.item())


def compute_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(probs, dim=1)
    acc = preds.eq(labels).float().mean()
    return float(acc.item())


def compute_entropy(probs: torch.Tensor) -> float:
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1).mean()
    return float(entropy.item())


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            acc_bin = accuracies[mask].float().mean()
            conf_bin = confidences[mask].mean()
            ece += mask.float().mean() * (acc_bin - conf_bin).abs()
    return float(ece.item())


class FlexClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_channels, hidden), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden, out_channels))
        self.clf = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.clf(x)


def maybe_patch_tspn_classifier(
    model_plain: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    key_in = "network.clf.clf.0.weight"
    if key_in not in state_dict:
        return

    in_features = int(state_dict[key_in].shape[1])
    hidden = int(state_dict[key_in].shape[0])
    key_out = None
    dropout = False
    if "network.clf.clf.3.weight" in state_dict:
        key_out = "network.clf.clf.3.weight"
        dropout = True
    elif "network.clf.clf.2.weight" in state_dict:
        key_out = "network.clf.clf.2.weight"
    if key_out is None:
        return
    out_features = int(state_dict[key_out].shape[0])

    current = getattr(model_plain, "clf", None)
    if current is not None and hasattr(current, "clf"):
        seq = current.clf
        if isinstance(seq, nn.Sequential) and len(seq) >= 2:
            cur_hidden = getattr(seq[0], "out_features", None)
            cur_out = getattr(seq[-1], "out_features", None)
            cur_dropout = any(isinstance(layer, nn.Dropout) for layer in seq)
            if cur_hidden == hidden and cur_out == out_features and cur_dropout == dropout:
                return

    model_plain.clf = FlexClassifier(in_features, hidden, out_features, dropout).to(device)


def add_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    snr_lin = 10 ** (snr_db / 10.0)
    xpower = x.pow(2).mean(dim=(1, 2), keepdim=True)
    npower = xpower / snr_lin
    noise = torch.randn_like(x) * torch.sqrt(npower)
    return x + noise


def _baseline_like(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        mean = x.mean(dim=1, keepdim=True)
        return mean.expand_as(x)
    return torch.zeros_like(x)


def _importance_grad_input(
    model: torch.nn.Module,
    x: torch.Tensor,
    axis: str,
) -> Tuple[torch.Tensor, int]:
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())
    score = logits[:, pred].sum()
    grad = torch.autograd.grad(score, x, allow_unused=True)[0]
    if grad is None:
        if axis == "channel":
            importance = x.abs().sum(dim=1).squeeze(0)
        else:
            importance = x.abs().sum(dim=2).squeeze(0)
    else:
        if axis == "channel":
            importance = (grad * x).abs().sum(dim=1).squeeze(0)
        else:
            importance = (grad * x).abs().sum(dim=2).squeeze(0)
    return importance, pred


def compute_del_ins_auc(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    steps: int = 20,
    max_samples: int = 200,
    baseline: str = "zero",
    seed: int = 17,
    snr_db: Optional[float] = None,
    axis: str = "time",
) -> Tuple[Optional[float], Optional[float], int]:
    if max_samples <= 0:
        return None, None, 0
    if len(dataset) == 0:
        return None, None, 0
    indices = np.arange(len(dataset))
    if max_samples > 0 and len(indices) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    fractions = np.linspace(0, 1, steps + 1)
    del_scores: List[float] = []
    ins_scores: List[float] = []

    model.eval()
    for batch in loader:
        x, _ = batch
        x = x.to(device)
        if snr_db is not None:
            x = add_noise(x, snr_db)
        with torch.enable_grad():
            importance, pred = _importance_grad_input(model, x, axis)
        order = torch.argsort(importance, descending=True)
        original = x.detach()
        base = _baseline_like(original, baseline)
        L = original.shape[2] if axis == "channel" else original.shape[1]

        del_curve = []
        ins_curve = []
        for frac in fractions:
            k = int(round(frac * L))
            k = min(k, L)

            if k == 0:
                x_del = original
                x_ins = base
            else:
                idx = order[:k]
                x_del = original.clone()
                if axis == "channel":
                    x_del[:, :, idx] = base[:, :, idx]
                else:
                    x_del[:, idx, :] = base[:, idx, :]
                x_ins = base.clone()
                if axis == "channel":
                    x_ins[:, :, idx] = original[:, :, idx]
                else:
                    x_ins[:, idx, :] = original[:, idx, :]

            with torch.no_grad():
                prob_del = torch.softmax(model(x_del), dim=1)[0, pred].item()
                prob_ins = torch.softmax(model(x_ins), dim=1)[0, pred].item()
            del_curve.append(prob_del)
            ins_curve.append(prob_ins)

        del_auc = float(np.trapezoid(del_curve, fractions))
        ins_auc = float(np.trapezoid(ins_curve, fractions))
        del_scores.append(del_auc)
        ins_scores.append(ins_auc)

    return float(np.mean(del_scores)), float(np.mean(ins_scores)), len(subset)


def gather_probs_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_db: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if snr_db is not None:
                x = add_noise(x, snr_db)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs)
            labels_list.append(y.long())
    probs_all = torch.cat(probs_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    return probs_all, labels_all


def gather_logits_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_db: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if snr_db is not None:
                x = add_noise(x, snr_db)
            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y.long())
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    return logits_all, labels_all


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float:
    temperature = torch.ones(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.detach().item())


def pick_dataset_class(args_obj: Any) -> Any:
    dataset_task = getattr(args_obj, "dataset_task", "")
    dataset_class = DATASET_TASK_CLASS.get(dataset_task)
    if dataset_task != "DIRG_020_basic":
        if dataset_class is None:
            raise ValueError(f"Unknown dataset_task: {dataset_task}")
        return dataset_class

    data_dir = ensure_trailing_sep(str(getattr(args_obj, "data_dir", "")))
    target = getattr(args_obj, "target", "")
    if target:
        data_pref = os.path.join(data_dir, f"data_{target}.npy")
        alt_pref = os.path.join(data_dir, f"{target}_data.npy")
        if os.path.exists(data_pref):
            return getattr(datasets_module, "DIRG_020_basic")
        if os.path.exists(alt_pref):
            return getattr(datasets_module, "Default_dataset")
    if dataset_class is None:
        raise ValueError(f"Unknown dataset_task: {dataset_task}")
    return dataset_class


class FullNpyDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_full_dataset(args_obj: Any) -> Dataset:
    data_dir = ensure_trailing_sep(str(getattr(args_obj, "data_dir", "")))
    target_raw = getattr(args_obj, "target", None)
    if target_raw is None:
        target_raw = getattr(args_obj, "target_list", None)
    targets: List[str] = []
    if isinstance(target_raw, str):
        targets = [target_raw]
    elif isinstance(target_raw, (list, tuple)):
        for item in target_raw:
            if isinstance(item, (list, tuple)):
                targets.extend([str(x) for x in item])
            else:
                targets.append(str(item))
    elif target_raw is not None:
        targets = [str(target_raw)]

    if not targets:
        raise ValueError("Target is required for full-data evaluation.")

    data_list = []
    label_list = []
    for target in targets:
        candidates = [
            (f"data_{target}.npy", f"label_{target}.npy"),
            (f"{target}_data.npy", f"{target}_label.npy"),
        ]
        found = False
        for data_name, label_name in candidates:
            data_path = os.path.join(data_dir, data_name)
            label_path = os.path.join(data_dir, label_name)
            if os.path.exists(data_path) and os.path.exists(label_path):
                data = np.load(data_path).astype(np.float32, copy=False)
                labels = np.load(label_path).astype(np.float32, copy=False)
                data_list.append(data)
                label_list.append(labels)
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"No full-data files found for target={target} in {data_dir}"
            )

    data_all = np.concatenate(data_list, axis=0)
    labels_all = np.concatenate(label_list, axis=0)
    return FullNpyDataset(data_all, labels_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NLL, ECE, Entropy, Acc and optional Deletion/Insertion AUC."
    )
    parser.add_argument("--save-root", required=True, help="Root dir: save/task_xxx")
    parser.add_argument("--data-dir", required=True, help="Local dataset directory.")
    parser.add_argument(
        "--filter-target",
        default="",
        help="Only evaluate targets in this comma-separated list.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save CSV results (default: save-root/reports).",
    )
    parser.add_argument("--snr-db", type=float, default=-1, help="SNR (dB) for noise, set <0 to disable.")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--auc-max-samples", type=int, default=0)
    parser.add_argument("--auc-steps", type=int, default=20)
    parser.add_argument(
        "--auc-axis",
        choices=("time", "channel"),
        default="time",
        help="Masking axis for del/ins AUC: time (default) or channel (feature dimension).",
    )
    parser.add_argument(
        "--auc-only",
        action="store_true",
        help="Only compute del/ins AUC (skip acc/nll/ece/entropy/temp-scaling).",
    )
    parser.add_argument("--temperature-scaling", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-reported-acc",
        action="store_true",
        help="Use test_result.csv test_acc when available (still computes other metrics).",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        return torch.device("cpu")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    save_root = normalize_path(args.save_root)
    data_dir = ensure_trailing_sep(str(normalize_path(args.data_dir)))
    output_dir = normalize_path(args.output_dir) if args.output_dir else save_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_db = args.snr_db if args.snr_db is not None and args.snr_db >= 0 else None
    device = pick_device()
    target_filter = [t.strip() for t in args.filter_target.split(",") if t.strip()]

    tspn_index = scan_tspn_configs(Path(ROOT_DIR) / "configs")

    run_dirs = find_run_dirs(save_root)
    if not run_dirs:
        print(f"No runs found under {save_root}")
        return

    records: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        hparams_path = run_dir / "logs" / "version_0" / "hparams.yaml"
        hparams = parse_hparams(hparams_path)
        model_name = str(hparams.get("model", ""))
        dataset_task = str(hparams.get("dataset_task", ""))
        target_raw = hparams.get("target") or parse_target_from_name(run_dir.name)
        target_values = []
        if isinstance(target_raw, str):
            target_values = [target_raw]
        elif isinstance(target_raw, (list, tuple)):
            for item in target_raw:
                if isinstance(item, (list, tuple)):
                    target_values.extend([str(x) for x in item])
                else:
                    target_values.append(str(item))
        elif target_raw is not None:
            target_values = [str(target_raw)]
        target = "+".join(target_values) if target_values else ""
        if target_filter and not any(t in target_filter for t in target_values):
            continue
        seed = hparams.get("seed")
        seed_idx = parse_seed_idx(run_dir.name)

        ckpt_path, ckpt_val_loss = find_best_checkpoint(run_dir)
        record: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "model": model_name,
            "variant": "",
            "dataset_task": dataset_task,
            "target": target,
            "seed": seed if seed is not None else "",
            "seed_idx": seed_idx if seed_idx is not None else "",
            "ckpt_path": str(ckpt_path) if ckpt_path else "",
            "ckpt_val_loss": ckpt_val_loss if ckpt_val_loss is not None else "",
            "n_samples": "",
            "n_samples_auc": "",
            "auc_axis": args.auc_axis,
            "acc": "",
            "acc_eval": "",
            "acc_reported": "",
            "nll": "",
            "ece": "",
            "entropy": "",
            "del_auc": "",
            "ins_auc": "",
            "temp": "",
            "nll_ts": "",
            "ece_ts": "",
            "status": "ok",
            "error": "",
        }

        if not ckpt_path:
            record["status"] = "error"
            record["error"] = "No checkpoint found."
            records.append(record)
            continue

        try:
            args_obj = SimpleNamespace(**hparams)
            args_obj.data_dir = data_dir
            args_obj.device = str(device)

            state = torch.load(ckpt_path, map_location=device)
            model_pl = None
            variant = ""

            if model_name in TSPN_FAMILY:
                key = (model_name, dataset_task)
                candidates = tspn_index.get(key, [])
                for cfg_path in candidates:
                    configs = load_config(cfg_path)
                    model_plain = build_model(configs, args_obj)
                    if model_name == "TSPN":
                        maybe_patch_tspn_classifier(model_plain, state["state_dict"], device)
                    candidate = Basic_plmodel(model_plain, args_obj)
                    try:
                        candidate.load_state_dict(state["state_dict"])
                    except Exception:
                        continue
                    model_pl = candidate
                    variant = cfg_path.stem.replace("config_", "")
                    break
                if model_pl is None:
                    raise RuntimeError("No matching TSPN config found for checkpoint.")
            else:
                model_plain = build_model({}, args_obj)
                model_pl = Basic_plmodel(model_plain, args_obj)
                model_pl.load_state_dict(state["state_dict"])

            model_pl.to(device)
            model_pl.eval()

            record["variant"] = variant

            test_dataset = build_full_dataset(args_obj)
            loader = DataLoader(
                test_dataset,
                batch_size=int(getattr(args_obj, "batch_size", 64)),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=device.type == "cuda",
            )

            if not args.auc_only:
                probs_all, labels_all = gather_probs_and_labels(
                    model_pl, loader, device, snr_db=snr_db
                )

                record["n_samples"] = int(labels_all.numel())
                acc_eval = compute_accuracy(probs_all, labels_all)
                record["acc_eval"] = acc_eval
                reported_acc = read_reported_test_acc(run_dir)
                record["acc_reported"] = reported_acc if reported_acc is not None else ""
                if args.use_reported_acc and reported_acc is not None:
                    record["acc"] = reported_acc
                else:
                    record["acc"] = acc_eval
                record["nll"] = compute_nll(probs_all, labels_all)
                record["ece"] = compute_ece(probs_all, labels_all, n_bins=args.ece_bins)
                record["entropy"] = compute_entropy(probs_all)
            else:
                record["n_samples"] = len(test_dataset)

            if args.auc_max_samples > 0:
                del_auc, ins_auc, used = compute_del_ins_auc(
                    model_pl,
                    test_dataset,
                    device=device,
                    steps=args.auc_steps,
                    max_samples=args.auc_max_samples,
                    baseline="zero",
                    seed=17,
                    snr_db=snr_db,
                    axis=args.auc_axis,
                )
                record["del_auc"] = del_auc if del_auc is not None else ""
                record["ins_auc"] = ins_auc if ins_auc is not None else ""
                record["n_samples_auc"] = used
            else:
                record["n_samples_auc"] = 0

            if args.temperature_scaling and not args.auc_only:
                dataset_class = pick_dataset_class(args_obj)
                val_dataset = dataset_class(args_obj, flag="val")
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=int(getattr(args_obj, "batch_size", 64)),
                    shuffle=False,
                    num_workers=int(args.num_workers),
                    pin_memory=device.type == "cuda",
                )
                val_logits, val_labels = gather_logits_and_labels(
                    model_pl, val_loader, device, snr_db=snr_db
                )
                temp = fit_temperature(val_logits, val_labels)
                record["temp"] = temp
                scaled_probs = torch.softmax(torch.log(probs_all + 1e-12) / temp, dim=1)
                record["nll_ts"] = compute_nll(scaled_probs, labels_all)
                record["ece_ts"] = compute_ece(scaled_probs, labels_all, n_bins=args.ece_bins)

        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        records.append(record)
        print(f"[{record['status']}] {run_dir}")

    if not records:
        print("No records collected.")
        return

    raw_path = output_dir / "uncertainty_explain_by_seed.csv"
    fieldnames = list(records[0].keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {raw_path}")

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        numeric_cols = [
            "acc",
            "acc_eval",
            "acc_reported",
            "nll",
            "ece",
            "entropy",
            "del_auc",
            "ins_auc",
            "temp",
            "nll_ts",
            "ece_ts",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        group_cols = ["model", "variant", "dataset_task", "target"]
        summary = df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
        summary.columns = [
            "_".join([c for c in col if c]) if isinstance(col, tuple) else col
            for col in summary.columns
        ]
        summary_path = output_dir / "uncertainty_explain_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    except Exception as exc:
        print(f"Summary skipped: {exc}")


if __name__ == "__main__":
    main()
