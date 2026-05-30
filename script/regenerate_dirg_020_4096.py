#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for v7.3 MAT files
    h5py = None


DOMAIN_RE = re.compile(r"Speed:\s*(\d+)\s*Hz,\s*Load:\s*(\d+)\s*N")


def parse_domain_description(text: str) -> tuple[int, int]:
    match = DOMAIN_RE.search(text or "")
    if not match:
        raise ValueError(f"Unrecognized Domain_description format: {text!r}")
    return int(match.group(1)), int(match.group(2))


def select_signal_array(mat_dict: dict) -> np.ndarray:
    candidates = []
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            candidates.append(value)
    if not candidates:
        raise ValueError("No numeric array found in MAT contents.")
    arr = max(candidates, key=lambda x: x.size)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {arr.shape}")
    return arr


def load_mat_signal(path: Path) -> np.ndarray:
    try:
        mat = loadmat(path)
        return select_signal_array(mat)
    except NotImplementedError as exc:
        if h5py is None:
            raise RuntimeError("MAT v7.3 detected but h5py is not installed.") from exc
        with h5py.File(path, "r") as f:
            datasets = []
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "shape") and len(obj.shape) >= 2:
                    datasets.append(obj)
            if not datasets:
                raise ValueError("No numeric dataset found in MAT file.")
            arr = np.array(max(datasets, key=lambda x: x.size))
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array after squeeze, got shape {arr.shape}")
        return arr


def orient_signal(signal: np.ndarray, channels: int | None) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal array, got shape {signal.shape}")
    if channels is None:
        return signal

    if signal.shape[1] == channels:
        return signal
    if signal.shape[0] == channels:
        return signal.T

    # If neither dimension matches channels, attempt best-effort orientation.
    if signal.shape[0] > signal.shape[1] and signal.shape[1] > channels:
        return signal[:, :channels]
    if signal.shape[1] > signal.shape[0] and signal.shape[0] > channels:
        return signal[:channels, :].T

    raise ValueError(
        f"Cannot align signal shape {signal.shape} to channels={channels}"
    )


def segment_signal(signal: np.ndarray, window: int) -> np.ndarray:
    total = signal.shape[0]
    if total % window != 0:
        raise ValueError(f"Signal length {total} is not divisible by window {window}")
    return signal.reshape(-1, window, signal.shape[1])


def load_metadata(metadata_path: Path, dataset_name: str, file_prefix: str) -> pd.DataFrame:
    df = pd.read_excel(metadata_path)
    df = df[df["Name"] == dataset_name]
    df = df[df["File"].str.startswith(file_prefix)]
    if df.empty:
        raise ValueError(
            f"No rows found for Name={dataset_name!r} with File prefix {file_prefix!r}"
        )
    df = df.copy()
    df["Label"] = df["Label"].astype(int)
    df["Domain_id"] = df["Domain_id"].astype(int)
    df[["Speed", "Load"]] = df["Domain_description"].apply(
        lambda x: pd.Series(parse_domain_description(x))
    )
    return df


def build_datasets(
    meta_df: pd.DataFrame,
    raw_dir: Path,
    out_dir: Path,
    window: int,
    channels: int | None,
    channel_indices: list[int] | None,
    name_prefix: str,
    limit_files: int,
    strict: bool,
    save_summary: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    grouped = meta_df.groupby(["Speed", "Load", "Domain_id"], sort=True)
    for (speed, load, domain_id), group in grouped:
        group = group.sort_values("File")
        if limit_files > 0:
            group = group.head(limit_files)

        dataset_name = f"{name_prefix}_{speed}Hz_{load}N"
        segments_all = []
        labels_all = []
        file_count = 0

        for _, row in group.iterrows():
            file_name = row["File"]
            label = int(row["Label"])
            path = raw_dir / file_name
            if not path.exists():
                message = f"Missing raw file: {path}"
                if strict:
                    raise FileNotFoundError(message)
                print(f"WARNING: {message}")
                continue

            signal = load_mat_signal(path)
            signal = orient_signal(signal, channels)
            if channel_indices is not None:
                signal = signal[:, channel_indices]

            segments = segment_signal(signal, window).astype(np.float32, copy=False)
            labels = np.full((segments.shape[0],), label, dtype=np.float32)

            segments_all.append(segments)
            labels_all.append(labels)
            file_count += 1

            print(
                f"[{dataset_name}] {file_name}: label={label} "
                f"segments={segments.shape[0]}"
            )

        if not segments_all:
            print(f"[{dataset_name}] skipped (no data)")
            continue

        data = np.concatenate(segments_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)

        out_data = out_dir / f"data_{dataset_name}.npy"
        out_label = out_dir / f"label_{dataset_name}.npy"
        np.save(out_data, data)
        np.save(out_label, labels)

        unique, counts = np.unique(labels, return_counts=True)
        label_summary = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)])
        print(
            f"[{dataset_name}] saved {out_data.name}, {out_label.name} "
            f"| shape={data.shape} | labels={label_summary}"
        )

        summaries.append(
            {
                "dataset_name": dataset_name,
                "domain_id": domain_id,
                "speed_hz": speed,
                "load_n": load,
                "files": file_count,
                "samples": int(data.shape[0]),
                "channels": int(data.shape[2]),
                "window": window,
            }
        )

    if save_summary and summaries:
        summary_path = out_dir / "dirg_020_domain_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate RM_020_DIRG datasets (B x 4096 x C) from PHMbench raw MAT files."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/mnt/e/dataset/PHMbench-raw_data/metadata.xlsx"),
        help="Path to metadata.xlsx.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/mnt/e/dataset/PHMbench-raw_data/raw/RM_020_DIRG"),
        help="Path to RM_020_DIRG raw MAT files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/e/dataset/generate/PHMbench_DIRG_020_4096"),
        help="Output directory for generated .npy files.",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="DIRG_020",
        help="Prefix used in output dataset names (data_{name}.npy).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4096,
        help="Window length (L).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=6,
        help="Expected number of channels (C) in the raw data.",
    )
    parser.add_argument(
        "--channel_indices",
        type=str,
        default="",
        help="Optional comma-separated channel indices (0-based) to keep.",
    )
    parser.add_argument(
        "--domain_ids",
        type=str,
        default="",
        help="Comma-separated domain IDs to include (e.g., '1,2,3').",
    )
    parser.add_argument(
        "--speeds",
        type=str,
        default="",
        help="Comma-separated speeds (Hz) to include (e.g., '100,200').",
    )
    parser.add_argument(
        "--loads",
        type=str,
        default="",
        help="Comma-separated loads (N) to include (e.g., '0,1000').",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=0,
        help="If >0, limit the number of files per domain (useful for quick tests).",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing raw files instead of failing.",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        help="Write a CSV summary of generated domains to out_dir.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    meta_df = load_metadata(
        metadata_path=args.metadata,
        dataset_name="RM_020_DIRG",
        file_prefix="C",
    )

    if args.domain_ids:
        domain_ids = set(parse_int_list(args.domain_ids))
        meta_df = meta_df[meta_df["Domain_id"].isin(domain_ids)]

    if args.speeds:
        speeds = set(parse_int_list(args.speeds))
        meta_df = meta_df[meta_df["Speed"].isin(speeds)]

    if args.loads:
        loads = set(parse_int_list(args.loads))
        meta_df = meta_df[meta_df["Load"].isin(loads)]

    if meta_df.empty:
        raise ValueError("No metadata rows remain after filtering.")

    channel_indices = None
    if args.channel_indices:
        channel_indices = parse_int_list(args.channel_indices)

    build_datasets(
        meta_df=meta_df,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        window=args.window,
        channels=args.channels,
        channel_indices=channel_indices,
        name_prefix=args.name_prefix,
        limit_files=args.limit_files,
        strict=not args.allow_missing,
        save_summary=args.save_summary,
    )


if __name__ == "__main__":
    main()
