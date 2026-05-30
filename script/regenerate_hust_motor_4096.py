#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np


CLASS_ORDER = ["BF", "BOW", "BROKEN", "H", "MISAL", "UNBAL"]
HZ_LIST_DEFAULT = [5, 10, 20, 30]


def parse_raw_segments(path: Path, window: int, channels: int) -> np.ndarray:
    """
    Parse a HUST raw txt file and return segments shaped [B, window, channels].
    The file format is expected to contain a line: 'Time (seconds) and Data Channels'
    followed by numeric rows (time + channels).
    """
    found_data = False
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not found_data:
                if "Time (seconds) and Data Channels" in line:
                    found_data = True
                continue
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < channels + 1:
                continue
            try:
                values = [float(x) for x in parts[: channels + 1]]
            except ValueError:
                continue
            rows.append(values)

    if not found_data:
        raise ValueError(f"Data section not found in {path}")
    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    arr = np.array(rows, dtype=np.float32)
    if arr.shape[1] < channels + 1:
        raise ValueError(f"Expected at least {channels + 1} columns in {path}")

    signal = arr[:, 1 : 1 + channels]
    if signal.shape[0] % window != 0:
        raise ValueError(
            f"{path} rows ({signal.shape[0]}) not divisible by window {window}"
        )
    segments = signal.reshape(-1, window, channels)
    return segments


def build_dataset(raw_dir: Path, out_dir: Path, hz_list, window: int = 4096, channels: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)

    for hz in hz_list:
        segments_all = []
        labels_all = []

        for label, cls in enumerate(CLASS_ORDER):
            filename = f"{cls}_{hz}HZ.txt"
            path = raw_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing raw file: {path}")

            segments = parse_raw_segments(path, window=window, channels=channels)
            segments_all.append(segments.astype(np.float32))
            labels_all.append(np.full((segments.shape[0],), label, dtype=np.float32))

            print(f"[{hz}HZ] {filename}: segments={segments.shape[0]}")

        data = np.concatenate(segments_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)

        out_data = out_dir / f"HUST_{hz}Hz_6_data.npy"
        out_label = out_dir / f"HUST_{hz}Hz_6_label.npy"

        np.save(out_data, data)
        np.save(out_label, labels)

        # Basic summary
        unique, counts = np.unique(labels, return_counts=True)
        summary = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)])
        print(f"[{hz}HZ] saved: {out_data.name}, {out_label.name} | shape={data.shape} | labels={summary}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate HUST motor 4096 datasets from raw txt files.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/mnt/e/dataset/HUST motor multimodal dataset/Raw data"),
        help="Path to raw txt directory (contains *_HZ.txt files).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/e/dataset/generate/HUST_bearing_new/HUST_motor_4096"),
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--hz_list",
        type=str,
        default="5,10,20,30",
        help="Comma-separated list of HZ values to generate (e.g., '20' or '5,10,20,30').",
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
        default=4,
        help="Number of channels (C).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hz_list = [int(x.strip()) for x in args.hz_list.split(",") if x.strip()]
    if not hz_list:
        raise ValueError("hz_list is empty")

    build_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        hz_list=hz_list,
        window=args.window,
        channels=args.channels,
    )


if __name__ == "__main__":
    main()
