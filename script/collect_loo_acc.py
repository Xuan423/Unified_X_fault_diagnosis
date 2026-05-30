import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "+".join(str(item) for item in value)
    return str(value)


def find_latest_version(logs_path: Path) -> Optional[Path]:
    versions = [p for p in logs_path.glob("version_*") if p.is_dir()]
    if not versions:
        return None

    def version_key(path: Path) -> int:
        match = re.search(r"version_(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return max(versions, key=version_key)


def read_hparams(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_last_metric(path: Path, metric_name: str) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    for row in reversed(rows):
        if metric_name in row:
            value = parse_float(row.get(metric_name))
            if value is not None:
                return value
        for key, raw_value in row.items():
            if key.startswith(metric_name):
                value = parse_float(raw_value)
                if value is not None:
                    return value
    return None


def extract_iteration(exp_name: str) -> Optional[int]:
    match = re.search(r"_it(\d+)", exp_name)
    return int(match.group(1)) if match else None


def iter_experiments(save_root: Path, dataset_task: str) -> Iterable[Dict[str, Any]]:
    task_path = save_root / f"task_{dataset_task}"
    if not task_path.is_dir():
        return

    for model_path in task_path.glob("model_*"):
        if not model_path.is_dir():
            continue
        for exp_path in model_path.iterdir():
            if not exp_path.is_dir():
                continue
            logs_path = exp_path / "logs"
            version_path = find_latest_version(logs_path)
            if version_path is None:
                continue

            hparams = read_hparams(version_path / "hparams.yaml")
            test_acc = get_last_metric(version_path / "metrics.csv", "test_acc")
            if test_acc is None:
                test_acc = get_last_metric(exp_path / "test_result.csv", "test_acc")

            yield {
                "dataset_task": hparams.get("dataset_task", dataset_task),
                "model": hparams.get("model", model_path.name.replace("model_", "", 1)),
                "exp": exp_path.name,
                "iteration": extract_iteration(exp_path.name),
                "target": normalize_field(hparams.get("target")),
                "source": normalize_field(hparams.get("source")),
                "test_acc": test_acc,
                "exp_mtime": exp_path.stat().st_mtime,
            }


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": "", "std": "", "best": ""}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect leave-one-out test accuracy.")
    parser.add_argument("--save-root", default="save")
    parser.add_argument("--dataset-task", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--iteration-count", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()

    save_root = Path(args.save_root)
    output_prefix = Path(args.output_prefix)
    models = set(args.models or [])
    targets = set(args.targets or [])

    latest_by_key: Dict[Any, Dict[str, Any]] = {}
    for row in iter_experiments(save_root, args.dataset_task) or []:
        if models and row["model"] not in models:
            continue
        if targets and row["target"] not in targets:
            continue
        if row["iteration"] is None:
            continue
        key = (row["dataset_task"], row["model"], row["target"], row["source"], row["iteration"])
        previous = latest_by_key.get(key)
        if previous is None or row["exp_mtime"] > previous["exp_mtime"]:
            latest_by_key[key] = row

    raw_rows = sorted(
        latest_by_key.values(),
        key=lambda row: (row["model"], row["target"], row["source"], row["iteration"]),
    )
    raw_fieldnames = [
        "dataset_task",
        "model",
        "target",
        "source",
        "iteration",
        "test_acc",
        "exp",
    ]
    write_csv(output_prefix.with_name(output_prefix.name + "_raw.csv"), raw_fieldnames, raw_rows)

    grouped: Dict[Any, Dict[int, Optional[float]]] = {}
    for row in raw_rows:
        key = (row["dataset_task"], row["model"], row["target"], row["source"])
        grouped.setdefault(key, {})[row["iteration"]] = row["test_acc"]

    summary_rows: List[Dict[str, Any]] = []
    for key, acc_map in sorted(grouped.items()):
        dataset_task, model, target, source = key
        values = [v for v in acc_map.values() if isinstance(v, float)]
        stats = mean_std(values)
        summary_row = {
            "dataset_task": dataset_task,
            "model": model,
            "target": target,
            "source": source,
            "completed_runs": len(values),
            "mean_acc": stats["mean"],
            "std_acc": stats["std"],
            "best_acc": stats["best"],
        }
        for index in range(args.iteration_count):
            summary_row[f"acc_it{index}"] = acc_map.get(index, "")
        summary_rows.append(summary_row)

    summary_fieldnames = (
        ["dataset_task", "model", "target", "source", "completed_runs"]
        + [f"acc_it{index}" for index in range(args.iteration_count)]
        + ["mean_acc", "std_acc", "best_acc"]
    )
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    write_csv(summary_path, summary_fieldnames, summary_rows)

    overall_groups: Dict[str, List[float]] = {}
    for row in summary_rows:
        value = parse_float(row["mean_acc"])
        if value is not None:
            overall_groups.setdefault(row["model"], []).append(value)

    overall_rows: List[Dict[str, Any]] = []
    for model, values in sorted(overall_groups.items()):
        stats = mean_std(values)
        overall_rows.append(
            {
                "dataset_task": args.dataset_task,
                "model": model,
                "completed_tasks": len(values),
                "mean_task_acc": stats["mean"],
                "std_task_acc": stats["std"],
                "best_task_acc": stats["best"],
            }
        )

    overall_path = output_prefix.with_name(output_prefix.name + "_overall.csv")
    write_csv(
        overall_path,
        ["dataset_task", "model", "completed_tasks", "mean_task_acc", "std_task_acc", "best_task_acc"],
        overall_rows,
    )

    print(f"Wrote raw results: {output_prefix.with_name(output_prefix.name + '_raw.csv')}")
    print(f"Wrote leave-one-out summary: {summary_path}")
    print(f"Wrote model overall summary: {overall_path}")


if __name__ == "__main__":
    main()
