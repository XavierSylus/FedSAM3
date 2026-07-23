#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("paper_figure_generation_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 2 mean +/- sample-std training curves from training_history.json."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing training history: {path}")
    text = path.read_text(encoding="utf-8")
    text = text.replace("Infinity", "999999.0").replace("NaN", "null")
    return json.loads(text)


def finite_float(value: Any, label: str, invalid_min: float | None = None) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite value for {label}: {value}")
    if invalid_min is not None and result >= invalid_min:
        return float("nan")
    return result


def series_from_round_list(
    history: dict[str, Any],
    field: str,
    label: str,
    invalid_min: float | None,
) -> dict[int, float]:
    rounds = history.get("rounds")
    values = history.get(field)
    if not isinstance(rounds, list) or not isinstance(values, list):
        raise ValueError(f"{label} must contain list fields 'rounds' and '{field}'")
    if len(rounds) != len(values):
        raise ValueError(f"{label} rounds/{field} length mismatch: {len(rounds)} vs {len(values)}")
    return {
        int(round_id): finite_float(value, f"{label}:{field}:round_{round_id}", invalid_min)
        for round_id, value in zip(rounds, values)
    }


def series_from_val_metrics(
    history: dict[str, Any],
    metric: str,
    label: str,
    invalid_min: float | None,
) -> dict[int, float]:
    rows = history.get("val_metrics")
    if not isinstance(rows, list):
        raise ValueError(f"{label} must contain list field 'val_metrics'")
    series: dict[int, float] = {}
    for row in rows:
        round_id = int(row["round"])
        series[round_id] = finite_float(
            row.get(metric),
            f"{label}:val_{metric}:round_{round_id}",
            invalid_min,
        )
    return series


def align_seed_series(
    histories: list[tuple[int, dict[str, Any]]],
    metric_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    metric_name = metric_config["name"]
    invalid_min = metric_config.get("invalid_min")
    per_seed: list[dict[int, float]] = []
    for seed, history in histories:
        label = f"seed_{seed}"
        if metric_name == "train_loss":
            series = series_from_round_list(history, "avg_losses", label, invalid_min)
        elif metric_name == "grad_conflict_deg":
            series = series_from_round_list(history, "grad_conflict_deg", label, invalid_min)
        else:
            series = series_from_val_metrics(history, metric_name, label, invalid_min)
        per_seed.append(series)

    rounds = sorted(set().union(*(set(series) for series in per_seed)))
    matrix = np.array([[series.get(round_id, np.nan) for round_id in rounds] for series in per_seed], dtype=float)
    drop_incomplete = bool(metric_config.get("drop_incomplete_rounds", False))
    if drop_incomplete:
        valid_columns = np.isfinite(matrix).all(axis=0)
        if not valid_columns.any():
            raise ValueError(f"No complete rounds remain for {metric_name}")
        rounds = [round_id for round_id, keep in zip(rounds, valid_columns) if keep]
        matrix = matrix[:, valid_columns]
    if not np.isfinite(matrix).all():
        missing = []
        for seed_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                if not np.isfinite(value):
                    missing.append(f"seed_{histories[seed_idx][0]}:round_{rounds[col_idx]}")
        raise ValueError(f"Missing {metric_name} observations: {', '.join(missing[:10])}")
    return np.array(rounds, dtype=int), matrix


def load_group_histories(config: dict[str, Any]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    fig2 = config["fig2"]
    history_root = resolve_path(fig2["history_root"])
    seeds = [int(seed) for seed in fig2["seeds"]]
    group_histories: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for group in fig2["groups"]:
        group_key = group["key"]
        histories: list[tuple[int, dict[str, Any]]] = []
        for seed in seeds:
            path = history_root / f"seed_{seed}" / group_key / "training_history.json"
            histories.append((seed, load_history(path)))
        group_histories[group_key] = histories
    return group_histories


def compute_curves(
    config: dict[str, Any],
    group_histories: dict[str, list[tuple[int, dict[str, Any]]]],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    expected_seed_count = len(config["fig2"]["seeds"])
    all_group_keys = [group["key"] for group in config["fig2"]["groups"]]

    for group in config["fig2"]["groups"]:
        group_key = group["key"]
        curves[group_key] = {}
    for metric in config["fig2"]["metrics"]:
        metric_group_keys = metric.get("groups", all_group_keys)
        for group_key in metric_group_keys:
            histories = group_histories[group_key]
            if len(histories) != expected_seed_count:
                raise ValueError(f"{group_key} has {len(histories)} seeds, expected {expected_seed_count}")
            rounds, matrix = align_seed_series(histories, metric)
            if matrix.shape[0] != expected_seed_count:
                raise ValueError(f"{group_key}:{metric['name']} seed count mismatch")
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=int(config["std_ddof"]))
            curves[group_key][metric["name"]] = (rounds, mean, std)
    return curves


def write_summary_csv(
    path: Path,
    config: dict[str, Any],
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "group", "round", "mean", "sample_std", "seed_count"])
        seed_count = len(config["fig2"]["seeds"])
        all_group_keys = [group["key"] for group in config["fig2"]["groups"]]
        for metric in config["fig2"]["metrics"]:
            metric_name = metric["name"]
            metric_group_keys = set(metric.get("groups", all_group_keys))
            for group in config["fig2"]["groups"]:
                group_key = group["key"]
                if group_key not in metric_group_keys:
                    continue
                rounds, mean, std = curves[group_key][metric_name]
                for round_id, mean_value, std_value in zip(rounds, mean, std):
                    writer.writerow(
                        [
                            metric_name,
                            group_key,
                            int(round_id),
                            f"{mean_value:.10f}",
                            f"{std_value:.10f}",
                            seed_count,
                        ]
                    )


def draw_figure(
    config: dict[str, Any],
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    output_dir: Path,
) -> list[Path]:
    fig2 = config["fig2"]
    plt.rcParams.update(config["matplotlib_rc"])

    fig, axes = plt.subplots(2, 2, figsize=tuple(fig2["figsize"]), dpi=int(fig2["dpi"]))
    axes_by_metric = {metric["name"]: ax for metric, ax in zip(fig2["metrics"], axes.flat)}
    fig.patch.set_facecolor("#ffffff")

    for metric in fig2["metrics"]:
        ax = axes_by_metric[metric["name"]]
        ax.set_facecolor("#ffffff")
        all_group_keys = [group["key"] for group in fig2["groups"]]
        metric_group_keys = set(metric.get("groups", all_group_keys))
        for group in fig2["groups"]:
            group_key = group["key"]
            if group_key not in metric_group_keys:
                continue
            rounds, mean, std = curves[group_key][metric["name"]]
            color = group["color"]
            ax.plot(rounds, mean, color=color, linewidth=2.0, label=group["label"])
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)

        ax.set_title(metric["title"], fontsize=10, weight="semibold")
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel(metric["ylabel"], fontsize=9)
        ax.grid(True, color="#e4e1dc", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(fig2["title"], fontsize=11, weight="semibold", y=0.985)
    fig.text(
        0.5,
        0.002,
        "Solid lines show the three-seed mean; shaded bands show +/-1 sample standard deviation.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), w_pad=1.6, h_pad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = fig2["output_basename"]
    paths = [output_dir / f"{basename}.{ext}" for ext in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_dir"])
    group_histories = load_group_histories(config)
    curves = compute_curves(config, group_histories)
    figure_paths = draw_figure(config, curves, output_dir)
    summary_path = output_dir / f"{config['fig2']['output_basename']}_summary.csv"
    write_summary_csv(summary_path, config, curves)

    for path in figure_paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
