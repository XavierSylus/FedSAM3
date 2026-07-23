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
        description="Generate Fig. 4 endpoint comparison from final Table 3 seed-level CSV data."
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, str]], metric: str, group_key: str) -> np.ndarray:
    values = np.array([float(row[metric]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{group_key}:{metric} contains non-finite values")
    return values


def require_seed_set(rows: list[dict[str, str]], expected_seeds: list[int], group_key: str) -> None:
    actual = sorted(int(row["seed"]) for row in rows)
    expected = sorted(expected_seeds)
    if actual != expected:
        raise ValueError(f"{group_key} seed mismatch: expected {expected}, got {actual}")


def load_group_rows(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    fig4 = config["fig4"]
    main_rows = read_csv_rows(resolve_path(fig4["table3_main_by_seed"]))
    d_rows = read_csv_rows(resolve_path(fig4["table3_fedprox_d_by_seed"]))
    expected_seeds = [int(seed) for seed in fig4["seeds"]]

    rows_by_group: dict[str, list[dict[str, str]]] = {}
    for group in fig4["groups"]:
        key = group["key"]
        if group["source"] == "table3_main":
            source_group = group["source_group"]
            rows = [row for row in main_rows if row["group"] == source_group]
        elif group["source"] == "table3_fedprox_d":
            rows = list(d_rows)
        else:
            raise ValueError(f"Unknown Fig. 4 source for {key}: {group['source']}")

        require_seed_set(rows, expected_seeds, key)
        rows_by_group[key] = rows
    return rows_by_group


def compute_endpoint_values(
    config: dict[str, Any],
    rows_by_group: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, float]]:
    std_ddof = int(config["std_ddof"])
    stats: dict[str, dict[str, float]] = {}

    for group in config["fig4"]["groups"]:
        key = group["key"]
        rows = rows_by_group[key]
        stats[key] = {}
        for metric in config["fig4"]["metrics"]:
            values = finite_values(rows, metric["column"], key)
            stats[key][f"{metric['name']}_mean"] = float(values.mean())
            stats[key][f"{metric['name']}_std"] = float(values.std(ddof=std_ddof))
            stats[key][f"{metric['name']}_n"] = float(values.size)
    return stats


def write_summary_csv(path: Path, config: dict[str, Any], stats: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "group", "mean", "sample_std", "seed_count"])
        for metric in config["fig4"]["metrics"]:
            metric_name = metric["name"]
            for group in config["fig4"]["groups"]:
                key = group["key"]
                writer.writerow(
                    [
                        metric_name,
                        key,
                        f"{stats[key][f'{metric_name}_mean']:.10f}",
                        f"{stats[key][f'{metric_name}_std']:.10f}",
                        int(stats[key][f"{metric_name}_n"]),
                    ]
                )


def add_value_labels(ax: plt.Axes, values: np.ndarray, stds: np.ndarray, fmt: str, offset: float) -> None:
    for idx, (value, std) in enumerate(zip(values, stds)):
        ax.text(
            idx,
            value + std + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2b2b2b",
        )


def draw_panel(
    ax: plt.Axes,
    groups: list[dict[str, str]],
    values: np.ndarray,
    stds: np.ndarray,
    metric: dict[str, Any],
) -> None:
    x = np.arange(len(groups))
    ax.bar(
        x,
        values,
        yerr=stds,
        color=[group["color"] for group in groups],
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
    )
    ax.set_title(metric["title"], fontsize=10, pad=8, weight="semibold")
    ax.set_ylabel(metric["ylabel"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([group["key"] for group in groups], fontsize=9)
    ax.set_ylim(*metric["ylim"])
    ax.grid(axis="y", color="#e6e1d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a4a4a")
    ax.spines["bottom"].set_color("#4a4a4a")
    ax.tick_params(axis="y", labelsize=8)
    add_value_labels(ax, values, stds, metric["value_format"], float(metric["label_offset"]))


def draw_figure(
    config: dict[str, Any],
    stats: dict[str, dict[str, float]],
    output_dir: Path,
) -> list[Path]:
    fig4 = config["fig4"]
    groups = fig4["groups"]
    plt.rcParams.update(config["matplotlib_rc"])

    fig, axes = plt.subplots(1, 2, figsize=tuple(fig4["figsize"]), dpi=int(fig4["dpi"]))
    fig.patch.set_facecolor("#ffffff")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    for ax, metric in zip(axes, fig4["metrics"]):
        metric_name = metric["name"]
        means = np.array([stats[group["key"]][f"{metric_name}_mean"] for group in groups], dtype=float)
        stds = np.array([stats[group["key"]][f"{metric_name}_std"] for group in groups], dtype=float)
        draw_panel(ax, groups, means, stds, metric)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group["color"], label=group["label"])
        for group in groups
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.suptitle(fig4["title"], fontsize=11, weight="semibold", y=0.98)
    fig.text(
        0.5,
        0.005,
        "Mean +/- sample std computed from final Table 3 seed-level rows. Higher Dice is better; lower HD95 is better.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.20, 1, 0.94), w_pad=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = fig4["output_basename"]
    paths = [output_dir / f"{basename}.{ext}" for ext in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_dir"])
    rows_by_group = load_group_rows(config)
    stats = compute_endpoint_values(config, rows_by_group)
    figure_paths = draw_figure(config, stats, output_dir)
    summary_path = output_dir / f"{config['fig4']['output_basename']}_summary.csv"
    write_summary_csv(summary_path, config, stats)

    for path in figure_paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
