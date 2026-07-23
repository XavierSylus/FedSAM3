#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ["A", "B", "C", "D"]
COLORS = {
    "A": "#8b8f97",
    "B": "#4f7fa7",
    "C": "#2f6f8f",
    "D": "#b45f4d",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = (
        Path.home()
        / "Desktop"
        / "paper_data"
        / "0424\u4e3b\u7ed3\u679c\u8bf4\u660e\u56fe"
    )
    parser = argparse.ArgumentParser(
        description="Reproduce table4_endpoint_comparison from source zip files."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "figure_repro_outputs")
    return parser.parse_args()


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing source zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as bundle:
        text = bundle.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def build_endpoint_values(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abc_rows = read_csv_from_zip(
        data_dir / "A_B_C.zip",
        "A_B_C/server_data/main_table_group_summary.csv",
    )
    abc_by_group = {row["group"]: row for row in abc_rows}

    dice_mean = [
        float(abc_by_group["group_a"]["final_dice_mean"]),
        float(abc_by_group["group_b"]["final_dice_mean"]),
        float(abc_by_group["group_c"]["final_dice_mean"]),
    ]
    dice_std = [
        float(abc_by_group["group_a"]["final_dice_std"]),
        float(abc_by_group["group_b"]["final_dice_std"]),
        float(abc_by_group["group_c"]["final_dice_std"]),
    ]
    hd95_mean = [
        float(abc_by_group["group_a"]["final_hd95_mean"]),
        float(abc_by_group["group_b"]["final_hd95_mean"]),
        float(abc_by_group["group_c"]["final_hd95_mean"]),
    ]
    hd95_std = [
        float(abc_by_group["group_a"]["final_hd95_std"]),
        float(abc_by_group["group_b"]["final_hd95_std"]),
        float(abc_by_group["group_c"]["final_hd95_std"]),
    ]

    d_rows = read_csv_from_zip(
        data_dir / "fedprox_d_paper.zip",
        "fedprox_d_paper/fedprox_d_by_seed.csv",
    )
    d_final_dice = np.array([float(row["final_dice"]) for row in d_rows], dtype=float)
    d_final_hd95 = np.array([float(row["final_hd95"]) for row in d_rows], dtype=float)

    dice_mean.append(float(d_final_dice.mean()))
    dice_std.append(float(d_final_dice.std()))
    hd95_mean.append(float(d_final_hd95.mean()))
    hd95_std.append(float(d_final_hd95.std()))

    return (
        np.round(np.array(dice_mean, dtype=float), 4),
        np.round(np.array(dice_std, dtype=float), 4),
        np.round(np.array(hd95_mean, dtype=float), 3),
        np.round(np.array(hd95_std, dtype=float), 3),
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
    values: np.ndarray,
    stds: np.ndarray,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    value_fmt: str,
    label_offset: float,
) -> None:
    x = np.arange(len(GROUPS))
    ax.bar(
        x,
        values,
        yerr=stds,
        color=[COLORS[group] for group in GROUPS],
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
    )
    ax.set_title(title, fontsize=10, pad=8, weight="semibold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e6e1d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a4a4a")
    ax.spines["bottom"].set_color("#4a4a4a")
    ax.tick_params(axis="y", labelsize=8)
    add_value_labels(ax, values, stds, value_fmt, label_offset)


def main() -> None:
    args = parse_args()
    final_dice_mean, final_dice_std, final_hd95_mean, final_hd95_std = build_endpoint_values(args.data_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.95), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    draw_panel(
        axes[0],
        final_dice_mean,
        final_dice_std,
        "Final Dice",
        "Endpoint overlap",
        (0.80, 0.905),
        "{:.4f}",
        0.003,
    )
    draw_panel(
        axes[1],
        final_hd95_mean,
        final_hd95_std,
        "Final HD95 (mm)",
        "Endpoint boundary error",
        (8.0, 13.2),
        "{:.3f}",
        0.12,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["A"], label="A: image-only FedAvg"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["B"], label="B: hetero FedAvg"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["C"], label="C: hetero FedAvg + restricted routing"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["D"], label="D: hetero FedProx"),
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
    fig.suptitle(
        "Main endpoint comparison across A/B/C/D protocols",
        fontsize=11,
        weight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.005,
        "Mean +/- std from Table 4. Higher Dice is better; lower HD95 is better.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.20, 1, 0.94), w_pad=1.8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "table4_endpoint_comparison.png"
    pdf_path = args.output_dir / "table4_endpoint_comparison.pdf"
    svg_path = args.output_dir / "table4_endpoint_comparison.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()
