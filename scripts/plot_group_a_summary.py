#!/usr/bin/env python3
"""
FedSAM3-Cream · Group A 训练收尾三联图
======================================
从 training_history.json 生成三联图供论文直接使用：
  子图1：Segmentation Loss 收敛曲线
  子图2：Volume-level Dice 曲线
  子图3：HD95 曲线（过滤 inf 点）

用法：
    # 使用默认路径
    python scripts/plot_group_a_summary.py

    # 手动指定 history 文件和输出目录
    python scripts/plot_group_a_summary.py \\
        --history data/checkpoints/training_history.json \\
        --out_dir results/paper_figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.labelsize": 11,
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.35,
})

C_BLUE   = "#4C72B0"
C_GREEN  = "#55A868"
C_RED    = "#DD4444"


def _load(history_path: Path) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_val_series(val_metrics: list, key: str) -> tuple[list, list]:
    """从 val_metrics 列表中提取 (rounds, values)，自动跳过 inf/NaN。"""
    rounds, values = [], []
    for m in val_metrics:
        v = m.get(key)
        if v is None:
            continue
        if key == "hd95" and (v == float("inf") or (isinstance(v, float) and np.isnan(v))):
            continue
        rounds.append(m["round"])
        values.append(v)
    return rounds, values


def plot_group_a_summary(history_path: Path, out_dir: Path) -> None:
    history = _load(history_path)

    rounds       = history.get("rounds", [])
    seg_losses   = history.get("avg_seg_losses", history.get("avg_losses", []))
    val_metrics  = history.get("val_metrics", [])

    if not rounds:
        print("❌ training_history.json 中 rounds 为空，无法绘图。")
        sys.exit(1)

    dice_rounds, dice_vals   = _extract_val_series(val_metrics, "dice")
    hd95_rounds, hd95_vals   = _extract_val_series(val_metrics, "hd95")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── 子图1：Seg Loss ──
    ax = axes[0]
    ax.plot(rounds, seg_losses[:len(rounds)], color=C_BLUE, linewidth=1.8, marker="o", markersize=2)
    ax.set_title("Segmentation Loss", fontweight="bold")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")

    # ── 子图2：Dice ──
    ax = axes[1]
    if dice_vals:
        ax.plot(dice_rounds, dice_vals, color=C_GREEN, linewidth=1.8, marker="o", markersize=3)
        best_dice = max(dice_vals)
        best_round = dice_rounds[dice_vals.index(best_dice)]
        ax.axhline(best_dice, color=C_RED, linestyle="--", linewidth=1.0,
                   label=f"Best={best_dice:.4f} (R{best_round})")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Dice data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Validation Dice (Volume-level)", fontweight="bold")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Dice Coefficient")
    ax.set_ylim(0, 1.05)

    # ── 子图3：HD95 ──
    ax = axes[2]
    if hd95_vals:
        ax.plot(hd95_rounds, hd95_vals, color=C_RED, linewidth=1.8, marker="x", markersize=4)
        best_hd = min(hd95_vals)
        best_r  = hd95_rounds[hd95_vals.index(best_hd)]
        ax.axhline(best_hd, color=C_BLUE, linestyle="--", linewidth=1.0,
                   label=f"Best={best_hd:.2f}mm (R{best_r})")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No valid HD95 data\n(all inf →空切片门控生效)", ha="center",
                va="center", transform=ax.transAxes, fontsize=9)
    ax.set_title("Validation HD95 (non-empty slices only)", fontweight="bold")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("HD95 (mm)")

    fig.suptitle("Group A · ImageOnly Baseline · Training Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "group_a_summary.pdf"
    out_png = out_dir / "group_a_summary.png"
    fig.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ PDF → {out_pdf}")
    print(f"✅ PNG → {out_png}")

    # 终端打印关键数字
    if dice_vals:
        print(f"\n📊 Group A 关键指标摘要：")
        print(f"   最佳 Val Dice  = {max(dice_vals):.4f}（第 {dice_rounds[dice_vals.index(max(dice_vals))]} 轮）")
        print(f"   最终 Val Dice  = {dice_vals[-1]:.4f}")
    if hd95_vals:
        print(f"   最佳 HD95      = {min(hd95_vals):.2f} mm（第 {hd95_rounds[hd95_vals.index(min(hd95_vals))]} 轮）")
        print(f"   最终 HD95      = {hd95_vals[-1]:.2f} mm")
    if seg_losses:
        print(f"   最终 Seg Loss  = {seg_losses[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Group A 训练三联收尾图")
    parser.add_argument(
        "--history",
        default="data/checkpoints/training_history.json",
        help="training_history.json 路径（默认: data/checkpoints/training_history.json）"
    )
    parser.add_argument(
        "--out_dir",
        default="results/paper_figures",
        help="输出目录（默认: results/paper_figures）"
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"❌ 未找到文件: {history_path}")
        print("   请先完成 Group A 训练，或通过 --history 指定正确路径。")
        sys.exit(1)

    plot_group_a_summary(history_path, Path(args.out_dir))


if __name__ == "__main__":
    main()
