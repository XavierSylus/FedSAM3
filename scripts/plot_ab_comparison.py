#!/usr/bin/env python3
"""
FedSAM3-Cream · Group A vs Group B 对比图生成脚本
===================================================
从两组 training_history.json 生成论文对比图：
  1. Dice / HD95 最终指标柱状图（论文核心对比图）
  2. Dice 收敛曲线对比（A vs B 叠加）
  3. Seg Loss 收敛曲线对比（A vs B 叠加）
  4. 梯度冲突角曲线（Group B 专属）

用法：
    python scripts/plot_ab_comparison.py
    python scripts/plot_ab_comparison.py \
        --history_a logs/group_a/checkpoints/training_history.json \
        --history_b logs/group_b/checkpoints/training_history.json \
        --out_dir results/ab_comparison_figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.labelsize": 12,
    "font.size":      12,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_A      = "#4C72B0"   # Group A — 蓝
C_B      = "#DD4444"   # Group B — 红
C_REFS   = "gray"


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_val_series(history: dict, metric_key: str):
    """Support both new schema (val_metrics) and legacy flat keys (val_dice/val_hd95)."""
    val_metrics = history.get("val_metrics", [])
    if isinstance(val_metrics, list) and len(val_metrics) > 0:
        rounds = [int(m.get("round", i + 1)) for i, m in enumerate(val_metrics)]
        values = [m.get(metric_key, None) for m in val_metrics]
        return rounds, values

    legacy_key = f"val_{metric_key}"
    values = history.get(legacy_key, [])
    rounds = history.get("rounds", list(range(1, len(values) + 1)))
    return rounds[:len(values)], values


def _get_train_loss_series(history: dict):
    """Support both new schema (avg_losses) and legacy key (train_loss)."""
    values = history.get("avg_losses", history.get("train_loss", []))
    rounds = history.get("rounds", list(range(1, len(values) + 1)))
    return rounds[:len(values)], values


def _best_dice(history: dict) -> float:
    _, vals = _get_val_series(history, "dice")
    vals = [float(v) for v in vals if v is not None]
    return float(max(vals)) if vals else 0.0


def _final_hd95(history: dict) -> float:
    _, vals = _get_val_series(history, "hd95")
    valid = [float(v) for v in vals if v is not None and np.isfinite(v) and v > 0]
    return float(np.mean(valid[-5:])) if valid else None


def plot_bar_comparison(ha: dict, hb: dict, out_dir: Path,
                        manual_a_dice: float = None, manual_a_hd95: float = None):
    """Dice / HD95 最终指标柱状图"""
    dice_a = manual_a_dice if manual_a_dice is not None else _best_dice(ha)
    dice_b = _best_dice(hb)
    hd_a   = manual_a_hd95 if manual_a_hd95 is not None else _final_hd95(ha)
    hd_b   = _final_hd95(hb)

    has_hd = hd_a is not None and hd_b is not None
    ncols  = 2 if has_hd else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    ax1 = axes[0] if has_hd else axes

    # --- Dice ---
    bars = ax1.bar(["Group A\n(Image-only FedAvg)", "Group B\n(Heterogeneous FedAvg)"],
                   [dice_a, dice_b], color=[C_A, C_B], width=0.45, zorder=3)
    for bar, val in zip(bars, [dice_a, dice_b]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Best Dice Coefficient")
    ax1.set_title("Dice Score Comparison", fontweight="bold")
    lo = min(dice_a, dice_b) - 0.005
    hi = max(dice_a, dice_b) + 0.005
    if hi > lo:
        ax1.set_ylim(lo, hi)

    # --- HD95（仅当数据可用时）---
    if has_hd:
        ax2 = axes[1]
        bars2 = ax2.bar(["Group A\n(Image-only FedAvg)", "Group B\n(Heterogeneous FedAvg)"],
                        [hd_a, hd_b], color=[C_A, C_B], width=0.45, zorder=3)
        for bar, val in zip(bars2, [hd_a, hd_b]):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f"{val:.2f} mm", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax2.set_ylabel("HD95 (mm)  ↓ Lower is Better")
        ax2.set_title("HD95 Comparison\n(Gradient Conflict Impact)", fontweight="bold")
        ax2.set_ylim(0, max(hd_a, hd_b) * 1.35)

        delta = hd_b - hd_a
        pct   = f"+{delta/hd_a*100:.1f}%" if hd_a > 0 else "N/A"
        ax2.annotate(f"Δ = +{delta:.2f} mm\n({pct})",
                     xy=(1, hd_b), xytext=(1.35, (hd_a + hd_b) / 2),
                     fontsize=10, color=C_B,
                     arrowprops=dict(arrowstyle="->", color=C_B, lw=1.5))

    fig.suptitle("Group A vs Group B · Final Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "ab_bar_comparison.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ 柱状对比图 → {out}")


def plot_convergence(ha: dict, hb: dict, out_dir: Path):
    """Dice + Seg Loss 收敛曲线对比"""
    val_rounds_a, dice_a = _get_val_series(ha, "dice")
    val_rounds_b, dice_b = _get_val_series(hb, "dice")
    loss_rounds_a, loss_a = _get_train_loss_series(ha)
    loss_rounds_b, loss_b = _get_train_loss_series(hb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if dice_a:
        ax1.plot(val_rounds_a[:len(dice_a)], dice_a, color=C_A, linewidth=1.8,
                 label="Group A (Image-only FedAvg)")
    if dice_b:
        ax1.plot(val_rounds_b[:len(dice_b)], dice_b, color=C_B, linewidth=1.8,
                 linestyle="--", label="Group B (Heterogeneous FedAvg)")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Validation Dice")
    ax1.set_title("Dice Convergence", fontweight="bold")
    ax1.legend(fontsize=9)

    if loss_a:
        ax2.plot(loss_rounds_a[:len(loss_a)], loss_a, color=C_A, linewidth=1.8,
                 label="Group A")
    if loss_b:
        ax2.plot(loss_rounds_b[:len(loss_b)], loss_b, color=C_B, linewidth=1.8,
                 linestyle="--", label="Group B")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Training Loss (Seg)")
    ax2.set_title("Segmentation Loss Convergence", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Group A vs Group B · Training Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "ab_convergence.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ 收敛曲线图 → {out}")


def plot_grad_conflict_b(hb: dict, out_dir: Path):
    """Group B 梯度冲突角 60 轮曲线图"""
    rounds_b = hb.get("rounds", [])
    conf = hb.get("grad_conflict_deg", [])
    valid = [(r, v) for r, v in zip(rounds_b, conf) if v is not None]
    if not valid:
        print("[SKIP] grad_conflict_deg 为空，跳过梯度冲突曲线。")
        return
    rs, vs = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rs, vs, color=C_B, linewidth=2.0, marker="D", markersize=4,
            label="Image-only vs Multimodal (Adapter)")
    ax.axhline(90, color=C_REFS, linestyle="--", linewidth=1.2,
               label="90° — Orthogonal (Zero Gradient Alignment)")
    mean_val = float(np.mean(vs))
    ax.axhline(mean_val, color=C_B, linestyle=":", linewidth=1.2,
               label=f"Mean = {mean_val:.1f}°")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Gradient Conflict Angle (°)")
    ax.set_title("Cross-Modal Gradient Conflict in Heterogeneous FedAvg\n"
                 "(Group B · Adapter Parameters)", fontweight="bold")
    ax.set_ylim(0, 190)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / "group_b_grad_conflict.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ 梯度冲突曲线 → {out}")


def main():
    parser = argparse.ArgumentParser(description="Group A vs B 对比图")
    parser.add_argument("--history_a",
                        default="logs/group_a/checkpoints/training_history.json")
    parser.add_argument("--history_b",
                        default="logs/group_b/checkpoints/training_history.json")
    parser.add_argument("--out_dir", default="results/ab_comparison_figures")
    parser.add_argument("--manual_a_dice", type=float, default=None,
                        help="手动指定 Group A 最佳 Dice（当 history_a 缺少该字段时使用）")
    parser.add_argument("--manual_a_hd95", type=float, default=None,
                        help="手动指定 Group A HD95（当 history_a 缺少该字段时使用）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ha = _load(args.history_a)
    hb = _load(args.history_b)

    print(f"▶ Group A: {len(ha.get('rounds', []))} 轮记录")
    print(f"▶ Group B: {len(hb.get('rounds', []))} 轮记录\n")
    print("▶ 生成对比图表 ...")

    plot_bar_comparison(ha, hb, out_dir,
                        manual_a_dice=args.manual_a_dice,
                        manual_a_hd95=args.manual_a_hd95)
    plot_convergence(ha, hb, out_dir)
    plot_grad_conflict_b(hb, out_dir)

    print(f"\n✅ 全部图表已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
