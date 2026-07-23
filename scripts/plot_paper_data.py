#!/usr/bin/env python3
"""
FedSAM3-Cream · 论文数据可视化脚本
=====================================
从 training_history.json 中读取训练过程记录，
一键生成论文 Methodology 章节所需的四类图表：

  1. LR 调度曲线（3.4 节 · 训练流程）
  2. GPU 显存峰值曲线（3.4 节 · CPU-Offload 效果）
  3. 每轮耗时曲线（3.4 节 · 串行训练效率）
  4. 梯度冲突余弦角曲线（3.1 节 · 梯度冲突问题，Group A 无数据则跳过）

用法：
    python scripts/plot_paper_data.py
    python scripts/plot_paper_data.py --history data/federated_split/checkpoints/training_history.json
    python scripts/plot_paper_data.py --out_dir results/paper_figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.labelsize": 11,
    "font.size":      11,
    "axes.grid":      True,
    "grid.alpha":     0.35,
})

# ────────────────────────────────────────────
#  颜色常量（顶会学术配色）
# ────────────────────────────────────────────
C_BLUE   = "#4C72B0"
C_RED    = "#DD4444"
C_GREEN  = "#55A868"
C_ORANGE = "#E8A838"


def _load_history(history_path: str) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_lr_curve(rounds, lr_history, out_dir: Path):
    """LR 调度曲线"""
    if not lr_history:
        print("[SKIP] lr_history 为空，跳过 LR 曲线绘制。")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rounds, lr_history, color=C_BLUE, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Cosine Annealing + Warmup)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    out = out_dir / "paper_lr_schedule.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ LR 曲线 → {out}")


def plot_gpu_memory(rounds, gpu_mem_mb, out_dir: Path):
    """GPU 显存峰值曲线"""
    if not gpu_mem_mb or all(v == 0 for v in gpu_mem_mb):
        print("[SKIP] gpu_mem_mb 全为 0（可能是 CPU 训练环境），跳过显存曲线。")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rounds, gpu_mem_mb, color=C_GREEN, linewidth=1.8, marker="s", markersize=3)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory per Round (CPU-Offload + Serial Training)")
    plt.tight_layout()
    out = out_dir / "paper_gpu_memory.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ GPU 显存曲线 → {out}")


def plot_round_time(rounds, round_time_sec, out_dir: Path):
    """每轮耗时曲线"""
    if not round_time_sec:
        print("[SKIP] round_time_sec 为空，跳过耗时曲线。")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rounds, round_time_sec, color=C_ORANGE, linewidth=1.8, marker="^", markersize=3)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Time per Round (sec)")
    ax.set_title("Training Time per Communication Round")
    # 标注均值
    mean_t = float(np.mean(round_time_sec))
    ax.axhline(mean_t, color=C_RED, linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_t:.1f}s")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / "paper_round_time.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 耗时曲线 → {out}")


def plot_grad_conflict(rounds, grad_conflict_deg, out_dir: Path):
    """梯度冲突余弦角曲线（Group A 无多模态时自动跳过）"""
    valid = [(r, v) for r, v in zip(rounds, grad_conflict_deg) if v is not None]
    if not valid:
        print("[SKIP] grad_conflict_deg 全为 None（Group A 单模态实验），跳过梯度冲突曲线。")
        return
    rs, vs = zip(*valid)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rs, vs, color=C_RED, linewidth=1.8, marker="D", markersize=4)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1.0, label="90° (Orthogonal)")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Gradient Conflict Angle (°)")
    ax.set_title("Gradient Conflict between Image-only & Multimodal Clients\n"
                 "(Adapter Parameters, Cosine Angle)")
    ax.set_ylim(0, 185)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / "paper_grad_conflict.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 梯度冲突曲线 → {out}")


def plot_combined(rounds, history: dict, out_dir: Path):
    """生成 2×2 合并大图（方便直接插入论文）"""
    lr_ok  = bool(history.get("lr_history"))
    mem_ok = bool(history.get("gpu_mem_mb")) and any(v > 0 for v in history["gpu_mem_mb"])
    time_ok = bool(history.get("round_time_sec"))
    conf_ok = any(v is not None for v in history.get("grad_conflict_deg", []))

    panels = []
    if lr_ok:   panels.append(("LR Schedule",          history["lr_history"],       "LR",     C_BLUE,   "o"))
    if mem_ok:  panels.append(("Peak GPU Memory (MB)", history["gpu_mem_mb"],       "MB",     C_GREEN,  "s"))
    if time_ok: panels.append(("Time / Round (sec)",   history["round_time_sec"],   "sec",    C_ORANGE, "^"))
    if conf_ok:
        conf_vals = [v if v is not None else float("nan")
                     for v in history["grad_conflict_deg"]]
        panels.append(("Grad Conflict (°)", conf_vals, "Degree", C_RED, "D"))

    if not panels:
        print("[SKIP] 无任何论文数据字段可绘图。")
        return

    n = len(panels)
    ncols = min(2, n)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows))
    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for i, (title, data, ylabel, color, marker) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        ax.plot(rounds[:len(data)], data[:len(rounds)],
                color=color, linewidth=1.8, marker=marker, markersize=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(ylabel)

    # 隐藏多余子图
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("FedSAM3-Cream · Training Dynamics (Paper Data)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "paper_combined.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ 合并大图    → {out}")


def main():
    parser = argparse.ArgumentParser(description="FedSAM3-Cream 论文数据可视化")
    parser.add_argument(
        "--history",
        default="data/federated_split/checkpoints/training_history.json",
        help="training_history.json 路径"
    )
    parser.add_argument(
        "--out_dir",
        default="results/paper_figures",
        help="输出目录（默认: results/paper_figures）"
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"❌ 未找到 training_history.json: {history_path}")
        print("   请先运行训练后再执行此脚本。")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ 加载训练历史: {history_path}")
    history = _load_history(str(history_path))

    rounds = history.get("rounds", [])
    if not rounds:
        print("❌ training_history.json 中 rounds 为空，无法绘图。")
        sys.exit(1)

    print(f"  共 {len(rounds)} 轮记录 (Round {rounds[0]} ~ {rounds[-1]})\n")
    print("▶ 生成论文数据图表 ...")

    plot_lr_curve(rounds, history.get("lr_history", []), out_dir)
    plot_gpu_memory(rounds, history.get("gpu_mem_mb", []), out_dir)
    plot_round_time(rounds, history.get("round_time_sec", []), out_dir)
    plot_grad_conflict(rounds, history.get("grad_conflict_deg", []), out_dir)
    plot_combined(rounds, history, out_dir)

    print(f"\n✅ 全部图表已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
