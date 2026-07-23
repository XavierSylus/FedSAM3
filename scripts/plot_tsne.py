#!/usr/bin/env python3
"""
FedSAM3-Cream · t-SNE 特征对齐可视化脚本
========================================
用途：
    从多个训练轮次的 checkpoint 中提取全局/客户端局部特征表示，
    使用 sklearn t-SNE 降维后绘制 1×N 并排子图，可视化随联邦训练
    推进，异构模态特征从"分散"逐渐"向全局锚点聚拢"的动态演变。

核心创新展示：
    text-only 客户端 (client_1) 的局部表示通过 Decoupled Contrastive
    Aggregation 机制，在训练后期与 image-only 客户端 (client_2) 共同
    向全局语义锚点 (global_rep) 对齐，证明跨模态知识传递的有效性。

用法示例：
    # 基本用法（传入三个轮次的 checkpoint）
    python scripts/plot_tsne.py \\
        --checkpoints results/checkpoint_round_1.pth \\
                      results/checkpoint_round_25.pth \\
                      results/checkpoint_round_50.pth \\
        --labels "Round 1" "Round 25" "Round 50" \\
        --output results/figures/tsne_alignment.pdf

    # 使用 use_dummy 生成演示图（无需真实 checkpoint）
    python scripts/plot_tsne.py --demo

依赖：
    pip install torch scikit-learn matplotlib seaborn

版本：v1.0  |  作者：FedSAM3-Cream Team  |  日期：2026-03
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")           # 无 GUI 后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch 未安装，仅 --demo 模式可用。")

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    raise ImportError("请先安装 scikit-learn：pip install scikit-learn")


# ──────────────────────────────────────────────
#  配色方案（顶会级学术配色）
# ──────────────────────────────────────────────
PALETTE = {
    # 全局锚点：金色 ★ 星形标记
    "global_text":   "#E8A838",   # 暖金——text 全局锚点
    "global_image":  "#E85D38",   # 橘红——image 全局锚点
    # 客户端局部：蓝紫系
    "text_only":     "#4C72B0",   # 深蓝——text-only 客户端
    "image_only":    "#DD8452",   # 橘棕——image-only 客户端
    "multimodal":    "#55A868",   # 翠绿——multimodal 客户端
    # 背景
    "bg":            "#FAFAFA",
    "grid":          "#E8E8E8",
}

MODALITY_CONFIG = {
    # client_id 关键字 → (显示名, 颜色, 标记)
    "text":        ("Text-only Clients",    PALETTE["text_only"],   "o"),
    "image":       ("Image-only Clients",   PALETTE["image_only"],  "s"),
    "multimodal":  ("Multimodal Clients",   PALETTE["multimodal"],  "^"),
    "global_text": ("Global Text Anchor",   PALETTE["global_text"], "*"),
    "global_image":("Global Image Anchor",  PALETTE["global_image"],"*"),
}


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def _tensor_to_numpy(t) -> np.ndarray:
    """安全地将 Tensor/ndarray 转换为 float32 ndarray。"""
    if HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.array(t, dtype=np.float32)


def _infer_modality(client_id: str) -> str:
    """根据 client_id 推断客户端模态类型。"""
    cid = str(client_id).lower()
    if "text" in cid:
        return "text"
    elif "image" in cid:
        return "image"
    else:
        return "multimodal"


def load_checkpoint_features(
    ckpt_path: str,
) -> Dict[str, np.ndarray]:
    """
    从单个 checkpoint 加载所有特征，返回字典：
        {
            "global_text":   ndarray (D,) or (1, D),
            "global_image":  ndarray (D,) or (1, D),
            "text":          ndarray (N_text, D),   # text-only 客户端
            "image":         ndarray (N_img, D),    # image-only 客户端
            "multimodal":    ndarray (N_mm, D),     # multimodal 客户端
        }
    """
    if not HAS_TORCH:
        raise RuntimeError("加载 checkpoint 需要 PyTorch。")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    features: Dict[str, list] = {
        "global_text": [],
        "global_image": [],
        "text": [],
        "image": [],
        "multimodal": [],
    }

    # ── 全局锚点 ──
    if "global_text_rep" in ckpt and ckpt["global_text_rep"] is not None:
        gtr = _tensor_to_numpy(ckpt["global_text_rep"])
        features["global_text"].append(gtr.reshape(1, -1) if gtr.ndim == 1 else gtr)

    if "global_image_rep" in ckpt and ckpt["global_image_rep"] is not None:
        gir = _tensor_to_numpy(ckpt["global_image_rep"])
        features["global_image"].append(gir.reshape(1, -1) if gir.ndim == 1 else gir)

    # ── 客户端局部表示 ──
    client_states: dict = ckpt.get("client_states", {})
    for client_id, state in client_states.items():
        modality = _infer_modality(client_id)

        # 兼容两种存储格式
        local_rep = None
        if isinstance(state, dict):
            local_rep = state.get("local_reps", state.get("local_rep", None))
        elif HAS_TORCH and isinstance(state, torch.Tensor):
            local_rep = state

        if local_rep is None:
            continue

        arr = _tensor_to_numpy(local_rep)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        features[modality].append(arr)

    # 拼接同模态的所有样本
    result: Dict[str, np.ndarray] = {}
    for key, arrs in features.items():
        if arrs:
            result[key] = np.concatenate(arrs, axis=0)

    if not result:
        raise ValueError(
            f"Checkpoint {ckpt_path} 中未找到任何特征数据。\n"
            "请确认 checkpoint 包含以下键：\n"
            "  global_text_rep, global_image_rep, client_states"
        )

    return result


def build_tsne_input(
    feature_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    将特征字典拼接为 t-SNE 输入矩阵。

    Returns:
        X:        (N_total, D) 特征矩阵
        labels:   (N_total,)  int 标签索引
        types:    list of str，每类的名称
    """
    type_keys = [k for k in MODALITY_CONFIG.keys() if k in feature_dict]
    X_parts, y_parts = [], []

    for idx, key in enumerate(type_keys):
        arr = feature_dict[key]
        X_parts.append(arr)
        y_parts.append(np.full(arr.shape[0], idx, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y, type_keys


def compute_tsne(
    X: np.ndarray,
    perplexity: float = 25.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """对输入矩阵做标准化后运行 t-SNE，返回 (N, 2) 嵌入。"""
    # 标准化——对高维特征有稳定性帮助
    X_scaled = StandardScaler().fit_transform(X)

    # 安全检查：样本数少时调低 perplexity
    n_samples = X_scaled.shape[0]
    safe_perp = min(perplexity, max(5.0, n_samples / 4.0))

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perp,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        n_jobs=1,           # 避免服务器线程冲突
    )
    return tsne.fit_transform(X_scaled)


# ──────────────────────────────────────────────
#  绘图核心
# ──────────────────────────────────────────────

def plot_one_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    y: np.ndarray,
    type_keys: List[str],
    panel_title: str,
    show_legend: bool = False,
):
    """在单个子图 ax 上绘制 t-SNE 散点图。"""
    ax.set_facecolor(PALETTE["bg"])

    # 网格
    ax.grid(True, linestyle="--", linewidth=0.5, color=PALETTE["grid"], zorder=0)

    for idx, key in enumerate(type_keys):
        mask = y == idx
        if not mask.any():
            continue

        cfg_key = key  # e.g. "text", "image", "global_text"
        name, color, marker = MODALITY_CONFIG[cfg_key]

        is_global = key.startswith("global_")
        size    = 280 if is_global else 60
        alpha   = 1.0 if is_global else 0.78
        lw      = 1.5 if is_global else 0.6
        zorder  = 10 if is_global else 5

        scatter = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="white" if is_global else color,
            linewidths=lw,
            label=name,
            zorder=zorder,
        )
        # 全局锚点额外添加光晕效果
        if is_global:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                marker=marker,
                s=size * 2.5,
                alpha=0.18,
                edgecolors="none",
                zorder=zorder - 1,
            )

    # 标题与轴
    ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE Dim 1", fontsize=10, labelpad=4)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10, labelpad=4)
    ax.tick_params(labelsize=8)

    # 去掉上/右边框（学术风格）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    if show_legend:
        legend = ax.legend(
            loc="lower right",
            framealpha=0.92,
            fontsize=8.5,
            markerscale=1.2,
            edgecolor="#CCCCCC",
            facecolor="white",
        )
        legend.get_frame().set_linewidth(0.8)


def make_figure(
    all_features: List[Dict[str, np.ndarray]],
    panel_labels: List[str],
    output_path: str,
    dpi: int = 300,
    perplexity: float = 25.0,
):
    """
    主绘图函数：对每个 checkpoint 分别运行 t-SNE，绘制 1×N 并排子图。
    """
    n_panels = len(all_features)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.2 * n_panels, 4.8),
        dpi=dpi,
    )
    if n_panels == 1:
        axes = [axes]

    # 设置整体字体
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "axes.labelsize": 10,
        "font.size":      10,
    })
    fig.patch.set_facecolor("white")

    # 共享图例的 handles/labels（取最后一个 panel 收集）
    first_legend_ax = None

    for i, (feat_dict, label) in enumerate(zip(all_features, panel_labels)):
        X, y, type_keys = build_tsne_input(feat_dict)
        print(f"  [{i+1}/{n_panels}] {label} → t-SNE 降维 "
              f"(N={X.shape[0]}, D={X.shape[1]}) ...", flush=True)
        embedding = compute_tsne(X, perplexity=perplexity)

        show_legend = (i == n_panels - 1)   # 仅最后一个 panel 显示图例
        plot_one_panel(
            axes[i], embedding, y, type_keys,
            panel_title=label,
            show_legend=show_legend,
        )
        if show_legend:
            first_legend_ax = axes[i]

    # 总标题
    fig.suptitle(
        "Cross-Modal Feature Alignment via Decoupled Contrastive Aggregation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"\n✅ 图像已保存至：{out.resolve()}")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Demo 模式（无需真实 checkpoint）
# ──────────────────────────────────────────────

def _make_demo_features(round_idx: int, n_clients: int = 3, D: int = 128) -> Dict[str, np.ndarray]:
    """
    生成合成演示特征：随着 round_idx 增大，各模态特征向全局锚点靠拢。
    """
    rng = np.random.RandomState(42 + round_idx)

    # 进度因子 [0, 1]，衡量聚拢程度
    progress = round_idx / 5.0   # 0=Round1, 0.5=Round25, 1.0=Round50

    # 全局锚点（固定）
    g_text  = np.array([[ 2.0,  0.5] + [0.0] * (D - 2)], dtype=np.float32)
    g_image = np.array([[-2.0,  0.5] + [0.0] * (D - 2)], dtype=np.float32)

    # 各客户端局部点——初始分散，随 progress 向全局锚点聚拢
    scale = 3.5 * (1 - 0.8 * progress)   # 噪声范围随训练减小

    text_reps = (g_text  + rng.randn(n_clients, D).astype(np.float32) * scale).astype(np.float32)
    img_reps  = (g_image + rng.randn(n_clients, D).astype(np.float32) * scale).astype(np.float32)
    mm_center = (g_text + g_image) / 2
    mm_reps   = (mm_center + rng.randn(n_clients, D).astype(np.float32) * scale * 0.8).astype(np.float32)

    return {
        "global_text":  g_text,
        "global_image": g_image,
        "text":         text_reps,
        "image":        img_reps,
        "multimodal":   mm_reps,
    }


# ──────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedSAM3-Cream t-SNE 特征对齐可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints", "-c",
        nargs="+",
        default=None,
        help="checkpoint 文件路径列表，按顺序对应各子图（支持1~N个）",
    )
    parser.add_argument(
        "--labels", "-l",
        nargs="+",
        default=None,
        help="各子图的标题标签，数量需与 --checkpoints 相同",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/figures/tsne_alignment.pdf",
        help="输出图像路径（支持 .pdf / .png / .svg）",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=25.0,
        help="t-SNE 困惑度参数",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出分辨率（DPI）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用合成数据生成演示图（无需真实 checkpoint）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        # ── Demo 模式 ──────────────────────────
        print("▶ Demo 模式：生成合成演示数据 ...")
        demo_rounds = [0, 3, 5]
        demo_labels = ["Round 1 (Initial)", "Round 25 (Mid)", "Round 50 (Final)"]
        all_features = [_make_demo_features(r) for r in demo_rounds]
        panel_labels = demo_labels
        output = args.output if args.output != "results/figures/tsne_alignment.pdf" \
                             else "results/figures/tsne_alignment_demo.pdf"
    else:
        # ── 真实 checkpoint 模式 ───────────────
        if not args.checkpoints:
            print("❌ 请通过 --checkpoints 指定 checkpoint 路径，或使用 --demo 生成演示图。")
            sys.exit(1)

        for p in args.checkpoints:
            if not Path(p).exists():
                print(f"❌ checkpoint 不存在：{p}")
                sys.exit(1)

        n_ckpt = len(args.checkpoints)
        panel_labels = args.labels if args.labels else [f"Round {i+1}" for i in range(n_ckpt)]
        if len(panel_labels) != n_ckpt:
            print("❌ --labels 数量与 --checkpoints 数量不匹配。")
            sys.exit(1)

        print(f"▶ 加载 {n_ckpt} 个 checkpoint ...")
        all_features = []
        for path, label in zip(args.checkpoints, panel_labels):
            print(f"  → {label}: {path}")
            feats = load_checkpoint_features(path)
            all_features.append(feats)

        output = args.output

    # 绘图
    print("\n▶ 运行 t-SNE & 绘图 ...")
    make_figure(
        all_features=all_features,
        panel_labels=panel_labels,
        output_path=output,
        dpi=args.dpi,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
