"""
GroupA 分割结果可视化脚本
论文用：从验证集随机取 N 张切片，生成 [原图 | GT(WT) | Pred(WT)] 三列对比图
用法:
    python scripts/visualize_segmentation.py \
        --checkpoint data/federated_split/checkpoints/best_model.pth \
        --val_json   data/federated_split/val_split.json \
        --data_root  G:/BraTS2020 \
        --output_dir data/federated_split/checkpoints/vis \
        --n_samples  16 \
        --img_size   256
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib


def load_case(case_info: dict, data_root: Path, img_size: int):
    """从 val_split.json 的一条记录中加载 (image, gt_wt_mask)"""
    # 取 flair 图像（images list 第0项）
    flair_rel = case_info['image'][0]
    flair_path = data_root / flair_rel

    img_data = nib.load(str(flair_path)).get_fdata()  # (H, W, D)

    # 选前景最多的切片
    mask_rel = case_info.get('label', '')
    mask_path = data_root / mask_rel if mask_rel else None
    if mask_path and mask_path.exists():
        mask_vol = nib.load(str(mask_path)).get_fdata()  # (H, W, D)
        fg_per_slice = (mask_vol > 0).sum(axis=(0, 1))    # (D,)
        z = int(np.argmax(fg_per_slice))
        gt_raw = mask_vol[:, :, z]                         # (H, W)
    else:
        z = img_data.shape[2] // 2
        gt_raw = np.zeros(img_data.shape[:2])

    slice_img = img_data[:, :, z]  # (H, W), FLAIR

    # Brain Z-score
    brain_mask = slice_img > 0
    if brain_mask.sum() > 0:
        mu, sigma = slice_img[brain_mask].mean(), slice_img[brain_mask].std()
        slice_img = (slice_img - mu) / (sigma + 1e-8)
        slice_img[~brain_mask] = 0.0

    # (3, H, W)，3通道灰度输入
    img_t = torch.from_numpy(slice_img).float().unsqueeze(0).repeat(3, 1, 1)
    img_t = F.interpolate(img_t.unsqueeze(0), size=(img_size, img_size),
                          mode='bilinear', align_corners=False).squeeze(0)

    # GT: WT = 所有前景（label ∈ {1,2,4}）
    gt_wt = (gt_raw > 0).astype(np.uint8)
    gt_t = torch.from_numpy(gt_wt).float()
    gt_t = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0),
                         size=(img_size, img_size), mode='nearest').squeeze()

    return img_t, gt_t, slice_img  # 返回原始 numpy 用于显示


@torch.no_grad()
def predict(model, img_t: torch.Tensor, device: str) -> np.ndarray:
    """返回 WT 二值预测 (H, W) numpy"""
    inp = img_t.unsqueeze(0).to(device)              # (1, 3, H, W)
    out = model(inp)

    if isinstance(out, dict):
        logits = out.get('logits', list(out.values())[0])
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    probs = torch.sigmoid(logits)                    # (1, C, H, W)
    if probs.shape[1] > 1:
        # channel 1 = WT（与 _load_mask 的 channel1 一致）
        pred = (probs[0, 1] > 0.5).cpu().numpy()
    else:
        pred = (probs[0, 0] > 0.5).cpu().numpy()

    return pred.astype(np.uint8)


def make_grid(cases, model, data_root, img_size, device, n_cols=4):
    """生成 n_samples × 3 的对比图（原图 | GT | Pred）"""
    n = len(cases)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows * 3, n_cols,
        figsize=(n_cols * 3, n_rows * 9),
        dpi=150
    )
    # 确保 axes 始终是二维
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    col_titles = ['FLAIR', 'GT (WT)', 'Pred (WT)']

    for i, case in enumerate(cases):
        row_block = (i // n_cols) * 3
        col = i % n_cols

        img_t, gt_t, img_np = load_case(case, data_root, img_size)
        pred_np = predict(model, img_t, device)
        gt_np = gt_t.numpy()

        # 显示用归一化 FLAIR（取 channel 0）
        display_img = img_t[0].numpy()

        layers = [display_img, gt_np, pred_np]
        cmaps  = ['gray', 'Reds', 'Blues']

        for j, (arr, cmap) in enumerate(zip(layers, cmaps)):
            ax = axes[row_block + j, col]
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=arr.max() if j == 0 else 1)
            ax.axis('off')
            if i < n_cols:
                ax.set_title(col_titles[j], fontsize=9, pad=2)

        # 标注病例名
        case_name = Path(case['image'][0]).parts[0]
        axes[row_block, col].set_title(
            f"{case_name}\n{col_titles[0]}", fontsize=7, pad=2
        )

    # 隐藏多余的子图
    for idx in range(n, n_rows * n_cols):
        r = (idx // n_cols) * 3
        c = idx % n_cols
        for jr in range(3):
            axes[r + jr, c].axis('off')

    plt.suptitle('Segmentation Results: FLAIR | GT (WT) | Pred (WT)',
                 fontsize=13, fontweight='bold', y=1.002)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='GroupA 分割可视化')
    parser.add_argument('--checkpoint', default='data/federated_split/checkpoints/best_model.pth')
    parser.add_argument('--val_json',   default='', help='val_split.json 路径（与 --val_dir 二选一）')
    parser.add_argument('--val_dir',    default='', help='直接指定含 BraTS* 子目录的验证集目录（与 --val_json 二选一）')
    parser.add_argument('--data_root',  default='', help='BraTS2020 源数据根目录（使用 --val_json 时必填）')
    parser.add_argument('--output_dir', default='data/federated_split/checkpoints/vis')
    parser.add_argument('--n_samples',  type=int, default=16, help='取多少张切片')
    parser.add_argument('--n_cols',     type=int, default=4,  help='每行列数')
    parser.add_argument('--sam3_checkpoint', default='data/checkpoints/sam3.pt',
                        help='SAM3 预训练权重路径（与训练时一致）')
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--text_dim',   type=int, default=768, help='text_dim，需与训练时一致')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 case_info 列表：支持两种来源
    if args.val_dir:
        # 直接扫描目录中的 BraTS* 病例文件夹
        val_dir = Path(args.val_dir)
        case_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir() and 'BraTS' in d.name])
        if not case_dirs:
            raise FileNotFoundError(f"在 {val_dir} 中未找到 BraTS* 文件夹")
        data_root = val_dir  # 图像直接在各病例文件夹下
        val_data = []
        for d in case_dirs:
            flair = list(d.glob('*_flair.nii')) + list(d.glob('*_flair.nii.gz'))
            seg   = list(d.glob('*_seg.nii'))   + list(d.glob('*_seg.nii.gz'))
            if flair and seg:
                # 构造与 val_split.json 相同格式的 case_info
                val_data.append({
                    'image': [str(flair[0].relative_to(val_dir))],
                    'label': str(seg[0].relative_to(val_dir)),
                })
        print(f"[Vis] 从目录扫描到 {len(val_data)} 个有效病例")
    elif args.val_json:
        data_root = Path(args.data_root)
        with open(args.val_json, encoding='utf-8') as f:
            val_data = json.load(f)['data']
        val_data = [c for c in val_data if c.get('label', '')]
    else:
        raise ValueError("必须提供 --val_dir 或 --val_json 其中之一")

    n = min(args.n_samples, len(val_data))
    sampled = random.sample(val_data, n)
    print(f"[Vis] 从 {len(val_data)} 个验证样本中随机取 {n} 个")

    # 加载模型
    print(f"[Vis] 加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    from src.integrated_model import SAM3MedicalIntegrated
    model = SAM3MedicalIntegrated(
        img_size=args.img_size,
        num_classes=3,
        embed_dim=768,
        adapter_dim=64,
        text_dim=args.text_dim,
        use_sam3=True,
        freeze_encoder=True,
        use_adapter=True,
        sam3_checkpoint=args.sam3_checkpoint,
        device=args.device,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [警告] 缺失键: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.to(args.device)
    model.eval()
    print(f"[Vis] 模型加载完成，设备: {args.device}")

    # 生成并保存网格图
    print("[Vis] 生成可视化图...")
    fig = make_grid(sampled, model, data_root, args.img_size, args.device, args.n_cols)
    out_path = output_dir / 'seg_comparison_grid.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[Vis] ✓ 已保存: {out_path}")

    # 额外保存单张大图（论文主图用）
    print("[Vis] 生成论文主图（单行 4 样本）...")
    paper_cases = sampled[:4]
    fig2, axes2 = plt.subplots(3, 4, figsize=(14, 9), dpi=200)
    col_titles = ['FLAIR', 'GT (WT)', 'Pred (WT)']
    for col, case in enumerate(paper_cases):
        img_t, gt_t, _ = load_case(case, data_root, args.img_size)
        pred_np = predict(model, img_t, args.device)
        gt_np = gt_t.numpy()
        display = img_t[0].numpy()
        case_name = Path(case['image'][0]).parts[0]
        for row, (arr, cmap) in enumerate(zip([display, gt_np, pred_np], ['gray', 'Reds', 'Blues'])):
            ax = axes2[row, col]
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=arr.max() if row == 0 else 1)
            ax.axis('off')
            if row == 0:
                ax.set_title(case_name, fontsize=7, pad=3)
        axes2[0, col].set_xlabel(col_titles[0], fontsize=8)

    for row, title in enumerate(col_titles):
        axes2[row, 0].set_ylabel(title, fontsize=10, rotation=90, labelpad=4)
        axes2[row, 0].yaxis.label.set_visible(True)

    plt.suptitle('FedSAM3-Cream GroupA: Segmentation Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    paper_path = output_dir / 'seg_paper_figure.png'
    fig2.savefig(paper_path, bbox_inches='tight', dpi=200)
    plt.close(fig2)
    print(f"[Vis] ✓ 论文主图已保存: {paper_path}")
    print("[Vis] 完成！")


if __name__ == '__main__':
    main()
