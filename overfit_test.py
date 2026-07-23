"""
快速过拟合测试 (Overfit Sanity Test)
测试模型是否能在 1 张固定图像上收敛
用法（项目根目录下）：
    python overfit_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from src.integrated_model import SAM3MedicalIntegrated

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 初始化模型（与主训练完全相同的参数）──
model = SAM3MedicalIntegrated(
    img_size=1024,
    num_classes=1,
    use_sam3=True,
    freeze_encoder=True,
    use_adapter=True,
    device=DEVICE,
    sam3_checkpoint="data/checkpoints/sam3.pt",
).to(DEVICE)
model.train()

trainable = model.get_trainable_params()
print(f"可训练参数量: {sum(p.numel() for p in trainable):,}")

optimizer = torch.optim.AdamW(trainable, lr=1e-4)

# ── 构造 1 张固定图像 + 固定 mask（模拟椭圆形肿瘤）──
torch.manual_seed(42)
image = torch.randn(1, 3, 1024, 1024).to(DEVICE)

mask = torch.zeros(1, 1, 1024, 1024).to(DEVICE)
mask[:, :, 350:650, 350:650] = 1.0  # 300×300 的正方形"肿瘤"

gt_fg = mask.sum().item() / mask.numel()
print(f"GT 前景比例: {gt_fg*100:.1f}%\n")

# ── 训练 100 步 ──
for step in range(101):
    out = model(image)

    if isinstance(out, dict):
        logits = out.get('logits', out.get('out'))
    elif hasattr(out, 'logits'):
        logits = out.logits
    else:
        logits = out

    # 尺寸对齐
    if logits.shape[2:] != mask.shape[2:]:
        logits = F.interpolate(logits, size=mask.shape[2:], mode='bilinear', align_corners=False)

    # 取第一个通道（与 evaluate_model 一致）
    if logits.shape[1] > 1:
        logits = logits[:, 0:1]

    # BCE + Dice 联合损失（不用 Tversky，排除 loss 函数干扰）
    prob = torch.sigmoid(logits)
    bce  = F.binary_cross_entropy_with_logits(logits, mask)
    inter = (prob * mask).sum()
    dice_loss = 1 - (2 * inter + 1e-6) / (prob.sum() + mask.sum() + 1e-6)
    loss = bce + dice_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        pred_fg = (prob > 0.5).float().sum().item() / prob.numel()
        print(f"Step {step:3d} | loss={loss.item():.4f} "
              f"| bce={bce.item():.4f} | dice_loss={dice_loss.item():.4f} "
              f"| pred_fg={pred_fg*100:.1f}%")

print("\n── 结果解读 ──")
print("loss 降到 < 0.3 且 pred_fg 接近 29%  →  模型能学习，问题在初始化/loss配置")
print("loss 几乎不动 / pred_fg 99%+          →  梯度链路有根本问题，需要排查 forward")
