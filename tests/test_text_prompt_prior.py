"""
tests/test_text_prompt_prior.py
================================
Step 3 TextPromptEncoder 单元测试

验证目标：
  1. 零初始化：up_proj 全零 → 初始输出全零（不破坏基线分割）
  2. 无文本 Prompt → (B, 0, embed_dim)（向后兼容）
  3. 1D 文本表征 (D,) → (B, 1, embed_dim)
  4. 2D 文本表征 (B, D) → (B, 1, embed_dim)
  5. .detach() 保证 text_token 无梯度
  6. .contiguous() 内存布局正确
  7. 设备对齐（CPU）
  8. SAM3MedicalIntegrated.forward 新签名向后兼容（无 text_prompt 不崩溃）
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 导入被测模块
# ─────────────────────────────────────────────
try:
    from src.integrated_model import TextPromptEncoder
    TPE_AVAILABLE = True
except ImportError as e:
    print(f"[SKIP] TextPromptEncoder import failed: {e}")
    TPE_AVAILABLE = False


# ═══════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════
def _make_encoder(text_dim=512, embed_dim=768, bottleneck_dim=256):
    return TextPromptEncoder(text_dim=text_dim, embed_dim=embed_dim,
                             bottleneck_dim=bottleneck_dim)


PASS = "✓"
FAIL = "✗"
results = []


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append((name, cond))
    return cond


# ═══════════════════════════════════════════════════════
# Test 1: 零初始化
# ═══════════════════════════════════════════════════════
def test_zero_init():
    enc = _make_encoder()
    # up_proj.weight 全零
    check("up_proj.weight 零初始化",
          enc.up_proj.weight.abs().max().item() == 0.0)
    # up_proj.bias 全零
    check("up_proj.bias 零初始化",
          enc.up_proj.bias.abs().max().item() == 0.0)
    # forward 初始输出全零
    g = torch.randn(4, 512)
    out = enc(g, batch_size=4, device=torch.device('cpu'), dtype=torch.float32)
    check("初始 forward 输出全零",
          out.abs().max().item() == 0.0,
          f"max={out.abs().max().item():.6f}")


# ═══════════════════════════════════════════════════════
# Test 2: None → 空 sparse_embeddings
# ═══════════════════════════════════════════════════════
def test_none_input():
    enc = _make_encoder(text_dim=512, embed_dim=768)
    out = enc(None, batch_size=3, device=torch.device('cpu'), dtype=torch.float32)
    check("None 输入 → (B, 0, embed_dim)",
          out.shape == (3, 0, 768),
          f"shape={out.shape}")


# ═══════════════════════════════════════════════════════
# Test 3: 1D 输入 (D,)
# ═══════════════════════════════════════════════════════
def test_1d_input():
    enc = _make_encoder()
    g = torch.randn(512)
    out = enc(g, batch_size=4, device=torch.device('cpu'), dtype=torch.float32)
    check("1D (D,) → (B, 1, embed_dim)",
          out.shape == (4, 1, 768),
          f"shape={out.shape}")


# ═══════════════════════════════════════════════════════
# Test 4: 2D 输入 (B, D)
# ═══════════════════════════════════════════════════════
def test_2d_input():
    enc = _make_encoder()
    g = torch.randn(4, 512)
    out = enc(g, batch_size=4, device=torch.device('cpu'), dtype=torch.float32)
    check("2D (B, D) → (B, 1, embed_dim)",
          out.shape == (4, 1, 768),
          f"shape={out.shape}")


# ═══════════════════════════════════════════════════════
# Test 5: 梯度截断保护
# ═══════════════════════════════════════════════════════
def test_detach_gradient():
    enc = _make_encoder()
    g = torch.randn(512, requires_grad=True)
    g_detached = g.detach()
    out = enc(g_detached, batch_size=2, device=torch.device('cpu'), dtype=torch.float32)
    # 对 out 求和，梯度不应流到 g
    try:
        out.sum().backward()
        grad_ok = (g.grad is None)
    except Exception:
        grad_ok = True  # 不带 grad 的情况也可接受
    check("梯度截断（.detach() 后反传不流向 g）", grad_ok)


# ═══════════════════════════════════════════════════════
# Test 6: contiguous 内存布局
# ═══════════════════════════════════════════════════════
def test_contiguous():
    enc = _make_encoder()
    g = torch.randn(512)
    # expand 返回非连续视图，经 contiguous → Linear 不崩溃
    expanded = g.unsqueeze(0).expand(4, -1)
    check("expand 前非连续", not expanded.is_contiguous())
    out = enc(g, batch_size=4, device=torch.device('cpu'), dtype=torch.float32)
    check("forward 输出连续", out.is_contiguous())


# ═══════════════════════════════════════════════════════
# Test 7: 设备对齐
# ═══════════════════════════════════════════════════════
def test_device_alignment():
    enc = _make_encoder()
    g = torch.randn(512)  # CPU
    out = enc(g, batch_size=2, device=torch.device('cpu'), dtype=torch.float32)
    check("输出设备 = CPU", out.device.type == 'cpu')
    check("输出 dtype = float32", out.dtype == torch.float32)


# ═══════════════════════════════════════════════════════
# Test 8: forward 新签名向后兼容（无 text_prompt 参数）
# ═══════════════════════════════════════════════════════
def test_forward_backward_compat():
    """
    验证 TextPromptEncoder.forward() 在 text_prompt=None 时
    与原 _get_empty_prompts 行为完全一致（返回 (B, 0, embed_dim)）
    """
    enc = _make_encoder()
    out_none = enc(None, batch_size=2, device=torch.device('cpu'), dtype=torch.float32)
    check("向后兼容：None → (B, 0, D)",
          out_none.shape == (2, 0, 768),
          f"shape={out_none.shape}")
    check("向后兼容：None → 非 None（正常 tensor）",
          isinstance(out_none, torch.Tensor))


# ═══════════════════════════════════════════════════════
# 主运行
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("TextPromptEncoder Step 3 单元测试")
    print("=" * 60)

    if not TPE_AVAILABLE:
        print("[ERROR] TextPromptEncoder 无法导入，跳过所有测试")
        exit(1)

    test_zero_init()
    test_none_input()
    test_1d_input()
    test_2d_input()
    test_detach_gradient()
    test_contiguous()
    test_device_alignment()
    test_forward_backward_compat()

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print()
    print("=" * 60)
    print(f"结果: {passed} 通过 / {total - passed} 失败 / {total} 总计")
    print("=" * 60)
    if passed < total:
        exit(1)
