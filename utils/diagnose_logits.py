"""
Logits崩塌诊断工具 (2026-03-01)

用于诊断分割模型输出的Logits异常问题,包括:
1. Logits全为负数(预测全为背景)
2. Logits全为正数(预测全为前景)
3. Logits范围异常

使用方法:
    from utils.diagnose_logits import diagnose_model_output

    # 在训练循环中
    output = model(images)
    diagnose_model_output(output['logits'], masks, verbose=True)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def diagnose_model_output(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    诊断模型输出的Logits统计信息

    Args:
        logits: 模型输出的logits (B, C, H, W)
        targets: 真实标签 (B, C, H, W)，可选
        verbose: 是否打印详细信息

    Returns:
        统计信息字典
    """
    stats = {}

    # 基本统计
    stats['min'] = logits.min().item()
    stats['max'] = logits.max().item()
    stats['mean'] = logits.mean().item()
    stats['std'] = logits.std().item()

    # Sigmoid后的概率
    probs = torch.sigmoid(logits)
    stats['prob_min'] = probs.min().item()
    stats['prob_max'] = probs.max().item()
    stats['prob_mean'] = probs.mean().item()

    # 预测分布
    preds = (probs > 0.5).float()
    stats['pred_foreground_ratio'] = preds.mean().item()

    # 如果有targets,计算实际前景比例
    if targets is not None:
        stats['true_foreground_ratio'] = (targets > 0.5).float().mean().item()

        # 计算预测与真实的差距
        stats['prediction_bias'] = stats['pred_foreground_ratio'] - stats['true_foreground_ratio']

    # 异常检测
    warnings = []

    # 检测1: Logits全为负数
    if stats['max'] < 0:
        warnings.append("⚠️ CRITICAL: 所有logits为负数！模型预测全为背景。")
        warnings.append("   可能原因: (1) 分割头bias初始化不当 (2) 编码器特征崩塌 (3) RoPE错误")
        warnings.append("   建议: 检查RoPE设置、Adapter权重、分割头初始化")

    # 检测2: Logits全为正数
    elif stats['min'] > 0:
        warnings.append("⚠️ WARNING: 所有logits为正数！模型预测全为前景。")
        warnings.append("   可能原因: 分割头bias过大或特征异常")

    # 检测3: Logits范围异常大
    logit_range = stats['max'] - stats['min']
    if logit_range > 50:
        warnings.append(f"⚠️ WARNING: Logits范围异常大 ({logit_range:.2f})")
        warnings.append("   可能原因: 梯度爆炸、特征尺度不匹配")

    # 检测4: 预测严重偏离真实分布
    if targets is not None:
        if abs(stats['prediction_bias']) > 0.5:
            warnings.append(f"⚠️ WARNING: 预测偏差过大 ({stats['prediction_bias']:.4f})")
            warnings.append(f"   真实前景比例: {stats['true_foreground_ratio']:.4f}")
            warnings.append(f"   预测前景比例: {stats['pred_foreground_ratio']:.4f}")

    # 检测5: 概率分布退化
    if stats['prob_std'] < 0.01:
        warnings.append("⚠️ WARNING: 概率分布退化（标准差过小）")
        warnings.append("   可能原因: 模型输出趋于一致，缺乏区分度")

    stats['warnings'] = warnings

    # 打印诊断信息
    if verbose:
        print("\n" + "=" * 60)
        print("Logits 诊断报告")
        print("=" * 60)

        print(f"\n【Logits 统计】")
        print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  均值: {stats['mean']:.4f} ± {stats['std']:.4f}")

        print(f"\n【概率分布 (Sigmoid后)】")
        print(f"  范围: [{stats['prob_min']:.4f}, {stats['prob_max']:.4f}]")
        print(f"  均值: {stats['prob_mean']:.4f}")

        print(f"\n【预测分布】")
        print(f"  预测前景比例: {stats['pred_foreground_ratio']:.4f}")
        if targets is not None:
            print(f"  真实前景比例: {stats['true_foreground_ratio']:.4f}")
            print(f"  偏差: {stats['prediction_bias']:.4f}")

        if warnings:
            print(f"\n【异常检测】")
            for warning in warnings:
                print(warning)
        else:
            print(f"\n【异常检测】")
            print("✓ 未发现明显异常")

        print("=" * 60 + "\n")

    return stats


def suggest_bias_initialization(
    foreground_ratio: float,
    verbose: bool = True
) -> float:
    """
    根据数据集前景比例,建议分割头的bias初始化值

    Args:
        foreground_ratio: 数据集中前景像素的占比 (0-1)
        verbose: 是否打印建议

    Returns:
        建议的bias初始值 (logit space)
    """
    # 裁剪到有效范围
    p = max(0.001, min(0.999, foreground_ratio))

    # 计算bias (logit space)
    # 使初始sigmoid(bias) ≈ p
    suggested_bias = np.log(p / (1.0 - p))

    if verbose:
        print("\n" + "=" * 60)
        print("分割头Bias初始化建议")
        print("=" * 60)
        print(f"数据集前景比例: {foreground_ratio:.4f}")
        print(f"建议Bias值: {suggested_bias:.4f}")
        print(f"初始概率: {torch.sigmoid(torch.tensor(suggested_bias)).item():.4f}")
        print("\n实现方法 (在MaskDecoder.__init__中):")
        print(f"```python")
        print(f"if self.final_conv.bias is not None:")
        print(f"    nn.init.constant_(self.final_conv.bias, {suggested_bias:.4f})")
        print(f"```")
        print("=" * 60 + "\n")

    return suggested_bias


def check_rope_frequencies(
    model: torch.nn.Module,
    expected_img_size: int = 256,
    verbose: bool = True
) -> Dict[str, any]:
    """
    检查模型中所有RoPE频率缓存是否匹配img_size

    Args:
        model: SAM3模型
        expected_img_size: 期望的图像尺寸
        verbose: 是否打印详细信息

    Returns:
        检查结果字典
    """
    results = {
        'total_blocks': 0,
        'rope_enabled_blocks': 0,
        'mismatched_blocks': [],
        'is_all_correct': True
    }

    # 计算期望的seq_len
    patch_size = 16  # SAM3/ViT默认
    expected_seq_len = (expected_img_size // patch_size) ** 2

    # 遍历所有blocks
    if hasattr(model, 'wrapped_blocks'):
        results['total_blocks'] = len(model.wrapped_blocks)

        for i, wrapped_block in enumerate(model.wrapped_blocks):
            # 获取原始block
            original_block = wrapped_block.original_block if hasattr(wrapped_block, 'original_block') else wrapped_block

            # 检查attn层
            if not hasattr(original_block, 'attn'):
                continue

            attn = original_block.attn

            # 检查是否启用RoPE
            if not getattr(attn, 'use_rope', False):
                continue

            results['rope_enabled_blocks'] += 1

            # 检查freqs_cis
            if not hasattr(attn, 'freqs_cis') or attn.freqs_cis is None:
                results['mismatched_blocks'].append({
                    'block_idx': i,
                    'issue': 'freqs_cis is None'
                })
                results['is_all_correct'] = False
                continue

            # 获取当前形状
            current_seq_len = attn.freqs_cis.shape[0]

            # 检查是否匹配
            if current_seq_len != expected_seq_len:
                results['mismatched_blocks'].append({
                    'block_idx': i,
                    'current_seq_len': current_seq_len,
                    'expected_seq_len': expected_seq_len,
                    'issue': 'size mismatch'
                })
                results['is_all_correct'] = False

    # 打印结果
    if verbose:
        print("\n" + "=" * 60)
        print("RoPE 频率检查报告")
        print("=" * 60)
        print(f"期望图像尺寸: {expected_img_size}x{expected_img_size}")
        print(f"期望序列长度: {expected_seq_len}")
        print(f"总Block数: {results['total_blocks']}")
        print(f"启用RoPE的Block数: {results['rope_enabled_blocks']}")

        if results['mismatched_blocks']:
            print(f"\n⚠️ 发现 {len(results['mismatched_blocks'])} 个不匹配的Block:")
            for mismatch in results['mismatched_blocks']:
                print(f"  - Block {mismatch['block_idx']}: {mismatch['issue']}")
                if 'current_seq_len' in mismatch:
                    print(f"    当前seq_len: {mismatch['current_seq_len']}, 期望: {mismatch['expected_seq_len']}")

            print(f"\n建议: 调用 model.reset_rope_frequencies() 重置")
        else:
            print(f"\n✓ 所有RoPE频率缓存正确!")

        print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Logits 诊断工具测试")
    print("=" * 60)

    # 测试1: 模拟崩塌的logits
    print("\n【测试1: Logits崩塌 (全为负数)】")
    collapsed_logits = torch.randn(2, 1, 256, 256) - 15  # 强制为负数
    collapsed_targets = torch.randint(0, 2, (2, 1, 256, 256)).float() * 0.05  # 5%前景

    stats1 = diagnose_model_output(collapsed_logits, collapsed_targets, verbose=True)

    # 测试2: 正常的logits
    print("\n【测试2: 正常Logits】")
    normal_logits = torch.randn(2, 1, 256, 256) * 2
    normal_targets = torch.randint(0, 2, (2, 1, 256, 256)).float() * 0.1

    stats2 = diagnose_model_output(normal_logits, normal_targets, verbose=True)

    # 测试3: Bias建议
    print("\n【测试3: Bias初始化建议】")
    suggest_bias_initialization(0.01, verbose=True)  # 1%前景
    suggest_bias_initialization(0.10, verbose=True)  # 10%前景

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
