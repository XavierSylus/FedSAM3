#!/usr/bin/env python3
"""
生成虚拟文本特征文件用于测试
为每个 BraTS 样本创建随机的 768 维文本特征向量

功能：
1. 读取 federated_split 中的客户端数据集配置
2. 为需要 text_feature 的样本生成 .npy 文件
3. 使用随机正态分布生成特征向量（可选：L2 归一化）

注意：这仅用于测试数据加载流程！
真实应用需要从医学报告中提取文本特征（如使用 BioBERT/PubMedBERT）
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Set


def extract_text_feature_paths(json_path: Path) -> Set[str]:
    """
    从客户端数据集 JSON 中提取所有文本特征路径

    Args:
        json_path: 客户端 dataset.json 文件路径

    Returns:
        文本特征路径集合（去重）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_paths = set()
    for item in data['data']:
        text_feature = item.get('text_feature')
        if text_feature and text_feature != 'empty':
            text_paths.add(text_feature)

    return text_paths


def generate_dummy_text_feature(
    output_path: Path,
    embed_dim: int = 768,
    normalize: bool = True
):
    """
    生成虚拟文本特征向量

    Args:
        output_path: 输出 .npy 文件路径
        embed_dim: 嵌入维度（默认 768，匹配 BERT/BiomedBERT）
        normalize: 是否 L2 归一化
    """
    # 生成随机正态分布向量
    feature = np.random.randn(embed_dim).astype(np.float32)

    # L2 归一化
    if normalize:
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

    # 保存为 .npy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature)


def main():
    """主函数"""
    print("=" * 80)
    print("FedSAM3-Cream 虚拟文本特征生成器")
    print("⚠️  仅用于测试数据加载流程！")
    print("=" * 80)
    print()

    # ==================== 路径配置 ====================
    project_root = Path(__file__).parent.parent
    federated_root = project_root / 'data' / 'federated_split'
    data_root = project_root / 'data' / 'source_images' / 'BraTS2020' / 'BraTS2020_TrainingData' / 'BraTS2020_Training'

    # 客户端列表
    clients = [
        'client1_text_only',
        'client3_multimodal'  # client2_image_only 不需要文本特征
    ]

    # ==================== Step 1: 收集所有需要的文本特征路径 ====================
    print("[1/3] 收集文本特征路径...")
    all_text_paths = set()

    for client_name in clients:
        json_path = federated_root / client_name / 'dataset.json'

        if not json_path.exists():
            print(f"  ⚠️  跳过 {client_name}: dataset.json 不存在")
            continue

        client_paths = extract_text_feature_paths(json_path)
        all_text_paths.update(client_paths)
        print(f"  ✓ {client_name}: {len(client_paths)} 个文本特征")

    print(f"\n  总计需要生成: {len(all_text_paths)} 个文本特征文件")

    # ==================== Step 2: 生成虚拟文本特征 ====================
    print("\n[2/3] 生成虚拟文本特征...")

    generated_count = 0
    skipped_count = 0

    for text_path_str in sorted(all_text_paths):
        output_path = data_root / text_path_str

        # 检查文件是否已存在
        if output_path.exists():
            skipped_count += 1
            continue

        # 生成虚拟特征
        generate_dummy_text_feature(
            output_path=output_path,
            embed_dim=768,
            normalize=True
        )
        generated_count += 1

        if generated_count % 10 == 0:
            print(f"  已生成 {generated_count}/{len(all_text_paths)} 个文件...")

    print(f"\n  ✓ 新生成: {generated_count} 个文件")
    print(f"  ✓ 跳过（已存在）: {skipped_count} 个文件")

    # ==================== Step 3: 验证生成结果 ====================
    print("\n[3/3] 验证生成结果...")

    success = True
    for text_path_str in all_text_paths:
        output_path = data_root / text_path_str

        if not output_path.exists():
            print(f"  ❌ 文件缺失: {output_path}")
            success = False

    if success:
        print(f"  ✓ 所有文件验证通过！")
    else:
        print(f"  ❌ 部分文件生成失败")
        return 1

    # ==================== 完成 ====================
    print("\n" + "=" * 80)
    print("✅ 虚拟文本特征生成完成！")
    print("=" * 80)
    print()
    print("📁 生成位置:", data_root)
    print(f"📊 文件统计: {generated_count + skipped_count} 个 .npy 文件")
    print()
    print("⚠️  重要提示:")
    print("  1. 这些是随机生成的虚拟特征，仅用于测试数据加载")
    print("  2. 真实应用需要从医学报告提取真实文本特征")
    print("  3. 推荐使用: BioBERT, PubMedBERT, ClinicalBERT 等医学预训练模型")
    print()
    print("🎯 下一步:")
    print("  1. 运行测试脚本: python scripts/test_data_loading.py")
    print("  2. 检查数据加载是否正常")
    print("  3. 开始联邦学习训练")
    print()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
