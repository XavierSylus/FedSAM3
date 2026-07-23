#!/usr/bin/env python3
"""
FedSAM3-Cream 数据加载测试脚本
验证三客户端异构数据加载

功能：
1. 测试 Client 1 (text_only) 的数据加载
2. 测试 Client 2 (image_only) 的数据加载
3. 测试 Client 3 (multimodal) 的数据加载
4. 验证 "empty" 标记是否被正确处理为全零张量

作者: FedSAM3-Cream Team
日期: 2026-02-27
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from data.multimodal_dataset import MultimodalMedicalDataset


def test_client_loading(
    client_name: str,
    expected_modality: str,
    data_root: str = "data/source_images"
):
    """
    测试单个客户端的数据加载

    Args:
        client_name: 客户端名称（如 'client1_text_only'）
        expected_modality: 预期的模态类型
        data_root: BraTS 2020 数据的根目录
    """
    print(f"\n{'='*70}")
    print(f"测试 {client_name} ({expected_modality})")
    print(f"{'='*70}")

    # 构建路径
    json_path = project_root / "data" / "federated_split" / client_name / "dataset.json"

    if not json_path.exists():
        print(f"❌ 错误: 数据集文件不存在: {json_path}")
        print(f"请先运行: python scripts/recreate_client_splits.py")
        return False

    try:
        # 创建数据集
        dataset = MultimodalMedicalDataset(
            json_path=str(json_path),
            data_root=data_root,
            image_size=256,
            embed_dim=768,
            max_samples=5  # 只测试前 5 个样本
        )

        print(f"✓ 数据集加载成功: {len(dataset)} 个样本")

        # 创建 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Windows 上设为 0
        )

        print(f"✓ DataLoader 创建成功")

        # 遍历数据
        for batch_idx, (images, masks, text_features) in enumerate(loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  - images: {images.shape}")
            print(f"    └─ min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
            print(f"  - masks: {masks.shape}")
            print(f"    └─ unique values={torch.unique(masks).tolist()}")
            print(f"  - text_features: {text_features.shape}")
            print(f"    └─ L2 norm (mean)={text_features.norm(dim=1).mean():.4f}")

            # ===========================================
            # 🔥 关键验证：模态特性
            # ===========================================
            if expected_modality == 'text_only':
                # Client 1: 图像应该全为零
                image_sum = images.abs().sum().item()
                if image_sum == 0:
                    print(f"  ✅ 验证通过: 图像为全零张量 (text_only)")
                else:
                    print(f"  ❌ 验证失败: text_only 客户端的图像应该全为零, 实际 sum={image_sum}")
                    return False

                # 文本特征应该非零（真实数据）
                text_sum = text_features.abs().sum().item()
                if text_sum > 0:
                    print(f"  ✅ 验证通过: 文本特征非零 (有真实数据)")
                else:
                    print(f"  ⚠️  警告: text_only 客户端的文本特征为零")

            elif expected_modality == 'image_only':
                # Client 2: 文本特征应该全为零
                text_sum = text_features.abs().sum().item()
                if text_sum == 0:
                    print(f"  ✅ 验证通过: 文本特征为全零向量 (image_only)")
                else:
                    print(f"  ❌ 验证失败: image_only 客户端的文本特征应该全为零, 实际 sum={text_sum}")
                    return False

                # 图像应该非零（真实数据）
                image_sum = images.abs().sum().item()
                if image_sum > 0:
                    print(f"  ✅ 验证通过: 图像非零 (有真实数据)")
                else:
                    print(f"  ⚠️  警告: image_only 客户端的图像为零")

            elif expected_modality == 'multimodal':
                # Client 3: 图像和文本特征都应该非零
                image_sum = images.abs().sum().item()
                text_sum = text_features.abs().sum().item()

                if image_sum > 0 and text_sum > 0:
                    print(f"  ✅ 验证通过: 图像和文本特征都非零 (multimodal)")
                else:
                    print(f"  ❌ 验证失败: multimodal 客户端应该有真实数据")
                    print(f"     图像 sum={image_sum}, 文本 sum={text_sum}")
                    return False

            # 只测试 2 个 batch
            if batch_idx >= 1:
                break

        print(f"\n✅ {client_name} 数据加载测试通过！")
        return True

    except FileNotFoundError as e:
        print(f"❌ 文件不存在错误: {e}")
        print(f"\n请确认:")
        print(f"  1. BraTS 2020 数据位于: {data_root}")
        print(f"  2. 文本特征文件 (.npy) 已提取")
        return False

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("FedSAM3-Cream 数据加载测试")
    print("异构客户端验证")
    print("=" * 70)

    # 配置数据根目录（根据你的实际路径调整）
    data_root = "data/source_images/BraTS2020/BraTS2020_TrainingData/BraTS2020_Training"  # ← BraTS 2020 数据实际路径

    # 测试配置
    test_cases = [
        ('client1_text_only', 'text_only'),
        ('client2_image_only', 'image_only'),
        ('client3_multimodal', 'multimodal'),
    ]

    # 运行测试
    results = {}
    for client_name, expected_modality in test_cases:
        success = test_client_loading(client_name, expected_modality, data_root)
        results[client_name] = success

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    all_passed = True
    for client_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {client_name:25s} : {status}")
        if not success:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("🎉 所有客户端数据加载测试通过！")
        print()
        print("下一步:")
        print("  1. 更新 setup_serial_clients.py")
        print("  2. 更新 configs/baseline.yaml")
        print("  3. 运行联邦学习训练")
        print()
        return 0
    else:
        print("❌ 部分测试失败，请检查上述错误信息")
        print()
        print("常见问题:")
        print("  1. 数据集未划分 → 运行: python scripts/recreate_client_splits.py")
        print("  2. data_root 路径错误 → 修改本文件第 155 行的 data_root 变量")
        print("  3. 文本特征文件缺失 → 确保 .npy 文件存在于 data/source_images/")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
