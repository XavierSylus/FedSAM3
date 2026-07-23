#!/usr/bin/env python3
"""
FedSAM3-Cream 数据集重新划分脚本
Multi-modal Federated Medical Image Segmentation

功能：
1. 读取 BraTS 2020 训练集 (train_split.json)
2. 按 fold (0/1/2) 严格分配到三个异构客户端
3. 模拟 CreamFL 的模态缺失场景（绝对隔离）

作者: FedSAM3-Cream Team
日期: 2026-02-27
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import sys


def validate_input_data(train_json_path: Path) -> Dict[str, Any]:
    """
    验证输入数据的合法性

    Args:
        train_json_path: 训练集 JSON 文件路径

    Returns:
        完整的训练集数据字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据格式错误
    """
    if not train_json_path.exists():
        raise FileNotFoundError(
            f"❌ 训练集文件不存在: {train_json_path}\n"
            f"请确保已运行数据预处理脚本。"
        )

    print(f"  读取文件: {train_json_path}")

    with open(train_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' not in data:
        raise ValueError("❌ train_split.json 缺少 'data' 字段")

    samples = data['data']
    print(f"  ✓ 读取到 {len(samples)} 个训练样本")

    # 统计 fold 分布
    fold_counts = defaultdict(int)
    for idx, sample in enumerate(samples):
        if 'fold' not in sample:
            raise ValueError(f"❌ 第 {idx} 个样本缺少 'fold' 字段")

        if 'image' not in sample:
            raise ValueError(f"❌ 第 {idx} 个样本缺少 'image' 字段")

        if 'text_feature' not in sample:
            raise ValueError(f"❌ 第 {idx} 个样本缺少 'text_feature' 字段")

        fold_counts[sample['fold']] += 1

    print(f"  ✓ Fold 分布: {dict(sorted(fold_counts.items()))}")

    # 验证 fold 完整性
    expected_folds = {0, 1, 2}
    actual_folds = set(fold_counts.keys())
    if actual_folds != expected_folds:
        raise ValueError(
            f"❌ Fold 分布异常\n"
            f"期望: {expected_folds}\n"
            f"实际: {actual_folds}"
        )

    return data


def create_client_dataset(
    samples: List[Dict],
    modality: str,
    description: str
) -> Dict[str, Any]:
    """
    创建客户端数据集

    Args:
        samples: 原始样本列表
        modality: 模态类型 ('text_only' | 'image_only' | 'multimodal')
        description: 数据集描述

    Returns:
        处理后的数据集字典
    """
    processed_samples = []

    for sample in samples:
        # 深拷贝避免修改原始数据
        new_sample = sample.copy()

        if modality == 'text_only':
            # ========================================
            # Client 1 (text_only): 仅保留文本特征
            # ========================================
            new_sample['image'] = 'empty'
            # text_feature 和 label 保持不变

        elif modality == 'image_only':
            # ========================================
            # Client 2 (image_only): 仅保留图像
            # ========================================
            new_sample['text_feature'] = 'empty'
            # image 和 label 保持不变

        elif modality == 'multimodal':
            # ========================================
            # Client 3 (multimodal): 完整保留所有模态
            # ========================================
            # 保持原样，不做任何修改
            pass

        else:
            raise ValueError(f"❌ 未知的 modality 类型: {modality}")

        processed_samples.append(new_sample)

    return {
        'data': processed_samples,
        'modality': modality,
        'description': description
    }


def verify_client_dataset(
    dataset_path: Path,
    expected_modality: str,
    expected_count: int
) -> None:
    """
    验证客户端数据集的正确性

    Args:
        dataset_path: 数据集 JSON 文件路径
        expected_modality: 预期的模态类型
        expected_count: 预期的样本数量

    Raises:
        AssertionError: 验证失败
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        client_data = json.load(f)

    # 验证样本数量
    actual_count = len(client_data['data'])
    assert actual_count == expected_count, \
        f"样本数量不匹配: 期望 {expected_count}, 实际 {actual_count}"

    # 验证 modality 字段
    assert client_data['modality'] == expected_modality, \
        f"Modality 不匹配: 期望 {expected_modality}, 实际 {client_data['modality']}"

    # 验证第一个样本的字段
    if actual_count > 0:
        sample = client_data['data'][0]

        if expected_modality == 'text_only':
            assert sample['image'] == 'empty', \
                f"text_only 客户端的 image 应为 'empty', 实际: {sample['image']}"
            assert sample['text_feature'] != 'empty', \
                f"text_only 客户端应保留 text_feature"

        elif expected_modality == 'image_only':
            assert sample['image'] != 'empty', \
                f"image_only 客户端应保留 image"
            assert sample['text_feature'] == 'empty', \
                f"image_only 客户端的 text_feature 应为 'empty', 实际: {sample['text_feature']}"

        elif expected_modality == 'multimodal':
            assert sample['image'] != 'empty', \
                f"multimodal 客户端应保留 image"
            assert sample['text_feature'] != 'empty', \
                f"multimodal 客户端应保留 text_feature"


def main():
    """主函数"""
    print("=" * 80)
    print("FedSAM3-Cream 数据集重新划分")
    print("Multi-modal Federated Medical Image Segmentation")
    print("=" * 80)
    print()

    # ==================== 路径配置 ====================
    project_root = Path(__file__).parent.parent
    federated_root = project_root / 'data' / 'federated_split'
    train_json = federated_root / 'train_split.json'

    # 确保不触碰验证集和测试集
    val_json = federated_root / 'val_split.json'
    test_json = federated_root / 'test_split.json'

    # ==================== Step 1: 验证输入 ====================
    print("[1/5] 验证输入数据...")
    try:
        data = validate_input_data(train_json)
        all_samples = data['data']
    except Exception as e:
        print(f"\n❌ 输入验证失败: {e}")
        sys.exit(1)

    # 确认不会修改验证集和测试集
    print(f"\n  ⚠ 确认以下文件将保持不变:")
    print(f"    - {val_json.name}")
    print(f"    - {test_json.name}")

    # ==================== Step 2: 按 fold 分组 ====================
    print("\n[2/5] 按 fold 分组数据...")
    fold_groups = defaultdict(list)
    for sample in all_samples:
        fold_groups[sample['fold']].append(sample)

    print(f"  Fold 0: {len(fold_groups[0])} 样本 → Client 1 (text_only)")
    print(f"  Fold 1: {len(fold_groups[1])} 样本 → Client 2 (image_only)")
    print(f"  Fold 2: {len(fold_groups[2])} 样本 → Client 3 (multimodal)")

    # ==================== Step 3: 客户端配置 ====================
    print("\n[3/5] 创建客户端配置...")

    client_configs = [
        {
            'name': 'client1_text_only',
            'fold': 0,
            'modality': 'text_only',
            'description': 'Client 1: BraTS 2020 text-only features (Fold 0, 33.3%)'
        },
        {
            'name': 'client2_image_only',
            'fold': 1,
            'modality': 'image_only',
            'description': 'Client 2: BraTS 2020 image-only data (Fold 1, 33.3%)'
        },
        {
            'name': 'client3_multimodal',
            'fold': 2,
            'modality': 'multimodal',
            'description': 'Client 3: BraTS 2020 paired image+text (Fold 2, 33.3%)'
        }
    ]

    for config in client_configs:
        print(f"  - {config['name']}: {config['description']}")

    # ==================== Step 4: 生成客户端数据集 ====================
    print("\n[4/5] 生成客户端数据集...")

    for config in client_configs:
        client_dir = federated_root / config['name']

        # 如果目录已存在，先备份
        if client_dir.exists():
            backup_dir = client_dir.parent / f"{config['name']}_backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            print(f"  ⚠ {config['name']} 已存在，备份到 {backup_dir.name}/")
            shutil.move(str(client_dir), str(backup_dir))

        # 创建新目录
        client_dir.mkdir(parents=True, exist_ok=True)

        # 生成数据集
        dataset = create_client_dataset(
            samples=fold_groups[config['fold']],
            modality=config['modality'],
            description=config['description']
        )

        # 保存 JSON
        output_json = client_dir / 'dataset.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ {config['name']}: {len(dataset['data'])} 样本 → {output_json.name}")

    # ==================== Step 5: 严格验证输出 ====================
    print("\n[5/5] 验证输出数据完整性...")

    total_samples = 0
    try:
        for config in client_configs:
            dataset_path = federated_root / config['name'] / 'dataset.json'
            expected_count = len(fold_groups[config['fold']])

            verify_client_dataset(
                dataset_path=dataset_path,
                expected_modality=config['modality'],
                expected_count=expected_count
            )

            total_samples += expected_count
            print(f"  ✓ {config['name']}: 验证通过 ({expected_count} 样本)")

        # 确保样本总数守恒
        assert total_samples == len(all_samples), \
            f"样本总数不匹配: {total_samples} != {len(all_samples)}"

        print(f"\n  ✓ 样本总数守恒: {total_samples} == {len(all_samples)}")

    except AssertionError as e:
        print(f"\n❌ 验证失败: {e}")
        sys.exit(1)

    # ==================== 完成 ====================
    print("\n" + "=" * 80)
    print("✅ 数据集重新划分完成！")
    print("=" * 80)
    print(f"\n📁 输出目录: {federated_root.absolute()}")
    print("\n📊 数据统计:")
    print(f"  - Client 1 (text_only):    {len(fold_groups[0])} 样本 (33.3%)")
    print(f"  - Client 2 (image_only):   {len(fold_groups[1])} 样本 (33.3%)")
    print(f"  - Client 3 (multimodal):   {len(fold_groups[2])} 样本 (33.3%)")
    print(f"  - 总计:                    {total_samples} 样本 (100%)")
    print("\n📝 下一步:")
    print("  1. 检查生成的 client*/dataset.json 文件")
    print("  2. 修改 Dataset 类以处理 'empty' 标识（见任务 2）")
    print("  3. 更新 setup_serial_clients.py 的客户端配置")
    print("  4. 运行训练脚本验证数据加载")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
