#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习数据集划分脚本 - 增强版
功能：
1. 清洗 TextBraTS.json 数据，过滤无效条目
2. 严格 70/10/20 划分（Train/Val/Test）
3. 物理文件复制到客户端目录
4. 生成干净的 JSON 索引文件
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


# ==================== 配置参数 ====================
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
NUM_CLIENTS = 3

# 路径配置
BASE_DIR = Path(__file__).parent.parent
SOURCE_JSON = BASE_DIR / "data" / "source_images" / "TextBraTS.json"
SOURCE_DATA_DIR = BASE_DIR / "data" / "source_images" / "BraTS2020" / "BraTS2020_TrainingData" / "BraTS2020_Training"
TARGET_DIR = BASE_DIR / "data" / "federated_split"

# ==================== 辅助函数 ====================

def clean_data(raw_data: List[Dict]) -> List[Dict]:
    """
    清洗数据：过滤掉无效条目
    - image 为空列表
    - label 为空或 None
    - text_feature 为空或 None
    """
    valid_data = []
    invalid_count = 0

    for idx, item in enumerate(raw_data):
        # 检查必要字段
        if not item.get("image") or not isinstance(item["image"], list) or len(item["image"]) == 0:
            print(f"⚠️  条目 {idx} 无效: image 为空")
            invalid_count += 1
            continue

        if not item.get("label"):
            print(f"⚠️  条目 {idx} 无效: label 为空")
            invalid_count += 1
            continue

        if not item.get("text_feature"):
            print(f"⚠️  条目 {idx} 无效: text_feature 为空")
            invalid_count += 1
            continue

        valid_data.append(item)

    print(f"\n✅ 数据清洗完成:")
    print(f"   - 原始条目数: {len(raw_data)}")
    print(f"   - 无效条目数: {invalid_count}")
    print(f"   - 有效条目数: {len(valid_data)}")

    return valid_data


def split_data(data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按照 70/10/20 比例划分数据集
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算划分点
    total = len(shuffled_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]

    print(f"\n📊 数据集划分 (随机种子={RANDOM_SEED}):")
    print(f"   - Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"   - Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"   - Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")

    return train_data, val_data, test_data


def get_case_folder_name(image_path: str) -> str:
    """
    从 image 路径提取病例文件夹名称
    例如: "BraTS20_Training_002/BraTS20_Training_002_flair.nii.gz" -> "BraTS20_Training_002"
    """
    return image_path.split('/')[0]


def copy_case_files(item: Dict, target_case_dir: Path, source_base: Path) -> Dict:
    """
    复制单个病例的所有文件到目标目录

    参数:
        item: JSON 中的数据条目
        target_case_dir: 目标病例目录
        source_base: 源数据根目录

    返回:
        更新后的数据条目（带有新的路径）
    """
    # 创建目标目录
    target_case_dir.mkdir(parents=True, exist_ok=True)

    # 提取病例文件夹名称
    case_folder = get_case_folder_name(item["image"][0])
    source_case_dir = source_base / case_folder

    # 新的条目（用于生成 JSON）
    new_item = {
        "fold": item.get("fold", 0),
        "image": [],
        "label": "",
        "text_feature": ""
    }

    # 1. 复制所有 image 文件（处理 .nii.gz -> .nii 扩展名差异）
    copied_images = []
    for img_path in item["image"]:
        # JSON 中是 .nii.gz，但实际文件是 .nii
        actual_img_path = img_path.replace('.nii.gz', '.nii')
        source_file = source_base / actual_img_path

        if source_file.exists():
            filename = source_file.name
            target_file = target_case_dir / filename
            shutil.copy2(source_file, target_file)
            copied_images.append(f"{case_folder}/{filename}")
        else:
            print(f"   ⚠️  文件不存在: {source_file}")

    new_item["image"] = copied_images

    # 2. 复制 label 文件
    label_path = item["label"].replace('.nii.gz', '.nii')
    source_label = source_base / label_path

    if source_label.exists():
        filename = source_label.name
        target_label = target_case_dir / filename
        shutil.copy2(source_label, target_label)
        new_item["label"] = f"{case_folder}/{filename}"
    else:
        print(f"   ⚠️  标签文件不存在: {source_label}")

    # 3. 复制 text_feature 文件（如果存在）
    text_feature_path = item["text_feature"]
    source_text = source_base / text_feature_path

    # text_feature 可能不存在，这里灵活处理
    if source_text.exists():
        filename = source_text.name
        target_text = target_case_dir / filename
        shutil.copy2(source_text, target_text)
        new_item["text_feature"] = f"{case_folder}/{filename}"
    else:
        # 如果 .npy 文件不存在，保留原始路径（供后续生成）
        new_item["text_feature"] = item["text_feature"]

    return new_item


def distribute_to_clients(data: List[Dict], split_type: str, source_base: Path, target_base: Path) -> List[Dict]:
    """
    轮询分配数据到客户端并复制文件

    参数:
        data: 数据列表
        split_type: "train" 或 "val"
        source_base: 源数据目录
        target_base: 目标根目录

    返回:
        包含所有客户端数据的列表（用于生成 JSON）
    """
    all_items = []

    for idx, item in enumerate(data):
        # 轮询分配到客户端
        client_id = (idx % NUM_CLIENTS) + 1
        client_name = f"client_{client_id}"

        # 目标路径: data/federated_split/{split_type}/{client_name}/private/{case_folder}/
        case_folder = get_case_folder_name(item["image"][0])
        target_case_dir = target_base / split_type / client_name / "private" / case_folder

        # 复制文件并获取新条目
        new_item = copy_case_files(item, target_case_dir, source_base)
        new_item["fold"] = client_id - 1  # fold 0, 1, 2 对应 client_1, 2, 3

        all_items.append(new_item)

    print(f"\n✅ {split_type.upper()} 数据分配完成:")
    for client_id in range(1, NUM_CLIENTS + 1):
        count = sum(1 for item in all_items if item["fold"] == client_id - 1)
        print(f"   - Client {client_id}: {count} 个病例")

    return all_items


def distribute_test_global(data: List[Dict], source_base: Path, target_base: Path) -> List[Dict]:
    """
    将测试集分配到全局目录

    参数:
        data: 测试数据列表
        source_base: 源数据目录
        target_base: 目标根目录

    返回:
        测试集数据列表（用于生成 JSON）
    """
    all_items = []

    for item in data:
        # 目标路径: data/federated_split/test/global/{case_folder}/
        case_folder = get_case_folder_name(item["image"][0])
        target_case_dir = target_base / "test" / "global" / case_folder

        # 复制文件并获取新条目
        new_item = copy_case_files(item, target_case_dir, source_base)
        new_item["fold"] = 0  # 测试集统一 fold=0

        all_items.append(new_item)

    print(f"\n✅ TEST 数据分配完成:")
    print(f"   - Global: {len(all_items)} 个病例")

    return all_items


def save_json(data: List[Dict], output_path: Path):
    """
    保存 JSON 文件
    """
    output_data = {"data": data}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"   📄 已保存: {output_path}")


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("🚀 联邦学习数据集划分脚本 - 增强版")
    print("=" * 70)

    # 1. 检查源文件
    if not SOURCE_JSON.exists():
        print(f"❌ 错误: 找不到源 JSON 文件: {SOURCE_JSON}")
        return

    if not SOURCE_DATA_DIR.exists():
        print(f"❌ 错误: 找不到源数据目录: {SOURCE_DATA_DIR}")
        return

    # 2. 删除旧的 federated_split 目录
    if TARGET_DIR.exists():
        print(f"\n🗑️  删除旧的输出目录: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
        print("   ✅ 删除成功")

    # 3. 读取并清洗数据
    print(f"\n📖 读取源 JSON: {SOURCE_JSON}")
    with open(SOURCE_JSON, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    raw_data = raw_json.get("data", [])
    valid_data = clean_data(raw_data)

    if len(valid_data) == 0:
        print("❌ 错误: 没有有效数据！")
        return

    # 4. 划分数据集
    train_data, val_data, test_data = split_data(valid_data)

    # 5. 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 6. 分发并复制文件
    print("\n" + "=" * 70)
    print("📦 开始物理文件分发...")
    print("=" * 70)

    # Train 集
    print("\n[1/3] 处理 TRAIN 数据...")
    train_items = distribute_to_clients(train_data, "train", SOURCE_DATA_DIR, TARGET_DIR)

    # Val 集
    print("\n[2/3] 处理 VAL 数据...")
    val_items = distribute_to_clients(val_data, "val", SOURCE_DATA_DIR, TARGET_DIR)

    # Test 集
    print("\n[3/3] 处理 TEST 数据...")
    test_items = distribute_test_global(test_data, SOURCE_DATA_DIR, TARGET_DIR)

    # 7. 生成 JSON 索引文件
    print("\n" + "=" * 70)
    print("📝 生成 JSON 索引文件...")
    print("=" * 70)

    save_json(train_items, TARGET_DIR / "train_split.json")
    save_json(val_items, TARGET_DIR / "val_split.json")
    save_json(test_items, TARGET_DIR / "test_split.json")

    # 8. 验证生成的 JSON（检查是否有空 image）
    print("\n" + "=" * 70)
    print("🔍 验证生成的 JSON 文件...")
    print("=" * 70)

    for json_file, items in [
        ("train_split.json", train_items),
        ("val_split.json", val_items),
        ("test_split.json", test_items)
    ]:
        empty_image_count = sum(1 for item in items if not item.get("image") or len(item["image"]) == 0)

        if empty_image_count > 0:
            print(f"   ⚠️  {json_file}: 发现 {empty_image_count} 个空 image 条目！")
        else:
            print(f"   ✅ {json_file}: 所有条目的 image 均有效")

    # 9. 完成
    print("\n" + "=" * 70)
    print("🎉 数据划分完成！")
    print("=" * 70)
    print(f"\n输出目录: {TARGET_DIR}")
    print(f"   - train_split.json: {len(train_items)} 个样本")
    print(f"   - val_split.json: {len(val_items)} 个样本")
    print(f"   - test_split.json: {len(test_items)} 个样本")
    print(f"   - 总计: {len(train_items) + len(val_items) + len(test_items)} 个样本")
    print("\n✅ 所有文件已物理复制到目标目录！")
    print("=" * 70)


if __name__ == "__main__":
    main()
