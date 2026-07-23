r"""
准备联邦学习自定义数据划分脚本
=================================

目标:
- Client 1 (text_only): 30% BraTS 2020 文本数据，影像为空
- Client 2 (image_only): 100% BraTS 2018 影像数据
- Client 3 (multimodal): 50% BraTS 2020 成对数据（影像+文本）

输出: G:\FedSAM3-Cream存档\FedSAM3-Cream26.1.存档\data\federated_split
    ├── client1_text_only/
    │   └── dataset.json
    ├── client2_image_only/
    │   └── dataset.json
    └── client3_multimodal/
        └── dataset.json

使用 JSON 相对路径引用源数据，节省存储空间。
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


# ===================== 配置参数 =====================
BASE_DIR = Path(r"G:\FedSAM3-Cream存档\FedSAM3-Cream26.1.存档")
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source_images"
OUTPUT_DIR = DATA_DIR / "federated_split"

# BraTS 数据集路径
BRATS2020_JSON = SOURCE_DIR / "TextBraTS2020.json"
BRATS2018_JSON = SOURCE_DIR / "TextBraTS2018.json"
BRATS2020_IMAGE_DIR = SOURCE_DIR / "BraTS2020" / "BraTS2020_TrainingData"
BRATS2018_HGG_DIR = SOURCE_DIR / "BraTS2018" / "HGG"
BRATS2018_LGG_DIR = SOURCE_DIR / "BraTS2018" / "LGG"

# 划分比例
CLIENT1_RATIO = 0.30  # Client 1: 30% BraTS 2020 文本
CLIENT3_RATIO = 0.50  # Client 3: 50% BraTS 2020 成对数据

# 随机种子（可复现）
RANDOM_SEED = 42


def setup_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def load_json(json_path: Path) -> Dict[str, Any]:
    """加载 JSON 数据"""
    print(f"[INFO] 加载数据集: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] 共 {len(data['data'])} 条数据")
    return data


def split_brats2020(data: List[Dict], client1_ratio: float, client3_ratio: float):
    """
    划分 BraTS 2020 数据集

    Args:
        data: BraTS 2020 数据列表
        client1_ratio: Client 1 比例 (文本)
        client3_ratio: Client 3 比例 (成对)

    Returns:
        (client1_data, client3_data)
    """
    total_samples = len(data)
    client1_count = int(total_samples * client1_ratio)
    client3_count = int(total_samples * client3_ratio)

    print(f"\n[INFO] BraTS 2020 划分统计:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - Client 1 (文本): {client1_count} 样本 ({client1_ratio*100:.1f}%)")
    print(f"  - Client 3 (成对): {client3_count} 样本 ({client3_ratio*100:.1f}%)")
    print(f"  - 未使用: {total_samples - client1_count - client3_count} 样本")

    # 随机打乱
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 划分
    client1_data = shuffled_data[:client1_count]
    client3_data = shuffled_data[client1_count:client1_count + client3_count]

    return client1_data, client3_data


def create_client1_dataset(data: List[Dict], output_dir: Path, source_base: Path):
    """
    创建 Client 1 数据集（仅文本，影像为空）

    Client 1 特点: text_only
    - 保留 text_feature
    - 保留 label
    - image 路径保留但标记为 "empty"（后续加载时处理）
    """
    print(f"\n[INFO] 创建 Client 1 数据集 (text_only)")

    client_dir = output_dir / "client1_text_only"
    client_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据集 JSON
    client_data = []
    for item in data:
        new_item = {
            "fold": item.get("fold", 1),
            "image": "empty",  # 标记影像为空
            "label": item["label"],
            "text_feature": item["text_feature"]
        }
        client_data.append(new_item)

    dataset_json = {
        "data": client_data,
        "modality": "text_only",
        "description": "Client 1: 30% BraTS 2020 with text only (image is empty)"
    }

    # 保存 JSON
    output_json_path = client_dir / "dataset.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[SUCCESS] Client 1 数据集已保存: {output_json_path}")
    print(f"          样本数: {len(client_data)}")


def create_client2_dataset(data: List[Dict], output_dir: Path, source_base: Path):
    """
    创建 Client 2 数据集（100% BraTS 2018 影像）

    Client 2 特点: image_only
    - 保留 image (4 个模态)
    - 保留 label
    - text_feature 设为 "empty"
    """
    print(f"\n[INFO] 创建 Client 2 数据集 (image_only)")

    client_dir = output_dir / "client2_image_only"
    client_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据集 JSON
    client_data = []
    for item in data:
        new_item = {
            "fold": item.get("fold", 1),
            "image": item["image"],  # 保留影像路径
            "label": item["label"],
            "text_feature": "empty"  # 标记文本为空
        }
        client_data.append(new_item)

    dataset_json = {
        "data": client_data,
        "modality": "image_only",
        "description": "Client 2: 100% BraTS 2018 with image only (text is empty)"
    }

    # 保存 JSON
    output_json_path = client_dir / "dataset.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[SUCCESS] Client 2 数据集已保存: {output_json_path}")
    print(f"          样本数: {len(client_data)}")


def create_client3_dataset(data: List[Dict], output_dir: Path, source_base: Path):
    """
    创建 Client 3 数据集（50% BraTS 2020 成对数据）

    Client 3 特点: multimodal
    - 保留 image (4 个模态)
    - 保留 text_feature
    - 保留 label
    """
    print(f"\n[INFO] 创建 Client 3 数据集 (multimodal)")

    client_dir = output_dir / "client3_multimodal"
    client_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据集 JSON（完整保留）
    client_data = []
    for item in data:
        new_item = {
            "fold": item.get("fold", 1),
            "image": item["image"],
            "label": item["label"],
            "text_feature": item["text_feature"]
        }
        client_data.append(new_item)

    dataset_json = {
        "data": client_data,
        "modality": "multimodal",
        "description": "Client 3: 50% BraTS 2020 with paired image+text"
    }

    # 保存 JSON
    output_json_path = client_dir / "dataset.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[SUCCESS] Client 3 数据集已保存: {output_json_path}")
    print(f"          样本数: {len(client_data)}")


def create_summary_report(output_dir: Path):
    """创建汇总报告"""
    print(f"\n{'='*60}")
    print(f"联邦数据划分完成!")
    print(f"{'='*60}")

    # 读取每个客户端的数据集
    clients = [
        ("client1_text_only", "Client 1 (text_only)"),
        ("client2_image_only", "Client 2 (image_only)"),
        ("client3_multimodal", "Client 3 (multimodal)")
    ]

    total_samples = 0
    for client_name, display_name in clients:
        json_path = output_dir / client_name / "dataset.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sample_count = len(data['data'])
                modality = data['modality']
                total_samples += sample_count
                print(f"{display_name:30s}: {sample_count:4d} 样本 [{modality}]")

    print(f"{'='*60}")
    print(f"{'总计':30s}: {total_samples:4d} 样本")
    print(f"\n输出目录: {output_dir}")
    print(f"\n提示: JSON 文件使用相对路径引用源数据，无需复制实际文件。")


def main():
    """主函数"""
    print("="*60)
    print("联邦学习数据划分脚本")
    print("="*60)

    # 设置随机种子
    setup_seed(RANDOM_SEED)

    # 检查输入文件
    if not BRATS2020_JSON.exists():
        raise FileNotFoundError(f"未找到 BraTS 2020 JSON: {BRATS2020_JSON}")
    if not BRATS2018_JSON.exists():
        raise FileNotFoundError(f"未找到 BraTS 2018 JSON: {BRATS2018_JSON}")

    # 加载数据集
    brats2020_data = load_json(BRATS2020_JSON)
    brats2018_data = load_json(BRATS2018_JSON)

    # 划分 BraTS 2020
    client1_data, client3_data = split_brats2020(
        brats2020_data['data'],
        CLIENT1_RATIO,
        CLIENT3_RATIO
    )

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] 输出目录: {OUTPUT_DIR}")

    # 创建各客户端数据集
    create_client1_dataset(client1_data, OUTPUT_DIR, SOURCE_DIR)
    create_client2_dataset(brats2018_data['data'], OUTPUT_DIR, SOURCE_DIR)
    create_client3_dataset(client3_data, OUTPUT_DIR, SOURCE_DIR)

    # 生成汇总报告
    create_summary_report(OUTPUT_DIR)


if __name__ == "__main__":
    main()
