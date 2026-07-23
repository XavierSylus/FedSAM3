"""
自动创建 Public 数据集脚本
从剩余的 BraTS2020 数据中为 client_2 和 client_3 创建 public 数据（用于 CreamFL 对比学习）
"""

import os
import shutil
import random
from pathlib import Path
from typing import Set, List

# =============== 配置参数 ===============
SOURCE_DATA_DIR = Path("data/source_images/BraTS2020/BraTS2020_TrainingData/BraTS2020_TrainingData")
FEDERATED_SPLIT_DIR = Path("data/federated_split/train")

# 每个客户端分配的 public 样本数
PUBLIC_SAMPLES_PER_CLIENT = 150

# 目标客户端
TARGET_CLIENTS = ["client_2", "client_3"]

# 随机种子（保证可复现）
RANDOM_SEED = 42


def get_all_case_ids(source_dir: Path) -> Set[str]:
    """
    获取源数据目录中所有病例的 ID

    Returns:
        Set of case IDs (e.g., {'BraTS20_Training_001', ...})
    """
    case_ids = set()
    if not source_dir.exists():
        print(f"❌ 错误：源数据目录不存在: {source_dir}")
        return case_ids

    for case_dir in source_dir.iterdir():
        if case_dir.is_dir() and case_dir.name.startswith("BraTS"):
            case_ids.add(case_dir.name)

    print(f"✅ 源数据目录共有 {len(case_ids)} 个病例")
    return case_ids


def get_used_case_ids(federated_dir: Path) -> Set[str]:
    """
    获取所有客户端已使用的病例 ID

    Returns:
        Set of used case IDs
    """
    used_ids = set()

    for client_dir in federated_dir.iterdir():
        if not client_dir.is_dir() or not client_dir.name.startswith("client"):
            continue

        private_dir = client_dir / "private"
        if not private_dir.exists():
            continue

        for case_dir in private_dir.iterdir():
            if case_dir.is_dir() and case_dir.name.startswith("BraTS"):
                used_ids.add(case_dir.name)

    print(f"✅ 已使用的病例数: {len(used_ids)}")
    return used_ids


def select_public_cases(
    all_cases: Set[str],
    used_cases: Set[str],
    num_clients: int,
    samples_per_client: int
) -> dict:
    """
    从未使用的病例中随机选择 public 数据

    Returns:
        {client_id: [case_ids]}
    """
    # 计算可用病例
    available_cases = list(all_cases - used_cases)
    total_needed = num_clients * samples_per_client

    print(f"\n📊 数据分配统计:")
    print(f"  - 可用病例数: {len(available_cases)}")
    print(f"  - 需要病例数: {total_needed} ({num_clients} 客户端 × {samples_per_client} 样本)")

    if len(available_cases) < total_needed:
        print(f"⚠️  警告：可用病例不足！需要 {total_needed}，但只有 {len(available_cases)}")
        print(f"    将为每个客户端分配 {len(available_cases) // num_clients} 个样本")
        samples_per_client = len(available_cases) // num_clients

    # 随机打乱
    random.seed(RANDOM_SEED)
    random.shuffle(available_cases)

    # 分配给客户端
    allocation = {}
    start_idx = 0
    for i, client_id in enumerate(TARGET_CLIENTS):
        end_idx = start_idx + samples_per_client
        allocation[client_id] = available_cases[start_idx:end_idx]
        start_idx = end_idx
        print(f"  - {client_id}: {len(allocation[client_id])} 个样本")

    return allocation


def copy_case_to_public(
    case_id: str,
    source_dir: Path,
    target_dir: Path,
    copy_mask: bool = False
):
    """
    复制病例到 public 目录

    Args:
        case_id: 病例 ID
        source_dir: 源数据目录
        target_dir: 目标 public 目录
        copy_mask: 是否复制掩码（默认 False，因为 public 数据用于无监督对比学习）
    """
    source_case_dir = source_dir / case_id
    target_case_dir = target_dir / case_id

    if not source_case_dir.exists():
        print(f"  ⚠️  跳过：源病例不存在 {case_id}")
        return

    # 创建目标目录
    target_case_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    copied_files = 0
    for file in source_case_dir.iterdir():
        if not file.is_file():
            continue

        # 跳过掩码文件（除非明确要求复制）
        if not copy_mask and '_seg.nii' in file.name:
            continue

        # 复制文件
        target_file = target_case_dir / file.name
        if not target_file.exists():
            shutil.copy2(file, target_file)
            copied_files += 1

    return copied_files


def create_public_datasets(allocation: dict, source_dir: Path, federated_dir: Path):
    """
    为每个客户端创建 public 数据集
    """
    print(f"\n🚀 开始创建 public 数据集...")

    for client_id, case_ids in allocation.items():
        print(f"\n📦 处理 {client_id}:")
        print(f"  - 目标样本数: {len(case_ids)}")

        # 创建 public 目录
        public_dir = federated_dir / client_id / "public"
        public_dir.mkdir(parents=True, exist_ok=True)

        # 复制病例
        success_count = 0
        for i, case_id in enumerate(case_ids, 1):
            copied = copy_case_to_public(
                case_id=case_id,
                source_dir=source_dir,
                target_dir=public_dir,
                copy_mask=False  # Public 数据不需要掩码
            )
            if copied:
                success_count += 1

            # 每 50 个打印一次进度
            if i % 50 == 0:
                print(f"    进度: {i}/{len(case_ids)}")

        print(f"  ✅ {client_id} - 成功创建 {success_count} 个 public 样本")
        print(f"     路径: {public_dir}")


def verify_public_data(federated_dir: Path):
    """
    验证 public 数据是否创建成功
    """
    print(f"\n🔍 验证 Public 数据...")

    for client_id in TARGET_CLIENTS:
        public_dir = federated_dir / client_id / "public"

        if not public_dir.exists():
            print(f"  ❌ {client_id}: public 目录不存在")
            continue

        # 统计病例数
        case_dirs = [d for d in public_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")]

        # 统计文件数
        total_files = 0
        for case_dir in case_dirs:
            total_files += len(list(case_dir.glob("*.nii*")))

        print(f"  ✅ {client_id}: {len(case_dirs)} 个病例, 共 {total_files} 个文件")


def main():
    print("=" * 80)
    print("🎯 FedSAM3-Cream Public 数据集创建脚本")
    print("=" * 80)

    # 1. 获取所有病例
    all_cases = get_all_case_ids(SOURCE_DATA_DIR)
    if not all_cases:
        print("❌ 无法找到源数据，请检查路径配置")
        return

    # 2. 获取已使用的病例
    used_cases = get_used_case_ids(FEDERATED_SPLIT_DIR)

    # 3. 分配 public 数据
    allocation = select_public_cases(
        all_cases=all_cases,
        used_cases=used_cases,
        num_clients=len(TARGET_CLIENTS),
        samples_per_client=PUBLIC_SAMPLES_PER_CLIENT
    )

    # 4. 创建 public 数据集
    create_public_datasets(
        allocation=allocation,
        source_dir=SOURCE_DATA_DIR,
        federated_dir=FEDERATED_SPLIT_DIR
    )

    # 5. 验证
    verify_public_data(FEDERATED_SPLIT_DIR)

    print("\n" + "=" * 80)
    print("✅ Public 数据集创建完成！")
    print("=" * 80)
    print("\n💡 下一步:")
    print("  1. 检查 data/federated_split/train/client_2/public/")
    print("  2. 检查 data/federated_split/train/client_3/public/")
    print("  3. 运行训练脚本测试数据加载")


if __name__ == "__main__":
    main()
