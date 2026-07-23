#!/usr/bin/env python3
"""
重建干净验证集

从 test/client_X/private/image/ 中取未被 train 使用的病例，
复制为 val/client_X/private/BraTS20_Training_XXX/ 的平铺结构。
保证 val ∩ train = ∅，消除数据泄漏。
不修改任何 train 或 test 数据，无需重跑训练。
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEDERATED_SPLIT = PROJECT_ROOT / "data" / "federated_split"
VAL_CASES_PER_CLIENT = 20


def collect_train_case_names() -> set:
    """收集所有 train 客户端已使用的病例名称"""
    used = set()
    train_root = FEDERATED_SPLIT / "train"
    for client_dir in train_root.iterdir():
        private_dir = client_dir / "private"
        if private_dir.exists():
            for case in private_dir.iterdir():
                if case.is_dir():
                    used.add(case.name)
    print(f"[OK] Train 已用病例总数: {len(used)}")
    return used


def find_test_cases(client_name: str) -> list:
    """从 test/client_X/private/image/ 获取病例目录列表（兼容两种目录结构）"""
    for subdir in ["image", ""]:
        candidate = FEDERATED_SPLIT / "test" / client_name / "private"
        if subdir:
            candidate = candidate / subdir
        if candidate.exists():
            cases = sorted([
                d for d in candidate.iterdir()
                if d.is_dir() and "BraTS" in d.name
            ])
            if cases:
                print(f"[OK] test/{client_name} 来源: {candidate}，共 {len(cases)} 个病例")
                return cases
    print(f"[WARN] test/{client_name}/private/ 无法找到 BraTS 病例目录")
    return []


def copy_case(src: Path, dst_client_private: Path):
    """复制 flair 图像和 seg 标签（验证不需要文本特征）"""
    case_dst = dst_client_private / src.name
    case_dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if "_flair.nii" in f.name or "_seg.nii" in f.name:
            shutil.copy2(f, case_dst / f.name)


def verify_no_overlap(val_cases: set, train_cases: set):
    overlap = val_cases & train_cases
    if overlap:
        raise RuntimeError(f"发现 val/train 重叠病例: {overlap}")
    print(f"[OK] 零重叠验证通过，val 共 {len(val_cases)} 个病例")


def main():
    train_cases = collect_train_case_names()
    all_val_names = set()
    used_names = set()

    for client_name in ["client_2", "client_3"]:
        test_cases = find_test_cases(client_name)
        clean_cases = [
            c for c in test_cases
            if c.name not in train_cases and c.name not in used_names
        ]

        n = min(VAL_CASES_PER_CLIENT, len(clean_cases))
        if n < 5:
            raise RuntimeError(
                f"{client_name} 可用干净病例不足（仅 {len(clean_cases)} 个）。"
                f"请检查 test/{client_name}/private/ 目录结构。"
            )

        old = FEDERATED_SPLIT / "val" / client_name / "private"
        if old.exists():
            shutil.rmtree(old)
            print(f"[清除] 旧 val/{client_name}/private/")

        val_private = FEDERATED_SPLIT / "val" / client_name / "private"
        for case_dir in clean_cases[:n]:
            copy_case(case_dir, val_private)
            all_val_names.add(case_dir.name)
            used_names.add(case_dir.name)

        print(f"[OK] val/{client_name}/private: {n} 个病例")

    verify_no_overlap(all_val_names, train_cases)
    print("\n验证集重建完成，可直接重跑实验。")


if __name__ == "__main__":
    main()
