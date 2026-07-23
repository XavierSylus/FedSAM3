"""
准备 BraTS 联邦学习数据集
将 TextBraTS.json 和 BraTS2020 数据分割并分配给三个客户端：
- 客户端1：纯文本（TextBraTS.json）
- 客户端2：纯图像（BraTS2020）
- 客户端3：文本+图像（TextBraTS.json + BraTS2020）
"""
import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_textbrats_json(json_path: str) -> List[Dict]:
    """加载 TextBraTS.json 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['data']


def get_brats2020_training_cases(brats2020_dir: str) -> List[str]:
    """获取 BraTS2020 训练数据的所有病例"""
    # 尝试多个可能的路径
    possible_paths = [
        Path(brats2020_dir) / "BraTS2020_TrainingData" / "BraTS2020_Training",
        Path(brats2020_dir) / "BraTS2020_TrainingData",
        Path(brats2020_dir)
    ]
    
    training_dir = None
    for path in possible_paths:
        if path.exists():
            training_dir = path
            break
    
    if training_dir is None:
        raise ValueError(f"训练数据目录不存在，尝试的路径: {possible_paths}")
    
    # 获取所有病例文件夹
    case_dirs = [d.name for d in training_dir.iterdir() if d.is_dir()]
    return sorted(case_dirs), training_dir


def split_data(
    data_list: List,
    train_ratio: float = 0.4,
    test_ratio: float = 0.4,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    将数据列表分割为训练集、测试集和验证集
    
    Args:
        data_list: 数据列表
        train_ratio: 训练集比例（默认0.4）
        test_ratio: 测试集比例（默认0.4）
        val_ratio: 验证集比例（默认0.2）
    Returns:
        (train_list, test_list, val_list)
    """
    random.seed(seed)
    
    # 验证比例
    total_ratio = train_ratio + test_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和 {total_ratio} != 1.0，将自动归一化")
        train_ratio = train_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
    
    # 打乱数据
    shuffled_data = data_list.copy()
    random.shuffle(shuffled_data)
    
    # 计算划分点
    total = len(shuffled_data)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count
    
    train_data = shuffled_data[:train_count]
    test_data = shuffled_data[train_count:train_count + test_count]
    val_data = shuffled_data[train_count + test_count:]
    
    return train_data, test_data, val_data


def split_into_two_sets(data_list: List, seed: int = 42) -> Tuple[List, List]:
    """将数据列表平均分成两部分"""
    random.seed(seed)
    shuffled = data_list.copy()
    random.shuffle(shuffled)
    
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def copy_textbrats_entry(
    entry: Dict,
    source_base: str,
    dest_dir: Path,
    copy_mode: bool = True
) -> bool:
    """
    复制 TextBraTS.json 中的一个条目到目标目录
    
    Args:
        entry: JSON 条目
        source_base: 源数据基础目录
        dest_dir: 目标目录
    Returns:
        是否成功复制
    """
    try:
        # 创建目标目录
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图像文件
        for img_path in entry['image']:
            src_img = Path(source_base) / img_path
            if src_img.exists():
                dest_img = dest_dir / img_path.split('/')[-1]
                if copy_mode:
                    shutil.copy2(src_img, dest_img)
                else:
                    shutil.move(str(src_img), str(dest_img))
        
        # 复制标签文件
        if 'label' in entry:
            src_label = Path(source_base) / entry['label']
            if src_label.exists():
                dest_label = dest_dir / entry['label'].split('/')[-1]
                if copy_mode:
                    shutil.copy2(src_label, dest_label)
                else:
                    shutil.move(str(src_label), str(dest_label))
        
        # 复制文本特征文件
        if 'text_feature' in entry:
            src_text = Path(source_base) / entry['text_feature']
            if src_text.exists():
                dest_text = dest_dir / entry['text_feature'].split('/')[-1]
                if copy_mode:
                    shutil.copy2(src_text, dest_text)
                else:
                    shutil.move(str(src_text), str(dest_text))
        
        return True
    except Exception as e:
        print(f"复制条目失败: {e}")
        return False


def copy_brats2020_case(
    case_name: str,
    source_dir: Path,
    dest_dir: Path,
    copy_mode: bool = True
) -> bool:
    """
    复制 BraTS2020 的一个病例到目标目录
    
    Args:
        case_name: 病例名称（如 "BraTS20_Training_001"）
        source_dir: 源目录
        dest_dir: 目标目录
    Returns:
        是否成功复制
    """
    try:
        source_case_dir = source_dir / case_name
        if not source_case_dir.exists():
            print(f"警告: 病例目录不存在: {source_case_dir}")
            return False
        
        dest_case_dir = dest_dir / case_name
        dest_case_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所有文件
        for file_path in source_case_dir.glob("*"):
            if file_path.is_file():
                dest_file = dest_case_dir / file_path.name
                if copy_mode:
                    shutil.copy2(file_path, dest_file)
                else:
                    shutil.move(str(file_path), str(dest_file))
        
        return True
    except Exception as e:
        print(f"复制病例失败 {case_name}: {e}")
        return False


def prepare_federated_brats_data(
    textbrats_json_path: str,
    brats2020_dir: str,
    output_base_dir: str = "data",
    source_base_dir: str = None,
    seed: int = 42,
    copy_mode: bool = True
):
    """
    准备联邦学习数据集
    
    Args:
        textbrats_json_path: TextBraTS.json 文件路径
        brats2020_dir: BraTS2020 目录路径
        output_base_dir: 输出基础目录
        source_base_dir: 源数据基础目录（用于 TextBraTS.json 的文件路径）
        seed: 随机种子
        copy_mode: True=复制，False=移动
    """
    print("=" * 60)
    print("准备 BraTS 联邦学习数据集")
    print("=" * 60)
    
    # 加载 TextBraTS.json
    print("\n[1/6] 加载 TextBraTS.json...")
    textbrats_data = load_textbrats_json(textbrats_json_path)
    print(f"  TextBraTS.json 条目数: {len(textbrats_data)}")
    
    # 获取 BraTS2020 训练数据
    print("\n[2/6] 获取 BraTS2020 训练数据...")
    brats2020_cases, brats_training_dir = get_brats2020_training_cases(brats2020_dir)
    print(f"  BraTS2020 病例数: {len(brats2020_cases)}")
    print(f"  训练数据目录: {brats_training_dir}")
    
    # 分割 TextBraTS.json 数据
    print("\n[3/6] 分割 TextBraTS.json 数据...")
    text_train, text_test, text_val = split_data(
        textbrats_data,
        train_ratio=0.4,
        test_ratio=0.4,
        val_ratio=0.2,
        seed=seed
    )
    print(f"  训练集: {len(text_train)}")
    print(f"  测试集: {len(text_test)}")
    print(f"  验证集: {len(text_val)}")
    
    # 将训练集和测试集各分成两部分
    text_train_1, text_train_2 = split_into_two_sets(text_train, seed=seed)
    text_test_1, text_test_2 = split_into_two_sets(text_test, seed=seed+1)
    
    print(f"\n  TextBraTS 训练集分割:")
    print(f"    训练集1: {len(text_train_1)} (给客户端1)")
    print(f"    训练集2: {len(text_train_2)} (给客户端3)")
    print(f"  TextBraTS 测试集分割:")
    print(f"    测试集1: {len(text_test_1)} (给客户端1)")
    print(f"    测试集2: {len(text_test_2)} (给客户端3)")
    
    # 分割 BraTS2020 数据
    print("\n[4/6] 分割 BraTS2020 数据...")
    brats_train, brats_test, brats_val = split_data(
        brats2020_cases,
        train_ratio=0.4,
        test_ratio=0.4,
        val_ratio=0.2,
        seed=seed+100
    )
    print(f"  训练集: {len(brats_train)}")
    print(f"  测试集: {len(brats_test)}")
    print(f"  验证集: {len(brats_val)}")
    
    # 将训练集和测试集各分成两部分
    brats_train_1, brats_train_2 = split_into_two_sets(brats_train, seed=seed+100)
    brats_test_1, brats_test_2 = split_into_two_sets(brats_test, seed=seed+101)
    
    print(f"\n  BraTS2020 训练集分割:")
    print(f"    训练集1: {len(brats_train_1)} (给客户端2)")
    print(f"    训练集2: {len(brats_train_2)} (给客户端3)")
    print(f"  BraTS2020 测试集分割:")
    print(f"    测试集1: {len(brats_test_1)} (给客户端2)")
    print(f"    测试集2: {len(brats_test_2)} (给客户端3)")
    
    # 设置源数据基础目录
    if source_base_dir is None:
        source_base_dir = Path(textbrats_json_path).parent
    else:
        source_base_dir = Path(source_base_dir)
    
    brats_source_dir = brats_training_dir
    output_base = Path(output_base_dir)
    
    # 准备客户端1数据（纯文本）
    print("\n[5/6] 准备客户端1数据（纯文本）...")
    client1_train_dir = output_base / "train" / "client_1" / "private"
    client1_test_dir = output_base / "test" / "client_1" / "private"
    client1_val_dir = output_base / "val" / "client_1" / "private"
    
    # 训练集
    for entry in text_train_1:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client1_train_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    # 测试集
    for entry in text_test_1:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client1_test_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    # 验证集
    for entry in text_val:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client1_val_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    print(f"  客户端1: 训练集 {len(text_train_1)}, 测试集 {len(text_test_1)}, 验证集 {len(text_val)}")
    
    # 准备客户端2数据（纯图像）
    print("\n[6/6] 准备客户端2数据（纯图像）...")
    client2_train_dir = output_base / "train" / "client_2" / "private"
    client2_test_dir = output_base / "test" / "client_2" / "private"
    client2_val_dir = output_base / "val" / "client_2" / "private"
    
    # 训练集
    for case_name in brats_train_1:
        copy_brats2020_case(case_name, brats_source_dir, client2_train_dir, copy_mode)
    
    # 测试集
    for case_name in brats_test_1:
        copy_brats2020_case(case_name, brats_source_dir, client2_test_dir, copy_mode)
    
    # 验证集
    for case_name in brats_val:
        copy_brats2020_case(case_name, brats_source_dir, client2_val_dir, copy_mode)
    
    print(f"  客户端2: 训练集 {len(brats_train_1)}, 测试集 {len(brats_test_1)}, 验证集 {len(brats_val)}")
    
    # 准备客户端3数据（文本+图像）
    print("\n[7/6] 准备客户端3数据（文本+图像）...")
    client3_train_text_dir = output_base / "train" / "client_3" / "private" / "text"
    client3_train_image_dir = output_base / "train" / "client_3" / "private" / "image"
    client3_test_text_dir = output_base / "test" / "client_3" / "private" / "text"
    client3_test_image_dir = output_base / "test" / "client_3" / "private" / "image"
    client3_val_text_dir = output_base / "val" / "client_3" / "private" / "text"
    client3_val_image_dir = output_base / "val" / "client_3" / "private" / "image"
    
    # TextBraTS 训练集
    for entry in text_train_2:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client3_train_text_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    # TextBraTS 测试集
    for entry in text_test_2:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client3_test_text_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    # TextBraTS 验证集（使用全部验证集）
    for entry in text_val:
        case_name = entry['image'][0].split('/')[0]
        dest_dir = client3_val_text_dir / case_name
        copy_textbrats_entry(entry, str(source_base_dir), dest_dir, copy_mode)
    
    # BraTS2020 训练集
    for case_name in brats_train_2:
        copy_brats2020_case(case_name, brats_source_dir, client3_train_image_dir, copy_mode)
    
    # BraTS2020 测试集
    for case_name in brats_test_2:
        copy_brats2020_case(case_name, brats_source_dir, client3_test_image_dir, copy_mode)
    
    # BraTS2020 验证集（使用全部验证集）
    for case_name in brats_val:
        copy_brats2020_case(case_name, brats_source_dir, client3_val_image_dir, copy_mode)
    
    print(f"  客户端3:")
    print(f"    文本: 训练集 {len(text_train_2)}, 测试集 {len(text_test_2)}, 验证集 {len(text_val)}")
    print(f"    图像: 训练集 {len(brats_train_2)}, 测试集 {len(brats_test_2)}, 验证集 {len(brats_val)}")
    
    # 保存分配信息到 JSON
    print("\n保存分配信息...")
    allocation_info = {
        "textbrats": {
            "total": len(textbrats_data),
            "train_1": len(text_train_1),
            "train_2": len(text_train_2),
            "test_1": len(text_test_1),
            "test_2": len(text_test_2),
            "val": len(text_val)
        },
        "brats2020": {
            "total": len(brats2020_cases),
            "train_1": len(brats_train_1),
            "train_2": len(brats_train_2),
            "test_1": len(brats_test_1),
            "test_2": len(brats_test_2),
            "val": len(brats_val)
        },
        "clients": {
            "client_1": {
                "type": "text_only",
                "train": len(text_train_1),
                "test": len(text_test_1),
                "val": len(text_val)
            },
            "client_2": {
                "type": "image_only",
                "train": len(brats_train_1),
                "test": len(brats_test_1),
                "val": len(brats_val)
            },
            "client_3": {
                "type": "multimodal",
                "text": {
                    "train": len(text_train_2),
                    "test": len(text_test_2),
                    "val": len(text_val)
                },
                "image": {
                    "train": len(brats_train_2),
                    "test": len(brats_test_2),
                    "val": len(brats_val)
                }
            }
        }
    }
    
    info_path = output_base / "data_allocation.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(allocation_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n分配信息已保存到: {info_path}")
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print("\n数据目录结构:")
    print(f"  {output_base}/train/client_1/private/  (纯文本)")
    print(f"  {output_base}/train/client_2/private/  (纯图像)")
    print(f"  {output_base}/train/client_3/private/text/  (文本)")
    print(f"  {output_base}/train/client_3/private/image/  (图像)")
    print("\n类似地，test/ 和 val/ 目录也有相同的结构")


def main():
    parser = argparse.ArgumentParser(
        description="准备 BraTS 联邦学习数据集"
    )
    
    parser.add_argument(
        '--textbrats_json',
        type=str,
        default='data/source_images/TextBraTS.json',
        help='TextBraTS.json 文件路径（默认: data/source_images/TextBraTS.json）'
    )
    parser.add_argument(
        '--brats2020_dir',
        type=str,
        default='data/source_images/BraTS2020',
        help='BraTS2020 目录路径（默认: data/source_images/BraTS2020）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='输出基础目录（默认: data）'
    )
    parser.add_argument(
        '--source_base',
        type=str,
        default=None,
        help='源数据基础目录（用于 TextBraTS.json 的文件路径，默认: TextBraTS.json 所在目录）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='移动文件而不是复制（默认: 复制）'
    )
    
    args = parser.parse_args()
    
    prepare_federated_brats_data(
        textbrats_json_path=args.textbrats_json,
        brats2020_dir=args.brats2020_dir,
        output_base_dir=args.output,
        source_base_dir=args.source_base,
        seed=args.seed,
        copy_mode=not args.move
    )


if __name__ == "__main__":
    main()
