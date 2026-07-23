"""
将原始数据集划分为训练集、验证集和测试集
用于准备联邦学习的数据集
"""
import os
import shutil
import random
from pathlib import Path
from typing import Tuple
import argparse


def split_train_val_test(
    source_images_dir: str,
    source_masks_dir: str,
    output_base_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_mode: bool = True
) -> dict:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        source_images_dir: 源图像目录路径
        source_masks_dir: 源掩码目录路径
        output_base_dir: 输出基础目录（如 "data"）
        train_ratio: 训练集比例（默认0.7，即70%）
        val_ratio: 验证集比例（默认0.15，即15%）
        test_ratio: 测试集比例（默认0.15，即15%）
        seed: 随机种子
        copy_mode: True=复制文件，False=移动文件
    
    Returns:
        包含每个数据集统计的字典
    """
    # 设置随机种子
    random.seed(seed)
    
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和 {total_ratio} != 1.0，将自动归一化")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # 转换为Path对象
    source_images = Path(source_images_dir)
    source_masks = Path(source_masks_dir)
    output_base = Path(output_base_dir)
    
    # 检查源目录
    if not source_images.exists():
        raise ValueError(f"源图像目录不存在: {source_images_dir}")
    if not source_masks.exists():
        raise ValueError(f"源掩码目录不存在: {source_masks_dir}")
    
    # 创建输出目录
    train_images_dir = output_base / "train_raw" / "images"
    train_masks_dir = output_base / "train_raw" / "masks"
    val_images_dir = output_base / "val_raw" / "images"
    val_masks_dir = output_base / "val_raw" / "masks"
    test_images_dir = output_base / "test_raw" / "images"
    test_masks_dir = output_base / "test_raw" / "masks"
    
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, 
                     val_masks_dir, test_images_dir, test_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_images.glob(f"*{ext}")))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {source_images_dir} 中未找到图像文件")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算划分点
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count
    
    print("=" * 60)
    print("数据集划分: 训练集 / 验证集 / 测试集")
    print("=" * 60)
    print(f"总图像数: {total_files}")
    print(f"\n划分结果:")
    print(f"  训练集: {train_count} ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_count} ({val_ratio*100:.1f}%)")
    print(f"  测试集: {test_count} ({test_ratio*100:.1f}%)")
    
    # 划分文件
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]
    
    results = {
        'train': {'images': 0, 'masks': 0},
        'val': {'images': 0, 'masks': 0},
        'test': {'images': 0, 'masks': 0}
    }
    
    # 处理训练集
    print(f"\n处理训练集...")
    for img_file in train_images:
        dest_img = train_images_dir / img_file.name
        if copy_mode:
            shutil.copy2(img_file, dest_img)
        else:
            shutil.move(str(img_file), str(dest_img))
        results['train']['images'] += 1
        
        # 查找对应的掩码文件
        mask_file = find_mask_file(img_file, source_masks)
        if mask_file:
            dest_mask = train_masks_dir / mask_file.name
            if copy_mode:
                shutil.copy2(mask_file, dest_mask)
            else:
                shutil.move(str(mask_file), str(dest_mask))
            results['train']['masks'] += 1
    
    # 处理验证集
    print(f"\n处理验证集...")
    for img_file in val_images:
        dest_img = val_images_dir / img_file.name
        if copy_mode:
            shutil.copy2(img_file, dest_img)
        else:
            shutil.move(str(img_file), str(dest_img))
        results['val']['images'] += 1
        
        # 查找对应的掩码文件
        mask_file = find_mask_file(img_file, source_masks)
        if mask_file:
            dest_mask = val_masks_dir / mask_file.name
            if copy_mode:
                shutil.copy2(mask_file, dest_mask)
            else:
                shutil.move(str(mask_file), str(dest_mask))
            results['val']['masks'] += 1
    
    # 处理测试集
    print(f"\n处理测试集...")
    for img_file in test_images:
        dest_img = test_images_dir / img_file.name
        if copy_mode:
            shutil.copy2(img_file, dest_img)
        else:
            shutil.move(str(img_file), str(dest_img))
        results['test']['images'] += 1
        
        # 查找对应的掩码文件
        mask_file = find_mask_file(img_file, source_masks)
        if mask_file:
            dest_mask = test_masks_dir / mask_file.name
            if copy_mode:
                shutil.copy2(mask_file, dest_mask)
            else:
                shutil.move(str(mask_file), str(dest_mask))
            results['test']['masks'] += 1
    
    return results


def find_mask_file(img_file: Path, masks_dir: Path) -> Path:
    """
    查找图像对应的掩码文件
    
    Args:
        img_file: 图像文件路径
        masks_dir: 掩码目录路径
    Returns:
        掩码文件路径，如果未找到返回None
    """
    possible_mask_names = [
        img_file.name,  # 相同文件名
        f"mask_{img_file.stem}.png",
        f"{img_file.stem}_mask.png",
        f"{img_file.stem}.png"
    ]
    
    for mask_name in possible_mask_names:
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            return mask_path
    
    return None


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="将数据集划分为训练集、验证集和测试集"
    )
    
    parser.add_argument(
        '--source_images',
        type=str,
        required=True,
        help='源图像目录路径'
    )
    parser.add_argument(
        '--source_masks',
        type=str,
        required=True,
        help='源掩码目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='输出基础目录（默认: data）'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='训练集比例（默认: 0.7，即70%%）'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例（默认: 0.15，即15%%）'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='测试集比例（默认: 0.15，即15%%）'
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
    
    print("=" * 60)
    print("数据集划分工具: 训练集 / 验证集 / 测试集")
    print("=" * 60)
    print(f"源图像目录: {args.source_images}")
    print(f"源掩码目录: {args.source_masks}")
    print(f"输出目录: {args.output}")
    print(f"\n划分比例:")
    print(f"  训练集: {args.train_ratio*100:.1f}%")
    print(f"  验证集: {args.val_ratio*100:.1f}%")
    print(f"  测试集: {args.test_ratio*100:.1f}%")
    print(f"模式: {'移动' if args.move else '复制'}")
    print("=" * 60)
    
    try:
        results = split_train_val_test(
            source_images_dir=args.source_images,
            source_masks_dir=args.source_masks,
            output_base_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_mode=not args.move
        )
        
        print("\n" + "=" * 60)
        print("数据集划分完成！")
        print("=" * 60)
        print("\n最终统计:")
        print(f"\n训练集:")
        print(f"  图像: {results['train']['images']} 张")
        print(f"  掩码: {results['train']['masks']} 对")
        print(f"\n验证集:")
        print(f"  图像: {results['val']['images']} 张")
        print(f"  掩码: {results['val']['masks']} 对")
        print(f"\n测试集:")
        print(f"  图像: {results['test']['images']} 张")
        print(f"  掩码: {results['test']['masks']} 对")
        
        print(f"\n输出目录结构:")
        print(f"  {args.output}/train_raw/images/")
        print(f"  {args.output}/train_raw/masks/")
        print(f"  {args.output}/val_raw/images/")
        print(f"  {args.output}/val_raw/masks/")
        print(f"  {args.output}/test_raw/images/")
        print(f"  {args.output}/test_raw/masks/")
        
        print(f"\n下一步:")
        print(f"  1. 使用 prepare_federated_dataset.py 准备训练集的联邦学习数据")
        print(f"  2. （可选）使用 prepare_federated_dataset.py 准备验证集和测试集")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
