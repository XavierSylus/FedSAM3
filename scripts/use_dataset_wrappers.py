"""
使用数据集包装器的示例
演示如何将 wrappers.py 的功能集成到现有数据加载器中
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from src.data.dataset_wrappers import TrainDataset, ValDataset, wrap_dataset
from data.dataset_loader import MedicalImageDataset, create_data_loaders


def example_1_basic_wrapper():
    """示例 1: 基本包装器使用"""
    print("=" * 60)
    print("示例 1: 基本包装器使用")
    print("=" * 60)
    
    # 创建基础数据集
    base_dataset = MedicalImageDataset(
        data_dir="data/train/client_1",
        mode="private",
        image_size=1024,
        has_mask=True
    )
    
    # 使用训练数据集包装器
    train_dataset = TrainDataset(
        base_dataset,
        inp_size=512,
        augment=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 测试加载
    for batch in train_loader:
        print(f"输入形状: {batch['inp'].shape}")
        print(f"掩码形状: {batch['gt'].shape}")
        break
    
    print("✓ 基本包装器使用成功")


def example_2_validation_wrapper():
    """示例 2: 验证数据集包装器"""
    print("=" * 60)
    print("示例 2: 验证数据集包装器")
    print("=" * 60)
    
    # 创建基础数据集
    base_dataset = MedicalImageDataset(
        data_dir="data/val/client_1",
        mode="private",
        image_size=1024,
        has_mask=True
    )
    
    # 使用验证数据集包装器
    val_dataset = ValDataset(
        base_dataset,
        inp_size=512,
        augment=False  # 验证集不使用数据增强
    )
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,  # 验证集不打乱
        num_workers=0
    )
    
    # 测试加载
    for batch in val_loader:
        print(f"输入形状: {batch['inp'].shape}")
        print(f"掩码形状: {batch['gt'].shape}")
        break
    
    print("✓ 验证数据集包装器使用成功")


def example_3_convenience_function():
    """示例 3: 使用便捷函数"""
    print("=" * 60)
    print("示例 3: 使用便捷函数")
    print("=" * 60)
    
    # 创建基础数据集
    base_dataset = MedicalImageDataset(
        data_dir="data/train/client_1",
        mode="private",
        image_size=1024,
        has_mask=True
    )
    
    # 使用便捷函数包装
    train_dataset = wrap_dataset(
        base_dataset,
        mode="train",
        inp_size=512,
        augment=True
    )
    
    val_dataset = wrap_dataset(
        base_dataset,
        mode="val",
        inp_size=512,
        augment=False
    )
    
    print(f"训练数据集长度: {len(train_dataset)}")
    print(f"验证数据集长度: {len(val_dataset)}")
    print("✓ 便捷函数使用成功")


def example_4_integrate_with_existing_loader():
    """示例 4: 与现有数据加载器集成"""
    print("=" * 60)
    print("示例 4: 与现有数据加载器集成")
    print("=" * 60)
    
    # 使用现有的 create_data_loaders 函数
    client_configs = [
        {'client_id': 'client_1', 'has_private': True, 'has_public': True},
        {'client_id': 'client_2', 'has_private': True, 'has_public': True},
    ]
    
    try:
        # 创建基础数据加载器
        base_loaders = create_data_loaders(
            data_root="data",
            split="train",
            client_configs=client_configs,
            batch_size=4,
            image_size=1024,
            shuffle=True
        )
        
        # 包装每个客户端的数据集
        wrapped_loaders = []
        for private_loader, public_loader in base_loaders:
            # 包装私有数据加载器
            if private_loader is not None:
                base_dataset = private_loader.dataset
                wrapped_dataset = wrap_dataset(
                    base_dataset,
                    mode="train",
                    inp_size=512,
                    augment=True
                )
                wrapped_private_loader = DataLoader(
                    wrapped_dataset,
                    batch_size=private_loader.batch_size,
                    shuffle=True,
                    num_workers=0
                )
            else:
                wrapped_private_loader = None
            
            # 包装公开数据加载器（如果有）
            if public_loader is not None:
                base_dataset = public_loader.dataset
                wrapped_dataset = wrap_dataset(
                    base_dataset,
                    mode="train",
                    inp_size=512,
                    augment=True
                )
                wrapped_public_loader = DataLoader(
                    wrapped_dataset,
                    batch_size=public_loader.batch_size,
                    shuffle=True,
                    num_workers=0
                )
            else:
                wrapped_public_loader = None
            
            wrapped_loaders.append((wrapped_private_loader, wrapped_public_loader))
        
        print(f"成功包装 {len(wrapped_loaders)} 个客户端的数据加载器")
        print("✓ 与现有数据加载器集成成功")
        
    except Exception as e:
        print(f"集成测试失败（可能是数据路径问题）: {e}")
        print("这是正常的，如果数据目录不存在")


def example_5_custom_augmentation():
    """示例 5: 自定义数据增强"""
    print("=" * 60)
    print("示例 5: 自定义数据增强")
    print("=" * 60)
    
    # 创建基础数据集
    base_dataset = MedicalImageDataset(
        data_dir="data/train/client_1",
        mode="private",
        image_size=1024,
        has_mask=True
    )
    
    # 使用包装器，启用数据增强
    train_dataset = TrainDataset(
        base_dataset,
        inp_size=512,
        augment=True,  # 启用随机翻转
        gt_resize=512  # 掩码尺寸
    )
    
    # 测试多次获取同一索引，查看数据增强效果
    idx = 0
    samples = [train_dataset[idx] for _ in range(5)]
    
    print(f"获取索引 {idx} 的样本 5 次（每次可能因随机翻转而不同）")
    print(f"所有样本的输入形状一致: {all(s['inp'].shape == samples[0]['inp'].shape for s in samples)}")
    print("✓ 自定义数据增强工作正常")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集包装器使用示例")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="选择要运行的示例 (1-5)"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_wrapper,
        2: example_2_validation_wrapper,
        3: example_3_convenience_function,
        4: example_4_integrate_with_existing_loader,
        5: example_5_custom_augmentation,
    }
    
    print("\n注意: 这些示例需要实际的数据目录")
    print("如果数据目录不存在，某些示例可能会失败，这是正常的\n")
    
    try:
        examples[args.example]()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()






