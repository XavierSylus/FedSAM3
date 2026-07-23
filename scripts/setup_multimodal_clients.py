"""
多模态串行客户端设置脚本（支持文本特征）

此模块负责为串行训练模式设置支持文本特征的客户端配置。
使用新的 MultimodalMedicalDataset 来加载 TextBraTS.json 和 .npy 文本特征。
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from data.multimodal_dataset import MultimodalMedicalDataset, create_multimodal_dataloaders


def setup_multimodal_clients(
    json_path: str,
    data_root: Optional[str] = None,
    batch_size: int = 4,
    img_size: int = 1024,
    max_samples: Optional[int] = None,
    embed_dim: int = 768,
    folds: Optional[Dict[str, List[int]]] = None
) -> Dict[str, Dict]:
    """
    设置支持文本特征的串行训练客户端配置

    此函数为每个客户端准备多模态数据加载器（图像 + 文本特征）。
    模型将在训练循环中按需创建和销毁。

    Args:
        json_path: JSON 元数据文件路径（如 "data/source_images/TextBraTS.json"）
        data_root: 数据根目录（如果 JSON 中的路径是相对路径）
        batch_size: 批次大小
        img_size: 图像尺寸
        max_samples: 最大样本数（用于快速测试）
        embed_dim: 嵌入维度
        folds: 折划分配置（用于交叉验证）
               例如: {'train': [0, 1, 2], 'val': [3], 'test': [4]}
               如果为 None，则加载所有数据

    Returns:
        客户端配置字典，格式:
        {
            'client_multimodal': {
                'modality': 'multimodal',
                'private_loader': DataLoader,  # 返回 (image, mask, text_features)
                'public_loader': None,
                'embed_dim': int,
                'has_text_features': True
            }
        }
    """
    print(f"[设置多模态客户端] JSON 文件: {json_path}")
    print(f"[设置多模态客户端] 数据根目录: {data_root or 'JSON 文件所在目录'}")

    client_configs = {}

    try:
        # 方案 1：如果指定了 folds，按 fold 划分数据
        if folds is not None:
            print(f"  使用交叉验证，fold 配置: {folds}")

            loaders = create_multimodal_dataloaders(
                json_path=json_path,
                data_root=data_root,
                batch_size=batch_size,
                image_size=img_size,
                num_workers=0,
                shuffle_train=True,
                folds=folds
            )

            # 为训练集创建客户端配置
            if 'train' in loaders:
                train_loader = loaders['train']

                client_configs['client_multimodal'] = {
                    'modality': 'multimodal',
                    'private_loader': train_loader,
                    'public_loader': None,
                    'embed_dim': embed_dim,
                    'has_text_features': True
                }

                print(f"  ✓ 客户端 'client_multimodal' 配置成功")
                print(f"    - 训练数据: {len(train_loader.dataset)} 样本")
                print(f"    - 模态类型: multimodal（图像 + 文本特征）")

        # 方案 2：加载所有数据（不划分 fold）
        else:
            print(f"  加载所有数据（不划分 fold）")

            dataset = MultimodalMedicalDataset(
                json_path=json_path,
                data_root=data_root,
                fold=None,  # 加载所有 fold
                image_size=img_size,
                max_samples=max_samples
            )

            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            client_configs['client_multimodal'] = {
                'modality': 'multimodal',
                'private_loader': train_loader,
                'public_loader': None,
                'embed_dim': embed_dim,
                'has_text_features': True
            }

            print(f"  ✓ 客户端 'client_multimodal' 配置成功")
            print(f"    - 训练数据: {len(dataset)} 样本")
            print(f"    - 模态类型: multimodal（图像 + 文本特征）")

    except Exception as e:
        print(f"  ❌ 多模态客户端配置失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not client_configs:
        raise RuntimeError(
            "没有成功配置任何客户端！\n"
            "请检查：\n"
            "1. JSON 文件路径是否正确\n"
            "2. 数据根目录是否正确\n"
            "3. .npy 文本特征文件是否存在"
        )

    print(f"[完成] 成功配置 {len(client_configs)} 个多模态客户端")
    return client_configs


def setup_mixed_clients(
    json_path: str,
    data_root: Optional[str] = None,
    old_data_root: str = "data/federated_split",
    batch_size: int = 4,
    img_size: int = 1024,
    max_samples: Optional[int] = None,
    embed_dim: int = 768
) -> Dict[str, Dict]:
    """
    设置混合客户端配置（多模态 + 纯图像客户端）

    此函数创建两种类型的客户端：
    1. 多模态客户端：使用 TextBraTS.json 和文本特征
    2. 纯图像客户端：使用旧的数据加载器（不使用文本特征）

    Args:
        json_path: 多模态数据的 JSON 文件路径
        data_root: 多模态数据的根目录
        old_data_root: 旧数据的根目录（纯图像客户端）
        batch_size: 批次大小
        img_size: 图像尺寸
        max_samples: 最大样本数
        embed_dim: 嵌入维度

    Returns:
        混合客户端配置字典
    """
    print("[设置混合客户端] 多模态 + 纯图像")

    client_configs = {}

    # 1. 配置多模态客户端
    print("\n[1/2] 配置多模态客户端...")
    try:
        multimodal_configs = setup_multimodal_clients(
            json_path=json_path,
            data_root=data_root,
            batch_size=batch_size,
            img_size=img_size,
            max_samples=max_samples,
            embed_dim=embed_dim
        )
        client_configs.update(multimodal_configs)
    except Exception as e:
        print(f"  ⚠️  多模态客户端配置失败: {e}")

    # 2. 配置纯图像客户端（使用旧的数据加载器）
    print("\n[2/2] 配置纯图像客户端...")
    try:
        from data.dataset_loader import create_data_loaders

        # 定义纯图像客户端
        image_client_types = [
            {'client_id': 'client_image_1', 'has_private': True, 'has_public': False},
            {'client_id': 'client_image_2', 'has_private': True, 'has_public': False},
        ]

        loaders = create_data_loaders(
            data_root=old_data_root,
            split="train",
            client_configs=image_client_types,
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            max_samples=max_samples
        )

        for i, (private_loader, public_loader) in enumerate(loaders):
            if private_loader is not None:
                client_id = image_client_types[i]['client_id']
                client_configs[client_id] = {
                    'modality': 'image_only',
                    'private_loader': private_loader,
                    'public_loader': public_loader,
                    'embed_dim': embed_dim,
                    'has_text_features': False
                }
                print(f"  ✓ 客户端 '{client_id}' 配置成功")
                print(f"    - 训练数据: {len(private_loader.dataset)} 样本")
                print(f"    - 模态类型: image_only")

    except Exception as e:
        print(f"  ⚠️  纯图像客户端配置失败: {e}")

    if not client_configs:
        raise RuntimeError("没有成功配置任何客户端！")

    print(f"\n[完成] 成功配置 {len(client_configs)} 个混合客户端")
    return client_configs


if __name__ == "__main__":
    """测试多模态客户端设置"""
    print("=" * 80)
    print("多模态客户端设置 - 测试模式")
    print("=" * 80)

    # 测试配置
    json_path = "data/source_images/TextBraTS.json"
    data_root = "data/source_images"

    # 方案 1：只配置多模态客户端
    print("\n【方案 1】只配置多模态客户端")
    print("-" * 80)
    try:
        configs = setup_multimodal_clients(
            json_path=json_path,
            data_root=data_root,
            batch_size=2,
            img_size=1024,
            max_samples=5  # 测试用
        )

        print(f"\n配置的客户端: {list(configs.keys())}")
        for client_id, cfg in configs.items():
            print(f"  {client_id}: {cfg['modality']}, 文本特征: {cfg['has_text_features']}")

        # 测试加载一个批次
        print("\n测试加载第一个批次...")
        loader = configs['client_multimodal']['private_loader']
        images, masks, text_features = next(iter(loader))
        print(f"  ✓ 图像形状: {images.shape}")
        print(f"  ✓ 掩码形状: {masks.shape}")
        print(f"  ✓ 文本特征形状: {text_features.shape}")

    except Exception as e:
        print(f"✗ 失败: {e}")

    # 方案 2：配置混合客户端（需要旧数据目录存在）
    print("\n\n【方案 2】配置混合客户端（多模态 + 纯图像）")
    print("-" * 80)
    try:
        configs = setup_mixed_clients(
            json_path=json_path,
            data_root=data_root,
            old_data_root="data/federated_split",
            batch_size=2,
            img_size=1024,
            max_samples=5
        )

        print(f"\n配置的客户端: {list(configs.keys())}")
        for client_id, cfg in configs.items():
            print(f"  {client_id}: {cfg['modality']}, 文本特征: {cfg.get('has_text_features', False)}")

    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
