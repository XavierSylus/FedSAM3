"""
串行客户端设置脚本（修复版）

修复说明:
1. 从配置对象读取客户端列表（不再硬编码）
2. 支持 dataset.json 格式的数据加载
3. 增强错误提示，显示搜索的绝对路径
4. 支持多种数据格式（旧格式目录结构 + 新格式JSON）
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from data.multimodal_dataset import MultimodalMedicalDataset


def load_from_json(
    json_path: Path,
    batch_size: int,
    img_size: int,
    max_samples: Optional[int] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    从 dataset.json 文件加载数据

    Args:
        json_path: dataset.json 文件路径
        batch_size: 批次大小
        img_size: 图像尺寸
        max_samples: 最大样本数

    Returns:
        (private_loader, public_loader) 或 (None, None)
    """
    abs_path = json_path.resolve()

    if not json_path.exists():
        print(f"    ❌ JSON文件不存在: {json_path}")
        print(f"       绝对路径: {abs_path}")
        return None, None

    try:
        # 读取JSON配置
        with open(json_path, 'r') as f:
            dataset_config = json.load(f)

        # 推断data_root（JSON文件所在目录的上级目录）
        data_root = json_path.parent.parent.parent  # data/federated_split/clientX -> data/

        # 创建数据集
        dataset = MultimodalMedicalDataset(
            json_file=str(json_path),
            data_root=str(data_root),
            image_size=img_size,
            max_samples=max_samples
        )

        if len(dataset) == 0:
            print(f"    ⚠️ 数据集为空: {json_path}")
            return None, None

        # 创建DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        print(f"    ✅ 成功加载 {len(dataset)} 个样本")

        # 假设JSON格式的数据都是私有数据
        # 如果需要区分private/public，需要在JSON中添加标记
        return loader, None

    except Exception as e:
        print(f"    ❌ 加载JSON失败: {e}")
        print(f"       文件: {abs_path}")
        import traceback
        traceback.print_exc()
        return None, None


def setup_serial_clients(
    config: object,  # FederatedConfig对象
    data_root: str = None,
    batch_size: int = 1,
    img_size: int = 256,
    max_samples: Optional[int] = None,
    embed_dim: int = 768
) -> Dict[str, Dict]:
    """
    设置串行训练的客户端配置（修复版）

    Args:
        config: 配置对象，包含 clients 列表
        data_root: 数据根目录（如果config中没有）
        batch_size: 批次大小
        img_size: 图像尺寸
        max_samples: 最大样本数
        embed_dim: 嵌入维度

    Returns:
        客户端配置字典
    """
    # 获取数据根目录
    if data_root is None:
        data_root = getattr(config, 'data_root', 'data')

    print(f"[设置串行客户端 v2.0] 数据根目录: {data_root}")
    print(f"  当前工作目录: {Path.cwd()}")
    print(f"  数据根绝对路径: {Path(data_root).resolve()}")

    client_configs = {}

    # 从配置对象获取客户端列表
    clients_list = getattr(config, 'clients', None)

    if clients_list is None:
        print("\n❌ 错误: 配置中没有 'clients' 字段！")
        print("   请确保配置文件包含 'federated.clients' 列表")
        return {}

    print(f"\n找到 {len(clients_list)} 个客户端配置\n")

    for client_info in clients_list:
        client_id = client_info.get('client_id')
        modality = client_info.get('modality', 'multimodal')
        enabled = client_info.get('enabled', True)
        data_source = client_info.get('data_source')

        if not enabled:
            print(f"  ⏭️  跳过客户端 {client_id}: 已禁用")
            continue

        print(f"  配置客户端: {client_id} (模态: {modality})")

        try:
            private_loader = None
            public_loader = None

            # 方式1: 从data_source指定的JSON文件加载
            if data_source and Path(data_source).suffix == '.json':
                json_path = Path(data_source)
                print(f"    使用JSON数据源: {json_path}")
                private_loader, public_loader = load_from_json(
                    json_path, batch_size, img_size, max_samples
                )

            # 方式2: 从旧格式目录结构加载
            else:
                print(f"    尝试使用旧格式目录结构...")
                from data.dataset_loader import create_data_loaders

                client_type_config = {
                    'client_id': client_id,
                    'has_private': True,
                    'has_public': False
                }

                loaders = create_data_loaders(
                    data_root=data_root,
                    split="train",
                    client_configs=[client_type_config],
                    batch_size=batch_size,
                    image_size=img_size,
                    shuffle=True,
                    max_samples=max_samples
                )

                if loaders and len(loaders) > 0:
                    private_loader, public_loader = loaders[0]

            # 验证加载器
            if private_loader is None and public_loader is None:
                print(f"    ⚠️ 跳过 {client_id}: 无法创建任何数据加载器")
                continue

            # 存储配置
            client_configs[client_id] = {
                'modality': modality,
                'private_loader': private_loader,
                'public_loader': public_loader,
                'embed_dim': embed_dim
            }

            print(f"    ✅ 客户端 {client_id} 配置成功")
            print()

        except Exception as e:
            print(f"    ❌ 客户端 {client_id} 配置失败: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # 最终检查
    if len(client_configs) == 0:
        print("\n" + "="*70)
        print("❌ 致命错误: 没有成功配置任何客户端！")
        print("="*70)
        print("\n可能的原因:")
        print("  1. data_source 路径不正确")
        print("  2. dataset.json 文件不存在或格式错误")
        print("  3. 旧格式目录结构不存在")
        print("\n建议:")
        print("  1. 运行诊断脚本: python diagnose_data_structure.py")
        print("  2. 检查配置文件中的 data_source 路径")
        print("  3. 确认数据已准备好")
        print("="*70 + "\n")
        raise RuntimeError("没有成功配置任何客户端！请检查数据目录和配置。")

    print(f"\n✅ 成功配置 {len(client_configs)} 个客户端: {list(client_configs.keys())}\n")
    return client_configs
