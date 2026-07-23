"""
测试配置加载和解耦聚合功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import FederatedConfig


def test_config_loading():
    """测试配置文件加载"""
    print("="*60)
    print("测试配置加载")
    print("="*60)

    configs_to_test = [
        ("configs/baseline.yaml", "Baseline"),
        ("configs/proposed_method.yaml", "Proposed Method")
    ]

    for config_path, name in configs_to_test:
        print(f"\n[测试] {name} ({config_path})")
        print("-"*60)

        try:
            # 加载配置
            config = FederatedConfig.from_yaml(config_path)

            # 显示关键信息
            print(f"  ✓ 配置加载成功")
            print(f"  - 解耦聚合: {'✅ 开启' if config.use_decoupled_agg else '❌ 关闭'}")
            print(f"  - Batch Size: {config.batch_size}")
            print(f"  - Learning Rate: {config.lr}")
            print(f"  - 混合精度: {'✅' if config.use_amp else '❌'}")

            # 显示客户端配置
            if config.clients:
                print(f"  - 客户端数量: {len(config.clients)}")
                for i, client in enumerate(config.clients, 1):
                    if client.get('enabled', True):
                        print(f"    {i}. {client['client_id']:10s} | {client['modality']:12s} | {client['data_source']}")
            else:
                print(f"  - 客户端配置: None (将从 data_root 自动加载)")

        except Exception as e:
            print(f"  ❌ 配置加载失败: {e}")
            import traceback
            traceback.print_exc()


def test_modality_extraction():
    """测试模态信息提取"""
    print("\n" + "="*60)
    print("测试模态信息提取")
    print("="*60)

    config_path = "configs/proposed_method.yaml"
    config = FederatedConfig.from_yaml(config_path)

    if config.clients:
        # 模拟 federated_trainer 中的逻辑
        client_modality_map = {c['client_id']: c['modality'] for c in config.clients if c.get('enabled', True)}

        print(f"\n客户端模态映射:")
        for client_id, modality in client_modality_map.items():
            print(f"  {client_id:10s} -> {modality}")

        # 模拟排序后的模态列表
        client_ids_sorted = sorted(client_modality_map.keys())
        client_modalities = [client_modality_map.get(cid, 'multimodal') for cid in client_ids_sorted]

        print(f"\n排序后的客户端 ID: {client_ids_sorted}")
        print(f"对应的模态列表: {client_modalities}")

        print(f"\n✓ 模态提取成功")
    else:
        print(f"⚠️ 配置中没有客户端信息")


if __name__ == "__main__":
    test_config_loading()
    test_modality_extraction()

    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)
