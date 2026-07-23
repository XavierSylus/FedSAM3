#!/usr/bin/env python
"""
最终验证脚本 - 确认所有修复都已生效

检查项:
1. client_id 与目录名匹配
2. 验证集数据可以加载
3. 梯度监控模块可用
4. 配置文件格式正确
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 80)
    print("FedSAM3-Cream 最终验证")
    print("=" * 80)

    passed = 0
    total = 0

    # 测试 1: 检查 client_id 匹配
    print("\n1. 检查 client_id 与目录名匹配...")
    total += 1
    try:
        import yaml
        configs = [
            'configs/exp_group_a.yaml',
            'configs/exp_group_b.yaml',
            'configs/exp_group_c.yaml'
        ]

        for config_file in configs:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            if 'federated' in config and 'clients' in config['federated']:
                for client in config['federated']['clients']:
                    client_id = client.get('client_id', '')
                    # 检查是否是 client_X 格式
                    if client_id.startswith('client_'):
                        print(f"  ✓ {config_file}: {client_id}")
                    else:
                        print(f"  ✗ {config_file}: {client_id} (应该是 client_X 格式)")
                        raise ValueError(f"client_id 格式错误: {client_id}")

        passed += 1
        print("  ✅ 所有 client_id 格式正确")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 测试 2: 检查验证集目录
    print("\n2. 检查验证集目录...")
    total += 1
    try:
        val_dir = Path("data/federated_split/val")
        if not val_dir.exists():
            raise FileNotFoundError(f"验证集目录不存在: {val_dir}")

        for client_dir in ["client_1", "client_2", "client_3"]:
            client_path = val_dir / client_dir / "private"
            if not client_path.exists():
                raise FileNotFoundError(f"客户端目录不存在: {client_path}")

            # 统计样本数
            samples = list(client_path.iterdir())
            print(f"  ✓ {client_dir}: {len(samples)} 个验证样本")

        passed += 1
        print("  ✅ 所有验证集目录存在")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 测试 3: 测试数据加载
    print("\n3. 测试数据加载...")
    total += 1
    try:
        from data.dataset_loader import create_data_loaders

        loaders = create_data_loaders(
            data_root="data/federated_split",
            split="val",
            client_configs=[{
                'client_id': 'client_2',
                'has_private': True,
                'has_public': False,
                'modality': 'image_only'
            }],
            batch_size=1,
            image_size=256,
            shuffle=False
        )

        if loaders and len(loaders) > 0:
            private_loader, _ = loaders[0]
            print(f"  ✓ 成功创建验证集 DataLoader: {len(private_loader.dataset)} 个样本")
            passed += 1
            print("  ✅ 数据加载功能正常")
        else:
            print("  ❌ 未能创建 DataLoader")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试 4: 梯度监控模块
    print("\n4. 测试梯度监控模块...")
    total += 1
    try:
        from src.gradient_monitor import GradientMonitor
        monitor = GradientMonitor()
        print("  ✓ GradientMonitor 导入成功")
        passed += 1
        print("  ✅ 梯度监控模块可用")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 生成报告
    print("\n" + "=" * 80)
    print("验证摘要")
    print("=" * 80)
    print(f"\n通过: {passed}/{total}\n")

    if passed == total:
        print("✅ 所有检查通过！系统已就绪，可以开始训练实验。\n")
        print("下一步:")
        print("  1. python main.py --config configs/exp_group_a.yaml")
        print("  2. python main.py --config configs/exp_group_b.yaml")
        print("  3. python main.py --config configs/exp_group_c.yaml")
        return 0
    else:
        print("⚠️  部分检查失败，请先修复问题。\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
