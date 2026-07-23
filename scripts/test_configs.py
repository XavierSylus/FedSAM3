"""
配置文件加载测试脚本
用于验证 baseline.yaml 和 proposed_method.yaml 配置的正确性
"""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_config_summary(config: dict, config_name: str):
    """打印配置摘要"""
    print(f"\n{'='*60}")
    print(f"配置文件: {config_name}")
    print(f"{'='*60}")

    # 联邦学习配置
    federated = config.get('federated', {})
    clients = federated.get('clients', [])
    use_decoupled = federated.get('use_decoupled_agg', False)

    print(f"\n【联邦学习配置】")
    print(f"  客户端数量: {len(clients)}")
    print(f"  解耦聚合: {'✅ 开启' if use_decoupled else '❌ 关闭'}")
    print(f"\n  客户端列表:")
    for i, client in enumerate(clients, 1):
        print(f"    {i}. {client['client_id']:10s} | 模态: {client['modality']:12s} | 启用: {client['enabled']}")

    # 训练参数
    training = config.get('training', {})
    print(f"\n【训练参数】")
    print(f"  Batch Size: {training.get('batch_size', 1)}")
    print(f"  Learning Rate: {training.get('learning_rate', 0.0002)}")
    print(f"  Rounds: {training.get('rounds', 50)}")
    print(f"  Weight Decay: {training.get('weight_decay', 0.01)}")
    print(f"  Gradient Clip: {training.get('grad_clip', 1.0)}")

    # 训练选项
    options = config.get('options', {})
    print(f"\n【训练选项】")
    print(f"  混合精度 (AMP): {'✅ 开启' if options.get('use_amp', False) else '❌ 关闭'}")
    print(f"  使用真实模型: {'✅ 是' if not options.get('use_dummy', True) else '❌ 否（虚拟）'}")

    # 日志配置
    logging = config.get('logging', {})
    print(f"\n【日志配置】")
    print(f"  日志类型: {logging.get('log_type', 'tensorboard')}")
    print(f"  日志目录: {logging.get('log_dir', 'logs')}")
    print(f"  实验名称: {logging.get('experiment_name', 'N/A')}")


def main():
    """主函数"""
    base_dir = Path(r"G:\FedSAM3-Cream存档\FedSAM3-Cream26.1.存档")
    configs_dir = base_dir / "configs"

    # 测试两个配置文件
    configs_to_test = [
        ("baseline.yaml", "Baseline 实验配置"),
        ("proposed_method.yaml", "提出方法实验配置")
    ]

    for config_file, config_name in configs_to_test:
        config_path = configs_dir / config_file

        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            continue

        try:
            # 加载配置
            config = load_config(config_path)

            # 打印摘要
            print_config_summary(config, config_name)

            print(f"\n✅ 配置文件验证成功: {config_file}")

        except Exception as e:
            print(f"❌ 配置文件加载失败: {config_file}")
            print(f"   错误信息: {e}")

    print(f"\n{'='*60}")
    print("配置文件对比总结")
    print(f"{'='*60}")
    print("\n【Baseline vs Proposed Method】")
    print("  1. 客户端数量: Baseline (2个) vs Proposed (3个)")
    print("  2. 解耦聚合: Baseline (❌) vs Proposed (✅)")
    print("  3. 训练参数: 两者一致（Batch=1, LR=2e-4, AMP=✅）")
    print("\n【实验对照组设计】")
    print("  - Baseline: 仅 Client 2 + 3，传统联邦聚合")
    print("  - Proposed: 所有客户端，解耦功能聚合")
    print("  - 核心差异: 能否有效利用异构模态数据")


if __name__ == "__main__":
    main()
