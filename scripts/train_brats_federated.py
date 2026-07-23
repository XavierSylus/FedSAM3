"""
联邦学习训练入口脚本
使用真实 BraTS 数据运行异构客户端联邦学习训练
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主联邦学习训练入口"""
    try:
        # 导入配置管理器和训练器
        from src.config_manager import load_config
        from src.federated_trainer import FederatedTrainer
        
        # 1. 加载配置
        config = load_config()
        
        # 2. 初始化训练器
        trainer = FederatedTrainer(config)
        
        # 3. 运行训练
        exit_code = trainer.train()
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return 130
        
    except Exception as e:
        print(f"训练过程中发生错误: {type(e).__name__}")
        print(f"错误详情: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
