"""
快速启动联邦学习训练脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FedSAM3-Cream 联邦学习训练 - 快速启动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数运行（真实数据）
  python run_training.py
  
  # 使用自定义参数
  python run_training.py --rounds 100 --batch_size 4
  
  # 使用虚拟数据测试
  python run_training.py --use_dummy
  
  # 查看帮助
  python run_training.py --help
        """
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/federated_split',
        help='数据根目录（默认: data/federated_split）'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help='训练轮数（默认: 50）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批次大小（默认: 6）'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率（默认: 0.0002）'
    )
    parser.add_argument(
        '--use_dummy',
        action='store_true',
        help='使用虚拟数据（用于快速测试）'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大样本数（用于冒烟测试）'
    )
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=None,
        help='本地训练轮数'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help='图像尺寸'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=None,
        help='嵌入维度'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=None,
        help='注意力头数'
    )
    
    args = parser.parse_args()
    
    if args.use_dummy:
        # 使用虚拟数据
        print("=" * 60)
        print("使用虚拟数据运行训练（测试模式）")
        print("=" * 60)
        from scripts.main_federated import main as main_dummy
        main_dummy()
    else:
        # 使用真实数据
        print("=" * 60)
        print("使用真实 BraTS 数据运行训练")
        print("=" * 60)
        from scripts.train_brats_federated import main as main_real
        
        # 准备参数
        sys.argv = ['train_brats_federated.py']
        if args.data_root:
            sys.argv.extend(['--data_root', args.data_root])
        if args.rounds:
            sys.argv.extend(['--rounds', str(args.rounds)])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.lr:
            sys.argv.extend(['--lr', str(args.lr)])
        if args.max_samples:
            sys.argv.extend(['--max_samples', str(args.max_samples)])
        if args.local_epochs:
            sys.argv.extend(['--local_epochs', str(args.local_epochs)])
        if args.img_size:
            sys.argv.extend(['--img_size', str(args.img_size)])
        if args.embed_dim:
            sys.argv.extend(['--embed_dim', str(args.embed_dim)])
        if args.num_heads:
            sys.argv.extend(['--num_heads', str(args.num_heads)])
        
        exit_code = main_real()
        sys.exit(exit_code)
