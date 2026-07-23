"""
测试日志记录功能
验证 TensorBoard 和 WandB 日志记录是否正常工作
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import create_logger
import torch


def test_tensorboard_logging():
    """测试 TensorBoard 日志记录"""
    print("=" * 60)
    print("测试 TensorBoard 日志记录")
    print("=" * 60)
    
    try:
        logger = create_logger(
            log_type="tensorboard",
            experiment_name="test_logging",
            log_dir="logs"
        )
        
        # 模拟训练过程，记录关键指标
        print("\n记录训练指标...")
        for round_num in range(1, 11):
            # 模拟损失值（逐渐下降）
            train_loss = 1.4 - (round_num - 1) * 0.03
            seg_loss = train_loss * 0.7
            cream_loss = train_loss * 0.3
            
            logger.log({
                'Train_Loss': train_loss,
                'Seg_Loss': seg_loss,
                'Cream_Loss': cream_loss,
            }, step=round_num)
            
            # 每5轮记录一次验证指标
            if round_num % 5 == 0:
                val_dice = 0.5 + (round_num - 1) * 0.05
                val_iou = val_dice * 0.8
                val_hd95 = 20.0 - (round_num - 1) * 1.5
                
                logger.log({
                    'Val_Dice': val_dice,
                    'Val_IoU': val_iou,
                    'Val_HD95': val_hd95,
                }, step=round_num)
        
        # 记录总结
        logger.log_summary({
            'final_train_loss': 1.1,
            'final_val_dice': 0.95,
            'best_val_dice': 0.96
        })
        
        logger.close()
        print("✓ TensorBoard 日志记录测试完成")
        print(f"  日志目录: logs/tensorboard/test_logging")
        print("  查看日志: tensorboard --logdir logs/tensorboard")
        return True
        
    except Exception as e:
        print(f"✗ TensorBoard 日志记录测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wandb_logging():
    """测试 WandB 日志记录（如果可用）"""
    print("\n" + "=" * 60)
    print("测试 WandB 日志记录（如果可用）")
    print("=" * 60)
    
    try:
        logger = create_logger(
            log_type="wandb",
            experiment_name="test_logging_wandb",
            project_name="FedSAM3-Cream-Test"
        )
        
        # 模拟训练过程
        print("\n记录训练指标...")
        for round_num in range(1, 6):
            train_loss = 1.4 - (round_num - 1) * 0.03
            seg_loss = train_loss * 0.7
            cream_loss = train_loss * 0.3
            
            logger.log({
                'Train_Loss': train_loss,
                'Seg_Loss': seg_loss,
                'Cream_Loss': cream_loss,
            }, step=round_num)
        
        logger.log_summary({
            'final_train_loss': 1.1,
            'final_val_dice': 0.95
        })
        
        logger.close()
        print("✓ WandB 日志记录测试完成")
        if logger.wandb_run:
            print(f"  访问 WandB: {logger.wandb_run.url}")
        return True
        
    except Exception as e:
        print(f"⚠ WandB 日志记录测试跳过: {e}")
        print("  （这可能是正常的，如果 WandB 未安装或未登录）")
        return False


def test_both_logging():
    """测试同时使用 TensorBoard 和 WandB"""
    print("\n" + "=" * 60)
    print("测试同时使用 TensorBoard 和 WandB")
    print("=" * 60)
    
    try:
        logger = create_logger(
            log_type="both",
            experiment_name="test_logging_both",
            project_name="FedSAM3-Cream-Test",
            log_dir="logs"
        )
        
        # 记录一些指标
        logger.log({
            'Train_Loss': 1.2,
            'Seg_Loss': 0.84,
            'Cream_Loss': 0.36,
        }, step=1)
        
        logger.close()
        print("✓ 同时使用 TensorBoard 和 WandB 测试完成")
        return True
        
    except Exception as e:
        print(f"⚠ 同时使用测试跳过: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("日志记录功能测试")
    print("=" * 60)
    
    results = []
    
    # 测试 TensorBoard
    results.append(("TensorBoard", test_tensorboard_logging()))
    
    # 测试 WandB（可选）
    results.append(("WandB", test_wandb_logging()))
    
    # 测试同时使用
    results.append(("Both", test_both_logging()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败/跳过"
        print(f"{name:20s}: {status}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n提示:")
    print("1. TensorBoard 日志保存在 logs/tensorboard/ 目录")
    print("2. 使用 'tensorboard --logdir logs/tensorboard' 查看日志")
    print("3. WandB 需要先安装: pip install wandb")
    print("4. WandB 需要先登录: wandb login")

