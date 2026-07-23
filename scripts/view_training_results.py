"""
查看训练结果脚本
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def view_training_results(checkpoint_dir="data/checkpoints"):
    """查看训练结果"""
    checkpoint_path = Path(checkpoint_dir)
    history_path = checkpoint_path / "training_history.json"
    
    if not history_path.exists():
        print(f"训练历史文件不存在: {history_path}")
        print("请先运行训练: python run_training.py")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print("=" * 60)
    print("训练结果总结")
    print("=" * 60)
    
    # 基本信息
    print(f"\n训练轮数: {history['final_stats']['total_rounds']}")
    print(f"训练时间: {history.get('training_time', '未知')}")
    
    # 损失信息
    if history['avg_losses']:
        print(f"\n损失统计:")
        print(f"  初始损失: {history['avg_losses'][0]:.4f}")
        print(f"  最终损失: {history['avg_losses'][-1]:.4f}")
        print(f"  损失下降: {history['avg_losses'][0] - history['avg_losses'][-1]:.4f}")
        min_loss = min(history['avg_losses'])
        min_round = history['rounds'][history['avg_losses'].index(min_loss)]
        print(f"  最低损失: {min_loss:.4f} (第 {min_round} 轮)")
    
    # 模型统计
    print(f"\n模型统计:")
    stats = history['final_stats']
    print(f"  总参数量: {stats['total_params']:,}")
    print(f"  可训练参数: {stats['trainable_params']:,}")
    print(f"  冻结参数: {stats['frozen_params']:,}")
    
    # 全局表示
    if history['global_text_rep_norms']:
        print(f"\n全局表示范数:")
        print(f"  文本表示: {history['global_text_rep_norms'][-1]:.4f}")
        print(f"  图像表示: {history['global_image_rep_norms'][-1]:.4f}")
    
    # 客户端损失（最后一轮）
    if history['client_losses']:
        print(f"\n最后一轮客户端损失:")
        last_client_losses = history['client_losses'][-1]
        for client_id, loss in last_client_losses.items():
            print(f"  {client_id}: {loss:.4f}")
    
    # 验证集指标
    if 'val_metrics' in history and history['val_metrics']:
        print(f"\n验证集指标:")
        print(f"  评估次数: {len(history['val_metrics'])}")
        
        # 最佳指标
        best_dice = max(m['dice'] for m in history['val_metrics'])
        best_iou = max(m['iou'] for m in history['val_metrics'])
        best_dice_round = next(m['round'] for m in history['val_metrics'] if m['dice'] == best_dice)
        
        print(f"  最佳 Dice: {best_dice:.4f} (第 {best_dice_round} 轮)")
        print(f"  最佳 IoU: {best_iou:.4f}")
        
        # 最终指标
        if 'final_val_metrics' in history:
            final = history['final_val_metrics']
            print(f"\n  最终指标:")
            print(f"    Dice: {final.get('dice', 0.0):.4f}")
            print(f"    IoU: {final.get('iou', 0.0):.4f}")
            if 'hd95' in final and final['hd95'] != float('inf'):
                print(f"    HD95: {final['hd95']:.2f} mm")
            else:
                print(f"    HD95: N/A")
    
    print("\n" + "=" * 60)
    print(f"详细历史记录文件: {history_path}")
    print("=" * 60)
    
    # 询问是否显示详细历史
    try:
        show_detail = input("\n是否显示详细训练历史? (y/n): ").strip().lower()
        if show_detail == 'y':
            print("\n详细训练历史:")
            print("-" * 60)
            for i, (round_num, avg_loss) in enumerate(zip(history['rounds'], history['avg_losses'])):
                if i % 10 == 0 or i == len(history['rounds']) - 1:
                    print(f"第 {round_num:3d} 轮: 平均损失 = {avg_loss:.4f}")
    except:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="查看训练结果")
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints',
                       help='检查点目录（默认: data/checkpoints）')
    args = parser.parse_args()
    
    view_training_results(args.checkpoint_dir)















