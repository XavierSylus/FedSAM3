import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_metrics(history_path, output_dir):
    # 读取历史数据
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {history_path}")
        return

    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 绘制训练损失 (Training Loss)
    if 'rounds' in history and 'avg_losses' in history:
        rounds = history['rounds']
        losses = history['avg_losses']
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.title('Training Loss per Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        loss_path = output_dir / 'training_loss_curve.png'
        plt.savefig(loss_path)
        plt.close()
        print(f"训练损失曲线已保存: {loss_path}")

    # 2. 绘制验证指标 (Validation Metrics - Dice & IoU)
    if 'val_metrics' in history and history['val_metrics']:
        val_rounds = [m['round'] for m in history['val_metrics']]
        val_dice = [m['dice'] for m in history['val_metrics']]
        val_iou = [m['iou'] for m in history['val_metrics']]
        # Handle HD95: Replace infinity with a large number or None for plotting
        val_hd95 = []
        for m in history['val_metrics']:
            v = m['hd95']
            if v == float('inf') or v == 'Infinity':
                val_hd95.append(None) # Don't plot inf
            else:
                val_hd95.append(v)
        
        # Plot Dice & IoU
        plt.figure(figsize=(10, 6))
        plt.plot(val_rounds, val_dice, marker='s', linestyle='-', color='g', label='Validation Dice')
        plt.plot(val_rounds, val_iou, marker='^', linestyle='--', color='orange', label='Validation IoU')
        
        plt.title('Validation Metrics per Round (Dice & IoU)')
        plt.xlabel('Round')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        
        val_path = output_dir / 'validation_metrics_curve.png'
        plt.savefig(val_path)
        plt.close()
        print(f"验证指标曲线(Dice/IoU)已保存: {val_path}")

        # Plot HD95 separately (different scale)
        plt.figure(figsize=(10, 6))
        # Filter out Nones for plotting
        valid_hd95_rounds = [r for r, v in zip(val_rounds, val_hd95) if v is not None]
        valid_hd95_values = [v for v in val_hd95 if v is not None]
        
        if valid_hd95_values:
            plt.plot(valid_hd95_rounds, valid_hd95_values, marker='x', linestyle='-', color='r', label='Validation HD95')
            plt.title('Validation HD95 per Round (Lower is Better)')
            plt.xlabel('Round')
            plt.ylabel('HD95 (mm)')
            plt.grid(True)
            plt.legend()
            
            hd95_path = output_dir / 'validation_hd95_curve.png'
            plt.savefig(hd95_path)
            plt.close()
            print(f"验证指标曲线(HD95)已保存: {hd95_path}")
        else:
             print("提示: HD95 全为无穷大(Inf)，跳过绘制 HD95 曲线。")

        # 3. 绘制验证损失 (Validation Loss) - [Added]
        val_loss = []
        has_val_loss = False
        for m in history['val_metrics']:
            if 'val_loss' in m:
                val_loss.append(m['val_loss'])
                has_val_loss = True
            else:
                val_loss.append(None)
        
        if has_val_loss:
            plt.figure(figsize=(10, 6))
            valid_loss_rounds = [r for r, v in zip(val_rounds, val_loss) if v is not None]
            valid_loss_values = [v for v in val_loss if v is not None]
            
            plt.plot(valid_loss_rounds, valid_loss_values, marker='d', linestyle='-', color='purple', label='Validation Loss')
            plt.title('Validation Loss per Round')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            val_loss_path = output_dir / 'validation_loss_curve.png'
            plt.savefig(val_loss_path)
            plt.close()
            print(f"验证损失曲线已保存: {val_loss_path}")

    else:
        print("警告: 未找到验证集指标数据 (val_metrics)，跳过绘制。")

if __name__ == "__main__":
    # 配置路径
    current_dir = Path(os.getcwd())
    history_file = current_dir / 'data' / 'checkpoints' / 'training_history.json'
    output_directory = current_dir
    
    print(f"正在读取历史文件: {history_file}")
    plot_metrics(history_file, output_directory)
