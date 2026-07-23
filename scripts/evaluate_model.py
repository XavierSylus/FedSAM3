"""
模型评估脚本
使用 MedicalMetricsCalculator 和 PredictionSaver 对训练好的模型进行评估

用法:
    # 评估最新检查点
    python scripts/evaluate_model.py --checkpoint_dir checkpoints
    
    # 评估指定检查点
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pth
    
    # 评估并保存结果
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pth --save_dir results/evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import SAM3_Medical, DEVICE
from src.metrics import MedicalMetricsCalculator
from src.saver import PredictionSaver
from data.dataset_loader import create_data_loaders


def detect_num_classes_from_checkpoint(checkpoint: dict) -> Optional[int]:
    """
    从检查点的 state_dict 中自动检测类别数
    
    检测方法：
    1. 优先检查 mask_decoder.upsample4.weight 的形状
       形状应该是 [num_classes, decoder_dim//8, 2, 2]
       其中 decoder_dim//8 通常是 32（如果 decoder_dim=256）
    2. 备选：检查 mask_decoder.upsample4.bias 的形状 [num_classes]
    3. 备选：检查其他 upsample 层的输出通道数
    
    Args:
        checkpoint: 检查点字典
    
    Returns:
        检测到的类别数，如果无法检测则返回 None
    """
    # 获取 state_dict
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
    if state_dict is None:
        return None
    
    # 方法1: 检查 mask_decoder.upsample4.weight 的形状
    # ConvTranspose2d 权重形状: [in_channels, out_channels, kernel_h, kernel_w]
    # upsample4 = ConvTranspose2d(decoder_dim//8, num_classes, 2, 2)
    # 所以形状是 [decoder_dim//8, num_classes, 2, 2] = [32, num_classes, 2, 2]
    # 例如: num_classes=1 -> [32, 1, 2, 2]
    #      num_classes=4 -> [32, 4, 2, 2]
    key = 'mask_decoder.upsample4.weight'
    if key in state_dict:
        weight_shape = state_dict[key].shape
        if len(weight_shape) == 4:  # [in_channels, out_channels, H, W]
            # num_classes 是 out_channels，在索引 1
            num_classes = weight_shape[1]
            # 验证合理性（类别数通常在 1-10 之间）
            if 1 <= num_classes <= 10:
                print(f"  [检测] 从 {key} 检测到类别数: {num_classes} (形状: {weight_shape})")
                return int(num_classes)
    
    # 方法2: 检查 mask_decoder.upsample4.bias 的形状
    # 形状应该是 [num_classes]
    key = 'mask_decoder.upsample4.bias'
    if key in state_dict:
        bias_shape = state_dict[key].shape
        if len(bias_shape) == 1:  # [num_classes]
            num_classes = bias_shape[0]
            if 1 <= num_classes <= 10:
                print(f"  [检测] 从 {key} 检测到类别数: {num_classes} (形状: {bias_shape})")
                return int(num_classes)
    
    # 方法3: 尝试查找所有包含 'upsample4' 和 'weight' 的键
    for key in state_dict.keys():
        if 'upsample4' in key and 'weight' in key:
            weight_shape = state_dict[key].shape
            if len(weight_shape) == 4:  # [in_channels, out_channels, H, W]
                # num_classes 是 out_channels，在索引 1
                num_classes = weight_shape[1]
                if 1 <= num_classes <= 10:
                    print(f"  [检测] 从 {key} 检测到类别数: {num_classes} (形状: {weight_shape})")
                    return int(num_classes)
    
    # 方法4: 尝试从模型配置中获取（如果存在）
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        if 'num_classes' in model_config:
            num_classes = model_config['num_classes']
            print(f"  [检测] 从 model_config 检测到类别数: {num_classes}")
            return int(num_classes)
    
    print("  [检测] 无法自动检测类别数")
    return None


def load_model(checkpoint_path: Path, device: str = DEVICE, num_classes: Optional[int] = None) -> nn.Module:
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 设备
        num_classes: 手动指定类别数（如果为None，则自动检测）
    
    Returns:
        加载的模型
    """
    print(f"\n加载模型检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从检查点获取模型配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print("  ✓ 从检查点读取模型配置")
    else:
        # 尝试自动检测类别数
        detected_num_classes = detect_num_classes_from_checkpoint(checkpoint)
        
        if detected_num_classes is not None:
            print(f"  ✓ 自动检测到类别数: {detected_num_classes}")
            num_classes_to_use = detected_num_classes
        elif num_classes is not None:
            print(f"  使用指定的类别数: {num_classes}")
            num_classes_to_use = num_classes
        else:
            # 默认配置
            num_classes_to_use = 4  # BraTS: 背景, NCR/NET, ED, ET
            print(f"  警告: 无法检测类别数，使用默认值: {num_classes_to_use}")
        
        # 使用默认配置
        model_config = {
            'img_size': 1024,
            'embed_dim': 768,
            'decoder_dim': 256,
            'num_classes': num_classes_to_use,
            'adapter_skip': 64,
            'use_text_fusion': False
        }
    
    # 如果手动指定了 num_classes，覆盖配置
    if num_classes is not None:
        model_config['num_classes'] = num_classes
        print(f"  使用手动指定的类别数: {num_classes}")
    
    # 创建模型
    model = SAM3_Medical(**model_config)
    
    # 加载权重
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise ValueError("检查点中未找到模型权重")
        print("  ✓ 模型权重加载成功")
    except RuntimeError as e:
        # 如果加载失败，尝试使用 strict=False
        print(f"  警告: 严格模式加载失败，尝试非严格模式...")
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
        elif 'state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['state_dict'], strict=False
            )
        else:
            raise ValueError("检查点中未找到模型权重")
        
        if missing_keys:
            print(f"  警告: 缺失的键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  警告: 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"  警告: 意外的键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  警告: 意外的键: {unexpected_keys}")
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ 模型加载成功")
    print(f"    配置: {model_config}")
    
    return model


def evaluate_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    calculator: MedicalMetricsCalculator,
    saver: Optional[PredictionSaver] = None,
    device: str = DEVICE,
    save_predictions: bool = False,
    num_classes: int = 4
) -> Dict[str, float]:
    """
    在数据集上评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        calculator: 指标计算器
        saver: 结果保存器（可选）
        device: 设备
        save_predictions: 是否保存预测结果
    
    Returns:
        评估指标字典
    """
    model.eval()
    
    all_metrics = []
    image_names = []
    
    print(f"\n开始评估（共 {len(dataloader)} 个批次）...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 处理不同的批次格式
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('inp', batch.get('inputs')))
                labels = batch.get('label', batch.get('gt', batch.get('targets', batch.get('mask'))))
                names = batch.get('name', batch.get('image_name', [f"sample_{batch_idx}"]))
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                    names = batch[2] if len(batch) > 2 else [f"sample_{batch_idx}"]
                else:
                    images = batch[0]
                    labels = None
                    names = [f"sample_{batch_idx}"]
            else:
                images = batch
                labels = None
                names = [f"sample_{batch_idx}"]
            
            if labels is None:
                print(f"  警告: 批次 {batch_idx} 没有标签，跳过")
                continue
            
            # 移动到设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 模型推理
            y_pred = model(images)  # (B, C, H, W)
            
            # 计算指标（逐样本）
            batch_size = y_pred.shape[0]
            for i in range(batch_size):
                y_pred_i = y_pred[i:i+1]  # (1, C, H, W)
                y_true_i = labels[i:i+1]  # (1, H, W)
                
                # 计算指标
                metrics = calculator.calculate_metrics(y_pred_i, y_true_i)
                all_metrics.append(metrics)
                
                # 获取图像名称
                if isinstance(names, list) and i < len(names):
                    image_name = names[i]
                else:
                    image_name = f"sample_{batch_idx}_{i}"
                image_names.append(image_name)
                
                # 保存预测结果（如果需要）
                if save_predictions and saver is not None:
                    # 获取仿射变换矩阵（如果可用）
                    affine = np.eye(4)  # 默认单位矩阵
                    if isinstance(batch, dict) and 'affine' in batch:
                        affine = batch['affine'][i] if isinstance(batch['affine'], (list, np.ndarray)) else batch['affine']
                    
                    saver.save_single_nifti(
                        pred=y_pred_i.cpu(),
                        save_name=image_name,
                        affine=affine,
                        is_logits=True
                    )
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0:
                print(f"  进度: {batch_idx + 1}/{len(dataloader)} 批次")
    
    # 计算平均指标
    if not all_metrics:
        print("  警告: 没有有效的评估结果")
        return {}
    
    # 计算所有样本的平均指标
    mean_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        if values:
            # 过滤掉 inf 值
            valid_values = [v for v in values if v != float('inf')]
            if valid_values:
                mean_metrics[key] = float(np.mean(valid_values))
            else:
                mean_metrics[key] = float('inf')
    
    # 保存批量指标（如果提供了保存器）
    if saver is not None:
        csv_path = saver.save_batch_metrics_to_csv(
            metrics_list=all_metrics,
            filename="evaluation_results.csv",
            include_image_names=True,
            image_names=image_names
        )
        print(f"\n  ✓ 指标已保存到: {csv_path}")
    
    return mean_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估训练好的模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估最新检查点
  python scripts/evaluate_model.py --checkpoint_dir checkpoints --data_root data
  
  # 评估指定检查点并保存结果
  python scripts/evaluate_model.py \\
      --checkpoint checkpoints/best_model.pth \\
      --data_root data \\
      --save_dir results/evaluation \\
      --save_predictions
  
  # 评估验证集
  python scripts/evaluate_model.py \\
      --checkpoint checkpoints/best_model.pth \\
      --data_root data \\
      --split val \\
      --save_dir results/val_evaluation
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='检查点文件路径（.pth）'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='检查点目录（将自动查找 latest_checkpoint.pth 或 best_model.pth，默认: 自动查找）'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='数据根目录（默认: data）'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='评估数据集分割（默认: val）'
    )
    parser.add_argument(
        '--client_id',
        type=str,
        default=None,
        help='客户端ID（如果为None，评估所有客户端）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小（默认: 4）'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='结果保存目录（如果为None，不保存）'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='是否保存预测掩码（需要 --save_dir）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备（cuda/cpu，默认: 自动检测）'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='模型类别数（如果为None，则自动从检查点检测，默认: 自动检测）'
    )
    
    args = parser.parse_args()
    
    # 确定设备
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载检查点
    checkpoint_path = None
    
    # 如果用户指定了检查点
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"错误: 指定的检查点文件不存在: {checkpoint_path}")
            return 1
    elif args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"错误: 指定的检查点目录不存在: {checkpoint_dir}")
            return 1
        
        # 尝试查找检查点
        for name in ['latest_checkpoint.pth', 'best_model.pth', 'final_model.pth']:
            candidate = checkpoint_dir / name
            if candidate.exists():
                checkpoint_path = candidate
                break
        
        if checkpoint_path is None:
            # 查找所有 .pth 文件
            pth_files = list(checkpoint_dir.glob("*.pth"))
            if pth_files:
                # 使用最新的文件
                checkpoint_path = max(pth_files, key=lambda p: p.stat().st_mtime)
                print(f"  找到检查点: {checkpoint_path}")
    
    # 如果用户没有指定，尝试自动查找
    if checkpoint_path is None or not checkpoint_path.exists():
        print("\n未指定检查点，尝试自动查找...")
        
        # 常见的检查点位置
        common_locations = [
            Path("data/checkpoints/final_model.pth"),
            Path("data/checkpoints/best_model.pth"),
            Path("data/checkpoints/latest_checkpoint.pth"),
            Path("checkpoints/final_model.pth"),
            Path("checkpoints/best_model.pth"),
            Path("checkpoints/latest_checkpoint.pth"),
        ]
        
        # 查找存在的检查点文件
        for location in common_locations:
            if location.exists():
                checkpoint_path = location
                print(f"  ✓ 自动找到检查点: {checkpoint_path}")
                break
        
        # 如果还是没找到，尝试查找目录
        if checkpoint_path is None or not checkpoint_path.exists():
            for dir_path in [Path("data/checkpoints"), Path("checkpoints")]:
                if dir_path.exists():
                    pth_files = list(dir_path.glob("*.pth"))
                    if pth_files:
                        checkpoint_path = max(pth_files, key=lambda p: p.stat().st_mtime)
                        print(f"  ✓ 自动找到检查点: {checkpoint_path}")
                        break
    
    # 最终检查
    if checkpoint_path is None or not checkpoint_path.exists():
        print("\n错误: 未找到检查点文件")
        print("\n请使用以下方式之一指定检查点:")
        print("  1. 使用 --checkpoint 指定文件路径:")
        print("     python scripts/evaluate_model.py --checkpoint data/checkpoints/final_model.pth")
        print("  2. 使用 --checkpoint_dir 指定目录:")
        print("     python scripts/evaluate_model.py --checkpoint_dir data/checkpoints")
        print("\n常见检查点位置:")
        print("  - data/checkpoints/final_model.pth")
        print("  - data/checkpoints/best_model.pth")
        print("  - checkpoints/latest_checkpoint.pth")
        return 1
    
    # 加载模型
    try:
        model = load_model(checkpoint_path, device=device, num_classes=args.num_classes)
        
        # 获取模型的 num_classes
        model_num_classes = model.num_classes
        print(f"\n模型类别数: {model_num_classes}")
        
        if model_num_classes == 1:
            print("  警告: 检测到单类分割模型（num_classes=1）")
            print("  注意: MedicalMetricsCalculator 是为 BraTS 多类分割设计的")
            print("  单类分割模型将使用二值分割评估（WT区域）")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        print("\n提示: 如果遇到类别数不匹配的错误，可以尝试:")
        print("  1. 使用 --num_classes 1 指定单类分割模型")
        print("  2. 使用 --num_classes 4 指定多类分割模型（BraTS）")
        import traceback
        traceback.print_exc()
        return 1
    
    # 创建数据加载器
    print(f"\n创建数据加载器（split={args.split}）...")
    try:
        # 客户端配置
        if args.client_id:
            client_configs = [{'client_id': args.client_id, 'has_private': True, 'has_public': False}]
        else:
            # 评估所有客户端
            client_configs = [
                {'client_id': 'client_1', 'has_private': True, 'has_public': False},
                {'client_id': 'client_2', 'has_private': True, 'has_public': False},
                {'client_id': 'client_3', 'has_private': True, 'has_public': False},
            ]
        
        dataloaders = create_data_loaders(
            data_root=args.data_root,
            split=args.split,
            client_configs=client_configs,
            batch_size=args.batch_size,
            image_size=1024,
            shuffle=False  # 评估时不打乱
        )
        
        print(f"  ✓ 创建了 {len(dataloaders)} 个数据加载器")
    except Exception as e:
        print(f"错误: 创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 初始化评估工具
    calculator = MedicalMetricsCalculator(device=device)
    
    saver = None
    if args.save_dir:
        saver = PredictionSaver(args.save_dir)
        print(f"\n结果将保存到: {args.save_dir}")
    
    # 评估每个客户端
    print("\n" + "=" * 60)
    print("开始评估")
    print("=" * 60)
    
    all_results = {}
    
    for client_idx, (client_config, dataloader) in enumerate(zip(client_configs, dataloaders)):
        client_id = client_config['client_id']
        print(f"\n[客户端 {client_idx + 1}/{len(client_configs)}] {client_id}")
        print("-" * 60)
        
        try:
            metrics = evaluate_on_dataset(
                model=model,
                dataloader=dataloader,
                calculator=calculator,
                saver=saver,
                device=device,
                save_predictions=args.save_predictions,
                num_classes=model.num_classes
            )
            
            all_results[client_id] = metrics
            
            # 打印结果
            if metrics:
                print(f"\n评估结果 ({client_id}):")
                print(f"  WT Dice:  {metrics.get('WT_Dice', 0.0):.4f}")
                print(f"  TC Dice:  {metrics.get('TC_Dice', 0.0):.4f}")
                print(f"  ET Dice:  {metrics.get('ET_Dice', 0.0):.4f}")
                print(f"  Mean Dice: {metrics.get('Mean_Dice', 0.0):.4f}")
                print()
                print(f"  WT HD95:  {metrics.get('WT_HD95', float('inf')):.4f}")
                print(f"  TC HD95:  {metrics.get('TC_HD95', float('inf')):.4f}")
                print(f"  ET HD95:  {metrics.get('ET_HD95', float('inf')):.4f}")
                mean_hd95 = metrics.get('Mean_HD95', float('inf'))
                if mean_hd95 != float('inf'):
                    print(f"  Mean HD95: {mean_hd95:.4f}")
                else:
                    print(f"  Mean HD95: inf")
        except Exception as e:
            print(f"  错误: 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算总体平均
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("总体平均结果")
        print("=" * 60)
        
        # 计算所有客户端的平均
        avg_metrics = {}
        for key in ['WT_Dice', 'TC_Dice', 'ET_Dice', 'Mean_Dice', 'WT_HD95', 'TC_HD95', 'ET_HD95', 'Mean_HD95']:
            values = [r[key] for r in all_results.values() if key in r]
            if values:
                valid_values = [v for v in values if v != float('inf')]
                if valid_values:
                    avg_metrics[key] = float(np.mean(valid_values))
                else:
                    avg_metrics[key] = float('inf')
        
        print(f"  WT Dice:  {avg_metrics.get('WT_Dice', 0.0):.4f}")
        print(f"  TC Dice:  {avg_metrics.get('TC_Dice', 0.0):.4f}")
        print(f"  ET Dice:  {avg_metrics.get('ET_Dice', 0.0):.4f}")
        print(f"  Mean Dice: {avg_metrics.get('Mean_Dice', 0.0):.4f}")
        print()
        mean_hd95 = avg_metrics.get('Mean_HD95', float('inf'))
        if mean_hd95 != float('inf'):
            print(f"  Mean HD95: {mean_hd95:.4f}")
        else:
            print(f"  Mean HD95: inf")
        
        # 保存总体结果
        if saver:
            saver.save_metrics_to_csv(
                metrics_dict=avg_metrics,
                filename="overall_metrics.csv",
                append=False
            )
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

