"""
PyTorch 训练脚本 - 集成 WandB
支持记录 loss 和 accuracy，包含 learning_rate 和 epochs 超参数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from typing import Optional, Dict, Any
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import create_logger


class Trainer:
    """
    通用训练器类
    支持 WandB 日志记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "pytorch-training",
        wandb_entity: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 训练设备
            use_wandb: 是否使用 WandB
            wandb_project: WandB 项目名称
            wandb_entity: WandB entity（用户名或团队名）
            experiment_name: 实验名称
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化 WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    reinit=True
                )
                print(f"✓ WandB 已初始化: {wandb.run.url if wandb.run else 'N/A'}")
            except Exception as e:
                print(f"⚠ WandB 初始化失败: {e}")
                self.use_wandb = False
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            optimizer: 优化器
            criterion: 损失函数
            epoch: 当前 epoch 编号
        
        Returns:
            训练指标字典
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 处理不同的批次格式
            if isinstance(batch, dict):
                inputs = batch.get('image', batch.get('inp', batch.get('inputs')))
                targets = batch.get('label', batch.get('gt', batch.get('targets', batch.get('mask'))))
            elif isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            if targets is not None:
                # 如果是分割任务，可能需要调整 targets 的形状
                if outputs.dim() == 4 and targets.dim() == 3:
                    # 分割任务：outputs (B, C, H, W), targets (B, H, W)
                    targets = targets.unsqueeze(1)
                elif outputs.dim() == 4 and targets.dim() == 4:
                    # 分割任务：outputs (B, C, H, W), targets (B, C, H, W)
                    pass
                elif outputs.dim() == 2 and targets.dim() == 1:
                    # 分类任务：outputs (B, num_classes), targets (B,)
                    pass
                
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算准确率（仅对分类任务）
                if outputs.dim() == 2:  # 分类任务
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            else:
                # 如果没有标签，只计算损失（可能需要其他方式）
                loss = torch.tensor(0.0, device=self.device)
            
            # 累计损失
            running_loss += loss.item()
            
            # 记录到 WandB（每个批次）
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/epoch": epoch,
                    "train/batch": batch_idx
                })
        
        # 计算平均指标
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(
        self,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            criterion: 损失函数
            epoch: 当前 epoch 编号
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 处理不同的批次格式
                if isinstance(batch, dict):
                    inputs = batch.get('image', batch.get('inp', batch.get('inputs')))
                    targets = batch.get('label', batch.get('gt', batch.get('targets', batch.get('mask'))))
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if targets is not None:
                    # 调整 targets 形状（如果需要）
                    if outputs.dim() == 4 and targets.dim() == 3:
                        targets = targets.unsqueeze(1)
                    
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()
                    
                    # 计算准确率（仅对分类任务）
                    if outputs.dim() == 2:  # 分类任务
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
        
        # 计算平均指标
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        epochs: int,
        learning_rate: float,
        optimizer_type: str = "adam",
        criterion: Optional[nn.Module] = None,
        save_path: Optional[str] = None,
        save_best: bool = True
    ):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            optimizer_type: 优化器类型 ("adam", "sgd", "adamw")
            criterion: 损失函数（如果为 None，使用默认）
            save_path: 模型保存路径
            save_best: 是否保存最佳模型
        """
        # 设置默认损失函数
        if criterion is None:
            # 根据任务类型选择损失函数
            # 这里使用交叉熵损失（适用于分类任务）
            criterion = nn.CrossEntropyLoss()
        
        # 创建优化器
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 记录超参数到 WandB
        if self.use_wandb:
            wandb.config.update({
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer": optimizer_type,
                "batch_size": self.train_loader.batch_size,
                "device": self.device
            })
        
        # 最佳验证准确率
        best_val_acc = 0.0
        
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"学习率: {learning_rate}")
        print(f"优化器: {optimizer_type}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print("=" * 60)
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics = self.train_epoch(optimizer, criterion, epoch)
            
            # 验证
            val_metrics = self.validate(criterion, epoch) if self.val_loader else {}
            
            # 打印指标
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
            if val_metrics:
                print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # 记录到 WandB
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy']
                }
                if val_metrics:
                    log_dict.update({
                        "val/loss": val_metrics['loss'],
                        "val/accuracy": val_metrics['accuracy']
                    })
                wandb.log(log_dict)
            
            # 保存最佳模型
            if save_best and val_metrics and val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': best_val_acc,
                    }, save_path)
                    print(f"  ✓ 保存最佳模型 (验证准确率: {best_val_acc:.2f}%)")
        
        # 记录最终结果到 WandB
        if self.use_wandb:
            wandb.run.summary.update({
                "best_val_accuracy": best_val_acc if self.val_loader else train_metrics['accuracy'],
                "final_train_loss": train_metrics['loss'],
                "final_train_accuracy": train_metrics['accuracy']
            })
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print("=" * 60)
        
        # 关闭 WandB
        if self.use_wandb:
            wandb.finish()


def create_dummy_dataset(size: int = 100, num_classes: int = 10):
    """创建虚拟数据集用于测试"""
    from torch.utils.data import TensorDataset
    
    # 创建虚拟数据
    images = torch.randn(size, 3, 32, 32)
    labels = torch.randint(0, num_classes, (size,))
    
    return TensorDataset(images, labels)


def create_dummy_model(num_classes: int = 10):
    """创建虚拟模型用于测试"""
    import torchvision.models as models
    
    # 使用预训练的 ResNet18
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch 训练脚本 - 集成 WandB")
    
    # 超参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"], help="优化器类型")
    
    # WandB 配置
    parser.add_argument("--wandb_project", type=str, default="pytorch-training", help="WandB 项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity（用户名或团队名）")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 WandB")
    
    # 模型和数据配置
    parser.add_argument("--num_classes", type=int, default=10, help="分类类别数")
    parser.add_argument("--use_dummy", action="store_true", help="使用虚拟数据（用于测试）")
    
    # 其他配置
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径")
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 创建虚拟数据集（用于演示）
    if args.use_dummy:
        print("使用虚拟数据集...")
        train_dataset = create_dummy_dataset(size=1000, num_classes=args.num_classes)
        val_dataset = create_dummy_dataset(size=200, num_classes=args.num_classes)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 创建模型
        model = create_dummy_model(num_classes=args.num_classes)
    else:
        print("请提供真实的数据集和模型")
        print("示例:")
        print("  from your_dataset import YourDataset")
        print("  from your_model import YourModel")
        print("  train_dataset = YourDataset(...)")
        print("  model = YourModel(...)")
        return
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        experiment_name=args.experiment_name
    )
    
    # 开始训练
    trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()

