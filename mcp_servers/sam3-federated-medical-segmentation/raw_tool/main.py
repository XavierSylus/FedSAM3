"""
SAM3 Federated Medical Image Segmentation Model - MCP Tool Entry Point

整合了：
1. SAM3真实模型（sam3-main）
2. SAM-Adapter适配器机制
3. CreamFL多模态联邦学习
4. FedFMS联邦架构
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add parent directories to path
# 当作为MCP工具运行时，路径结构：mcp_servers/sam3-federated-medical-segmentation/raw_tool/main.py
# 需要找到项目根目录：从 raw_tool 向上4级到项目根目录
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/sam3_federated_model/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "core_projects" / "sam3-main"))
sys.path.insert(0, str(project_root / "core_projects" / "SAM-Adapter-PyTorch-main"))
sys.path.insert(0, str(project_root / "core_projects" / "CreamFL-main" / "src"))
sys.path.insert(0, str(project_root / "core_projects" / "FedFMS-main"))

try:
    from src.integrated_model import SAM3MedicalIntegrated, DEVICE
    from src.integrated_client import IntegratedClientTrainer
    from src.server import CreamAggregator
except ImportError as e:
    print(f"Warning: Could not import integrated modules: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:5]}")


class SAM3FederatedModelLoader:
    """SAM3联邦学习模型加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - model_arch: str, 'sam3' or 'sam3-adapter'
                - sam3_checkpoint: str, SAM3预训练权重路径
                - freeze_encoder: bool, 是否冻结编码器
                - use_adapter: bool, 是否使用适配器
                - img_size: int, 图像尺寸
                - num_classes: int, 类别数
                - device: str, 设备
        """
        self.config = config
        self.device = torch.device(config.get('device', DEVICE))
        self.model = None
        
    def load_model(self) -> SAM3MedicalIntegrated:
        """加载整合的SAM3模型"""
        self.model = SAM3MedicalIntegrated(
            img_size=self.config.get('img_size', 1024),
            num_classes=self.config.get('num_classes', 1),
            adapter_dim=self.config.get('adapter_dim', 64),
            use_sam3=self.config.get('use_sam3', True),
            freeze_encoder=self.config.get('freeze_encoder', True),
            use_adapter=self.config.get('use_adapter', True),
            sam3_checkpoint=self.config.get('sam3_checkpoint'),
            device=str(self.device)
        )
        return self.model
    
    def get_trainable_params_info(self) -> Dict[str, Any]:
        """获取可训练参数信息"""
        if self.model is None:
            return {}
        
        trainable_params = self.model.get_trainable_params()
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'trainable_params': total_trainable,
            'total_params': total_params,
            'trainable_ratio': total_trainable / total_params if total_params > 0 else 0
        }


def run_inference(
    model: SAM3MedicalIntegrated,
    image_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行推理
    
    Args:
        model: 加载的模型
        image_path: 图像路径
        config: 配置字典
    
    Returns:
        推理结果字典
    """
    device = torch.device(config.get('device', DEVICE))
    img_size = config.get('img_size', 1024)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, return_features=False)
        mask = torch.sigmoid(output['logits']).cpu().numpy()[0, 0]
    
    return {
        'mask': mask.tolist(),
        'shape': list(mask.shape),
        'device': str(device)
    }


def main():
    """主入口"""
    import json
    
    # 示例配置
    config = {
        'use_sam3': True,
        'freeze_encoder': True,
        'use_adapter': True,
        'img_size': 1024,
        'num_classes': 1,
        'adapter_dim': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sam3_checkpoint': None  # 设置SAM3权重路径
    }
    
    # 加载模型
    loader = SAM3FederatedModelLoader(config)
    model = loader.load_model()
    
    print("Model loaded successfully!")
    
    # 打印参数信息
    param_info = loader.get_trainable_params_info()
    print(f"Trainable parameters: {param_info.get('trainable_params', 0):,} / {param_info.get('total_params', 0):,}")
    print(f"Trainable ratio: {param_info.get('trainable_ratio', 0):.2%}")
    
    # 示例推理（如果提供图像路径）
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = run_inference(model, image_path, config)
        print(f"Inference result shape: {result['shape']}")


if __name__ == "__main__":
    main()
