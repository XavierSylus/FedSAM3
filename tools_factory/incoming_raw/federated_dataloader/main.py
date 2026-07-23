"""
Federated DataLoader - MCP Tool Entry Point

封装了 federated_dataloader.py 的功能，支持：
1. .npy 格式数据（原有格式）
2. .nii.gz 格式数据（BraTS/AMOS数据集）
3. 多模态支持（T1/T2/FLAIR）
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from torch.utils.data import DataLoader

# ============================================
# 路径设置
# ============================================
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/federated_dataloader/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core_projects" / "FedFMS-main"))

try:
    from dataloaders.federated_dataloader import (
        Dataset,
        ProstateDataset,
        BraTSDataset,
        AMOSDataset,
    )
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:5]}")


class FederatedDataLoaderLoader:
    """联邦数据加载器加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器
        
        Args:
            config: 配置字典
                - dataset_type: str, 数据集类型 ('Dataset', 'ProstateDataset', 'BraTSDataset', 'AMOSDataset')
                - data_path: str, 数据路径
                - data_format: str, 数据格式 ('nii.gz' 或 'npy')
                - modalities: list, 模态列表 ['T1', 'T2', 'FLAIR']
                - client_idx: int, 客户端索引
                - split: str, 数据集分割 ('train', 'val', 'test')
        """
        self.config = config
        self.dataset = None
        self.dataloader = None
        
    def load(self) -> Any:
        """
        加载数据加载器配置
        
        Returns:
            配置字典
        """
        return {
            'dataset_type': self.config.get('dataset_type', 'BraTSDataset'),
            'config': self.config,
            'status': 'ready'
        }
    
    def build_dataset(self) -> Any:
        """
        构建数据集
        
        Returns:
            数据集对象
        """
        dataset_type = self.config.get('dataset_type', 'BraTSDataset')
        
        common_params = {
            'data_path': self.config.get('data_path'),
            'client_idx': self.config.get('client_idx'),
            'freq_site_idx': self.config.get('freq_site_idx'),
            'split': self.config.get('split', 'train'),
            'transform': self.config.get('transform'),
            'client_name': self.config.get('client_name'),
        }
        
        if dataset_type == 'BraTSDataset':
            self.dataset = BraTSDataset(
                **common_params,
                modalities=self.config.get('modalities', ['T1', 'T2', 'FLAIR']),
                data_format=self.config.get('data_format', 'nii.gz')
            )
        elif dataset_type == 'AMOSDataset':
            self.dataset = AMOSDataset(
                **common_params,
                modalities=self.config.get('modalities', ['T1', 'T2', 'FLAIR']),
                data_format=self.config.get('data_format', 'nii.gz')
            )
        elif dataset_type == 'ProstateDataset':
            self.dataset = ProstateDataset(**common_params)
        elif dataset_type == 'Dataset':
            self.dataset = Dataset(**common_params)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        return self.dataset
    
    def build_dataloader(self) -> Any:
        """
        构建数据加载器
        
        Returns:
            DataLoader对象
        """
        if self.dataset is None:
            self.build_dataset()
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=self.config.get('shuffle', True),
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', False)
        )
        
        return self.dataloader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            数据集信息字典
        """
        if self.dataset is None:
            return {
                'status': 'not_built',
                'dataset_type': self.config.get('dataset_type', 'BraTSDataset')
            }
        
        return {
            'status': 'built',
            'dataset_type': self.config.get('dataset_type', 'BraTSDataset'),
            'dataset_length': len(self.dataset),
            'has_dataloader': self.dataloader is not None
        }


def run_inference(
    builder: Dict[str, Any],
    action: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行数据集构建或数据加载
    
    Args:
        builder: 构建器对象（配置字典）
        action: 操作类型 ('build_dataset', 'build_dataloader', 'get_sample', 'get_batch')
        config: 配置字典
    
    Returns:
        结果字典
    """
    loader = FederatedDataLoaderLoader(config)
    
    if action == 'build_dataset':
        dataset = loader.build_dataset()
        info = loader.get_dataset_info()
        return {
            'status': 'success',
            'action': 'build_dataset',
            'dataset_type': config.get('dataset_type', 'BraTSDataset'),
            'dataset_info': info,
            'message': f'Dataset {config.get("dataset_type", "BraTSDataset")} built successfully'
        }
    
    elif action == 'build_dataloader':
        dataloader = loader.build_dataloader()
        info = loader.get_dataset_info()
        return {
            'status': 'success',
            'action': 'build_dataloader',
            'dataset_type': config.get('dataset_type', 'BraTSDataset'),
            'dataset_info': info,
            'batch_size': config.get('batch_size', 1),
            'message': 'DataLoader built successfully'
        }
    
    elif action == 'get_sample':
        dataset = loader.build_dataset()
        idx = config.get('sample_idx', 0)
        if idx >= len(dataset):
            return {
                'status': 'error',
                'error': f'Sample index {idx} out of range. Dataset has {len(dataset)} samples.'
            }
        
        sample = dataset[idx]
        
        # Convert to serializable format
        result = {
            'status': 'success',
            'action': 'get_sample',
            'sample_idx': idx,
            'image_shape': list(sample['image'].shape) if isinstance(sample['image'], np.ndarray) else list(sample['image'].size()),
            'label_shape': list(sample['label'].shape) if isinstance(sample['label'], np.ndarray) else list(sample['label'].size()),
        }
        
        # Include image/label data if requested
        if config.get('include_data', False):
            if isinstance(sample['image'], torch.Tensor):
                result['image'] = sample['image'].cpu().numpy().tolist()
            else:
                result['image'] = sample['image'].tolist()
            
            if isinstance(sample['label'], torch.Tensor):
                result['label'] = sample['label'].cpu().numpy().tolist()
            else:
                result['label'] = sample['label'].tolist()
        
        return result
    
    elif action == 'get_batch':
        dataloader = loader.build_dataloader()
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        result = {
            'status': 'success',
            'action': 'get_batch',
            'batch_size': batch['image'].shape[0],
            'image_shape': list(batch['image'].shape),
            'label_shape': list(batch['label'].shape),
        }
        
        if config.get('include_data', False):
            result['image'] = batch['image'].cpu().numpy().tolist()
            result['label'] = batch['label'].cpu().numpy().tolist()
        
        return result
    
    elif action == 'get_info':
        dataset = loader.build_dataset()
        info = loader.get_dataset_info()
        return {
            'status': 'success',
            'action': 'get_info',
            'dataset_type': config.get('dataset_type', 'BraTSDataset'),
            'dataset_info': info
        }
    
    else:
        return {
            'status': 'error',
            'error': f'Unknown action: {action}. Must be one of: build_dataset, build_dataloader, get_sample, get_batch, get_info'
        }


def main():
    """主入口，用于直接运行测试"""
    import json
    
    # 示例配置 - BraTS数据集
    config = {
        'dataset_type': 'BraTSDataset',
        'data_path': './data/BraTS',
        'data_format': 'nii.gz',
        'modalities': ['T1', 'T2', 'FLAIR'],
        'client_idx': 0,
        'split': 'train',
        'batch_size': 1,
        'shuffle': True,
    }
    
    # 加载构建器
    loader = FederatedDataLoaderLoader(config)
    builder = loader.load()
    
    print("DataLoader builder loaded successfully!")
    print(f"Dataset type: {builder['dataset_type']}")
    
    # 测试构建
    if len(sys.argv) > 1:
        action = sys.argv[1]
        result = run_inference(builder, action, config)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()






