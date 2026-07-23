"""
SAM-Adapter Block Modules - MCP Tool Entry Point

封装了 SAM-Adapter 项目中的 block.py 模块功能，提供：
1. MergeAndConv - 合并和卷积模块
2. SideClassifer - 侧分类器
3. UpsampleSKConv - 上采样SK卷积
4. SKSPP - SK空间金字塔池化模块
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np

# ============================================
# 路径设置
# ============================================
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/sam_adapter_block/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core_projects" / "SAM-Adapter-PyTorch-main"))

try:
    from models.block import (
        MergeAndConv,
        SideClassifer,
        UpsampleSKConv,
        SKSPP,
    )
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:5]}")


class SAMAdapterBlockLoader:
    """SAM-Adapter Block模块加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器
        
        Args:
            config: 配置字典
                - block_type: str, 模块类型 ('MergeAndConv', 'SideClassifer', 'UpsampleSKConv', 'SKSPP')
                - 其他模块特定的参数...
        """
        self.config = config
        self.block = None
        self.block_type = config.get('block_type', 'MergeAndConv')
        
    def load(self) -> Any:
        """
        加载模块构建器（返回配置信息）
        
        Returns:
            配置字典
        """
        return {
            'block_type': self.block_type,
            'config': self.config,
            'status': 'ready'
        }
    
    def build_block(self) -> Any:
        """
        构建SAM-Adapter Block模块
        
        Returns:
            构建的模块对象
        """
        block_type = self.config.get('block_type', 'MergeAndConv')
        
        if block_type == 'MergeAndConv':
            return self._build_merge_and_conv()
        elif block_type == 'SideClassifer':
            return self._build_side_classifier()
        elif block_type == 'UpsampleSKConv':
            return self._build_upsample_sk_conv()
        elif block_type == 'SKSPP':
            return self._build_skspp()
        else:
            raise ValueError(f"Unknown block_type: {block_type}. Must be one of: MergeAndConv, SideClassifer, UpsampleSKConv, SKSPP")
    
    def _build_merge_and_conv(self):
        """构建MergeAndConv模块"""
        return MergeAndConv(
            ic=self.config.get('ic', 256),
            oc=self.config.get('oc', 256),
            inner=self.config.get('inner', 32)
        )
    
    def _build_side_classifier(self):
        """构建SideClassifer模块"""
        return SideClassifer(
            ic=self.config.get('ic', 256),
            n_class=self.config.get('n_class', 1),
            M=self.config.get('M', 2),
            kernel_size=self.config.get('kernel_size', 1)
        )
    
    def _build_upsample_sk_conv(self):
        """构建UpsampleSKConv模块"""
        return UpsampleSKConv(
            ic=self.config.get('ic', 256),
            oc=self.config.get('oc', 128),
            reduce=self.config.get('reduce', 4)
        )
    
    def _build_skspp(self):
        """构建SKSPP模块"""
        return SKSPP(
            features=self.config.get('features', 256),
            WH=self.config.get('WH', 64),
            M=self.config.get('M', 2),
            G=self.config.get('G', 1),
            r=self.config.get('r', 16),
            stride=self.config.get('stride', 1),
            L=self.config.get('L', 32)
        )
    
    def get_block_info(self) -> Dict[str, Any]:
        """
        获取模块信息
        
        Returns:
            模块信息字典
        """
        if self.block is None:
            return {
                'status': 'not_built',
                'block_type': self.block_type
            }
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.block.parameters())
        trainable_params = sum(p.numel() for p in self.block.parameters() if p.requires_grad)
        
        return {
            'status': 'built',
            'block_type': self.block_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'has_block': True
        }


def run_inference(
    builder: Dict[str, Any],
    action: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行模块构建或前向传播
    
    Args:
        builder: 构建器对象（配置字典）
        action: 操作类型 ('build', 'forward')
        config: 配置字典
    
    Returns:
        结果字典
    """
    loader = SAMAdapterBlockLoader(config)
    
    if action == 'build':
        block = loader.build_block()
        loader.block = block
        
        info = loader.get_block_info()
        return {
            'status': 'success',
            'action': 'build',
            'block_type': config.get('block_type', 'MergeAndConv'),
            'block_info': info,
            'message': f'Block {config.get("block_type", "MergeAndConv")} built successfully'
        }
    
    elif action == 'forward':
        # 构建模块
        block = loader.build_block()
        loader.block = block
        
        # 获取输入数据
        input_shape = config.get('input_shape', [1, 256, 64, 64])
        input_data = config.get('input_data')
        
        # 创建输入tensor
        if input_data is not None:
            if isinstance(input_data, list):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            # 使用随机数据
            input_tensor = torch.randn(*input_shape)
        
        # 设置为评估模式
        block.eval()
        
        # 前向传播
        with torch.no_grad():
            output = block(input_tensor)
        
        # 转换输出为可序列化格式
        if isinstance(output, list):
            output_data = [out.cpu().numpy().tolist() for out in output]
            output_shape = [list(out.shape) for out in output]
        else:
            output_data = output.cpu().numpy().tolist()
            output_shape = list(output.shape)
        
        info = loader.get_block_info()
        return {
            'status': 'success',
            'action': 'forward',
            'block_type': config.get('block_type', 'MergeAndConv'),
            'input_shape': list(input_tensor.shape),
            'output_shape': output_shape,
            'output_data': output_data,
            'block_info': info
        }
    
    elif action == 'get_info':
        block = loader.build_block()
        loader.block = block
        info = loader.get_block_info()
        return {
            'status': 'success',
            'action': 'get_info',
            'block_type': config.get('block_type', 'MergeAndConv'),
            'block_info': info
        }
    
    else:
        return {
            'status': 'error',
            'error': f'Unknown action: {action}. Must be one of: build, forward, get_info'
        }


def main():
    """主入口，用于直接运行测试"""
    import json
    
    # 示例配置 - 构建MergeAndConv模块
    config = {
        'block_type': 'MergeAndConv',
        'ic': 256,
        'oc': 256,
        'inner': 32,
        'input_shape': [1, 256, 64, 64]
    }
    
    # 加载构建器
    loader = SAMAdapterBlockLoader(config)
    builder = loader.load()
    
    print("Block builder loaded successfully!")
    print(f"Block type: {builder['block_type']}")
    
    # 测试构建
    if len(sys.argv) > 1:
        action = sys.argv[1]
        result = run_inference(builder, action, config)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()








