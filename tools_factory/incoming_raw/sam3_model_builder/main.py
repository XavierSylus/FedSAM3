"""
SAM3 Model Builder - MCP Tool Entry Point

封装了 sam3.model_builder 模块的功能，提供：
1. 构建SAM3图像模型 (build_sam3_image_model)
2. 构建SAM3视频模型 (build_sam3_video_model)
3. 构建跟踪器 (build_tracker)
4. 下载检查点 (download_ckpt_from_hf)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================
# 路径设置
# ============================================
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/sam3_model_builder/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core_projects" / "sam3-main"))

try:
    from sam3.model_builder import (
        build_sam3_image_model,
        build_sam3_video_model,
        build_tracker,
        download_ckpt_from_hf,
    )
    import torch
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:5]}")


class SAM3ModelBuilderLoader:
    """SAM3模型构建器加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器
        
        Args:
            config: 配置字典
                - model_type: str, 'image' 或 'video' 或 'tracker'
                - device: str, 'cuda' 或 'cpu'
                - 其他模型构建参数...
        """
        self.config = config
        self.model = None
        self.model_type = config.get('model_type', 'image')
        
    def load(self) -> Any:
        """
        加载模型构建器（返回配置信息）
        
        Returns:
            配置字典
        """
        return {
            'model_type': self.model_type,
            'config': self.config,
            'status': 'ready'
        }
    
    def build_model(self) -> Any:
        """
        构建SAM3模型
        
        Returns:
            构建的模型对象
        """
        model_type = self.config.get('model_type', 'image')
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'image':
            return self._build_image_model()
        elif model_type == 'video':
            return self._build_video_model()
        elif model_type == 'tracker':
            return self._build_tracker()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'image', 'video', or 'tracker'")
    
    def _build_image_model(self):
        """构建SAM3图像模型"""
        return build_sam3_image_model(
            bpe_path=self.config.get('bpe_path'),
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            eval_mode=self.config.get('eval_mode', True),
            checkpoint_path=self.config.get('checkpoint_path'),
            load_from_HF=self.config.get('load_from_HF', True),
            enable_segmentation=self.config.get('enable_segmentation', True),
            enable_inst_interactivity=self.config.get('enable_inst_interactivity', False),
            compile=self.config.get('compile', False),
        )
    
    def _build_video_model(self):
        """构建SAM3视频模型"""
        return build_sam3_video_model(
            checkpoint_path=self.config.get('checkpoint_path'),
            load_from_HF=self.config.get('load_from_HF', True),
            bpe_path=self.config.get('bpe_path'),
            has_presence_token=self.config.get('has_presence_token', True),
            geo_encoder_use_img_cross_attn=self.config.get('geo_encoder_use_img_cross_attn', True),
            strict_state_dict_loading=self.config.get('strict_state_dict_loading', True),
            apply_temporal_disambiguation=self.config.get('apply_temporal_disambiguation', True),
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            compile=self.config.get('compile', False),
        )
    
    def _build_tracker(self):
        """构建SAM3跟踪器"""
        return build_tracker(
            apply_temporal_disambiguation=self.config.get('apply_temporal_disambiguation', True),
            with_backbone=self.config.get('with_backbone', False),
            compile_mode='default' if self.config.get('compile', False) else None,
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {
                'status': 'not_built',
                'model_type': self.model_type
            }
        
        return {
            'status': 'built',
            'model_type': self.model_type,
            'device': str(next(self.model.parameters()).device) if hasattr(self.model, 'parameters') else 'unknown',
            'has_model': True
        }


def run_inference(
    builder: Dict[str, Any],
    action: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行模型构建或下载操作
    
    Args:
        builder: 构建器对象（配置字典）
        action: 操作类型 ('build_image', 'build_video', 'build_tracker', 'download_ckpt')
        config: 配置字典
    
    Returns:
        结果字典
    """
    loader = SAM3ModelBuilderLoader(config)
    
    if action == 'build_image':
        model = loader._build_image_model()
        return {
            'status': 'success',
            'model_type': 'image',
            'model_info': loader.get_model_info(),
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown'
        }
    
    elif action == 'build_video':
        model = loader._build_video_model()
        return {
            'status': 'success',
            'model_type': 'video',
            'model_info': loader.get_model_info(),
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown'
        }
    
    elif action == 'build_tracker':
        model = loader._build_tracker()
        return {
            'status': 'success',
            'model_type': 'tracker',
            'model_info': loader.get_model_info()
        }
    
    elif action == 'download_ckpt':
        checkpoint_path = download_ckpt_from_hf()
        return {
            'status': 'success',
            'action': 'download_ckpt',
            'checkpoint_path': checkpoint_path,
            'message': f'Checkpoint downloaded to: {checkpoint_path}'
        }
    
    else:
        return {
            'status': 'error',
            'error': f'Unknown action: {action}. Must be one of: build_image, build_video, build_tracker, download_ckpt'
        }


def main():
    """主入口，用于直接运行测试"""
    import json
    
    # 示例配置 - 构建图像模型
    config = {
        'model_type': 'image',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_mode': True,
        'load_from_HF': True,
        'enable_segmentation': True,
        'enable_inst_interactivity': False,
        'compile': False,
    }
    
    # 加载构建器
    loader = SAM3ModelBuilderLoader(config)
    builder = loader.load()
    
    print("Model builder loaded successfully!")
    print(f"Model type: {builder['model_type']}")
    
    # 测试构建（如果提供操作参数）
    if len(sys.argv) > 1:
        action = sys.argv[1]
        result = run_inference(builder, action, config)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
