"""Template MCP server used by the Tools Factory.

In the SciToolAgent-inspired architecture, each raw tool is compiled into a
standalone MCP server. This file acts as a base template that the
``ToolBuilder`` fills with tool-specific metadata (e.g., inputs, outputs,
model paths).

Enhanced version: Automatically imports and calls functions from raw_tool/main.py
"""

# 联邦客户端训练器 MCP 服务器模板（增强版：自动调用raw_tool/main.py）

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP


# Add raw_tool to path for imports
_raw_tool_path = Path(__file__).parent / "raw_tool"
if _raw_tool_path.exists():
    sys.path.insert(0, str(_raw_tool_path.parent))
    sys.path.insert(0, str(_raw_tool_path))


app = FastMCP("sam3-model-builder")


# Try to import tool functions from raw_tool/main.py
_MODEL_LOADER = None
_MODEL = None
_MODEL_LOADER_CLASS = None

try:
    # Try to import different possible loader classes
    from raw_tool.main import run_inference
    
    # Try multiple possible loader class names
    loader_classes = [
        "SAM3ModelBuilderLoader",
        "SAM3FederatedModelLoader",
        "SAM3ModelLoader", 
        "ModelLoader",
        "Loader"
    ]
    
    _MODEL_LOADER_CLASS = None
    for loader_name in loader_classes:
        try:
            loader_module = __import__("raw_tool.main", fromlist=[loader_name])
            if hasattr(loader_module, loader_name):
                _MODEL_LOADER_CLASS = getattr(loader_module, loader_name)
                print(f"Found loader class: {loader_name}")
                break
        except (ImportError, AttributeError):
            continue
    
    if _MODEL_LOADER_CLASS is None:
        raise ImportError("Could not find any loader class in raw_tool.main")
    
    def _load_model(config: Dict[str, Any]) -> Any:
        """Load model using detected loader class."""
        global _MODEL_LOADER, _MODEL
        if _MODEL is None:
            _MODEL_LOADER = _MODEL_LOADER_CLASS(config)
            # Try different load method names
            if hasattr(_MODEL_LOADER, 'load_model'):
                _MODEL = _MODEL_LOADER.load_model()
            elif hasattr(_MODEL_LOADER, 'load'):
                _MODEL = _MODEL_LOADER.load()
            else:
                _MODEL = _MODEL_LOADER
        return _MODEL
    
    _HAS_TOOL = True
except ImportError as e:
    print(f"Warning: Could not import tool functions: {e}")
    print("Falling back to echo mode.")
    _HAS_TOOL = False
    
    def _load_model(config: Dict[str, Any]) -> Any:
        """Fallback: return mock model."""
        return {"model_path": config.get("sam3_checkpoint", config.get("pretrained_path", "dummy_model.pt"))}


@app.tool()
def run_tool(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the SAM3 Model Builder tool.

    This function supports:
    1. Building SAM3 image models (build_image)
    2. Building SAM3 video models (build_video)
    3. Building SAM3 tracker models (build_tracker)
    4. Downloading checkpoints from HuggingFace (download_ckpt)

    Args:
        inputs: Dictionary containing:
            - action: str, action to perform ('build_image', 'build_video', 'build_tracker', 'download_ckpt')
            - model_type: str, model type ('image', 'video', 'tracker') - used when action is build_*
            - device: str, device to use ('cuda' or 'cpu')
            - bpe_path: str, optional path to BPE tokenizer
            - eval_mode: bool, whether to set model to eval mode (default: true)
            - checkpoint_path: str, optional path to checkpoint
            - load_from_HF: bool, whether to load from HuggingFace (default: true)
            - enable_segmentation: bool, whether to enable segmentation head (default: true)
            - enable_inst_interactivity: bool, whether to enable instance interactivity (default: false)
            - compile: bool, whether to compile model (default: false)
            - has_presence_token: bool, whether model has presence token (default: true)
            - geo_encoder_use_img_cross_attn: bool, whether geo encoder uses image cross attention (default: true)
            - strict_state_dict_loading: bool, whether to strictly load state dict (default: true)
            - apply_temporal_disambiguation: bool, whether to apply temporal disambiguation (default: true)
            - with_backbone: bool, whether tracker includes backbone (default: false)

    Returns:
        Dictionary containing:
            - status: str, execution status ('success' or 'error')
            - model_type: str, model type (if model was built)
            - model_info: dict, model information
            - device: str, device used
            - checkpoint_path: str, downloaded checkpoint path (if action is download_ckpt)
            - message: str, optional message
            - error: str, error message (if status is error)
    """
    if not _HAS_TOOL:
        # Fallback: echo mode
        return {
            "status": "ok",
            "echo_inputs": inputs,
            "message": "Tool functions not available, running in echo mode"
        }
    
    try:
        # Extract action
        action = inputs.get('action', 'build_image')
        
        # Extract configuration
        config = {
            'model_type': inputs.get('model_type', 'image'),
            'device': inputs.get('device', 'cuda' if sys.platform != 'darwin' else 'cpu'),
            'bpe_path': inputs.get('bpe_path'),
            'eval_mode': inputs.get('eval_mode', True),
            'checkpoint_path': inputs.get('checkpoint_path'),
            'load_from_HF': inputs.get('load_from_HF', True),
            'enable_segmentation': inputs.get('enable_segmentation', True),
            'enable_inst_interactivity': inputs.get('enable_inst_interactivity', False),
            'compile': inputs.get('compile', False),
            'has_presence_token': inputs.get('has_presence_token', True),
            'geo_encoder_use_img_cross_attn': inputs.get('geo_encoder_use_img_cross_attn', True),
            'strict_state_dict_loading': inputs.get('strict_state_dict_loading', True),
            'apply_temporal_disambiguation': inputs.get('apply_temporal_disambiguation', True),
            'with_backbone': inputs.get('with_backbone', False),
        }
        
        # Load builder
        builder = _load_model(config)
        
        # Run inference (which handles building models or downloading checkpoints)
        result = run_inference(builder, action, config)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
def get_model_info() -> Dict[str, Any]:
    """Get information about the model builder.
    
    Returns:
        Dictionary containing model builder configuration and information.
    """
    if _MODEL_LOADER is None:
        return {
            "status": "not_loaded",
            "message": "Model builder not loaded yet. Call run_tool first."
        }
    
    info = {}
    if hasattr(_MODEL_LOADER, 'get_model_info'):
        info = _MODEL_LOADER.get_model_info()
    elif hasattr(_MODEL_LOADER, 'config'):
        info = {
            "config": _MODEL_LOADER.config,
            "model_type": _MODEL_LOADER.model_type if hasattr(_MODEL_LOADER, 'model_type') else 'unknown'
        }
    
    return {
        "status": "loaded",
        "config": _MODEL_LOADER.config if hasattr(_MODEL_LOADER, 'config') else {},
        "has_model": _MODEL is not None,
        "model_info": info
    }


if __name__ == "__main__":
    # Entrypoint for running this template as a standalone MCP server.
    app.run()


