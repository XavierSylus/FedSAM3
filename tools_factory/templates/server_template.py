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


app = FastMCP("{{server_name}}")


# Try to import tool functions from raw_tool/main.py
_MODEL_LOADER = None
_MODEL = None
_MODEL_LOADER_CLASS = None

try:
    # Try to import different possible loader classes
    from raw_tool.main import run_inference
    
    # Try multiple possible loader class names
    loader_classes = [
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
    """Execute the SAM3 federated medical image segmentation tool.

    This function supports:
    1. Model Architecture: Load SAM3 model with adapter
    2. Pretrained Weights: Load SAM3 checkpoint
    3. Parameter Freezing: Freeze encoder layers for fine-tuning
    4. Inference: Run segmentation on medical images

    Args:
        inputs: Dictionary containing:
            - use_sam3: bool, whether to use real SAM3 model
            - freeze_encoder: bool, whether to freeze encoder (default: true)
            - use_adapter: bool, whether to use adapter modules (default: true)
            - sam3_checkpoint: str, optional path to SAM3 pretrained weights
            - img_size: int, input image size (default: 1024)
            - num_classes: int, number of classes (default: 1)
            - adapter_dim: int, adapter dimension (default: 64)
            - image_path: str, path to input image for inference
            - device: str, device to use ('cuda' or 'cpu')

    Returns:
        Dictionary containing:
            - status: str, execution status
            - mask: list[list[float]], segmentation mask (if image_path provided)
            - shape: list[int], output shape
            - model_info: dict, model configuration info
            - trainable_params_info: dict, parameter information
    """
    if not _HAS_TOOL:
        # Fallback: echo mode
        return {
            "status": "ok",
            "echo_inputs": inputs,
            "message": "Tool functions not available, running in echo mode"
        }
    
    try:
        # Extract configuration (support both tool formats)
        config = {}
        
        # For sam3-federated-medical-segmentation format
        if 'use_sam3' in inputs:
            config = {
                'use_sam3': inputs.get('use_sam3', True),
                'freeze_encoder': inputs.get('freeze_encoder', True),
                'use_adapter': inputs.get('use_adapter', True),
                'sam3_checkpoint': inputs.get('sam3_checkpoint'),
                'img_size': inputs.get('img_size', 1024),
                'num_classes': inputs.get('num_classes', 1),
                'adapter_dim': inputs.get('adapter_dim', 64),
                'device': inputs.get('device', 'cuda' if sys.platform != 'darwin' else 'cpu')
            }
        # For sam3-medical-segmentation format
        else:
            config = {
                'model_arch': inputs.get('model_arch', 'resnet56'),
                'pretrained_path': inputs.get('pretrained_path'),
                'freeze_encoder': inputs.get('freeze_encoder', True),
                'freeze_layers': inputs.get('freeze_layers', []),
                'img_size': inputs.get('img_size', 1024),
                'num_classes': inputs.get('num_classes', 1),
                'device': inputs.get('device', 'cuda' if sys.platform != 'darwin' else 'cpu')
            }
        
        # Load model
        model = _load_model(config)
        
        # Prepare result
        result = {
            "status": "ok",
            "model_info": {
                "use_sam3": config['use_sam3'],
                "freeze_encoder": config['freeze_encoder'],
                "use_adapter": config['use_adapter'],
                "sam3_checkpoint": config['sam3_checkpoint'],
                "img_size": config['img_size'],
                "num_classes": config['num_classes'],
                "adapter_dim": config['adapter_dim']
            }
        }
        
        # Get trainable parameters info
        if _MODEL_LOADER is not None:
            if hasattr(_MODEL_LOADER, 'get_trainable_params_info'):
                param_info = _MODEL_LOADER.get_trainable_params_info()
                result["trainable_params_info"] = param_info
            elif hasattr(_MODEL_LOADER, 'config'):
                result["model_info"] = _MODEL_LOADER.config
        
        # Run inference if image_path provided
        if 'image_path' in inputs and inputs['image_path']:
            inference_result = run_inference(model, inputs['image_path'], config)
            result.update({
                "mask": inference_result.get('mask'),
                "shape": inference_result.get('shape'),
                "device": inference_result.get('device')
            })
        else:
            result["message"] = "Model loaded successfully. Provide 'image_path' for inference."
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model.
    
    Returns:
        Dictionary containing model architecture and configuration.
    """
    if _MODEL_LOADER is None:
        return {
            "status": "not_loaded",
            "message": "Model not loaded yet. Call run_tool first."
        }
    
    param_info = _MODEL_LOADER.get_trainable_params_info() if _MODEL_LOADER else {}
    
    return {
        "status": "loaded",
        "config": _MODEL_LOADER.config if hasattr(_MODEL_LOADER, 'config') else {},
        "has_model": _MODEL is not None,
        "trainable_params_info": param_info
    }


if __name__ == "__main__":
    # Entrypoint for running this template as a standalone MCP server.
    app.run()


