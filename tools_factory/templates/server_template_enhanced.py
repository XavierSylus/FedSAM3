"""Enhanced MCP server template with automatic tool integration.

This template automatically imports and calls functions from raw_tool/main.py,
supporting:
1. Model architecture loading
2. Pretrained weight loading  
3. Parameter freezing
4. Inference execution
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from mcp import FastMCP, tool  # type: ignore[import]


# Add raw_tool to path for imports
_raw_tool_path = Path(__file__).parent / "raw_tool"
if _raw_tool_path.exists():
    sys.path.insert(0, str(_raw_tool_path.parent))
    sys.path.insert(0, str(_raw_tool_path))


app = FastMCP("{{server_name}}")


# Try to import tool functions from raw_tool/main.py
_MODEL_LOADER = None
_MODEL = None

try:
    from raw_tool.main import SAM3ModelLoader, run_inference
    
    def _load_model(config: Dict[str, Any]) -> Any:
        """Load model using SAM3ModelLoader."""
        global _MODEL_LOADER, _MODEL
        if _MODEL is None:
            _MODEL_LOADER = SAM3ModelLoader(config)
            _MODEL = _MODEL_LOADER.load()
        return _MODEL
    
    _HAS_TOOL = True
except ImportError as e:
    print(f"Warning: Could not import tool functions: {e}")
    print("Falling back to echo mode.")
    _HAS_TOOL = False
    
    def _load_model(config: Dict[str, Any]) -> Any:
        """Fallback: return mock model."""
        return {"model_path": config.get("pretrained_path", "dummy_model.pt")}


@tool
def run_tool(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the SAM3 medical image segmentation tool.

    This function supports three key operations:
    1. Model Architecture: Load ResNet-based encoder from CreamFL
    2. Pretrained Weights: Load weights from checkpoint file
    3. Parameter Freezing: Freeze encoder layers for fine-tuning

    Args:
        inputs: Dictionary containing:
            - model_arch: str, architecture name (resnet11/resnet56/resnet18/resnet50)
            - pretrained_path: str, optional path to pretrained weights
            - freeze_encoder: bool, whether to freeze encoder (default: true)
            - freeze_layers: list[str], specific layers to freeze
            - img_size: int, input image size (default: 1024)
            - num_classes: int, number of classes (default: 1)
            - image_path: str, path to input image for inference
            - device: str, device to use ('cuda' or 'cpu')

    Returns:
        Dictionary containing:
            - status: str, execution status
            - mask: list[list[float]], segmentation mask (if image_path provided)
            - shape: list[int], output shape
            - model_info: dict, model configuration info
    """
    if not _HAS_TOOL:
        # Fallback: echo mode
        return {
            "status": "ok",
            "echo_inputs": inputs,
            "message": "Tool functions not available, running in echo mode"
        }
    
    try:
        # Extract configuration
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
                "architecture": config['model_arch'],
                "pretrained_path": config['pretrained_path'],
                "freeze_encoder": config['freeze_encoder'],
                "freeze_layers": config['freeze_layers'],
                "img_size": config['img_size'],
                "num_classes": config['num_classes']
            }
        }
        
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


@tool
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
    
    return {
        "status": "loaded",
        "config": _MODEL_LOADER.config if hasattr(_MODEL_LOADER, 'config') else {},
        "has_model": _MODEL is not None
    }


if __name__ == "__main__":
    # Entrypoint for running this template as a standalone MCP server.
    app.run()
