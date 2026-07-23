"""
SAM3 Medical Image Segmentation Model - MCP Tool Entry Point

This module provides a unified interface for:
1. Model architecture loading (ResNet from CreamFL)
2. Pretrained weight loading
3. Parameter freezing configuration
4. Medical image segmentation inference
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

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core_projects" / "CreamFL-main" / "src"))

try:
    from networks.resnet_fedml import ResNet, Bottleneck, resnet56_fedml, resnet11_fedml
    from networks.resnet import resnet18, resnet50
    from networks.resnet_client import resnet18_client, resnet_50
except ImportError as e:
    print(f"Warning: Could not import CreamFL networks: {e}")
    print("Please ensure core_projects/CreamFL-main/src is in PYTHONPATH")

# Import SAM3 model components
try:
    from src.model import SAM3_Medical, Adapter, MaskDecoder
except ImportError:
    # Fallback: define minimal components if SAM3 not available
    print("Warning: SAM3_Medical not found, using minimal implementation")


class SAM3ModelLoader:
    """Loader for SAM3 Medical Image Segmentation Model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SAM3 model loader.
        
        Args:
            config: Configuration dictionary containing:
                - model_arch: str, one of ['resnet11', 'resnet56', 'resnet18', 'resnet50']
                - pretrained_path: str, path to pretrained weights (optional)
                - freeze_encoder: bool, whether to freeze image encoder
                - freeze_layers: List[str], specific layers to freeze (optional)
                - img_size: int, input image size (default: 1024)
                - num_classes: int, number of segmentation classes (default: 1)
                - device: str, device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        
    def load_model_architecture(self) -> nn.Module:
        """Load model architecture based on config."""
        arch = self.config.get('model_arch', 'resnet56')
        num_classes = self.config.get('num_classes', 10)
        
        arch_map = {
            'resnet11': lambda: resnet11_fedml(class_num=num_classes, pretrained=False),
            'resnet56': lambda: resnet56_fedml(class_num=num_classes, pretrained=False),
            'resnet18': lambda: resnet18(pretrained=False, num_classes=num_classes),
            'resnet50': lambda: resnet50(pretrained=False, num_classes=num_classes),
            'resnet18_client': lambda: resnet18_client(pretrained=False, 
                                                       embed_dim=512, 
                                                       num_class=num_classes,
                                                       is_train=False,
                                                       scale=128,
                                                       phase='extract_conv_feature'),
        }
        
        if arch not in arch_map:
            raise ValueError(f"Unknown architecture: {arch}. Available: {list(arch_map.keys())}")
        
        model = arch_map[arch]()
        return model.to(self.device)
    
    def load_pretrained_weights(self, model: nn.Module, weight_path: str) -> nn.Module:
        """Load pretrained weights into model."""
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Pretrained weights not found: {weight_path}")
        
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        # Load weights
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from {weight_path}")
        except Exception as e:
            print(f"Warning: Partial weight loading: {e}")
            model.load_state_dict(new_state_dict, strict=False)
        
        return model
    
    def freeze_parameters(self, model: nn.Module) -> nn.Module:
        """Freeze model parameters based on config."""
        freeze_encoder = self.config.get('freeze_encoder', True)
        freeze_layers = self.config.get('freeze_layers', [])
        
        if freeze_encoder:
            # Freeze entire encoder
            for param in model.parameters():
                param.requires_grad = False
            print("Frozen: All encoder parameters")
        
        # Freeze specific layers
        if freeze_layers:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
                    print(f"Frozen: {name}")
        
        # Count frozen/trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        return model
    
    def build_sam3_model(self) -> nn.Module:
        """Build complete SAM3 model with encoder, adapter, and decoder."""
        img_size = self.config.get('img_size', 1024)
        num_classes = self.config.get('num_classes', 1)
        embed_dim = self.config.get('embed_dim', 768)
        
        # Load image encoder
        encoder = self.load_model_architecture()
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_path'):
            encoder = self.load_pretrained_weights(encoder, self.config['pretrained_path'])
        
        # Freeze encoder if specified
        if self.config.get('freeze_encoder', True):
            encoder = self.freeze_parameters(encoder)
        
        # Build SAM3 model (simplified version)
        # In full implementation, this would use SAM3_Medical class
        class SimpleSAM3(nn.Module):
            def __init__(self, encoder, num_classes=1):
                super().__init__()
                self.encoder = encoder
                # Simple decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, num_classes, 4, 2, 1),
                )
            
            def forward(self, x):
                # Extract features (assuming encoder returns features)
                if hasattr(self.encoder, 'extract_conv_feature'):
                    features = self.encoder.extract_conv_feature(x)
                else:
                    # Fallback: use forward pass
                    features = self.encoder.conv1(x)
                    features = self.encoder.bn1(features)
                    features = self.encoder.relu(features)
                    features = self.encoder.maxpool(features)
                    features = self.encoder.layer1(features)
                    features = self.encoder.layer2(features)
                    features = self.encoder.layer3(features)
                    features = self.encoder.layer4(features)
                
                # Decode to segmentation mask
                mask = self.decoder(features)
                return mask
        
        model = SimpleSAM3(encoder, num_classes=num_classes)
        return model.to(self.device)
    
    def load(self) -> nn.Module:
        """Load and configure model."""
        self.model = self.build_sam3_model()
        self.model.eval()
        return self.model


def run_inference(
    model: nn.Module,
    image_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run inference on a medical image.
    
    Args:
        model: Loaded SAM3 model
        image_path: Path to input image
        config: Configuration dictionary
        
    Returns:
        Dictionary containing:
            - mask: Segmentation mask (numpy array)
            - shape: Output shape
            - device: Device used
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    img_size = config.get('img_size', 1024)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # Assuming single channel output
    
    return {
        'mask': mask.tolist(),
        'shape': list(mask.shape),
        'device': str(device)
    }


def main():
    """Main entry point for MCP tool."""
    import json
    
    # Example configuration
    config = {
        'model_arch': 'resnet56',
        'pretrained_path': None,  # Set to path if available
        'freeze_encoder': True,
        'freeze_layers': ['layer1', 'layer2'],
        'img_size': 1024,
        'num_classes': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Load model
    loader = SAM3ModelLoader(config)
    model = loader.load()
    
    print("Model loaded successfully!")
    print(f"Model architecture: {config['model_arch']}")
    print(f"Frozen encoder: {config['freeze_encoder']}")
    
    # Example: run inference (if image path provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = run_inference(model, image_path, config)
        print(f"Inference result shape: {result['shape']}")


if __name__ == "__main__":
    main()
