# MCP工具封装模板

## 📋 目录结构

创建一个新的MCP工具需要以下文件：

```
tools_factory/incoming_raw/
└── your_tool_name/              # 工具文件夹名称（会变成server名称）
    ├── main.py                 # 工具入口文件（必需）
    └── tool.yaml               # 工具配置文件（可选）
```

---

## 🛠️ 模板1: 基础工具模板

### main.py 模板

```python
"""
Your Tool Name - MCP Tool Entry Point

工具描述：这里描述你的工具是做什么的
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================
# 1. 路径设置（必需）
# ============================================
# 当作为MCP工具运行时，路径结构：mcp_servers/your-tool-name/raw_tool/main.py
# 需要找到项目根目录：从 raw_tool 向上4级到项目根目录
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/your_tool_name/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

# 添加项目路径到sys.path（根据需要调整）
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))  # 如果有src目录
sys.path.insert(0, str(project_root / "core_projects" / "your_module"))  # 如果有其他模块


# ============================================
# 2. 导入依赖（根据你的工具调整）
# ============================================
try:
    import torch
    import numpy as np
    # 导入你的模块
    from your_module import YourClass
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")


# ============================================
# 3. Loader类（必需）
# ============================================
class YourToolLoader:
    """工具加载器类
    
    类名规则：
    - 必须以"Loader"结尾
    - 推荐格式：YourToolLoader 或 YourToolModelLoader
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器
        
        Args:
            config: 配置字典，包含工具所需的所有参数
                - param1: 参数1描述
                - param2: 参数2描述
        """
        self.config = config
        self.model = None  # 或其他资源
        
    def load(self) -> Any:
        """
        加载模型/资源
        
        Returns:
            加载的模型或资源对象
        """
        if self.model is None:
            # 在这里实现加载逻辑
            # self.model = YourClass(self.config)
            pass
        return self.model
    
    def load_model(self) -> Any:
        """
        加载模型（可选，如果使用load_model而不是load）
        
        Returns:
            加载的模型对象
        """
        return self.load()  # 或者实现不同的逻辑


# ============================================
# 4. 推理函数（必需）
# ============================================
def run_inference(
    model: Any,
    input_data: str,  # 或Dict, List等
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行推理/处理
    
    Args:
        model: 加载的模型或资源
        input_data: 输入数据（路径、数据等）
        config: 配置字典
    
    Returns:
        结果字典，包含：
            - result: 主要结果
            - shape: 输出形状（如果适用）
            - device: 使用的设备（如果适用）
    """
    # 在这里实现推理逻辑
    result = {
        'result': None,  # 你的结果
        'shape': None,   # 输出形状
        'device': str(config.get('device', 'cpu'))
    }
    return result


# ============================================
# 5. 主函数（可选，用于直接测试）
# ============================================
def main():
    """主入口，用于直接运行测试"""
    config = {
        'param1': 'value1',
        'param2': 'value2',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    loader = YourToolLoader(config)
    model = loader.load()
    print("Model loaded successfully!")
    
    # 测试推理（如果提供输入）
    if len(sys.argv) > 1:
        input_data = sys.argv[1]
        result = run_inference(model, input_data, config)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

---

## 📝 模板2: tool.yaml 配置文件模板

### tool.yaml 模板

```yaml
# MCP工具配置文件

# Server名称（必需）
# 如果不提供，将使用文件夹名称（下划线替换为连字符）
server_name: your-tool-name

# 模型路径（可选）
# 如果工具需要预训练权重，可以在这里指定
model_path: null  # 或 "path/to/model.pth"

# 输入参数定义（必需）
inputs:
  param1: str      # 参数1：字符串类型
  param2: int      # 参数2：整数类型
  param3: bool     # 参数3：布尔类型
  param4: list[str] # 参数4：字符串列表
  param5: dict     # 参数5：字典类型
  # 可选参数用注释说明
  # optional_param: str  # 可选参数

# 输出结果定义（必需）
outputs:
  result: any      # 主要结果
  shape: list[int] # 输出形状
  device: str      # 使用的设备
  info: dict       # 其他信息
```

---

## 🎯 模板3: 完整示例（图像处理工具）

### main.py 完整示例

```python
"""
Image Processing Tool - MCP Tool Entry Point

功能：图像处理和增强
"""

import sys
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np

# 路径设置
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    project_root = _current_file.parent.parent.parent.parent
else:
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))


class ImageProcessorLoader:
    """图像处理工具加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = None
        
    def load(self) -> Any:
        """加载处理器"""
        if self.processor is None:
            # 初始化处理器
            self.processor = {
                'mode': self.config.get('mode', 'enhance'),
                'device': self.config.get('device', 'cpu')
            }
        return self.processor
    
    def get_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        return {
            'mode': self.config.get('mode'),
            'device': self.config.get('device')
        }


def run_inference(
    processor: Dict[str, Any],
    image_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """运行图像处理"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # 处理图像（示例）
    mode = processor.get('mode', 'enhance')
    if mode == 'enhance':
        # 增强处理
        processed = img_array  # 实际处理逻辑
    else:
        processed = img_array
    
    return {
        'result': processed.tolist(),
        'shape': list(processed.shape),
        'device': processor.get('device', 'cpu'),
        'mode': mode
    }


def main():
    """主入口"""
    config = {
        'mode': 'enhance',
        'device': 'cpu'
    }
    
    loader = ImageProcessorLoader(config)
    processor = loader.load()
    print("Processor loaded!")
    
    if len(sys.argv) > 1:
        result = run_inference(processor, sys.argv[1], config)
        print(f"Processed image shape: {result['shape']}")


if __name__ == "__main__":
    main()
```

### tool.yaml 完整示例

```yaml
server_name: image-processor

inputs:
  mode: str        # 处理模式：'enhance', 'filter', 'transform'
  image_path: str  # 输入图像路径
  device: str      # 设备：'cpu' 或 'cuda'

outputs:
  result: list[list[float]]  # 处理后的图像数据
  shape: list[int]           # 图像形状 [H, W, C]
  device: str                # 使用的设备
  mode: str                  # 使用的处理模式
```

---

## 🔧 模板4: 机器学习模型工具

### main.py ML模型模板

```python
"""
ML Model Tool - MCP Tool Entry Point

功能：加载和运行机器学习模型
"""

import sys
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn

# 路径设置
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    project_root = _current_file.parent.parent.parent.parent
else:
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class MLModelLoader:
    """机器学习模型加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        
    def load_model(self) -> nn.Module:
        """加载模型"""
        if self.model is None:
            # 创建模型
            model_arch = self.config.get('model_arch', 'resnet18')
            num_classes = self.config.get('num_classes', 10)
            
            # 根据架构创建模型
            if model_arch == 'resnet18':
                from torchvision.models import resnet18
                self.model = resnet18(num_classes=num_classes)
            # 添加其他架构...
            
            # 加载预训练权重（如果提供）
            checkpoint_path = self.config.get('checkpoint_path')
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            
            # 冻结参数（如果指定）
            if self.config.get('freeze_encoder', False):
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.model.to(self.device)
            self.model.eval()
            
        return self.model
    
    def get_trainable_params_info(self) -> Dict[str, Any]:
        """获取可训练参数信息"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }


def run_inference(
    model: nn.Module,
    input_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """运行推理"""
    from PIL import Image
    import torchvision.transforms as transforms
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(input_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return {
        'prediction': int(pred),
        'confidence': float(confidence),
        'probabilities': probs[0].cpu().numpy().tolist(),
        'device': str(device)
    }


def main():
    """主入口"""
    config = {
        'model_arch': 'resnet18',
        'num_classes': 10,
        'checkpoint_path': None,
        'freeze_encoder': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    loader = MLModelLoader(config)
    model = loader.load_model()
    
    param_info = loader.get_trainable_params_info()
    print(f"Model loaded: {param_info['total_params']:,} parameters")
    
    if len(sys.argv) > 1:
        result = run_inference(model, sys.argv[1], config)
        print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
```

---

## 📋 使用步骤

### 步骤1: 创建工具文件夹

```bash
cd tools_factory/incoming_raw
mkdir your_tool_name
cd your_tool_name
```

### 步骤2: 创建main.py

复制上面的模板，根据你的需求修改：
- 修改类名（必须以`Loader`结尾）
- 修改导入的模块
- 实现`load()`或`load_model()`方法
- 实现`run_inference()`函数

### 步骤3: 创建tool.yaml（可选）

定义输入输出参数：

```yaml
server_name: your-tool-name
inputs:
  param1: str
  param2: int
outputs:
  result: any
```

### 步骤4: 编译MCP Server

```bash
cd E:\FedSAM3-Cream
python -m tools_factory.builder
```

### 步骤5: 测试

```bash
cd mcp_servers/your-tool-name
python server.py
```

---

## ✅ 检查清单

创建工具前，确保：

- [ ] **类名正确**: 以`Loader`结尾（如`YourToolLoader`）
- [ ] **路径设置**: 正确设置`sys.path`
- [ ] **导入成功**: 所有依赖都可以导入
- [ ] **方法存在**: `load()`或`load_model()`方法存在
- [ ] **函数存在**: `run_inference()`函数存在
- [ ] **类型注解**: 函数有正确的类型注解
- [ ] **文档字符串**: 函数有文档字符串（用于MCP工具描述）

---

## 🎯 命名规范

### Loader类名

```python
# ✅ 正确
class YourToolLoader:
class ImageProcessorLoader:
class MLModelLoader:
class SAM3FederatedModelLoader:

# ❌ 错误
class YourTool:  # 没有Loader后缀
class loader:   # 小写开头
```

### 方法名

```python
# ✅ 正确（二选一）
def load(self) -> Any:
def load_model(self) -> Any:

# ✅ 可选：获取信息的方法
def get_info(self) -> Dict[str, Any]:
def get_trainable_params_info(self) -> Dict[str, Any]:
```

### 函数名

```python
# ✅ 必需：推理函数
def run_inference(model, input_data, config) -> Dict[str, Any]:
```

---

## 🔍 常见问题

### Q1: 类名找不到？

**A**: 确保类名以`Loader`结尾，且在`main.py`中定义。

### Q2: 导入失败？

**A**: 检查`sys.path`设置，确保模块路径正确。

### Q3: 方法不存在？

**A**: 确保实现了`load()`或`load_model()`方法。

### Q4: 装饰器错误？

**A**: 模板会自动处理，确保使用`@app.tool()`（在server.py中）。

---

## 📚 参考示例

查看现有工具作为参考：

- **基础工具**: `tools_factory/incoming_raw/sam3_medical_model/`
- **完整工具**: `tools_factory/incoming_raw/sam3_federated_model/`

---

**现在你可以使用这些模板创建自己的MCP工具了！** 🚀
