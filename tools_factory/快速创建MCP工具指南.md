# 快速创建MCP工具指南

## 🚀 5分钟快速开始

### 步骤1: 创建工具文件夹

```bash
cd tools_factory/incoming_raw
mkdir my_first_tool
cd my_first_tool
```

### 步骤2: 复制模板文件

```bash
# 复制main.py模板
cp ../templates/tool_template_main.py main.py

# 复制tool.yaml模板（可选）
cp ../templates/tool_template.yaml tool.yaml
```

### 步骤3: 修改模板

编辑 `main.py`，修改以下部分：

1. **修改类名**（第47行）:
   ```python
   class MyFirstToolLoader:  # 修改这里
   ```

2. **修改导入**（第30-35行）:
   ```python
   try:
       import torch  # 添加你的依赖
       import numpy as np
   except ImportError as e:
       ...
   ```

3. **实现load方法**（第60-70行）:
   ```python
   def load(self) -> Any:
       if self.model is None:
           # 实现你的加载逻辑
           self.model = YourModel(self.config)
       return self.model
   ```

4. **实现run_inference函数**（第95-110行）:
   ```python
   def run_inference(model, input_data, config):
       # 实现你的推理逻辑
       return {'result': your_result}
   ```

### 步骤4: 修改tool.yaml（可选）

编辑 `tool.yaml`，定义输入输出：

```yaml
server_name: my-first-tool
inputs:
  input_path: str
  param1: int
outputs:
  result: any
  status: str
```

### 步骤5: 编译

```bash
cd E:\FedSAM3-Cream
python -m tools_factory.builder
```

### 步骤6: 测试

```bash
cd mcp_servers/my-first-tool
python server.py
```

---

## 📝 完整示例：创建一个简单的文本处理工具

### 1. 创建文件夹和文件

```bash
cd tools_factory/incoming_raw
mkdir text_processor
cd text_processor
```

### 2. 创建main.py

```python
"""
Text Processor Tool - MCP Tool Entry Point
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 路径设置
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    project_root = _current_file.parent.parent.parent.parent
else:
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))


class TextProcessorLoader:
    """文本处理工具加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = None
        
    def load(self) -> Dict[str, Any]:
        """加载处理器"""
        if self.processor is None:
            self.processor = {
                'mode': self.config.get('mode', 'uppercase'),
                'language': self.config.get('language', 'en')
            }
        return self.processor


def run_inference(
    processor: Dict[str, Any],
    text: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """处理文本"""
    mode = processor.get('mode', 'uppercase')
    
    if mode == 'uppercase':
        result = text.upper()
    elif mode == 'lowercase':
        result = text.lower()
    elif mode == 'reverse':
        result = text[::-1]
    else:
        result = text
    
    return {
        'result': result,
        'original_length': len(text),
        'processed_length': len(result),
        'mode': mode
    }


def main():
    """主入口"""
    config = {
        'mode': 'uppercase',
        'language': 'en'
    }
    
    loader = TextProcessorLoader(config)
    processor = loader.load()
    print("Processor loaded!")
    
    if len(sys.argv) > 1:
        result = run_inference(processor, sys.argv[1], config)
        print(f"Result: {result['result']}")


if __name__ == "__main__":
    main()
```

### 3. 创建tool.yaml

```yaml
server_name: text-processor

inputs:
  text: str        # 输入文本
  mode: str        # 处理模式：'uppercase', 'lowercase', 'reverse'
  language: str    # 语言（可选）

outputs:
  result: str              # 处理后的文本
  original_length: int     # 原始长度
  processed_length: int    # 处理后长度
  mode: str                # 使用的模式
```

### 4. 编译和测试

```bash
# 编译
cd E:\FedSAM3-Cream
python -m tools_factory.builder

# 测试
cd mcp_servers/text-processor
python server.py
```

---

## ✅ 检查清单

创建工具后，检查：

- [ ] **类名正确**: 以`Loader`结尾
- [ ] **路径设置**: `sys.path`正确设置
- [ ] **导入成功**: 所有依赖可以导入
- [ ] **方法存在**: `load()`或`load_model()`存在
- [ ] **函数存在**: `run_inference()`存在
- [ ] **类型注解**: 函数有类型注解
- [ ] **文档字符串**: 函数有文档字符串
- [ ] **编译成功**: `python -m tools_factory.builder`无错误
- [ ] **Server启动**: `python server.py`可以启动

---

## 🎯 模板文件位置

- **main.py模板**: `tools_factory/templates/tool_template_main.py`
- **tool.yaml模板**: `tools_factory/templates/tool_template.yaml`
- **完整文档**: `tools_factory/MCP工具封装模板.md`

---

## 📚 参考

- **基础工具示例**: `tools_factory/incoming_raw/sam3_medical_model/`
- **完整工具示例**: `tools_factory/incoming_raw/sam3_federated_model/`
- **模板文件**: `tools_factory/templates/`

---

**现在开始创建你的第一个MCP工具吧！** 🎉
