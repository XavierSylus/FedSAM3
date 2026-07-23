"""
MCP工具main.py模板

使用方法：
1. 复制此文件到 tools_factory/incoming_raw/your_tool_name/main.py
2. 根据你的需求修改以下部分：
   - 工具描述
   - 导入的模块
   - Loader类名和实现
   - run_inference函数实现
3. 运行 python -m tools_factory.builder 编译
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================
# 1. 路径设置（必需 - 不要修改这部分逻辑）
# ============================================
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/your_tool_name/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

# 添加项目路径（根据你的项目结构调整）
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))  # 如果有src目录
# sys.path.insert(0, str(project_root / "core_projects" / "your_module"))  # 取消注释并修改


# ============================================
# 2. 导入依赖（根据你的工具修改）
# ============================================
try:
    # import torch
    # import numpy as np
    # from your_module import YourClass
    pass
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")


# ============================================
# 3. Loader类（必需 - 修改类名和实现）
# ============================================
class YourToolLoader:  # TODO: 修改类名（必须以Loader结尾）
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
            # TODO: 在这里实现加载逻辑
            # self.model = YourClass(self.config)
            self.model = {"status": "loaded", "config": self.config}
        return self.model
    
    def load_model(self) -> Any:
        """
        加载模型（可选，如果使用load_model而不是load）
        
        注意：server.py会优先尝试load_model()，如果没有则尝试load()
        
        Returns:
            加载的模型对象
        """
        return self.load()  # 或者实现不同的逻辑
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取工具信息（可选）
        
        Returns:
            工具信息字典
        """
        return {
            "config": self.config,
            "has_model": self.model is not None
        }


# ============================================
# 4. 推理函数（必需 - 修改实现）
# ============================================
def run_inference(
    model: Any,
    input_data: str,  # TODO: 根据你的输入类型修改（str, Dict, List等）
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行推理/处理
    
    Args:
        model: 加载的模型或资源
        input_data: 输入数据（路径、数据等）
        config: 配置字典
    
    Returns:
        结果字典，必须包含：
            - 至少一个结果字段
            - 可选：shape, device, info等
    """
    # TODO: 在这里实现推理逻辑
    result = {
        'result': None,  # 你的主要结果
        'status': 'ok',
        # 'shape': None,   # 输出形状（如果适用）
        # 'device': str(config.get('device', 'cpu'))  # 使用的设备（如果适用）
    }
    return result


# ============================================
# 5. 主函数（可选，用于直接测试）
# ============================================
def main():
    """主入口，用于直接运行测试"""
    # TODO: 根据你的工具修改配置
    config = {
        'param1': 'value1',
        'param2': 'value2',
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    loader = YourToolLoader(config)  # TODO: 修改为你的Loader类名
    model = loader.load()
    print("Model loaded successfully!")
    
    # 测试推理（如果提供输入）
    if len(sys.argv) > 1:
        input_data = sys.argv[1]
        result = run_inference(model, input_data, config)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
