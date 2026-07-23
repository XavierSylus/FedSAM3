# MCP Server: sam3-model-builder

这是由 Tools Factory 自动从 `sam3_model_builder` 生成的 MCP server。

## 功能说明

这个工具封装了 `sam3.model_builder` 模块的功能，提供以下操作：

1. **构建SAM3图像模型** (`build_image`) - 构建用于图像分割的SAM3模型
2. **构建SAM3视频模型** (`build_video`) - 构建用于视频跟踪的SAM3模型
3. **构建跟踪器** (`build_tracker`) - 构建SAM3跟踪器模块
4. **下载检查点** (`download_ckpt`) - 从HuggingFace下载SAM3预训练权重

## 目录结构

- `server.py`        : FastMCP 入口，`python server.py` 即可启动 MCP server
- `raw_tool/`        : 原始的工具代码（包括 `main.py`、`tool.yaml` 等）

## 运行方式

```bash
cd mcp_servers/sam3-model-builder
python server.py
```

然后在 MCP 客户端中把这个进程配置为本地 MCP server 即可使用。

## 使用示例

### 1. 构建SAM3图像模型

```python
inputs = {
    "action": "build_image",
    "model_type": "image",
    "device": "cuda",
    "eval_mode": True,
    "load_from_HF": True,
    "enable_segmentation": True,
    "enable_inst_interactivity": False
}
result = run_tool(inputs)
```

### 2. 构建SAM3视频模型

```python
inputs = {
    "action": "build_video",
    "model_type": "video",
    "device": "cuda",
    "apply_temporal_disambiguation": True,
    "load_from_HF": True
}
result = run_tool(inputs)
```

### 3. 构建跟踪器

```python
inputs = {
    "action": "build_tracker",
    "model_type": "tracker",
    "apply_temporal_disambiguation": True,
    "with_backbone": False
}
result = run_tool(inputs)
```

### 4. 下载检查点

```python
inputs = {
    "action": "download_ckpt"
}
result = run_tool(inputs)
# result["checkpoint_path"] 包含下载的检查点路径
```

## 输入参数

- `action` (str, 必需): 操作类型 - 'build_image', 'build_video', 'build_tracker', 'download_ckpt'
- `model_type` (str): 模型类型 - 'image', 'video', 'tracker'
- `device` (str): 设备 - 'cuda' 或 'cpu'
- `checkpoint_path` (str, 可选): 检查点路径
- `load_from_HF` (bool): 是否从HuggingFace加载（默认: true）
- 其他参数请参考 `tool.yaml` 文件

## 输出格式

```python
{
    "status": "success" | "error",
    "model_type": "image" | "video" | "tracker",
    "model_info": {...},
    "device": "cuda" | "cpu",
    "checkpoint_path": "...",  # 当action为download_ckpt时
    "message": "...",  # 可选
    "error": "..."  # 当status为error时
}
```
