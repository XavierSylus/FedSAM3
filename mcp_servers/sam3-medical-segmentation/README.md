# MCP Server: sam3-medical-segmentation

这是由 Tools Factory 自动从 `sam3_medical_model` 生成的 MCP server。

## 目录结构

- `server.py`        : FastMCP 入口，`python server.py` 即可启动 MCP server
- `raw_tool/`        : 你原始的工具代码（包括 `main.py`、权重文件等）

## 运行方式

```bash
cd mcp_servers/sam3-medical-segmentation
python server.py
```

然后在 MCP 客户端中把这个进程配置为本地 MCP server 即可使用。
