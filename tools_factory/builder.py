"""Tools Factory builder for compiling raw tools into MCP servers.

This module is inspired by the SciToolAgent architecture, where arbitrary
scientific tools (scripts, models) are:

1. Discovered in a ``incoming_raw/`` drop zone.
2. Analyzed (typically via an LLM) to infer inputs, outputs, and semantics.
3. Compiled into standardized MCP servers using code templates.
"""

# 联邦客户端训练器 MCP 服务器编译器

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import shutil
import textwrap
import yaml


class ToolBuilder:
    """Compiler-like builder that generates MCP servers for raw tools.

    Attributes:
        root_dir: Project root directory containing the ``tools_factory`` tree.
    """

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize the ToolBuilder.

        Args:
            root_dir: Path to the MatterSwarm repository root.
        """
        self.root_dir = Path(root_dir).resolve()
        self.incoming_dir = self.root_dir / "tools_factory" / "incoming_raw"
        self.templates_dir = self.root_dir / "tools_factory" / "templates"
        self.mcp_output_dir = self.root_dir / "mcp_servers"

    # ------------------------------------------------------------------
    # High-level public API
    # ------------------------------------------------------------------

    def build_all(self) -> List[Path]:
        """Compile all raw tools under ``incoming_raw/`` into MCP servers.

        Workflow（对应你描述的使用方式）:

        1. 准备素材：每个原始工具放在一个子文件夹里，例如

           ``tools_factory/incoming_raw/my_precursor_model/``

           里面至少包含：

           - ``main.py``: 你原来用来跑模型/推理的入口脚本
           - 可选的权重文件（例如 ``model.pth`` 或其它数据文件）
           - 可选的 ``tool.yaml``: 用于覆盖 server_name / model_path 等元数据

        2. 丢进去：把整个文件夹拖到 ``incoming_raw/``。
        3. 运行编译器：在项目根目录下运行 ``python -m tools_factory.builder``。
        4. 结果：在 ``mcp_servers/`` 下面得到一个可以直接 ``python server.py`` 运行的 MCP server。

        Returns:
            A list of paths to generated MCP server directories.
        """
        tools = self.scan_incoming()
        generated: List[Path] = []
        for t in tools:
            server_dir = self.generate_mcp_server(t)
            generated.append(server_dir)
        return generated

    def scan_incoming(self) -> List[Path]:
        """Scan for new raw tool folders in ``incoming_raw``.

        In the full SciToolAgent design, each folder may contain:
        - Python scripts
        - Model weights (e.g., ``.pt``)
        - A small YAML/JSON metadata file

        Returns:
            A list of paths to candidate tool directories.
        """
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        candidates: List[Path] = []
        for child in self.incoming_dir.iterdir():
            if child.is_dir():
                candidates.append(child)
        return candidates

    def analyze_tool_metadata(self, tool_dir: Path) -> Dict[str, Any]:
        """Analyze a raw tool folder and infer its interface.

        This is a placeholder for an LLM-powered analysis that would:
        - Read source code/docstrings.
        - Identify function signatures (inputs/outputs).
        - Infer semantic tags for the knowledge graph (e.g., domain, modality).

        Args:
            tool_dir: Path to the raw tool directory.

        Returns:
            A dictionary with extracted metadata suitable for templating.
        """
        # TODO: Integrate with an LLM and static analysis pipeline.
        metadata_file = tool_dir / "tool.yaml"
        if metadata_file.exists():
            base = yaml.safe_load(metadata_file.read_text(encoding="utf-8")) or {}
        else:
            base = {}

        # 一些通用的默认值，YAML 里没写就自动补上
        server_name = str(base.get("server_name") or tool_dir.name.replace("_", "-"))
        # 默认为当前目录下的第一个 .pth /.pt 文件；否则 fallback 到 dummy_model.pt
        model_path = base.get("model_path")
        if model_path is None:
            weight = next(
                (
                    p.name
                    for p in tool_dir.iterdir()
                    if p.suffix in {".pt", ".pth"} and p.is_file()
                ),
                "dummy_model.pt",
            )
            model_path = weight

        inputs = base.get("inputs") or {"dummy_input": "str"}
        outputs = base.get("outputs") or {"dummy_output": "str"}

        return {
            "server_name": server_name,
            "model_path": model_path,
            "inputs": inputs,
            "outputs": outputs,
        }

    def generate_mcp_server(self, tool_dir: Path) -> Path:
        """Generate an MCP server from a raw tool directory.

        This method takes the raw code and wraps it into a FastMCP-based
        server using the ``templates/server_template.py`` file. In a complete
        implementation, it would:

        - Copy or vendor the raw code into a new server package.
        - Render the server template with tool-specific metadata.
        - Optionally emit a Dockerfile from ``Dockerfile.template``.

        Args:
            tool_dir: Path to the raw tool folder.

        Returns:
            Path to the generated MCP server module directory.
        """
        metadata = self.analyze_tool_metadata(tool_dir)
        server_name = metadata.get("server_name", tool_dir.name)

        # 1. 创建目标目录，并把原始工具整体拷贝进去（方便在 server.py 里 import main.py）
        server_dir = self.mcp_output_dir / server_name
        if server_dir.exists():
            # 简单容错：如果已经存在老版本，先删掉再重新生成
            shutil.rmtree(server_dir)
        server_dir.mkdir(parents=True, exist_ok=True)

        # 把原始工具代码 vendor 进去，放在 server 目录下的 raw_tool/ 子目录
        raw_dst = server_dir / "raw_tool"
        shutil.copytree(tool_dir, raw_dst)

        # 2. 渲染 server_template.py -> server.py
        template_path = self.templates_dir / "server_template.py"
        template_text = template_path.read_text(encoding="utf-8")

        rendered = (
            template_text.replace("{{server_name}}", server_name)
            .replace("{{model_path}}", str(metadata.get("model_path", "dummy_model.pt")))
        )

        server_file = server_dir / "server.py"
        server_file.write_text(rendered, encoding="utf-8")

        # 3. 生成一个最小的 README，告诉用户怎么跑这个 server
        readme_text = textwrap.dedent(
            f"""
            # MCP Server: {server_name}

            这是由 Tools Factory 自动从 `{tool_dir.name}` 生成的 MCP server。

            ## 目录结构

            - `server.py`        : FastMCP 入口，`python server.py` 即可启动 MCP server
            - `raw_tool/`        : 你原始的工具代码（包括 `main.py`、权重文件等）

            ## 运行方式

            ```bash
            cd mcp_servers/{server_name}
            python server.py
            ```

            然后在 MCP 客户端中把这个进程配置为本地 MCP server 即可使用。
            """
        ).strip() + "\n"
        (server_dir / "README.md").write_text(readme_text, encoding="utf-8")

        # 4. 可选：将来可以在这里根据 Dockerfile.template 渲染 Dockerfile
        #    目前先保留为后续拓展点。

        return server_dir


__all__ = ["ToolBuilder"]


def main() -> None:
    """CLI 入口：允许你直接 `python -m tools_factory.builder` 触发构建。

    对应你的自然工作流：“双击 builder.py” / “python -m tools_factory.builder”。
    """
    project_root = Path(__file__).resolve().parents[1]
    builder = ToolBuilder(project_root)

    servers = builder.build_all()
    if not servers:
        print("No tools found in tools_factory/incoming_raw. 请先把你的工具文件夹拖进去。")
        return

    print("生成完成，以下 MCP servers 已创建：")
    for s in servers:
        print(f" - {s}")


if __name__ == "__main__":
    main()
