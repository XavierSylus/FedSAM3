## Tools Factory 使用说明（手把手版）

这个目录负责把你现有的科研代码（例如：前驱体预测、性质预测模型等）**一键编译成 MCP Server**，方便在 MatterSwarm 里统一调用。

> 你可以把它理解成一个“小型编译器”：
> - 输入：你的 `main.py` + 权重 `.pth` + 可选的 `tool.yaml` 配置。
> - 输出：`mcp_servers/xxx/` 下一个可以直接 `python server.py` 跑起来的 MCP server。

---

## 1. 准备一个原始工具（以“前驱体预测模型”为例）

假设你已经有一个可以在命令行运行的前驱体预测工程，它的大致结构是：

```text
my_precursor_tool/
  main.py          # 你平时用来跑推理的入口脚本（必需）
  model.pth        # 训练好的权重文件（建议放在同一目录）
  utils.py         # 其他辅助代码（可选）
  ...
```

不需要对你的原始代码做太多修改，**Tools Factory 会把整个文件夹原样“打包”进去**。

### 1.1 可选：写一个简单的 `tool.yaml`

在同一个目录 `my_precursor_tool/` 里，新建文件 `tool.yaml`，用于告诉编译器一些元信息（不写也可以，编译器会自动猜）：

```yaml
server_name: precursor-predictor        # 生成的 MCP server 名称（不用写就用文件夹名）
model_path: model.pth                   # 模型权重文件名（不用写就自动找第一个 .pt/.pth）

inputs:                                 # （可选）输入参数定义，只是元数据，方便后续接入
  formula: str

outputs:                                # （可选）输出结构定义
  precursors: list[str]
```

> 最简用法：**哪怕你只放了 `main.py + model.pth`，不写 `tool.yaml`，也能正常编译。**

---

## 2. 把工具“丢进” `incoming_raw/`

在本项目根目录下，结构大致是这样的：

```text
MatterSwarm/
  tools_factory/
    incoming_raw/
    templates/
    builder.py
    README.md  ← 就是你现在在看的这个文件
```

你只需要做一件事：**把你的工具文件夹整个拷贝到 `incoming_raw/` 下面。**

例如，把 `my_precursor_tool/` 拷到这里：

```text
tools_factory/
  incoming_raw/
    my_precursor_tool/
      main.py
      model.pth
      utils.py
      tool.yaml   # 如果你写了的话
```

> 可以直接用资源管理器拖拽，也可以用 `cp` / `xcopy` 命令，完全看你习惯。

---

## 3. 运行编译器：一键生成 MCP Server

### 3.1 激活你的 Python 环境

在项目根目录（包含 `requirements.txt` 的地方）打开终端，确保你已经安装过依赖：

```bash
cd /path/to/MatterSwarm

# 如果还没装过依赖，先装一遍
pip install -r requirements.txt
```

（Windows 下记得先激活你自己的虚拟环境 / conda 环境。）

### 3.2 一行命令编译所有工具

仍然在项目根目录，执行：

```bash
python -m tools_factory.builder
```

可能发生两种情况：

- **如果 `incoming_raw/` 里暂时是空的**：
  - 终端会提示类似：
    - `No tools found in tools_factory/incoming_raw. 请先把你的工具文件夹拖进去。`

- **如果里面有一个或多个工具文件夹**：
  - 编译器会依次处理每个文件夹，并输出生成结果路径，例如：

    ```text
    生成完成，以下 MCP servers 已创建：
     - /path/to/MatterSwarm/mcp_servers/my_precursor_tool
    ```

> 编译器会自动：
> - 创建 `mcp_servers/<server_name>/` 目录（如果已存在旧版本会先删除再重建）。
> - 把你的原始工具整体拷贝到 `raw_tool/` 子目录。
> - 根据模板生成 `server.py` 和一个简单的 `README.md`。

---

## 4. 看看编译结果长什么样

成功之后，项目里会多出一个或多个 MCP server 目录，例如：

```text
mcp_servers/
  my_precursor_tool/
    server.py      # MCP server 入口文件
    README.md      # 自动生成的使用说明
    raw_tool/      # 你的原始工具工程（main.py, model.pth, ...）
      main.py
      model.pth
      utils.py
      tool.yaml
```

你可以先打开这个 `README.md` 看看里面的说明，大致会类似：

```bash
cd mcp_servers/my_precursor_tool
python server.py
```

这条命令会启动一个基于 FastMCP 的 MCP Server。

> 注意：目前模板里的 `run_tool()` 只是一个“回声”示例，还没真正调用你的 `main.py`。  
> 但是这套壳子已经是标准 MCP 结构，后续只需要在模板里加上真正的推理调用即可。

---

## 5. 如何在 MCP 客户端里使用这个 Server（概念说明）

不同 MCP 客户端（例如 Claude Desktop / 其他 IDE 插件）有各自的配置方式，一般思路是：

1. 把 `mcp_servers/my_precursor_tool/server.py` 作为一个本地命令行服务。
2. 在客户端的配置文件里，添加一个新的 MCP server 条目，类似：

   - 命令：`python /path/to/MatterSwarm/mcp_servers/my_precursor_tool/server.py`
   - 工作目录：`/path/to/MatterSwarm/mcp_servers/my_precursor_tool`
   - 环境变量：如有需要，设置模型路径 / API Key 等。

3. 重启 MCP 客户端后，你应该能在工具列表里看到一个名叫 `my_precursor_tool`（或你在 `tool.yaml` 里指定的 `server_name`）的工具。

之后 MatterSwarm 的后端 `MCPExecutor` 也可以按这个 `server_name` 来连接和调用这个工具。

---

## 6. 常见小问题 & 提示

1. **编译时提示目录为空 / 找不到工具？**
   - 检查你的工具是不是放在 `tools_factory/incoming_raw/` 下面的一层子目录里，而不是更深或更浅。
   - 确认该子目录不是空的，而且确实是一个“文件夹”。

2. **忘了写 `tool.yaml` 会怎样？**
   - 没关系：
     - `server_name` 会用文件夹名（下划线会被换成 `-`）。
     - `model_path` 会自动扫描第一个 `.pt` 或 `.pth` 文件。
     - `inputs` / `outputs` 会填入一个简单的默认结构，只是用于元数据展示。

3. **想要支持真正的推理逻辑怎么办？**
   - 目前模板里的 `run_tool()` 只是简单返回输入和模型路径：
     - 你可以在 `tools_factory/templates/server_template.py` 基础上，增加：
       - `from raw_tool import main`（或你自己的入口函数）。
       - 在 `run_tool()` 里调用 `main.run_inference(**inputs)` 或类似逻辑。
   - 修改模板后，再重新运行 `python -m tools_factory.builder` 即可批量更新所有 server。

4. **重复编译会不会冲突？**
   - 不会：当目标 `mcp_servers/<server_name>/` 已经存在时，编译器会先删掉旧目录再完整重建。

---

## 7. 如果你不确定某一步怎么做

你只要记住一句话：

> “把有 `main.py` 和 `.pth` 的文件夹丢进 `tools_factory/incoming_raw/`，然后在项目根目录跑：  
> `python -m tools_factory.builder`。”  

如果遇到任何报错或不确定的地方，可以把终端输出复制给我，我可以根据你的实际工具结构帮你调整下一步该怎么做。 😄


