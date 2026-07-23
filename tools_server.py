from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FedSAM3-Tools")

import torch
@mcp.tool()
def check_gpu_status() -> str:
    """
    检查当前 GPU 的可用性和显存剩余情况。
    在开始训练前，请先调用此工具确认资源是否充足。
    """
    if not torch.cuda.is_available():
        return "没有检测到 GPU，正在使用 CPU (不建议运行 SAM3)"
    
    device_count = torch.cuda.device_count()
    status_report = []
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        free_mem = props.total_memory - torch.cuda.memory_allocated(i)
        free_mem_gb = free_mem / 1024**3
        status_report.append(f"GPU {i} ({props.name}): 剩余显存 {free_mem_gb:.2f} GB")
        
    return "\n".join(status_report)


if __name__ == "__main__":
    mcp.run()
