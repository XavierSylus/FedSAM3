#!/usr/bin/env python
"""
FedSAM3-Cream 快速测试脚本

一键运行所有诊断和测试，快速发现问题。

Author: FedSAM3-Cream Team
Date: 2026-03-28
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str, allow_failure: bool = False) -> bool:
    """
    运行命令并显示结果

    Args:
        cmd: 要运行的命令
        description: 命令描述
        allow_failure: 是否允许失败

    Returns:
        是否成功
    """
    print("\n" + "=" * 80)
    print(f"运行: {description}")
    print("=" * 80)
    print(f"命令: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"✅ {description} - 成功")
        return True
    else:
        if allow_failure:
            print(f"⚠️  {description} - 失败（但允许）")
            return True
        else:
            print(f"❌ {description} - 失败")
            return False


def main():
    """主函数"""
    print("=" * 80)
    print("FedSAM3-Cream 快速测试脚本")
    print("=" * 80)

    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    print(f"\n项目根目录: {project_root}")

    # 测试列表
    tests = [
        {
            "cmd": "python tools/diagnose_and_fix.py",
            "description": "运行诊断工具",
            "allow_failure": True  # 诊断工具可能发现问题
        },
        {
            "cmd": "python scripts/validate_dataset.py --check-leakage --verbose",
            "description": "验证数据集完整性",
            "allow_failure": False
        },
        {
            "cmd": "python src/gradient_monitor.py",
            "description": "测试梯度监控模块",
            "allow_failure": False
        },
        {
            "cmd": "python -c \"import src.gradient_monitor; print('✓ 梯度监控模块导入成功')\"",
            "description": "验证梯度监控模块可导入",
            "allow_failure": False
        }
    ]

    # 运行测试
    results = []
    for test in tests:
        success = run_command(
            test["cmd"],
            test["description"],
            test.get("allow_failure", False)
        )
        results.append({
            "description": test["description"],
            "success": success
        })

    # 生成报告
    print("\n\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{i}. {status} {result['description']}")

    # 建议
    print("\n" + "=" * 80)
    print("下一步建议")
    print("=" * 80)

    if passed == total:
        print("""
✅ 所有测试通过！可以开始运行实验:

1. Group A (纯视觉基线):
   python main.py --config configs/exp_group_a.yaml

2. Group B (文本污染对照):
   python main.py --config configs/exp_group_b.yaml

3. Group C (解耦蒸馏终极方案):
   python main.py --config configs/exp_group_c.yaml

查看详细诊断指南:
   docs/DIAGNOSTIC_GUIDE.md
        """)
    else:
        print("""
⚠️  存在失败的测试，请先修复问题:

1. 查看上面的错误信息
2. 参考诊断指南: docs/DIAGNOSTIC_GUIDE.md
3. 重新运行本脚本验证修复

常见问题:
- 数据集不存在: 运行 python scripts/recreate_client_splits.py
- 模块导入失败: 检查 Python 环境和依赖
        """)

    # 返回退出码
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
