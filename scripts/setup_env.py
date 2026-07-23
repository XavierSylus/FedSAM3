#!/usr/bin/env python3
"""
设置环境变量以解决 OpenMP 库冲突
"""
import os

# 设置 KMP_DUPLICATE_LIB_OK 以解决 OpenMP 库冲突
# 这是一个已知的 PyTorch + NumPy 在 Windows 上的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("✅ 环境变量已设置: KMP_DUPLICATE_LIB_OK=TRUE")
