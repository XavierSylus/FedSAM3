#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集检查脚本 - FedSAM3-Cream 项目
功能：递归扫描数据集目录，提取医疗影像的详细信息
支持格式：.nii.gz (3D/4D), .png, .jpg
作者：FedSAM3-Cream 团队
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import traceback

import numpy as np
import nibabel as nib
from PIL import Image


class DatasetInfoExtractor:
    """数据集信息提取器"""

    def __init__(self, data_dir: str, output_csv: str = "dataset_info.csv"):
        """
        初始化数据集信息提取器

        Args:
            data_dir: 数据集根目录
            output_csv: 输出的 CSV 文件路径
        """
        self.data_dir = Path(data_dir)
        self.output_csv = output_csv
        self.results = []  # 存储所有文件的信息
        self.errors = []   # 存储错误信息

        # 支持的文件扩展名
        self.supported_extensions = ['.nii.gz', '.nii', '.png', '.jpg', '.jpeg']

    def is_mask_file(self, filename: str) -> bool:
        """
        判断是否为 mask/label/ground truth 文件

        Args:
            filename: 文件名

        Returns:
            是否为掩码文件
        """
        filename_lower = filename.lower()
        keywords = ['mask', 'label', 'gt', 'seg', 'segmentation']
        return any(keyword in filename_lower for keyword in keywords)

    def extract_spacing_from_nifti(self, img: nib.Nifti1Image) -> Tuple[float, float, float]:
        """
        从 NIfTI 图像中提取像素间距（spacing）

        Args:
            img: nibabel 加载的 NIfTI 图像对象

        Returns:
            (x_spacing, y_spacing, z_spacing) 三维间距元组
        """
        try:
            # 优先从 header 中获取
            header = img.header
            zooms = header.get_zooms()

            # 对于 4D 数据，只取前三个维度的间距
            if len(zooms) >= 3:
                return tuple(zooms[:3])
            else:
                return tuple(zooms)

        except Exception as e:
            # 如果 header 失败，尝试从 affine 矩阵计算
            try:
                affine = img.affine
                spacing = np.abs(np.diag(affine)[:3])
                return tuple(spacing)
            except:
                return (None, None, None)

    def get_middle_slice_info(self, data: np.ndarray) -> Tuple[str, List]:
        """
        获取中间切片信息及其前5个非零值

        Args:
            data: numpy 数组

        Returns:
            (切片形状字符串, 前5个非零值列表)
        """
        shape = data.shape

        # 处理不同维度的数据
        if len(shape) == 2:  # 2D 图像（PNG/JPG）
            slice_data = data
        elif len(shape) == 3:  # 3D 数据 (H, W, D)
            mid_idx = shape[2] // 2
            slice_data = data[:, :, mid_idx]
        elif len(shape) == 4:  # 4D 数据 (H, W, D, C) 或 (C, H, W, D)
            # 判断哪个维度是深度维度（通常是最大的那个）
            if shape[2] > shape[0] and shape[2] > shape[3]:
                # (H, W, D, C) 格式
                mid_idx = shape[2] // 2
                slice_data = data[:, :, mid_idx, 0]  # 取第一个通道
            else:
                # (C, H, W, D) 格式
                mid_idx = shape[3] // 2
                slice_data = data[0, :, :, mid_idx]  # 取第一个通道
        else:
            return ("N/A", [])

        # 获取前5个非零值
        nonzero_values = slice_data[slice_data != 0]
        top5_nonzero = nonzero_values.flatten()[:5].tolist() if len(nonzero_values) > 0 else []

        return (str(slice_data.shape), top5_nonzero)

    def process_nifti_file(self, file_path: Path) -> Dict:
        """
        处理 NIfTI 格式文件（.nii.gz / .nii）

        Args:
            file_path: 文件路径

        Returns:
            包含文件信息的字典
        """
        try:
            # 加载 NIfTI 文件
            img = nib.load(str(file_path))
            data = img.get_fdata()

            # 基本信息
            info = {
                '文件名': file_path.name,
                '相对路径': str(file_path.relative_to(self.data_dir)),
                '数据类型': str(data.dtype),
                '数据形状': str(data.shape),
                '数据范围': f"{data.min():.4f} ~ {data.max():.4f}",
            }

            # 提取 spacing 信息
            spacing = self.extract_spacing_from_nifti(img)
            info['像素间距'] = f"({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})" if spacing[0] is not None else "N/A"

            # 如果是 mask 文件，计算唯一值
            if self.is_mask_file(file_path.name):
                unique_values = np.unique(data)
                if len(unique_values) <= 20:  # 只显示类别数较少的情况
                    info['唯一数值'] = str(unique_values.tolist())
                else:
                    info['唯一数值'] = f"共 {len(unique_values)} 个唯一值"
            else:
                info['唯一数值'] = "N/A (非掩码文件)"

            # 获取中间切片信息
            slice_shape, top5_values = self.get_middle_slice_info(data)
            info['中间层切片形状'] = slice_shape
            info['切片前5个非零值'] = str(top5_values) if top5_values else "N/A"

            return info

        except Exception as e:
            error_msg = f"处理文件 {file_path.name} 时出错: {str(e)}"
            self.errors.append(error_msg)
            return {
                '文件名': file_path.name,
                '相对路径': str(file_path.relative_to(self.data_dir)),
                '错误信息': str(e)
            }

    def process_image_file(self, file_path: Path) -> Dict:
        """
        处理普通图像文件（.png / .jpg）

        Args:
            file_path: 文件路径

        Returns:
            包含文件信息的字典
        """
        try:
            # 加载图像
            img = Image.open(str(file_path))
            data = np.array(img)

            # 基本信息
            info = {
                '文件名': file_path.name,
                '相对路径': str(file_path.relative_to(self.data_dir)),
                '数据类型': str(data.dtype),
                '数据形状': str(data.shape),
                '数据范围': f"{data.min()} ~ {data.max()}",
                '像素间距': "N/A (普通图像)",
            }

            # 如果是 mask 文件，计算唯一值
            if self.is_mask_file(file_path.name):
                unique_values = np.unique(data)
                if len(unique_values) <= 20:
                    info['唯一数值'] = str(unique_values.tolist())
                else:
                    info['唯一数值'] = f"共 {len(unique_values)} 个唯一值"
            else:
                info['唯一数值'] = "N/A (非掩码文件)"

            # 2D 图像的"中间切片"就是它本身
            slice_shape, top5_values = self.get_middle_slice_info(data)
            info['中间层切片形状'] = slice_shape
            info['切片前5个非零值'] = str(top5_values) if top5_values else "N/A"

            return info

        except Exception as e:
            error_msg = f"处理文件 {file_path.name} 时出错: {str(e)}"
            self.errors.append(error_msg)
            return {
                '文件名': file_path.name,
                '相对路径': str(file_path.relative_to(self.data_dir)),
                '错误信息': str(e)
            }

    def scan_directory(self):
        """递归扫描数据集目录，收集所有支持的文件信息"""
        print(f"\n开始扫描数据集目录: {self.data_dir}")
        print("=" * 80)

        file_count = 0

        # 递归遍历目录
        for root, dirs, files in os.walk(self.data_dir):
            for filename in files:
                file_path = Path(root) / filename

                # 检查文件扩展名
                if file_path.suffix.lower() in ['.gz']:
                    # 检查是否是 .nii.gz
                    if file_path.stem.endswith('.nii'):
                        file_count += 1
                        print(f"\n[{file_count}] 处理 NIfTI 文件: {file_path.name}")
                        info = self.process_nifti_file(file_path)
                        self.results.append(info)

                elif file_path.suffix.lower() in ['.nii']:
                    file_count += 1
                    print(f"\n[{file_count}] 处理 NIfTI 文件: {file_path.name}")
                    info = self.process_nifti_file(file_path)
                    self.results.append(info)

                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    file_count += 1
                    print(f"\n[{file_count}] 处理图像文件: {file_path.name}")
                    info = self.process_image_file(file_path)
                    self.results.append(info)

        print(f"\n{'=' * 80}")
        print(f"扫描完成！共处理 {file_count} 个文件，其中 {len(self.errors)} 个文件出现错误。")

    def print_markdown_table(self):
        """在控制台打印 Markdown 格式的表格"""
        if not self.results:
            print("\n没有找到任何支持的文件。")
            return

        print("\n\n")
        print("=" * 80)
        print("数据集信息汇总表（Markdown 格式）")
        print("=" * 80)
        print()

        # 表头
        headers = ['文件名', '数据类型', '数据形状', '像素间距', '数据范围',
                   '唯一数值', '中间层切片形状', '切片前5个非零值']

        # 打印表头
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join([" --- " for _ in headers]) + "|"

        print(header_line)
        print(separator_line)

        # 打印数据行
        for result in self.results:
            if '错误信息' in result:
                # 错误行
                row = f"| {result.get('文件名', 'N/A')} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | {result.get('错误信息', 'Unknown')} |"
            else:
                row = f"| {result.get('文件名', 'N/A')} | {result.get('数据类型', 'N/A')} | " \
                      f"{result.get('数据形状', 'N/A')} | {result.get('像素间距', 'N/A')} | " \
                      f"{result.get('数据范围', 'N/A')} | {result.get('唯一数值', 'N/A')} | " \
                      f"{result.get('中间层切片形状', 'N/A')} | {result.get('切片前5个非零值', 'N/A')} |"
            print(row)

        print()
        print("=" * 80)

    def save_to_csv(self):
        """将结果保存为 CSV 文件"""
        if not self.results:
            print("\n没有数据可以保存。")
            return

        try:
            # 准备 CSV 字段
            fieldnames = ['文件名', '相对路径', '数据类型', '数据形状', '像素间距',
                         '数据范围', '唯一数值', '中间层切片形状', '切片前5个非零值', '错误信息']

            with open(self.output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    # 确保所有字段都存在
                    row = {field: result.get(field, 'N/A') for field in fieldnames}
                    writer.writerow(row)

            print(f"\n✓ CSV 文件已保存至: {self.output_csv}")

        except Exception as e:
            print(f"\n✗ 保存 CSV 文件时出错: {str(e)}")
            traceback.print_exc()

    def print_error_summary(self):
        """打印错误汇总"""
        if self.errors:
            print("\n\n")
            print("=" * 80)
            print("错误汇总")
            print("=" * 80)
            for idx, error in enumerate(self.errors, 1):
                print(f"{idx}. {error}")
            print("=" * 80)

    def run(self):
        """执行完整的数据集扫描流程"""
        print("\n" + "=" * 80)
        print("FedSAM3-Cream 数据集信息提取工具")
        print("=" * 80)

        # 检查目录是否存在
        if not self.data_dir.exists():
            print(f"\n错误：数据集目录不存在: {self.data_dir}")
            return

        # 扫描目录
        self.scan_directory()

        # 打印 Markdown 表格
        self.print_markdown_table()

        # 保存 CSV
        self.save_to_csv()

        # 打印错误汇总
        self.print_error_summary()

        print("\n处理完成！")


def main():
    """主函数：解析命令行参数并运行脚本"""
    parser = argparse.ArgumentParser(
        description='FedSAM3-Cream 数据集信息提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python check_dataset_info.py --data_dir ./data/BraTS2020
  python check_dataset_info.py --data_dir ./data/ISIC2018 --output dataset_report.csv
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='数据集根目录路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='dataset_info.csv',
        help='输出的 CSV 文件名（默认: dataset_info.csv）'
    )

    args = parser.parse_args()

    # 创建提取器并运行
    extractor = DatasetInfoExtractor(
        data_dir=args.data_dir,
        output_csv=args.output
    )
    extractor.run()


if __name__ == "__main__":
    main()
