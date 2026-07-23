"""
FeTS2022 Preprocessor - MCP Tool Entry Point

封装了 preprocess_FeTS2022.py 的功能，支持：
1. 将NIfTI格式的FeTS2022数据集转换为numpy格式
2. 图像归一化和窗口化处理
3. 多站点数据处理（支持站点: '1', '4', '5', '6', '13', '16', '18', '20', '21'）
4. 图像和标签的尺寸调整（图像1024x1024，标签256x256）
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import os

# ============================================
# 路径设置
# ============================================
_current_file = Path(__file__).resolve()
if 'raw_tool' in str(_current_file):
    # MCP环境：raw_tool/main.py -> 向上4级到项目根
    project_root = _current_file.parent.parent.parent.parent
else:
    # 直接运行环境：incoming_raw/fets2022_preprocessor/main.py -> 向上3级到项目根
    project_root = _current_file.parent.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core_projects" / "FedFMS-main"))

try:
    import SimpleITK as sitk
    import cv2
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")


def convert_from_nii_to_png(img: np.ndarray) -> np.ndarray:
    """
    将NIfTI图像转换为PNG格式（归一化到0-255）
    
    Args:
        img: 输入图像数组
    
    Returns:
        归一化后的图像数组（uint8格式）
    """
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class FeTS2022PreprocessorLoader:
    """FeTS2022数据预处理器加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
                - base_path: str, 原始数据路径（默认: '/mnt/diskB/name/FeTS2022'）
                - save_base_path: str, 保存路径（默认: '/mnt/diskB/name/FeTS2022_FedDG_1024'）
                - sites: list[str], 要处理的站点列表（默认: ['1', '4', '5', '6', '13', '16', '18', '20', '21']）
                - split: str, 数据集分割（默认: 'train'）
                - image_size: int, 输出图像尺寸（默认: 1024）
                - label_size: int, 输出标签尺寸（默认: 256）
        """
        self.config = config
        self.processed_sites = []
        
    def load(self) -> Dict[str, Any]:
        """
        加载预处理器配置
        
        Returns:
            配置字典
        """
        return {
            'base_path': self.config.get('base_path', '/mnt/diskB/name/FeTS2022'),
            'save_base_path': self.config.get('save_base_path', '/mnt/diskB/name/FeTS2022_FedDG_1024'),
            'sites': self.config.get('sites', ['1', '4', '5', '6', '13', '16', '18', '20', '21']),
            'split': self.config.get('split', 'train'),
            'image_size': self.config.get('image_size', 1024),
            'label_size': self.config.get('label_size', 256),
            'status': 'ready'
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取预处理器信息
        
        Returns:
            预处理器信息字典
        """
        return {
            'config': self.config,
            'processed_sites': self.processed_sites,
            'channels': {'1': 3, '4': 3, '5': 3, '6': 3, '13': 3, '16': 3, '18': 3, '20': 3, '21': 3}
        }


def preprocess_site(
    site: str,
    base_path: str,
    save_base_path: str,
    split: str = 'train',
    image_size: int = 1024,
    label_size: int = 256
) -> Dict[str, Any]:
    """
    预处理单个站点的数据
    
    Args:
        site: 站点ID
        base_path: 原始数据路径
        save_base_path: 保存路径
        split: 数据集分割
        image_size: 输出图像尺寸
        label_size: 输出标签尺寸
    
    Returns:
        处理结果字典
    """
    channels = {'1': 3, '4': 3, '5': 3, '6': 3, '13': 3, '16': 3, '18': 3, '20': 3, '21': 3}
    
    if site not in list(channels.keys()):
        return {
            'status': 'error',
            'error': f'Invalid site: {site}. Must be one of {list(channels.keys())}'
        }
    
    try:
        sitedir = os.path.join(base_path, site)
        save_sitedir = os.path.join(save_base_path, site)
        imgsdir = os.path.join(sitedir, 'images')
        labelsdir = os.path.join(sitedir, 'labels')
        savesdir = os.path.join(save_sitedir, 'data_npy')
        labelnpydir = os.path.join(save_sitedir, 'label_npy')
        
        # 创建保存目录
        if not os.path.exists(savesdir):
            os.makedirs(savesdir)
        if not os.path.exists(labelnpydir):
            os.makedirs(labelnpydir)
        
        freqsdir = os.path.join(save_sitedir, 'freq_amp_npy')
        if not os.path.exists(freqsdir):
            os.makedirs(freqsdir)
        
        # 获取所有样本
        ossitedir = os.listdir(imgsdir)
        images, labels = [], []
        save_path, label_path, freq_path = [], [], []
        
        processed_samples = 0
        total_slices = 0
        
        for j, sample in enumerate(ossitedir):
            imgdir = os.path.join(imgsdir, sample)
            labeldir = os.path.join(labelsdir, sample)
            savedir = os.path.join(savesdir, sample)
            savelabeldir = os.path.join(labelnpydir, sample)
            savefreqdir = os.path.join(freqsdir, sample)
            
            # 读取图像和标签
            label_v = sitk.ReadImage(labeldir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            
            # 标签处理：只保留类别4（肿瘤），转换为二值标签
            label_v[label_v == 4] = 1
            label_v[label_v != 1] = 0
            
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)
            
            # 处理每个切片
            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                # 跳过全零标签（每5个保留一个）
                if (np.all(label == 0)) and i % 5 != 0:
                    continue
                
                image = np.array(image_v[i-1:i+2, :, :])
                image = np.transpose(image, (1, 2, 0))
                
                labels.append(label)
                images.append(image)
                save_path.append(savedir + str(i) + '.npy')
                label_path.append(savelabeldir + str(i) + '.npy')
                freq_path.append(savefreqdir + str(i) + '.npy')
                total_slices += 1
        
        # 转换为numpy数组
        labels = np.array(labels).astype(int)
        images = np.array(images)
        
        # 处理每个样本：调整尺寸并保存
        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]
            
            # 调整尺寸
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (label_size, label_size), interpolation=cv2.INTER_NEAREST)
            label = np.expand_dims(label.astype(np.int64), axis=-1)
            
            # 保存
            np.save(save_path[idx], image)
            np.save(label_path[idx], label)
            processed_samples += 1
        
        return {
            'status': 'success',
            'site': site,
            'total_samples': len(ossitedir),
            'processed_slices': processed_samples,
            'total_slices': total_slices,
            'images_shape': list(images.shape) if len(images) > 0 else None,
            'labels_shape': list(labels.shape) if len(labels) > 0 else None,
            'save_dir': savesdir,
            'label_dir': labelnpydir
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'site': site,
            'error': str(e),
            'error_type': type(e).__name__
        }


def run_inference(
    preprocessor: Dict[str, Any],
    action: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行预处理操作
    
    Args:
        preprocessor: 预处理器配置字典
        action: 操作类型 ('preprocess_site', 'preprocess_all_sites', 'get_info')
        config: 配置字典，包含：
            - site: str, 站点ID（当action为preprocess_site时必需）
            - base_path: str, 原始数据路径
            - save_base_path: str, 保存路径
            - sites: list[str], 站点列表（当action为preprocess_all_sites时）
            - split: str, 数据集分割
            - image_size: int, 图像尺寸
            - label_size: int, 标签尺寸
    
    Returns:
        结果字典
    """
    base_path = config.get('base_path', preprocessor.get('base_path', '/mnt/diskB/name/FeTS2022'))
    save_base_path = config.get('save_base_path', preprocessor.get('save_base_path', '/mnt/diskB/name/FeTS2022_FedDG_1024'))
    split = config.get('split', preprocessor.get('split', 'train'))
    image_size = config.get('image_size', preprocessor.get('image_size', 1024))
    label_size = config.get('label_size', preprocessor.get('label_size', 256))
    
    if action == 'preprocess_site':
        site = config.get('site')
        if site is None:
            return {
                'status': 'error',
                'error': 'site parameter is required for preprocess_site action'
            }
        
        result = preprocess_site(
            site=site,
            base_path=base_path,
            save_base_path=save_base_path,
            split=split,
            image_size=image_size,
            label_size=label_size
        )
        return result
    
    elif action == 'preprocess_all_sites':
        sites = config.get('sites', preprocessor.get('sites', ['1', '4', '5', '6', '13', '16', '18', '20', '21']))
        results = []
        
        for site in sites:
            result = preprocess_site(
                site=site,
                base_path=base_path,
                save_base_path=save_base_path,
                split=split,
                image_size=image_size,
                label_size=label_size
            )
            results.append(result)
        
        # 统计信息
        success_count = sum(1 for r in results if r.get('status') == 'success')
        error_count = len(results) - success_count
        total_slices = sum(r.get('processed_slices', 0) for r in results if r.get('status') == 'success')
        
        return {
            'status': 'success',
            'action': 'preprocess_all_sites',
            'total_sites': len(sites),
            'success_sites': success_count,
            'error_sites': error_count,
            'total_slices': total_slices,
            'results': results
        }
    
    elif action == 'get_info':
        return {
            'status': 'success',
            'action': 'get_info',
            'base_path': base_path,
            'save_base_path': save_base_path,
            'split': split,
            'image_size': image_size,
            'label_size': label_size,
            'available_sites': ['1', '4', '5', '6', '13', '16', '18', '20', '21']
        }
    
    else:
        return {
            'status': 'error',
            'error': f'Unknown action: {action}. Must be one of: preprocess_site, preprocess_all_sites, get_info'
        }


def main():
    """主入口，用于直接运行测试"""
    import json
    
    # 示例配置
    config = {
        'base_path': '/mnt/diskB/name/FeTS2022',
        'save_base_path': '/mnt/diskB/name/FeTS2022_FedDG_1024',
        'sites': ['1'],
        'split': 'train',
        'image_size': 1024,
        'label_size': 256
    }
    
    # 加载预处理器
    loader = FeTS2022PreprocessorLoader(config)
    preprocessor = loader.load()
    
    print("FeTS2022 Preprocessor loaded successfully!")
    print(f"Base path: {preprocessor['base_path']}")
    print(f"Save path: {preprocessor['save_base_path']}")
    
    # 测试预处理
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == 'preprocess_site' and len(sys.argv) > 2:
            config['site'] = sys.argv[2]
        elif action == 'preprocess_all_sites':
            pass  # 使用默认站点列表
        elif action == 'get_info':
            pass
        else:
            print(f"Usage: python main.py <action> [site]")
            print(f"Actions: preprocess_site <site>, preprocess_all_sites, get_info")
            return
        
        result = run_inference(preprocessor, action, config)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()

