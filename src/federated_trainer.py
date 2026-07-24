"""
FedSAM3-Cream 联邦学习训练器

此模块将训练循环逻辑封装到 FederatedTrainer 类中，提高代码的可维护性和可复用性。
严格保持与原脚本 scripts/train_brats_federated.py 的逻辑一致性。
"""

# ★ Fix Medium 4: 服务器无 GUI 环境下强制使用非交互式 Matplotlib 后端
import matplotlib
matplotlib.use('Agg')

import os
import math
import time
import random
import logging
import traceback
import platform
import hashlib
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Dict, Tuple, Optional, Any
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to python path to allow running as script directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_BRATS_NUM_CLASSES   = 3      # BraTS 肿瘤子区域（WT/TC/ET）
_BERT_TEXT_DIM       = 768    # BERT-base 输出维度
_CONTRASTIVE_DIM     = 1024   # 对比学习嵌入空间维度
_EARLY_STOP_PATIENCE = 20     # Early Stopping 耐心轮数
_VAL_PLOT_INTERVAL   = 5      # 验证集评估 & 绘图间隔（轮）

from src.config_manager import FederatedConfig
from src.integrated_model import SAM3MedicalIntegrated as SAM3_Medical
# ★ Fix Critical 2: 引入三个具体 Trainer 子类，移除已废弃的 ClientTrainer
from src.client import TextOnlyTrainer, ImageOnlyTrainer, MultimodalTrainer
from src.server import CreamAggregator
from src.logger import create_logger
from src.update_diagnostics import (
    compute_parameter_group_diagnostics,
    flatten_parameter_group_diagnostics,
)
from src.parameter_groups import (
    PARAMETER_GROUPS,
    allowed_modalities,
    classify_trainable_parameters,
)
from data.heterogeneous_dataset_loader import create_heterogeneous_data_loaders
from data_processing.brats_region_contract import (
    logits_to_brats_labels,
    regions_to_brats_labels,
)


# ★ Fix Critical 2: 工厂函数 - 根据 modality 字符串实例化正确的 Trainer 子类
def create_client_trainer(modality: str, **kwargs):
    """
    根据模态类型实例化对应的客户端训练器子类。

    Args:
        modality: 模态类型字符串，取值为 'text_only' | 'image_only' | 'multimodal'
        **kwargs: 传递给 Trainer 构造函数的参数
                  (private_loader, public_loader, device, use_amp,
                   local_epochs, dataset_name, embed_dim, grad_clip)
    Returns:
        对应的 Trainer 子类实例
    Raises:
        ValueError: 如果 modality 不在已知类型中
    """
    modality_map = {
        'text_only': TextOnlyTrainer,
        'image_only': ImageOnlyTrainer,
        'multimodal': MultimodalTrainer,
    }
    trainer_cls = modality_map.get(modality)
    if trainer_cls is None:
        raise ValueError(
            f"未知的客户端模态类型: '{modality}'，"
            f"有效值为 {list(modality_map.keys())}"
        )
    trainer = trainer_cls(**kwargs)
    trainer.client_modality = modality
    return trainer


def compute_gradient_conflict_from_vectors(
    client_grad_vectors: List[Optional[torch.Tensor]],
    client_modalities: List[str],
) -> Optional[float]:
    """Compute the adapter-gradient conflict angle from real client gradient vectors."""
    try:
        idx_img = client_modalities.index('image_only')
        idx_multi = client_modalities.index('multimodal')
    except ValueError:
        return None

    img_vec = client_grad_vectors[idx_img]
    multi_vec = client_grad_vectors[idx_multi]
    if img_vec is None or multi_vec is None:
        print("[WARNING][GradConflict] 缺少真实 adapter 梯度向量，跳过冲突角计算")
        return None

    img_vec = img_vec.detach().float().flatten().cpu()
    multi_vec = multi_vec.detach().float().flatten().cpu()
    if img_vec.numel() == 0 or multi_vec.numel() == 0:
        print("[WARNING][GradConflict] adapter 梯度向量为空，跳过冲突角计算")
        return None
    if img_vec.numel() != multi_vec.numel():
        print(
            "[WARNING][GradConflict] adapter 梯度向量维度不一致，"
            f"image_only={img_vec.numel()} multimodal={multi_vec.numel()}，跳过冲突角计算"
        )
        return None

    img_norm = torch.linalg.norm(img_vec)
    multi_norm = torch.linalg.norm(multi_vec)
    if img_norm.item() <= 1e-12 or multi_norm.item() <= 1e-12:
        print("[WARNING][GradConflict] adapter 梯度范数为 0，跳过冲突角计算")
        return None

    cos_sim = torch.dot(img_vec, multi_vec) / (img_norm * multi_norm)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0).item()
    angle_deg = math.degrees(math.acos(cos_sim))
    print(
        f"[GradConflict] true adapter gradient conflict angle: {angle_deg:.2f}° "
        f"(cos_sim={cos_sim:.4f}, adapter_elems={img_vec.numel()})"
    )
    return angle_deg


class FederatedTrainer:
    """
    联邦学习训练器
    
    封装了联邦学习训练的完整流程，包括：
    - 环境和模型初始化
    - 客户端配置和训练
    - 服务器聚合
    - 验证集评估
    - 检查点管理
    - 训练可视化
    """
    
    def __init__(self, config: FederatedConfig):
        """
        初始化联邦训练器
        
        Args:
            config: FederatedConfig 配置对象
        """
        self.config = config
        
        # 核心组件（延迟初始化）
        self.device = None
        self.global_model = None
        self.server = None
        self.client_configs = None
        self.client_trainers = None
        self.client_states = None
        self.client_sample_counts: Dict[str, int] = {}
        self.val_loader = None
        self.logger = None
        
        # 训练状态
        self.training_history = {
            'rounds': [],
            'avg_losses': [],
            'avg_seg_losses': [],
            'avg_cream_losses': [],
            'client_losses': [],
            'global_text_rep_norms': [],
            'global_image_rep_norms': [],
            'val_metrics': [],
            'run_metadata': {},
            # ★ 论文数据收集（新增）
            'lr_history': [],          # 每轮主干 LR（float）
            'gpu_mem_mb': [],          # 每轮 GPU 峰值显存（MB），CPU 环境为 0
            'round_time_sec': [],      # 每轮完整训练耗时（秒）
            'grad_conflict_deg': [],   # adapter 梯度冲突角（度），无图像或多模态对时记录 None
            'parameter_group_diagnostics': [],
            'parameter_group_effectiveness': [],
            'aggregation_audits': [],
        }
        self.last_val_metrics = {}
        self.best_val_dice = 0.0
        
        # 检查点目录：绑定到各组的 log_dir 下，避免不同实验组互相覆盖
        if not self.config.log_dir:
            self.config.log_dir = str(Path(self.config.data_root) / "logs")
        self.checkpoint_dir = Path(self.config.log_dir) / "checkpoints"

    @staticmethod
    def _normalize_client_id(client_id: str) -> str:
        return str(client_id).replace("_", "").lower()

    @staticmethod
    def _missing_modality_client_ratio(
        client_configs: Dict[str, Dict[str, Any]]
    ) -> float:
        if not client_configs:
            return 0.0
        incomplete_count = sum(
            1
            for config in client_configs.values()
            if str(config.get("modality", "")).strip() != "multimodal"
        )
        return incomplete_count / len(client_configs)

    def _get_enabled_config_clients(self) -> Dict[str, Dict[str, Any]]:
        enabled_clients: Dict[str, Dict[str, Any]] = {}
        if not getattr(self.config, "clients", None):
            return enabled_clients

        for client_cfg in self.config.clients:
            if not client_cfg.get("enabled", True):
                continue
            client_id = str(client_cfg.get("client_id", "")).strip()
            if not client_id:
                raise ValueError("config.clients contains empty client_id")
            norm_id = self._normalize_client_id(client_id)
            if norm_id in enabled_clients:
                raise ValueError(f"Duplicate client_id in config.clients: {client_id}")
            enabled_clients[norm_id] = client_cfg
        return enabled_clients

    def _validate_federated_protocol(
        self, enabled_clients: Dict[str, Dict[str, Any]]
    ) -> None:
        modalities = {
            str(client_cfg.get("modality", "")).strip()
            for client_cfg in enabled_clients.values()
        }
        invalid_modalities = modalities - {
            "text_only", "image_only", "multimodal"
        }
        if not modalities or invalid_modalities:
            raise ValueError(
                "Federated protocol has invalid enabled client modalities: "
                f"{sorted(invalid_modalities)}"
            )

        aggregation_method = str(
            getattr(self.config, "aggregation_method", "")
        ).lower()
        routing_mode = str(getattr(self.config, "routing_mode", "")).lower()
        sample_weight_unit = str(
            getattr(self.config, "sample_weight_unit", "")
        ).lower()
        update_policy = str(
            getattr(self.config, "unoptimized_update_policy", "")
        ).lower()
        expected_policy = {
            "unrestricted": "include_zero",
            "restricted": "exclude_and_renormalize",
        }

        if aggregation_method != "fedavg":
            raise ValueError(
                "Federated protocol requires aggregation_method='fedavg', "
                f"got {aggregation_method!r}"
            )
        if routing_mode not in expected_policy:
            raise ValueError(
                "Federated protocol requires routing_mode 'unrestricted' or "
                f"'restricted', got {routing_mode!r}"
            )
        if sample_weight_unit != "private_cases":
            raise ValueError(
                "Federated protocol requires sample_weight_unit='private_cases', "
                f"got {sample_weight_unit!r}"
            )
        if update_policy != expected_policy[routing_mode]:
            raise ValueError(
                "Federated protocol has an invalid unoptimized update policy: "
                f"routing_mode={routing_mode!r}, policy={update_policy!r}"
            )

    def _validate_client_protocol_after_filter(self) -> None:
        if not bool(getattr(self.config, "strict_protocol_check", True)):
            return
        if not self.client_configs:
            raise ValueError("Protocol check failed: filtered client_configs is empty")

        expected_clients = self._get_enabled_config_clients()
        if not expected_clients:
            return

        actual_norm_to_id: Dict[str, str] = {}
        for client_id in self.client_configs.keys():
            norm_id = self._normalize_client_id(client_id)
            if norm_id in actual_norm_to_id:
                raise ValueError(f"Protocol check failed: duplicate actual client id {client_id}")
            actual_norm_to_id[norm_id] = client_id

        expected_ids = set(expected_clients.keys())
        actual_ids = set(actual_norm_to_id.keys())
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        if missing:
            raise ValueError(f"Protocol check failed: missing clients {missing}")
        if unexpected:
            raise ValueError(f"Protocol check failed: unexpected clients {unexpected}")

        modality_mismatches = []
        for norm_id, expected_cfg in expected_clients.items():
            actual_id = actual_norm_to_id[norm_id]
            expected_modality = str(expected_cfg.get("modality", "")).strip()
            actual_modality = str(self.client_configs[actual_id].get("modality", "")).strip()
            if expected_modality != actual_modality:
                modality_mismatches.append(
                    f"{actual_id}: expected={expected_modality}, actual={actual_modality}"
                )
        if modality_mismatches:
            raise ValueError("Protocol check failed: " + "; ".join(modality_mismatches))

        self._validate_federated_protocol(expected_clients)

    def _derive_private_case_counts(self) -> Dict[str, int]:
        """Return the fixed FedAvg weight for each enabled private dataset."""
        if not self.client_configs:
            raise RuntimeError("Cannot derive private case counts without clients")

        private_case_counts: Dict[str, int] = {}
        for client_id in sorted(self.client_configs):
            client_config = self.client_configs[client_id]
            private_loader = client_config.get("private_loader")
            if private_loader is None or not hasattr(private_loader, "dataset"):
                raise RuntimeError(
                    f"Client {client_id} has no private loader dataset for FedAvg "
                    "sample weighting"
                )
            private_case_count = len(private_loader.dataset)
            if (
                isinstance(private_case_count, bool)
                or not isinstance(private_case_count, int)
                or private_case_count <= 0
            ):
                raise ValueError(
                    f"Client {client_id} private_case_count must be a positive "
                    f"integer, got {private_case_count!r}"
                )
            private_case_counts[client_id] = private_case_count

        return private_case_counts

    def _should_enable_text_assist_in_seg(self) -> bool:
        """Keep text-assisted segmentation independent from aggregation routing."""
        return bool(getattr(self.config, "enable_text_assist_in_seg", True))

    def _segmentation_trainer_kwargs(self) -> Dict[str, Any]:
        contract = {
            "segmentation_loss": getattr(self.config, "segmentation_loss", None),
            "seg_dice_weight": getattr(self.config, "seg_dice_weight", None),
            "seg_bce_weight": getattr(self.config, "seg_bce_weight", None),
            "seg_dice_smooth": getattr(self.config, "seg_dice_smooth", None),
        }
        missing = [name for name, value in contract.items() if value is None]
        if missing:
            raise ValueError(
                f"segmentation training contract is incomplete: missing {missing}"
            )
        if int(getattr(self.config, "num_classes", 0)) != 3:
            raise ValueError("BraTS segmentation requires num_classes=3")

        thresholds = getattr(self.config, "segmentation_thresholds", None)
        if thresholds is None or len(thresholds) != 3:
            raise ValueError(
                "segmentation inference contract requires thresholds for [WT, TC, ET]"
            )
        contract["segmentation_thresholds"] = tuple(thresholds)
        return contract

    def _select_client_initial_state(
        self,
        round_global_state: Dict[str, torch.Tensor],
        cached_client_state: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Return the common round-global initialization required by the 2x2 protocol."""
        policy = str(
            getattr(self.config, "client_init_policy", "round_global")
        ).lower()
        if policy != "round_global":
            raise ValueError(f"Unsupported client_init_policy: {policy}")
        return round_global_state

    def _should_restore_client_optimizer(self) -> bool:
        """Optimizer state must not cross rounds in the strict 2x2 protocol."""
        return bool(getattr(self.config, "persist_client_optimizer", False))

    def _restore_round_global_before_aggregation(
        self, round_global_state: Dict[str, torch.Tensor]
    ) -> None:
        self.server.apply_trainable_parameters(round_global_state)

    @staticmethod
    def _stable_json_dumps(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _sha256_json(cls, payload: Any) -> str:
        return hashlib.sha256(cls._stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def _build_run_identity(self) -> Dict[str, Any]:
        config_snapshot = self.config.to_dict()
        client_entries: List[Dict[str, Any]] = []
        if self.client_configs:
            for client_id in sorted(self.client_configs.keys(), key=self._normalize_client_id):
                client_cfg = self.client_configs[client_id]
                client_entries.append({
                    "client_id": str(client_id),
                    "modality": str(client_cfg.get("modality", "")).strip(),
                    "enabled": bool(client_cfg.get("enabled", True)),
                    "data_source": client_cfg.get("data_source"),
                })

        protocol_payload = {
            "experiment_name": getattr(self.config, "experiment_name", None),
            "seed": int(getattr(self.config, "seed", 3407)),
            "rounds": int(getattr(self.config, "rounds", 0)),
            "batch_size": int(getattr(self.config, "batch_size", 0)),
            "local_epochs": int(getattr(self.config, "local_epochs", 0)),
            "learning_rate": float(getattr(self.config, "lr", 0.0)),
            "seg_head_lr": float(getattr(self.config, "seg_head_lr", 0.0)),
            "adapter_lr": float(getattr(self.config, "adapter_lr", 0.0)),
            "weight_decay": float(getattr(self.config, "weight_decay", 0.0)),
            "lambda_cream": float(getattr(self.config, "lambda_cream", 0.0)),
            "aggregation_method": str(getattr(self.config, "aggregation_method", "")).lower(),
            "routing_mode": str(getattr(self.config, "routing_mode", "")).lower(),
            "sample_weight_unit": str(
                getattr(self.config, "sample_weight_unit", "")
            ).lower(),
            "unoptimized_update_policy": str(
                getattr(self.config, "unoptimized_update_policy", "")
            ).lower(),
            "client_sample_count_unit": "private_case_count",
            "client_sample_counts": {
                client_id: int(self.client_sample_counts[client_id])
                for client_id in sorted(self.client_sample_counts)
            },
            "baseline_method": str(getattr(self.config, "baseline_method", "none")).lower(),
            "fedprox_mu": float(getattr(self.config, "fedprox_mu", 0.0)),
            "client_init_policy": str(
                getattr(self.config, "client_init_policy", "round_global")
            ).lower(),
            "persist_client_optimizer": bool(
                getattr(self.config, "persist_client_optimizer", False)
            ),
            "strict_protocol_check": bool(getattr(self.config, "strict_protocol_check", True)),
            "proxy_client_id": getattr(self.config, "proxy_client_id", None),
            "segmentation": {
                "loss": getattr(self.config, "segmentation_loss", None),
                "dice_weight": getattr(self.config, "seg_dice_weight", None),
                "bce_weight": getattr(self.config, "seg_bce_weight", None),
                "smooth": getattr(self.config, "seg_dice_smooth", None),
                "thresholds": getattr(self.config, "segmentation_thresholds", None),
            },
            "data_root": str(getattr(self.config, "data_root", "")),
            "clients": client_entries,
            "missing_modality_client_ratio": self._missing_modality_client_ratio(
                self.client_configs or {}
            ),
        }

        return {
            "config_hash": self._sha256_json(config_snapshot),
            "protocol_hash": self._sha256_json(protocol_payload),
            "protocol_payload": protocol_payload,
        }

    def _set_random_seed(self) -> None:
        seed = int(getattr(self.config, "seed", 3407))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Prefer deterministic behavior for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Seed] Using random seed: {seed}")

    def _collect_run_metadata(self) -> Dict[str, Any]:
        run_identity = self._build_run_identity()
        metadata: Dict[str, Any] = {
            "seed": int(getattr(self.config, "seed", 3407)),
            "experiment_name": getattr(self.config, "experiment_name", None),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": self.device,
            "hostname": platform.node(),
            "run_started_at": datetime.now().isoformat(),
            "config_hash": run_identity["config_hash"],
            "protocol_hash": run_identity["protocol_hash"],
            "protocol_payload": run_identity["protocol_payload"],
        }
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                text=True,
            ).strip()
        except Exception:
            commit = "unknown"
        metadata["git_commit"] = commit
        return metadata

    def setup_environment(self):
        """初始化环境和全局模型"""
        print("\n[1/4] 初始化全局模型...")

        # 确定实际使用的设备
        self.device = self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，切换到 CPU")
            self.device = "cpu"
        self._set_random_seed()

        # Initialize the globally shared BraTS [WT, TC, ET] model.
        self.global_model = SAM3_Medical(
            img_size=self.config.img_size,
            num_classes=self.config.num_classes,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            use_sam3=not self.config.use_mock,
            sam3_checkpoint=self.config.sam3_checkpoint if not self.config.use_mock else None,
            text_dim=_BERT_TEXT_DIM,
            contrastive_dim=_CONTRASTIVE_DIM
        ).to(self.device)

        print(f"  [OK] 全局模型已初始化")

        # ★★★ Critical Fix: 显式重置 RoPE 频率缓存 ★★★
        # 问题：SAM3 预训练权重中的 freqs_cis 是 1024 尺寸，但我们用 256
        # 解决：在训练开始前强制重置，避免每个 forward 都动态重计算
        print(f"  [RoPE] 正在为 img_size={self.config.img_size} 重置 RoPE 频率缓存...")
        try:
            if hasattr(self.global_model, 'reset_rope_frequencies'):
                # ✅ 修复：reset_rope_frequencies() 不接受 img_size 参数
                # RoPE频率会根据模型初始化时的img_size自动计算
                rope_reset_count = self.global_model.reset_rope_frequencies(verbose=True)
                print(f"  [OK] RoPE 频率已重置（方法 1: reset_rope_frequencies, {rope_reset_count} blocks）")
            elif hasattr(self.global_model, 'wrapped_blocks'):
                # 手动遍历所有 blocks 重置
                rope_reset_count = 0
                for i, wrapped_block in enumerate(self.global_model.wrapped_blocks):
                    original_block = wrapped_block.block if hasattr(wrapped_block, 'block') else wrapped_block
                    attn = original_block.attn if hasattr(original_block, 'attn') else None

                    if attn and hasattr(attn, '_setup_rope_freqs') and getattr(attn, 'use_rope', False):
                        attn._setup_rope_freqs()
                        rope_reset_count += 1

                if rope_reset_count > 0:
                    print(f"  [OK] RoPE 频率已重置（方法 2: 手动重置 {rope_reset_count} 个 blocks）")
                else:
                    print(f"  [SKIP] 未找到需要重置的 RoPE blocks")
            else:
                print(f"  [SKIP] 模型不支持 RoPE 重置，跳过")
        except Exception as e:
            print(f"  [警告] RoPE 重置失败: {e}")
            print(f"  将在 forward 时动态重计算（性能会降低）")

        # 初始化服务器
        print("[2/4] 初始化服务器...")
        self.server = CreamAggregator(
            self.global_model,
            device=self.device,
            aggregation_method=self.config.aggregation_method,
            global_rep_alpha=self.config.global_rep_alpha,
            strict_aggregation_guard=getattr(self.config, "strict_aggregation_guard", False),
            proxy_k_batches=getattr(self.config, "proxy_k_batches", 1),
        )
        print(f"  [OK] 服务器已初始化")
    
    def setup_clients(self):
        """设置客户端配置和训练器（串行训练模式）"""
        print("[3/5] 设置客户端配置（串行训练模式）...")
        
        try:
            # 导入串行训练辅助函数
            from scripts.setup_serial_clients import setup_serial_clients
            
            # 获取客户端配置（不创建模型）
            self.client_configs = setup_serial_clients(
                data_root=self.config.data_root,
                batch_size=self.config.batch_size,
                img_size=self.config.img_size,
                max_samples=self.config.max_samples,
                embed_dim=self.config.embed_dim
            )

            # config.clients 白名单过滤：由 yaml 决定当前实验组加载哪些客户端
            # 注意：config 里 client_id 可能无下划线（client2），
            # 而 setup_serial_clients 返回的 key 带下划线（client_2），需兼容两种格式
            if hasattr(self.config, 'clients') and self.config.clients:
                allowed_ids = {
                    c['client_id'] for c in self.config.clients
                    if c.get('enabled', True)
                }
                self.client_configs = {
                    k: v for k, v in self.client_configs.items()
                    if any(k == aid or k.replace('_', '') == aid.replace('_', '') for aid in allowed_ids)
                }
                print(f"[Config-Driven Filter] 保留客户端: {list(self.client_configs.keys())}")

            self._validate_client_protocol_after_filter()
            self.client_sample_counts = self._derive_private_case_counts()


            print(f"    - {len(self.client_configs)} 个客户端配置已创建")
            for client_id, cfg in self.client_configs.items():
                print(
                    f"      * {client_id}: {cfg['modality']}, "
                    f"private_case_count={self.client_sample_counts[client_id]}"
                )
        except Exception as e:
            print(f"\n错误: 客户端配置设置失败")
            print(f"详细信息: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 初始化客户端 Trainer 对象（无模型）
        print("[4/5] 初始化客户端训练器（无状态）...")
        self.client_trainers = {}
        enable_text_assist_in_seg = self._should_enable_text_assist_in_seg()
        segmentation_kwargs = self._segmentation_trainer_kwargs()
        for client_id, cfg in self.client_configs.items():
            # ★ Fix Critical 2: 使用工厂函数，根据模态实例化对应子类
            self.client_trainers[client_id] = create_client_trainer(
                modality=cfg['modality'],
                private_loader=cfg['private_loader'],
                public_loader=cfg['public_loader'],
                device=self.device,
                use_amp=self.config.use_amp,
                local_epochs=self.config.local_epochs,
                dataset_name='BraTS',
                grad_clip=getattr(self.config, 'grad_clip', 1.0),
                accumulation_steps=getattr(self.config, 'accumulation_steps', 1),
                enable_text_assist_in_seg=enable_text_assist_in_seg,
                baseline_method=getattr(self.config, 'baseline_method', 'none'),
                fedprox_mu=getattr(self.config, 'fedprox_mu', 0.0),
                text_loss_temperature=self.config.text_loss_temperature,
                **segmentation_kwargs,
            )

        self._materialize_runtime_trainable_modules()

        # ★ Fix Medium 5: 初始化本地表征使用 contrastive_dim 而非 embed_dim
        contrastive_dim = getattr(self.config, 'contrastive_dim', _CONTRASTIVE_DIM)
        print("[5/5] 初始化客户端状态缓存（CPU）...")
        self.client_states = {}
        for client_id in self.client_configs.keys():
            initial_trainable_state_cpu = self.server.get_trainable_parameter_snapshot()

            self.client_states[client_id] = {
                'weights': initial_trainable_state_cpu,
                'opt_state': None,
                'local_reps': torch.zeros(contrastive_dim)  # ★ 使用 contrastive_dim
            }
        
        print(f"    - 客户端状态缓存已初始化 ({len(self.client_states)} 个客户端)")
        if len(self.client_states) > 0:
            first_client_id = list(self.client_states.keys())[0]
            cache_size_mb = sum(p.numel() for p in self.client_states[first_client_id]['weights'].values()) * 4 / 1024 / 1024
            print(f"    - 每个客户端缓存大小: ~{cache_size_mb:.1f} MB (仅可训练参数)")

    def _materialize_runtime_trainable_modules(self) -> None:
        image_client = next(
            (
                cfg for cfg in self.client_configs.values()
                if cfg.get("modality") in {"image_only", "multimodal"}
            ),
            None,
        )
        if image_client is None:
            return

        private_loader = image_client.get("private_loader")
        if private_loader is None:
            raise RuntimeError(
                "Cannot initialize runtime output modules without an image private loader"
            )
        try:
            batch = next(iter(private_loader))
        except StopIteration as exc:
            raise RuntimeError(
                "Cannot initialize runtime output modules from an empty image loader"
            ) from exc

        if isinstance(batch, dict):
            images = batch.get("image", batch.get("inp"))
        elif isinstance(batch, (list, tuple)):
            images = batch[0] if batch else None
        else:
            images = batch
        if not isinstance(images, torch.Tensor) or images.ndim < 4:
            raise RuntimeError(
                "Image loader must provide a batched image tensor for model initialization"
            )

        was_training = self.global_model.training
        try:
            self.global_model.eval()
            with torch.no_grad():
                self.global_model(images[:1].to(self.device))
        finally:
            self.global_model.train(was_training)

        registry_ids = {
            id(parameter)
            for parameter in self.global_model.get_trainable_params()
        }
        for adapter_attr in ("_sam3_adapter_conv", "_mock_adapter_conv"):
            output_adapter = getattr(self.global_model, adapter_attr, None)
            if output_adapter is None:
                continue
            missing = [
                name
                for name, parameter in output_adapter.named_parameters()
                if id(parameter) not in registry_ids
            ]
            if missing:
                raise RuntimeError(
                    f"{adapter_attr} parameters missing from training registry: {missing}"
                )
    
    def setup_validation(self):
        """准备验证集数据加载器"""
        print("\n准备验证集数据加载器...")

        # val_client_ids 优先使用 client_configs.keys()：
        # 该集合的 client_id（如 client_2）已由 setup_serial_clients 验证，
        # 与磁盘目录命名（带下划线）严格一致。
        # config.clients 中的 client_id（如 client2，无下划线）与目录不匹配，
        # 不能作为路径构造的来源。
        val_client_ids = []
        if self.client_configs:
            # 过滤掉 text_only（无图像，无法做 Dice 评估）
            val_client_ids = [
                cid for cid, cfg in self.client_configs.items()
                if cfg.get('modality') != 'text_only'
            ]
        if not val_client_ids:
            print("  ⚠ 未能从 client_configs 中找到图像客户端，跳过验证集加载")

        val_loaders = []
        for client_id in val_client_ids:
            try:
                _modality = self.client_configs[client_id].get('modality', 'image_only')
                val_loaders_dict = create_heterogeneous_data_loaders(
                    data_root=self.config.data_root,
                    split="val",
                    client_configs=[{
                        'client_id': client_id,
                        'modality': _modality,
                    }],
                    batch_size=self.config.batch_size,
                    image_size=self.config.img_size,
                    shuffle=False,
                    max_samples=self.config.max_samples,
                    include_text_features=False,
                    is_validation=True,
                    load_public=False,
                )
                if client_id in val_loaders_dict:
                    val_private, _ = val_loaders_dict[client_id]
                    if val_private is not None:
                        val_loaders.append(val_private)
            except Exception as e:
                print(f"  警告: 无法加载验证集 ({client_id}): {e}")
        
        # 合并所有验证集
        if val_loaders:
            val_datasets = [loader.dataset for loader in val_loaders]
            val_concat_dataset = ConcatDataset(val_datasets)
            self.val_loader = DataLoader(val_concat_dataset, batch_size=self.config.batch_size, shuffle=False)
            print(f"  [OK] 验证集准备完成，共 {len(val_concat_dataset)} 个样本")
        else:
            self.val_loader = None
            print("  ⚠ 未找到验证集数据，将跳过验证集评估")
    
    def setup_logging(self):
        """初始化日志记录器"""
        if self.config.log_type != 'none':
            print("\n初始化日志记录器...")
            config_dict = self.config.to_dict()
            self.logger = create_logger(
                log_type=self.config.log_type,
                experiment_name=self.config.experiment_name,
                project_name=self.config.wandb_project,
                log_dir=self.config.log_dir or str(Path(self.config.data_root) / "logs"),
                wandb_entity=self.config.wandb_entity,
                config=config_dict
            )
            print(f"[OK] 日志记录器已初始化（类型: {self.config.log_type}）")
    
    def train(self):
        """主训练循环"""
        # 环境设置
        self.setup_environment()
        
        # 客户端设置
        self.setup_clients()
        
        # 验证集设置
        self.setup_validation()
        
        # 日志设置
        self.setup_logging()
        self.training_history['run_metadata'] = self._collect_run_metadata()
        print(
            f"[RunMeta] git={self.training_history['run_metadata'].get('git_commit', 'unknown')}, "
            f"seed={self.training_history['run_metadata'].get('seed')}, device={self.device}, "
            f"protocol={self.training_history['run_metadata'].get('protocol_hash', 'unknown')[:12]}"
        )
        
        print("\n" + "=" * 60)
        print("开始联邦学习训练...")
        print("=" * 60)
        
        # 检查点恢复
        start_round = 1
        if self.config.resume_from or self.config.resume_from_checkpoint:
            start_round = self._resume_from_checkpoint()
        
        # Early Stopping 状态
        self.best_val_dice = 0.0
        _saved_best_dice = 0.0
        _patience_counter = 0
        _best_model_path = self.checkpoint_dir / "best_model.pth"

        # 主训练循环
        for round_num in range(start_round, self.config.rounds + 1):
            self._train_single_round(round_num)

            # 验证
            if self.val_loader is not None and (
                round_num % getattr(self.config, 'val_interval', 1) == 0 or round_num == 1
            ):
                self._evaluate_validation(round_num)

                # Early Stopping 逻辑
                if self.best_val_dice > _saved_best_dice:
                    _saved_best_dice = self.best_val_dice
                    _patience_counter = 0
                    # 保存最佳模型权重
                    try:
                        _best_model_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self.global_model.state_dict(),
                            _best_model_path
                        )
                        print(f"  [Best] 新最佳 Val Dice={_saved_best_dice:.4f}，已保存 best_model.pth")
                    except Exception as _e:
                        print(f"  [WARN] best_model 保存失败: {_e}")
                else:
                    _patience_counter += 1
                    print(f"  [EarlyStopping] patience={_patience_counter}/{_EARLY_STOP_PATIENCE}，当前最佳 Dice={_saved_best_dice:.4f}")
                    if _patience_counter >= _EARLY_STOP_PATIENCE:
                        print(f"\n🚨 [Early Stopping] 触发！连续 {_EARLY_STOP_PATIENCE} 次验证无提升，停止训练。")
                        print(f"   最佳 Val Dice = {_saved_best_dice:.4f}（已保存至 {_best_model_path}）")
                        break

            # 检查点保存
            if self.config.checkpoint_interval > 0 and round_num % self.config.checkpoint_interval == 0:
                print(f"\n  [检查点] 保存检查点（第 {round_num} 轮）...")
                try:
                    self.save_checkpoint(round_num)
                except Exception as e:
                    print(f"  [FAIL] 检查点保存失败: {e}")
                    traceback.print_exc()

            # 绘制训练曲线
            if round_num == 1 or round_num == self.config.rounds or round_num % _VAL_PLOT_INTERVAL == 0:
                try:
                    self.plot_training_curves(round_num)
                except Exception as e:
                    print(f"  [FAIL] 绘图失败: {e}")

        # 训练完成
        self._finalize_training()

        return 0

    def _prepare_round_global_reps(self) -> Dict[str, torch.Tensor]:
        """Build one round-global proxy pair and share it with every client."""
        disable_refresh = bool(
            getattr(self.config, "disable_global_rep_update", False)
        )
        if disable_refresh:
            round_global_reps = self.server.get_global_reps()
            proxy_client_id = "frozen_server_representations"
        else:
            configured_proxy_id = str(
                getattr(self.config, "proxy_client_id", "") or ""
            ).strip()
            if not configured_proxy_id:
                if float(getattr(self.config, "lambda_cream", 0.0)) > 0:
                    raise RuntimeError(
                        "Cream training requires server.proxy_client_id"
                    )
                round_global_reps = self.server.get_global_reps()
                proxy_client_id = "server_representations"
            else:
                matching_sources = [
                    (client_id, cfg)
                    for client_id, cfg in self.client_configs.items()
                    if self._normalize_client_id(client_id)
                    == self._normalize_client_id(configured_proxy_id)
                ]
                if len(matching_sources) != 1:
                    raise RuntimeError(
                        f"Configured proxy client not found: {configured_proxy_id}"
                    )
                proxy_client_id, proxy_config = matching_sources[0]
                if proxy_config.get("modality") != "multimodal":
                    raise RuntimeError(
                        f"Proxy client must be multimodal: {proxy_client_id}"
                    )
                proxy_loader = proxy_config.get("public_loader")
                if proxy_loader is None:
                    raise RuntimeError(
                        f"Proxy client has no public loader: {proxy_client_id}"
                    )
                global_image_rep, global_text_rep = (
                    self.server.generate_and_dispatch_global_proxies(
                        self.global_model,
                        proxy_loader,
                        self.device,
                    )
                )
                round_global_reps = {
                    "global_image_rep": global_image_rep,
                    "global_text_rep": global_text_rep,
                }

        invalid = [
            name
            for name in ("global_image_rep", "global_text_rep")
            if not isinstance(round_global_reps.get(name), torch.Tensor)
            or round_global_reps[name].numel() == 0
            or not torch.isfinite(round_global_reps[name]).all()
            or torch.linalg.vector_norm(
                round_global_reps[name].detach().float()
            ).item() <= 1e-12
        ]
        if invalid:
            raise RuntimeError(
                f"Invalid shared round-global proxies from {proxy_client_id}: {invalid}"
            )

        print(
            f"[Protocol] Shared round-global proxies generated from {proxy_client_id}"
        )
        return {
            name: round_global_reps[name].detach()
            for name in ("global_image_rep", "global_text_rep")
        }

    def _persist_parameter_group_diagnostics(
        self,
        round_num: int,
        diagnostics: Dict[str, Any],
    ) -> None:
        output_dir = Path(self.config.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_dir / "parameter_group_diagnostics.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"round": round_num, **diagnostics},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        rows = flatten_parameter_group_diagnostics(round_num, diagnostics)
        csv_path = output_dir / "parameter_group_diagnostics.csv"
        fieldnames = [
            "round",
            "row_type",
            "client_id",
            "client_a",
            "client_b",
            "modality_a",
            "modality_b",
            "parameter_group",
            "update_l2",
            "reference_l2",
            "relative_drift",
            "update_rms",
            "numel",
            "parameter_count",
            "nonzero_parameter_count",
            "nonzero_parameter_ratio",
            "sample_weight",
            "cosine_similarity",
            "angle_deg",
            "is_negative",
            "conflict_status",
            "shared_numel",
            "shared_parameter_count",
            "pair_count",
            "negative_pair_count",
            "negative_cosine_ratio",
            "conflict_rate",
            "mean_cosine_similarity",
            "mean_angle_deg",
            "shared_pair_count",
            "no_shared_pair_count",
            "undefined_pair_count",
            "routing_mode",
            "aggregation_client_ids",
            "aggregation_participation",
        ]
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="raise",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _finalize_parameter_group_effectiveness(
        self,
        client_reports: Dict[str, Dict[str, Any]],
        client_modalities: Dict[str, str],
        aggregation_audit: Dict[str, Any],
        aggregated_state: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, Any]]:
        active_client_ids = aggregation_audit.get("active_client_ids")
        parameter_audit = aggregation_audit.get("parameters")
        if not isinstance(active_client_ids, list) or not isinstance(parameter_audit, dict):
            raise RuntimeError("Aggregation audit is missing active clients or parameter entries")
        if set(active_client_ids) != set(client_reports):
            raise RuntimeError("Parameter group reports do not match aggregation clients")
        if set(active_client_ids) != set(client_modalities):
            raise RuntimeError("Aggregation modalities do not match aggregation clients")
        if set(parameter_audit) != set(aggregated_state):
            raise RuntimeError("Aggregation audit keys do not match aggregated parameters")

        routing_mode = aggregation_audit.get("routing_mode")
        if routing_mode != self.config.routing_mode:
            raise RuntimeError("Aggregation audit routing mode does not match configuration")
        parameter_groups = classify_trainable_parameters(parameter_audit)

        for client_id in active_client_ids:
            report = client_reports[client_id]
            if report.get("modality") != client_modalities[client_id]:
                raise RuntimeError(f"Client modality mismatch in effectiveness report: {client_id}")
            groups = report.get("groups")
            if not isinstance(groups, dict) or set(groups) != set(PARAMETER_GROUPS):
                raise RuntimeError(f"Client effectiveness groups are incomplete: {client_id}")

        for parameter_name in sorted(parameter_audit):
            entry = parameter_audit[parameter_name]
            parameter_group = parameter_groups[parameter_name]
            if entry.get("parameter_group") != parameter_group:
                raise RuntimeError(
                    f"Aggregation audit parameter group mismatch: {parameter_name}"
                )
            eligible_client_ids = entry.get("eligible_client_ids")
            normalized_weights = entry.get("normalized_weights")
            if not isinstance(eligible_client_ids, list) or not isinstance(normalized_weights, dict):
                raise RuntimeError(f"Aggregation audit entry is incomplete: {parameter_name}")
            if len(eligible_client_ids) != len(set(eligible_client_ids)):
                raise RuntimeError(f"Aggregation audit has duplicate eligible client: {parameter_name}")
            if not set(eligible_client_ids).issubset(active_client_ids):
                raise RuntimeError(f"Aggregation audit has unknown eligible client: {parameter_name}")
            if routing_mode == "unrestricted":
                if set(eligible_client_ids) != set(active_client_ids):
                    raise RuntimeError(
                        f"Unrestricted aggregation excluded an active client: {parameter_name}"
                    )
            else:
                allowed = allowed_modalities(parameter_group)
                for client_id in eligible_client_ids:
                    if client_modalities[client_id] not in allowed:
                        raise RuntimeError(
                            "Restricted aggregation admitted a routing-ineligible client: "
                            f"{client_id}:{parameter_name}"
                        )
            if set(normalized_weights) != set(eligible_client_ids):
                raise RuntimeError(f"Aggregation weights do not match eligible clients: {parameter_name}")

            for client_id in eligible_client_ids:
                normalized_weight = normalized_weights[client_id]
                if not isinstance(normalized_weight, (int, float)) or normalized_weight <= 0.0:
                    raise RuntimeError(
                        f"Aggregation has an invalid normalized weight: {client_id}:{parameter_name}"
                    )
                group_report = client_reports[client_id]["groups"][parameter_group]
                group_report["aggregation_eligible_count"] += 1
                group_report["aggregated_count"] += 1

        for client_id in active_client_ids:
            for parameter_group in PARAMETER_GROUPS:
                group_report = client_reports[client_id]["groups"][parameter_group]
                if group_report["aggregated_count"] > group_report["aggregation_eligible_count"]:
                    raise RuntimeError(
                        f"Aggregation count exceeds eligibility: {client_id}:{parameter_group}"
                    )
                group_report["aggregation_eligible"] = (
                    group_report["aggregation_eligible_count"] > 0
                )
                group_report["aggregated"] = group_report["aggregated_count"] > 0
        return client_reports

    def _persist_parameter_group_effectiveness(
        self,
        round_num: int,
        client_reports: Dict[str, Dict[str, Any]],
    ) -> None:
        output_dir = Path(self.config.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "parameter_group_effectiveness.jsonl"
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"round": round_num, "clients": client_reports},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


    def _train_single_round(self, round_num: int):
        """执行单轮联邦训练"""
        _round_start_time = time.time()
        # 重置 GPU 显存峰值计数器（仅在 CUDA 可用时有效）
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"\n{'=' * 60}")
        print(f"Round {round_num}/{self.config.rounds}")
        print(f"{'=' * 60}")
        
        round_client_updates = {}
        round_client_stats = {}
        round_client_grad_vectors = {}
        round_client_effectiveness = {}
        fedprox_mode = str(getattr(self.config, 'baseline_method', 'none')).lower() == 'fedprox'
        round_global_state = self.server.get_trainable_parameter_snapshot()
        round_buffer_snapshot = self.server.capture_round_buffer_snapshot()
        round_global_reference_state = round_global_state if fedprox_mode else None
        if fedprox_mode:
            print(
                f"[Baseline] FedProx active: mu={getattr(self.config, 'fedprox_mu', 0.0):.4f}, "
                f"shared_round_global_keys={len(round_global_reference_state)}"
            )
        round_global_reps = self._prepare_round_global_reps()
        
        # === 串行训练：逐个客户端训练 ===
        for client_idx, (client_id, cfg) in enumerate(self.client_configs.items(), 1):
            print(f"\n[Client {client_idx}/{len(self.client_configs)}] Training {client_id} ({cfg['modality']})...")
            
            # Step 1: 加载客户端状态到全局模型（GPU）
            print(f"  [1/5] Loading {client_id} state to GPU...")
            state_to_load = self._select_client_initial_state(
                round_global_state=round_global_state,
                cached_client_state=self.client_states[client_id]['weights'],
            )
            print("      [Protocol] Using shared round-global snapshot as client init")
            self.server.restore_round_buffer_snapshot(
                round_buffer_snapshot,
                reason=f"before_client:{client_id}",
            )
            self.server.apply_trainable_parameters(state_to_load)
            self.global_model.to(self.device)
            
            # Step 2: 动态裁剪优化器（根据模态构建参数列表）
            # ★ 精准路由（2026-03-14重构）：不同模态的客户端只优化自己负责的参数，
            # 彻底消除 backward 后无关参数出现 None 梯度的警告与 AMP 崩溃。
            print(f"  [2/5] Creating AdamW optimizer (modality-aware)...")
            client_modality = cfg['modality']
            if client_modality == 'text_only':
                # 文本客户端：仅优化文本投影层，绝不包含图像/分割参数
                opt_params = list(self.global_model.fusion_head.text_proj.parameters())
                print(f"      [text_only] optimizer params = fusion_head.text_proj only")
            elif client_modality == 'image_only':
                # 图像客户端：从全量可训练参数中剔除 fusion_head.text_proj 的参数
                text_proj_param_ids = {
                    id(p) for p in self.global_model.fusion_head.text_proj.parameters()
                }
                opt_params = [
                    p for p in self.global_model.get_trainable_params()
                    if id(p) not in text_proj_param_ids
                ]
                print(f"      [image_only] optimizer params = get_trainable_params() - text_proj")
            else:
                # multimodal：传入完整的可训练参数（原默认行为）
                opt_params = list(self.global_model.get_trainable_params())
                print(f"      [multimodal] optimizer params = get_trainable_params() (full)")
            # ★ 痛点3-a 修复（2026-03-17）：Adapter/PEFT 参数强制 lr=1e-4
            # 诊断：Logits 飙到 82.45 是因为 Adapter 参数随视觉主干 lr 一起爆炸更新。
            # 修复：将 opt_params 拆分为三组：
            #   - medical_seg_head：Zero-Init 冷启动需要 1e-3 暴力唤醒
            #   - adapter_* / lora_* 等 PEFT 参数：强制 lr=1e-4 防过拟合
            #   - 其余可训练参数（mask_decoder 主干等）：使用 config.lr
            _SEG_HEAD_LR = self.config.seg_head_lr
            _ADAPTER_LR = self.config.adapter_lr
            _WEIGHT_DECAY = self.config.weight_decay
            _SEG_HEAD_KW    = ('medical_seg_head',)
            _PEFT_KEYWORDS  = ('adapter', 'lora', 'text_adapter')
            # 用 named_parameters 获取模型名称，以便按关键词分组
            _all_named_params = dict(self.global_model.named_parameters())
            _seg_head_param_ids = set()
            _peft_param_ids     = set()
            for _name, _p in self.global_model.named_parameters():
                if not _p.requires_grad:
                    continue
                if any(kw in _name.lower() for kw in _SEG_HEAD_KW):
                    _seg_head_param_ids.add(id(_p))
                elif any(kw in _name.lower() for kw in _PEFT_KEYWORDS):
                    _peft_param_ids.add(id(_p))
            # 筛选当前模态 opt_params 中三类参数
            _seg_head_params = [p for p in opt_params if id(p) in _seg_head_param_ids]
            _peft_params     = [p for p in opt_params if id(p) in _peft_param_ids]
            _main_params     = [p for p in opt_params if id(p) not in _seg_head_param_ids and id(p) not in _peft_param_ids]
            _param_groups = []
            if _seg_head_params:
                _param_groups.append({
                    'params': _seg_head_params,
                    'lr': _SEG_HEAD_LR,
                    'initial_lr': _SEG_HEAD_LR,
                    'weight_decay': _WEIGHT_DECAY,
                })
                print(f"      [LR] Seg Head params:     {len(_seg_head_params)} params → lr={_SEG_HEAD_LR:.0e} (Zero-Init 唤醒, cosine 调度)")
            if _peft_params:
                _param_groups.append({
                    'params': _peft_params,
                    'lr': _ADAPTER_LR,
                    'initial_lr': _ADAPTER_LR,
                })
                print(f"      [LR] PEFT/Adapter params: {len(_peft_params)} params → lr={_ADAPTER_LR:.0e} (cosine 调度)")
            if _main_params:
                _param_groups.append({
                    'params': _main_params,
                    'lr': self.config.lr,
                    'initial_lr': self.config.lr,
                })
                print(f"      [LR] Main params:         {len(_main_params)} params → lr={self.config.lr:.2e} (可调度)")
            if not _param_groups:
                _param_groups = [{'params': opt_params, 'lr': self.config.lr, 'initial_lr': self.config.lr}]
            optimizer = torch.optim.AdamW(
                _param_groups, weight_decay=_WEIGHT_DECAY
            )

            # LR 调度器：计算 cosine 衰减比例，各参数组按 initial_lr × cosine_factor 更新
            # seg_head 1e-3 → ~1e-5，main params 5e-5 → 1e-6，后期 LR 比值保持在 10:1
            _lr_base   = self.config.lr
            _lr_min    = getattr(self.config, 'lr_min', 1e-6)
            _warmup    = getattr(self.config, 'lr_warmup_rounds', 0)
            _scheduler = getattr(self.config, 'lr_scheduler', 'none')

            if _warmup > 0 and round_num <= _warmup:
                _cosine_factor = round_num / _warmup
            elif _scheduler == 'cosine':
                _progress = (round_num - _warmup) / max(self.config.rounds - _warmup, 1)
                _cosine_factor = _lr_min / _lr_base + 0.5 * (1 - _lr_min / _lr_base) * (1 + math.cos(math.pi * _progress))
            elif _scheduler == 'linear':
                _progress = round_num / max(self.config.rounds, 1)
                _cosine_factor = 1.0 - _progress * (1 - _lr_min / _lr_base)
            else:
                _cosine_factor = 1.0

            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * _cosine_factor


            
            # 严格 2x2：所有实验格都使用轮内新优化器，禁止跨轮状态混杂。
            if (
                self._should_restore_client_optimizer()
                and self.client_states[client_id]['opt_state'] is not None
            ):
                try:
                    optimizer.load_state_dict(self.client_states[client_id]['opt_state'])
                    print("      Optimizer state restored")
                except Exception as e:
                    raise RuntimeError("Failed to restore client optimizer state") from e
            
            # Step 3: 训练
            print(f"  [3/5] Training...")
            trainer = self.client_trainers[client_id]

            try:
                global_reps = round_global_reps

                # ★ 核心修复：所有客户端统一使用 trainer.run() 训练
                # client.py Phase 2 已通过多态分发解决模态逻辑：
                # - TextOnlyTrainer: 计算文本对比学习损失，不做分割
                # - ImageOnlyTrainer: 分割 + Cream Loss（无文本特征）
                # - MultimodalTrainer: 分割 + Cream Loss（含文本特征）
                #
                # ★ Fix Critical 2: 删除不存在的 client_modality 参数
                # trainer.run() 签名为 (model, optimizer, global_reps, lambda_cream)
                # ★ Logits 监控：将当前联邦轮次注入模型，供 forward() 内监控代码使用
                self.global_model._monitor_epoch = round_num
                if cfg['modality'] == 'multimodal':
                    print(f"      [LossRule] total = seg + ({self.config.lambda_cream:.4f} * cream)")
                elif cfg['modality'] == 'image_only':
                    print("      [LossRule] total = seg (cream monitor only)")
                elif cfg['modality'] == 'text_only':
                    print("      [LossRule] total = cream (+prox if enabled), lambda_cream ignored by design")
                if fedprox_mode:
                    print(f"      [Baseline] FedProx enabled (mu={getattr(self.config, 'fedprox_mu', 0.0):.4f})")

                updated_weights, img_rep, txt_rep, stats = trainer.run(
                    model=self.global_model,
                    optimizer=optimizer,
                    global_reps=global_reps,
                    lambda_cream=self.config.lambda_cream,
                    global_reference_state=round_global_reference_state
                )
                grad_vector = getattr(trainer, '_last_adapter_grad_vector', None)
                round_client_grad_vectors[client_id] = (
                    grad_vector.detach().cpu() if grad_vector is not None else None
                )

                # 合并图像和文本表征为单一 local_reps（用于聚合）
                # 优先使用图像表征，如果不存在则使用文本表征
                local_reps = img_rep if img_rep is not None else txt_rep

                if updated_weights is None:
                    raise RuntimeError(
                        f"Client {client_id} returned no optimizer upload payload"
                    )
                optimizer_parameter_names = set(
                    trainer.get_active_optimizer_parameter_names()
                )
                upload_parameter_names = set(updated_weights)
                if upload_parameter_names != optimizer_parameter_names:
                    raise RuntimeError(
                        f"Client {client_id} upload keys do not equal its optimizer "
                        f"parameter keys: upload_only="
                        f"{sorted(upload_parameter_names - optimizer_parameter_names)}, "
                        f"optimizer_only="
                        f"{sorted(optimizer_parameter_names - upload_parameter_names)}"
                    )
                for parameter_name in sorted(upload_parameter_names):
                    round_global_parameter = round_global_state.get(parameter_name)
                    if round_global_parameter is None:
                        raise RuntimeError(
                            f"Client {client_id} uploaded parameter absent from the "
                            f"round-global snapshot: {parameter_name}"
                        )
                    parameter_delta = (
                        updated_weights[parameter_name] - round_global_parameter
                    )
                    if not torch.isfinite(parameter_delta).all():
                        raise RuntimeError(
                            f"Client {client_id} produced a non-finite parameter delta: "
                            f"{parameter_name}"
                        )
                client_effectiveness = trainer.collect_parameter_group_effectiveness(
                    model=self.global_model,
                    round_global_state=round_global_state,
                    uploaded_state=updated_weights,
                )
                client_effectiveness["client_id"] = client_id
                round_client_effectiveness[client_id] = client_effectiveness

                print(
                    f"      Loss: {stats.get('avg_loss', 0):.4f}, "
                    f"Seg: {stats.get('avg_seg_loss', 0):.4f}, "
                    f"Cream: {stats.get('avg_cream_loss', 0):.4f}, "
                    f"NonSegComponent: {stats.get('avg_non_seg_component', 0):.4f}"
                )
                print(f"      Batches: {stats.get('num_batches', 0)}, LR: {self.config.lr * _cosine_factor:.2e}")

            except Exception as e:
                logging.error(f"Training failed with error:\n{traceback.format_exc()}")
                # 重新抛出，让错误可见
                raise
            
            # Step 4: 保存客户端状态回 CPU（内存安全）
            print(f"  [4/5] Saving state to CPU cache...")

            self.client_states[client_id]['weights'] = {
                parameter_name: updated_weights[parameter_name].detach().cpu().clone()
                for parameter_name in sorted(upload_parameter_names)
            }

            if self._should_restore_client_optimizer():
                self.client_states[client_id]['opt_state'] = optimizer.state_dict()
            else:
                self.client_states[client_id]['opt_state'] = None

            # Keep local representations for diagnostics; global round proxies are
            # generated from fixed public multimodal data, not client uploads.
            if local_reps is not None:
                self.client_states[client_id]['local_reps'] = local_reps.cpu() if local_reps.device.type != 'cpu' else local_reps
            else:
                # 理论上不应该发生，因为每个客户端至少应该有一个表征
                raise ValueError(f"客户端 {client_id} 没有返回任何表征（img_rep 和 txt_rep 均为 None）")

            # Step 5: 收集更新
            print(f"  [5/5] Collecting updates...")

            round_client_updates[client_id] = self.client_states[client_id]['weights']
            print(f"      [OK] Collected optimizer-scoped upload from {client_id}")

            round_client_stats[client_id] = stats
            self.server.restore_round_buffer_snapshot(
                round_buffer_snapshot,
                reason=f"after_client:{client_id}",
            )
            
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  [OK] {client_id} completed")
        
        # === 服务器聚合 ===
        client_ids_sorted = sorted(round_client_updates.keys())
        client_modality_map = {
            client_id: str(self.client_configs[client_id]['modality'])
            for client_id in client_ids_sorted
        }
        client_sample_counts = {
            client_id: self.client_sample_counts[client_id]
            for client_id in client_ids_sorted
        }
        client_modalities = [client_modality_map[client_id] for client_id in client_ids_sorted]
        print(
            f"\n[Aggregation] routing={self.config.routing_mode}, "
            f"clients={client_ids_sorted}, private_case_counts={client_sample_counts}"
        )

        grad_conflict_deg = compute_gradient_conflict_from_vectors(
            [round_client_grad_vectors[client_id] for client_id in client_ids_sorted],
            client_modalities,
        )
        self.server._last_grad_conflict_deg = grad_conflict_deg

        self.server.restore_round_buffer_snapshot(
            round_buffer_snapshot,
            reason="before_aggregation",
        )
        self._restore_round_global_before_aggregation(round_global_state)
        print("  [Protocol] Restored round-global parameters and buffers before server aggregation")

        aggregated_state = self.server.aggregate_weights(
            round_global_parameters=round_global_state,
            client_updates={
                client_id: round_client_updates[client_id]
                for client_id in client_ids_sorted
            },
            client_modalities=client_modality_map,
            client_sample_counts=client_sample_counts,
            routing_mode=self.config.routing_mode,
        )
        aggregation_audit = getattr(self.server, "_last_aggregation_audit", None)
        if not isinstance(aggregation_audit, dict):
            raise RuntimeError("Server did not produce the required aggregation audit")
        round_client_effectiveness = self._finalize_parameter_group_effectiveness(
            client_reports=round_client_effectiveness,
            client_modalities=client_modality_map,
            aggregation_audit=aggregation_audit,
            aggregated_state=aggregated_state,
        )
        self._persist_parameter_group_effectiveness(
            round_num,
            round_client_effectiveness,
        )

        round_group_diagnostics = compute_parameter_group_diagnostics(
            round_global_state=round_global_state,
            client_updates={
                cid: round_client_updates[cid]
                for cid in client_ids_sorted
                if round_client_updates[cid] is not None
            },
            client_modalities=client_modality_map,
            client_sample_counts=client_sample_counts,
            aggregation_audit=aggregation_audit,
            aggregated_state=aggregated_state,
        )
        self._persist_parameter_group_diagnostics(
            round_num,
            round_group_diagnostics,
        )
        
        # Apply only aggregated named parameters, then discard all client-local buffers.
        self.server.apply_trainable_parameters(aggregated_state)
        self.server.restore_round_buffer_snapshot(
            round_buffer_snapshot,
            reason="after_aggregation",
        )
        self.training_history['aggregation_audits'].append(
            {
                'round': round_num,
                **aggregation_audit,
                'buffer_distribution': self.server.get_round_buffer_distribution_audit(),
            }
        )
        
        # 获取更新后的全局表示
        updated_global_reps = self.server.get_global_reps()
        
        # 计算平均训练损失
        client_losses = [s.get('avg_loss', 0.0) for s in round_client_stats.values()]
        client_seg_losses = [s.get('avg_seg_loss', 0.0) for s in round_client_stats.values()]
        client_cream_losses = [s.get('avg_cream_loss', 0.0) for s in round_client_stats.values()]
        
        avg_train_loss = sum(client_losses) / len(client_losses) if len(client_losses) > 0 else 0.0
        avg_seg_loss = sum(client_seg_losses) / len(client_seg_losses) if len(client_seg_losses) > 0 else 0.0
        avg_cream_loss = sum(client_cream_losses) / len(client_cream_losses) if len(client_cream_losses) > 0 else 0.0
        
        # 日志记录
        print(f"  ✓ Round {round_num} Summary:")
        print(f"    Avg Loss: {avg_train_loss:.4f} (Seg: {avg_seg_loss:.4f}, Cream: {avg_cream_loss:.4f})")
        
        # 记录到日志系统
        if self.logger is not None:
            log_metrics = {
                'Train_Loss': avg_train_loss,
                'Seg_Loss': avg_seg_loss,
                'Cream_Loss': avg_cream_loss,
            }
            # 如果存在最后一次验证指标，也一并记录
            if self.last_val_metrics:
                val_dice = self.last_val_metrics.get('dice', 0.0)
                val_iou = self.last_val_metrics.get('iou', 0.0)
                val_hd95 = self.last_val_metrics['hd95']
                log_metrics['Val_Dice'] = val_dice
                log_metrics['Val_IoU'] = val_iou
                log_metrics['Val_HD95_Pixel'] = val_hd95
            self.logger.log(log_metrics, step=round_num)
        
        # 记录训练历史
        self.training_history['rounds'].append(round_num)
        self.training_history['avg_losses'].append(avg_train_loss)
        self.training_history['avg_seg_losses'].append(avg_seg_loss)
        self.training_history['avg_cream_losses'].append(avg_cream_loss)
        self.training_history['client_losses'].append({
            client_id: stat.get('avg_loss', 0.0) for client_id, stat in round_client_stats.items()
        })
        self.training_history['parameter_group_diagnostics'].append(
            round_group_diagnostics
        )
        self.training_history['parameter_group_effectiveness'].append(
            {"round": round_num, "clients": round_client_effectiveness}
        )
        self.training_history['global_text_rep_norms'].append(
            updated_global_reps['global_text_rep'].norm().item()
        )
        self.training_history['global_image_rep_norms'].append(
            updated_global_reps['global_image_rep'].norm().item()
        )
        # ★ 论文数据收集：LR / 显存 / 耗时 / 梯度冲突
        self.training_history['lr_history'].append(float(self.config.lr * _cosine_factor))
        _peak_mem_mb = 0.0
        if torch.cuda.is_available():
            _peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        self.training_history['gpu_mem_mb'].append(round(_peak_mem_mb, 2))
        _elapsed = time.time() - _round_start_time
        self.training_history['round_time_sec'].append(round(_elapsed, 2))
        # grad_conflict_deg is computed from real adapter gradients and is None
        # when there is no image-only/multimodal client pair.
        _conflict = getattr(self.server, '_last_grad_conflict_deg', None)
        self.training_history['grad_conflict_deg'].append(_conflict)
        print(f"  [PaperData] Round {round_num}: LR={self.config.lr * _cosine_factor:.2e}, "
              f"GPU峰值={_peak_mem_mb:.0f}MB, 耗时={_elapsed:.1f}s, "
              f"GradConflict={f'{_conflict:.1f}°' if _conflict is not None else 'N/A'}")
        
        # 详细总结（每10轮或第1轮）
        if round_num % 10 == 0 or round_num == 1:
            print(f"\n  === Round {round_num} Detailed Summary ===")
            print(f"    Training Loss: {avg_train_loss:.4f}")
            print(f"    Seg Loss: {avg_seg_loss:.4f}")
            print(f"    Cream Loss: {avg_cream_loss:.4f}")
            print(f"    Global Text Rep Norm: {updated_global_reps['global_text_rep'].norm().item():.4f}")
            print(f"    Global Image Rep Norm: {updated_global_reps['global_image_rep'].norm().item():.4f}")
            print(f"    Participating Clients: {len(round_client_updates)}")
            # 显示每个客户端的损失
            for client_id, stats in round_client_stats.items():
                print(f"      * {client_id}: Loss={stats.get('avg_loss', 0):.4f}, Seg={stats.get('avg_seg_loss', 0):.4f}, Cream={stats.get('avg_cream_loss', 0):.4f}")
    
    def _evaluate_validation(self, round_num: int):
        """验证集评估"""
        print(f"\n[Validation] Evaluating on validation set...")
        try:
            # 使用全局模型进行验证
            verbose_diagnosis = (round_num == 1 or round_num % 50 == 0)
            
            # 使用任意一个 trainer 的 validate 方法
            any_trainer = list(self.client_trainers.values())[0]
            val_metrics = any_trainer.validate(
                model=self.global_model,
                test_loader=self.val_loader,
                compute_hd95=True,
                verbose=verbose_diagnosis
            )
            
            print(f"  Dice: {val_metrics.get('dice', 0):.4f}, IoU: {val_metrics.get('iou', 0):.4f}")
            print(f"  HD95: {val_metrics['hd95']:.2f} pixel")
            
            # 记录验证指标到日志系统
            if self.logger is not None:
                val_dice = val_metrics.get('dice', 0.0)
                val_iou = val_metrics.get('iou', 0.0)
                log_metrics = {
                    'Val_Dice': val_dice,
                    'Val_IoU': val_iou,
                    'Val_HD95_Pixel': val_metrics['hd95'],
                }
                for region in ("WT", "TC", "ET"):
                    log_metrics[f'Val_{region}_Dice'] = val_metrics[f'{region}_dice']
                    log_metrics[f'Val_{region}_IoU'] = val_metrics[f'{region}_iou']
                    log_metrics[f'Val_{region}_HD95_Pixel'] = val_metrics[f'{region}_hd95']
                    log_metrics[f'Val_{region}_Empty_FP_Rate'] = val_metrics[
                        f'{region}_empty_fp_rate'
                    ]
                    log_metrics[f'Val_{region}_Empty_FN_Rate'] = val_metrics[
                        f'{region}_empty_fn_rate'
                    ]
                self.logger.log(log_metrics, step=round_num)
            
            # 保存最后一次验证指标
            self.last_val_metrics = val_metrics.copy()

            # ★ 修复C：Early Stopping 追踪逻辑——必须用 MAX 判断，不能简单覆盖
            # 将 best_val_dice 的 MAX 判断移至此处，作为唯一可信的更新点
            _current_dice = val_metrics.get('dice', 0.0)
            if _current_dice > self.best_val_dice:
                self.best_val_dice = _current_dice
                print(f"  [EarlyStopping] ★ 新最佳 Val Dice={self.best_val_dice:.4f}（已更新 self.best_val_dice）")
            else:
                print(f"  [EarlyStopping] 当前 Dice={_current_dice:.4f} ≤ 历史最佳 Dice={self.best_val_dice:.4f}，不更新")

            history_entry = {'round': round_num, **val_metrics}
            history_entry['val_loss'] = val_metrics.get('val_loss', 0.0)
            self.training_history['val_metrics'].append(history_entry)

            # 显示验证集指标
            print(f"\n    Validation Metrics:")
            print(f"      Dice: {val_metrics.get('dice', 0.0):.4f}")
            print(f"      IoU: {val_metrics.get('iou', 0.0):.4f}")
            print(f"      HD95: {val_metrics['hd95']:.2f} pixel")
                
        except Exception as e:
            print(f"  ⚠ Validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _resume_from_checkpoint(self) -> int:
        """从检查点恢复训练"""
        checkpoint_path = Path(self.config.resume_from or self.config.resume_from_checkpoint)
        
        if checkpoint_path.exists():
            try:
                print(f"\n从检查点恢复: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 加载服务器状态
                if 'server_state' in checkpoint:
                    self.server.load_state_dict(checkpoint['server_state'], strict=True)
                    print("  [OK] 服务器状态已恢复")
                
                # 获取恢复的轮数
                resume_round = checkpoint.get('round', 0)
                print(f"  [OK] 将从第 {resume_round + 1} 轮继续训练")
                
                # 加载训练历史
                self.training_history = checkpoint.get('training_history', self.training_history)
                print("  [OK] 训练历史已恢复")
                
                # 加载客户端状态缓存
                if 'client_states' in checkpoint and checkpoint['client_states'] is not None:
                    self.client_states = checkpoint['client_states']
                    print("  [OK] 客户端状态缓存已恢复")
                
                return resume_round + 1
                
            except Exception as e:
                print(f"  [FAIL] 检查点恢复失败: {e}")
                print("  将从头开始训练...")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print(f"  ⚠ 检查点文件不存在: {checkpoint_path}")
            print("  将从头开始训练...")
            return 1
    
    def _finalize_training(self):
        """训练完成后的处理"""
        print("\n" + "=" * 60)
        print("联邦学习训练完成！")
        print("=" * 60)
        
        # 最终模型统计
        print("\n最终全局模型统计:")
        final_model = self.server.get_global_model()
        total_params = sum(p.numel() for p in final_model.parameters())
        trainable_params = sum(p.numel() for p in final_model.get_trainable_params())
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数量: {trainable_params:,}")
        print(f"  - 冻结参数量: {total_params - trainable_params:,}")
        
        # 保存最终模型
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / "final_model.pth"
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'global_text_rep': self.server.global_text_rep.cpu(),
            'global_image_rep': self.server.global_image_rep.cpu(),
        }, save_path)
        print(f"\n最终模型已保存到: {save_path}")
        
        # 保存最终检查点
        print("\n保存最终检查点...")
        try:
            final_checkpoint_path = self.save_checkpoint(self.config.rounds)
            print(f"最终检查点已保存到: {final_checkpoint_path}")
        except Exception as e:
            print(f"  [FAIL] 最终检查点保存失败: {e}")
        
        # 保存训练历史记录
        history_path = self.checkpoint_dir / "training_history.json"
        self.training_history['final_stats'] = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'frozen_params': int(total_params - trainable_params),
            'total_rounds': self.config.rounds,
            'final_avg_loss': float(self.training_history['avg_losses'][-1]) if self.training_history['avg_losses'] else 0.0
        }
        self.training_history['training_time'] = datetime.now().isoformat()
        self.training_history.setdefault('run_metadata', {})
        self.training_history['run_metadata']['run_finished_at'] = datetime.now().isoformat()
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        run_meta_path = self.checkpoint_dir / "run_metadata.json"
        with open(run_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history.get('run_metadata', {}), f, indent=2, ensure_ascii=False)
        print(f"Run metadata saved to: {run_meta_path}")
        print(f"训练历史已保存到: {history_path}")
        
        # 最终验证集评估
        if self.val_loader is not None:
            print("\n" + "=" * 60)
            print("最终验证集评估")
            print("=" * 60)
            try:
                any_trainer = list(self.client_trainers.values())[0]
                final_val_metrics = any_trainer.validate(
                    model=self.global_model,
                    test_loader=self.val_loader,
                    compute_hd95=True,
                    verbose=True
                )
                
                print(f"\n最终评估指标:")
                print(f"  Dice 系数: {final_val_metrics.get('dice', 0.0):.4f}")
                print(f"  IoU: {final_val_metrics.get('iou', 0.0):.4f}")
                print(f"  HD95: {final_val_metrics['hd95']:.2f} pixel")
                
                # 保存分割掩码
                if self.config.save_masks:
                    print(f"\n保存分割掩码...")
                    mask_save_dir = self.checkpoint_dir / "segmentation_masks"
                    try:
                        self.save_segmentation_masks(
                            self.global_model,
                            self.val_loader,
                            mask_save_dir,
                            self.config.max_masks
                        )
                        print(f"  [OK] 分割掩码已保存到: {mask_save_dir}")
                    except Exception as e:
                        print(f"  [FAIL] 保存分割掩码失败: {e}")
                
                self.training_history['final_val_metrics'] = final_val_metrics
                
            except Exception as e:
                print(f"最终验证集评估失败: {e}")
                import traceback
                traceback.print_exc()
            print("=" * 60)
        
        # 记录最终总结到日志系统
        if self.logger is not None:
            summary = {}
            if self.training_history['avg_losses']:
                summary['final_train_loss'] = self.training_history['avg_losses'][-1]
                summary['initial_train_loss'] = self.training_history['avg_losses'][0]
            if self.training_history.get('avg_cream_losses'):
                summary['final_cream_loss'] = self.training_history['avg_cream_losses'][-1]
            if self.training_history['val_metrics']:
                best_dice = max(m['dice'] for m in self.training_history['val_metrics'])
                best_iou = max(m['iou'] for m in self.training_history['val_metrics'])
                summary['best_val_dice'] = best_dice
                summary['best_val_iou'] = best_iou
            if 'final_val_metrics' in self.training_history:
                final_metrics = self.training_history['final_val_metrics']
                summary['final_val_dice'] = final_metrics.get('dice', 0.0)
                summary['final_val_hd95_pixel'] = final_metrics['hd95']
            self.logger.log_summary(summary)
            self.logger.close()
        
        # 打印训练总结
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        if self.training_history['avg_losses']:
            print(f"初始损失: {self.training_history['avg_losses'][0]:.4f}")
            print(f"最终损失: {self.training_history['avg_losses'][-1]:.4f}")
        
        if self.training_history['val_metrics']:
            best_dice = max(m['dice'] for m in self.training_history['val_metrics'])
            best_dice_round = next(m['round'] for m in self.training_history['val_metrics'] if m['dice'] == best_dice)
            print(f"\n训练过程中的最佳验证指标:")
            print(f"  Dice 系数: {best_dice:.4f} (第 {best_dice_round} 轮)")
        
        print(f"\n文件保存位置:")
        print(f"  - 模型文件: {save_path}")
        print(f"  - 训练历史: {history_path}")
        print("=" * 60)

    def save_checkpoint(self, round_num: int) -> Path:
        """保存训练检查点"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'round': round_num,
            'server_state': self.server.get_state_dict(),
            'training_history': self.training_history,
            'client_states': self.client_states,
            'config': self.config.to_dict(),
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"  [OK] 检查点已保存: {checkpoint_path}")
        
        # 清理旧检查点
        keep_max = self.config.keep_max_checkpoints or self.config.keep_checkpoint_max
        if keep_max > 0:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_round_*.pth"),
                key=lambda x: int(x.stem.split('_')[-1]),
                reverse=True
            )
            for old_checkpoint in checkpoint_files[keep_max:]:
                try:
                    old_checkpoint.unlink()
                    print(f"  [OK] 已删除旧检查点: {old_checkpoint.name}")
                except Exception as e:
                    print(f"  ⚠ 删除旧检查点失败 ({old_checkpoint.name}): {e}")
        
        return checkpoint_path
    
    def plot_training_curves(self, current_round: int):
        """生成并保存训练曲线图"""
        plot_dir = self.checkpoint_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[绘图] 正在生成训练曲线（第 {current_round} 轮）...")
        
        try:
            rounds_x = self.training_history['rounds']
            
            if not rounds_x:
                print("  ⚠ 没有训练数据可绘制")
                return
            
            # 1. 损失曲线
            plt.figure(figsize=(12, 6))
            plt.plot(rounds_x, self.training_history['avg_losses'], label='Train Loss', marker='o', linewidth=2)
            
            if self.training_history.get('avg_seg_losses'):
                plt.plot(rounds_x, self.training_history['avg_seg_losses'], label='Seg Loss', linestyle=':', marker='s', linewidth=1.5)
            
            if self.training_history.get('avg_cream_losses'):
                plt.plot(rounds_x, self.training_history['avg_cream_losses'], label='Cream Loss', linestyle='--', marker='^', linewidth=1.5)
            
            # 添加验证损失
            if self.training_history.get('val_metrics'):
                val_rounds = [m['round'] for m in self.training_history['val_metrics']]
                val_losses = [m.get('val_loss', 0.0) for m in self.training_history['val_metrics']]
                if val_losses and any(l > 0 for l in val_losses):
                    plt.plot(val_rounds, val_losses, label='Val Loss', marker='x', linestyle='-.', linewidth=2, color='red')
            
            plt.title(f'Training Loss Curves (Round {current_round})', fontsize=14, fontweight='bold')
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / 'loss_curve.png', dpi=150)
            plt.close()
            print(f"  [OK] 损失曲线已保存: {plot_dir / 'loss_curve.png'}")
            
            # 2. 验证指标曲线（Dice & IoU）
            if self.training_history['val_metrics']:
                val_rounds = [m['round'] for m in self.training_history['val_metrics']]
                val_dice = [m['dice'] for m in self.training_history['val_metrics']]
                val_iou = [m['iou'] for m in self.training_history['val_metrics']]
                
                plt.figure(figsize=(12, 6))
                plt.plot(val_rounds, val_dice, label='Dice Score', marker='s', linewidth=2, color='blue')
                plt.plot(val_rounds, val_iou, label='IoU Score', marker='^', linewidth=2, color='green')
                plt.title(f'Validation Metrics: Dice & IoU (Round {current_round})', fontsize=14, fontweight='bold')
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.ylim([0, 1.0])
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'metrics_dice_iou.png', dpi=150)
                plt.close()
                print(f"  [OK] Dice/IoU 曲线已保存: {plot_dir / 'metrics_dice_iou.png'}")
                
                # 3. HD95 曲线
                val_hd95 = [m['hd95'] for m in self.training_history['val_metrics']]
                
                if val_hd95:
                    plt.figure(figsize=(12, 6))
                    plt.plot(
                        val_rounds,
                        val_hd95,
                        label='HD95', color='red', marker='x', linewidth=2
                    )
                    plt.title(f'Validation Metric: HD95 (Round {current_round})', fontsize=14, fontweight='bold')
                    plt.xlabel('Round', fontsize=12)
                    plt.ylabel('HD95 (pixel)', fontsize=12)
                    plt.legend(loc='best', fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plot_dir / 'metrics_hd95.png', dpi=150)
                    plt.close()
                    print(f"  [OK] HD95 曲线已保存: {plot_dir / 'metrics_hd95.png'}")
                else:
                    print("  [WARN] No validation HD95 history; curve was not generated")
                    
        except Exception as e:
            print(f"  [FAIL] 绘图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_segmentation_masks(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: Path,
        max_samples: int = 50
    ) -> Path:
        """保存分割掩码到文件"""
        save_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        thresholds = getattr(self.config, "segmentation_thresholds", None)
        if thresholds is None:
            raise ValueError("segmentation thresholds are required to save masks")
        
        saved_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if saved_count >= max_samples:
                    break
                    
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, masks = batch[0], batch[1]
                else:
                    images = batch
                    masks = None
                
                images = images.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                else:
                    continue
                
                # 前向传播
                pred = model(images)
                
                # 处理模型返回字典的情况
                if isinstance(pred, dict):
                    pred = pred.get('logits', pred.get('out', list(pred.values())[0]))

                pred_labels = logits_to_brats_labels(
                    pred,
                    thresholds=thresholds,
                    channel_dim=1,
                )
                true_labels = regions_to_brats_labels(masks, channel_dim=1)
                
                # 处理每个样本
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if saved_count >= max_samples:
                        break
                    
                    pred_mask_uint8 = pred_labels[i].cpu().numpy().astype(np.uint8)
                    true_mask_uint8 = true_labels[i].cpu().numpy().astype(np.uint8)
                    
                    # 保存预测掩码
                    pred_img = Image.fromarray(pred_mask_uint8, mode='L')
                    pred_path = save_dir / f"pred_mask_{saved_count:04d}.png"
                    pred_img.save(pred_path)
                    
                    # 保存真实掩码
                    true_img = Image.fromarray(true_mask_uint8, mode='L')
                    true_path = save_dir / f"true_mask_{saved_count:04d}.png"
                    true_img.save(true_path)
                    
                    saved_count += 1
        
        return save_dir
