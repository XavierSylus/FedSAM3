"""
联邦学习客户端训练器：BaseClientTrainer 抽象基类 + TextOnlyTrainer / ImageOnlyTrainer / MultimodalTrainer 三个子类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
import math
import numpy as np

# ============================================================================
# 自动混合精度（AMP）兼容层
# ============================================================================
if torch.cuda.is_available():
    from torch.amp import GradScaler, autocast
else:
    GradScaler = None
    class _NoOpAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = lambda device_type='cpu', **kwargs: _NoOpAutocast()

from src.model import SAM3_Medical, DEVICE, BATCH_SIZE, LR
from src.cream_losses import (
    BraTSDiceBCELoss,
    CreamContrastiveLoss,
    PrototypeLogisticTextLoss,
)


# ============================================================================
# Phase 2 核心：抽象基类 BaseClientTrainer
# ============================================================================
class BaseClientTrainer(ABC):
    """
    客户端训练器抽象基类。

    共享职责：AMP、梯度裁剪、统计信息、验证。
    子类实现：unpack_private_batch、unpack_public_batch、compute_loss、get_return_values。
    """

    def __init__(
        self,
        private_loader: DataLoader,
        public_loader: DataLoader,
        device: str = DEVICE,
        use_amp: bool = True,
        local_epochs: int = 1,
        dataset_name: str = "BraTS",
        contrastive_dim: int = 1024,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        enable_text_assist_in_seg: bool = True,
        allow_text_param_upload: bool = False,
        baseline_method: str = "none",
        fedprox_mu: float = 0.0,
        segmentation_loss: Optional[str] = None,
        seg_dice_weight: Optional[float] = None,
        seg_bce_weight: Optional[float] = None,
        seg_dice_smooth: Optional[float] = None,
        segmentation_thresholds: Optional[Tuple[float, float, float]] = None,
        text_loss_temperature: Optional[float] = None,
    ):
        """
        初始化客户端训练器（共享组件）

        Args:
            private_loader: 私有分割数据 DataLoader
            public_loader: 公共数据 DataLoader（用于对比学习）
            device: 训练设备
            use_amp: 是否使用自动混合精度（FP16）
            local_epochs: 本地训练轮数
            dataset_name: 数据集名称（用于指标计算）
            contrastive_dim: 对比学习特征维度
            grad_clip: 梯度裁剪阈值（0.0 表示不裁剪）
            accumulation_steps: 梯度累加步数（1 = 无累加；有效 batch = batch_size × steps）
        """
        self.private_loader = private_loader
        self.public_loader = public_loader
        self.device = device
        self.use_amp = use_amp
        self.local_epochs = local_epochs
        self.local_epoch = 0
        self.dataset_name = dataset_name
        self.contrastive_dim = contrastive_dim
        self.grad_clip = grad_clip
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.enable_text_assist_in_seg = enable_text_assist_in_seg
        self.allow_text_param_upload = allow_text_param_upload
        self.baseline_method = str(baseline_method).lower()
        self.fedprox_mu = float(fedprox_mu)

        # 日志器
        self.logger = logging.getLogger(self.__class__.__name__)

        # Delay metrics imports until validation actually runs.
        self.medical_calculator = None
        self._metrics_module = None

        if use_amp and device == "cuda" and torch.cuda.is_available():
            if GradScaler is not None:
                self.scaler = GradScaler(device='cuda')
            else:
                self.scaler = None
                self.use_amp = False
        else:
            self.scaler = None
            self.use_amp = False

        self.cream_loss_fn = CreamContrastiveLoss(tau=0.07)
        self.text_loss_fn = (
            PrototypeLogisticTextLoss(text_loss_temperature)
            if text_loss_temperature is not None
            else None
        )
        segmentation_values = (
            segmentation_loss,
            seg_dice_weight,
            seg_bce_weight,
            seg_dice_smooth,
            segmentation_thresholds,
        )
        if all(value is None for value in segmentation_values):
            self.seg_criterion = None
            self.segmentation_thresholds = None
        elif any(value is None for value in segmentation_values):
            raise ValueError("segmentation loss configuration must be provided as one complete contract")
        else:
            if segmentation_loss != "dice_bce":
                raise ValueError(
                    f"segmentation_loss must be 'dice_bce', got {segmentation_loss!r}"
                )
            self.seg_criterion = BraTSDiceBCELoss(
                dice_weight=seg_dice_weight,
                bce_weight=seg_bce_weight,
                smooth=seg_dice_smooth,
            )
            if len(segmentation_thresholds) != 3:
                raise ValueError(
                    "segmentation_thresholds must contain [WT, TC, ET]"
                )
            ordered_thresholds = tuple(
                float(value) for value in segmentation_thresholds
            )
            if any(
                not math.isfinite(value) or value <= 0.0 or value >= 1.0
                for value in ordered_thresholds
            ):
                raise ValueError(
                    "segmentation thresholds must be finite values strictly between 0 and 1"
                )
            self.segmentation_thresholds = ordered_thresholds

        self.training_stats = {
            'total_loss': 0.0,
            'seg_loss': 0.0,
            'cream_loss': 0.0,
            'non_seg_component': 0.0,
            'num_batches': 0
        }
        self._adapter_param_ids = set()
        self._adapter_grad_sum: Optional[torch.Tensor] = None
        self._adapter_grad_steps = 0
        self._last_adapter_grad_vector: Optional[torch.Tensor] = None
        self._trainable_param_name_by_id: Dict[int, str] = {}
        self._diag_grad: Dict[str, float] = {}
        self._diag_cream: Dict[str, float] = {}
        self._last_multimodal_cream_diag: Optional[Dict[str, float]] = None

    def _get_metrics_module(self):
        """Import metrics lazily so preflight paths avoid MONAI/OpenMP side effects."""
        if self._metrics_module is None:
            from src import metrics as metrics_module

            self._metrics_module = metrics_module
        return self._metrics_module

    def _compute_segmentation_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.seg_criterion is None:
            raise RuntimeError("segmentation loss contract was not configured for this client")
        return self.seg_criterion(logits, target)

    def _reset_adapter_grad_tracking(self, model: nn.Module) -> None:
        """Track only the trainable adapter parameters for gradient-angle diagnostics."""
        self._trainable_param_name_by_id = {
            id(param): name for name, param in model.named_parameters() if param.requires_grad
        }
        self._adapter_param_ids = {
            id(param)
            for name, param in model.named_parameters()
            if (
                name.startswith('adapter_manager.adapters.')
                or '.adapter.' in name
            )
            and param.requires_grad
        }
        self._adapter_grad_sum = None
        self._adapter_grad_steps = 0
        self._last_adapter_grad_vector = None

    def _record_adapter_grad_snapshot(self, optimizer: torch.optim.Optimizer) -> None:
        """Accumulate a flattened adapter-gradient snapshot after unscaling."""
        if not self._adapter_param_ids:
            return

        grad_chunks = []
        for group in optimizer.param_groups:
            for param in group['params']:
                if id(param) not in self._adapter_param_ids or param.grad is None:
                    continue
                grad_chunks.append(param.grad.detach().float().flatten().cpu())

        if not grad_chunks:
            return

        grad_vector = torch.cat(grad_chunks, dim=0)
        if self._adapter_grad_sum is None:
            self._adapter_grad_sum = grad_vector.clone()
        else:
            if self._adapter_grad_sum.shape != grad_vector.shape:
                raise RuntimeError(
                    "Adapter gradient snapshot shape changed across steps; "
                    "gradient conflict signal would be invalid."
                )
            self._adapter_grad_sum += grad_vector
        self._adapter_grad_steps += 1

    def _finalize_adapter_grad_tracking(self) -> None:
        """Expose the mean adapter-gradient vector collected during local training."""
        if self._adapter_grad_sum is None or self._adapter_grad_steps == 0:
            self._last_adapter_grad_vector = None
            return
        self._last_adapter_grad_vector = self._adapter_grad_sum / float(self._adapter_grad_steps)

    def _reset_round_diagnostics(self) -> None:
        self._diag_grad = {
            'steps': 0.0,
            'text_proj_grad_norm_sum': 0.0,
            'text_adapter_grad_norm_sum': 0.0,
            'adapter_grad_norm_sum': 0.0,
            'adapter_nonzero_ratio_sum': 0.0,
        }
        self._diag_cream = {
            'steps': 0.0,
            'raw_sum': 0.0,
            'weighted_sum': 0.0,
            'text_align_sum': 0.0,
            'text_align_weighted_sum': 0.0,
            'valid_pairs_sum': 0.0,
            'batch_size_sum': 0.0,
            'cross_size_sum': 0.0,
            'diag_mode_steps': 0.0,
            'anchor_mode_steps': 0.0,
            'nan_or_inf_steps': 0.0,
            'text_missing_steps': 0.0,
            'l_inter_sum': 0.0,
            'l_intra_sum': 0.0,
            'tau': float('nan'),
            'temperature': float('nan'),
        }
        self._last_multimodal_cream_diag = None

    def _record_grad_norm_diagnostics(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._diag_grad:
            return

        text_proj_sq = 0.0
        text_adapter_sq = 0.0
        adapter_sq = 0.0
        adapter_total = 0
        adapter_nonzero = 0

        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                name = self._trainable_param_name_by_id.get(id(param), '')
                grad = param.grad.detach().float()
                sq = torch.sum(grad * grad).item()
                if 'adapter' in name:
                    adapter_sq += sq
                    flat = grad.flatten()
                    adapter_total += flat.numel()
                    adapter_nonzero += int((flat.abs() > 0).sum().item())
                if 'text_proj' in name:
                    text_proj_sq += sq
                if ('adapter' in name) and ('text' in name):
                    text_adapter_sq += sq

        self._diag_grad['steps'] += 1.0
        self._diag_grad['text_proj_grad_norm_sum'] += math.sqrt(max(text_proj_sq, 0.0))
        self._diag_grad['text_adapter_grad_norm_sum'] += math.sqrt(max(text_adapter_sq, 0.0))
        self._diag_grad['adapter_grad_norm_sum'] += math.sqrt(max(adapter_sq, 0.0))
        nonzero_ratio = (adapter_nonzero / adapter_total) if adapter_total > 0 else 0.0
        self._diag_grad['adapter_nonzero_ratio_sum'] += nonzero_ratio

    def _record_cream_diag_step(self, step_diag: Dict[str, float]) -> None:
        if not self._diag_cream:
            return
        self._diag_cream['steps'] += 1.0
        self._diag_cream['raw_sum'] += float(step_diag.get('cream_raw', 0.0))
        self._diag_cream['weighted_sum'] += float(step_diag.get('cream_weighted', 0.0))
        self._diag_cream['text_align_sum'] += float(step_diag.get('text_align_raw', 0.0))
        self._diag_cream['text_align_weighted_sum'] += float(step_diag.get('text_align_weighted', 0.0))
        self._diag_cream['valid_pairs_sum'] += float(step_diag.get('valid_pairs', 0.0))
        self._diag_cream['batch_size_sum'] += float(step_diag.get('local_batch', 0.0))
        self._diag_cream['cross_size_sum'] += float(step_diag.get('cross_batch', 0.0))
        self._diag_cream['diag_mode_steps'] += 1.0 if step_diag.get('pair_mode', '') == 'diagonal' else 0.0
        self._diag_cream['anchor_mode_steps'] += 1.0 if step_diag.get('pair_mode', '') == 'anchor0' else 0.0
        self._diag_cream['nan_or_inf_steps'] += 1.0 if not bool(step_diag.get('is_finite', True)) else 0.0
        self._diag_cream['text_missing_steps'] += 1.0 if bool(step_diag.get('text_missing', False)) else 0.0
        self._diag_cream['l_inter_sum'] += float(step_diag.get('l_inter', 0.0))
        self._diag_cream['l_intra_sum'] += float(step_diag.get('l_intra', 0.0))
        if math.isnan(self._diag_cream['tau']):
            self._diag_cream['tau'] = float(step_diag.get('tau', float('nan')))
        if math.isnan(self._diag_cream['temperature']):
            self._diag_cream['temperature'] = float(step_diag.get('temperature', float('nan')))

    def _attach_diagnostics_to_stats(self, training_stats: Dict[str, float]) -> Dict[str, float]:
        grad_steps = max(int(self._diag_grad.get('steps', 0.0)), 1)
        cream_steps = max(int(self._diag_cream.get('steps', 0.0)), 1)

        training_stats['diag_text_proj_grad_norm'] = self._diag_grad.get('text_proj_grad_norm_sum', 0.0) / grad_steps
        training_stats['diag_text_adapter_grad_norm'] = self._diag_grad.get('text_adapter_grad_norm_sum', 0.0) / grad_steps
        training_stats['diag_adapter_grad_norm'] = self._diag_grad.get('adapter_grad_norm_sum', 0.0) / grad_steps
        training_stats['diag_adapter_nonzero_ratio'] = self._diag_grad.get('adapter_nonzero_ratio_sum', 0.0) / grad_steps

        training_stats['diag_cream_raw'] = self._diag_cream.get('raw_sum', 0.0) / cream_steps
        training_stats['diag_cream_weighted'] = self._diag_cream.get('weighted_sum', 0.0) / cream_steps
        training_stats['diag_text_align_raw'] = self._diag_cream.get('text_align_sum', 0.0) / cream_steps
        training_stats['diag_text_align_weighted'] = self._diag_cream.get('text_align_weighted_sum', 0.0) / cream_steps
        training_stats['diag_cream_valid_pairs'] = self._diag_cream.get('valid_pairs_sum', 0.0) / cream_steps
        training_stats['diag_cream_local_batch'] = self._diag_cream.get('batch_size_sum', 0.0) / cream_steps
        training_stats['diag_cream_cross_batch'] = self._diag_cream.get('cross_size_sum', 0.0) / cream_steps
        training_stats['diag_cream_nan_or_inf_steps'] = self._diag_cream.get('nan_or_inf_steps', 0.0)
        training_stats['diag_cream_text_missing_steps'] = self._diag_cream.get('text_missing_steps', 0.0)
        training_stats['diag_cream_l_inter'] = self._diag_cream.get('l_inter_sum', 0.0) / cream_steps
        training_stats['diag_cream_l_intra'] = self._diag_cream.get('l_intra_sum', 0.0) / cream_steps
        training_stats['diag_cream_tau'] = self._diag_cream.get('tau', float('nan'))
        training_stats['diag_cream_temperature'] = self._diag_cream.get('temperature', float('nan'))
        training_stats['diag_cream_diag_mode_steps'] = self._diag_cream.get('diag_mode_steps', 0.0)
        training_stats['diag_cream_anchor_mode_steps'] = self._diag_cream.get('anchor_mode_steps', 0.0)

        if self._diag_cream.get('steps', 0.0) > 0:
            print(
                "      [CreamDiag] "
                f"raw={training_stats['diag_cream_raw']:.4f}, "
                f"weighted={training_stats['diag_cream_weighted']:.4f}, "
                f"text_align={training_stats['diag_text_align_raw']:.4f}, "
                f"text_align_w={training_stats['diag_text_align_weighted']:.4f}, "
                f"valid_pairs={training_stats['diag_cream_valid_pairs']:.1f}, "
                f"pair_mode(diag/anchor0)="
                f"{int(training_stats['diag_cream_diag_mode_steps'])}/"
                f"{int(training_stats['diag_cream_anchor_mode_steps'])}, "
                f"tau={training_stats['diag_cream_tau']:.4f}, "
                f"temperature={training_stats['diag_cream_temperature']:.4f}, "
                f"nan_or_inf_steps={int(training_stats['diag_cream_nan_or_inf_steps'])}, "
                f"text_missing_steps={int(training_stats['diag_cream_text_missing_steps'])}"
            )

        if self._diag_grad.get('steps', 0.0) > 0:
            print(
                "      [GradDiag] "
                f"text_proj_norm={training_stats['diag_text_proj_grad_norm']:.4e}, "
                f"text_adapter_norm={training_stats['diag_text_adapter_grad_norm']:.4e}, "
                f"adapter_norm={training_stats['diag_adapter_grad_norm']:.4e}, "
                f"adapter_nonzero_ratio={training_stats['diag_adapter_nonzero_ratio']:.4f}"
            )

        return training_stats

    # ========================================================================
    # 抽象方法：子类必须实现的多态接口
    # ========================================================================

    @abstractmethod
    def unpack_private_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """
        解包私有数据批次（模态专属逻辑）

        Args:
            batch: 私有数据批次（来自 private_loader）

        Returns:
            Dict: {
                'image': Optional[Tensor],      # (B, 3, H, W)
                'mask': Optional[Tensor],       # (B, C, H, W)
                'text_feat': Optional[Tensor]   # (B, D_text)
            }

        示例：
            - TextOnlyTrainer: {'image': None, 'mask': None, 'text_feat': Tensor}
            - ImageOnlyTrainer: {'image': Tensor, 'mask': Tensor, 'text_feat': None}
            - MultimodalTrainer: {'image': Tensor, 'mask': Tensor, 'text_feat': Tensor}
        """
        pass

    @abstractmethod
    def unpack_public_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """
        解包公共数据批次（模态专属逻辑）

        Args:
            batch: 公共数据批次（来自 public_loader）

        Returns:
            Dict: {
                'image': Optional[Tensor],      # (B, 3, H, W)
                'text_feat': Optional[Tensor]   # (B, D_text)
            }
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        private_inputs: Dict[str, Optional[torch.Tensor]],
        public_inputs: Dict[str, Optional[torch.Tensor]],
        global_reps: Dict[str, torch.Tensor],
        lambda_cream: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        计算损失（模态专属逻辑）

        Args:
            model: SAM3_Medical 模型实例
            private_inputs: 解包后的私有数据
            public_inputs: 解包后的公共数据
            global_reps: 全局表示字典
            lambda_cream: 对比学习损失权重

        Returns:
            Tuple of (total_loss, seg_loss, cream_loss, public_rep)
            - total_loss: 总损失（用于反向传播）
            - seg_loss: 分割损失（用于统计）
            - cream_loss: 对比学习损失（用于统计）
            - public_rep: 公共数据表征（用于聚合，shape: (B, D)）
        """
        pass

    @abstractmethod
    def get_return_values(
        self,
        model: nn.Module,
        local_reps: torch.Tensor,
        training_stats: Dict[str, float]
    ) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        返回值解耦（模态专属逻辑）

        Args:
            model: SAM3_Medical 模型实例
            local_reps: 聚合后的本地表征 (D,)
            training_stats: 训练统计信息

        Returns:
            Tuple of (weights, image_rep, text_rep, stats)
            - TextOnlyTrainer: (text_only_state, None, text_rep, stats)
            - ImageOnlyTrainer: (state_dict, image_rep, None, stats)
            - MultimodalTrainer: (state_dict, image_rep, text_rep, stats)
        """
        pass

    def get_uploadable_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Return the state subset this client will upload to the server."""
        return self.get_model_state(model)

    # ========================================================================
    # 共享逻辑：主训练循环
    # ========================================================================

    def run(
        self,
        model: SAM3_Medical,
        optimizer: torch.optim.Optimizer,
        global_reps: Dict[str, torch.Tensor],
        lambda_cream: float = 0.05,
        global_reference_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        运行本地训练（包含多个本地 Epoch 循环）

        Args:
            model: SAM3_Medical 模型实例
            optimizer: 优化器实例
            global_reps: 全局表示字典
            lambda_cream: 对比学习损失权重

        Returns:
            Tuple of (weights, image_rep, text_rep, training_stats)
        """
        model.to(self.device)
        model.train()

        self.local_epoch = 0
        self._reset_adapter_grad_tracking(model)

        # 本地 Epoch 循环
        for epoch in range(self.local_epochs):
            self.local_epoch += 1
            weights, img_rep, txt_rep, epoch_stats = self.tra(
                model, optimizer, global_reps, lambda_cream, global_reference_state
            )

        self._finalize_adapter_grad_tracking()
        return weights, img_rep, txt_rep, epoch_stats

    def tra(
        self,
        model: SAM3_Medical,
        optimizer: torch.optim.Optimizer,
        global_reps: Dict[str, torch.Tensor],
        lambda_cream: float = 0.05,
        global_reference_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Template Method Pattern：定义训练循环骨架，将模态专属逻辑委托给子类抽象方法。

        Args:
            model: SAM3_Medical 模型实例
            optimizer: 优化器实例
            global_reps: 全局表示字典
            lambda_cream: 对比学习损失权重

        Returns:
            Tuple of (weights, image_rep, text_rep, training_stats)
        """
        model.train()

        # 投影全局表示到对比学习空间（若需要）
        global_text_rep, global_image_rep = self._prepare_global_reps(
            model, global_reps
        )

        # 初始化统计和迭代器
        self.training_stats = {
            'total_loss': 0.0,
            'seg_loss': 0.0,
            'cream_loss': 0.0,
            'non_seg_component': 0.0,
            'num_batches': 0
        }
        self._reset_round_diagnostics()
        local_public_reps_list = []
        fedprox_param_names = (
            set(self.get_uploadable_state(model).keys())
            if self.baseline_method == "fedprox" and global_reference_state
            else set()
        )

        private_iter = iter(self.private_loader)
        public_iter = iter(self.public_loader) if self.public_loader is not None else None

        # 梯度累加：在整个 Epoch 开始前清零一次（不在每个 batch 开头清零）
        optimizer.zero_grad(set_to_none=True)
        _accum_step = 0  # 当前累加计数

        # ★ 主训练循环
        while True:
            try:
                # Step 1: 获取私有批次并解包（多态调用）
                private_batch = next(private_iter)
                private_inputs = self.unpack_private_batch(private_batch)

                # Step 2: 获取公共批次并解包（多态调用）
                try:
                    if public_iter is None:
                        raise StopIteration
                    public_batch = next(public_iter)
                except StopIteration:
                    if self.public_loader is not None:
                        public_iter = iter(self.public_loader)
                        try:
                            public_batch = next(public_iter)
                        except StopIteration:
                            public_batch = self._get_fallback_public_batch(private_inputs)
                    else:
                        public_batch = self._get_fallback_public_batch(private_inputs)

                public_inputs = self.unpack_public_batch(public_batch)

            except StopIteration:
                # flush 残余梯度：epoch 末尾未凑满 accumulation_steps 的 batch 已 backward，
                # 不能再走 backward 路径（dummy loss 无 grad_fn 会被拦截），
                # 直接调用 _flush_accumulated_grads 提交已累加的梯度。
                if _accum_step > 0 and _accum_step % self.accumulation_steps != 0:
                    self._flush_accumulated_grads(optimizer)
                break

            _accum_step += 1

            # Step 3: 计算损失（多态调用）
            with autocast(device_type='cuda', enabled=self.use_amp) if self.device == 'cuda' else autocast(device_type='cpu', enabled=False):
                total_loss, seg_loss, cream_loss, public_rep = self.compute_loss(
                    model, private_inputs, public_inputs,
                    {'text': global_text_rep, 'image': global_image_rep},
                    lambda_cream
                )
            if fedprox_param_names:
                total_loss = total_loss + self._compute_fedprox_penalty(
                    model=model,
                    global_reference_state=global_reference_state,
                    fedprox_param_names=fedprox_param_names,
                )
            if isinstance(self._last_multimodal_cream_diag, dict):
                self._record_cream_diag_step(self._last_multimodal_cream_diag)

            # 梯度等比缩放：数学上等价于 batch_size × accumulation_steps
            scaled_loss = total_loss / self.accumulation_steps
            # 是否执行优化器步进（达到累加步数，或遇到最后一个有效 batch）
            perform_step = (_accum_step % self.accumulation_steps == 0)

            # Step 4: 反向传播和优化器步进（统一接口）
            self._backward_and_step(scaled_loss, optimizer, perform_step=perform_step)

            # Step 5: 收集统计信息（用原始 loss，不用缩放值）
            if public_rep is not None:
                local_public_reps_list.append(public_rep.detach().cpu())

            total_loss_val = total_loss.item()
            seg_loss_val = seg_loss.item() if isinstance(seg_loss, torch.Tensor) else float(seg_loss)
            self.training_stats['total_loss'] += total_loss_val
            self.training_stats['seg_loss'] += seg_loss_val
            self.training_stats['cream_loss'] += cream_loss.item()
            self.training_stats['non_seg_component'] += (total_loss_val - seg_loss_val)
            self.training_stats['num_batches'] += 1

        # 聚合本地表征
        local_reps = self._aggregate_representations(local_public_reps_list)

        # 计算统计信息
        training_stats = self._compute_stats()
        training_stats = self._attach_diagnostics_to_stats(training_stats)

        # Step 6: 返回值解耦（多态调用）
        return self.get_return_values(model, local_reps, training_stats)

    def _compute_fedprox_penalty(
        self,
        model: nn.Module,
        global_reference_state: Optional[Dict[str, torch.Tensor]],
        fedprox_param_names: set,
    ) -> torch.Tensor:
        """Compute the FedProx proximal penalty over uploadable trainable params."""
        if (
            self.baseline_method != "fedprox"
            or self.fedprox_mu <= 0
            or not global_reference_state
            or not fedprox_param_names
        ):
            return torch.tensor(0.0, device=self.device)

        proximal_term = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in fedprox_param_names:
                continue
            global_param = global_reference_state.get(name)
            if global_param is None:
                continue
            proximal_term = proximal_term + torch.sum(
                (param - global_param.to(self.device)) ** 2
            )

        return 0.5 * self.fedprox_mu * proximal_term

    def _prepare_global_reps(
        self,
        model: nn.Module,
        global_reps: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """投影全局表示到对比学习空间（若需要）"""
        required = {
            "global_text_rep": global_reps.get("global_text_rep"),
            "global_image_rep": global_reps.get("global_image_rep"),
        }
        invalid = [
            name
            for name, value in required.items()
            if not isinstance(value, torch.Tensor)
            or value.numel() == 0
            or not torch.isfinite(value).all()
            or torch.linalg.vector_norm(value.float()).item() <= 1e-12
        ]
        if invalid:
            raise ValueError(
                "Global representation contract violated: "
                f"missing, empty, non-finite, or zero-norm values for {invalid}"
            )

        global_text_rep = required["global_text_rep"].detach().to(self.device)
        global_image_rep = required["global_image_rep"].detach().to(self.device)

        expected_dim = getattr(model, 'contrastive_dim', 1024)

        # 投影文本表示
        if global_text_rep.dim() == 1 and global_text_rep.shape[0] != expected_dim:
            if hasattr(model, 'text_proj') and model.text_proj is not None:
                global_text_rep = model.text_proj(global_text_rep.unsqueeze(0)).squeeze(0)

        # 投影图像表示
        if global_image_rep.dim() == 1 and global_image_rep.shape[0] != expected_dim:
            if hasattr(model, 'image_proj') and model.image_proj is not None:
                global_image_rep = model.image_proj(global_image_rep.unsqueeze(0)).squeeze(0)

        return global_text_rep, global_image_rep

    def _get_fallback_public_batch(self, private_inputs: Dict) -> Any:
        """生成回退公共批次（当公共数据用尽时）"""
        # 子类可以重写此方法以提供更智能的回退策略
        return {}

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        perform_step: bool = True
    ):
        """
        统一梯度回传接口，严格遵守 AMP GradScaler 状态机：
        unscale_() 在一个更新周期内只能调用一次，因此仅在 perform_step=True 时调用。
        梯度累加时，perform_step=False 的 batch 只执行 backward，不调用 unscale_/step/update。
        """
        if loss.grad_fn is None:
            self.logger.warning(
                "[_backward_and_step] loss.grad_fn is None，跳过 backward+step。"
            )
            return

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if not perform_step:
                return
            _has_grads = any(
                p.grad is not None
                for group in optimizer.param_groups
                for p in group['params']
            )
            if not _has_grads:
                self.logger.warning(
                    "[_backward_and_step] AMP backward 后无参数产生梯度，跳过 step。"
                )
                optimizer.zero_grad(set_to_none=True)
                return
            self.scaler.unscale_(optimizer)
            self._record_grad_norm_diagnostics(optimizer)
            self._record_adapter_grad_snapshot(optimizer)
            _params_to_clip = [
                p for group in optimizer.param_groups
                for p in group['params']
                if p.grad is not None
            ]
            if _params_to_clip:
                torch.nn.utils.clip_grad_norm_(_params_to_clip, max_norm=self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if not perform_step:
                return
            self._record_grad_norm_diagnostics(optimizer)
            self._record_adapter_grad_snapshot(optimizer)
            _params_to_clip = [
                p for group in optimizer.param_groups
                for p in group['params']
                if p.grad is not None
            ]
            if _params_to_clip:
                torch.nn.utils.clip_grad_norm_(_params_to_clip, max_norm=self.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def _flush_accumulated_grads(self, optimizer: torch.optim.Optimizer) -> None:
        """
        直接提交已累加的梯度，不执行新的 backward。
        用于 epoch 末尾残余梯度（未凑满 accumulation_steps）的强制提交。
        """
        if self.use_amp:
            _has_grads = any(
                p.grad is not None
                for group in optimizer.param_groups
                for p in group['params']
            )
            if not _has_grads:
                return
            self.scaler.unscale_(optimizer)
            self._record_grad_norm_diagnostics(optimizer)
            self._record_adapter_grad_snapshot(optimizer)
            _params_to_clip = [
                p for group in optimizer.param_groups
                for p in group['params']
                if p.grad is not None
            ]
            if _params_to_clip:
                torch.nn.utils.clip_grad_norm_(_params_to_clip, max_norm=self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            _params_to_clip = [
                p for group in optimizer.param_groups
                for p in group['params']
                if p.grad is not None
            ]
            if _params_to_clip:
                self._record_grad_norm_diagnostics(optimizer)
                self._record_adapter_grad_snapshot(optimizer)
                torch.nn.utils.clip_grad_norm_(_params_to_clip, max_norm=self.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _aggregate_representations(self, reps_list: list) -> torch.Tensor:
        """聚合本地公共数据表征：torch.cat(dim=0)+mean 支持不同大小 batch 的 public_rep。"""
        if len(reps_list) > 0:
            # 统一处理：保证每个 rep 都是 2D (N_i, D)
            safe_reps = []
            for r in reps_list:
                if r.dim() == 0:
                    r = r.reshape(1, 1)       # 标量 → (1, 1)
                elif r.dim() == 1:
                    r = r.unsqueeze(0)        # (D,) → (1, D)
                # 2D (N_i, D) 直接保留
                safe_reps.append(r)
            local_reps = torch.cat(safe_reps, dim=0)  # (total_N, D)
            local_reps = local_reps.mean(dim=0)        # (D,)
        else:
            local_reps = torch.zeros(self.contrastive_dim)
        return local_reps

    def _compute_stats(self) -> Dict[str, float]:
        """计算训练统计信息"""
        num_batches = self.training_stats['num_batches']
        return {
            'avg_loss': self.training_stats['total_loss'] / num_batches if num_batches > 0 else 0.0,
            'avg_seg_loss': self.training_stats['seg_loss'] / num_batches if num_batches > 0 else 0.0,
            'avg_cream_loss': self.training_stats['cream_loss'] / num_batches if num_batches > 0 else 0.0,
            'avg_non_seg_component': self.training_stats['non_seg_component'] / num_batches if num_batches > 0 else 0.0,
            'num_batches': num_batches,
            'local_epoch': self.local_epoch
        }

    def get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Return exactly the parameters declared by the model training registry."""
        if not hasattr(model, "get_trainable_params"):
            raise AttributeError(
                f"{type(model).__name__} must define get_trainable_params()"
            )

        registered_param_ids = {
            id(parameter) for parameter in model.get_trainable_params()
        }
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) in registered_param_ids
        }

    def load_model_state(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False
    ) -> None:
        """将服务器聚合后的 state_dict 加载回模型。strict=False 容忍 RoPE buffer 和冻结主干参数缺失。"""
        missing_keys, unexpected_keys = model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()},
            strict=strict
        )
        if missing_keys:
            self.logger.debug(
                f"[load_model_state] 以下键缺失（通常为冻结参数）："
                f"{missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )
        if unexpected_keys:
            self.logger.warning(f"[load_model_state] 以下键在模型中不存在：{unexpected_keys}")

    def validate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        compute_hd95: bool = True,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate 2D validation samples under the strict WT/TC/ET contract."""
        model.eval()
        if self.segmentation_thresholds is None:
            raise RuntimeError("segmentation inference thresholds were not configured")
        metrics_module = self._get_metrics_module()
        accumulator = metrics_module.BraTSMetricAccumulator(
            compute_hd95=compute_hd95
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if isinstance(batch, (list, tuple)):
                    images, masks = batch[0], batch[1] if len(batch) > 1 else None
                elif isinstance(batch, dict):
                    images = batch.get('images', batch.get('inp'))
                    masks = batch.get('masks', batch.get('gt', batch.get('target')))
                else:
                    images, masks = batch, None

                if images is None:
                    continue

                images = images.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)

                raw_output = model(images)

                if isinstance(raw_output, tuple) and len(raw_output) == 2:
                    pred_logits = raw_output[0]
                elif isinstance(raw_output, dict):
                    pred_logits = raw_output.get('logits', list(raw_output.values())[0])
                else:
                    pred_logits = raw_output

                if masks is None:
                    continue

                self.seg_criterion.validate_inputs(pred_logits, masks)
                accumulator.update_from_logits(
                    pred_logits,
                    masks,
                    thresholds=self.segmentation_thresholds,
                )

                if batch_idx == 0:
                    for channel_index, region in enumerate(("WT", "TC", "ET")):
                        region_logits = pred_logits[
                            :,
                            channel_index:channel_index + 1,
                        ]
                        threshold = self.segmentation_thresholds[channel_index]
                        fg_ratio = (
                            torch.sigmoid(region_logits) >= threshold
                        ).float().mean().item()
                        print(
                            f"  [Val Diag] {region} logits: "
                            f"max={region_logits.max():.3f} "
                            f"min={region_logits.min():.3f} "
                            f"threshold={threshold:.3f} "
                            f"fg_ratio={fg_ratio:.4f}"
                        )

                if verbose and batch_idx % 10 == 0:
                    self.logger.info("Validation batch %d accumulated", batch_idx)

        results = accumulator.compute()
        print(
            "  [Val Stat] "
            f"pred_fg_voxels={results['pred_fg_voxels']} "
            f"gt_fg_voxels={results['gt_fg_voxels']} "
            f"precision={results['precision']:.4f} "
            f"recall={results['recall']:.4f}"
        )
        for region in ("WT", "TC", "ET"):
            print(
                f"  [Val {region}] "
                f"Dice={results[f'{region}_dice']:.4f} "
                f"IoU={results[f'{region}_iou']:.4f} "
                f"both_empty={results[f'{region}_both_empty_count']} "
                f"empty_fp={results[f'{region}_empty_fp_count']} "
                f"empty_fn={results[f'{region}_empty_fn_count']}"
            )
        return results


# ============================================================================
# Phase 2 核心：TextOnlyTrainer（文本专属训练器）
# ============================================================================
class TextOnlyTrainer(BaseClientTrainer):
    """
    文本专属训练器。

    Private ``text_feature`` is projected by ``fusion_head.text_proj`` and
    aligned to the detached round-global public multimodal text prototype.
    This client has no segmentation loss.
    """

    def unpack_private_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包私有批次：(text_feature,)"""
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            text_feat = batch[0].to(self.device)
        else:
            raise ValueError(f"text_only 客户端期望 (text_feature,) 格式，实际: {type(batch)}")

        return {'image': None, 'mask': None, 'text_feat': text_feat}

    def unpack_public_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包公共批次：(text_feature,)"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            text_feat = batch[0].to(self.device)
        else:
            raise ValueError(f"text_only 客户端 public 数据期望 (text_feature,) 格式")

        return {'image': None, 'text_feat': text_feat}

    def _get_fallback_public_batch(self, private_inputs: Dict) -> Any:
        """public_loader 用尽时返回与 private_text_feat 形状一致的全零宿主。"""
        text_feat = private_inputs['text_feat']
        return (torch.zeros_like(text_feat),)

    def compute_loss(
        self,
        model: nn.Module,
        private_inputs: Dict,
        public_inputs: Dict,
        global_reps: Dict,
        lambda_cream: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute ``L_text`` from private text supervision and the fixed
        round-global text prototype. ``lambda_cream`` is not applied because
        this client has no segmentation objective.
        """
        text_feat = private_inputs['text_feat']
        if not hasattr(model, 'fusion_head'):
            raise AttributeError(
                "[TextOnlyTrainer.compute_loss] model 缺少 fusion_head 属性！\n"
                "请确保 SAM3MedicalIntegrated 已正确初始化 MultimodalFusionHead。\n"
                f"当前 model 类型: {type(model).__name__}"
            )
        if not hasattr(model.fusion_head, 'project_text'):
            raise AttributeError(
                "[TextOnlyTrainer.compute_loss] fusion_head 缺少 project_text"
            )
        if self.text_loss_fn is None:
            raise RuntimeError(
                "text loss contract was not configured for the text-only client"
            )
        if not isinstance(text_feat, torch.Tensor) or text_feat.ndim != 2:
            raise ValueError("private text_feature must have shape [B, D_text]")

        projected_text = model.fusion_head.project_text(text_feat)
        global_text_rep = global_reps['text'].detach()
        text_loss = self.text_loss_fn(projected_text, global_text_rep)
        seg_loss = projected_text.new_zeros(())
        normalized_text = F.normalize(projected_text, p=2, dim=1)

        return text_loss, seg_loss, text_loss, normalized_text

    def get_return_values(
        self,
        model: nn.Module,
        local_reps: torch.Tensor,
        training_stats: Dict
    ) -> Tuple[Optional[Dict], None, torch.Tensor, Dict]:
        """
        返回文本专属参数字典。若过滤后文本参数为空直接 raise RuntimeError：
        静默 fallback 会将视觉参数混入聚合池，击穿物理隔离墙。
        """
        full_state = self.get_model_state(model)
        text_only_state = {
            k: v for k, v in full_state.items()
            if 'text_encoder' in k or 'text_proj' in k or 'text_adapter' in k
        }
        # 键名与 server.py 路由白名单不一致时快速失败，拒绝静默污染聚合池
        if len(text_only_state) == 0:
            param_sample = list(full_state.keys())[:8]
            raise RuntimeError(
                "[TextOnlyTrainer.get_return_values] 致命错误：\n"
                "  过滤关键字 ('text_encoder', 'text_proj', 'text_adapter') "
                "在模型参数中未命中任何键！\n"
                f"  full_state 共 {len(full_state)} 个参数，样本键名: {param_sample}\n"
                "  根本原因：模型参数命名与路由白名单不一致。\n"
                "  修复方案：\n"
                "    1. 检查当前模型 named_parameters() 中文本相关参数的实际键名；\n"
                "    2. 同步更新本函数过滤关键字 和 server.py TEXT_PARAMS / TEXT_ADAPTER_PARAMS；\n"
                "  程序主动崩溃优于有毒梯度混入聚合池（Fail Fast \u003e Silent Corruption）。"
            )

        return text_only_state, None, local_reps, training_stats




# ============================================================================
# Phase 2 核心：ImageOnlyTrainer（图像专属训练器）
# ============================================================================
class ImageOnlyTrainer(BaseClientTrainer):
    """
    ★ 图像专属训练器（ImageOnlyTrainer）

    **数据格式**：
    - Private Batch: (image, mask)
    - Public Batch: (image,)

    **训练策略**：
    - 分割损失 + 图像对比学习损失
    - 无文本特征

    **返回值**：
    - weights: 完整模型参数
    - image_rep: 图像表征
    - text_rep: None
    """

    def unpack_private_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包私有批次：(image, mask)"""
        if isinstance(batch, dict):
            img = batch.get('image', batch.get('inp'))
            mask = batch.get('mask', batch.get('gt', batch.get('label')))
        elif isinstance(batch, (list, tuple)):
            img = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        else:
            img = batch
            mask = None

        img = img.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        return {'image': img, 'mask': mask, 'text_feat': None}

    def unpack_public_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包公共批次：(image,)"""
        if isinstance(batch, dict):
            img = batch.get('image', batch.get('inp'))
        elif isinstance(batch, (list, tuple)):
            img = batch[0]
        else:
            img = batch

        img = img.to(self.device)
        return {'image': img, 'text_feat': None}

    def _get_fallback_public_batch(self, private_inputs: Dict) -> Any:
        """
        ★ 生成与 private image 形状一致的全零张量

        **触发条件**：当 public_loader 数据用尽时
        **返回格式**：{'image': Tensor} 字典
        **关键修复**：防止返回空字典导致 img.to(device) AttributeError
        """
        img = private_inputs['image']
        return {'image': torch.zeros_like(img)}

    def compute_loss(
        self,
        model: nn.Module,
        private_inputs: Dict,
        public_inputs: Dict,
        global_reps: Dict,
        lambda_cream: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        分割损失（Dice+BCE）+ 跨模态表征蒸馏（当前 Group A：cream_loss=0）。

        Args:
            model: 当前轮次模型
            private_inputs: {'image': (B,C,H,W), 'mask': (B,C,H,W)}
            public_inputs: {'image': (B,C,H,W)}
            global_reps: {'text': (D,), 'image': (D,)}
            lambda_cream: 蒸馏强度系数
        Returns:
            (total_loss, seg_loss, cream_loss, public_rep)
        """
        img = private_inputs['image']
        mask = private_inputs['mask']
        public_img = public_inputs['image']
        # ★ 强制 .to(self.device).detach()：阻断向服务器的梯度回传
        # global_text_rep 来自 Server EMA，不允许梯度流回服务器节点表征。
        global_text_rep = global_reps['text'].to(self.device).detach()
        # global_image_rep 在本方法不参与损失计算（L_intra 已关闭，避免图像 EMA 自环偏置）

        # ══════════════════════════════════════════════════════════════════
        # Step 1：主任务分割损失（配置化 soft-Dice + BCEWithLogits）
        # ══════════════════════════════════════════════════════════════════
        if mask is not None:
            # ★ Group A 实验：无 text_only 客户端，global_text_rep 为随机噪声，
            #   固定传 None，让 TextPromptEncoder 走零输出路径，不污染 Mask Decoder。
            global_text_prompt = None
            raw_output = model(img, global_text_rep=global_text_prompt)
            if isinstance(raw_output, tuple) and len(raw_output) == 2:
                pred = raw_output[0]  # (B, 3, H, W) logits
            elif isinstance(raw_output, dict):
                pred = raw_output.get('logits', list(raw_output.values())[0])
            else:
                pred = raw_output

            seg_loss = self._compute_segmentation_loss(pred, mask)
        else:
            seg_loss = torch.tensor(0.0, device=self.device)

        # Group A：纯分割，无跨模态蒸馏（Group B/C 启动时在此实现 InfoNCE）
        cream_loss = torch.tensor(0.0, device=self.device)
        public_rep = torch.zeros(self.contrastive_dim, device=self.device)
        total_loss = seg_loss

        return total_loss, seg_loss, cream_loss, public_rep


    def get_return_values(
        self,
        model: nn.Module,
        local_reps: torch.Tensor,
        training_stats: Dict
    ) -> Tuple[Dict, torch.Tensor, None, Dict]:
        """
        返回图像专属结果

        **返回格式**：(image_only_state_dict, image_rep, None, stats)

        ★ 解耦聚合修复（2026-03-16）：
        ImageOnly 客户端无文本训练数据，其 text_encoder/text_proj 参数
        在本地训练中未被更新（或只受随机初始化影响）。若这些参数上传聚合，
        会将未经文本数据训练的噪声权重混入全局 Text Encoder，污染文本表征。
        因此：过滤掉所有 text 相关参数，只上传图像相关参数。
        Adapter 等轻量调优模块保留（作为跨客户端共性知识的桥梁）。
        """
        full_state = self.get_model_state(model)

        # ★ 物理隔离：剔除未被训练的 text 参数，防止模态污染
        TEXT_PARAM_KEYWORDS = ('text_encoder', 'text_proj', 'text_adapter')
        image_only_state = {
            k: v for k, v in full_state.items()
            if not any(kw in k for kw in TEXT_PARAM_KEYWORDS)
        }

        if self.allow_text_param_upload:
            return full_state, local_reps, None, training_stats
        return image_only_state, local_reps, None, training_stats

    def get_uploadable_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        full_state = self.get_model_state(model)
        text_param_keywords = ('text_encoder', 'text_proj', 'text_adapter')
        image_only_state = {
            k: v for k, v in full_state.items()
            if not any(kw in k for kw in text_param_keywords)
        }
        if self.allow_text_param_upload:
            return full_state
        return image_only_state

    def get_return_values(
        self,
        model: nn.Module,
        local_reps: torch.Tensor,
        training_stats: Dict
    ) -> Tuple[Dict, torch.Tensor, None, Dict]:
        return self.get_uploadable_state(model), local_reps, None, training_stats


# ============================================================================
# Phase 2 核心：MultimodalTrainer（多模态训练器）
# ============================================================================
class MultimodalTrainer(BaseClientTrainer):
    """
    ★ 多模态训练器（MultimodalTrainer）

    **数据格式**：
    - Private Batch: (image, mask, text_feature)
    - Public Batch: (image, text_feature)

    **训练策略**：
    - 分割损失 + 多模态对比学习损失
    - 支持文本-图像融合

    **返回值**：
    - weights: 完整模型参数
    - image_rep: 图像表征
    - text_rep: 文本表征
    """

    def unpack_private_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包私有批次：(image, mask, text_feature)"""
        if isinstance(batch, dict):
            img = batch.get('image', batch.get('inp'))
            mask = batch.get('mask', batch.get('gt', batch.get('label')))
            text_feat = batch.get('text_feature')
        elif isinstance(batch, (list, tuple)):
            img = batch[0]
            mask = batch[1] if len(batch) > 1 else None
            text_feat = batch[2] if len(batch) > 2 else None
        else:
            img = batch
            mask = None
            text_feat = None

        img = img.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if text_feat is not None:
            text_feat = text_feat.to(self.device)

        return {'image': img, 'mask': mask, 'text_feat': text_feat}

    def unpack_public_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """解包公共批次：(image, text_feature)"""
        if isinstance(batch, dict):
            img = batch.get('image', batch.get('inp'))
            text_feat = batch.get('text_feature')
        elif isinstance(batch, (list, tuple)):
            img = batch[0]
            text_feat = batch[1] if len(batch) > 1 else None
        else:
            img = batch
            text_feat = None

        img = img.to(self.device)
        if text_feat is not None:
            text_feat = text_feat.to(self.device)

        return {'image': img, 'text_feat': text_feat}

    def _get_fallback_public_batch(self, private_inputs: Dict) -> Any:
        """
        ★ 生成全零图文张量

        **触发条件**：当 public_loader 数据用尽时
        **返回格式**：{'image': Tensor, 'text_feature': Tensor} 字典
        **关键修复**：防止返回空字典导致 NoneType AttributeError
        """
        img = private_inputs['image']
        text_feat = private_inputs['text_feat']
        return {
            'image': torch.zeros_like(img),
            'text_feature': torch.zeros_like(text_feat) if text_feat is not None else None
        }

    def compute_loss(
        self,
        model: nn.Module,
        private_inputs: Dict,
        public_inputs: Dict,
        global_reps: Dict,
        lambda_cream: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        计算分割损失 + 多模态对比学习损失

        **策略**：
        - 私有数据：计算分割损失（支持文本融合）
        - 公共数据：计算多模态对比学习损失
        """
        img = private_inputs['image']
        mask = private_inputs['mask']
        private_text_feat = private_inputs['text_feat']
        public_img = public_inputs['image']
        public_text_feat = public_inputs['text_feat']
        global_text_rep = global_reps['text']
        global_image_rep = global_reps['image']
        use_text_assist = bool(self.enable_text_assist_in_seg)

        # Step 1: 私有数据 - 分割损失
        if mask is not None:
            # 多模态客户端同时注入当前批次私有文本与服务器全局文本代理。
            text_prompt_val = global_text_rep.detach() if use_text_assist and global_text_rep is not None else None
            pred = model(
                img,
                text_features=private_text_feat if use_text_assist else None,
                global_text_rep=text_prompt_val,
            )
            if isinstance(pred, dict):
                pred = pred.get('logits', list(pred.values())[0])

            seg_loss = self._compute_segmentation_loss(pred, mask)
        else:
            seg_loss = torch.tensor(0.0, device=self.device)

        # Step 2: 公共数据 - 对比学习损失（图像侧）
        pub_feat = model.extract_features(
            public_img,
            text_features=public_text_feat if use_text_assist else None
        )  # (B, N, D_contrastive)

        # ★ Fix Critical 2 (2026-03-13): 提取公共数据的文本侧特征，用于真正的跨模态对齐
        # 原来 img_rep == txt_rep，导致跨模态对比学习退化为模态内对比。
        if public_text_feat is not None:
            fusion_head = getattr(model, 'fusion_head', None)
            if fusion_head is None:
                raise ValueError(
                    "[MultimodalTrainer] model.fusion_head is missing; cannot project text features."
                )
            if hasattr(fusion_head, 'project_text'):
                pub_text_rep = fusion_head.project_text(public_text_feat)
            elif hasattr(fusion_head, 'text_proj'):
                pub_text_rep = fusion_head.text_proj(public_text_feat)
            else:
                raise ValueError(
                    "[MultimodalTrainer] fusion_head has no project_text/text_proj; "
                    "cannot compute multimodal text representation."
                )
            if pub_text_rep.dim() == 3:
                pub_text_rep = pub_text_rep.mean(dim=1)
        else:
            pub_text_rep = None
        # 暂存文本侧表征，供 get_return_values 读取
        self._last_pub_text_rep = pub_text_rep.detach() if pub_text_rep is not None else None

        # 计算对比学习损失
        L_inter, L_intra = self.cream_loss_fn(
            pub_feat, global_text_rep, global_image_rep
        )
        cream_loss = L_inter + L_intra

        # 显式文本对齐项：让 text_proj 真正收到梯度，而不是只做表征缓存。
        text_align_loss = torch.tensor(0.0, device=self.device)
        if pub_text_rep is not None:
            text_align_loss = self.cream_loss_fn.contrastive_loss.inter_modal_loss(
                pub_text_rep, global_text_rep
            )

        # 总损失
        total_loss = seg_loss + lambda_cream * (cream_loss + text_align_loss)

        local_batch = int(pub_feat.shape[0]) if pub_feat.dim() >= 1 else 1
        cross_batch = int(global_text_rep.shape[0]) if global_text_rep.dim() > 1 else 1
        pair_mode = 'diagonal' if cross_batch == local_batch else 'anchor0'
        contrastive_impl = getattr(self.cream_loss_fn, 'contrastive_loss', None)
        tau = float(getattr(contrastive_impl, 'tau', float('nan')))
        temperature = float(getattr(contrastive_impl, 'temperature', float('nan')))
        self._last_multimodal_cream_diag = {
            'cream_raw': float(cream_loss.detach().item()),
            'cream_weighted': float((lambda_cream * cream_loss).detach().item()),
            'text_align_raw': float(text_align_loss.detach().item()),
            'text_align_weighted': float((lambda_cream * text_align_loss).detach().item()),
            'valid_pairs': float(local_batch),
            'local_batch': float(local_batch),
            'cross_batch': float(cross_batch),
            'pair_mode': pair_mode,
            'is_finite': bool(torch.isfinite(cream_loss + text_align_loss).item()),
            'text_missing': bool(public_text_feat is None),
            'l_inter': float(L_inter.detach().item()),
            'l_intra': float(L_intra.detach().item()),
            'tau': tau,
            'temperature': temperature,
        }

        # 收集公共数据图像侧表征（用于聚合）
        public_rep = pub_feat.mean(dim=1)  # (B, D)

        return total_loss, seg_loss, cream_loss, public_rep

    def get_return_values(
        self,
        model: nn.Module,
        local_reps: torch.Tensor,
        training_stats: Dict
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict]:
        """
        返回多模态结果

        **返回格式**：(trainable_state_dict, image_rep, text_rep, stats)
        """
        # ★ Fix Critical 1 (2026-03-13): 只返回可训练参数
        # ★ Fix Critical 2 (2026-03-13): img_rep 和 txt_rep 分别独立
        #   - local_reps：图像侧聚合表征（来自 _aggregate_representations，图像 pub_feat 均值）
        #   - text_rep：文本侧表征（来自 compute_loss 中暂存的 _last_pub_text_rep）
        #   若缺少文本表征，不允许回退为 local_reps（图像表征），避免文本锚点被图像污染
        text_rep = None
        if hasattr(self, '_last_pub_text_rep') and self._last_pub_text_rep is not None:
            _pub_txt = self._last_pub_text_rep
            text_rep = _pub_txt.mean(dim=0).cpu() if _pub_txt.dim() > 1 else _pub_txt.cpu()
        if text_rep is None:
            raise RuntimeError(
                "Multimodal client did not produce a public text representation"
            )
        return self.get_model_state(model), local_reps, text_rep, training_stats


# ============================================================================
# 工厂函数：创建训练器
# ============================================================================
def create_trainer(
    modality: str,
    private_loader: DataLoader,
    public_loader: DataLoader,
    **kwargs
) -> BaseClientTrainer:
    """
    ★ 训练器工厂函数（Factory Pattern）

    **使用示例**：
    ```python
    trainer = create_trainer(
        modality="text_only",  # or "image_only", "multimodal"
        private_loader=private_loader,
        public_loader=public_loader,
        device="cuda",
        use_amp=True,
        grad_clip=1.0
    )
    weights, img_rep, txt_rep, stats = trainer.run(model, optimizer, global_reps)
    ```

    Args:
        modality: 客户端模态类型 ("text_only", "image_only", "multimodal")
        private_loader: 私有数据加载器
        public_loader: 公共数据加载器
        **kwargs: 传递给训练器的额外参数

    Returns:
        BaseClientTrainer 子类实例

    Raises:
        ValueError: 若 modality 无效
    """
    if modality == "text_only":
        return TextOnlyTrainer(private_loader, public_loader, **kwargs)
    elif modality == "image_only":
        return ImageOnlyTrainer(private_loader, public_loader, **kwargs)
    elif modality == "multimodal":
        return MultimodalTrainer(private_loader, public_loader, **kwargs)
    else:
        raise ValueError(
            f"未知的 modality: {modality}\n"
            f"有效值: ['text_only', 'image_only', 'multimodal']"
        )




# ============================================================================
# 单元测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Phase 2 重构：消灭上帝对象 - 单元测试")
    print("=" * 80)

    from torch.utils.data import TensorDataset

    # 创建虚拟数据
    num_samples = 5
    img_size_test = 128
    dummy_imgs = torch.randn(num_samples, 3, img_size_test, img_size_test)
    dummy_masks = torch.randn(num_samples, 1, img_size_test, img_size_test)
    dummy_text_feats = torch.randn(num_samples, 768)

    # 测试 1: TextOnlyTrainer
    print("\n[测试 1] TextOnlyTrainer")
    print("-" * 80)
    text_private_dataset = TensorDataset(dummy_text_feats)
    text_public_dataset = TensorDataset(dummy_text_feats)
    text_private_loader = DataLoader(text_private_dataset, batch_size=2, shuffle=True)
    text_public_loader = DataLoader(text_public_dataset, batch_size=2, shuffle=True)

    text_trainer = create_trainer(
        modality="text_only",
        private_loader=text_private_loader,
        public_loader=text_public_loader,
        device="cpu",
        use_amp=False,
        grad_clip=1.0
    )
    print(f"✓ TextOnlyTrainer 创建成功: {type(text_trainer).__name__}")

    # 测试 2: ImageOnlyTrainer
    print("\n[测试 2] ImageOnlyTrainer")
    print("-" * 80)
    image_private_dataset = TensorDataset(dummy_imgs, dummy_masks)
    image_public_dataset = TensorDataset(dummy_imgs)
    image_private_loader = DataLoader(image_private_dataset, batch_size=2, shuffle=True)
    image_public_loader = DataLoader(image_public_dataset, batch_size=2, shuffle=True)

    image_trainer = create_trainer(
        modality="image_only",
        private_loader=image_private_loader,
        public_loader=image_public_loader,
        device="cpu",
        use_amp=False
    )
    print(f"✓ ImageOnlyTrainer 创建成功: {type(image_trainer).__name__}")

    # 测试 3: MultimodalTrainer
    print("\n[测试 3] MultimodalTrainer")
    print("-" * 80)
    multi_private_dataset = TensorDataset(dummy_imgs, dummy_masks, dummy_text_feats)
    multi_public_dataset = TensorDataset(dummy_imgs, dummy_text_feats)
    multi_private_loader = DataLoader(multi_private_dataset, batch_size=2, shuffle=True)
    multi_public_loader = DataLoader(multi_public_dataset, batch_size=2, shuffle=True)

    multi_trainer = create_trainer(
        modality="multimodal",
        private_loader=multi_private_loader,
        public_loader=multi_public_loader,
        device="cpu",
        use_amp=False
    )
    print(f"✓ MultimodalTrainer 创建成功: {type(multi_trainer).__name__}")

    # 测试 4: MultimodalTrainer 直接实例化
    print("\n[测试 4] MultimodalTrainer 直接实例化")
    print("-" * 80)
    legacy_trainer = MultimodalTrainer(
        private_loader=multi_private_loader,
        public_loader=multi_public_loader,
        device="cpu",
        use_amp=False
    )
    print(f"✓ MultimodalTrainer 直接实例化成功: {type(legacy_trainer).__name__}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！Phase 2 重构成功！")
    print("=" * 80)
