"""Strict BraTS [WT, TC, ET] segmentation metrics."""

import math
from collections.abc import Sequence
from typing import Dict

import numpy as np
import torch

from data_processing.brats_region_contract import REGION_NAMES, close_nested_regions

try:
    from medpy import metric as medpy_metric

    HAS_MEDPY = True
except ImportError:
    medpy_metric = None
    HAS_MEDPY = False

try:
    from scipy import ndimage

    HAS_SCIPY = True
except ImportError:
    ndimage = None
    HAS_SCIPY = False


def image_diagonal(shape: Sequence[int]) -> float:
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"image shape must be positive [H, W], got {tuple(shape)}")
    height, width = (int(size) for size in shape)
    return math.sqrt((height - 1) ** 2 + (width - 1) ** 2)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute symmetric surface HD95 for two non-empty 2D binary masks."""
    pred = np.asarray(pred, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        raise ValueError(
            f"HD95 masks must share shape [H, W], got {pred.shape} and {target.shape}"
        )
    if not pred.any() or not target.any():
        raise ValueError("empty-mask HD95 must be resolved by the metric contract")
    if np.array_equal(pred, target):
        return 0.0

    if HAS_MEDPY:
        return float(medpy_metric.binary.hd95(pred, target))
    if not HAS_SCIPY:
        raise RuntimeError("HD95 requires medpy or scipy for non-empty masks")

    structure = ndimage.generate_binary_structure(2, 1)
    pred_surface = pred ^ ndimage.binary_erosion(
        pred,
        structure=structure,
        border_value=0,
    )
    target_surface = target ^ ndimage.binary_erosion(
        target,
        structure=structure,
        border_value=0,
    )
    distance_to_target = ndimage.distance_transform_edt(~target_surface)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        (
            distance_to_target[pred_surface],
            distance_to_pred[target_surface],
        )
    )
    return float(np.percentile(distances, 95))


class BraTSMetricAccumulator:
    """Macro-average 2D sample metrics with explicit empty-region accounting."""

    def __init__(self, *, compute_hd95: bool) -> None:
        self.compute_hd95 = bool(compute_hd95)
        self._stats = {
            region: {
                "num_samples": 0,
                "dice_sum": 0.0,
                "iou_sum": 0.0,
                "hd95_sum": 0.0,
                "both_empty_count": 0,
                "empty_fp_count": 0,
                "empty_fn_count": 0,
                "both_nonempty_count": 0,
                "pred_voxels": 0,
                "gt_voxels": 0,
                "tp_voxels": 0,
            }
            for region in REGION_NAMES
        }

    @staticmethod
    def _validate_target(target: torch.Tensor) -> None:
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a torch.Tensor")
        if target.ndim != 4 or target.shape[1] != len(REGION_NAMES):
            raise ValueError(
                f"target must have shape [B, 3, H, W] in order {REGION_NAMES}, "
                f"got {tuple(target.shape)}"
            )
        if target.dtype != torch.float32:
            raise TypeError(f"target must use torch.float32, got {target.dtype}")
        if not torch.isfinite(target).all():
            raise ValueError("target must contain only finite values")
        if torch.any((target != 0.0) & (target != 1.0)):
            raise ValueError("target must be binary with values in {0, 1}")

        wt = target[:, 0].bool()
        tc = target[:, 1].bool()
        et = target[:, 2].bool()
        if torch.any(et & ~tc) or torch.any(tc & ~wt):
            raise ValueError("target must satisfy ET subset TC subset WT")

    @staticmethod
    def _validate_prediction(
        pred_regions: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if not isinstance(pred_regions, torch.Tensor):
            raise TypeError("pred_regions must be a torch.Tensor")
        if tuple(pred_regions.shape) != tuple(target.shape):
            raise ValueError(
                "prediction and target must share shape [B, 3, H, W]; "
                f"got prediction={tuple(pred_regions.shape)}, target={tuple(target.shape)}"
            )
        if pred_regions.device != target.device:
            raise ValueError(
                f"prediction and target must share a device, "
                f"got {pred_regions.device} and {target.device}"
            )
        if torch.is_floating_point(pred_regions) and not torch.isfinite(pred_regions).all():
            raise ValueError("prediction must contain only finite values")
        if torch.any((pred_regions != 0) & (pred_regions != 1)):
            raise ValueError("prediction must be binary with values in {0, 1}")

    def update_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        thresholds: Sequence[float],
    ) -> None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim != 4 or logits.shape[1] != len(REGION_NAMES):
            raise ValueError(
                f"logits must have shape [B, 3, H, W] in order {REGION_NAMES}, "
                f"got {tuple(logits.shape)}"
            )
        if not torch.is_floating_point(logits) or logits.is_complex():
            raise TypeError("logits must use a real floating-point dtype")
        if len(thresholds) != len(REGION_NAMES):
            raise ValueError(f"thresholds must follow channel order {REGION_NAMES}")

        threshold_tensor = torch.as_tensor(
            thresholds,
            dtype=logits.dtype,
            device=logits.device,
        )
        if not torch.isfinite(threshold_tensor).all() or torch.any(
            (threshold_tensor <= 0.0) | (threshold_tensor >= 1.0)
        ):
            raise ValueError("thresholds must be finite values strictly between 0 and 1")
        threshold_tensor = threshold_tensor.view(1, len(REGION_NAMES), 1, 1)
        pred_regions = torch.sigmoid(logits) >= threshold_tensor
        self.update(pred_regions, target)

    def update(
        self,
        pred_regions: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self._validate_target(target)
        self._validate_prediction(pred_regions, target)
        pred_regions = close_nested_regions(pred_regions, channel_dim=1)
        target_regions = target.bool()
        diagonal = image_diagonal(target.shape[-2:])

        for batch_index in range(target.shape[0]):
            for channel_index, region in enumerate(REGION_NAMES):
                pred_mask = pred_regions[batch_index, channel_index]
                target_mask = target_regions[batch_index, channel_index]
                pred_count = int(pred_mask.sum().item())
                target_count = int(target_mask.sum().item())
                intersection = int((pred_mask & target_mask).sum().item())
                stats = self._stats[region]

                stats["num_samples"] += 1
                stats["pred_voxels"] += pred_count
                stats["gt_voxels"] += target_count
                stats["tp_voxels"] += intersection

                if pred_count == 0 and target_count == 0:
                    dice = 1.0
                    iou = 1.0
                    hd95 = 0.0
                    stats["both_empty_count"] += 1
                elif target_count == 0:
                    dice = 0.0
                    iou = 0.0
                    hd95 = diagonal
                    stats["empty_fp_count"] += 1
                elif pred_count == 0:
                    dice = 0.0
                    iou = 0.0
                    hd95 = diagonal
                    stats["empty_fn_count"] += 1
                else:
                    dice = 2.0 * intersection / (pred_count + target_count)
                    union = pred_count + target_count - intersection
                    iou = intersection / union
                    hd95 = (
                        hausdorff_distance_95(
                            pred_mask.detach().cpu().numpy(),
                            target_mask.detach().cpu().numpy(),
                        )
                        if self.compute_hd95
                        else 0.0
                    )
                    stats["both_nonempty_count"] += 1

                stats["dice_sum"] += dice
                stats["iou_sum"] += iou
                if self.compute_hd95:
                    stats["hd95_sum"] += hd95

    def compute(self) -> Dict[str, float]:
        if any(self._stats[region]["num_samples"] == 0 for region in REGION_NAMES):
            raise RuntimeError("cannot compute metrics before at least one update")

        results: Dict[str, float] = {}
        region_dice = []
        region_iou = []
        region_hd95 = []
        total_pred = 0
        total_gt = 0
        total_tp = 0

        for region in REGION_NAMES:
            stats = self._stats[region]
            count = stats["num_samples"]
            dice = stats["dice_sum"] / count
            iou = stats["iou_sum"] / count
            region_dice.append(dice)
            region_iou.append(iou)
            total_pred += stats["pred_voxels"]
            total_gt += stats["gt_voxels"]
            total_tp += stats["tp_voxels"]

            results[f"{region}_dice"] = float(dice)
            results[f"{region}_iou"] = float(iou)
            results[f"{region}_num_samples"] = int(count)
            for state in (
                "both_empty",
                "empty_fp",
                "empty_fn",
                "both_nonempty",
            ):
                state_count = int(stats[f"{state}_count"])
                results[f"{region}_{state}_count"] = state_count
                results[f"{region}_{state}_rate"] = float(state_count / count)
            results[f"{region}_pred_fg_voxels"] = int(stats["pred_voxels"])
            results[f"{region}_gt_fg_voxels"] = int(stats["gt_voxels"])
            if self.compute_hd95:
                hd95 = stats["hd95_sum"] / count
                results[f"{region}_hd95"] = float(hd95)
                region_hd95.append(hd95)

        false_positive = total_pred - total_tp
        false_negative = total_gt - total_tp
        precision_denominator = total_tp + false_positive
        recall_denominator = total_tp + false_negative
        results["dice"] = float(np.mean(region_dice))
        results["iou"] = float(np.mean(region_iou))
        results["precision"] = (
            float(total_tp / precision_denominator)
            if precision_denominator > 0
            else 1.0
        )
        results["recall"] = (
            float(total_tp / recall_denominator)
            if recall_denominator > 0
            else 1.0
        )
        results["pred_fg_voxels"] = int(total_pred)
        results["gt_fg_voxels"] = int(total_gt)
        if self.compute_hd95:
            results["hd95"] = float(np.mean(region_hd95))
        return results


class MedicalMetricsCalculator:
    """One-batch interface backed by the strict BraTS metric contract."""

    def __init__(
        self,
        *,
        thresholds: Sequence[float],
        compute_hd95: bool = True,
    ) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        self.compute_hd95 = bool(compute_hd95)

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        accumulator = BraTSMetricAccumulator(compute_hd95=self.compute_hd95)
        accumulator.update_from_logits(
            y_pred,
            y,
            thresholds=self.thresholds,
        )
        return accumulator.compute()
