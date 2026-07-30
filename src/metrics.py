"""Strict BraTS [WT, TC, ET] segmentation metrics."""

import math
from collections.abc import Sequence
from typing import Any, Dict

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


def physical_diagonal(
    shape: Sequence[int],
    voxel_spacing: Sequence[float],
) -> float:
    if len(shape) != len(voxel_spacing) or len(shape) < 2:
        raise ValueError("shape and voxel_spacing must have the same dimension")
    extents = []
    for size, spacing in zip(shape, voxel_spacing):
        size = int(size)
        spacing = float(spacing)
        if size <= 0 or not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("shape must be positive and voxel spacing must be finite")
        extents.append((size - 1) * spacing)
    return math.sqrt(sum(extent ** 2 for extent in extents))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    voxel_spacing: Sequence[float] | None = None,
) -> float:
    """Compute symmetric surface HD95 for two non-empty binary masks."""
    pred = np.asarray(pred, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    if pred.ndim < 2 or target.ndim != pred.ndim or pred.shape != target.shape:
        raise ValueError(
            f"HD95 masks must share an N-D shape, got {pred.shape} and {target.shape}"
        )
    if not pred.any() or not target.any():
        raise ValueError("empty-mask HD95 must be resolved by the metric contract")
    if np.array_equal(pred, target):
        return 0.0

    spacing = None
    if voxel_spacing is not None:
        spacing = tuple(float(value) for value in voxel_spacing)
        if len(spacing) != pred.ndim or any(
            not math.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("voxel_spacing must contain one positive value per axis")

    if HAS_MEDPY:
        return float(
            medpy_metric.binary.hd95(
                pred,
                target,
                voxelspacing=spacing,
            )
        )
    if not HAS_SCIPY:
        raise RuntimeError("HD95 requires medpy or scipy for non-empty masks")

    structure = ndimage.generate_binary_structure(pred.ndim, 1)
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
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing,
    )
    distance_to_pred = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )
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


class BraTS3DHD95Accumulator:
    """Stream complete cases and macro-average physical 3D HD95."""

    def __init__(self) -> None:
        self._stats = {
            region: {
                "case_count": 0,
                "hd95_sum": 0.0,
                "both_empty_case_count": 0,
                "empty_fp_case_count": 0,
                "empty_fn_case_count": 0,
                "both_nonempty_case_count": 0,
            }
            for region in REGION_NAMES
        }
        self._current_case: Dict[str, Any] | None = None
        self._finalized_case_keys: set[str] = set()
        self._computed = False

    @staticmethod
    def _metadata_rows(
        metadata: Dict[str, Any],
        batch_size: int,
    ) -> list[Dict[str, Any]]:
        required = {
            "case_key",
            "case_id",
            "slice_index",
            "volume_shape",
            "voxel_spacing_mm",
        }
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            missing = sorted(required.difference(metadata or {}))
            raise ValueError(f"3D HD95 metadata is missing fields: {missing}")

        case_keys = metadata["case_key"]
        case_ids = metadata["case_id"]
        if (
            not isinstance(case_keys, (list, tuple))
            or not isinstance(case_ids, (list, tuple))
            or len(case_keys) != batch_size
            or len(case_ids) != batch_size
        ):
            raise ValueError("case_key and case_id metadata must match batch size")

        tensor_fields = {}
        expected_shapes = {
            "slice_index": (batch_size,),
            "volume_shape": (batch_size, 3),
            "voxel_spacing_mm": (batch_size, 3),
        }
        for field, expected_shape in expected_shapes.items():
            value = metadata[field]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{field} metadata must have shape {expected_shape}"
                )
            tensor_fields[field] = value.detach().cpu()

        rows = []
        for batch_index in range(batch_size):
            case_key = str(case_keys[batch_index])
            case_id = str(case_ids[batch_index])
            volume_shape = tuple(
                int(value)
                for value in tensor_fields["volume_shape"][batch_index].tolist()
            )
            voxel_spacing = tuple(
                float(value)
                for value in tensor_fields["voxel_spacing_mm"][batch_index].tolist()
            )
            slice_index = int(
                tensor_fields["slice_index"][batch_index].item()
            )
            if not case_key or not case_id:
                raise ValueError("case_key and case_id must be non-empty")
            if any(size <= 1 for size in volume_shape):
                raise ValueError(f"invalid NIfTI volume shape: {volume_shape}")
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in voxel_spacing
            ):
                raise ValueError(
                    f"invalid NIfTI voxel spacing: {voxel_spacing}"
                )
            if slice_index < 0 or slice_index >= volume_shape[2]:
                raise ValueError(
                    f"slice index {slice_index} is outside volume {volume_shape}"
                )
            rows.append(
                {
                    "case_key": case_key,
                    "case_id": case_id,
                    "slice_index": slice_index,
                    "volume_shape": volume_shape,
                    "voxel_spacing_mm": voxel_spacing,
                }
            )
        return rows

    def _start_case(
        self,
        metadata: Dict[str, Any],
        evaluation_shape: tuple[int, int],
    ) -> None:
        case_key = metadata["case_key"]
        if case_key in self._finalized_case_keys:
            raise RuntimeError(
                f"validation slices for case {metadata['case_id']} are not contiguous"
            )
        height, width = evaluation_shape
        if height <= 1 or width <= 1:
            raise ValueError(f"invalid evaluation shape: {evaluation_shape}")

        original_shape = metadata["volume_shape"]
        original_spacing = metadata["voxel_spacing_mm"]
        effective_spacing = (
            original_spacing[0] * (original_shape[0] - 1) / (height - 1),
            original_spacing[1] * (original_shape[1] - 1) / (width - 1),
            original_spacing[2],
        )
        depth = original_shape[2]
        self._current_case = {
            **metadata,
            "evaluation_shape": (height, width, depth),
            "effective_spacing_mm": effective_spacing,
            "pred": np.zeros(
                (len(REGION_NAMES), height, width, depth),
                dtype=np.bool_,
            ),
            "target": np.zeros(
                (len(REGION_NAMES), height, width, depth),
                dtype=np.bool_,
            ),
            "seen_slices": np.zeros(depth, dtype=np.bool_),
        }

    def _finalize_current_case(self) -> None:
        state = self._current_case
        if state is None:
            return
        missing_slices = np.flatnonzero(~state["seen_slices"])
        if missing_slices.size:
            raise RuntimeError(
                f"case {state['case_id']} has missing validation slices: "
                f"{missing_slices.tolist()}"
            )

        diagonal = physical_diagonal(
            state["evaluation_shape"],
            state["effective_spacing_mm"],
        )
        for channel_index, region in enumerate(REGION_NAMES):
            pred = state["pred"][channel_index]
            target = state["target"][channel_index]
            pred_nonempty = bool(pred.any())
            target_nonempty = bool(target.any())
            stats = self._stats[region]

            if not pred_nonempty and not target_nonempty:
                hd95 = 0.0
                stats["both_empty_case_count"] += 1
            elif pred_nonempty and not target_nonempty:
                hd95 = diagonal
                stats["empty_fp_case_count"] += 1
            elif not pred_nonempty and target_nonempty:
                hd95 = diagonal
                stats["empty_fn_case_count"] += 1
            else:
                hd95 = hausdorff_distance_95(
                    pred,
                    target,
                    voxel_spacing=state["effective_spacing_mm"],
                )
                stats["both_nonempty_case_count"] += 1

            stats["case_count"] += 1
            stats["hd95_sum"] += hd95

        self._finalized_case_keys.add(state["case_key"])
        self._current_case = None

    def update_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        thresholds: Sequence[float],
        metadata: Dict[str, Any],
    ) -> None:
        if self._computed:
            raise RuntimeError("cannot update 3D HD95 after compute")
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim != 4 or logits.shape[1] != len(REGION_NAMES):
            raise ValueError(
                f"logits must have shape [B, 3, H, W] in order {REGION_NAMES}"
            )
        if not torch.is_floating_point(logits) or logits.is_complex():
            raise TypeError("logits must use a real floating-point dtype")
        if len(thresholds) != len(REGION_NAMES):
            raise ValueError(f"thresholds must follow channel order {REGION_NAMES}")

        BraTSMetricAccumulator._validate_target(target)
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
        BraTSMetricAccumulator._validate_prediction(pred_regions, target)
        pred_regions = close_nested_regions(pred_regions, channel_dim=1)
        metadata_rows = self._metadata_rows(metadata, target.shape[0])

        for batch_index, row in enumerate(metadata_rows):
            if (
                self._current_case is None
                or row["case_key"] != self._current_case["case_key"]
            ):
                self._finalize_current_case()
                self._start_case(row, tuple(target.shape[-2:]))
            state = self._current_case
            if (
                row["volume_shape"] != state["volume_shape"]
                or row["voxel_spacing_mm"] != state["voxel_spacing_mm"]
            ):
                raise ValueError(
                    f"inconsistent NIfTI metadata for case {row['case_id']}"
                )

            slice_index = row["slice_index"]
            if state["seen_slices"][slice_index]:
                raise RuntimeError(
                    f"duplicate validation slice {slice_index} for case {row['case_id']}"
                )
            state["pred"][:, :, :, slice_index] = (
                pred_regions[batch_index].detach().cpu().numpy()
            )
            state["target"][:, :, :, slice_index] = (
                target[batch_index].detach().cpu().numpy().astype(np.bool_)
            )
            state["seen_slices"][slice_index] = True

    def compute(self) -> Dict[str, Any]:
        if self._computed:
            raise RuntimeError("3D HD95 can only be computed once")
        self._finalize_current_case()
        if not self._finalized_case_keys:
            raise RuntimeError("cannot compute 3D HD95 before a complete case")
        self._computed = True

        results: Dict[str, Any] = {}
        region_hd95 = []
        for region in REGION_NAMES:
            stats = self._stats[region]
            case_count = stats["case_count"]
            hd95 = stats["hd95_sum"] / case_count
            results[f"{region}_hd95"] = float(hd95)
            results[f"{region}_case_count"] = int(case_count)
            for state in (
                "both_empty",
                "empty_fp",
                "empty_fn",
                "both_nonempty",
            ):
                count = int(stats[f"{state}_case_count"])
                results[f"{region}_{state}_case_count"] = count
                results[f"{region}_{state}_case_rate"] = float(
                    count / case_count
                )
            region_hd95.append(hd95)

        results["hd95"] = float(np.mean(region_hd95))
        results["hd95_unit"] = "mm"
        results["hd95_dimension"] = "3d_case"
        results["hd95_aggregation"] = "macro_case_then_region"
        results["hd95_empty_policy"] = "physical_volume_diagonal_mm"
        results["num_cases"] = len(self._finalized_case_keys)
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
