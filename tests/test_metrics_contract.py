import math

import pytest
import torch

from src.metrics import BraTS3DHD95Accumulator, BraTSMetricAccumulator


def _volume_metadata(case_key, case_id, slice_index, shape, spacing):
    return {
        "case_key": [case_key],
        "case_id": [case_id],
        "slice_index": torch.tensor([slice_index], dtype=torch.int64),
        "volume_shape": torch.tensor([shape], dtype=torch.int64),
        "voxel_spacing_mm": torch.tensor([spacing], dtype=torch.float64),
    }


def test_empty_region_rules_are_explicit_and_false_positives_are_penalized():
    pred = torch.zeros(4, 3, 3, 4, dtype=torch.bool)
    target = torch.zeros(4, 3, 3, 4, dtype=torch.float32)

    pred[1, 0, 1, 1] = True
    target[2, 0, 1, 1] = 1.0
    pred[3, 0, 1, 1] = True
    target[3, 0, 1, 1] = 1.0

    accumulator = BraTSMetricAccumulator(compute_hd95=True)
    accumulator.update(pred, target)
    metrics = accumulator.compute()

    diagonal = math.sqrt((3 - 1) ** 2 + (4 - 1) ** 2)
    assert metrics["WT_dice"] == pytest.approx(0.5)
    assert metrics["WT_iou"] == pytest.approx(0.5)
    assert metrics["WT_hd95"] == pytest.approx(diagonal / 2.0)
    assert metrics["WT_both_empty_rate"] == pytest.approx(0.25)
    assert metrics["WT_empty_fp_rate"] == pytest.approx(0.25)
    assert metrics["WT_empty_fn_rate"] == pytest.approx(0.25)
    assert metrics["WT_both_nonempty_rate"] == pytest.approx(0.25)

    assert metrics["TC_dice"] == pytest.approx(1.0)
    assert metrics["TC_iou"] == pytest.approx(1.0)
    assert metrics["TC_hd95"] == pytest.approx(0.0)
    assert metrics["TC_both_empty_rate"] == pytest.approx(1.0)
    assert metrics["ET_both_empty_rate"] == pytest.approx(1.0)
    assert math.isfinite(metrics["hd95"])


def test_configured_thresholds_and_nested_closure_apply_to_all_regions():
    probabilities = torch.full((1, 3, 2, 2), 0.01, dtype=torch.float32)
    probabilities[0, 0, 0, 0] = 0.40
    probabilities[0, 1, 0, 0] = 0.60
    probabilities[0, 2, 0, 0] = 0.95
    target = torch.zeros_like(probabilities)
    target[0, :, 0, 0] = 1.0

    accumulator = BraTSMetricAccumulator(compute_hd95=True)
    accumulator.update_from_logits(
        torch.logit(probabilities),
        target,
        thresholds=(0.5, 0.7, 0.9),
    )
    metrics = accumulator.compute()

    for region in ("WT", "TC", "ET"):
        assert metrics[f"{region}_dice"] == pytest.approx(1.0)
        assert metrics[f"{region}_iou"] == pytest.approx(1.0)
        assert metrics[f"{region}_hd95"] == pytest.approx(0.0)


def test_metric_contract_rejects_non_nested_target():
    pred = torch.zeros(1, 3, 2, 2, dtype=torch.bool)
    target = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
    target[0, 2, 0, 0] = 1.0

    accumulator = BraTSMetricAccumulator(compute_hd95=False)
    with pytest.raises(ValueError, match="ET subset TC subset WT"):
        accumulator.update(pred, target)


def test_metric_accumulator_combines_batches_without_dropping_empty_samples():
    pred = torch.zeros(2, 3, 2, 2, dtype=torch.bool)
    target = torch.zeros(2, 3, 2, 2, dtype=torch.float32)
    pred[1, 0, 0, 0] = True

    accumulator = BraTSMetricAccumulator(compute_hd95=True)
    accumulator.update(pred[:1], target[:1])
    accumulator.update(pred[1:], target[1:])
    metrics = accumulator.compute()

    assert metrics["WT_num_samples"] == 2
    assert metrics["WT_both_empty_count"] == 1
    assert metrics["WT_empty_fp_count"] == 1
    assert metrics["WT_dice"] == pytest.approx(0.5)


def test_3d_hd95_uses_nifti_spacing_after_in_plane_resize():
    accumulator = BraTS3DHD95Accumulator()
    metadata_shape = (5, 4, 2)
    spacing = (2.0, 3.0, 5.0)

    for slice_index in range(metadata_shape[2]):
        logits = torch.full((1, 3, 3, 4), -20.0)
        target = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
        if slice_index == 0:
            logits[0, :, 2, 1] = 20.0
            target[0, :, 1, 1] = 1.0
        accumulator.update_from_logits(
            logits,
            target,
            thresholds=(0.5, 0.5, 0.5),
            metadata=_volume_metadata(
                "client_2/case_a",
                "case_a",
                slice_index,
                metadata_shape,
                spacing,
            ),
        )

    metrics = accumulator.compute()

    for region in ("WT", "TC", "ET"):
        assert metrics[f"{region}_hd95"] == pytest.approx(4.0)
        assert metrics[f"{region}_case_count"] == 1
    assert metrics["hd95"] == pytest.approx(4.0)
    assert metrics["hd95_unit"] == "mm"
    assert metrics["hd95_dimension"] == "3d_case"
    assert metrics["hd95_aggregation"] == "macro_case_then_region"


def test_3d_hd95_empty_mismatch_uses_physical_volume_diagonal():
    accumulator = BraTS3DHD95Accumulator()
    metadata_shape = (5, 4, 2)
    spacing = (2.0, 3.0, 5.0)

    for slice_index in range(metadata_shape[2]):
        logits = torch.full((1, 3, 3, 4), -20.0)
        target = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
        if slice_index == 0:
            target[0, :, 1, 1] = 1.0
        accumulator.update_from_logits(
            logits,
            target,
            thresholds=(0.5, 0.5, 0.5),
            metadata=_volume_metadata(
                "client_2/case_a",
                "case_a",
                slice_index,
                metadata_shape,
                spacing,
            ),
        )

    metrics = accumulator.compute()
    physical_diagonal = math.sqrt(8.0 ** 2 + 9.0 ** 2 + 5.0 ** 2)

    for region in ("WT", "TC", "ET"):
        assert metrics[f"{region}_hd95"] == pytest.approx(physical_diagonal)
        assert metrics[f"{region}_empty_fn_case_count"] == 1
    assert metrics["hd95_empty_policy"] == "physical_volume_diagonal_mm"


def test_3d_hd95_rejects_incomplete_case_reconstruction():
    accumulator = BraTS3DHD95Accumulator()
    logits = torch.full((1, 3, 3, 4), -20.0)
    target = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    accumulator.update_from_logits(
        logits,
        target,
        thresholds=(0.5, 0.5, 0.5),
        metadata=_volume_metadata(
            "client_2/case_a",
            "case_a",
            0,
            (3, 4, 2),
            (1.0, 1.0, 1.0),
        ),
    )

    with pytest.raises(RuntimeError, match="missing validation slices"):
        accumulator.compute()
