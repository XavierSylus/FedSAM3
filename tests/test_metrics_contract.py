import math

import pytest
import torch

from src.metrics import BraTSMetricAccumulator


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
