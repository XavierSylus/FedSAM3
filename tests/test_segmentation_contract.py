from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.config_manager import FederatedConfig
from src.cream_losses import BraTSDiceBCELoss


def _nested_target() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1, 1], [1, 0]],
                [[1, 0], [1, 0]],
                [[0, 0], [1, 0]],
            ]
        ],
        dtype=torch.float32,
    )


def test_dice_bce_matches_declared_formula_and_preserves_gradients():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target = _nested_target()
    criterion = BraTSDiceBCELoss(
        dice_weight=1.25,
        bce_weight=0.75,
        smooth=1.0,
    )

    loss = criterion(logits, target)

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    expected_dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    expected_bce = F.binary_cross_entropy_with_logits(logits, target)
    expected = 1.25 * expected_dice + 0.75 * expected_bce

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_empty_target_channels_are_not_skipped():
    logits = torch.full((1, 3, 2, 2), 4.0, requires_grad=True)
    target = torch.zeros_like(logits, dtype=torch.float32)
    criterion = BraTSDiceBCELoss(
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0,
    )

    loss = criterion(logits, target)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (torch.zeros(1, 3, 2, 2, dtype=torch.long), "torch.float32"),
        (torch.zeros(1, 1, 2, 2), r"\[B, 3, H, W\]"),
        (torch.full((1, 3, 2, 2), 0.5), "binary"),
    ],
)
def test_target_contract_rejects_invalid_dtype_shape_and_values(target, message):
    criterion = BraTSDiceBCELoss(
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        criterion(torch.zeros(1, 3, 2, 2), target)


def test_target_contract_rejects_non_nested_regions():
    target = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
    target[:, 2, 0, 0] = 1.0
    criterion = BraTSDiceBCELoss(
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0,
    )

    with pytest.raises(ValueError, match="ET subset TC subset WT"):
        criterion(torch.zeros_like(target), target)


def test_yaml_segmentation_contract_has_explicit_ordered_thresholds(tmp_path: Path):
    config_path = tmp_path / "contract.yaml"
    config_path.write_text(
        """
model:
  num_classes: 3
segmentation:
  loss: dice_bce
  dice_weight: 1.0
  bce_weight: 1.0
  smooth: 1.0
  thresholds:
    WT: 0.5
    TC: 0.6
    ET: 0.7
""".strip(),
        encoding="utf-8",
    )

    config = FederatedConfig.from_yaml(str(config_path))

    assert config.segmentation_loss == "dice_bce"
    assert config.seg_dice_weight == 1.0
    assert config.seg_bce_weight == 1.0
    assert config.seg_dice_smooth == 1.0
    assert config.segmentation_thresholds == (0.5, 0.6, 0.7)
