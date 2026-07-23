import pytest
import torch

from data_processing.brats_region_contract import (
    REGION_NAMES,
    brats_labels_to_regions,
    logits_to_brats_labels,
    regions_to_brats_labels,
)


def test_raw_brats_labels_convert_to_wt_tc_et_and_round_trip():
    labels = torch.tensor(
        [
            [0, 1, 2, 4],
            [4, 2, 1, 0],
        ],
        dtype=torch.long,
    )

    regions = brats_labels_to_regions(labels)

    expected = torch.tensor(
        [
            [[0, 1, 1, 1], [1, 1, 1, 0]],
            [[0, 1, 0, 1], [1, 0, 1, 0]],
            [[0, 0, 0, 1], [1, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    assert REGION_NAMES == ("WT", "TC", "ET")
    assert regions.shape == (3, 2, 4)
    assert regions.dtype == torch.float32
    assert torch.equal(regions, expected)
    assert torch.equal(regions_to_brats_labels(regions, channel_dim=0), labels)


def test_regions_to_labels_applies_nested_closure():
    non_nested_regions = torch.tensor(
        [
            [[0, 1, 0, 0]],
            [[0, 0, 1, 0]],
            [[0, 0, 0, 1]],
        ],
        dtype=torch.float32,
    )

    labels = regions_to_brats_labels(non_nested_regions, channel_dim=0)

    assert torch.equal(labels, torch.tensor([[0, 2, 1, 4]], dtype=torch.long))


def test_logits_to_labels_uses_per_channel_thresholds():
    probabilities = torch.tensor(
        [
            [[0.40, 0.60, 0.10, 0.10]],
            [[0.10, 0.10, 0.80, 0.10]],
            [[0.10, 0.10, 0.10, 0.95]],
        ],
        dtype=torch.float32,
    )

    labels = logits_to_brats_labels(
        torch.logit(probabilities),
        thresholds=(0.5, 0.7, 0.9),
        channel_dim=0,
    )

    assert torch.equal(labels, torch.tensor([[0, 2, 1, 4]], dtype=torch.long))


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (torch.tensor([[0.0, 1.0]]), "integer dtype"),
        (torch.tensor([[0, 3]], dtype=torch.long), "unsupported BraTS labels"),
    ],
)
def test_raw_label_contract_rejects_invalid_input(labels, message):
    with pytest.raises((TypeError, ValueError), match=message):
        brats_labels_to_regions(labels)


def test_region_contract_rejects_invalid_channel_shape_and_values():
    with pytest.raises(ValueError, match="exactly 3 channels"):
        regions_to_brats_labels(torch.zeros(2, 4, 4), channel_dim=0)

    invalid_regions = torch.zeros(3, 4, 4)
    invalid_regions[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        regions_to_brats_labels(invalid_regions, channel_dim=0)
