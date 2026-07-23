from collections.abc import Sequence

import torch


REGION_NAMES = ("WT", "TC", "ET")
BRATS_LABEL_VALUES = (0, 1, 2, 4)


def _resolve_channel_dim(tensor: torch.Tensor, channel_dim: int) -> int:
    if tensor.ndim < 3:
        raise ValueError(
            f"Region tensor must have at least 3 dimensions, got shape {tuple(tensor.shape)}"
        )
    resolved = channel_dim if channel_dim >= 0 else tensor.ndim + channel_dim
    if resolved < 0 or resolved >= tensor.ndim:
        raise ValueError(
            f"channel_dim={channel_dim} is invalid for shape {tuple(tensor.shape)}"
        )
    if tensor.shape[resolved] != len(REGION_NAMES):
        raise ValueError(
            f"Region tensor must contain exactly 3 channels in order {REGION_NAMES}, "
            f"got shape {tuple(tensor.shape)} with channel_dim={channel_dim}"
        )
    return resolved


def _validate_binary_regions(
    regions: torch.Tensor,
    channel_dim: int,
) -> tuple[torch.Tensor, int]:
    if not isinstance(regions, torch.Tensor):
        raise TypeError("regions must be a torch.Tensor")
    resolved = _resolve_channel_dim(regions, channel_dim)
    if regions.is_complex():
        raise TypeError("regions must use a real-valued or boolean dtype")
    if torch.is_floating_point(regions) and not torch.isfinite(regions).all():
        raise ValueError("regions must contain only finite values")
    if torch.any((regions != 0) & (regions != 1)):
        raise ValueError("regions must be binary with values in {0, 1}")
    return regions.movedim(resolved, 0).bool(), resolved


def brats_labels_to_regions(labels: torch.Tensor) -> torch.Tensor:
    """Convert one BraTS label map to overlapping float32 channels [WT, TC, ET]."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.ndim not in (2, 3):
        raise ValueError(
            f"BraTS labels must have shape [H, W] or [D, H, W], got {tuple(labels.shape)}"
        )
    if labels.dtype == torch.bool or torch.is_floating_point(labels) or labels.is_complex():
        raise TypeError("BraTS labels must use an integer dtype")

    invalid_values = [
        value
        for value in torch.unique(labels).tolist()
        if value not in BRATS_LABEL_VALUES
    ]
    if invalid_values:
        raise ValueError(
            f"Found unsupported BraTS labels {invalid_values}; "
            f"expected only {BRATS_LABEL_VALUES}"
        )

    wt = labels != 0
    tc = (labels == 1) | (labels == 4)
    et = labels == 4
    return torch.stack((wt, tc, et), dim=0).to(dtype=torch.float32)


def close_nested_regions(
    regions: torch.Tensor,
    *,
    channel_dim: int,
) -> torch.Tensor:
    """Apply ET subset TC subset WT closure and return a boolean tensor."""
    channel_first, resolved = _validate_binary_regions(regions, channel_dim)
    wt, tc, et = channel_first.unbind(dim=0)
    closed_tc = tc | et
    closed_wt = wt | closed_tc
    closed = torch.stack((closed_wt, closed_tc, et), dim=0)
    return closed.movedim(0, resolved)


def regions_to_brats_labels(
    regions: torch.Tensor,
    *,
    channel_dim: int,
) -> torch.Tensor:
    """Convert overlapping [WT, TC, ET] channels to nested BraTS labels 0/1/2/4."""
    closed = close_nested_regions(regions, channel_dim=channel_dim)
    resolved = channel_dim if channel_dim >= 0 else closed.ndim + channel_dim
    wt, tc, et = closed.movedim(resolved, 0).unbind(dim=0)

    labels = torch.zeros_like(wt, dtype=torch.long)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 4
    return labels


def logits_to_brats_labels(
    logits: torch.Tensor,
    *,
    thresholds: Sequence[float],
    channel_dim: int,
) -> torch.Tensor:
    """Threshold three sigmoid logits, close their nesting, and emit labels 0/1/2/4."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not torch.is_floating_point(logits) or logits.is_complex():
        raise TypeError("logits must use a real floating-point dtype")
    resolved = _resolve_channel_dim(logits, channel_dim)
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    if len(thresholds) != len(REGION_NAMES):
        raise ValueError(
            f"thresholds must contain one value per channel {REGION_NAMES}"
        )

    threshold_tensor = torch.as_tensor(
        thresholds,
        dtype=logits.dtype,
        device=logits.device,
    )
    if not torch.isfinite(threshold_tensor).all():
        raise ValueError("thresholds must contain only finite values")
    if torch.any((threshold_tensor <= 0) | (threshold_tensor >= 1)):
        raise ValueError("thresholds must be strictly between 0 and 1")

    channel_first = logits.movedim(resolved, 0)
    threshold_shape = (len(REGION_NAMES),) + (1,) * (channel_first.ndim - 1)
    regions = torch.sigmoid(channel_first) >= threshold_tensor.view(threshold_shape)
    return regions_to_brats_labels(regions, channel_dim=0)
