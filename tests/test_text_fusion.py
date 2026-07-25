import torch

from src.models.text_fusion import GatedFusion


def _build_fusion() -> GatedFusion:
    return GatedFusion(
        image_channels=8,
        text_dim=6,
        hidden_dim=4,
    ).eval()


def test_gated_fusion_preserves_spatial_shape_and_finite_values():
    torch.manual_seed(3407)
    fusion = _build_fusion()
    image_features = torch.randn(2, 8, 4, 4)
    text_features = torch.randn(2, 6)

    fused = fusion(image_features, text_features)

    assert fused.shape == image_features.shape
    assert torch.isfinite(fused).all()


def test_gated_fusion_propagates_both_modalities():
    torch.manual_seed(3407)
    fusion = _build_fusion()
    image_features = torch.randn(2, 8, 4, 4, requires_grad=True)
    text_features = torch.randn(2, 6, requires_grad=True)

    fused = fusion(image_features, text_features)
    image_gradient, text_gradient = torch.autograd.grad(
        fused.square().mean(),
        (image_features, text_features),
    )

    for gradient in (image_gradient, text_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0

