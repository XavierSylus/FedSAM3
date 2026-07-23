from typing import FrozenSet


TEXT_ADAPTER = "TEXT_ADAPTER"
VISION_ADAPTER = "VISION_ADAPTER"
TEXT_PARAMS = "TEXT_PARAMS"
IMAGE_PARAMS = "IMAGE_PARAMS"
FUSION_PARAMS = "FUSION_PARAMS"

TEXT_ADAPTER_KEYWORDS = ("text_adapter",)
VISION_ADAPTER_KEYWORDS = (
    "adapters.",
    "wrapped_blocks.",
    "lora",
    "_adapter_conv",
)
FUSION_PARAM_KEYWORDS = (
    "fusion_head._text_projection",
    "fusion_head._fusion_gate",
    "text_prompt_encoder",
)
TEXT_PARAM_KEYWORDS = ("text_encoder", "text_proj")
IMAGE_PARAM_KEYWORDS = (
    "image_encoder",
    "backbone",
    "neck",
    "sam3_model",
    "mask_decoder",
    "segmentation_head",
    "prompt_encoder",
    "image_proj",
    "_output_conv",
    "medical_seg_head",
)

_TEXT_MODALITIES: FrozenSet[str] = frozenset({"text_only", "multimodal"})
_VISION_MODALITIES: FrozenSet[str] = frozenset({"image_only", "multimodal"})
_FUSION_MODALITIES: FrozenSet[str] = frozenset({"multimodal"})


def classify_parameter(param_name: str) -> str:
    if not param_name:
        raise ValueError("Parameter name must be non-empty")
    if any(keyword in param_name for keyword in TEXT_ADAPTER_KEYWORDS):
        return TEXT_ADAPTER
    if any(keyword in param_name for keyword in VISION_ADAPTER_KEYWORDS):
        return VISION_ADAPTER
    if any(keyword in param_name for keyword in FUSION_PARAM_KEYWORDS):
        return FUSION_PARAMS
    if any(keyword in param_name for keyword in TEXT_PARAM_KEYWORDS):
        return TEXT_PARAMS
    if any(keyword in param_name for keyword in IMAGE_PARAM_KEYWORDS):
        return IMAGE_PARAMS
    raise ValueError(
        f"Unclassified trainable/uploaded parameter: {param_name}"
    )


def allowed_modalities(param_group: str) -> FrozenSet[str]:
    if param_group in {TEXT_ADAPTER, TEXT_PARAMS}:
        return _TEXT_MODALITIES
    if param_group in {VISION_ADAPTER, IMAGE_PARAMS}:
        return _VISION_MODALITIES
    if param_group == FUSION_PARAMS:
        return _FUSION_MODALITIES
    raise ValueError(f"Unknown parameter group: {param_group}")


def is_vision_parameter(param_name: str) -> bool:
    return classify_parameter(param_name) in {VISION_ADAPTER, IMAGE_PARAMS}
