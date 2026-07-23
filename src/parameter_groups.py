from typing import FrozenSet


TEXT_ADAPTER = "TEXT_ADAPTER"
VISION_ADAPTER = "VISION_ADAPTER"
TEXT_PARAMS = "TEXT_PARAMS"
IMAGE_PARAMS = "IMAGE_PARAMS"
COMPAT_FALLBACK = "COMPAT_FALLBACK"

TEXT_ADAPTER_KEYWORDS = ("text_adapter",)
VISION_ADAPTER_KEYWORDS = ("adapters.", "wrapped_blocks.", "lora")
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
    "text_prompt_encoder",
)

_TEXT_MODALITIES: FrozenSet[str] = frozenset({"text_only", "multimodal"})
_VISION_MODALITIES: FrozenSet[str] = frozenset({"image_only", "multimodal"})


def classify_parameter(param_name: str) -> str:
    if any(keyword in param_name for keyword in TEXT_ADAPTER_KEYWORDS):
        return TEXT_ADAPTER
    if any(keyword in param_name for keyword in VISION_ADAPTER_KEYWORDS):
        return VISION_ADAPTER
    if any(keyword in param_name for keyword in TEXT_PARAM_KEYWORDS):
        return TEXT_PARAMS
    if any(keyword in param_name for keyword in IMAGE_PARAM_KEYWORDS):
        return IMAGE_PARAMS
    return COMPAT_FALLBACK


def allowed_modalities(param_group: str) -> FrozenSet[str]:
    if param_group in {TEXT_ADAPTER, TEXT_PARAMS}:
        return _TEXT_MODALITIES
    if param_group in {VISION_ADAPTER, IMAGE_PARAMS}:
        return _VISION_MODALITIES
    return frozenset()


def is_vision_parameter(param_name: str) -> bool:
    return classify_parameter(param_name) in {VISION_ADAPTER, IMAGE_PARAMS}
