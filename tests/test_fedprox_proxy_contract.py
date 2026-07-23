import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "src" / "client.py"
TRAINER_PATH = PROJECT_ROOT / "src" / "federated_trainer.py"
MODEL_PATH = PROJECT_ROOT / "src" / "integrated_model.py"
MAIN_PATH = PROJECT_ROOT / "main.py"


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_text_only_loss_does_not_implement_a_second_fedprox_term():
    method = _class_method(CLIENT_PATH, "TextOnlyTrainer", "compute_loss")
    names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
    string_literals = {
        node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "_MU" not in names
    assert "global_weights" not in string_literals


def test_round_uses_one_shared_fail_fast_proxy_for_all_clients():
    method = _class_method(
        TRAINER_PATH,
        "FederatedTrainer",
        "_prepare_round_global_reps",
    )
    source = ast.unparse(method)

    assert "proxy_client_id" in source
    assert "multimodal" in source
    assert "raise RuntimeError" in source
    assert "global_image_rep" in source
    assert "global_text_rep" in source


def test_missing_global_representations_never_fall_back_to_zero():
    method = _class_method(CLIENT_PATH, "BaseClientTrainer", "_prepare_global_reps")
    source = ast.unparse(method)

    assert "raise ValueError" in source
    assert "_zeros" not in source


def test_fusion_parameters_are_all_created_before_local_training():
    initializer = _class_method(MODEL_PATH, "MultimodalFusionHead", "__init__")
    apply_fusion = _class_method(
        MODEL_PATH,
        "MultimodalFusionHead",
        "apply_fusion",
    )
    extract_features = _class_method(
        MODEL_PATH,
        "SAM3MedicalIntegrated",
        "extract_features",
    )

    initializer_source = ast.unparse(initializer)
    apply_source = ast.unparse(apply_fusion)
    extract_source = ast.unparse(extract_features)

    assert "self._text_projection" in initializer_source
    assert "self._fusion_gate" in initializer_source
    assert "nn.Linear" not in apply_source
    assert "nn.Conv2d" not in apply_source
    assert "_enc_proj" not in extract_source


def test_upload_contains_only_named_trainable_parameters():
    method = _class_method(CLIENT_PATH, "BaseClientTrainer", "get_model_state")
    source = ast.unparse(method)

    assert "model.named_parameters()" in source
    assert "model.state_dict()" not in source
    assert "ALWAYS_INCLUDE_PREFIXES" not in source


def test_multimodal_text_representation_has_no_zero_fallback():
    method = _class_method(
        CLIENT_PATH,
        "MultimodalTrainer",
        "get_return_values",
    )
    source = ast.unparse(method)

    assert "raise RuntimeError" in source
    assert "zeros_like" not in source


def test_trainable_parameter_registry_matches_optimizer_contract():
    get_trainable = _class_method(
        MODEL_PATH,
        "SAM3MedicalIntegrated",
        "get_trainable_params",
    )
    initializer = _class_method(
        MODEL_PATH,
        "SAM3MedicalIntegrated",
        "__init__",
    )
    trainable_source = ast.unparse(get_trainable)
    initializer_source = ast.unparse(initializer)

    assert "self.fusion_head.parameters()" in trainable_source
    assert "self.text_prompt_encoder.parameters()" in trainable_source
    assert "parameter.requires_grad_(False)" in initializer_source


def test_main_supports_isolated_smoke_log_directory():
    parser_builder = next(
        node
        for node in ast.parse(MAIN_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
    )
    override_function = next(
        node
        for node in ast.parse(MAIN_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_cli_overrides"
    )

    assert "--log_dir" in ast.unparse(parser_builder)
    assert "config.log_dir = args.log_dir" in ast.unparse(override_function)
