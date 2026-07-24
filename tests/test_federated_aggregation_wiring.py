"""Static contracts for strict trainer-to-server aggregation wiring."""

import ast
import textwrap
from pathlib import Path


_TRAINER_PATH = Path(__file__).resolve().parents[1] / "src" / "federated_trainer.py"
_TRAINER_SOURCE = _TRAINER_PATH.read_text(encoding="utf-8")
_TRAINER_TREE = ast.parse(_TRAINER_SOURCE)


def _method(name):
    trainer_class = next(
        node
        for node in _TRAINER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "FederatedTrainer"
    )
    return next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _method_source(name):
    method = _method(name)
    lines = _TRAINER_SOURCE.splitlines()
    return "\n".join(lines[method.lineno - 1:method.end_lineno])


def _raise_message_literals(name):
    messages = []
    for node in ast.walk(_method(name)):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        messages.append(
            "".join(
                child.value
                for child in ast.walk(node.exc)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
        )
    return messages


def test_trainer_calls_strict_parameterwise_aggregation_api():
    aggregation_calls = [
        node
        for node in ast.walk(_method("_train_single_round"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "aggregate_weights"
    ]

    assert len(aggregation_calls) == 1
    call = aggregation_calls[0]
    assert not call.args
    assert {keyword.arg for keyword in call.keywords} == {
        "round_global_parameters",
        "client_updates",
        "client_modalities",
        "client_sample_counts",
        "routing_mode",
    }


def test_trainer_uses_private_dataset_length_as_the_only_sample_weight():
    derive_counts_source = _method_source("_derive_private_case_counts")
    derive_counts_tree = ast.parse(textwrap.dedent(derive_counts_source))
    dataset_length_calls = [
        node
        for node in ast.walk(derive_counts_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "dataset"
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "private_loader"
    ]

    assert len(dataset_length_calls) == 1
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "private_case_counts"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "private_case_count"
        for node in ast.walk(derive_counts_tree)
    )
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "private_case_count"
        and any(isinstance(operator, ast.LtE) for operator in node.ops)
        for node in ast.walk(derive_counts_tree)
    )
    assert "batch_size" not in derive_counts_source
    assert "local_epochs" not in derive_counts_source


def test_trainer_rejects_invalid_optimizer_uploads_without_state_dict_fallback():
    round_source = _method_source("_train_single_round")
    error_messages = _raise_message_literals("_train_single_round")

    assert "if updated_weights is None" in round_source
    assert "upload keys do not equal its optimizer" in round_source
    assert any(
        "uploaded parameter absent from the round-global snapshot" in message
        for message in error_messages
    )
    assert any(
        "produced a non-finite parameter delta" in message
        for message in error_messages
    )
    assert "falling back to global model" not in round_source
    assert "round_client_reps" not in round_source


def test_trainer_records_private_case_unit_and_server_aggregation_audit():
    identity_source = _method_source("_build_run_identity")
    round_source = _method_source("_train_single_round")

    assert '"client_sample_count_unit": "private_case_count"' in identity_source
    assert '"client_sample_counts"' in identity_source
    assert "self.training_history['aggregation_audits'].append" in round_source
    assert "Server did not produce the required aggregation audit" in round_source


def test_legacy_routing_switch_and_group_protocol_are_absent_from_trainer():
    assert "use_decoupled_agg" not in _TRAINER_SOURCE
    assert "_validate_group_protocol" not in _TRAINER_SOURCE
    assert "Group A" not in _TRAINER_SOURCE
    assert "Group B" not in _TRAINER_SOURCE
    assert "Group C" not in _TRAINER_SOURCE
