"""Contracts for server-owned buffer distribution in serial federation."""

import ast
from pathlib import Path

import torch
import torch.nn as nn

from src.server import CreamAggregator


class _BufferContractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.contrastive_dim = 2
        self.text_proj = nn.Linear(1, 1, bias=False)
        self.batch_norm = nn.BatchNorm1d(1, affine=False)
        self.register_buffer(
            "rope_cache", torch.zeros(2), persistent=False
        )

    def reset_rope_frequencies(self, verbose=False):
        del verbose
        self.rope_cache.copy_(
            torch.tensor([3.0, 5.0], device=self.rope_cache.device)
        )
        return 1


def _server():
    model = _BufferContractModel()
    return model, CreamAggregator(model, device="cpu", aggregation_method="fedavg")


def _all_named_buffers(model):
    return {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }


def _run_serial_buffer_sequence(client_order):
    model, server = _server()
    snapshot = server.capture_round_buffer_snapshot()
    client_values = {"client_1": 11.0, "client_2": 29.0, "client_3": -7.0}

    for client_id in client_order:
        server.restore_round_buffer_snapshot(
            snapshot, reason=f"before_client:{client_id}"
        )
        with torch.no_grad():
            model.batch_norm.running_mean.fill_(client_values[client_id])
            model.batch_norm.running_var.fill_(client_values[client_id] + 1.0)
            model.rope_cache.fill_(client_values[client_id] + 2.0)

    server.restore_round_buffer_snapshot(snapshot, reason="after_aggregation")
    return snapshot, _all_named_buffers(model), server.get_round_buffer_distribution_audit()


def test_client_order_cannot_change_server_owned_buffers():
    first_snapshot, first_buffers, first_audit = _run_serial_buffer_sequence(
        ["client_1", "client_2", "client_3"]
    )
    second_snapshot, second_buffers, second_audit = _run_serial_buffer_sequence(
        ["client_3", "client_2", "client_1"]
    )

    assert set(first_snapshot) == {"batch_norm.running_mean", "batch_norm.running_var", "batch_norm.num_batches_tracked"}
    assert set(first_snapshot) == set(second_snapshot)
    for name in first_snapshot:
        torch.testing.assert_close(first_buffers[name], first_snapshot[name])
        torch.testing.assert_close(second_buffers[name], second_snapshot[name])
        torch.testing.assert_close(first_buffers[name], second_buffers[name])
    torch.testing.assert_close(first_buffers["rope_cache"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(second_buffers["rope_cache"], torch.tensor([3.0, 5.0]))
    assert first_audit["buffer_key_count"] == 4
    assert first_audit["persistent_buffer_key_count"] == 3
    assert first_audit["nonpersistent_buffer_key_count"] == 1
    assert first_audit["restore_events"][-1]["reason"] == "after_aggregation"
    assert second_audit["restore_events"][-1]["reason"] == "after_aggregation"


def test_checkpoint_restores_parameters_and_server_owned_persistent_buffers():
    model, server = _server()
    checkpoint = server.get_state_dict()
    parameter_snapshot = checkpoint["trainable_parameters"]
    buffer_snapshot = checkpoint["persistent_buffers"]

    with torch.no_grad():
        model.text_proj.weight.fill_(17.0)
        model.batch_norm.running_mean.fill_(19.0)
        model.batch_norm.running_var.fill_(23.0)
        model.rope_cache.fill_(31.0)

    server.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(
        model.text_proj.weight, parameter_snapshot["text_proj.weight"]
    )
    for name, expected in buffer_snapshot.items():
        torch.testing.assert_close(dict(model.named_buffers())[name], expected)
    torch.testing.assert_close(model.rope_cache, torch.tensor([3.0, 5.0]))
    assert "model_state_dict" not in checkpoint


def test_aggregation_remains_parameter_only_and_reports_buffer_counts():
    model, server = _server()
    round_global = server.get_trainable_parameter_snapshot()
    aggregated = server.aggregate_weights(
        round_global_parameters=round_global,
        client_updates={"client_1": {"text_proj.weight": round_global["text_proj.weight"]}},
        client_modalities={"client_1": "text_only"},
        client_sample_counts={"client_1": 1},
        routing_mode="unrestricted",
    )

    assert set(aggregated) == set(round_global)
    assert not (set(aggregated) & set(dict(model.named_buffers())))
    assert server._last_aggregation_audit["parameter_key_count"] == 1
    assert server._last_aggregation_audit["buffer_key_count"] == 4
    assert server._last_aggregation_audit["persistent_buffer_key_count"] == 3
    assert server._last_aggregation_audit["nonpersistent_buffer_key_count"] == 1


def _method_source(path, class_name, method_name):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return "\n".join(source.splitlines()[method.lineno - 1:method.end_lineno])


def test_strict_round_path_has_no_model_state_dict_aggregation_entrypoint():
    root = Path(__file__).resolve().parents[1]
    trainer_round_source = _method_source(
        root / "src" / "federated_trainer.py",
        "FederatedTrainer",
        "_train_single_round",
    )
    server_aggregate_source = _method_source(
        root / "src" / "server.py", "CreamAggregator", "aggregate_weights"
    )
    server_checkpoint_source = _method_source(
        root / "src" / "server.py", "CreamAggregator", "load_state_dict"
    )

    assert "self.global_model.load_state_dict" not in trainer_round_source
    assert "load_trainable_state_dict" not in trainer_round_source
    assert "self.global_model.load_state_dict" not in server_aggregate_source
    assert "self.global_model.load_state_dict" not in server_checkpoint_source
    assert "capture_round_buffer_snapshot" in trainer_round_source
    assert trainer_round_source.count("restore_round_buffer_snapshot") == 4
