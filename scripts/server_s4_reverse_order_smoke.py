"""Run the S4 reverse-client-order real-data smoke gate on the target server."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import FederatedConfig  # noqa: E402
from src.federated_trainer import FederatedTrainer  # noqa: E402


S4_KEYS = {"client_order", "forward_evidence_dir"}
CLIENT_MAPPING_NAMES = (
    "client_configs",
    "client_trainers",
    "client_states",
    "client_sample_counts",
    "_initial_loader_randomness",
)
PROTOCOL_COMPARISON_KEYS = (
    "seed",
    "rounds",
    "batch_size",
    "local_epochs",
    "learning_rate",
    "seg_head_lr",
    "adapter_lr",
    "weight_decay",
    "lambda_cream",
    "aggregation_method",
    "routing_mode",
    "sample_weight_unit",
    "unoptimized_update_policy",
    "client_sample_count_unit",
    "client_sample_counts",
    "baseline_method",
    "fedprox_mu",
    "client_init_policy",
    "persist_client_optimizer",
    "strict_protocol_check",
    "proxy_client_id",
    "segmentation",
    "data_root",
    "clients",
    "missing_modality_client_ratio",
)
REGIONS = ("WT", "TC", "ET")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _canonical_json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _load_s4_contract(
    config_path: Path,
) -> tuple[FederatedConfig, dict[str, Any]]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise TypeError("S4 configuration must be a YAML mapping")
    contract = raw_config.get("s4_smoke")
    if not isinstance(contract, dict) or set(contract) != S4_KEYS:
        raise ValueError(f"s4_smoke must define exactly {sorted(S4_KEYS)}")

    client_order = contract["client_order"]
    if (
        not isinstance(client_order, list)
        or not client_order
        or any(
            not isinstance(client_id, str) or not client_id
            for client_id in client_order
        )
        or len(set(client_order)) != len(client_order)
    ):
        raise ValueError("s4_smoke.client_order must contain unique client IDs")
    forward_evidence_dir = contract["forward_evidence_dir"]
    if not isinstance(forward_evidence_dir, str) or not forward_evidence_dir.strip():
        raise ValueError("s4_smoke.forward_evidence_dir must be a non-empty path")

    enabled_clients = [
        client
        for client in raw_config.get("federated", {}).get("clients", [])
        if bool(client.get("enabled", True))
    ]
    enabled_order = [str(client.get("client_id")) for client in enabled_clients]
    if enabled_order != client_order:
        raise ValueError(
            "Enabled federated clients must be declared in s4_smoke.client_order"
        )

    config = FederatedConfig.from_yaml(str(config_path))
    if config.rounds != 1:
        raise ValueError("S4 is a one-round smoke gate")
    return config, contract


def _evidence_paths(evidence_dir: Path, completed_round: int) -> dict[str, Path]:
    checkpoint_dir = evidence_dir / "checkpoints"
    return {
        "final_model": checkpoint_dir / "final_model.pth",
        "final_checkpoint": (
            checkpoint_dir / f"checkpoint_round_{completed_round}.pth"
        ),
        "training_history": checkpoint_dir / "training_history.json",
        "run_metadata": checkpoint_dir / "run_metadata.json",
    }


def _load_forward_evidence(
    evidence_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    initial_paths = _evidence_paths(evidence_dir, completed_round=1)
    for path in (
        initial_paths["training_history"],
        initial_paths["run_metadata"],
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required S2 evidence is missing: {path}")
    history = _load_json(initial_paths["training_history"])
    metadata = _load_json(initial_paths["run_metadata"])
    rounds = history.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("S2 evidence has no completed round")
    completed_round = int(rounds[-1])
    paths = _evidence_paths(evidence_dir, completed_round)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required S2 evidence is missing: {path}")
    return history, metadata, paths


def _reorder_client_state(
    trainer: FederatedTrainer,
    client_order: list[str],
) -> None:
    for mapping_name in CLIENT_MAPPING_NAMES:
        mapping = getattr(trainer, mapping_name, None)
        if not isinstance(mapping, dict):
            raise TypeError(f"Trainer client state is not a dictionary: {mapping_name}")
        normalized = {
            trainer._normalize_client_id(client_id): client_id
            for client_id in mapping
        }
        desired = [
            trainer._normalize_client_id(client_id)
            for client_id in client_order
        ]
        if set(normalized) != set(desired) or len(normalized) != len(desired):
            raise ValueError(
                f"{mapping_name} does not match s4_smoke.client_order"
            )
        reordered = {
            normalized[normalized_id]: mapping[normalized[normalized_id]]
            for normalized_id in desired
        }
        setattr(trainer, mapping_name, reordered)


def _expected_restore_reasons(client_order: list[str]) -> list[str]:
    reasons: list[str] = []
    for client_id in client_order:
        reasons.extend(
            (
                f"before_client:{client_id}",
                f"after_client:{client_id}",
            )
        )
    reasons.extend(("before_aggregation", "after_aggregation"))
    return reasons


def _buffer_policy_passes(
    buffer_distribution: Any,
    client_order: list[str],
) -> bool:
    if not isinstance(buffer_distribution, dict):
        return False
    events = buffer_distribution.get("restore_events")
    if not isinstance(events, list):
        return False
    if any(not isinstance(event, dict) for event in events):
        return False
    if [event.get("reason") for event in events] != _expected_restore_reasons(
        client_order
    ):
        return False
    persistent_count = buffer_distribution.get("persistent_buffer_key_count")
    nonpersistent_count = buffer_distribution.get(
        "nonpersistent_buffer_key_count"
    )
    snapshot_count = buffer_distribution.get("snapshot_key_count")
    if snapshot_count != persistent_count:
        return False
    return all(
        isinstance(event, dict)
        and event.get("restored_persistent_buffer_key_count") == persistent_count
        and event.get("rebuilt_nonpersistent_buffer_key_count")
        == nonpersistent_count
        for event in events
    )


def _empty_region_metrics_pass(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    for region in REGIONS:
        count_keys = (
            f"{region}_both_empty_count",
            f"{region}_empty_fp_count",
            f"{region}_empty_fn_count",
            f"{region}_both_nonempty_count",
        )
        values = [metrics.get(key) for key in count_keys]
        sample_count = metrics.get(f"{region}_num_samples")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            )
            or sum(values) != sample_count
        ):
            return False
        for metric_name in ("dice", "iou", "hd95"):
            value = metrics.get(f"{region}_{metric_name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


def _audit_run(
    *,
    history: dict[str, Any],
    metadata: dict[str, Any],
    expected_order: list[str],
    routing_mode: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    reproducibility = history.get("round_reproducibility")
    aggregations = history.get("aggregation_audits")
    effectiveness = history.get("parameter_group_effectiveness")
    client_losses = history.get("client_losses")
    if not all(
        isinstance(value, list) and len(value) == 1
        for value in (
            reproducibility,
            aggregations,
            effectiveness,
            client_losses,
        )
    ):
        raise ValueError("Smoke evidence must contain exactly one completed round")

    round_record = reproducibility[0]
    aggregation = aggregations[0]
    effectiveness_clients = effectiveness[0].get("clients", {})
    loss_clients = client_losses[0]
    buffer_distribution = aggregation.get("buffer_distribution")
    data_manifest = metadata.get("data_manifest", {})
    expected_set = set(expected_order)
    active_client_ids = aggregation.get("active_client_ids", [])
    sample_counts = aggregation.get("client_sample_counts", {})
    checks = {
        "completed_one_round": history.get("rounds") == [1],
        "round_participation_order": (
            round_record.get("client_participation_order") == expected_order
        ),
        "metadata_participation_order": (
            data_manifest.get("client_participation_order") == expected_order
        ),
        "client_loss_order": (
            isinstance(loss_clients, dict)
            and list(loss_clients) == expected_order
        ),
        "all_clients_effective": (
            isinstance(effectiveness_clients, dict)
            and set(effectiveness_clients) == expected_set
        ),
        "all_clients_aggregated": (
            isinstance(active_client_ids, list)
            and set(active_client_ids) == expected_set
        ),
        "positive_private_case_counts": (
            isinstance(sample_counts, dict)
            and set(sample_counts) == expected_set
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in sample_counts.values()
            )
        ),
        "routing_mode": aggregation.get("routing_mode") == routing_mode,
        "buffer_distribution": _buffer_policy_passes(
            buffer_distribution,
            expected_order,
        ),
        "empty_region_metrics": _empty_region_metrics_pass(
            history.get("final_val_metrics")
        ),
    }
    details = {
        "git_commit": metadata.get("git_commit"),
        "client_participation_order": round_record.get(
            "client_participation_order"
        ),
        "buffer_distribution": buffer_distribution,
        "final_val_metrics": history.get("final_val_metrics"),
    }
    return checks, details


def _protocols_match(
    forward_metadata: dict[str, Any],
    reverse_metadata: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    forward = forward_metadata.get("protocol_payload")
    reverse = reverse_metadata.get("protocol_payload")
    if not isinstance(forward, dict) or not isinstance(reverse, dict):
        return False, {}
    comparison = {}
    for key in PROTOCOL_COMPARISON_KEYS:
        forward_value = _canonical_json_value(forward.get(key))
        reverse_value = _canonical_json_value(reverse.get(key))
        comparison[key] = {
            "forward": forward_value,
            "reverse": reverse_value,
            "equal": forward_value == reverse_value,
        }
    return all(item["equal"] for item in comparison.values()), comparison


def _artifacts_exist(paths: dict[str, Path]) -> tuple[bool, dict[str, str]]:
    serialized = {name: str(path) for name, path in paths.items()}
    return all(path.is_file() for path in paths.values()), serialized


def run(config_path: Path) -> int:
    config, contract = _load_s4_contract(config_path)
    client_order = [str(client_id) for client_id in contract["client_order"]]
    forward_order = list(reversed(client_order))
    log_dir = Path(config.log_dir)
    forward_evidence_dir = Path(str(contract["forward_evidence_dir"]))
    if forward_evidence_dir.resolve() == log_dir.resolve():
        raise ValueError("S4 output and S2 evidence directories must differ")

    forward_history, forward_metadata, forward_paths = _load_forward_evidence(
        forward_evidence_dir
    )
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(
            f"S4 log directory is not empty; preserve it before rerun: {log_dir}"
        )

    trainer = FederatedTrainer(config)
    trainer.setup_environment()
    trainer.setup_clients()
    _reorder_client_state(trainer, client_order)
    trainer.setup_validation()
    trainer.setup_logging()
    trainer.training_history["run_metadata"] = trainer._collect_run_metadata()

    trainer._train_single_round(config.rounds)
    trainer._evaluate_validation(config.rounds)
    trainer._finalize_training()

    reverse_history = trainer.training_history
    reverse_metadata = reverse_history["run_metadata"]
    forward_checks, forward_details = _audit_run(
        history=forward_history,
        metadata=forward_metadata,
        expected_order=forward_order,
        routing_mode=config.routing_mode,
    )
    reverse_checks, reverse_details = _audit_run(
        history=reverse_history,
        metadata=reverse_metadata,
        expected_order=client_order,
        routing_mode=config.routing_mode,
    )
    protocols_match, protocol_comparison = _protocols_match(
        forward_metadata,
        reverse_metadata,
    )
    reverse_paths = _evidence_paths(log_dir, completed_round=config.rounds)
    forward_artifacts, forward_artifact_paths = _artifacts_exist(forward_paths)
    reverse_artifacts, reverse_artifact_paths = _artifacts_exist(reverse_paths)

    checks = {
        "forward_s2_evidence": all(forward_checks.values()),
        "reverse_s4_evidence": all(reverse_checks.values()),
        "training_protocol_match": protocols_match,
        "forward_artifacts": forward_artifacts,
        "reverse_artifacts": reverse_artifacts,
    }
    result = {
        "gate": "S4",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_forward_order": forward_order,
        "expected_reverse_order": client_order,
        "forward": {
            "checks": forward_checks,
            "details": forward_details,
            "artifact_paths": forward_artifact_paths,
        },
        "reverse": {
            "checks": reverse_checks,
            "details": reverse_details,
            "artifact_paths": reverse_artifact_paths,
        },
        "protocol_comparison": protocol_comparison,
        "config_path": str(config_path),
    }
    result_path = log_dir / "s4_smoke_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[S4] Result saved to: {result_path}")
    print(f"[S4] Gate status: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return run(config_path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
