"""Audit and re-evaluate one completed formal 2x2 experiment cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROUNDS = 60
EXPECTED_METRIC_CONTRACT = {
    "hd95_unit": "mm",
    "hd95_dimension": "3d_case",
    "hd95_aggregation": "macro_case_then_region",
    "hd95_empty_policy": "physical_volume_diagonal_mm",
}
DIAGNOSTIC_FIELDNAMES = [
    "round",
    "row_type",
    "client_id",
    "client_a",
    "client_b",
    "modality_a",
    "modality_b",
    "parameter_group",
    "update_l2",
    "reference_l2",
    "relative_drift",
    "update_rms",
    "numel",
    "parameter_count",
    "nonzero_parameter_count",
    "nonzero_parameter_ratio",
    "sample_weight",
    "cosine_similarity",
    "angle_deg",
    "is_negative",
    "conflict_status",
    "shared_numel",
    "shared_parameter_count",
    "pair_count",
    "negative_pair_count",
    "negative_cosine_ratio",
    "conflict_rate",
    "mean_cosine_similarity",
    "mean_angle_deg",
    "shared_pair_count",
    "no_shared_pair_count",
    "undefined_pair_count",
    "routing_mode",
    "aggregation_client_ids",
    "aggregation_participation",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def require_exact_rounds(
    name: str,
    values: Sequence[Any],
    *,
    expected_rounds: int,
) -> None:
    expected = list(range(1, expected_rounds + 1))
    actual = [int(value) for value in values]
    if actual != expected:
        raise ValueError(
            f"{name} must contain exact rounds 1..{expected_rounds}; "
            f"got count={len(actual)}, first={actual[:3]}, last={actual[-3:]}"
        )


def expected_diagnostic_rows(
    rounds: Sequence[int],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rounds) != len(diagnostics):
        raise ValueError("Diagnostic history length does not match completed rounds")
    rows: list[dict[str, Any]] = []
    for round_num, round_diagnostics in zip(rounds, diagnostics):
        for client_id, groups in round_diagnostics.get("client_drift", {}).items():
            for parameter_group, metrics in groups.items():
                rows.append(
                    {
                        "round": int(round_num),
                        "row_type": "client_drift",
                        "client_id": client_id,
                        "parameter_group": parameter_group,
                        **metrics,
                    }
                )
        for item in round_diagnostics.get("pairwise_conflicts", []):
            rows.append(
                {
                    "round": int(round_num),
                    "row_type": "pairwise_conflict",
                    **item,
                }
            )
        for parameter_group, metrics in round_diagnostics.get(
            "conflict_summary", {}
        ).items():
            rows.append(
                {
                    "round": int(round_num),
                    "row_type": "conflict_summary",
                    "parameter_group": parameter_group,
                    **metrics,
                }
            )
        for parameter_group, metrics in round_diagnostics.get(
            "global_drift", {}
        ).items():
            row_metrics = dict(metrics)
            for field_name in (
                "aggregation_client_ids",
                "aggregation_participation",
            ):
                if field_name in row_metrics:
                    row_metrics[field_name] = json.dumps(
                        row_metrics[field_name],
                        sort_keys=True,
                    )
            rows.append(
                {
                    "round": int(round_num),
                    "row_type": "server_drift",
                    "parameter_group": parameter_group,
                    **row_metrics,
                }
            )
    return rows


def _serialize_csv_row(
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> dict[str, str]:
    extra_fields = set(row) - set(fieldnames)
    if extra_fields:
        raise ValueError(f"Unexpected diagnostic CSV fields: {sorted(extra_fields)}")
    return {
        field: "" if row.get(field) is None else str(row.get(field))
        for field in fieldnames
    }


def serialized_diagnostic_rows(
    rounds: Sequence[int],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        _serialize_csv_row(row, DIAGNOSTIC_FIELDNAMES)
        for row in expected_diagnostic_rows(rounds, diagnostics)
    ]


def assert_diagnostic_export_records(
    *,
    rounds: Sequence[int],
    diagnostics: Sequence[Mapping[str, Any]],
    jsonl_records: Sequence[Mapping[str, Any]],
    csv_fieldnames: Sequence[str] | None,
    csv_rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    expected_jsonl = [
        {"round": int(round_num), **dict(round_diagnostics)}
        for round_num, round_diagnostics in zip(rounds, diagnostics)
    ]
    if list(jsonl_records) != expected_jsonl:
        raise ValueError(
            "diagnostic JSONL does not exactly match training_history.json"
        )
    if list(csv_fieldnames or []) != DIAGNOSTIC_FIELDNAMES:
        raise ValueError("diagnostic CSV header violates the formal schema")

    expected_serialized = serialized_diagnostic_rows(rounds, diagnostics)
    if list(csv_rows) != expected_serialized:
        raise ValueError(
            "diagnostic CSV does not exactly match training_history.json; "
            f"expected_rows={len(expected_serialized)}, actual_rows={len(csv_rows)}"
        )
    return {
        "jsonl_records": len(jsonl_records),
        "csv_rows": len(csv_rows),
    }


def audit_diagnostic_exports(
    *,
    rounds: Sequence[int],
    diagnostics: Sequence[Mapping[str, Any]],
    jsonl_path: Path,
    csv_path: Path,
) -> dict[str, int]:
    jsonl_records = _read_jsonl(jsonl_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_rows = list(reader)
        fieldnames = reader.fieldnames
    return assert_diagnostic_export_records(
        rounds=rounds,
        diagnostics=diagnostics,
        jsonl_records=jsonl_records,
        csv_fieldnames=fieldnames,
        csv_rows=actual_rows,
    )


def _require_files(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty {name}: {path}")


def _require_history_lengths(history: Mapping[str, Any]) -> None:
    required_round_series = (
        "avg_losses",
        "avg_seg_losses",
        "avg_cream_losses",
        "client_losses",
        "global_text_rep_norms",
        "global_image_rep_norms",
        "val_metrics",
        "lr_history",
        "gpu_mem_mb",
        "round_time_sec",
        "grad_conflict_deg",
        "parameter_group_diagnostics",
        "parameter_group_effectiveness",
        "aggregation_audits",
        "round_reproducibility",
    )
    for name in required_round_series:
        values = history.get(name)
        if not isinstance(values, list) or len(values) != EXPECTED_ROUNDS:
            count = len(values) if isinstance(values, list) else None
            raise ValueError(
                f"training_history.{name} must contain {EXPECTED_ROUNDS} records; "
                f"got {count}"
            )


def _audit_text_logs(preflight_path: Path, console_path: Path) -> None:
    preflight_text = preflight_path.read_text(encoding="utf-8", errors="replace")
    if '"status": "ready"' not in preflight_text:
        raise ValueError("preflight.log does not contain ready status")

    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "联邦学习训练完成",
        "最终验证集评估",
        "最终评估指标",
        "训练历史已保存到",
    )
    missing = [marker for marker in required_markers if marker not in console_text]
    if missing:
        raise ValueError(f"console.log is incomplete; missing markers: {missing}")
    forbidden_markers = ("Traceback (most recent call last)", "Training failed:")
    found = [marker for marker in forbidden_markers if marker in console_text]
    if found:
        raise ValueError(f"console.log contains failure markers: {found}")


def audit_static_artifacts(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    training = config.get("training")
    logging_config = config.get("logging")
    if not isinstance(training, dict) or not isinstance(logging_config, dict):
        raise TypeError("Formal config is missing training or logging section")
    if int(training.get("rounds", 0)) != EXPECTED_ROUNDS:
        raise ValueError(f"Formal config must declare {EXPECTED_ROUNDS} rounds")
    log_dir = Path(str(logging_config.get("log_dir", "")))
    if not log_dir.is_absolute():
        raise ValueError("Formal logging.log_dir must be absolute")

    checkpoint_dir = log_dir / "checkpoints"
    paths = {
        "preflight log": log_dir / "preflight.log",
        "console log": log_dir / "console.log",
        "diagnostic JSONL": log_dir / "parameter_group_diagnostics.jsonl",
        "diagnostic CSV": log_dir / "parameter_group_diagnostics.csv",
        "effectiveness JSONL": log_dir / "parameter_group_effectiveness.jsonl",
        "training history": checkpoint_dir / "training_history.json",
        "run metadata": checkpoint_dir / "run_metadata.json",
        "final model": checkpoint_dir / "final_model.pth",
        "latest checkpoint": checkpoint_dir / "latest_checkpoint.pth",
        "round 60 checkpoint": checkpoint_dir / "checkpoint_round_60.pth",
    }
    _require_files(paths)
    _audit_text_logs(paths["preflight log"], paths["console log"])

    history = _read_json(paths["training history"])
    metadata = _read_json(paths["run metadata"])
    rounds = history.get("rounds")
    if not isinstance(rounds, list):
        raise TypeError("training_history.rounds must be a list")
    require_exact_rounds(
        "training_history.rounds",
        rounds,
        expected_rounds=EXPECTED_ROUNDS,
    )
    _require_history_lengths(history)
    require_exact_rounds(
        "training_history.val_metrics rounds",
        [entry.get("round") for entry in history["val_metrics"]],
        expected_rounds=EXPECTED_ROUNDS,
    )

    final_stats = history.get("final_stats")
    final_metrics = history.get("final_val_metrics")
    if not isinstance(final_stats, dict) or not isinstance(final_metrics, dict):
        raise ValueError("training_history is missing final stats or final metrics")
    if int(final_stats.get("total_rounds", 0)) != EXPECTED_ROUNDS:
        raise ValueError("training_history final_stats is not round 60")
    for key, expected in EXPECTED_METRIC_CONTRACT.items():
        if final_metrics.get(key) != expected:
            raise ValueError(
                f"Final metric contract mismatch: {key}={final_metrics.get(key)!r}"
            )

    config_seed = int(config.get("seed"))
    if metadata.get("seed") != config_seed:
        raise ValueError("YAML seed and run metadata seed do not match")
    if metadata.get("config_file_sha256") != _sha256(config_path):
        raise ValueError("YAML SHA256 and run metadata configuration SHA256 do not match")
    if history.get("run_metadata") != metadata:
        raise ValueError("Embedded and standalone run metadata do not match")
    data_sha = metadata.get("data_manifest_sha256")
    if not isinstance(data_sha, str) or len(data_sha) != 64:
        raise ValueError("Run metadata is missing the data manifest SHA256")

    diagnostics = history["parameter_group_diagnostics"]
    diagnostic_audit = audit_diagnostic_exports(
        rounds=rounds,
        diagnostics=diagnostics,
        jsonl_path=paths["diagnostic JSONL"],
        csv_path=paths["diagnostic CSV"],
    )
    effectiveness_records = _read_jsonl(paths["effectiveness JSONL"])
    require_exact_rounds(
        "parameter_group_effectiveness.jsonl rounds",
        [entry.get("round") for entry in effectiveness_records],
        expected_rounds=EXPECTED_ROUNDS,
    )
    if effectiveness_records != history["parameter_group_effectiveness"]:
        raise ValueError(
            "parameter_group_effectiveness.jsonl does not exactly match history"
        )

    return {
        "config": config,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "paths": paths,
        "history": history,
        "metadata": metadata,
        "diagnostic_audit": diagnostic_audit,
        "effectiveness_records": len(effectiveness_records),
    }


def _assert_tensor_mapping_equal(
    final_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    *,
    label: str,
) -> None:
    import torch

    missing = sorted(set(checkpoint_state) - set(final_state))
    if missing:
        raise ValueError(f"{label} contains keys missing from final model: {missing[:5]}")
    mismatched = [
        name
        for name, value in checkpoint_state.items()
        if not torch.equal(final_state[name].detach().cpu(), value.detach().cpu())
    ]
    if mismatched:
        raise ValueError(f"{label} differs from final model: {mismatched[:5]}")


def audit_checkpoint_contract(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    final_payload = torch.load(
        paths["final model"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(final_payload, dict) or set(final_payload) != {
        "model_state_dict",
        "global_text_rep",
        "global_image_rep",
    }:
        raise ValueError("final_model.pth violates the formal checkpoint schema")
    final_state = final_payload["model_state_dict"]
    if not isinstance(final_state, dict) or not final_state:
        raise ValueError("final_model.pth has no model_state_dict")

    checkpoint_summaries: dict[str, Any] = {}
    for artifact_name in ("latest checkpoint", "round 60 checkpoint"):
        checkpoint = torch.load(
            paths[artifact_name],
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or int(checkpoint.get("round", 0)) != EXPECTED_ROUNDS:
            raise ValueError(f"{artifact_name} is not the round 60 checkpoint")
        server_state = checkpoint.get("server_state")
        if not isinstance(server_state, dict):
            raise ValueError(f"{artifact_name} has no server_state")
        trainable_parameters = server_state.get("trainable_parameters")
        persistent_buffers = server_state.get("persistent_buffers")
        if not isinstance(trainable_parameters, dict) or not isinstance(
            persistent_buffers, dict
        ):
            raise ValueError(f"{artifact_name} has incomplete explicit server state")
        _assert_tensor_mapping_equal(
            final_state,
            trainable_parameters,
            label=f"{artifact_name} trainable parameters",
        )
        _assert_tensor_mapping_equal(
            final_state,
            persistent_buffers,
            label=f"{artifact_name} persistent buffers",
        )
        for representation_name in ("global_text_rep", "global_image_rep"):
            if not torch.equal(
                final_payload[representation_name].detach().cpu(),
                server_state[representation_name].detach().cpu(),
            ):
                raise ValueError(
                    f"{artifact_name} {representation_name} differs from final model"
                )
        checkpoint_summaries[artifact_name.replace(" ", "_")] = {
            "round": EXPECTED_ROUNDS,
            "trainable_parameter_count": len(trainable_parameters),
            "persistent_buffer_count": len(persistent_buffers),
            "sha256": _sha256(paths[artifact_name]),
        }

    return final_payload, {
        "final_model_sha256": _sha256(paths["final model"]),
        "model_state_key_count": len(final_state),
        **checkpoint_summaries,
    }


def _assert_metrics_equal(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    if set(expected) != set(actual):
        raise ValueError(
            "Re-evaluated metric keys do not match training history; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    mismatched: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, bool) or isinstance(expected_value, str):
            equal = expected_value == actual_value
        elif isinstance(expected_value, int):
            equal = expected_value == actual_value
        elif isinstance(expected_value, float):
            equal = math.isclose(
                expected_value,
                float(actual_value),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        else:
            equal = expected_value == actual_value
        if not equal:
            mismatched.append(key)
    if mismatched:
        raise ValueError(
            f"Re-evaluated metrics differ from training history: {mismatched}"
        )


def evaluate_final_model(
    config_path: Path,
    final_payload: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from src.config_manager import FederatedConfig
    from src.federated_trainer import FederatedTrainer

    runtime_config = FederatedConfig.from_yaml(str(config_path))
    runtime_config.log_type = "none"
    trainer = FederatedTrainer(runtime_config)
    trainer.setup_environment()
    trainer.setup_clients()
    trainer.setup_validation()
    trainer.global_model.load_state_dict(
        final_payload["model_state_dict"],
        strict=True,
    )

    data_manifest = trainer._build_data_manifest()
    data_manifest_sha256 = _sha256_json(data_manifest)
    if data_manifest_sha256 != expected_metadata.get("data_manifest_sha256"):
        raise ValueError("Current validation data manifest differs from training")

    validation_trainer = next(iter(trainer.client_trainers.values()))
    metrics = validation_trainer.validate(
        model=trainer.global_model,
        test_loader=trainer.val_loader,
        compute_hd95=True,
        verbose=True,
    )
    for key, expected in EXPECTED_METRIC_CONTRACT.items():
        if metrics.get(key) != expected:
            raise ValueError(f"Re-evaluation metric contract mismatch: {key}")
    return metrics


def _build_round_metric_rows(
    history: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, round_num in enumerate(history["rounds"]):
        metrics = history["val_metrics"][index]
        rows.append(
            {
                "round": round_num,
                "avg_loss": history["avg_losses"][index],
                "avg_seg_loss": history["avg_seg_losses"][index],
                "avg_cream_loss": history["avg_cream_losses"][index],
                "learning_rate": history["lr_history"][index],
                "gpu_mem_mb": history["gpu_mem_mb"][index],
                "round_time_sec": history["round_time_sec"][index],
                "grad_conflict_deg": history["grad_conflict_deg"][index],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "hd95_mm": metrics["hd95"],
                "WT_dice": metrics["WT_dice"],
                "TC_dice": metrics["TC_dice"],
                "ET_dice": metrics["ET_dice"],
                "WT_iou": metrics["WT_iou"],
                "TC_iou": metrics["TC_iou"],
                "ET_iou": metrics["ET_iou"],
                "WT_hd95_mm": metrics["WT_hd95"],
                "TC_hd95_mm": metrics["TC_hd95"],
                "ET_hd95_mm": metrics["ET_hd95"],
                "num_cases": metrics["num_cases"],
            }
        )
    return rows


def _write_outputs(
    *,
    verification_dir: Path,
    result: Mapping[str, Any],
    round_metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    verification_dir.mkdir(parents=True, exist_ok=False)
    json_path = verification_dir / "formal_verification.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config = result["configuration"]
    metrics = result["reevaluated_final_metrics"]
    csv_row = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "round": EXPECTED_ROUNDS,
        "routing_mode": config["routing_mode"],
        "baseline_method": config["baseline_method"],
        "fedprox_mu": config["fedprox_mu"],
        "training_git_commit": result["training_git_commit"],
        "config_file_sha256": result["config_file_sha256"],
        "data_manifest_sha256": result["data_manifest_sha256"],
        "final_model_sha256": result["checkpoint_audit"]["final_model_sha256"],
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "hd95_mm": metrics["hd95"],
        "WT_dice": metrics["WT_dice"],
        "TC_dice": metrics["TC_dice"],
        "ET_dice": metrics["ET_dice"],
        "WT_hd95_mm": metrics["WT_hd95"],
        "TC_hd95_mm": metrics["TC_hd95"],
        "ET_hd95_mm": metrics["ET_hd95"],
        "num_cases": metrics["num_cases"],
        "hd95_unit": metrics["hd95_unit"],
        "hd95_dimension": metrics["hd95_dimension"],
    }
    csv_path = verification_dir / "final_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)

    round_csv_path = verification_dir / "round_metrics.csv"
    round_fieldnames = list(round_metric_rows[0])
    with round_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=round_fieldnames)
        writer.writeheader()
        writer.writerows(round_metric_rows)


def verify_formal_cell(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_path.relative_to(PROJECT_ROOT)
    if not config_path.is_file():
        raise FileNotFoundError(f"Formal config does not exist: {config_path}")
    if _git_output("status", "--porcelain"):
        raise RuntimeError("Git worktree must be clean for formal verification")

    audit = audit_static_artifacts(config_path)
    metadata = audit["metadata"]
    verification_commit = _git_output("rev-parse", "HEAD")
    if metadata.get("git_commit") != verification_commit:
        raise ValueError("Training and verification Git commits do not match")

    final_payload, checkpoint_audit = audit_checkpoint_contract(audit["paths"])
    reevaluated_metrics = evaluate_final_model(
        config_path,
        final_payload,
        metadata,
    )
    _assert_metrics_equal(
        audit["history"]["final_val_metrics"],
        reevaluated_metrics,
    )
    round_metric_rows = _build_round_metric_rows(audit["history"])

    config = audit["config"]
    result = {
        "status": "PASS",
        "verified_at": _utc_now(),
        "config_path": str(config_path),
        "log_dir": str(audit["log_dir"]),
        "training_git_commit": metadata["git_commit"],
        "verification_git_commit": verification_commit,
        "config_file_sha256": metadata["config_file_sha256"],
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "sam3_checkpoint_sha256": _sha256(Path(config["model"]["sam3_checkpoint"])),
        "configuration": {
            "experiment_name": config["logging"]["experiment_name"],
            "seed": config["seed"],
            "rounds": config["training"]["rounds"],
            "routing_mode": config["federated"]["routing_mode"],
            "unoptimized_update_policy": config["aggregation"][
                "unoptimized_update_policy"
            ],
            "baseline_method": config["baseline"]["method"],
            "fedprox_mu": config["baseline"]["mu"],
        },
        "evaluation_contract": {
            "checkpoint": "final_model.pth",
            "round": EXPECTED_ROUNDS,
            **EXPECTED_METRIC_CONTRACT,
        },
        "export_audit": {
            "completed_rounds": EXPECTED_ROUNDS,
            "validation_records": len(audit["history"]["val_metrics"]),
            "effectiveness_records": audit["effectiveness_records"],
            "round_metrics_csv_rows": len(round_metric_rows),
            **audit["diagnostic_audit"],
        },
        "checkpoint_audit": checkpoint_audit,
        "historical_final_metrics": audit["history"]["final_val_metrics"],
        "reevaluated_final_metrics": reevaluated_metrics,
        "metrics_match": True,
    }
    verification_dir = audit["log_dir"] / "formal_verification"
    _write_outputs(
        verification_dir=verification_dir,
        result=result,
        round_metric_rows=round_metric_rows,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and re-evaluate one completed formal experiment cell"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Formal YAML configuration path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = verify_formal_cell(Path(args.config))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"Formal verification failed: {error}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
