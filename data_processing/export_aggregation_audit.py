from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDNAMES = [
    "schema_version",
    "seed",
    "cell",
    "training_git_commit",
    "round",
    "routing_mode",
    "parameter_name",
    "parameter_group",
    "client_id",
    "client_modality",
    "active",
    "actual_uploader",
    "eligible_uploader",
    "denominator_member",
    "zero_update",
    "zero_update_reason",
    "raw_sample_weight",
    "denominator_sample_weight",
    "denominator_sample_count_total",
    "normalized_weight",
    "empty_eligible",
    "formula_branch",
]


@dataclass(frozen=True)
class ExportResult:
    csv_path: Path
    jsonl_path: Path
    manifest_path: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )
        + "\n"
    ).encode("utf-8")


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return relative_to / path


def _read_archive_json_members(
    archive_path: Path,
    required_members: Iterable[str],
) -> dict[str, tuple[dict[str, Any], str, int]]:
    required = set(required_members)
    found: dict[str, tuple[dict[str, Any], str, int]] = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if member.name not in required:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Archive member is not a regular file: {member.name}")
            data = extracted.read()
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Archive JSON member must be an object: {member.name}")
            found[member.name] = (payload, _sha256_bytes(data), len(data))
            if set(found) == required:
                break
    missing = sorted(required - set(found))
    if missing:
        raise FileNotFoundError(
            f"Archive is missing required JSON members: {archive_path}: {missing}"
        )
    return found


def _require_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of client IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate client IDs")
    return value


def _require_positive_counts(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    counts: dict[str, int] = {}
    for client_id, count in value.items():
        if not isinstance(client_id, str):
            raise ValueError(f"{field} contains a non-string client ID")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{field}[{client_id}] must be a positive integer")
        counts[client_id] = count
    return counts


def _require_weight_map(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    weights: dict[str, float] = {}
    for client_id, weight in value.items():
        if not isinstance(client_id, str):
            raise ValueError(f"{field} contains a non-string client ID")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"{field}[{client_id}] must be numeric")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{field}[{client_id}] must be finite and positive")
        weights[client_id] = numeric
    return weights


def _formula_branch(routing_mode: str, empty_eligible: bool) -> str:
    if routing_mode == "unrestricted":
        return "eq5_unrestricted"
    if routing_mode == "uploader_renormalized":
        if empty_eligible:
            return "uploader_renormalized_empty_preserve_global"
        return "uploader_renormalized_nonempty"
    if empty_eligible:
        return "eq7_empty_preserve_global"
    return "eq7_restricted_nonempty"


def _parameter_record(
    *,
    schema_version: int,
    seed: int,
    cell: str,
    training_git_commit: str,
    round_num: int,
    routing_mode: str,
    parameter_name: str,
    entry: Mapping[str, Any],
    active_client_ids: list[str],
    client_sample_counts: dict[str, int],
) -> dict[str, Any]:
    parameter_group = entry.get("parameter_group")
    if not isinstance(parameter_group, str) or not parameter_group:
        raise ValueError(f"Parameter group is missing: {parameter_name}")
    uploaded = _require_string_list(
        entry.get("uploaded_client_ids"),
        f"{parameter_name}.uploaded_client_ids",
    )
    eligible = _require_string_list(
        entry.get("eligible_client_ids"),
        f"{parameter_name}.eligible_client_ids",
    )
    zero_update = _require_string_list(
        entry.get("zero_update_client_ids"),
        f"{parameter_name}.zero_update_client_ids",
    )
    sample_weights = _require_positive_counts(
        entry.get("sample_weights"),
        f"{parameter_name}.sample_weights",
    )
    normalized_weights = _require_weight_map(
        entry.get("normalized_weights"),
        f"{parameter_name}.normalized_weights",
    )
    empty_eligible = entry.get("empty_eligible")
    if not isinstance(empty_eligible, bool):
        raise ValueError(f"{parameter_name}.empty_eligible must be boolean")

    active = set(active_client_ids)
    uploaded_set = set(uploaded)
    eligible_set = set(eligible)
    zero_set = set(zero_update)
    if not uploaded_set.issubset(active):
        raise ValueError(f"{parameter_name} has an uploader outside active clients")
    if not eligible_set.issubset(active):
        raise ValueError(f"{parameter_name} has an eligible client outside active clients")
    if not zero_set.issubset(eligible_set):
        raise ValueError(f"{parameter_name} has a zero update outside the denominator")
    if empty_eligible != (not eligible):
        raise ValueError(f"{parameter_name} empty-set flag disagrees with eligibility")
    if set(sample_weights) != eligible_set:
        raise ValueError(f"{parameter_name} raw sample weights disagree with denominator")
    if set(normalized_weights) != eligible_set:
        raise ValueError(f"{parameter_name} normalized weights disagree with denominator")
    for client_id in eligible:
        if sample_weights[client_id] != client_sample_counts[client_id]:
            raise ValueError(f"{parameter_name} raw sample weight mismatch: {client_id}")

    missing_upload_zero = sorted(eligible_set - uploaded_set)
    numerical_zero_uploaders = sorted(zero_set & uploaded_set)
    if routing_mode == "unrestricted":
        if eligible != active_client_ids:
            raise ValueError(
                f"{parameter_name} unrestricted denominator is not all active clients"
            )
        missing_not_zero = sorted(set(missing_upload_zero) - zero_set)
        if missing_not_zero:
            raise ValueError(
                f"{parameter_name} missing-upload client is not a zero update: "
                f"{missing_not_zero}"
            )
    elif routing_mode == "uploader_renormalized":
        if eligible != uploaded:
            raise ValueError(
                f"{parameter_name} uploader-renormalized denominator must equal "
                "actual uploaders"
            )
    elif routing_mode == "restricted":
        if not eligible_set.issubset(uploaded_set):
            raise ValueError(
                f"{parameter_name} restricted eligibility includes a non-uploader"
            )
        if missing_upload_zero:
            raise ValueError(
                f"{parameter_name} restricted denominator contains a missing upload"
            )
    else:
        raise ValueError(f"Unsupported routing mode: {routing_mode}")

    denominator_total = sum(client_sample_counts[client_id] for client_id in eligible)
    if eligible:
        if denominator_total <= 0:
            raise ValueError(f"{parameter_name} has a non-positive denominator")
        if not math.isclose(
            sum(normalized_weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{parameter_name} normalized weights do not sum to one")
        for client_id in eligible:
            expected = client_sample_counts[client_id] / float(denominator_total)
            if not math.isclose(
                normalized_weights[client_id],
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{parameter_name} normalized weight mismatch: {client_id}"
                )
    elif sample_weights or normalized_weights or zero_update:
        raise ValueError(f"{parameter_name} empty set contains aggregation values")

    return {
        "schema_version": schema_version,
        "seed": seed,
        "cell": cell,
        "training_git_commit": training_git_commit,
        "round": round_num,
        "routing_mode": routing_mode,
        "parameter_name": parameter_name,
        "parameter_group": parameter_group,
        "active_clients": active_client_ids,
        "actual_uploaders": uploaded,
        "eligible_uploaders": sorted(uploaded_set & eligible_set),
        "denominator_clients": eligible,
        "zero_update_clients": zero_update,
        "missing_upload_zero_update_clients": missing_upload_zero,
        "numerical_zero_delta_uploaders": numerical_zero_uploaders,
        "raw_sample_weights": client_sample_counts,
        "denominator_sample_weights": sample_weights,
        "denominator_sample_count_total": denominator_total,
        "normalized_weights": normalized_weights,
        "empty_eligible": empty_eligible,
        "formula_branch": _formula_branch(routing_mode, empty_eligible),
    }


def _source_records(
    *,
    schema_version: int,
    seed: int,
    expected_commit: str,
    cell: str,
    client_modalities: dict[str, str],
    formal_verification: dict[str, Any],
    history: dict[str, Any],
) -> list[dict[str, Any]]:
    if formal_verification.get("status") != "PASS":
        raise ValueError(f"Formal verification is not PASS: {cell}")
    training_commit = formal_verification.get("training_git_commit")
    if training_commit != expected_commit:
        raise ValueError(f"Formal verification Git commit mismatch: {cell}")
    configuration = formal_verification.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Formal verification configuration is missing: {cell}")
    if configuration.get("seed") != seed:
        raise ValueError(f"Formal verification seed mismatch: {cell}")

    run_metadata = history.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError(f"Training history run metadata is missing: {cell}")
    if run_metadata.get("seed") != seed:
        raise ValueError(f"Training history seed mismatch: {cell}")
    if run_metadata.get("git_commit") != expected_commit:
        raise ValueError(f"Training history Git commit mismatch: {cell}")
    audits = history.get("aggregation_audits")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"Training history has no aggregation audits: {cell}")

    records: list[dict[str, Any]] = []
    observed_rounds: list[int] = []
    for audit in audits:
        if not isinstance(audit, dict):
            raise ValueError(f"Aggregation audit must be an object: {cell}")
        round_num = audit.get("round")
        if not isinstance(round_num, int) or round_num <= 0:
            raise ValueError(f"Aggregation audit round is invalid: {cell}")
        observed_rounds.append(round_num)
        routing_mode = audit.get("routing_mode")
        if routing_mode != configuration.get("routing_mode"):
            raise ValueError(f"Aggregation routing mode mismatch: {cell}: round {round_num}")
        active_client_ids = _require_string_list(
            audit.get("active_client_ids"),
            f"{cell}.round_{round_num}.active_client_ids",
        )
        if set(active_client_ids) != set(client_modalities):
            raise ValueError(f"Aggregation clients disagree with configured modalities: {cell}")
        client_sample_counts = _require_positive_counts(
            audit.get("client_sample_counts"),
            f"{cell}.round_{round_num}.client_sample_counts",
        )
        if set(client_sample_counts) != set(active_client_ids):
            raise ValueError(f"Aggregation sample counts disagree with active clients: {cell}")
        parameters = audit.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"Aggregation audit has no parameters: {cell}: round {round_num}")
        derived_empty_names: list[str] = []
        for parameter_name in sorted(parameters):
            entry = parameters[parameter_name]
            if not isinstance(parameter_name, str) or not isinstance(entry, dict):
                raise ValueError(f"Aggregation parameter entry is invalid: {cell}")
            record = _parameter_record(
                schema_version=schema_version,
                seed=seed,
                cell=cell,
                training_git_commit=training_commit,
                round_num=round_num,
                routing_mode=routing_mode,
                parameter_name=parameter_name,
                entry=entry,
                active_client_ids=active_client_ids,
                client_sample_counts=client_sample_counts,
            )
            records.append(record)
            if record["empty_eligible"]:
                derived_empty_names.append(parameter_name)
        reported_empty_names = audit.get("empty_eligible_parameter_names")
        if reported_empty_names is not None:
            reported = _require_string_list(
                reported_empty_names,
                f"{cell}.round_{round_num}.empty_eligible_parameter_names",
            )
            if reported != derived_empty_names:
                raise ValueError(
                    f"Empty-eligible parameter list mismatch: {cell}: round {round_num}"
                )

    history_rounds = history.get("rounds")
    if isinstance(history_rounds, list) and history_rounds != observed_rounds:
        raise ValueError(f"Training rounds disagree with aggregation audit rounds: {cell}")
    return records


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _csv_rows(
    records: Iterable[dict[str, Any]],
    client_modalities: Mapping[str, str],
) -> Iterable[dict[str, Any]]:
    for record in records:
        uploaded = set(record["actual_uploaders"])
        eligible_uploaders = set(record["eligible_uploaders"])
        denominator = set(record["denominator_clients"])
        zero_update = set(record["zero_update_clients"])
        missing = set(record["missing_upload_zero_update_clients"])
        numerical = set(record["numerical_zero_delta_uploaders"])
        for client_id in record["active_clients"]:
            if client_id in missing:
                zero_reason = "missing_upload_eq4"
            elif client_id in numerical:
                zero_reason = "uploaded_numerical_zero"
            else:
                zero_reason = ""
            yield {
                "schema_version": record["schema_version"],
                "seed": record["seed"],
                "cell": record["cell"],
                "training_git_commit": record["training_git_commit"],
                "round": record["round"],
                "routing_mode": record["routing_mode"],
                "parameter_name": record["parameter_name"],
                "parameter_group": record["parameter_group"],
                "client_id": client_id,
                "client_modality": client_modalities[client_id],
                "active": "true",
                "actual_uploader": _bool_text(client_id in uploaded),
                "eligible_uploader": _bool_text(client_id in eligible_uploaders),
                "denominator_member": _bool_text(client_id in denominator),
                "zero_update": _bool_text(client_id in zero_update),
                "zero_update_reason": zero_reason,
                "raw_sample_weight": record["raw_sample_weights"][client_id],
                "denominator_sample_weight": record[
                    "denominator_sample_weights"
                ].get(client_id, ""),
                "denominator_sample_count_total": record[
                    "denominator_sample_count_total"
                ],
                "normalized_weight": record["normalized_weights"].get(client_id, ""),
                "empty_eligible": _bool_text(record["empty_eligible"]),
                "formula_branch": record["formula_branch"],
            }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def export_aggregation_audit(config_path: str | Path) -> ExportResult:
    config_path = Path(config_path).resolve()
    config_data = config_path.read_bytes()
    config = json.loads(config_data.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Aggregation audit config must be an object")
    schema_version = config.get("schema_version")
    seed = config.get("seed")
    expected_commit = config.get("expected_training_git_commit")
    if not isinstance(schema_version, int) or schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")
    if not isinstance(seed, int) or seed <= 0:
        raise ValueError("seed must be a positive integer")
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise ValueError("expected_training_git_commit must be a 40-character commit")

    archive_root_value = config.get("archive_root")
    if not isinstance(archive_root_value, str) or not archive_root_value:
        raise ValueError("archive_root must be a path string")
    archive_root = _resolve_path(archive_root_value, relative_to=config_path.parent)
    client_modalities = config.get("clients")
    if (
        not isinstance(client_modalities, dict)
        or not client_modalities
        or not all(
            isinstance(client_id, str) and isinstance(modality, str)
            for client_id, modality in client_modalities.items()
        )
    ):
        raise ValueError("clients must map client IDs to modalities")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")

    all_records: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")
        cell = source.get("cell")
        archive_name = source.get("archive")
        formal_member = source.get("formal_verification_member")
        history_member = source.get("training_history_member")
        if not all(
            isinstance(value, str) and value
            for value in (cell, archive_name, formal_member, history_member)
        ):
            raise ValueError("Source paths and cell must be non-empty strings")
        archive_path = _resolve_path(archive_name, relative_to=archive_root)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Aggregation archive does not exist: {archive_path}")
        members = _read_archive_json_members(
            archive_path,
            [formal_member, history_member],
        )
        formal_verification, formal_sha256, formal_size = members[formal_member]
        history, history_sha256, history_size = members[history_member]
        records = _source_records(
            schema_version=schema_version,
            seed=seed,
            expected_commit=expected_commit,
            cell=cell,
            client_modalities=client_modalities,
            formal_verification=formal_verification,
            history=history,
        )
        all_records.extend(records)
        stat = archive_path.stat()
        source_manifest.append(
            {
                "cell": cell,
                "archive_path": str(archive_path),
                "archive_size_bytes": stat.st_size,
                "archive_mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "formal_verification_member": formal_member,
                "formal_verification_sha256": formal_sha256,
                "formal_verification_size_bytes": formal_size,
                "formal_verification_status": formal_verification["status"],
                "training_history_member": history_member,
                "training_history_sha256": history_sha256,
                "training_history_size_bytes": history_size,
                "training_git_commit": formal_verification["training_git_commit"],
                "round_count": len({record["round"] for record in records}),
                "parameter_record_count": len(records),
            }
        )

    all_records.sort(
        key=lambda record: (
            record["cell"],
            record["round"],
            record["parameter_group"],
            record["parameter_name"],
        )
    )
    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    output_dir_value = output.get("directory")
    csv_name = output.get("csv")
    jsonl_name = output.get("jsonl")
    manifest_name = output.get("manifest")
    if not all(
        isinstance(value, str) and value
        for value in (output_dir_value, csv_name, jsonl_name, manifest_name)
    ):
        raise ValueError("Output directory and filenames must be non-empty strings")
    output_dir = _resolve_path(output_dir_value, relative_to=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    jsonl_path = output_dir / jsonl_name
    manifest_path = output_dir / manifest_name

    csv_rows = list(_csv_rows(all_records, client_modalities))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(csv_rows)
    with jsonl_path.open("wb") as handle:
        for record in all_records:
            handle.write(_json_bytes(record))

    formula_counts = {
        "eq4_missing_upload_zero_update_parameters": sum(
            bool(record["missing_upload_zero_update_clients"])
            for record in all_records
        ),
        "eq5_unrestricted_parameters": sum(
            record["formula_branch"] == "eq5_unrestricted" for record in all_records
        ),
        "uploader_renormalized_nonempty_parameters": sum(
            record["formula_branch"] == "uploader_renormalized_nonempty"
            for record in all_records
        ),
        "uploader_renormalized_empty_preserve_global_parameters": sum(
            record["formula_branch"]
            == "uploader_renormalized_empty_preserve_global"
            for record in all_records
        ),
        "eq6_restricted_eligibility_parameters": sum(
            record["routing_mode"] == "restricted" for record in all_records
        ),
        "eq7_restricted_nonempty_parameters": sum(
            record["formula_branch"] == "eq7_restricted_nonempty"
            for record in all_records
        ),
        "eq7_empty_preserve_global_parameters": sum(
            record["formula_branch"] == "eq7_empty_preserve_global"
            for record in all_records
        ),
    }
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "data_processing/export_aggregation_audit.py",
        "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "generator_workspace_base_git_commit": _git_commit(),
        "generation_environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "config_path": str(config_path),
        "config_sha256": _sha256_bytes(config_data),
        "seed": seed,
        "cells": [source["cell"] for source in source_manifest],
        "expected_training_git_commit": expected_commit,
        "training_or_inference_performed": False,
        "source_type": "completed formally verified training archives",
        "aggregation_code_locations": [
            {
                "path": "src/server.py",
                "symbol": "CreamAggregator.aggregate_weights",
                "role": "actual uploaders, U/N/R eligibility, zero updates, weights, and empty-set branch",
            },
            {
                "path": "src/parameter_groups.py",
                "symbol": "allowed_modalities",
                "role": "restricted parameter-group modality whitelist",
            },
            {
                "path": "src/federated_trainer.py",
                "symbol": "FederatedTrainer._train_single_round",
                "role": "real server invocation and aggregation-audit persistence",
            },
        ],
        "sources": source_manifest,
        "formula_validation": {
            "status": "PASS",
            "equations": [
                "Eq. (4)",
                "Eq. (5)",
                "Eq. (6)",
                "Eq. (7)",
                "manifest-defined uploader-renormalized diagnostic control",
            ],
            "counts": formula_counts,
            "zero_update_semantics": {
                "missing_upload": "eligible denominator client absent from actual uploaders",
                "uploaded_numerical_zero": "actual uploader recorded with an all-zero delta",
            },
        },
        "parameter_record_count": len(all_records),
        "csv_row_count": len(csv_rows),
        "outputs": {
            "csv": {
                "path": str(csv_path),
                "sha256": _sha256_file(csv_path),
                "size_bytes": csv_path.stat().st_size,
            },
            "jsonl": {
                "path": str(jsonl_path),
                "sha256": _sha256_file(jsonl_path),
                "size_bytes": jsonl_path.stat().st_size,
            },
        },
    }
    manifest_path.write_bytes(_json_bytes(manifest, indent=2))
    return ExportResult(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Eq. (4)-(7) aggregation evidence from completed archives."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Aggregation audit JSON configuration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_aggregation_audit(args.config)
    print(
        json.dumps(
            {
                "csv": str(result.csv_path),
                "jsonl": str(result.jsonl_path),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
