import json
import math
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional

import torch

from src.parameter_groups import classify_parameter


_EPS = 1e-12


def _valid_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and (
        torch.is_floating_point(value) or torch.is_complex(value)
    )


def _update_stats(
    reference_state: Mapping[str, torch.Tensor],
    updated_state: Mapping[str, torch.Tensor],
    sample_weight: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    if sample_weight is not None and (
        not isinstance(sample_weight, int) or sample_weight <= 0
    ):
        raise ValueError("sample_weight must be a positive integer")
    accumulators: Dict[str, Dict[str, float]] = {}
    for name, updated in updated_state.items():
        reference = reference_state.get(name)
        if not _valid_tensor(updated) or not _valid_tensor(reference):
            raise ValueError(f"Invalid floating-point parameter tensor: {name}")
        if not torch.isfinite(updated).all() or not torch.isfinite(reference).all():
            raise ValueError(f"Non-finite parameter tensor: {name}")
        if updated.shape != reference.shape:
            raise ValueError(f"Shape mismatch for parameter {name}")

        group = classify_parameter(name)
        stats = accumulators.setdefault(
            group,
            {
                "update_sq": 0.0,
                "reference_sq": 0.0,
                "numel": 0.0,
                "parameter_count": 0.0,
                "nonzero_parameter_count": 0.0,
            },
        )
        delta = updated.detach().float().cpu() - reference.detach().float().cpu()
        reference_float = reference.detach().float().cpu()
        stats["update_sq"] += float(torch.sum(delta * delta).item())
        stats["reference_sq"] += float(
            torch.sum(reference_float * reference_float).item()
        )
        stats["numel"] += float(delta.numel())
        stats["parameter_count"] += 1.0
        if torch.count_nonzero(delta).item() > 0:
            stats["nonzero_parameter_count"] += 1.0

    result: Dict[str, Dict[str, Any]] = {}
    for group, stats in accumulators.items():
        update_l2 = math.sqrt(stats["update_sq"])
        reference_l2 = math.sqrt(stats["reference_sq"])
        parameter_count = int(stats["parameter_count"])
        nonzero_parameter_count = int(stats["nonzero_parameter_count"])
        result[group] = {
            "update_l2": update_l2,
            "reference_l2": reference_l2,
            "relative_drift": update_l2 / max(reference_l2, _EPS),
            "update_rms": math.sqrt(stats["update_sq"] / max(stats["numel"], 1.0)),
            "numel": int(stats["numel"]),
            "parameter_count": parameter_count,
            "nonzero_parameter_count": nonzero_parameter_count,
            "nonzero_parameter_ratio": nonzero_parameter_count / parameter_count,
        }
        if sample_weight is not None:
            result[group]["sample_weight"] = sample_weight
    return result


def _pair_conflicts(
    round_global_state: Mapping[str, torch.Tensor],
    client_updates: Mapping[str, Mapping[str, torch.Tensor]],
    client_modalities: Mapping[str, str],
) -> list:
    conflicts = []
    for client_a, client_b in combinations(sorted(client_updates), 2):
        state_a = client_updates[client_a]
        state_b = client_updates[client_b]
        groups_a = {classify_parameter(name) for name in state_a}
        groups_b = {classify_parameter(name) for name in state_b}
        observed_groups = groups_a.union(groups_b)
        shared_names = sorted(
            set(round_global_state).intersection(state_a).intersection(state_b)
        )
        accumulators: Dict[str, Dict[str, float]] = {}
        for name in shared_names:
            reference = round_global_state[name]
            value_a = state_a[name]
            value_b = state_b[name]
            if not all(_valid_tensor(value) for value in (reference, value_a, value_b)):
                raise ValueError(f"Invalid floating-point parameter tensor: {name}")
            if not all(
                torch.isfinite(value).all()
                for value in (reference, value_a, value_b)
            ):
                raise ValueError(f"Non-finite parameter tensor: {name}")
            if value_a.shape != reference.shape or value_b.shape != reference.shape:
                raise ValueError(f"Shape mismatch for parameter {name}")

            delta_a = value_a.detach().float().cpu() - reference.detach().float().cpu()
            delta_b = value_b.detach().float().cpu() - reference.detach().float().cpu()
            group = classify_parameter(name)
            stats = accumulators.setdefault(
                group,
                {
                    "dot": 0.0,
                    "norm_a_sq": 0.0,
                    "norm_b_sq": 0.0,
                    "numel": 0.0,
                    "parameter_count": 0.0,
                },
            )
            stats["dot"] += float(torch.sum(delta_a * delta_b).item())
            stats["norm_a_sq"] += float(torch.sum(delta_a * delta_a).item())
            stats["norm_b_sq"] += float(torch.sum(delta_b * delta_b).item())
            stats["numel"] += float(delta_a.numel())
            stats["parameter_count"] += 1.0

        for group in sorted(observed_groups):
            stats = accumulators.get(group)
            if stats is None:
                conflicts.append(
                    {
                        "client_a": client_a,
                        "client_b": client_b,
                        "modality_a": client_modalities.get(client_a),
                        "modality_b": client_modalities.get(client_b),
                        "parameter_group": group,
                        "cosine_similarity": None,
                        "angle_deg": None,
                        "is_negative": None,
                        "conflict_status": "no_shared_parameters",
                        "shared_numel": 0,
                        "shared_parameter_count": 0,
                    }
                )
                continue
            norm_product = math.sqrt(stats["norm_a_sq"] * stats["norm_b_sq"])
            if norm_product <= _EPS:
                conflicts.append(
                    {
                        "client_a": client_a,
                        "client_b": client_b,
                        "modality_a": client_modalities.get(client_a),
                        "modality_b": client_modalities.get(client_b),
                        "parameter_group": group,
                        "cosine_similarity": None,
                        "angle_deg": None,
                        "is_negative": None,
                        "conflict_status": "zero_norm",
                        "shared_numel": int(stats["numel"]),
                        "shared_parameter_count": int(stats["parameter_count"]),
                    }
                )
                continue
            cosine = max(-1.0, min(1.0, stats["dot"] / norm_product))
            conflicts.append(
                {
                    "client_a": client_a,
                    "client_b": client_b,
                    "modality_a": client_modalities.get(client_a),
                    "modality_b": client_modalities.get(client_b),
                    "parameter_group": group,
                    "cosine_similarity": cosine,
                    "angle_deg": math.degrees(math.acos(cosine)),
                    "is_negative": cosine < 0.0,
                    "conflict_status": "defined",
                    "shared_numel": int(stats["numel"]),
                    "shared_parameter_count": int(stats["parameter_count"]),
                }
            )
    return conflicts


def _summarize_conflicts(
    pairwise_conflicts: list,
    observed_groups: set,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, list] = {}
    for item in pairwise_conflicts:
        grouped.setdefault(item["parameter_group"], []).append(item)

    summary: Dict[str, Dict[str, Any]] = {}
    for group in sorted(observed_groups):
        items = grouped.get(group, [])
        defined_items = [
            item for item in items if item["conflict_status"] == "defined"
        ]
        pair_count = len(defined_items)
        negative_count = sum(1 for item in defined_items if item["is_negative"])
        if pair_count == 0:
            summary[group] = {
                "pair_count": 0,
                "negative_pair_count": 0,
                "negative_cosine_ratio": None,
                "conflict_rate": None,
                "mean_cosine_similarity": None,
                "mean_angle_deg": None,
                "shared_pair_count": sum(
                    item["shared_parameter_count"] > 0 for item in items
                ),
                "no_shared_pair_count": sum(
                    item["conflict_status"] == "no_shared_parameters"
                    for item in items
                ),
                "undefined_pair_count": len(items),
            }
            continue
        summary[group] = {
            "pair_count": pair_count,
            "negative_pair_count": negative_count,
            "negative_cosine_ratio": negative_count / pair_count,
            "conflict_rate": negative_count / pair_count,
            "mean_cosine_similarity": sum(
                item["cosine_similarity"] for item in defined_items
            )
            / pair_count,
            "mean_angle_deg": sum(
                item["angle_deg"] for item in defined_items
            ) / pair_count,
            "shared_pair_count": sum(
                item["shared_parameter_count"] > 0 for item in items
            ),
            "no_shared_pair_count": sum(
                item["conflict_status"] == "no_shared_parameters"
                for item in items
            ),
            "undefined_pair_count": len(items) - pair_count,
        }
    return summary


def _validate_aggregation_audit(
    round_global_state: Mapping[str, torch.Tensor],
    aggregated_state: Mapping[str, torch.Tensor],
    client_updates: Mapping[str, Mapping[str, torch.Tensor]],
    client_sample_counts: Mapping[str, int],
    aggregation_audit: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    routing_mode = aggregation_audit.get("routing_mode")
    if routing_mode not in {"unrestricted", "restricted"}:
        raise ValueError("Aggregation audit has an invalid routing mode")
    active_client_ids = aggregation_audit.get("active_client_ids")
    audit_sample_counts = aggregation_audit.get("client_sample_counts")
    parameter_audit = aggregation_audit.get("parameters")
    if not isinstance(active_client_ids, list) or not isinstance(parameter_audit, Mapping):
        raise ValueError("Aggregation audit is missing required parameter entries")
    if len(active_client_ids) != len(set(active_client_ids)):
        raise ValueError("Aggregation audit has duplicate active clients")
    if set(active_client_ids) != set(client_updates):
        raise ValueError("Aggregation audit clients do not match client updates")
    if audit_sample_counts != dict(client_sample_counts):
        raise ValueError("Aggregation audit sample counts do not match client sample counts")
    if set(aggregated_state) != set(round_global_state):
        raise ValueError("Aggregated state keys do not match round-global parameters")
    if set(parameter_audit) != set(aggregated_state):
        raise ValueError("Aggregation audit keys do not match aggregated parameters")
    for name, entry in parameter_audit.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Aggregation audit parameter entry is invalid: {name}")
        if entry.get("parameter_group") != classify_parameter(name):
            raise ValueError(f"Aggregation audit parameter group mismatch: {name}")
        eligible_client_ids = entry.get("eligible_client_ids")
        normalized_weights = entry.get("normalized_weights")
        if not isinstance(eligible_client_ids, list) or not isinstance(normalized_weights, Mapping):
            raise ValueError(f"Aggregation audit weights are invalid: {name}")
        if len(eligible_client_ids) != len(set(eligible_client_ids)):
            raise ValueError(f"Aggregation audit has duplicate eligible clients: {name}")
        if set(eligible_client_ids) != set(normalized_weights):
            raise ValueError(f"Aggregation audit weights do not match eligibility: {name}")
        if not set(eligible_client_ids).issubset(active_client_ids):
            raise ValueError(f"Aggregation audit contains an unknown client: {name}")
        if eligible_client_ids:
            for client_id in eligible_client_ids:
                weight = normalized_weights[client_id]
                if (
                    not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or weight <= 0.0
                ):
                    raise ValueError(
                        f"Aggregation audit has an invalid normalized weight: {name}"
                    )
            weight_sum = sum(float(normalized_weights[client_id]) for client_id in eligible_client_ids)
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Aggregation audit normalized weights do not sum to one: {name}")
    return parameter_audit


def _server_drift(
    round_global_state: Mapping[str, torch.Tensor],
    aggregated_state: Mapping[str, torch.Tensor],
    parameter_audit: Mapping[str, Mapping[str, Any]],
    routing_mode: str,
) -> Dict[str, Dict[str, Any]]:
    metrics = _update_stats(round_global_state, aggregated_state, sample_weight=None)
    participation: Dict[str, Dict[str, Dict[str, Any]]] = {}
    participants: Dict[str, set[str]] = {}
    for name, entry in parameter_audit.items():
        parameter_group = classify_parameter(name)
        participation.setdefault(parameter_group, {})[name] = {
            "eligible_client_ids": list(entry["eligible_client_ids"]),
            "normalized_weights": dict(entry["normalized_weights"]),
        }
        participants.setdefault(parameter_group, set()).update(
            entry["eligible_client_ids"]
        )
    for parameter_group, group_metrics in metrics.items():
        group_metrics["routing_mode"] = routing_mode
        group_metrics["aggregation_client_ids"] = sorted(
            participants.get(parameter_group, set())
        )
        group_metrics["aggregation_participation"] = participation.get(
            parameter_group,
            {},
        )
    return metrics


def compute_parameter_group_diagnostics(
    round_global_state: Mapping[str, torch.Tensor],
    client_updates: Mapping[str, Mapping[str, torch.Tensor]],
    client_modalities: Mapping[str, str],
    client_sample_counts: Mapping[str, int],
    aggregation_audit: Mapping[str, Any],
    aggregated_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    if set(client_updates) != set(client_modalities):
        raise ValueError("Client modalities do not match client updates")
    if set(client_updates) != set(client_sample_counts):
        raise ValueError("Client sample counts do not match client updates")
    for client_id, sample_count in client_sample_counts.items():
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(f"Client sample count must be positive: {client_id}")
    pairwise_conflicts = _pair_conflicts(
        round_global_state=round_global_state,
        client_updates=client_updates,
        client_modalities=client_modalities,
    )
    client_drift = {
        client_id: _update_stats(
            round_global_state,
            state,
            client_sample_counts[client_id],
        )
        for client_id, state in sorted(client_updates.items())
    }
    observed_groups = {
        group
        for groups in client_drift.values()
        for group in groups
    }
    parameter_audit = _validate_aggregation_audit(
        round_global_state,
        aggregated_state,
        client_updates,
        client_sample_counts,
        aggregation_audit,
    )
    global_drift = _server_drift(
        round_global_state,
        aggregated_state,
        parameter_audit,
        aggregation_audit["routing_mode"],
    )
    return {
        "client_drift": client_drift,
        "pairwise_conflicts": pairwise_conflicts,
        "conflict_summary": _summarize_conflicts(
            pairwise_conflicts,
            observed_groups,
        ),
        "global_drift": global_drift,
    }


def flatten_parameter_group_diagnostics(
    round_num: int,
    diagnostics: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for client_id, groups in diagnostics.get("client_drift", {}).items():
        for parameter_group, metrics in groups.items():
            rows.append(
                {
                    "round": round_num,
                    "row_type": "client_drift",
                    "client_id": client_id,
                    "parameter_group": parameter_group,
                    **metrics,
                }
            )

    for item in diagnostics.get("pairwise_conflicts", []):
        rows.append(
            {
                "round": round_num,
                "row_type": "pairwise_conflict",
                **item,
            }
        )

    for parameter_group, metrics in diagnostics.get("conflict_summary", {}).items():
        rows.append(
            {
                "round": round_num,
                "row_type": "conflict_summary",
                "parameter_group": parameter_group,
                **metrics,
            }
        )

    for parameter_group, metrics in diagnostics.get("global_drift", {}).items():
        row_metrics = dict(metrics)
        for field_name in ("aggregation_client_ids", "aggregation_participation"):
            if field_name in row_metrics:
                row_metrics[field_name] = json.dumps(
                    row_metrics[field_name],
                    sort_keys=True,
                )
        rows.append(
            {
                "round": round_num,
                "row_type": "server_drift",
                "parameter_group": parameter_group,
                **row_metrics,
            }
        )

    return rows
