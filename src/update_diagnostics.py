import math
from itertools import combinations
from typing import Any, Dict, Mapping, Optional

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
) -> Dict[str, Dict[str, float]]:
    accumulators: Dict[str, Dict[str, float]] = {}
    for name, updated in updated_state.items():
        reference = reference_state.get(name)
        if not _valid_tensor(updated) or not _valid_tensor(reference):
            continue
        if updated.shape != reference.shape:
            raise ValueError(f"Shape mismatch for parameter {name}")

        group = classify_parameter(name)
        stats = accumulators.setdefault(
            group,
            {"update_sq": 0.0, "reference_sq": 0.0, "numel": 0.0},
        )
        delta = updated.detach().float().cpu() - reference.detach().float().cpu()
        reference_float = reference.detach().float().cpu()
        stats["update_sq"] += float(torch.sum(delta * delta).item())
        stats["reference_sq"] += float(
            torch.sum(reference_float * reference_float).item()
        )
        stats["numel"] += float(delta.numel())

    result: Dict[str, Dict[str, float]] = {}
    for group, stats in accumulators.items():
        update_l2 = math.sqrt(stats["update_sq"])
        reference_l2 = math.sqrt(stats["reference_sq"])
        result[group] = {
            "update_l2": update_l2,
            "reference_l2": reference_l2,
            "relative_drift": update_l2 / max(reference_l2, _EPS),
            "numel": int(stats["numel"]),
        }
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
        shared_names = sorted(
            set(round_global_state).intersection(state_a).intersection(state_b)
        )
        accumulators: Dict[str, Dict[str, float]] = {}
        for name in shared_names:
            reference = round_global_state[name]
            value_a = state_a[name]
            value_b = state_b[name]
            if not all(_valid_tensor(value) for value in (reference, value_a, value_b)):
                continue
            if value_a.shape != reference.shape or value_b.shape != reference.shape:
                raise ValueError(f"Shape mismatch for parameter {name}")

            delta_a = value_a.detach().float().cpu() - reference.detach().float().cpu()
            delta_b = value_b.detach().float().cpu() - reference.detach().float().cpu()
            group = classify_parameter(name)
            stats = accumulators.setdefault(
                group,
                {"dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0, "numel": 0.0},
            )
            stats["dot"] += float(torch.sum(delta_a * delta_b).item())
            stats["norm_a_sq"] += float(torch.sum(delta_a * delta_a).item())
            stats["norm_b_sq"] += float(torch.sum(delta_b * delta_b).item())
            stats["numel"] += float(delta_a.numel())

        for group, stats in sorted(accumulators.items()):
            norm_product = math.sqrt(stats["norm_a_sq"] * stats["norm_b_sq"])
            if norm_product <= _EPS:
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
                    "shared_numel": int(stats["numel"]),
                }
            )
    return conflicts


def _summarize_conflicts(pairwise_conflicts: list) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, list] = {}
    for item in pairwise_conflicts:
        grouped.setdefault(item["parameter_group"], []).append(item)

    summary: Dict[str, Dict[str, float]] = {}
    for group, items in sorted(grouped.items()):
        pair_count = len(items)
        negative_count = sum(1 for item in items if item["is_negative"])
        summary[group] = {
            "pair_count": pair_count,
            "negative_pair_count": negative_count,
            "negative_cosine_ratio": negative_count / pair_count,
            "mean_cosine_similarity": sum(
                item["cosine_similarity"] for item in items
            )
            / pair_count,
            "mean_angle_deg": sum(item["angle_deg"] for item in items) / pair_count,
        }
    return summary


def compute_parameter_group_diagnostics(
    round_global_state: Mapping[str, torch.Tensor],
    client_updates: Mapping[str, Mapping[str, torch.Tensor]],
    client_modalities: Mapping[str, str],
    aggregated_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    pairwise_conflicts = _pair_conflicts(
        round_global_state=round_global_state,
        client_updates=client_updates,
        client_modalities=client_modalities,
    )
    client_drift = {
        client_id: _update_stats(round_global_state, state)
        for client_id, state in sorted(client_updates.items())
    }
    global_drift = (
        _update_stats(round_global_state, aggregated_state)
        if aggregated_state is not None
        else {}
    )
    return {
        "client_drift": client_drift,
        "pairwise_conflicts": pairwise_conflicts,
        "conflict_summary": _summarize_conflicts(pairwise_conflicts),
        "global_drift": global_drift,
    }
