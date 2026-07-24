"""
Server Aggregator: 基于 CreamFL 的全局聚合逻辑。
实现基于客户端表示相似性的加权聚合，以及异构多模态的解耦聚合路由。
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import torch.optim

from src.model import SAM3_Medical, DEVICE
from src.contrastive_aggregation import ContrastiveWeightAggregation
from src.knowledge_distillation import KnowledgeDistillation
from src.parameter_groups import (
    allowed_modalities,
    classify_parameter,
    classify_trainable_parameters,
    is_vision_parameter,
)


class CreamAggregator:
    """
    CreamFL 联邦聚合器。
    实现基于客户端表示相似性的加权聚合，以及异构模态间的物理隔离路由。
    """

    def __init__(
        self,
        global_model: SAM3_Medical,
        device: str = DEVICE,
        aggregation_method: str = "contrastive_weighted",
        global_rep_alpha: float = 0.9,
        strict_aggregation_guard: bool = False,
        proxy_k_batches: int = 1,
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        self.global_rep_alpha = global_rep_alpha
        self.strict_aggregation_guard = strict_aggregation_guard
        self.proxy_k_batches = max(1, int(proxy_k_batches))

        self.contrastive_aggregator = (
            ContrastiveWeightAggregation(device=device)
            if aggregation_method == "contrastive_weighted"
            else None
        )
        self.distiller = None

        representation_dim = int(global_model.contrastive_dim)
        if representation_dim <= 0:
            raise ValueError(
                f"contrastive_dim must be positive, got {representation_dim}"
            )
        self.global_text_rep = nn.functional.normalize(
            torch.randn(representation_dim, device=device), p=2, dim=0
        )
        self.global_image_rep = nn.functional.normalize(
            torch.randn(representation_dim, device=device), p=2, dim=0
        )

        # 梯度冲突角度（由 federated_trainer 在聚合后读取写入 training_history）
        self._last_grad_conflict_deg: Optional[float] = None
        self._last_aggregation_audit: Dict[str, Any] = {}
        self._round_buffer_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self._round_buffer_distribution_audit: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    # 内部工具方法
    # ──────────────────────────────────────────────────────────────────

    def _trainable_named_parameters(self) -> Dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in self.global_model.named_parameters()
            if parameter.requires_grad
        }

    def get_trainable_parameter_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a CPU snapshot of exactly the trainable named parameters."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._trainable_named_parameters().items()
        }

    def apply_trainable_parameters(
        self,
        parameter_snapshot: Dict[str, torch.Tensor],
    ) -> None:
        """Copy an exact trainable-parameter snapshot without touching buffers."""
        named_parameters = self._trainable_named_parameters()
        if set(parameter_snapshot) != set(named_parameters):
            missing = sorted(set(named_parameters) - set(parameter_snapshot))
            unexpected = sorted(set(parameter_snapshot) - set(named_parameters))
            raise ValueError(
                "Trainable parameter snapshot must match the model exactly; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        with torch.no_grad():
            for name, parameter in named_parameters.items():
                source = parameter_snapshot[name]
                if not isinstance(source, torch.Tensor):
                    raise TypeError(
                        f"Trainable parameter snapshot value must be a tensor: {name}"
                    )
                if source.shape != parameter.shape:
                    raise ValueError(
                        f"Trainable parameter snapshot shape mismatch for {name}: "
                        f"expected {tuple(parameter.shape)}, got {tuple(source.shape)}"
                    )
                if source.dtype != parameter.dtype:
                    raise ValueError(
                        f"Trainable parameter snapshot dtype mismatch for {name}: "
                        f"expected {parameter.dtype}, got {source.dtype}"
                    )
                parameter.copy_(source.to(parameter.device))

    def _buffer_inventory(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return registered persistent and nonpersistent buffers by full name."""
        named_buffers = dict(self.global_model.named_buffers())
        nonpersistent_names = set()
        for module_prefix, module in self.global_model.named_modules():
            for buffer_name in module._non_persistent_buffers_set:
                full_name = (
                    f"{module_prefix}.{buffer_name}"
                    if module_prefix
                    else buffer_name
                )
                if full_name in named_buffers:
                    nonpersistent_names.add(full_name)

        persistent_buffers = {
            name: buffer
            for name, buffer in named_buffers.items()
            if name not in nonpersistent_names
        }
        nonpersistent_buffers = {
            name: named_buffers[name]
            for name in sorted(nonpersistent_names)
        }
        return persistent_buffers, nonpersistent_buffers

    @staticmethod
    def _clone_buffer_snapshot(
        buffers: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            name: buffer.detach().cpu().clone()
            for name, buffer in buffers.items()
        }

    def capture_round_buffer_snapshot(self) -> Dict[str, torch.Tensor]:
        """Capture the server-owned persistent buffers for one federated round."""
        persistent_buffers, nonpersistent_buffers = self._buffer_inventory()
        self._round_buffer_snapshot = self._clone_buffer_snapshot(persistent_buffers)
        self._round_buffer_distribution_audit = {
            "buffer_key_count": len(persistent_buffers) + len(nonpersistent_buffers),
            "persistent_buffer_key_count": len(persistent_buffers),
            "nonpersistent_buffer_key_count": len(nonpersistent_buffers),
            "snapshot_key_count": len(self._round_buffer_snapshot),
            "restore_events": [],
        }
        return self._clone_buffer_snapshot(self._round_buffer_snapshot)

    def _rebuild_nonpersistent_buffers(self) -> int:
        """Rebuild deterministic nonpersistent buffers instead of copying clients' state."""
        _, nonpersistent_buffers = self._buffer_inventory()
        if not nonpersistent_buffers:
            return 0

        reset_rope_frequencies = getattr(
            self.global_model, "reset_rope_frequencies", None
        )
        if not callable(reset_rope_frequencies):
            raise RuntimeError(
                "Nonpersistent buffers require a deterministic "
                "model.reset_rope_frequencies() rebuild function"
            )
        rebuilt_count = reset_rope_frequencies(verbose=False)
        if rebuilt_count is None:
            return 0
        if isinstance(rebuilt_count, bool) or not isinstance(rebuilt_count, int):
            raise RuntimeError(
                "reset_rope_frequencies() must return an integer rebuild count"
            )
        return rebuilt_count

    def restore_round_buffer_snapshot(
        self,
        buffer_snapshot: Optional[Dict[str, torch.Tensor]] = None,
        *,
        reason: str,
    ) -> None:
        """Restore server buffers and deterministically rebuild nonpersistent buffers."""
        if buffer_snapshot is None:
            buffer_snapshot = self._round_buffer_snapshot
        if buffer_snapshot is None:
            raise RuntimeError("No round buffer snapshot is available to restore")

        persistent_buffers, nonpersistent_buffers = self._buffer_inventory()
        if set(buffer_snapshot) != set(persistent_buffers):
            missing = sorted(set(persistent_buffers) - set(buffer_snapshot))
            unexpected = sorted(set(buffer_snapshot) - set(persistent_buffers))
            raise ValueError(
                "Persistent buffer snapshot must match the model exactly; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        with torch.no_grad():
            for name, buffer in persistent_buffers.items():
                source = buffer_snapshot[name]
                if not isinstance(source, torch.Tensor):
                    raise TypeError(
                        f"Persistent buffer snapshot value must be a tensor: {name}"
                    )
                if source.shape != buffer.shape:
                    raise ValueError(
                        f"Persistent buffer snapshot shape mismatch for {name}: "
                        f"expected {tuple(buffer.shape)}, got {tuple(source.shape)}"
                    )
                if source.dtype != buffer.dtype:
                    raise ValueError(
                        f"Persistent buffer snapshot dtype mismatch for {name}: "
                        f"expected {buffer.dtype}, got {source.dtype}"
                    )
                buffer.copy_(source.to(buffer.device))

        rebuilt_count = self._rebuild_nonpersistent_buffers()
        self._round_buffer_distribution_audit.setdefault("restore_events", []).append(
            {
                "reason": str(reason),
                "restored_persistent_buffer_key_count": len(persistent_buffers),
                "rebuilt_nonpersistent_buffer_key_count": len(nonpersistent_buffers),
                "rebuilt_rope_block_count": rebuilt_count,
            }
        )

    def get_round_buffer_distribution_audit(self) -> Dict[str, Any]:
        """Return a detached plain-data record of the current round buffer policy."""
        return {
            **self._round_buffer_distribution_audit,
            "restore_events": [
                dict(event)
                for event in self._round_buffer_distribution_audit.get(
                    "restore_events", []
                )
            ],
        }

    def _is_vision_param(self, param_name: str) -> bool:
        """判断参数是否属于视觉侧（IMAGE_PARAMS 或 VISION_ADAPTER）。"""
        return is_vision_parameter(param_name)

    def _compute_similarity_weights(
        self,
        client_public_reps: List[torch.Tensor],
        consensus_rep: torch.Tensor
    ) -> torch.Tensor:
        """基于与共识表征的余弦相似度计算聚合权重。"""
        similarities = torch.stack([
            torch.dot(
                nn.functional.normalize(rep.to(self.device), p=2, dim=0),
                nn.functional.normalize(consensus_rep, p=2, dim=0)
            )
            for rep in client_public_reps
        ])
        weights = torch.softmax(similarities, dim=0)
        weights = torch.clamp(weights, min=0.0)
        return weights / (weights.sum() + 1e-8)

    def _safe_fill_missing_params(
        self,
        aggregated_state: Dict[str, torch.Tensor],
        location_tag: str = "",
        expected_param_names: Optional[set] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        防回归补齐 + 视觉参数守护。

        对缺失参数用 global_model 当前值（上轮聚合结果）补全。
        视觉侧参数缺失时打印 WARNING；已在 aggregated_state 中的参数绝不覆盖。
        """
        global_sd = self.global_model.state_dict()
        global_keys = set(global_sd.keys())
        aggregated_keys = set(aggregated_state.keys())
        missing = global_keys - aggregated_keys

        # strict guard should only audit keys expected to be aggregated in this round.
        if expected_param_names is None:
            expected_param_names = global_keys
        expected_missing = expected_param_names - aggregated_keys
        vision_missing = [k for k in expected_missing if self._is_vision_param(k)]
        non_vision_missing = [k for k in expected_missing if not self._is_vision_param(k)]

        vision_module_counts: Dict[str, int] = {}
        for k in vision_missing:
            mod = k.split('.')[0]
            vision_module_counts[mod] = vision_module_counts.get(mod, 0) + 1

        non_vision_module_counts: Dict[str, int] = {}
        for k in non_vision_missing:
            mod = k.split('.')[1] if k.count('.') >= 1 else k.split('.')[0]
            non_vision_module_counts[mod] = non_vision_module_counts.get(mod, 0) + 1

        self._last_aggregation_audit = {
            "location_tag": location_tag,
            "missing_total": len(missing),
            "expected_missing_total": len(expected_missing),
            "vision_missing": len(vision_missing),
            "non_vision_missing": len(non_vision_missing),
            "global_key_total": len(global_keys),
            "aggregated_key_total": len(aggregated_keys),
            "expected_key_total": len(expected_param_names),
            "vision_module_counts": vision_module_counts,
            "non_vision_module_counts": non_vision_module_counts,
            "expected_missing_sample": sorted(expected_missing)[:10],
        }
        print(
            f"  [{location_tag}][AggAudit] missing_total={len(missing)}, "
            f"expected_missing_total={len(expected_missing)}, "
            f"vision_missing={len(vision_missing)}, non_vision_missing={len(non_vision_missing)}"
        )
        print(
            f"  [{location_tag}][AggAudit] keyspace: global={len(global_keys)}, "
            f"aggregated={len(aggregated_keys)}, expected={len(expected_param_names)}"
        )
        if expected_missing:
            print(
                f"  [{location_tag}][AggAudit] expected_missing_sample="
                f"{sorted(expected_missing)[:10]}"
            )

        if vision_missing:
            print(
                f"  [{location_tag}][WARNING] 视觉参数未进入聚合池（{len(vision_missing)} 个，保留上轮值）："
                + "".join(f"\n    - {m}: {c} 个" for m, c in sorted(vision_module_counts.items()))
            )

            if self.strict_aggregation_guard:
                raise RuntimeError(
                    f"[{location_tag}] strict_aggregation_guard=True 且检测到视觉参数缺失: "
                    f"{len(vision_missing)} 个。请先修复聚合覆盖后再继续实验。"
                )

        if missing:
            for k in missing:
                aggregated_state[k] = global_sd[k].clone()

            if expected_missing:
                print(f"  [{location_tag}] 补齐 {len(missing)} 个参数（视觉: {len(vision_missing)}, 其他: {len(non_vision_missing)}）")
                for mod, cnt in sorted(non_vision_module_counts.items()):
                    print(f"    - {mod}: {cnt} 个（解耦聚合，由全局值保持）")

        still_missing = set(global_sd.keys()) - set(aggregated_state.keys())
        assert not still_missing, (
            f"[{location_tag}] 防回归断言失败：补齐后仍遗漏 {len(still_missing)} 个参数：\n"
            + "\n".join(f"  - {k}" for k in sorted(still_missing))
        )
        return aggregated_state

    def _get_participating_clients_dynamic(
        self,
        param_name: str,
        client_weights_list: List[Optional[Dict[str, torch.Tensor]]],
        client_modalities: Optional[List[str]] = None
    ) -> List[int]:
        """
        非对称解耦聚合路由器（Asymmetric Decoupled Aggregation Router）。

        按五级白名单优先级路由确定参与本参数聚合的客户端下标列表。
        无合格上传者时返回空列表，调用方收到空列表后必须 continue。
        """
        uploaded_indices = [
            i for i, w in enumerate(client_weights_list)
            if w is not None and param_name in w
        ]
        route_label = classify_parameter(param_name)

        if client_modalities is None or len(client_modalities) != len(client_weights_list):
            return uploaded_indices

        route_modalities = allowed_modalities(route_label)

        filtered_indices = [
            i for i in uploaded_indices if client_modalities[i] in route_modalities
        ]

        if len(filtered_indices) == 0 and len(uploaded_indices) > 0:
            print(
                f"[WARNING][ADA-Router] 参数 '{param_name}' "
                f"收到 {len(uploaded_indices)} 个客户端上传 "
                f"（上传方模态: {[client_modalities[i] for i in uploaded_indices]}），"
                f"但白名单强制路由后无合格客户端！"
                f"\n  路由判断: route={route_label}, "
                f"allowed={sorted(route_modalities)}"
                f"\n  → 跳过此参数，保留全局模型当前值。"
            )

        return filtered_indices

    def _apply_aggregated_state(
        self,
        aggregated_state: Dict[str, torch.Tensor],
        location_tag: str,
        expected_param_names: Optional[set] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一后处理：load_state_dict → RoPE 频率重置 → missing key 校验 → 防回归补齐。
        三条聚合分支共享此入口，消除重复代码。
        """
        def _rope_signature() -> Dict[str, Tuple[Tuple[int, ...], float, float]]:
            sig: Dict[str, Tuple[Tuple[int, ...], float, float]] = {}
            for key, value in self.global_model.state_dict().items():
                if any(tok in key for tok in ('freqs_cis', 'freqs_cos', 'freqs_sin', 'relative_coords')):
                    tensor = value.detach().float()
                    mean_val = float(tensor.mean().item()) if tensor.numel() > 0 else 0.0
                    std_val = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
                    sig[key] = (tuple(tensor.shape), mean_val, std_val)
            return sig

        rope_before = _rope_signature()
        missing_keys, _ = self.global_model.load_state_dict(aggregated_state, strict=False)
        rope_after_load = _rope_signature()

        if hasattr(self.global_model, 'reset_rope_frequencies'):
            count = self.global_model.reset_rope_frequencies(verbose=False)
            if count > 0:
                rope_after_reset = _rope_signature()
                changed_on_load = sum(
                    1 for key, value in rope_after_load.items()
                    if rope_before.get(key) != value
                )
                changed_on_reset = sum(
                    1 for key, value in rope_after_reset.items()
                    if rope_after_load.get(key) != value
                )
                reset_reason = "state_load_change" if changed_on_load > 0 else "forced_refresh_or_stale"
                print(
                    f"  [RoPE Fix] 重置了 {count} 个 Block 的 RoPE 频率 "
                    f"(reason={reset_reason}, rope_buffers={len(rope_after_reset)}, "
                    f"changed_on_load={changed_on_load}, changed_on_reset={changed_on_reset})"
                )

        trainable_names = {n for n, p in self.global_model.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing_keys if k in trainable_names]
        if missing_trainable:
            print(f"[注意] 聚合状态缺失 {len(missing_trainable)} 个可训练参数（将由防回归补齐）")

        return self._safe_fill_missing_params(
            aggregated_state,
            location_tag=location_tag,
            expected_param_names=expected_param_names,
        )

    # ──────────────────────────────────────────────────────────────────
    # 梯度冲突探针
    # ──────────────────────────────────────────────────────────────────

    def _probe_gradient_conflict(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_modalities: List[str]
    ):
        """
        伪梯度冲突探针（Group B/C 自动激活，Group A 静默跳过）。
        计算 ImageOnly 与 Multimodal 客户端在 adapter 参数上的梯度夹角。
        结果写入 self._last_grad_conflict_deg，由 federated_trainer 读取后存入 JSON。
        """
        self._last_grad_conflict_deg = None
        try:
            idx_img   = client_modalities.index('image_only')
            idx_multi = client_modalities.index('multimodal')
        except ValueError:
            return

        global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        delta_img = {
            k: (client_weights[idx_img][k] - global_state[k]).to(self.device)
            for k in client_weights[idx_img]
            if 'adapter' in k and k in global_state
        }
        delta_multi = {
            k: (client_weights[idx_multi][k] - global_state[k]).to(self.device)
            for k in client_weights[idx_multi]
            if 'adapter' in k and k in global_state
        }
        if delta_img and delta_multi:
            self._last_grad_conflict_deg = self.compute_gradient_conflict(delta_img, delta_multi)

    # ──────────────────────────────────────────────────────────────────
    # 核心聚合方法
    # ──────────────────────────────────────────────────────────────────

    def _validate_parameterwise_aggregation_inputs(
        self,
        round_global_parameters: Dict[str, torch.Tensor],
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_modalities: Dict[str, str],
        client_sample_counts: Dict[str, int],
        routing_mode: str,
    ) -> Tuple[List[str], Dict[str, str]]:
        if self.aggregation_method != "fedavg":
            raise ValueError(
                "Parameterwise U/R aggregation requires aggregation_method='fedavg'"
            )
        if routing_mode not in {"unrestricted", "restricted"}:
            raise ValueError(
                "routing_mode must be 'unrestricted' or 'restricted'"
            )

        active_client_ids = sorted(client_updates)
        if not active_client_ids:
            raise ValueError("At least one active client update is required")
        expected_client_ids = set(active_client_ids)
        if (
            set(client_modalities) != expected_client_ids
            or set(client_sample_counts) != expected_client_ids
        ):
            raise ValueError(
                "client_updates, client_modalities, and client_sample_counts "
                "must contain the same client IDs"
            )

        valid_modalities = {"text_only", "image_only", "multimodal"}
        for client_id in active_client_ids:
            sample_count = client_sample_counts[client_id]
            if (
                not isinstance(sample_count, int)
                or isinstance(sample_count, bool)
                or sample_count <= 0
            ):
                raise ValueError(
                    f"client_sample_counts[{client_id}] must be a positive integer"
                )
            if client_modalities[client_id] not in valid_modalities:
                raise ValueError(
                    f"Unsupported client modality for {client_id}: "
                    f"{client_modalities[client_id]}"
                )
            if not isinstance(client_updates[client_id], dict):
                raise TypeError(f"client_updates[{client_id}] must be a dictionary")

        model_parameters = self._trainable_named_parameters()
        if set(round_global_parameters) != set(model_parameters):
            missing = sorted(set(model_parameters) - set(round_global_parameters))
            unexpected = sorted(set(round_global_parameters) - set(model_parameters))
            raise ValueError(
                "round_global_parameters must contain exactly the trainable named "
                f"parameters; missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        parameter_groups = classify_trainable_parameters(model_parameters)
        for parameter_name, global_parameter in round_global_parameters.items():
            model_parameter = model_parameters[parameter_name]
            if not isinstance(global_parameter, torch.Tensor):
                raise TypeError(
                    f"round-global value for {parameter_name} must be a tensor"
                )
            if global_parameter.shape != model_parameter.shape:
                raise ValueError(
                    f"round-global shape mismatch for {parameter_name}: "
                    f"expected {tuple(model_parameter.shape)}, "
                    f"got {tuple(global_parameter.shape)}"
                )
            if global_parameter.dtype != model_parameter.dtype:
                raise ValueError(
                    f"round-global dtype mismatch for {parameter_name}: "
                    f"expected {model_parameter.dtype}, got {global_parameter.dtype}"
                )

        for client_id, update in client_updates.items():
            for parameter_name, local_parameter in update.items():
                if parameter_name not in model_parameters:
                    raise ValueError(
                        f"Client {client_id} uploaded a non-trainable or buffer key: "
                        f"{parameter_name}"
                    )
                if not isinstance(local_parameter, torch.Tensor):
                    raise TypeError(
                        f"Client {client_id} uploaded a non-tensor value for "
                        f"{parameter_name}"
                    )
                global_parameter = round_global_parameters[parameter_name]
                if local_parameter.shape != global_parameter.shape:
                    raise ValueError(
                        f"Client {client_id} upload shape mismatch for "
                        f"{parameter_name}: expected {tuple(global_parameter.shape)}, "
                        f"got {tuple(local_parameter.shape)}"
                    )
                if local_parameter.dtype != global_parameter.dtype:
                    raise ValueError(
                        f"Client {client_id} upload dtype mismatch for "
                        f"{parameter_name}: expected {global_parameter.dtype}, "
                        f"got {local_parameter.dtype}"
                    )

        return active_client_ids, parameter_groups

    def aggregate_weights(
        self,
        *,
        round_global_parameters: Dict[str, torch.Tensor],
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_modalities: Dict[str, str],
        client_sample_counts: Dict[str, int],
        routing_mode: str,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate optimizer-scoped parameter deltas under the U/R contract."""
        active_client_ids, parameter_groups = (
            self._validate_parameterwise_aggregation_inputs(
                round_global_parameters=round_global_parameters,
                client_updates=client_updates,
                client_modalities=client_modalities,
                client_sample_counts=client_sample_counts,
                routing_mode=routing_mode,
            )
        )

        aggregated_parameters: Dict[str, torch.Tensor] = {}
        parameter_audit: Dict[str, Dict[str, Any]] = {}
        empty_eligible_parameter_names: List[str] = []

        for parameter_name in sorted(round_global_parameters):
            parameter_group = parameter_groups[parameter_name]
            global_parameter = round_global_parameters[parameter_name].to(self.device)
            uploaded_client_ids = [
                client_id
                for client_id in active_client_ids
                if parameter_name in client_updates[client_id]
            ]

            if routing_mode == "unrestricted":
                eligible_client_ids = list(active_client_ids)
            else:
                allowed = allowed_modalities(parameter_group)
                eligible_client_ids = [
                    client_id
                    for client_id in uploaded_client_ids
                    if client_modalities[client_id] in allowed
                ]

            empty_eligible = not eligible_client_ids
            if empty_eligible:
                aggregated_parameters[parameter_name] = global_parameter.clone()
                empty_eligible_parameter_names.append(parameter_name)
                parameter_audit[parameter_name] = {
                    "parameter_group": parameter_group,
                    "uploaded_client_ids": uploaded_client_ids,
                    "eligible_client_ids": [],
                    "zero_update_client_ids": [],
                    "sample_weights": {},
                    "normalized_weights": {},
                    "empty_eligible": True,
                }
                continue

            denominator = sum(
                client_sample_counts[client_id]
                for client_id in eligible_client_ids
            )
            aggregated_delta = torch.zeros_like(global_parameter)
            zero_update_client_ids: List[str] = []
            normalized_weights: Dict[str, float] = {}

            for client_id in eligible_client_ids:
                normalized_weight = (
                    client_sample_counts[client_id] / float(denominator)
                )
                normalized_weights[client_id] = normalized_weight
                if parameter_name in client_updates[client_id]:
                    local_parameter = client_updates[client_id][parameter_name].to(
                        self.device
                    )
                    delta = local_parameter - global_parameter
                else:
                    delta = torch.zeros_like(global_parameter)

                if torch.count_nonzero(delta).item() == 0:
                    zero_update_client_ids.append(client_id)
                aggregated_delta = aggregated_delta + normalized_weight * delta

            aggregated_parameters[parameter_name] = (
                global_parameter + aggregated_delta
            )
            parameter_audit[parameter_name] = {
                "parameter_group": parameter_group,
                "uploaded_client_ids": uploaded_client_ids,
                "eligible_client_ids": eligible_client_ids,
                "zero_update_client_ids": zero_update_client_ids,
                "sample_weights": {
                    client_id: client_sample_counts[client_id]
                    for client_id in eligible_client_ids
                },
                "normalized_weights": normalized_weights,
                "empty_eligible": False,
            }

        persistent_buffers, nonpersistent_buffers = self._buffer_inventory()
        self._last_aggregation_audit = {
            "aggregation_method": "fedavg",
            "routing_mode": routing_mode,
            "active_client_ids": active_client_ids,
            "client_sample_counts": {
                client_id: client_sample_counts[client_id]
                for client_id in active_client_ids
            },
            "parameter_key_count": len(aggregated_parameters),
            "buffer_key_count": len(persistent_buffers) + len(nonpersistent_buffers),
            "persistent_buffer_key_count": len(persistent_buffers),
            "nonpersistent_buffer_key_count": len(nonpersistent_buffers),
            "empty_eligible_parameter_names": empty_eligible_parameter_names,
            "parameters": parameter_audit,
        }
        return aggregated_parameters

    def _update_global_reps_decoupled(
        self,
        image_reps: List[torch.Tensor],
        text_reps: List[torch.Tensor]
    ):
        """
        解耦 EMA 更新全局表征。
        image_reps 来自 image_only + multimodal，text_reps 来自 text_only + multimodal。
        """
        alpha = self.global_rep_alpha

        def _align_and_avg(reps: List[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
            aligned = []
            for rep in reps:
                rep = rep.to(self.device)
                if rep.dim() > 1:
                    rep = rep.mean(dim=0) if rep.shape[0] > 1 else rep.squeeze(0)
                curr = rep.shape[0]
                if curr > target_dim:
                    rep = rep[:target_dim]
                elif curr < target_dim:
                    rep = nn.functional.pad(rep, (0, target_dim - curr))
                aligned.append(rep)
            return torch.stack(aligned, dim=0).mean(dim=0) if aligned else None

        if image_reps:
            avg = _align_and_avg(image_reps, self.global_image_rep.shape[0])
            if avg is not None:
                self.global_image_rep = nn.functional.normalize(
                    alpha * self.global_image_rep + (1 - alpha) * avg, p=2, dim=0
                )

        if text_reps:
            avg = _align_and_avg(text_reps, self.global_text_rep.shape[0])
            if avg is not None:
                self.global_text_rep = nn.functional.normalize(
                    alpha * self.global_text_rep + (1 - alpha) * avg, p=2, dim=0
                )

    def aggregate_heterogeneous_clients(
        self,
        client_updates: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]],
        global_features_for_contrastive: Optional[torch.Tensor] = None,
        client_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        异构客户端聚合接口。

        处理三元组上传：(weights, image_rep, text_rep)
        - text_only  客户端: (None, None, text_rep)
        - image_only 客户端: (weights, image_rep, None)
        - multimodal 客户端: (weights, image_rep, text_rep)

        Args:
            client_updates:                  客户端上传的三元组列表
            global_features_for_contrastive: 用于对比权重计算的全局特征（可选）
            client_modalities:               客户端模态列表（用于解耦聚合）
        Returns:
            聚合后的全局模型状态字典
        """
        if not client_updates:
            print("警告: 没有客户端上传数据，保持全局模型不变")
            return self.global_model.state_dict()

        client_weights: List[Optional[Dict[str, torch.Tensor]]] = []
        image_reps: List[torch.Tensor] = []
        text_reps: List[torch.Tensor] = []
        participating_indices: List[int] = []

        for idx, (weights, img_rep, txt_rep) in enumerate(client_updates):
            modality = (
                client_modalities[idx]
                if client_modalities is not None and idx < len(client_modalities)
                else None
            )
            if weights is not None:
                client_weights.append(weights)
                participating_indices.append(idx)
            if img_rep is not None:
                image_reps.append(img_rep)
            if txt_rep is not None:
                if modality is None or modality in ('text_only', 'multimodal'):
                    text_reps.append(txt_rep)
                else:
                    # image_only 客户端意外上传了文本表征 → 拦截，防止随机视觉特征污染全局文本锚点
                    print(
                        f"[WARNING][Server] 客户端 {idx}（modality={modality}）"
                        f"意外上传了 txt_rep！已拦截，不纳入全局文本表征聚合池。"
                    )

        if not client_weights:
            print("⚠️ 警告: 没有客户端提供模型权重（所有客户端都是 text_only），跳过模型聚合")
            self._update_global_reps_decoupled(image_reps, text_reps)
            return self.global_model.state_dict()

        participating_reps = []
        for idx in participating_indices:
            _, img_rep, _ = client_updates[idx]
            if img_rep is not None:
                participating_reps.append(img_rep)
            else:
                print(f"警告: 客户端 {idx} 提供了权重但没有图像表征，使用零向量")
                participating_reps.append(torch.zeros(self.global_image_rep.shape[0]))

        participating_modalities = (
            [client_modalities[idx] for idx in participating_indices]
            if client_modalities is not None and len(client_modalities) == len(client_updates)
            else None
        )

        print(f"[Heterogeneous Aggregation] 参与模型聚合的客户端: {len(client_weights)}/{len(client_updates)}")
        print(f"[Heterogeneous Aggregation] 图像表征数量: {len(image_reps)}, 文本表征数量: {len(text_reps)}")

        aggregated_state = self.aggregate_weights(
            client_weights=client_weights,
            client_public_reps=participating_reps,
            global_features_for_contrastive=global_features_for_contrastive,
            client_modalities=participating_modalities
        )

        self._update_global_reps_decoupled(image_reps, text_reps)
        return aggregated_state

    def generate_and_dispatch_global_proxies(
        self,
        global_model,
        public_dataloader,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从多模态公共数据取固定 K 个 Batch，提取图像和文本全局表征。

        设计约束：
          ① 在 torch.no_grad() 上下文内执行，禁止构建正向计算图；
          ② 返回张量强制 .detach()，物理截断跨客户端反向传播路径；
          ③ 文本代理必须来自公共文本特征，禁止使用随机服务器向量代替；
          ④ 图像和文本代理均位于 contrastive_dim 空间。

        图像表征：调用 extract_features 提取，经 mean-pool 压缩为 1-D 向量。
        文本表征：使用 fusion_head.project_text 投影公共文本特征。

        Returns:
            (image_proxy, text_proxy)，形状均为 (D,)，已 detach。
        """
        global_model.eval()

        data_iter = iter(public_dataloader)
        aggregated_img_vec: Optional[torch.Tensor] = None
        aggregated_text_vec: Optional[torch.Tensor] = None
        used_batches = 0

        with torch.no_grad():
            while used_batches < self.proxy_k_batches:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                if isinstance(batch, dict):
                    pub_imgs = batch.get("image", batch.get("inp"))
                    public_text_features = batch.get("text_feature")
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    pub_imgs = batch[0]
                    public_text_features = batch[1]
                else:
                    raise RuntimeError(
                        "Proxy loader must provide (image, text_feature) batches"
                    )
                if not isinstance(pub_imgs, torch.Tensor):
                    raise RuntimeError("Proxy batch image must be a tensor")
                if not isinstance(public_text_features, torch.Tensor):
                    raise RuntimeError("Proxy batch text_feature must be a tensor")

                pub_imgs = pub_imgs.to(device)
                public_text_features = public_text_features.to(device)
                raw_feats = global_model.extract_features(pub_imgs)
                projected_text = global_model.fusion_head.project_text(
                    public_text_features
                )

                if raw_feats.dim() == 3:
                    img_vec = raw_feats.mean(dim=(0, 1))
                elif raw_feats.dim() == 2:
                    img_vec = raw_feats.mean(dim=0)
                else:
                    img_vec = raw_feats.flatten()
                img_vec = nn.functional.normalize(img_vec, p=2, dim=0)

                if projected_text.dim() == 3:
                    projected_text = projected_text.mean(dim=1)
                if projected_text.dim() != 2:
                    raise RuntimeError(
                        "Projected public text features must have shape (B, D)"
                    )
                text_vec = nn.functional.normalize(
                    projected_text.mean(dim=0), p=2, dim=0
                )

                if aggregated_img_vec is None:
                    aggregated_img_vec = torch.zeros_like(img_vec, device=self.device)
                    aggregated_text_vec = torch.zeros_like(
                        text_vec, device=self.device
                    )
                aggregated_img_vec = aggregated_img_vec + img_vec.detach().to(self.device)
                aggregated_text_vec = (
                    aggregated_text_vec + text_vec.detach().to(self.device)
                )
                pub_imgs = public_text_features = raw_feats = projected_text = None
                used_batches += 1

        if (
            used_batches == 0
            or aggregated_img_vec is None
            or aggregated_text_vec is None
        ):
            raise RuntimeError(
                "Cannot generate global proxies from an empty public dataloader"
            )

        image_proxy: torch.Tensor = nn.functional.normalize(
            aggregated_img_vec / float(used_batches), p=2, dim=0
        ).detach()
        text_proxy: torch.Tensor = nn.functional.normalize(
            aggregated_text_vec / float(used_batches), p=2, dim=0
        ).detach()
        aggregated_img_vec = aggregated_text_vec = None

        if image_proxy.shape != self.global_image_rep.shape:
            raise RuntimeError(
                "Image proxy dimension differs from the server representation contract: "
                f"{tuple(image_proxy.shape)} != {tuple(self.global_image_rep.shape)}"
            )
        if text_proxy.shape != self.global_text_rep.shape:
            raise RuntimeError(
                "Text proxy dimension differs from the server representation contract: "
                f"{tuple(text_proxy.shape)} != {tuple(self.global_text_rep.shape)}"
            )

        alpha = self.global_rep_alpha
        self.global_image_rep = nn.functional.normalize(
            alpha * self.global_image_rep + (1 - alpha) * image_proxy,
            p=2, dim=0
        )
        self.global_text_rep = nn.functional.normalize(
            alpha * self.global_text_rep + (1 - alpha) * text_proxy,
            p=2, dim=0
        )

        print(
            f"[generate_proxies] image_proxy.shape={image_proxy.shape}, "
            f"text_proxy.shape={text_proxy.shape}, used_batches={used_batches}/{self.proxy_k_batches}  "
            f"✓ (两张量已 detach，计算图已截断)"
        )
        return image_proxy, text_proxy

    @staticmethod
    def compute_gradient_conflict(
        grad_dict_img: Dict[str, torch.Tensor],
        grad_dict_multi: Dict[str, torch.Tensor],
    ) -> float:
        """
        梯度余弦相似度探针：诊断 ImageOnly 与 Multimodal 客户端的 adapter 梯度冲突。

        提取含 'adapter' 关键字的参数梯度，展平后拼接为 1-D 向量，计算余弦夹角（度）。
        0° = 完全同向（无冲突），90° = 正交，180° = 完全反向（严重毒化）。
        任一方无 adapter 梯度时，返回哨兵值 180.0。
        """
        img_grads = [
            v.detach().float().flatten()
            for k, v in grad_dict_img.items()
            if 'adapter' in k and v is not None
        ]
        multi_grads = [
            v.detach().float().flatten()
            for k, v in grad_dict_multi.items()
            if 'adapter' in k and v is not None
        ]

        if not img_grads or not multi_grads:
            print(
                "[WARNING][compute_gradient_conflict] "
                f"img_grads={len(img_grads)} 个, multi_grads={len(multi_grads)} 个，"
                "某一方无 adapter 梯度，无法计算冲突角，返回哨兵值 180.0°"
            )
            return 180.0

        vec_img   = torch.cat(img_grads, dim=0)
        vec_multi = torch.cat(multi_grads, dim=0)
        img_grads = multi_grads = None

        min_len   = min(vec_img.shape[0], vec_multi.shape[0])
        vec_img   = vec_img[:min_len]
        vec_multi = vec_multi[:min_len]

        img_norm = torch.linalg.norm(vec_img).item()
        multi_norm = torch.linalg.norm(vec_multi).item()
        dot_val = torch.dot(vec_img, vec_multi).item()
        img_nonzero_ratio = float((vec_img.abs() > 0).float().mean().item()) if min_len > 0 else 0.0
        multi_nonzero_ratio = float((vec_multi.abs() > 0).float().mean().item()) if min_len > 0 else 0.0
        print(
            "[GradConflictDiag] "
            f"img_norm={img_norm:.4e}, multi_norm={multi_norm:.4e}, "
            f"img_nonzero_ratio={img_nonzero_ratio:.4f}, "
            f"multi_nonzero_ratio={multi_nonzero_ratio:.4f}, "
            f"overlap_dim={min_len}, dot={dot_val:.4e}"
        )
        if img_norm <= 1e-12 or multi_norm <= 1e-12:
            print(
                "[WARNING][compute_gradient_conflict] "
                "one gradient vector norm is ~0; conflict angle is not reliable, return sentinel 180.0°"
            )
            return 180.0

        cos_sim = max(-1.0, min(1.0, nn.functional.cosine_similarity(
            vec_img.unsqueeze(0), vec_multi.unsqueeze(0)
        ).item()))
        angle_deg = math.degrees(math.acos(cos_sim))

        vec_img = vec_multi = None

        print(
            f"[GradConflict] adapter 梯度向量冲突角: {angle_deg:.2f}°  "
            f"(cos_sim={cos_sim:.4f}, adapter_params={min_len})"
        )
        if angle_deg > 90.0:
            print(
                f"  ⚠️ 冲突角 > 90°，ImageOnly 与 Multimodal 客户端 adapter 梯度方向相反！"
                f"建议检查 lambda_cream 权重或降低 multimodal 客户端学习率。"
            )
        return angle_deg

    # ──────────────────────────────────────────────────────────────────
    # 检查点接口
    # ──────────────────────────────────────────────────────────────────

    def get_global_reps(self) -> Dict[str, torch.Tensor]:
        return {
            'global_text_rep': self.global_text_rep.clone().cpu(),
            'global_image_rep': self.global_image_rep.clone().cpu(),
        }

    def get_global_model(self) -> SAM3_Medical:
        return self.global_model

    def set_global_model(self, model: SAM3_Medical):
        self.global_model = model.to(self.device)
        self._round_buffer_snapshot = None
        self._round_buffer_distribution_audit = {}

    def get_state_dict(self) -> Dict[str, Any]:
        """Save explicit server-owned parameters and persistent buffers."""
        persistent_buffers, _ = self._buffer_inventory()
        return {
            'trainable_parameters': self.get_trainable_parameter_snapshot(),
            'persistent_buffers': self._clone_buffer_snapshot(persistent_buffers),
            'global_text_rep': self.global_text_rep.cpu().clone(),
            'global_image_rep': self.global_image_rep.cpu().clone(),
            'aggregation_method': self.aggregation_method,
            'global_rep_alpha': self.global_rep_alpha,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Restore explicit server-owned parameters and persistent buffers."""
        required_keys = {'trainable_parameters', 'persistent_buffers'}
        missing_keys = sorted(required_keys - set(state_dict))
        if missing_keys:
            raise RuntimeError(
                "Server checkpoint is missing explicit state fields: "
                f"{missing_keys}"
            )
        self.apply_trainable_parameters(state_dict['trainable_parameters'])
        self.restore_round_buffer_snapshot(
            state_dict['persistent_buffers'],
            reason='checkpoint_restore',
        )

        if 'global_text_rep' in state_dict:
            self.global_text_rep = state_dict['global_text_rep'].to(self.device)
        if 'global_image_rep' in state_dict:
            self.global_image_rep = state_dict['global_image_rep'].to(self.device)

        saved_method = state_dict.get('aggregation_method')
        if saved_method and saved_method != self.aggregation_method:
            print(f"警告: 检查点中的聚合方法 ({saved_method}) 与当前配置 ({self.aggregation_method}) 不一致")

        saved_alpha = state_dict.get('global_rep_alpha')
        if saved_alpha and abs(saved_alpha - self.global_rep_alpha) > 1e-6:
            print(f"警告: 检查点中的 global_rep_alpha ({saved_alpha}) 与当前配置 ({self.global_rep_alpha}) 不一致")

    def setup_distillation(
        self,
        optimizer: torch.optim.Optimizer,
        kd_weight: float = 1.0,
        use_fp16: bool = False,
        grad_clip: float = 0.0
    ):
        """初始化知识蒸馏器。"""
        self.distiller = KnowledgeDistillation(
            model=self.global_model,
            optimizer=optimizer,
            device=self.device,
            kd_weight=kd_weight,
            use_fp16=use_fp16,
            grad_clip=grad_clip
        )

    def distill_global_model(
        self,
        public_loader,
        aggregated_img_features: torch.Tensor,
        aggregated_txt_features: Optional[torch.Tensor] = None,
        distill_index: Optional[List[int]] = None,
        modality_types: List[str] = ["image", "text"],
        num_epochs: int = 1
    ) -> Dict[str, List[float]]:
        """
        在公共数据集上对全局模型进行知识蒸馏。

        Args:
            public_loader:           公共数据集 DataLoader
            aggregated_img_features: 聚合后的图像特征 (N, D)
            aggregated_txt_features: 聚合后的文本特征 (N, D)（可选）
            distill_index:           公共数据索引列表（用于映射到聚合特征）
            modality_types:          要蒸馏的模态类型列表
            num_epochs:              蒸馏轮数
        Returns:
            训练历史字典
        """
        if self.distiller is None:
            raise RuntimeError("Distiller not initialized. Call setup_distillation() first.")

        aggregated_features = {'img_vec': aggregated_img_features.to(self.device)}
        if aggregated_txt_features is not None:
            aggregated_features['txt_vec'] = aggregated_txt_features.to(self.device)

        if distill_index is None:
            distill_index = list(range(aggregated_img_features.shape[0]))

        return self.distiller.distill(
            public_loader=public_loader,
            aggregated_features=aggregated_features,
            distill_index=distill_index,
            modality_types=modality_types,
            num_epochs=num_epochs
        )
