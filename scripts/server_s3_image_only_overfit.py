"""Run the S3 real-sample fixed-slice overfit gate on the target server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.heterogeneous_dataset_loader import (  # noqa: E402
    HeterogeneousBraTSDataset,
    heterogeneous_collate_fn,
)
from src.config_manager import FederatedConfig  # noqa: E402
from src.federated_trainer import FederatedTrainer  # noqa: E402
from src.parameter_groups import VISION_ADAPTER, classify_parameter  # noqa: E402


S3_KEYS = {
    "client_id",
    "min_source_wt_pixels",
    "min_loss_reduction_ratio",
    "min_logit_std",
    "min_predicted_wt_pixels",
}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class FixedImageMaskDataset(Dataset):
    def __init__(
        self,
        *,
        image: torch.Tensor,
        mask: torch.Tensor,
        case_id: str,
        slice_index: int,
    ) -> None:
        self.image = image.detach().cpu().clone()
        self.mask = mask.detach().cpu().clone()
        self.manifest = {
            "dataset_type": "fixed_real_slice",
            "case_ids": [case_id],
            "slice_index": int(slice_index),
            "image_sha256": _tensor_sha256(self.image),
            "mask_sha256": _tensor_sha256(self.mask),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return self.image.clone(), self.mask.clone()

    def get_reproducibility_manifest(self) -> dict[str, Any]:
        return dict(self.manifest)


def _image_only_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return heterogeneous_collate_fn(batch, "image_only")


def _load_s3_contract(config_path: Path) -> tuple[FederatedConfig, dict[str, Any]]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    s3_contract = raw_config.get("s3_overfit")
    if not isinstance(s3_contract, dict) or set(s3_contract) != S3_KEYS:
        raise ValueError(f"s3_overfit must define exactly {sorted(S3_KEYS)}")

    integer_keys = {
        "min_source_wt_pixels",
        "min_predicted_wt_pixels",
    }
    for key in integer_keys:
        value = s3_contract[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"s3_overfit.{key} must be a positive integer")

    ratio = s3_contract["min_loss_reduction_ratio"]
    if (
        isinstance(ratio, bool)
        or not math.isfinite(float(ratio))
        or not 0.0 < float(ratio) < 1.0
    ):
        raise ValueError(
            "s3_overfit.min_loss_reduction_ratio must be finite and in (0, 1)"
        )

    min_logit_std = s3_contract["min_logit_std"]
    if (
        isinstance(min_logit_std, bool)
        or not math.isfinite(float(min_logit_std))
        or float(min_logit_std) <= 0.0
    ):
        raise ValueError("s3_overfit.min_logit_std must be finite and positive")

    config = FederatedConfig.from_yaml(str(config_path))
    client_id = str(s3_contract["client_id"])
    enabled_clients = [
        client
        for client in config.clients or []
        if bool(client.get("enabled", True))
    ]
    if len(enabled_clients) != 1:
        raise ValueError("S3 requires exactly one enabled image_only client")
    enabled_client = enabled_clients[0]
    if (
        str(enabled_client.get("client_id")) != client_id
        or str(enabled_client.get("modality")) != "image_only"
    ):
        raise ValueError("S3 requires exactly one enabled image_only client")
    if config.batch_size != 1 or config.accumulation_steps != 1:
        raise ValueError("S3 requires batch_size=1 and accumulation_steps=1")
    if config.lambda_cream != 0.0 or config.baseline_method != "none":
        raise ValueError("S3 isolates segmentation and requires no Cream or FedProx")
    return config, s3_contract


def _select_fixed_slice(
    *,
    config: FederatedConfig,
    client_id: str,
    min_wt_pixels: int,
) -> tuple[FixedImageMaskDataset, dict[str, Any]]:
    source_dataset = HeterogeneousBraTSDataset(
        data_dir=str(Path(config.data_root) / "train" / client_id),
        mode="private",
        client_type="image_only",
        image_size=config.img_size,
        max_samples=config.max_samples,
        load_mask=True,
        include_text_features=False,
        is_validation=True,
    )
    if source_dataset.samples is None:
        raise RuntimeError("S3 source dataset did not enumerate deterministic slices")

    for sample_index, (case_index, slice_index) in enumerate(source_dataset.samples):
        image, mask = source_dataset[sample_index]
        wt_pixels = int(mask[0].sum().item())
        if wt_pixels < min_wt_pixels:
            continue
        case_id = source_dataset.case_folders[case_index].name
        fixed_dataset = FixedImageMaskDataset(
            image=image,
            mask=mask,
            case_id=case_id,
            slice_index=slice_index,
        )
        selection = {
            "case_id": case_id,
            "slice_index": int(slice_index),
            "source_wt_pixels": wt_pixels,
        }
        return fixed_dataset, selection

    raise RuntimeError(
        "No deterministic source slice satisfies the configured WT foreground minimum"
    )


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits", next(iter(output.values())))
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output does not contain a logits tensor")
    return output


def _evaluate_fixed_sample(
    trainer: FederatedTrainer,
    fixed_dataset: FixedImageMaskDataset,
    *,
    client_id: str,
) -> dict[str, Any]:
    image, mask = fixed_dataset[0]
    image = image.unsqueeze(0).to(trainer.device)
    mask = mask.unsqueeze(0).to(trainer.device)
    model = trainer.global_model
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = _extract_logits(model(image, global_text_rep=None))
            loss = trainer.client_trainers[client_id]._compute_segmentation_loss(
                logits,
                mask,
            )
            metric_logits = F.interpolate(
                logits.float(),
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    finally:
        model.train(was_training)

    values = torch.stack(
        (
            loss.detach().float(),
            metric_logits.detach().float().std(unbiased=False),
        )
    )
    if not torch.isfinite(values).all():
        raise RuntimeError("S3 fixed-sample evaluation produced non-finite values")
    wt_threshold = float(trainer.config.segmentation_thresholds[0])
    predicted_wt_pixels = int(
        (torch.sigmoid(metric_logits[:, 0]) >= wt_threshold).sum().item()
    )
    return {
        "loss": float(loss.item()),
        "logit_std": float(metric_logits.std(unbiased=False).item()),
        "predicted_wt_pixels": predicted_wt_pixels,
    }


def _parameter_update_audit(
    trainer: FederatedTrainer,
    initial_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    seg_head_updates = []
    vision_adapter_updates = []
    for name, parameter in trainer.global_model.named_parameters():
        if name not in initial_parameters:
            continue
        delta = parameter.detach().cpu() - initial_parameters[name]
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"S3 produced a non-finite parameter delta: {name}")
        if torch.count_nonzero(delta).item() == 0:
            continue
        if "medical_seg_head" in name:
            seg_head_updates.append(name)
        if classify_parameter(name) == VISION_ADAPTER:
            vision_adapter_updates.append(name)
    return {
        "seg_head_updated_names": sorted(seg_head_updates),
        "vision_adapter_updated_names": sorted(vision_adapter_updates),
    }


def _install_fixed_loader(
    trainer: FederatedTrainer,
    *,
    client_id: str,
    fixed_dataset: FixedImageMaskDataset,
) -> None:
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_image_only_collate,
        pin_memory=torch.cuda.is_available(),
    )
    trainer.client_configs[client_id]["private_loader"] = fixed_loader
    trainer.client_trainers[client_id].private_loader = fixed_loader
    trainer.client_sample_counts[client_id] = len(fixed_dataset)
    trainer._initial_loader_randomness[client_id]["private_loader"] = (
        trainer._configure_loader_stream(
            round_num=0,
            client_id=client_id,
            stream="private_initialization",
            loader=fixed_loader,
        )
    )


def run(config_path: Path) -> int:
    config, contract = _load_s3_contract(config_path)
    log_dir = Path(config.log_dir)
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(
            f"S3 log directory is not empty; preserve it before rerun: {log_dir}"
        )

    trainer = FederatedTrainer(config)
    trainer.setup_environment()
    trainer.setup_clients()

    client_id = str(contract["client_id"])
    fixed_dataset, selection = _select_fixed_slice(
        config=config,
        client_id=client_id,
        min_wt_pixels=int(contract["min_source_wt_pixels"]),
    )
    _install_fixed_loader(
        trainer,
        client_id=client_id,
        fixed_dataset=fixed_dataset,
    )
    trainer.setup_validation()
    trainer.setup_logging()
    trainer.training_history["run_metadata"] = trainer._collect_run_metadata()

    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.global_model.named_parameters()
        if parameter.requires_grad
    }
    initial_evaluation = _evaluate_fixed_sample(
        trainer,
        fixed_dataset,
        client_id=client_id,
    )
    print(
        "[S3] Fixed real slice: "
        f"case={selection['case_id']}, slice={selection['slice_index']}, "
        f"WT={selection['source_wt_pixels']}"
    )
    print(f"[S3] Initial evaluation: {initial_evaluation}")

    for round_num in range(1, config.rounds + 1):
        trainer._train_single_round(round_num)

    final_evaluation = _evaluate_fixed_sample(
        trainer,
        fixed_dataset,
        client_id=client_id,
    )
    parameter_updates = _parameter_update_audit(trainer, initial_parameters)
    initial_loss = float(initial_evaluation["loss"])
    final_loss = float(final_evaluation["loss"])
    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
    checks = {
        "loss_reduction": (
            loss_reduction_ratio
            >= float(contract["min_loss_reduction_ratio"])
        ),
        "nonconstant_logits": (
            float(final_evaluation["logit_std"])
            >= float(contract["min_logit_std"])
        ),
        "nonzero_wt_prediction": (
            int(final_evaluation["predicted_wt_pixels"])
            >= int(contract["min_predicted_wt_pixels"])
        ),
        "seg_head_update": bool(parameter_updates["seg_head_updated_names"]),
        "vision_adapter_update": bool(
            parameter_updates["vision_adapter_updated_names"]
        ),
    }
    result = {
        "gate": "S3",
        "passed": all(checks.values()),
        "checks": checks,
        "selection": selection,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "loss_reduction_ratio": loss_reduction_ratio,
        "parameter_updates": parameter_updates,
        "git_commit": trainer.training_history["run_metadata"].get("git_commit"),
        "config_path": str(config_path),
    }
    trainer.training_history["s3_overfit_result"] = result
    trainer._finalize_training()

    result_path = log_dir / "s3_overfit_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[S3] Final evaluation: {final_evaluation}")
    print(f"[S3] Result saved to: {result_path}")
    print(f"[S3] Gate status: {'PASS' if result['passed'] else 'FAIL'}")
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
