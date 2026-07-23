#!/usr/bin/env python3

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "core_projects/sam3-main/sam3/model_builder.py",
    "core_projects/SAM-Adapter-PyTorch-main/models/block.py",
    "core_projects/CreamFL-main/src/criterions/probemb.py",
    "core_projects/FedFMS-main/README.md",
)
IMPORTS = (
    "torch",
    "torchvision",
    "numpy",
    "yaml",
    "monai",
    "timm",
    "psutil",
    "nibabel",
    "sklearn",
    "skimage",
    "tensorboard",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_config(path: Path) -> dict:
    yaml = importlib.import_module("yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Invalid YAML root: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="FedSAM3 server training preflight")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    _require(config_path.is_file(), f"Config not found: {config_path}")

    for module_name in IMPORTS:
        importlib.import_module(module_name)

    for relative_path in CORE_FILES:
        path = PROJECT_ROOT / relative_path
        _require(path.is_file(), f"Required core source not found: {relative_path}")

    config = _load_config(config_path)
    model_config = config.get("model", {})
    federated_config = config.get("federated", {})
    server_config = config.get("server", {})

    checkpoint = PROJECT_ROOT / model_config.get(
        "sam3_checkpoint",
        "data/checkpoints/sam3.pt",
    )
    _require(checkpoint.is_file(), f"SAM3 checkpoint not found: {checkpoint}")

    enabled_clients = [
        client
        for client in federated_config.get("clients", [])
        if client.get("enabled", True)
    ]
    _require(enabled_clients, "No enabled clients in config")
    for client in enabled_clients:
        data_source = PROJECT_ROOT / str(client.get("data_source", ""))
        _require(
            data_source.is_file(),
            f"Dataset manifest not found for {client.get('client_id')}: {data_source}",
        )

    proxy_client_id = str(server_config.get("proxy_client_id", "")).strip()
    _require(proxy_client_id, "server.proxy_client_id is required")
    proxy_matches = [
        client
        for client in enabled_clients
        if str(client.get("client_id", "")).replace("_", "").lower()
        == proxy_client_id.replace("_", "").lower()
    ]
    _require(len(proxy_matches) == 1, f"Proxy client not found: {proxy_client_id}")
    _require(
        proxy_matches[0].get("modality") == "multimodal",
        f"Proxy client must be multimodal: {proxy_client_id}",
    )

    seed = config.get("seed")
    _require(isinstance(seed, int) and seed >= 0, "training.seed must be a non-negative integer")

    git_commit = _git_output("rev-parse", "HEAD")
    git_status = _git_output("status", "--porcelain")
    _require(not git_status, "Git worktree must be clean before server training")

    torch = importlib.import_module("torch")
    _require(torch.cuda.is_available(), "CUDA is required for production training")

    sys.path.insert(0, str(PROJECT_ROOT))
    importlib.import_module("src.integrated_model")
    importlib.import_module("src.federated_trainer")

    summary = {
        "status": "ready",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "git_commit": git_commit,
        "seed": seed,
        "proxy_client_id": proxy_client_id,
        "enabled_clients": [
            {
                "client_id": client.get("client_id"),
                "modality": client.get("modality"),
            }
            for client in enabled_clients
        ],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
