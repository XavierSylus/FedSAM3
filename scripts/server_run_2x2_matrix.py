"""Run and audit the formal four-cell federated experiment matrix."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "fedsam3_experiment_manifest.json"
EXPECTED_CELLS = (
    "U-FedAvg",
    "U-FedProx",
    "R-FedAvg",
    "R-FedProx",
)
CONTRACT_TESTS = (
    "tests/test_parameterwise_aggregation.py",
    "tests/test_fedprox_same_loss_contract.py",
    "tests/test_experiment_matrix_contract.py",
    "tests/test_server_2x2_runner_contract.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _log(handle: TextIO, message: str) -> None:
    handle.write(f"[{_now()}] {message}\n")
    handle.flush()


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


def _matrix_definition() -> tuple[list[dict[str, Any]], Path]:
    manifest = _read_json(MANIFEST_PATH)
    matrix = manifest["matrix"]
    if not isinstance(matrix, list):
        raise TypeError("Experiment manifest matrix must be a list")
    if tuple(entry.get("cell") for entry in matrix) != EXPECTED_CELLS:
        raise ValueError("Experiment manifest must declare the formal four-cell order")

    fixed = manifest.get("fixed_controls")
    if not isinstance(fixed, dict):
        raise TypeError("Experiment manifest fixed_controls must be an object")
    sample_weight = fixed.get("sample_weight")
    proxy = fixed.get("proxy")
    if not isinstance(sample_weight, dict) or not isinstance(proxy, dict):
        raise TypeError("Experiment manifest fixed controls are incomplete")

    cells: list[dict[str, Any]] = []
    log_dirs: list[Path] = []
    data_roots: set[str] = set()
    for entry in matrix:
        config_relative = Path(str(entry["config"]))
        config_path = (PROJECT_ROOT / config_relative).resolve()
        config_path.relative_to(PROJECT_ROOT)
        if not config_path.is_file():
            raise FileNotFoundError(f"Formal config is missing: {config_relative}")
        config = _load_yaml(config_path)

        federated = config.get("federated")
        training = config.get("training")
        server = config.get("server")
        aggregation = config.get("aggregation")
        baseline = config.get("baseline")
        logging = config.get("logging")
        sections = (federated, training, server, aggregation, baseline, logging)
        if any(not isinstance(section, dict) for section in sections):
            raise TypeError(f"Formal config is incomplete: {config_relative}")

        comparisons = {
            "seed": config.get("seed") == manifest.get("seed"),
            "rounds": training.get("rounds") == fixed.get("rounds"),
            "local_epochs": (
                training.get("local_epochs") == fixed.get("local_epochs")
            ),
            "routing_mode": (
                federated.get("routing_mode") == entry.get("routing_mode")
            ),
            "unoptimized_update_policy": (
                aggregation.get("unoptimized_update_policy")
                == entry.get("unoptimized_update_policy")
            ),
            "baseline_method": (
                baseline.get("method") == entry.get("baseline_method")
            ),
            "fedprox_mu": baseline.get("mu") == entry.get("fedprox_mu"),
            "sample_weight_unit": (
                aggregation.get("sample_weight_unit")
                == sample_weight.get("unit")
            ),
            "proxy_client_id": (
                server.get("proxy_client_id") == proxy.get("client_id")
            ),
            "proxy_k_batches": (
                server.get("proxy_k_batches") == proxy.get("batches")
            ),
            "client_init_policy": (
                federated.get("client_init_policy")
                == fixed.get("client_init_policy")
            ),
            "persist_client_optimizer": (
                federated.get("persist_client_optimizer")
                == fixed.get("persist_client_optimizer")
            ),
        }
        failed = [name for name, passed in comparisons.items() if not passed]
        if failed:
            raise ValueError(
                f"Formal config violates manifest controls: "
                f"{config_relative}: {failed}"
            )

        log_dir_value = logging.get("log_dir")
        data_root_value = config.get("data_root")
        if not isinstance(log_dir_value, str) or not log_dir_value:
            raise ValueError(f"Missing logging.log_dir: {config_relative}")
        if not isinstance(data_root_value, str) or not data_root_value:
            raise ValueError(f"Missing data_root: {config_relative}")
        log_dir = Path(log_dir_value)
        if not log_dir.is_absolute():
            raise ValueError(f"Formal log directory must be absolute: {log_dir}")
        log_dirs.append(log_dir)
        data_roots.add(data_root_value)
        cells.append(
            {
                "cell": str(entry["cell"]),
                "config_relative": str(config_relative.as_posix()),
                "config_path": config_path,
                "log_dir": log_dir,
            }
        )

    if len(data_roots) != 1:
        raise ValueError("All formal cells must use the same data_root")
    if len(set(log_dirs)) != len(log_dirs):
        raise ValueError("Formal cells must use distinct log directories")
    matrix_root = Path(os.path.commonpath([str(path) for path in log_dirs]))
    if any(path.parent != matrix_root for path in log_dirs):
        raise ValueError("Formal log directories must share one direct parent")
    return cells, matrix_root


def _require_empty_outputs(cells: list[dict[str, Any]]) -> None:
    for cell in cells:
        log_dir = cell["log_dir"]
        if log_dir.exists() and any(log_dir.iterdir()):
            raise FileExistsError(
                f"Formal result directory is not empty: {log_dir}"
            )


def _environment_snapshot() -> dict[str, Any]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name)] = distribution.version
    return {
        "captured_at": _now(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _copy_inputs(
    evidence_dir: Path,
    cells: list[dict[str, Any]],
) -> list[dict[str, str]]:
    input_dir = evidence_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    paths = [MANIFEST_PATH]
    paths.extend(cell["config_path"] for cell in cells)
    records = []
    for path in paths:
        destination = input_dir / path.name
        shutil.copy2(path, destination)
        records.append(
            {
                "source": str(path),
                "copy": str(destination),
                "sha256": _sha256(path),
            }
        )
    return records


def _run_command(command: list[str], log_handle: TextIO) -> None:
    _log(log_handle, f"COMMAND: {json.dumps(command, ensure_ascii=False)}")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _audit_cell(
    cell: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    checkpoint_dir = cell["log_dir"] / "checkpoints"
    artifact_paths = {
        "final_model": checkpoint_dir / "final_model.pth",
        "latest_checkpoint": checkpoint_dir / "latest_checkpoint.pth",
        "training_history": checkpoint_dir / "training_history.json",
        "run_metadata": checkpoint_dir / "run_metadata.json",
    }
    for name, path in artifact_paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing {name} evidence: {path}")

    history = _read_json(artifact_paths["training_history"])
    metadata = _read_json(artifact_paths["run_metadata"])
    rounds = history.get("rounds")
    final_metrics = history.get("final_val_metrics")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"Training history has no completed round: {cell['cell']}")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"Training history has no final metrics: {cell['cell']}")
    if metadata.get("git_commit") != git_commit:
        raise ValueError(f"Run metadata Git commit mismatch: {cell['cell']}")

    artifacts = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    artifacts["training_history"]["sha256"] = _sha256(
        artifact_paths["training_history"]
    )
    artifacts["run_metadata"]["sha256"] = _sha256(
        artifact_paths["run_metadata"]
    )
    return {
        "completed_round": int(rounds[-1]),
        "final_val_metrics": final_metrics,
        "artifacts": artifacts,
    }


def run() -> int:
    cells, matrix_root = _matrix_definition()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    evidence_dir = matrix_root / "_supervisor_logs" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    supervisor_path = evidence_dir / "supervisor.log"
    result_path = evidence_dir / "runner_result.json"
    installed_packages_path = evidence_dir / "installed_packages.json"

    result: dict[str, Any] = {
        "task": "formal_2x2_federated_brats_segmentation",
        "status": "RUNNING",
        "started_at": _now(),
        "evidence_dir": str(evidence_dir),
        "manifest_path": str(MANIFEST_PATH),
        "cells": [],
    }
    with supervisor_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        try:
            git_commit = _git_output("rev-parse", "HEAD")
            git_status = _git_output("status", "--porcelain")
            if git_status:
                raise RuntimeError("Git worktree must be clean before formal training")
            _require_empty_outputs(cells)

            disk_usage = shutil.disk_usage(matrix_root)
            environment = _environment_snapshot()
            _write_json(installed_packages_path, environment)
            result.update(
                {
                    "git_commit": git_commit,
                    "git_log": _git_output("show", "-s", "--format=fuller", "HEAD"),
                    "disk_free_bytes_at_start": disk_usage.free,
                    "environment_path": str(installed_packages_path),
                    "inputs": _copy_inputs(evidence_dir, cells),
                }
            )
            _write_json(result_path, result)
            _log(log_handle, f"EVIDENCE_DIR: {evidence_dir}")
            _log(log_handle, f"GIT_COMMIT: {git_commit}")

            test_command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *CONTRACT_TESTS,
            ]
            _run_command(test_command, log_handle)

            for cell in cells:
                cell_result: dict[str, Any] = {
                    "cell": cell["cell"],
                    "config": cell["config_relative"],
                    "log_dir": str(cell["log_dir"]),
                    "status": "RUNNING",
                    "started_at": _now(),
                }
                result["cells"].append(cell_result)
                _write_json(result_path, result)
                _log(log_handle, f"START CELL: {cell['cell']}")

                _run_command(
                    [
                        sys.executable,
                        "scripts/server_preflight.py",
                        "--config",
                        cell["config_relative"],
                    ],
                    log_handle,
                )
                _run_command(
                    [
                        sys.executable,
                        "main.py",
                        "--config",
                        cell["config_relative"],
                    ],
                    log_handle,
                )
                cell_result.update(
                    {
                        "status": "COMPLETE",
                        "finished_at": _now(),
                        "evidence": _audit_cell(cell, git_commit),
                    }
                )
                _write_json(result_path, result)
                _log(log_handle, f"COMPLETE CELL: {cell['cell']}")

            result.update(
                {
                    "status": "COMPLETE",
                    "finished_at": _now(),
                }
            )
            _write_json(result_path, result)
            _log(log_handle, "ALL FOUR CELLS COMPLETE")
            return 0
        except Exception as error:
            result.update(
                {
                    "status": "FAILED",
                    "finished_at": _now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_json(result_path, result)
            _log(log_handle, f"FAILED: {type(error).__name__}: {error}")
            return 1


if __name__ == "__main__":
    raise SystemExit(run())
