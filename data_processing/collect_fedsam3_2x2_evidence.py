"""Collect lightweight formal 2x2 evidence without extracting checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "fedsam3_2x2_final_evidence_package.json"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_text_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def target_directory(output_dir: Path, cell: Mapping[str, Any]) -> Path:
    cell_name = str(cell["cell"]).lower().replace("-", "_")
    return output_dir / f"seed_{cell['seed']}" / cell_name


def read_selected_members(
    archive_path: Path,
    archive_prefix: str,
    artifact_members: Mapping[str, str],
    optional_artifact_members: Mapping[str, str] | None = None,
) -> dict[str, bytes]:
    required = {
        f"{archive_prefix.rstrip('/')}/{relative}": logical_name
        for logical_name, relative in artifact_members.items()
    }
    optional_artifact_members = optional_artifact_members or {}
    optional = {
        f"{archive_prefix.rstrip('/')}/{relative}": logical_name
        for logical_name, relative in optional_artifact_members.items()
    }
    expected = {**required, **optional}
    found: dict[str, bytes] = {}
    checkpoint_prefix = f"{archive_prefix.rstrip('/')}/checkpoints/"
    crossed_checkpoint_payloads = False

    with tarfile.open(archive_path, mode="r:gz") as archive:
        while (member := archive.next()) is not None:
            required_names = set(required.values())
            if member.name.startswith(checkpoint_prefix):
                if required_names.issubset(found):
                    break
                crossed_checkpoint_payloads = True
                continue
            logical_name = expected.get(member.name)
            if logical_name is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            found[logical_name] = source.read()
            if required_names.issubset(found) and (
                set(optional.values()).issubset(found)
                or crossed_checkpoint_payloads
            ):
                break

    missing = sorted(set(artifact_members) - set(found))
    if missing:
        raise FileNotFoundError(
            f"Missing evidence members in {archive_path.name}: {missing}"
        )
    found["_crossed_checkpoint_payloads"] = (
        b"true" if crossed_checkpoint_payloads else b"false"
    )
    return found


def _artifact_filename(
    logical_name: str,
    required: Mapping[str, str],
    optional: Mapping[str, str],
) -> str:
    if logical_name == "config_yaml":
        return "config.yaml"
    if logical_name == "experiment_manifest_json":
        return "experiment_manifest.json"
    return Path({**required, **optional}[logical_name]).name


def _existing_artifacts(
    destination: Path,
    required: Mapping[str, str],
    optional: Mapping[str, str],
) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for logical_name in required:
        path = destination / _artifact_filename(logical_name, required, optional)
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete existing evidence directory: {path}")
        artifacts[logical_name] = path.read_bytes()
    for logical_name in optional:
        path = destination / _artifact_filename(logical_name, required, optional)
        if path.is_file():
            artifacts[logical_name] = path.read_bytes()
    for logical_name in ("config_yaml", "experiment_manifest_json"):
        path = destination / _artifact_filename(logical_name, required, optional)
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete existing evidence directory: {path}")
        artifacts[logical_name] = path.read_bytes()
    return artifacts


def _validate_matrix(config: Mapping[str, Any]) -> None:
    expected = config["expected"]
    cells = config["cells"]
    observed = [(int(cell["seed"]), str(cell["cell"])) for cell in cells]
    required = [
        (int(seed), str(cell))
        for seed in expected["seeds"]
        for cell in expected["cells"]
    ]
    if sorted(observed) != sorted(required):
        raise ValueError("Evidence configuration must declare each seed/cell once")


def collect(
    config_path: Path,
    output_override: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    _validate_matrix(config)
    archive_root = Path(config["archive_root"])
    output_dir = output_override or PROJECT_ROOT / config["output_dir"]
    output_dir = output_dir.resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Evidence output already exists: {output_dir}")
    if output_dir.exists() and (output_dir / "collection_manifest.json").exists():
        raise FileExistsError(f"Completed evidence output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    required_artifacts = config["archive_artifacts"]
    optional_artifacts = config.get("optional_archive_artifacts", {})
    for cell in config["cells"]:
        archive_path = archive_root / cell["archive"]
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing source archive: {archive_path}")
        config_source = (PROJECT_ROOT / cell["config"]).resolve()
        manifest_source = (PROJECT_ROOT / cell["manifest"]).resolve()
        config_source.relative_to(PROJECT_ROOT)
        manifest_source.relative_to(PROJECT_ROOT)

        destination = target_directory(output_dir, cell)
        reused_existing = destination.exists() and any(destination.iterdir())
        if reused_existing:
            if not resume:
                raise FileExistsError(f"Evidence cell already exists: {destination}")
            artifacts = _existing_artifacts(
                destination, required_artifacts, optional_artifacts
            )
            crossed_checkpoint_payloads = False
        else:
            destination.mkdir(parents=True, exist_ok=True)
            artifacts = read_selected_members(
                archive_path,
                str(cell["archive_prefix"]),
                required_artifacts,
                optional_artifacts,
            )
            crossed_checkpoint_payloads = (
                artifacts.pop("_crossed_checkpoint_payloads") == b"true"
            )
            artifacts["config_yaml"] = normalized_text_bytes(config_source)
            artifacts["experiment_manifest_json"] = normalized_text_bytes(
                manifest_source
            )

        artifact_records: dict[str, Any] = {}
        for logical_name, payload in artifacts.items():
            filename = _artifact_filename(
                logical_name, required_artifacts, optional_artifacts
            )
            output_path = destination / filename
            if not output_path.exists():
                output_path.write_bytes(payload)
            artifact_records[logical_name] = {
                "path": str(output_path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }

        final_csv_hash = artifact_records["final_metrics_csv"]["sha256"]
        records.append(
            {
                **cell,
                "archive_path": str(archive_path),
                "artifact_directory": str(destination.relative_to(output_dir)).replace(
                    "\\", "/"
                ),
                "missing_optional_artifacts": sorted(
                    set(optional_artifacts) - set(artifacts)
                ),
                "crossed_checkpoint_payloads": crossed_checkpoint_payloads,
                "final_metrics_csv_sha256": final_csv_hash,
                "artifacts": artifact_records,
            }
        )

    result = {
        "schema_version": 1,
        "status": "COLLECTED",
        "source_config": str(config_path),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "checkpoint_selection": config["checkpoint_selection"],
        "records": records,
    }
    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect lightweight evidence from the 12 formal archives"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = collect(args.config, args.output_dir, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
