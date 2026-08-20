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
) -> dict[str, bytes]:
    expected = {
        f"{archive_prefix.rstrip('/')}/{relative}": logical_name
        for logical_name, relative in artifact_members.items()
    }
    found: dict[str, bytes] = {}
    checkpoint_prefix = f"{archive_prefix.rstrip('/')}/checkpoints/"

    with tarfile.open(archive_path, mode="r:gz") as archive:
        while (member := archive.next()) is not None:
            if member.name.startswith(checkpoint_prefix) and len(found) != len(expected):
                missing = sorted(set(expected.values()) - set(found))
                raise RuntimeError(
                    "Required lightweight evidence was not found before checkpoint "
                    f"payloads in {archive_path.name}: {missing}"
                )
            logical_name = expected.get(member.name)
            if logical_name is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            found[logical_name] = source.read()
            if len(found) == len(expected):
                break

    missing = sorted(set(artifact_members) - set(found))
    if missing:
        raise FileNotFoundError(
            f"Missing evidence members in {archive_path.name}: {missing}"
        )
    return found


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


def collect(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    _validate_matrix(config)
    archive_root = Path(config["archive_root"])
    output_dir = output_override or PROJECT_ROOT / config["output_dir"]
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evidence output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    for cell in config["cells"]:
        archive_path = archive_root / cell["archive"]
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing source archive: {archive_path}")
        config_source = (PROJECT_ROOT / cell["config"]).resolve()
        manifest_source = (PROJECT_ROOT / cell["manifest"]).resolve()
        config_source.relative_to(PROJECT_ROOT)
        manifest_source.relative_to(PROJECT_ROOT)

        destination = target_directory(output_dir, cell)
        destination.mkdir(parents=True, exist_ok=False)
        artifacts = read_selected_members(
            archive_path,
            str(cell["archive_prefix"]),
            config["archive_artifacts"],
        )
        config_payload = normalized_text_bytes(config_source)
        manifest_payload = normalized_text_bytes(manifest_source)
        artifacts["config_yaml"] = config_payload
        artifacts["experiment_manifest_json"] = manifest_payload

        artifact_records: dict[str, Any] = {}
        for logical_name, payload in artifacts.items():
            if logical_name == "config_yaml":
                filename = "config.yaml"
            elif logical_name == "experiment_manifest_json":
                filename = "experiment_manifest.json"
            else:
                filename = Path(config["archive_artifacts"][logical_name]).name
            output_path = destination / filename
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = collect(args.config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
