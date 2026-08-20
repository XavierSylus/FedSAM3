#!/usr/bin/env python3
"""Build and validate the lightweight FedSAM3 P0 evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fedsam3_p0_evidence_package.json"
METRIC_FIELDS = ("dice", "iou", "hd95_mm")


class HashingReader:
    def __init__(self, raw: io.BufferedReader) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        if data:
            self.hasher.update(data)
        return data

    def tell(self) -> int:
        return self.raw.tell()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    return config, raw


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_config(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 18:
        errors.append("experiments must contain exactly 18 cells")
        return errors

    keys: set[tuple[str, int, str, str]] = set()
    archive_root = Path(str(config.get("archive_root", "")))
    for experiment in experiments:
        key = (
            str(experiment.get("protocol")),
            int(experiment.get("seed", -1)),
            str(experiment.get("routing")),
            str(experiment.get("aggregation")),
        )
        if key in keys:
            errors.append(f"duplicate experiment cell: {key}")
        keys.add(key)
        archive = archive_root / str(experiment.get("archive", ""))
        if not archive.is_file():
            errors.append(f"missing source archive: {archive}")
        for field in ("member_root", "config", "manifest", "cell"):
            if not experiment.get(field):
                errors.append(f"missing {field} for {key}")

    main_keys = {key for key in keys if key[0] == "main_2x2"}
    ratio_keys = {key for key in keys if key[0] == "ratio_2of3"}
    if len(main_keys) != 12:
        errors.append(f"main_2x2 expected 12 cells, found {len(main_keys)}")
    if len(ratio_keys) != 6:
        errors.append(f"ratio_2of3 expected 6 cells, found {len(ratio_keys)}")

    repository_files = config.get("repository_files", [])
    for relative in repository_files:
        path = REPO_ROOT / str(relative)
        if not path.is_file():
            errors.append(f"missing repository evidence file: {relative}")
    return errors


def csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")


def parse_final_metrics(data: bytes) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if len(rows) != 1:
        raise ValueError(f"final_metrics.csv expected one row, found {len(rows)}")
    row = rows[0]
    hd95 = row.get("hd95_mm", row.get("hd95"))
    if hd95 in (None, ""):
        raise ValueError("final_metrics.csv lacks hd95 or hd95_mm")
    return {
        "dice": float(row["dice"]),
        "iou": float(row["iou"]),
        "hd95_mm": float(hd95),
    }


def archive_destination(experiment: Mapping[str, Any], relative: str) -> str:
    return str(
        PurePosixPath("raw")
        / str(experiment["protocol"])
        / f"seed_{experiment['seed']}"
        / str(experiment["cell"])
        / relative
    )


def process_archive(
    zf: zipfile.ZipFile,
    archive_path: Path,
    experiment: Mapping[str, Any],
    selected_members: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    started = time.monotonic()
    member_root = str(experiment["member_root"]).rstrip("/")
    prefix = member_root + "/"
    extracted: list[dict[str, Any]] = []
    captured: dict[str, bytes] = {}

    with archive_path.open("rb") as raw:
        reader = HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = member.name[len(prefix) :]
                if relative not in selected_members:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                data = source.read()
                destination = archive_destination(experiment, relative)
                zf.writestr(destination, data)
                captured[relative] = data
                extracted.append(
                    {
                        "zip_path": destination,
                        "source_archive": archive_path.name,
                        "source_member": member.name,
                        "bytes": len(data),
                        "sha256": sha256_bytes(data),
                    }
                )
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            pass
        archive_sha256 = reader.hasher.hexdigest()

    archive_record = {
        "protocol": experiment["protocol"],
        "seed": experiment["seed"],
        "routing": experiment["routing"],
        "aggregation": experiment["aggregation"],
        "cell": experiment["cell"],
        "path": str(archive_path),
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return archive_record, extracted, captured


def result_rows(
    experiments: list[Mapping[str, Any]],
    captured_by_key: Mapping[tuple[str, int, str, str], Mapping[str, bytes]],
    archive_records: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for experiment in experiments:
        key = (
            str(experiment["protocol"]),
            int(experiment["seed"]),
            str(experiment["routing"]),
            str(experiment["aggregation"]),
        )
        captured = captured_by_key[key]
        final_data = captured.get("formal_verification/final_metrics.csv")
        verification_data = captured.get(
            "formal_verification/formal_verification.json"
        )
        if final_data is None:
            issues.append(f"missing final_metrics.csv for {key}")
            continue
        metrics = parse_final_metrics(final_data)
        verification: dict[str, Any] = {}
        if verification_data is not None:
            verification = json.loads(verification_data.decode("utf-8"))
        else:
            issues.append(f"missing formal_verification.json for {key}")
        record = {
            "protocol": key[0],
            "seed": key[1],
            "routing": key[2],
            "aggregation": key[3],
            "config": experiment["config"],
            "manifest": experiment["manifest"],
            "source_archive": archive_records[key]["archive"],
            "source_archive_sha256": archive_records[key]["sha256"],
            "formal_status": verification.get("status", "UNKNOWN"),
            "metrics_match": verification.get("metrics_match", "UNKNOWN"),
            "training_git_commit": verification.get("training_git_commit", ""),
            "verification_git_commit": verification.get(
                "verification_git_commit", ""
            ),
            "data_manifest_sha256": verification.get("data_manifest_sha256", ""),
            "checkpoint_rule": verification.get("evaluation_contract", {}).get(
                "checkpoint", ""
            ),
            "checkpoint_round": verification.get("evaluation_contract", {}).get(
                "round", ""
            ),
            "final_model_sha256": verification.get("checkpoint_audit", {}).get(
                "final_model_sha256", ""
            ),
            **metrics,
        }
        rows.append(record)
        if record["formal_status"] != "PASS":
            issues.append(f"formal verification not PASS for {key}")
        if record["metrics_match"] is not True:
            issues.append(f"formal metrics mismatch for {key}")
    return rows, issues


def paired_rows(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["protocol"]),
            int(row["seed"]),
            str(row["aggregation"]),
            str(row["routing"]),
        ): row
        for row in results
    }
    pairs: list[dict[str, Any]] = []
    base_keys = sorted({key[:3] for key in indexed})
    for protocol, seed, aggregation in base_keys:
        u = indexed.get((protocol, seed, aggregation, "U"))
        r = indexed.get((protocol, seed, aggregation, "R"))
        if u is None or r is None:
            continue
        pairs.append(
            {
                "protocol": protocol,
                "seed": seed,
                "aggregation": aggregation,
                "U_dice": u["dice"],
                "R_dice": r["dice"],
                "R_minus_U_dice": r["dice"] - u["dice"],
                "U_minus_R_dice": u["dice"] - r["dice"],
                "U_iou": u["iou"],
                "R_iou": r["iou"],
                "R_minus_U_iou": r["iou"] - u["iou"],
                "U_minus_R_iou": u["iou"] - r["iou"],
                "U_hd95_mm": u["hd95_mm"],
                "R_hd95_mm": r["hd95_mm"],
                "R_minus_U_hd95_mm": r["hd95_mm"] - u["hd95_mm"],
                "U_minus_R_hd95_mm": u["hd95_mm"] - r["hd95_mm"],
            }
        )
    return pairs


def summary_rows(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in results:
        key = (
            str(row["protocol"]),
            str(row["aggregation"]),
            str(row["routing"]),
        )
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "protocol": key[0],
            "aggregation": key[1],
            "routing": key[2],
            "n": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
        }
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_sample_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(record)
    return summary


def audit_summary_rows(
    config: Mapping[str, Any],
    captured_by_key: Mapping[tuple[str, int, str, str], Mapping[str, bytes]],
) -> tuple[list[dict[str, Any]], list[str]]:
    audit_config = config["representative_audit"]
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for routing in audit_config["routing_modes"]:
        key = (
            str(audit_config["protocol"]),
            int(audit_config["seed"]),
            str(routing),
            str(audit_config["aggregation"]),
        )
        data = captured_by_key.get(key, {}).get("parameter_group_diagnostics.jsonl")
        if data is None:
            issues.append(f"representative diagnostics missing for {key}")
            continue
        matching: dict[str, Any] | None = None
        for line in data.decode("utf-8").splitlines():
            parsed = json.loads(line)
            if int(parsed.get("round", -1)) == int(audit_config["round"]):
                matching = parsed
                break
        if matching is None:
            issues.append(f"representative round missing for {key}")
            continue

        client_drift = matching.get("client_drift", {})
        for group_name, metrics in sorted(matching.get("global_drift", {}).items()):
            participation = metrics.get("aggregation_participation", {})
            eligible_sets: Counter[str] = Counter()
            weight_patterns: Counter[str] = Counter()
            empty_count = 0
            for entry in participation.values():
                eligible = entry.get("eligible_client_ids", [])
                weights = entry.get("normalized_weights", {})
                eligible_sets["+".join(eligible) if eligible else "EMPTY"] += 1
                weight_patterns[json.dumps(weights, sort_keys=True)] += 1
                if not eligible:
                    empty_count += 1
            sample_weights = {
                client_id: group_metrics[group_name]["sample_weight"]
                for client_id, group_metrics in client_drift.items()
                if group_name in group_metrics
                and "sample_weight" in group_metrics[group_name]
            }
            rows.append(
                {
                    "protocol": key[0],
                    "seed": key[1],
                    "routing": key[2],
                    "aggregation": key[3],
                    "round": audit_config["round"],
                    "parameter_group": group_name,
                    "parameter_count": len(participation),
                    "empty_eligible_parameter_count": empty_count,
                    "eligible_set_counts": json.dumps(
                        eligible_sets, ensure_ascii=False, sort_keys=True
                    ),
                    "normalized_weight_pattern_counts": json.dumps(
                        weight_patterns, ensure_ascii=False, sort_keys=True
                    ),
                    "client_group_sample_weights": json.dumps(
                        sample_weights, ensure_ascii=False, sort_keys=True
                    ),
                    "actual_uploaders_per_parameter_persisted": False,
                    "zero_update_clients_per_parameter_persisted": False,
                    "raw_sample_weights_per_parameter_persisted": False,
                }
            )
    return rows, issues


def code_location_rows() -> list[dict[str, Any]]:
    targets = [
        ("src/server.py", "uploaded_client_ids =", "actual uploader collection"),
        ("src/server.py", "eligible_client_ids =", "U/R eligible client construction"),
        ("src/server.py", "zero_update_client_ids", "zero-update audit"),
        ("src/server.py", '"sample_weights":', "raw sample-weight audit"),
        ("src/server.py", '"normalized_weights":', "normalized-weight audit"),
        ("src/server.py", '"empty_eligible":', "empty eligible-set audit"),
        (
            "src/update_diagnostics.py",
            '"aggregation_participation"',
            "persisted diagnostics projection",
        ),
        (
            "src/federated_trainer.py",
            "_persist_parameter_group_diagnostics",
            "diagnostics persistence",
        ),
        (
            "scripts/server_verify_formal_cell.py",
            '"evaluation_contract"',
            "formal evaluation and checkpoint contract",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for relative, needle, purpose in targets:
        lines = (REPO_ROOT / relative).read_text(encoding="utf-8").splitlines()
        matches = [index for index, line in enumerate(lines, start=1) if needle in line]
        rows.append(
            {
                "file": relative,
                "line_numbers": ";".join(str(value) for value in matches),
                "purpose": purpose,
                "search_text": needle,
            }
        )
    return rows


def run_command_rows(experiments: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cell_names = {
        "u_fedavg": "U-FedAvg",
        "r_fedavg": "R-FedAvg",
        "u_fedprox": "U-FedProx",
        "r_fedprox": "R-FedProx",
    }
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        if experiment["protocol"] == "main_2x2":
            training_command = (
                "python scripts/server_run_2x2_matrix.py "
                f"--manifest {experiment['manifest']} "
                f"--cell {cell_names[str(experiment['cell'])]}"
            )
        else:
            training_command = f"python main.py --config {experiment['config']}"
        rows.append(
            {
                "protocol": experiment["protocol"],
                "seed": experiment["seed"],
                "cell": experiment["cell"],
                "config": experiment["config"],
                "training_command": training_command,
                "verification_command": (
                    "python scripts/server_verify_formal_cell.py "
                    f"--config {experiment['config']}"
                ),
            }
        )
    return rows


def readme_text(
    generated_at: str,
    source_commit: str,
    results: list[Mapping[str, Any]],
    issues: list[str],
) -> str:
    overall = "PARTIAL" if issues else "PASS"
    issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- 无。"
    return f"""# FedSAM3-Hetero P0证据包

生成时间：{generated_at}

仓库commit：`{source_commit}`

总体状态：`{overall}`

## 覆盖范围

- P0-1：包含正式2×2的12个实验单元，以及2/3敏感性检查的6个实验单元。每个单元保留轻量训练日志、逐轮诊断、正式复核、最终指标和完整原始归档SHA-256。
- P0-2：部分满足。训练产生的持久化诊断包含逐参数eligible clients、normalized weights和empty eligible事件，但没有持久化逐参数actual uploaders、zero-update clients和raw sample weights。具体边界见 `audit/P0_2_EVIDENCE_GAP.md`。
- P0-3：包含 `protocol/protocol_compare.csv`，明确主实验与2/3实验的客户端组成、参与率、seed、loss、routing、分母规则和评价口径。

## 主要入口

- `manifest/package_manifest.json`：机器可读总清单与验收状态。
- `manifest/source_archive_inventory.csv`：18个完整原始归档的位置、大小和SHA-256。
- `manifest/extracted_file_inventory.csv`：ZIP内所有提取证据的来源成员和SHA-256。
- `results/final_metrics_all_cells.csv`：18个单元的最终指标与正式复核信息。
- `results/paired_seed_deltas.csv`：同seed U/R配对差异，同时给出R-U和U-R，避免符号歧义。
- `results/summary_mean_sample_sd.csv`：均值与样本标准差。
- `protocol/run_commands.csv`：训练与正式复核命令。
- `audit/representative_seed3407_aggregation_audit_summary.csv`：代表seed的训练产出审计摘要。
- `repository_snapshot/`：配置、manifest、评价脚本、结果生成脚本和关键聚合代码快照。

## checkpoint说明

本轻量包不重复嵌入约168 GB的checkpoint归档。`final_metrics_all_cells.csv`记录正式评价采用的checkpoint、轮次及模型哈希；完整checkpoint保留在 `source_archive_inventory.csv` 指向的原始 `.tar.gz` 中。

## 数值纪律

- 结果来自已完成归档；本次未训练、未重新推理。
- 统计单位为三个固定seed，报告mean ± sample SD。
- 不提供p值或“显著”结论。
- 主实验与2/3实验只在各自协议内解释。

## 自动核验问题

{issue_lines}

结果记录数：{len(results)}。
"""


def gap_text() -> str:
    return """# P0-2聚合审计证据边界

## 已有训练产出能够证明的内容

- 代表seed每轮、每参数的eligible client集合。
- 每参数normalized aggregation weights。
- eligible集合为空的参数事件。
- 参数组级客户端sample weight、更新幅度、冲突与服务器漂移。
- 正式服务器代码中完整聚合审计字段的实现位置。

## 现有归档没有持久化的字段

- 每参数actual uploaders。
- 每参数zero-update clients。
- 每参数raw sample weights字典。

这些字段在 `src/server.py` 的运行时 `_last_aggregation_audit` 中构造，但写入 `parameter_group_diagnostics.jsonl` 时只投影保存了eligible clients和normalized weights。现有归档无法无损恢复数值为零但仍实际上传的参数，因此不能用配置或代码静态推断冒充实际运行证据。

## 验收结论

P0-2只能标记为“部分满足”。若导师要求完整逐参数运行审计，必须在服务器端增加不丢字段的正式持久化，再对至少一个代表seed重新运行；该工作不属于本次只读打包范围。
"""


def build_package(config_path: Path) -> Path:
    config, config_raw = load_config(config_path)
    errors = validate_config(config)
    if errors:
        raise SystemExit("Invalid package config:\n- " + "\n- ".join(errors))

    output_path = REPO_ROOT / str(config["output_path"])
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing package: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_commit = git_head()
    generated_at = datetime.now(timezone.utc).isoformat()
    experiments = list(config["experiments"])
    selected_members = set(config["selected_archive_members"])
    archive_root = Path(str(config["archive_root"]))
    archive_inventory: list[dict[str, Any]] = []
    extracted_inventory: list[dict[str, Any]] = []
    captured_by_key: dict[tuple[str, int, str, str], dict[str, bytes]] = {}
    archive_by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    print(f"Building {output_path}", flush=True)
    with zipfile.ZipFile(
        output_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for index, experiment in enumerate(experiments, start=1):
            key = (
                str(experiment["protocol"]),
                int(experiment["seed"]),
                str(experiment["routing"]),
                str(experiment["aggregation"]),
            )
            archive_path = archive_root / str(experiment["archive"])
            print(
                f"[{index:02d}/{len(experiments)}] hashing and extracting {archive_path.name}",
                flush=True,
            )
            archive_record, extracted, captured = process_archive(
                zf, archive_path, experiment, selected_members
            )
            archive_inventory.append(archive_record)
            archive_by_key[key] = archive_record
            extracted_inventory.extend(extracted)
            captured_by_key[key] = captured
            print(
                f"[{index:02d}/{len(experiments)}] done sha256={archive_record['sha256'][:16]} "
                f"files={len(extracted)} elapsed={archive_record['elapsed_seconds']}s",
                flush=True,
            )

        for relative in config["repository_files"]:
            source = REPO_ROOT / str(relative)
            destination = str(PurePosixPath("repository_snapshot") / str(relative))
            data = source.read_bytes()
            zf.writestr(destination, data)
            extracted_inventory.append(
                {
                    "zip_path": destination,
                    "source_archive": "repository",
                    "source_member": str(relative),
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )

        builder_relative = "data_processing/build_fedsam3_p0_evidence_package.py"
        builder_data = (REPO_ROOT / builder_relative).read_bytes()
        builder_destination = str(
            PurePosixPath("repository_snapshot") / builder_relative
        )
        zf.writestr(builder_destination, builder_data)
        extracted_inventory.append(
            {
                "zip_path": builder_destination,
                "source_archive": "repository",
                "source_member": builder_relative,
                "bytes": len(builder_data),
                "sha256": sha256_bytes(builder_data),
            }
        )
        config_destination = "repository_snapshot/configs/fedsam3_p0_evidence_package.json"
        zf.writestr(config_destination, config_raw)
        extracted_inventory.append(
            {
                "zip_path": config_destination,
                "source_archive": "repository",
                "source_member": str(config_path.relative_to(REPO_ROOT)),
                "bytes": len(config_raw),
                "sha256": sha256_bytes(config_raw),
            }
        )

        results, issues = result_rows(
            experiments, captured_by_key, archive_by_key
        )
        pairs = paired_rows(results)
        summaries = summary_rows(results)
        audit_rows, audit_issues = audit_summary_rows(config, captured_by_key)
        issues.extend(audit_issues)

        expected_counts = {"main_2x2": 12, "ratio_2of3": 6}
        for protocol, expected in expected_counts.items():
            actual = sum(1 for row in results if row["protocol"] == protocol)
            if actual != expected:
                issues.append(
                    f"{protocol} final result count expected {expected}, found {actual}"
                )
        data_hashes = {
            row["data_manifest_sha256"]
            for row in results
            if row["data_manifest_sha256"]
        }
        if len(data_hashes) != 1:
            issues.append(
                f"expected one shared test data manifest hash, found {len(data_hashes)}"
            )

        archive_csv = csv_bytes(
            archive_inventory,
            [
                "protocol",
                "seed",
                "routing",
                "aggregation",
                "cell",
                "path",
                "archive",
                "bytes",
                "sha256",
                "elapsed_seconds",
            ],
        )
        extracted_csv = csv_bytes(
            sorted(extracted_inventory, key=lambda row: row["zip_path"]),
            ["zip_path", "source_archive", "source_member", "bytes", "sha256"],
        )
        result_fields = [
            "protocol",
            "seed",
            "routing",
            "aggregation",
            "config",
            "manifest",
            "source_archive",
            "source_archive_sha256",
            "formal_status",
            "metrics_match",
            "training_git_commit",
            "verification_git_commit",
            "data_manifest_sha256",
            "checkpoint_rule",
            "checkpoint_round",
            "final_model_sha256",
            "dice",
            "iou",
            "hd95_mm",
        ]
        pair_fields = list(pairs[0]) if pairs else []
        summary_fields = list(summaries[0]) if summaries else []
        audit_fields = list(audit_rows[0]) if audit_rows else []
        protocol_fields = list(config["protocols"][0])
        command_rows = run_command_rows(experiments)
        command_fields = list(command_rows[0])
        code_rows = code_location_rows()
        code_fields = list(code_rows[0])

        generated_files = {
            "manifest/source_archive_inventory.csv": archive_csv,
            "manifest/extracted_file_inventory.csv": extracted_csv,
            "results/final_metrics_all_cells.csv": csv_bytes(results, result_fields),
            "results/paired_seed_deltas.csv": csv_bytes(pairs, pair_fields),
            "results/summary_mean_sample_sd.csv": csv_bytes(
                summaries, summary_fields
            ),
            "protocol/protocol_compare.csv": csv_bytes(
                config["protocols"], protocol_fields
            ),
            "protocol/run_commands.csv": csv_bytes(command_rows, command_fields),
            "audit/representative_seed3407_aggregation_audit_summary.csv": csv_bytes(
                audit_rows, audit_fields
            ),
            "audit/code_locations.csv": csv_bytes(code_rows, code_fields),
            "audit/P0_2_EVIDENCE_GAP.md": gap_text().encode("utf-8"),
        }
        for destination, data in generated_files.items():
            zf.writestr(destination, data)

        manifest = {
            "schema_version": 1,
            "package_name": config["package_name"],
            "generated_at_utc": generated_at,
            "source_commit": source_commit,
            "builder": builder_relative,
            "builder_sha256": sha256_bytes(builder_data),
            "config": str(config_path.relative_to(REPO_ROOT)),
            "config_sha256": sha256_bytes(config_raw),
            "training_or_inference_performed": False,
            "p0_status": {
                "P0-1": "PASS" if len(results) == 18 and not any("final" in issue for issue in issues) else "PARTIAL",
                "P0-2": "PARTIAL",
                "P0-3": "PASS",
                "overall": "PARTIAL",
            },
            "evidence_boundary": (
                "P0-2 remains partial because the archived runtime projection does not "
                "persist actual_uploaders, zero_update_clients, or raw sample weights per parameter."
            ),
            "source_archives": archive_inventory,
            "extracted_files": extracted_inventory,
            "results": results,
            "paired_seed_deltas": pairs,
            "summary_mean_sample_sd": summaries,
            "validation_issues": issues,
        }
        zf.writestr(
            "manifest/package_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        zf.writestr(
            "README.md",
            readme_text(generated_at, source_commit, results, issues).encode("utf-8"),
        )

    package_sha256 = sha256_file(output_path)
    print(f"Package complete: {output_path}", flush=True)
    print(f"Package bytes: {output_path.stat().st_size}", flush=True)
    print(f"Package SHA-256: {package_sha256}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, _ = load_config(config_path)
    errors = validate_config(config)
    if args.check_config:
        if errors:
            print("CONFIG_INVALID")
            for error in errors:
                print(f"- {error}")
            return 1
        print("CONFIG_VALID")
        print(f"experiments={len(config['experiments'])}")
        print(f"output={config['output_path']}")
        return 0
    if errors:
        print("CONFIG_INVALID", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    build_package(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
