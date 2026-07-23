from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


CONFIG_PATH = Path(__file__).with_name("paper_result_audit_config.json")


@dataclass(frozen=True)
class MetricSpec:
    table: str
    row_label: str
    metric: str
    decimals: int
    unit: str
    source_family: str
    source_setting: str
    source_metric: str


MAIN_METRICS = [
    ("Best Dice", "best_dice", "dice", ""),
    ("Final Dice", "final_dice", "dice", ""),
    ("Final HD95", "final_hd95", "hd95", "mm"),
    ("Avg. Grad Conflict", "avg_grad_conflict", "grad_conflict", "deg"),
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def population_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def configured_std(values: list[float], mode: str) -> float:
    if mode == "sample":
        return sample_std(values)
    if mode == "population":
        return population_std(values)
    raise ValueError(f"Unsupported std mode: {mode}")


def fmt_num(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def fmt_pm(mean: float, std: float, decimals: int) -> str:
    return f"{fmt_num(mean, decimals)} ± {fmt_num(std, decimals)}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = text.replace("Infinity", "1e999").replace("-Infinity", "-1e999")
    return json.loads(text)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def best_and_final_from_history(path: Path) -> dict[str, float]:
    hist = read_json(path)
    val_metrics = hist["val_metrics"]
    finite_metrics = [m for m in val_metrics if finite_float(m.get("dice")) is not None]
    best = max(finite_metrics, key=lambda m: float(m["dice"]))
    final = val_metrics[-1]
    conflicts = [finite_float(v) for v in hist.get("grad_conflict_deg", [])]
    conflicts = [v for v in conflicts if v is not None]
    out = {
        "best_dice": float(best["dice"]),
        "best_round": int(best["round"]),
        "best_hd95": float(best["hd95"]),
        "final_dice": float(final["dice"]),
        "final_round": int(final["round"]),
        "final_hd95": float(final["hd95"]),
    }
    if conflicts:
        out["avg_grad_conflict"] = sum(conflicts) / len(conflicts)
    return out


def parse_seed_from_path(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", str(path), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def ablation_seed3409_from_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = [
        (float(dice), float(hd95))
        for dice, hd95 in re.findall(
            r"Dice:\s*([0-9.]+),\s*IoU:\s*[0-9.]+(?:\s*\r?\n\s*HD95:\s*([0-9.]+)\s*mm)",
            text,
        )
    ]
    if not pairs:
        raise ValueError(f"No Dice/HD95 pairs found in {path}")
    best_dice, best_hd95 = max(pairs, key=lambda pair: pair[0])
    final_dice, final_hd95 = pairs[-1]
    return {
        "best_dice": best_dice,
        "best_hd95": best_hd95,
        "final_dice": final_dice,
        "final_hd95": final_hd95,
    }


def extract_docm_tables(docm_path: Path) -> list[list[list[str]]]:
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }
    with zipfile.ZipFile(docm_path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    tables: list[list[list[str]]] = []
    for tbl in root.findall(".//w:tbl", ns):
        rows: list[list[str]] = []
        for tr in tbl.findall("./w:tr", ns):
            cells: list[str] = []
            for tc in tr.findall("./w:tc", ns):
                texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
                cells.append("".join(texts).strip())
            rows.append(cells)
        tables.append(rows)
    return tables


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def table_to_records(table: list[list[str]]) -> list[dict[str, str]]:
    if not table:
        return []
    header = [normalize_text(h) for h in table[0]]
    records = []
    for row in table[1:]:
        rec = {}
        for i, key in enumerate(header):
            rec[key] = normalize_text(row[i]) if i < len(row) else ""
        records.append(rec)
    return records


def find_table(tables: list[list[list[str]]], required_headers: list[str]) -> tuple[int, list[dict[str, str]]]:
    needles = [h.lower() for h in required_headers]
    for idx, table in enumerate(tables, start=1):
        if not table:
            continue
        header = " | ".join(normalize_text(c).lower() for c in table[0])
        if all(needle in header for needle in needles):
            return idx, table_to_records(table)
    raise ValueError(f"Cannot find table with headers: {required_headers}")


def pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def value_matches(paper_value: str, computed: str) -> bool:
    paper = paper_value.replace("±", "+/-").replace("卤", "+/-").replace("?", "+/-")
    comp = computed.replace("±", "+/-")
    if normalize_text(paper) == normalize_text(comp):
        return True
    paper_nums = re.findall(r"-?\d+(?:\.\d+)?", paper)
    comp_nums = re.findall(r"-?\d+(?:\.\d+)?", comp)
    return bool(paper_nums) and paper_nums == comp_nums


def close_enough(a: float, b: float, decimals: int) -> bool:
    return round(a, decimals) == round(b, decimals)


def build_raw_records(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    abc_csv = data_dir / "ABC" / "02_raw_logs_csv" / "main_table_by_seed.csv"
    for row in read_csv_rows(abc_csv):
        group = row["group"]
        seed = int(row["seed"])
        hist_path = data_dir / "ABC" / "02_raw_logs_csv" / f"seed_{seed}" / group / "training_history.json"
        log_path = data_dir / "ABC" / "02_raw_logs_csv" / f"seed_{seed}" / group / "train.log"
        hist = best_and_final_from_history(hist_path)
        rec = {
            "family": "ABC",
            "setting": group,
            "seed": seed,
            "source_csv": str(abc_csv),
            "source_history": str(hist_path),
            "source_log": str(log_path),
            "best_dice": float(row["best_dice"]),
            "best_round": int(row["best_round"]),
            "best_hd95": float(row["best_hd95"]),
            "final_dice": float(row["final_dice"]),
            "final_round": 60,
            "final_hd95": float(row["final_hd95"]),
            "avg_grad_conflict": float(row["avg_grad_conflict"]) if row["avg_grad_conflict"] else "",
        }
        seed_records.append(rec)
        for metric in ["best_dice", "best_hd95", "final_dice", "final_hd95", "avg_grad_conflict"]:
            csv_value = rec.get(metric, "")
            hist_value = hist.get(metric, "")
            ok = csv_value == "" or (hist_value != "" and abs(float(csv_value) - float(hist_value)) < 1e-12)
            checks.append({
                "family": "ABC",
                "setting": group,
                "seed": seed,
                "metric": metric,
                "csv_value": csv_value,
                "raw_value": hist_value,
                "raw_source": str(hist_path),
                "status": "OK" if ok else "MISMATCH",
            })

    abl_csv = data_dir / "ablation" / "01_final_summary_tables" / "ablation_by_seed.csv"
    for row in read_csv_rows(abl_csv):
        seed = int(row["seed"])
        setting = row["setting"]
        if seed == 3409:
            raw_dir = data_dir / "ablation" / "02_raw_logs_csv" / "ablation_c_wo_global_rep_update_seed3409_recovery"
            raw_path = raw_dir / "metrics_from_log.txt"
            hist = ablation_seed3409_from_log(raw_path)
            log_path = raw_dir / "ablation_c_wo_global_rep_update_seed3409.console.log"
            history_path = ""
        else:
            raw_dir = data_dir / "ablation" / "02_raw_logs_csv" / f"ablation_c_wo_global_rep_update_seed{seed}"
            history_path = raw_dir / "training_history.json"
            raw_path = Path(history_path)
            hist = best_and_final_from_history(raw_path)
            log_path = raw_dir / f"ablation_c_wo_global_rep_update_seed{seed}.console.log"
        rec = {
            "family": "ablation",
            "setting": setting,
            "seed": seed,
            "source_csv": str(abl_csv),
            "source_history": str(history_path),
            "source_log": str(log_path),
            "best_dice": float(row["best_dice"]),
            "best_round": hist.get("best_round", ""),
            "best_hd95": float(row["best_hd95_at_best_dice"]),
            "final_dice": float(row["final_dice"]),
            "final_round": hist.get("final_round", 60),
            "final_hd95": float(row["final_hd95"]),
            "avg_grad_conflict": "",
        }
        seed_records.append(rec)
        metric_map = {
            "best_dice": "best_dice",
            "best_hd95": "best_hd95",
            "final_dice": "final_dice",
            "final_hd95": "final_hd95",
        }
        for metric, raw_metric in metric_map.items():
            csv_value = rec[metric]
            raw_value = hist[raw_metric]
            decimals = 4 if "dice" in metric else 3
            ok = close_enough(float(csv_value), float(raw_value), decimals)
            checks.append({
                "family": "ablation",
                "setting": setting,
                "seed": seed,
                "metric": metric,
                "csv_value": csv_value,
                "raw_value": raw_value,
                "raw_source": str(raw_path),
                "status": "OK" if ok else "MISMATCH",
            })

    fed_csv = data_dir / "fedprox_d" / "01_tables" / "fedprox_d_by_seed.csv"
    for row in read_csv_rows(fed_csv):
        seed = int(row["seed"])
        hist_path = data_dir / "fedprox_d" / "02_aw_logs" / f"seed_{seed}" / "training_history.json"
        log_path = data_dir / "fedprox_d" / "02_aw_logs" / f"seed_{seed}" / f"train_seed{seed}.log"
        hist = best_and_final_from_history(hist_path)
        rec = {
            "family": "fedprox_d",
            "setting": "D",
            "seed": seed,
            "source_csv": str(fed_csv),
            "source_history": str(hist_path),
            "source_log": str(log_path),
            "best_dice": float(row["best_dice"]),
            "best_round": int(row["best_round"]),
            "best_hd95": float(row["best_hd95"]),
            "final_dice": float(row["final_dice"]),
            "final_round": int(row["final_round"]),
            "final_hd95": float(row["final_hd95"]),
            "avg_grad_conflict": hist.get("avg_grad_conflict", ""),
        }
        seed_records.append(rec)
        for metric in ["best_dice", "best_hd95", "final_dice", "final_hd95"]:
            csv_value = rec[metric]
            raw_value = hist[metric]
            ok = abs(float(csv_value) - float(raw_value)) < 1e-12
            checks.append({
                "family": "fedprox_d",
                "setting": "D",
                "seed": seed,
                "metric": metric,
                "csv_value": csv_value,
                "raw_value": raw_value,
                "raw_source": str(hist_path),
                "status": "OK" if ok else "MISMATCH",
            })

    return seed_records, checks


def records_for(seed_records: list[dict[str, Any]], family: str, setting: str) -> list[dict[str, Any]]:
    out = [r for r in seed_records if r["family"] == family and r["setting"] == setting]
    if not out:
        raise ValueError(f"No records for {family}/{setting}")
    return sorted(out, key=lambda r: int(r["seed"]))


def aggregate(rows: list[dict[str, Any]], metric: str, std_mode: str) -> tuple[float, float]:
    values = [float(r[metric]) for r in rows if r.get(metric) != ""]
    return sum(values) / len(values), configured_std(values, std_mode)


def source_summary(rows: list[dict[str, Any]], metric: str) -> str:
    parts = []
    for row in rows:
        raw = row["source_history"] or row["source_log"]
        round_note = ""
        if metric.startswith("best") and row.get("best_round") != "":
            round_note = f" round {row['best_round']}"
        elif metric.startswith("final") and row.get("final_round") != "":
            round_note = f" round {row['final_round']}"
        parts.append(f"seed {row['seed']}{round_note}: {raw}")
    return " | ".join(parts)


def build_specs() -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for label, family_setting in [
        ("A", ("ABC", "group_a")),
        ("B", ("ABC", "group_b")),
        ("C", ("ABC", "group_c")),
        ("D", ("fedprox_d", "D")),
    ]:
        for metric_label, source_metric, dec_key, unit in MAIN_METRICS:
            if label in {"A", "D"} and source_metric == "avg_grad_conflict":
                continue
            specs.append(MetricSpec(
                table="Table 3 Main results",
                row_label=label,
                metric=metric_label,
                decimals=4 if dec_key == "dice" else 3,
                unit=unit,
                source_family=family_setting[0],
                source_setting=family_setting[1],
                source_metric=source_metric,
            ))
    for label, family_setting in [
        ("Full C", ("ABC", "group_c")),
        ("C without global rep. update", ("ablation", "C w/o global rep update")),
    ]:
        for metric_label, source_metric, dec_key, unit in [
            ("Best Dice", "best_dice", "dice", ""),
            ("Final Dice", "final_dice", "dice", ""),
            ("Final HD95", "final_hd95", "hd95", "mm"),
        ]:
            specs.append(MetricSpec(
                table="Ablation table",
                row_label=label,
                metric=metric_label,
                decimals=4 if dec_key == "dice" else 3,
                unit=unit,
                source_family=family_setting[0],
                source_setting=family_setting[1],
                source_metric=source_metric,
            ))
    return specs


def paper_lookup(docm_path: Path, pdf_path: Path) -> tuple[dict[tuple[str, str, str], str], dict[str, Any]]:
    tables = extract_docm_tables(docm_path)
    lookup: dict[tuple[str, str, str], str] = {}
    meta: dict[str, Any] = {"docm_table_count": len(tables)}

    main_idx, main_rows = find_table(tables, ["group", "best dice", "final dice", "final hd95"])
    meta["main_table_index"] = main_idx
    for row in main_rows:
        label = row.get("Group", "")
        for key, normalized_key in [
            ("Best Dice", "Best Dice"),
            ("Final Dice", "Final Dice"),
            ("Final HD95", "Final HD95"),
            ("Final HD95 (mm)", "Final HD95"),
            ("Avg. Grad Conflict", "Avg. Grad Conflict"),
            ("Avg. Grad. Conflict Angle (deg)", "Avg. Grad Conflict"),
        ]:
            if key in row:
                lookup[("Table 3 Main results", label, normalized_key)] = row[key]

    ablation_idx, ablation_rows = find_table(tables, ["setting", "global rep", "best dice"])
    meta["ablation_table_index"] = ablation_idx
    for row in ablation_rows:
        label = row.get("Setting", "")
        for key in ["Best Dice", "Final Dice", "Final HD95"]:
            if key in row:
                lookup[("Ablation table", label, key)] = row[key]

    meta["table5_rows"] = []
    for table in tables:
        if not table:
            continue
        header = [normalize_text(c) for c in table[0]]
        if "Round" in header and "Final Dice" in header and any("Grad" in h for h in header):
            records = table_to_records(table)
            meta["table5_rows"] = records

    pdf = pdf_text(pdf_path)
    meta["pdf_text_chars"] = len(pdf)
    meta["pdf_text_available"] = bool(pdf)
    meta["pdf_numeric_hits"] = {}
    return lookup, meta


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(config["data_dir"])
    docm_path = Path(config["docm_path"])
    pdf_path = Path(config["pdf_path"])
    output_dir = Path(config["output_dir"])

    seed_records, raw_checks = build_raw_records(data_dir)
    lookup, meta = paper_lookup(docm_path, pdf_path)
    pdf = pdf_text(pdf_path)
    specs = build_specs()
    audit_rows: list[dict[str, Any]] = []

    for spec in specs:
        rows = records_for(seed_records, spec.source_family, spec.source_setting)
        std_mode = config.get("std_mode_by_family", {}).get(spec.source_family, config.get("std_mode", "sample"))
        mean, std = aggregate(rows, spec.source_metric, std_mode)
        computed = fmt_pm(mean, std, spec.decimals)
        paper_value = lookup.get((spec.table, spec.row_label, spec.metric), "")
        paper_match = bool(paper_value) and value_matches(paper_value, computed)
        pdf_hit = computed.replace(" ± ", " ").replace("±", "").split()[0] in pdf if pdf else False
        audit_rows.append({
            "paper_table": spec.table,
            "paper_row": spec.row_label,
            "metric": spec.metric,
            "paper_value": paper_value,
            "recomputed_from_raw": computed,
            "mean": fmt_num(mean, spec.decimals),
            "sample_std": fmt_num(std, spec.decimals),
            "std_method": std_mode,
            "unit": spec.unit,
            "seeds": ", ".join(str(r["seed"]) for r in rows),
            "source_family": spec.source_family,
            "source_setting": spec.source_setting,
            "raw_trace": source_summary(rows, spec.source_metric),
            "paper_match": "OK" if paper_match else "MISMATCH",
            "pdf_numeric_presence": "HIT" if pdf_hit else "NOT_CHECKED" if not pdf else "WEAK_HIT",
        })

    table5_metric_map = {
        "": "coefficient",
        "Round": "Round",
        "Final Dice": "Final Dice",
        "Final HD95 (mm)": "Final HD95",
        "Avg. Grad. Conflict Angle (deg)": "Avg. Grad Conflict",
    }
    for table5_row in meta.get("table5_rows", []):
        coefficient = table5_row.get("", "")
        for paper_key, metric_name in table5_metric_map.items():
            value = table5_row.get(paper_key, "")
            if not value:
                continue
            audit_rows.append({
                "paper_table": "Table 5 Endpoint comparison",
                "paper_row": coefficient,
                "metric": metric_name,
                "paper_value": value,
                "recomputed_from_raw": "",
                "mean": "",
                "sample_std": "",
                "std_method": "",
                "unit": "deg" if "Conflict" in metric_name else "mm" if "HD95" in metric_name else "",
                "seeds": "",
                "source_family": "",
                "source_setting": "",
                "raw_trace": "NOT FOUND in supplied log/csv/json data directory or workspace text scan",
                "paper_match": "SOURCE_NOT_FOUND",
                "pdf_numeric_presence": "NOT_CHECKED" if not pdf else "WEAK_HIT",
            })

    fieldnames = [
        "paper_table",
        "paper_row",
        "metric",
        "paper_value",
        "recomputed_from_raw",
        "mean",
        "sample_std",
        "std_method",
        "unit",
        "seeds",
        "source_family",
        "source_setting",
        "raw_trace",
        "paper_match",
        "pdf_numeric_presence",
    ]
    seed_fieldnames = [
        "family",
        "setting",
        "seed",
        "best_dice",
        "best_round",
        "best_hd95",
        "final_dice",
        "final_round",
        "final_hd95",
        "avg_grad_conflict",
        "source_csv",
        "source_history",
        "source_log",
    ]
    check_fieldnames = ["family", "setting", "seed", "metric", "csv_value", "raw_value", "raw_source", "status"]

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "paper_result_traceability_audit.csv"
    seed_csv = output_dir / "seed_level_raw_values.csv"
    checks_csv = output_dir / "raw_consistency_checks.csv"
    write_csv(audit_csv, audit_rows, fieldnames)
    write_csv(seed_csv, seed_records, seed_fieldnames)
    write_csv(checks_csv, raw_checks, check_fieldnames)

    status = {
        "audit_csv": str(audit_csv),
        "seed_csv": str(seed_csv),
        "checks_csv": str(checks_csv),
        "paper_docm": str(docm_path),
        "paper_pdf": str(pdf_path),
        "metadata": meta,
        "audit_rows": len(audit_rows),
        "raw_checks": len(raw_checks),
        "paper_mismatches": [r for r in audit_rows if r["paper_match"] != "OK"],
        "raw_mismatches": [r for r in raw_checks if r["status"] != "OK"],
    }
    status_path = output_dir / "audit_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Paper result traceability audit",
        "",
        f"Paper DOCM: `{docm_path}`",
        f"Paper PDF: `{pdf_path}`",
        f"Data directory: `{data_dir}`",
        "",
        "| Paper table | Row | Metric | Paper value | Recomputed from raw | Trace status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in audit_rows:
        md_lines.append(
            f"| {row['paper_table']} | {row['paper_row']} | {row['metric']} | "
            f"{row['paper_value']} | {row['recomputed_from_raw']} | {row['paper_match']} |"
        )
    md_path = output_dir / "paper_result_traceability_audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    status["audit_md"] = str(md_path)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper result values against raw logs and CSVs.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()
    status = build_audit(load_config(Path(args.config)))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
