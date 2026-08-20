import argparse
import copy
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS, "m": M_NS}


def qn(namespace, local_name):
    return f"{{{namespace}}}{local_name}"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def paragraph_text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def find_paragraph(root, startswith):
    matches = [
        paragraph
        for paragraph in root.xpath(".//w:body/w:p", namespaces=NS)
        if paragraph_text(paragraph).startswith(startswith)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting with {startswith!r}, found {len(matches)}")
    return matches[0]


def set_paragraph_text(paragraph, text):
    paragraph_properties = paragraph.find("w:pPr", namespaces=NS)
    first_run_properties = paragraph.find(".//w:rPr", namespaces=NS)
    for child in list(paragraph):
        if child is not paragraph_properties:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, qn(W_NS, "r"))
    if first_run_properties is not None:
        run.append(copy.deepcopy(first_run_properties))
    text_node = etree.SubElement(run, qn(W_NS, "t"))
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(qn(XML_NS, "space"), "preserve")
    text_node.text = text


def set_cell_text(cell, text):
    paragraphs = cell.findall("w:p", namespaces=NS)
    if not paragraphs:
        paragraphs = [etree.SubElement(cell, qn(W_NS, "p"))]
    set_paragraph_text(paragraphs[0], text)
    for paragraph in paragraphs[1:]:
        cell.remove(paragraph)


def set_table_geometry(table, column_widths):
    total_width = sum(column_widths)
    table_properties = table.find("w:tblPr", namespaces=NS)
    if table_properties is None:
        table_properties = etree.SubElement(table, qn(W_NS, "tblPr"))

    table_width = table_properties.find("w:tblW", namespaces=NS)
    if table_width is None:
        table_width = etree.SubElement(table_properties, qn(W_NS, "tblW"))
    table_width.set(qn(W_NS, "w"), str(total_width))
    table_width.set(qn(W_NS, "type"), "dxa")

    table_indent = table_properties.find("w:tblInd", namespaces=NS)
    if table_indent is None:
        table_indent = etree.SubElement(table_properties, qn(W_NS, "tblInd"))
    table_indent.set(qn(W_NS, "w"), "120")
    table_indent.set(qn(W_NS, "type"), "dxa")

    table_layout = table_properties.find("w:tblLayout", namespaces=NS)
    if table_layout is None:
        table_layout = etree.SubElement(table_properties, qn(W_NS, "tblLayout"))
    table_layout.set(qn(W_NS, "type"), "fixed")

    table_grid = table.find("w:tblGrid", namespaces=NS)
    if table_grid is None:
        table_grid = etree.Element(qn(W_NS, "tblGrid"))
        table.insert(1, table_grid)
    for child in list(table_grid):
        table_grid.remove(child)
    for width in column_widths:
        grid_column = etree.SubElement(table_grid, qn(W_NS, "gridCol"))
        grid_column.set(qn(W_NS, "w"), str(width))

    for row in table.findall("w:tr", namespaces=NS):
        cells = row.findall("w:tc", namespaces=NS)
        if len(cells) != len(column_widths):
            raise ValueError(f"Expected {len(column_widths)} cells, found {len(cells)}")
        for cell, width in zip(cells, column_widths):
            cell_properties = cell.find("w:tcPr", namespaces=NS)
            if cell_properties is None:
                cell_properties = etree.Element(qn(W_NS, "tcPr"))
                cell.insert(0, cell_properties)
            for property_name in ("gridSpan", "vMerge"):
                property_node = cell_properties.find(f"w:{property_name}", namespaces=NS)
                if property_node is not None:
                    cell_properties.remove(property_node)
            cell_width = cell_properties.find("w:tcW", namespaces=NS)
            if cell_width is None:
                cell_width = etree.SubElement(cell_properties, qn(W_NS, "tcW"))
            cell_width.set(qn(W_NS, "w"), str(width))
            cell_width.set(qn(W_NS, "type"), "dxa")


def rebuild_table(table, headers, rows):
    source_rows = table.findall("w:tr", namespaces=NS)
    if len(source_rows) < 2:
        raise ValueError("The source table must contain a header and at least one data row")
    header_template = copy.deepcopy(source_rows[0])
    data_template = copy.deepcopy(source_rows[1])
    for row in source_rows:
        table.remove(row)

    new_rows = [header_template] + [copy.deepcopy(data_template) for _ in rows]
    for row in new_rows:
        cells = row.findall("w:tc", namespaces=NS)
        while len(cells) > len(headers):
            row.remove(cells.pop())
        while len(cells) < len(headers):
            new_cell = copy.deepcopy(cells[-1])
            row.append(new_cell)
            cells.append(new_cell)
        table.append(row)

    for cell, text in zip(new_rows[0].findall("w:tc", namespaces=NS), headers):
        set_cell_text(cell, text)
    for row_element, values in zip(new_rows[1:], rows):
        for cell, text in zip(row_element.findall("w:tc", namespaces=NS), values):
            set_cell_text(cell, text)

    table_grid = table.find("w:tblGrid", namespaces=NS)
    original_widths = []
    if table_grid is not None:
        for grid_column in table_grid.findall("w:gridCol", namespaces=NS):
            width = grid_column.get(qn(W_NS, "w"))
            if width and width.isdigit():
                original_widths.append(int(width))
    total_width = sum(original_widths) or 9360
    ratios = (0.15, 0.13, 0.14, 0.14, 0.14, 0.30)
    widths = [round(total_width * ratio) for ratio in ratios[:-1]]
    widths.append(total_width - sum(widths))
    set_table_geometry(table, widths)


def signed(value, decimals):
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"


def signed_mean_sd(values, decimals):
    return f"{signed(statistics.mean(values), decimals)} ± {statistics.stdev(values):.{decimals}f}"


def math_digest(xml_bytes):
    root = etree.fromstring(xml_bytes)
    math_nodes = root.xpath(".//m:oMath", namespaces=NS)
    canonical = b"".join(etree.tostring(node, method="c14n") for node in math_nodes)
    return len(math_nodes), sha256_bytes(canonical)


def load_results(manifest_path, config_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    seeds = [3407, 3408, 3409]
    aggregations = ["FedAvg", "FedProx"]
    routings = ["U", "R"]
    metrics = ["dice", "iou", "hd95_mm"]
    rows = manifest["source_rows"]

    expected_keys = {
        (seed, routing, aggregation)
        for seed in seeds
        for routing in routings
        for aggregation in aggregations
    }
    lookup = {(row["seed"], row["routing"], row["aggregation"]): row for row in rows}
    if len(rows) != 12 or set(lookup) != expected_keys:
        raise ValueError("The manifest does not contain the expected 12 unique seed/routing/aggregation cells")
    if any(not row["member"].endswith("formal_verification/final_metrics.csv") for row in rows):
        raise ValueError("At least one source row is not a formal final_metrics.csv result")
    if config["source"].get("expected_round") != 60:
        raise ValueError("The pre-fixed endpoint rule is not round 60")
    shared_definition = manifest["qa"]["data_integrity"]["shared_test_definition"]
    if "round 60" not in shared_definition or "32 cases" not in shared_definition:
        raise ValueError("The shared endpoint/test-set definition is incomplete")

    paired = {}
    for aggregation in aggregations:
        paired[aggregation] = {}
        for metric in metrics:
            paired[aggregation][metric] = [
                lookup[(seed, "R", aggregation)][metric] - lookup[(seed, "U", aggregation)][metric]
                for seed in seeds
            ]

    for aggregation in aggregations:
        for metric in metrics:
            summary = manifest["summary"][metric]
            expected_mean_delta = summary[f"{aggregation}_R"]["mean"] - summary[f"{aggregation}_U"]["mean"]
            if abs(statistics.mean(paired[aggregation][metric]) - expected_mean_delta) > 1e-12:
                raise ValueError(f"Paired delta mismatch for {aggregation}/{metric}")

    return manifest, config, seeds, paired


def build_paired_rows(seeds, paired):
    metric_specs = [
        ("dice", "Dice", 4),
        ("iou", "IoU", 4),
        ("hd95_mm", "HD95 (mm)", 2),
    ]
    table_rows = []
    report = []
    for aggregation in ("FedAvg", "FedProx"):
        for metric_key, metric_label, decimals in metric_specs:
            values = paired[aggregation][metric_key]
            table_rows.append(
                [
                    aggregation,
                    metric_label,
                    *[signed(value, decimals) for value in values],
                    signed_mean_sd(values, decimals),
                ]
            )
            report.append(
                {
                    "aggregation": aggregation,
                    "metric": metric_key,
                    "paired_delta_by_seed": dict(zip(seeds, values)),
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                }
            )
    return table_rows, report


def revise_document(source_path, output_path, manifest_path, config_path):
    manifest, config, seeds, paired = load_results(manifest_path, config_path)
    table_rows, paired_report = build_paired_rows(seeds, paired)
    source_bytes = Path(source_path).read_bytes()
    source_sha256 = sha256_bytes(source_bytes)

    with zipfile.ZipFile(source_path, "r") as source_package:
        source_entries = source_package.namelist()
        document_xml = source_package.read("word/document.xml")
        macro_bytes = source_package.read("word/vbaProject.bin") if "word/vbaProject.bin" in source_entries else None
        untouched_hashes = {
            name: sha256_bytes(source_package.read(name))
            for name in source_entries
            if name != "word/document.xml"
        }

    root = etree.fromstring(document_xml)
    original_math_count, original_math_hash = math_digest(document_xml)
    original_table_count = len(root.xpath(".//w:body/w:tbl", namespaces=NS))

    abstract = find_paragraph(root, "摘要—医学图像联邦学习中")
    set_paragraph_text(
        abstract,
        "摘要—医学图像联邦学习中，客户端可能仅持有文本表征、仅持有图像与标注，或持有配对图文；不同客户端因而具有不同的本地目标、优化作用域和可上传参数。本文提出 FedSAM3-Hetero，将这种差异显式写入客户端上传契约与服务器端逐参数聚合资格。U 模式在每个参数的样本加权分母中保留全部活跃客户端，并将未上传参数视为零更新；R 模式仅对满足参数组模态白名单的实际上传者重归一化。模型保持原始 SAM3 编码器及其几何编码逻辑，并在固定 BraTS 2020 异构划分、三个客户端和三个随机种子下比较 U/R 与 FedAvg/FedProx 的 2×2 组合。现有结果显示，在完整三客户端协议中，尽管三个种子的绝对指标波动较大，R 相对 U 在两种本地目标下的同种子配对差异方向一致，表现为更高的 Dice/IoU 和更低的 HD95；但在同时改变参与比例与活跃模态组成的 2/3 检查中，U/R 平均差异小于跨种子波动。由于 U/R 同时改变聚合资格、分母重归一化和参数的有效更新尺度，本文不将该差异归因为单一机制，也不主张任一路由普遍最优。本文的主要贡献是给出可记录、可复核的逐参数协作边界，并报告其在当前协议下的条件性结果。",
    )

    setup_paragraph = find_paragraph(root, "主实验固定代码定义的客户端角色")
    set_paragraph_text(
        setup_paragraph,
        "主实验固定代码定义的客户端角色、参数组、数据顺序、训练控制与三种随机种子，只比较 U/R 聚合契约和 FedAvg/FedProx 本地目标。表 2 汇总四个正式单元的均值 ± 样本标准差；表 3 进一步列出 R 相对同一基线 U 的逐种子配对差异。每轮参数审计同时核对上传范围、模态资格、零更新及样本权重。",
    )

    table2_caption = find_paragraph(root, "表 2. 代码对齐的 U/R")
    table2 = table2_caption.getnext()
    if table2 is None or table2.tag != qn(W_NS, "tbl"):
        raise ValueError("Table 2 does not immediately follow its caption")

    paired_caption = copy.deepcopy(table2_caption)
    set_paragraph_text(
        paired_caption,
        "表 3. 代码对齐主实验的同种子配对差异（paired delta，Δ=R−U；n=3）。最后一列为三个配对差异的均值 ± 样本标准差；Dice/IoU 的正值和 HD95 的负值表示 R 方向更优。",
    )
    paired_table = copy.deepcopy(table2)
    rebuild_table(
        paired_table,
        ["本地目标", "指标", "Δ3407", "Δ3408", "Δ3409", "均值 ± 样本SD"],
        table_rows,
    )

    result_paragraph = find_paragraph(root, "在 FedAvg 条件下，R 相对 U")
    paired_note = copy.deepcopy(result_paragraph)
    set_paragraph_text(
        paired_note,
        "注：Δ 按同一随机种子的 R−U 计算；结果来自预先固定的第 60 轮 formal_verification/final_metrics.csv，未选择最佳 seed 或最佳 checkpoint。",
    )

    table2.addnext(paired_caption)
    paired_caption.addnext(paired_table)
    paired_table.addnext(paired_note)

    set_paragraph_text(
        result_paragraph,
        "表 2 的四个正式单元均以均值 ± 样本标准差汇总；表 3 显示，在 FedAvg 和 FedProx 下，三个固定种子的 Dice 与 IoU 配对差异均为正、HD95 配对差异均为负，即 18/18 个“种子 × 本地目标 × 指标”比较方向一致。在 FedAvg 条件下，R−U 的 Dice、IoU 和 HD95 分别为 +0.0971 ± 0.0388、+0.0978 ± 0.0392 和 −11.87 ± 5.57 mm；在 FedProx 条件下，相应差异为 +0.1118 ± 0.0547、+0.1127 ± 0.0547 和 −12.48 ± 5.21 mm。上述结果仅为描述性统计：由于 n=3、未实施预先规定的假设检验且 U/R 同时改变资格集合与归一化尺度，本文不使用“显著”措辞，也不以任一最佳种子代替总体结果，不能据此确定单一机制的独立贡献或普遍优越性。",
    )

    sensitivity_intro = find_paragraph(root, "该检查只启用图像专属与多模态客户端")
    set_paragraph_text(
        sensitivity_intro,
        "该检查只启用图像专属与多模态客户端，在相同 FedAvg 框架、60 轮训练和三个随机种子下比较 U-FedAvg 与 R-FedAvg。相对于完整三客户端主比较，这一设置同时降低参与比例并移除文本专属客户端，因此只用于观察组合条件变化后的敏感性，不能分离参与比例或特定模态缺失的独立效应。表 4 报告各随机种子及其均值。",
    )
    sensitivity_caption = find_paragraph(root, "表 3. 2/3 参与")
    set_paragraph_text(
        sensitivity_caption,
        "表 4. 2/3 参与且活跃模态组成改变时的 U-FedAvg 与 R-FedAvg 最终结果。↑/↓ 表示数值越大/越小越好；HD95 的单位为 mm，正文差异按 U−R 计算。",
    )

    conclusion = find_paragraph(root, "FedSAM3-Hetero 将客户端模态异质性写成")
    set_paragraph_text(
        conclusion,
        "FedSAM3-Hetero 将客户端模态异质性写成可审计的本地目标、优化/上传作用域、五类参数组和服务器端 U/R 聚合契约。现有证据分为三个层级：历史 A–D 仅保留为探索性记录；代码对齐的 2×2 主比较显示，在绝对指标跨种子波动较大的情况下，R 在当前完整三客户端协议的两种本地目标及三个固定种子中均呈同向配对差异；当参与比例与活跃模态组成同时改变时，U/R 平均差异则小于跨种子波动。本文由此支持的是逐参数协作边界的可记录、可复核及其条件依赖性，而不是任一路由规则在所有联邦场景中的普遍最优性。",
    )

    revised_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    revised_math_count, revised_math_hash = math_digest(revised_xml)
    if (revised_math_count, revised_math_hash) != (original_math_count, original_math_hash):
        raise ValueError("OMML formula content changed during the revision")
    if len(root.xpath(".//w:body/w:tbl", namespaces=NS)) != original_table_count + 1:
        raise ValueError("The revised document does not contain exactly one additional table")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path, "r") as source_package, zipfile.ZipFile(output, "w") as output_package:
        for entry in source_package.infolist():
            data = revised_xml if entry.filename == "word/document.xml" else source_package.read(entry.filename)
            output_package.writestr(entry, data)

    with zipfile.ZipFile(output, "r") as output_package:
        if output_package.namelist() != source_entries:
            raise ValueError("Package entry order or membership changed")
        output_document_xml = output_package.read("word/document.xml")
        for name, expected_hash in untouched_hashes.items():
            if sha256_bytes(output_package.read(name)) != expected_hash:
                raise ValueError(f"Unexpected change to package part: {name}")
        if macro_bytes is not None and output_package.read("word/vbaProject.bin") != macro_bytes:
            raise ValueError("The VBA project changed")
        content_types = output_package.read("[Content_Types].xml")
        if b"application/vnd.ms-word.document.macroEnabled.main+xml" not in content_types:
            raise ValueError("The macro-enabled main document content type is missing")

    if sha256_bytes(Path(source_path).read_bytes()) != source_sha256:
        raise ValueError("The source document was modified")
    if math_digest(output_document_xml) != (original_math_count, original_math_hash):
        raise ValueError("Final package formula verification failed")

    output_root = etree.fromstring(output_document_xml)
    output_text = "\n".join(paragraph_text(p) for p in output_root.xpath(".//w:p", namespaces=NS))
    required_fragments = [
        "表 3. 代码对齐主实验的同种子配对差异",
        "+0.0978 ± 0.0392",
        "−12.48 ± 5.21",
        "表 4. 2/3 参与",
        "不以任一最佳种子代替总体结果",
    ]
    for fragment in required_fragments:
        if fragment not in output_text:
            raise ValueError(f"Missing required output fragment: {fragment}")

    return {
        "source": str(source_path),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": sha256_bytes(output.read_bytes()),
        "fixed_endpoint_round": config["source"]["expected_round"],
        "seed_count": len(seeds),
        "source_result_cells": len(manifest["source_rows"]),
        "paired_delta": paired_report,
        "formula_count": original_math_count,
        "formula_canonical_sha256": original_math_hash,
        "vba_project_present": macro_bytes is not None,
        "vba_project_sha256": sha256_bytes(macro_bytes) if macro_bytes is not None else None,
        "package_parts_unchanged_except_document_xml": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-docm", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-docm", required=True)
    args = parser.parse_args()
    report = revise_document(
        args.source_docm,
        args.output_docm,
        args.manifest,
        args.experiment_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
