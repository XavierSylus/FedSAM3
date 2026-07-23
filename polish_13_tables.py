from __future__ import annotations

import html
import os
import re
import sys
import zipfile
from pathlib import Path


TABLE_RE = re.compile(r"<w:tbl\b[^>]*>.*?</w:tbl>", re.DOTALL)
CELL_RE = re.compile(r"<w:tc\b[^>]*>.*?</w:tc>", re.DOTALL)
TEXT_RE = re.compile(r"<(?:w|m):t(?:\s[^>]*)?>(.*?)</(?:w|m):t>", re.DOTALL)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def wr(text: str) -> str:
    return (
        '<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        f"<w:t>{esc(text)}</w:t></w:r>"
    )


def extract_text(xml: str) -> str:
    return html.unescape("".join(TEXT_RE.findall(xml))).strip()


def set_size_9pt(tbl: str) -> str:
    tbl = re.sub(r'<w:sz\s+w:val="\d+"\s*/>', '<w:sz w:val="18"/>', tbl)
    tbl = re.sub(r'<w:szCs\s+w:val="\d+"\s*/>', '<w:szCs w:val="18"/>', tbl)
    return tbl


def set_table_widths(tbl: str, widths: list[int]) -> str:
    total = sum(widths)
    tbl = re.sub(r'<w:tblW\b[^>]*/>', f'<w:tblW w:w="{total}" w:type="dxa"/>', tbl, count=1)
    if re.search(r"<w:tblLayout\b[^>]*/>", tbl):
        tbl = re.sub(r'<w:tblLayout\b[^>]*/>', '<w:tblLayout w:type="fixed"/>', tbl, count=1)
    else:
        tbl = tbl.replace("</w:tblPr>", '<w:tblLayout w:type="fixed"/></w:tblPr>', 1)

    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", grid, tbl, count=1, flags=re.DOTALL)

    col_count = len(widths)

    def patch_cell(match: re.Match[str]) -> str:
        patch_cell.index += 1
        width = widths[(patch_cell.index - 1) % col_count]
        cell = match.group(0)
        if re.search(r"<w:tcW\b[^>]*/>", cell):
            cell = re.sub(r'<w:tcW\b[^>]*/>', f'<w:tcW w:w="{width}" w:type="dxa"/>', cell, count=1)
        else:
            cell = cell.replace("<w:tcPr>", f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>', 1)
        return cell

    patch_cell.index = 0
    return CELL_RE.sub(patch_cell, tbl)


def set_vertical_center(tbl: str) -> str:
    def patch_cell(match: re.Match[str]) -> str:
        cell = match.group(0)
        if re.search(r"<w:vAlign\b[^>]*/>", cell):
            return re.sub(r'<w:vAlign\b[^>]*/>', '<w:vAlign w:val="center"/>', cell, count=1)
        return cell.replace("</w:tcPr>", '<w:vAlign w:val="center"/></w:tcPr>', 1)

    return CELL_RE.sub(patch_cell, tbl)


def set_cell_text(cell: str, text: str) -> str:
    para_match = re.search(r"(<w:p\b[^>]*>)(.*?)(</w:p>)", cell, re.DOTALL)
    if not para_match:
        return cell
    p_start, p_body, p_end = para_match.groups()
    ppr_match = re.match(r"(<w:pPr>.*?</w:pPr>)", p_body, re.DOTALL)
    ppr = ppr_match.group(1) if ppr_match else ""
    new_para = f"{p_start}{ppr}{wr(text)}{p_end}"
    return cell[: para_match.start()] + new_para + cell[para_match.end() :]


def replace_cell_texts(tbl: str, replacements: dict[str, str]) -> str:
    def patch_cell(match: re.Match[str]) -> str:
        cell = match.group(0)
        text = extract_text(cell)
        if text in replacements:
            return set_cell_text(cell, replacements[text])
        return cell

    return CELL_RE.sub(patch_cell, tbl)


def set_cell_alignment(cell: str, align: str) -> str:
    def patch_para(match: re.Match[str]) -> str:
        para = match.group(0)
        if "<w:pPr>" in para:
            if re.search(r"<w:jc\b[^>]*/>", para):
                return re.sub(r'<w:jc\b[^>]*/>', f'<w:jc w:val="{align}"/>', para, count=1)
            return para.replace("</w:pPr>", f'<w:jc w:val="{align}"/></w:pPr>', 1)
        return para.replace(">", f"><w:pPr><w:jc w:val=\"{align}\"/></w:pPr>", 1)

    return re.sub(r"<w:p\b[^>]*>.*?</w:p>", patch_para, cell, flags=re.DOTALL)


def align_columns(tbl: str, col_count: int, center_cols: set[int]) -> str:
    def patch_cell(match: re.Match[str]) -> str:
        patch_cell.index += 1
        col = ((patch_cell.index - 1) % col_count) + 1
        cell = match.group(0)
        return set_cell_alignment(cell, "center" if col in center_cols else "left")

    patch_cell.index = 0
    return CELL_RE.sub(patch_cell, tbl)


def patch_table(tbl: str, idx: int) -> str:
    table1_replacements = {
        "Corresponding Module": "Module",
        "Main Function": "Function",
        "Allowed Updating Clients": "Clients",
        "Server-Side Processing": "Server Processing",
        "Shared parameter group": "Shared parameters",
        "Shared visual representation and general segmentation layers in the backbone": (
            "Shared visual representation and general segmentation backbone layers"
        ),
        "Provide shared representation capacity": "Shared representation capacity",
        "Vision parameter group": "Vision parameters",
        "Image representation extraction and segmentation supervision": (
            "Image representation and segmentation supervision"
        ),
        "Only visually supervised clients aggregate": "Visually supervised clients aggregate",
        "Multimodal parameter group": "Multimodal parameters",
        "Image-text interaction, text adaptation, and alignment modules": (
            "Image-text interaction, text adaptation, alignment modules"
        ),
        "Textual semantic injection and consistency modeling": "Text semantic injection and consistency modeling",
        "Only joint image-text clients aggregate": "Joint image-text clients aggregate",
        "Global image representation": "Global image representation",
        "Maintain cross-round image semantic anchors": "Maintain cross-round image semantic anchors",
        "Updated by EMA; excluded from client-side averaging": "EMA update; no client averaging",
        "Global text representation": "Global text representation",
        "Updated by EMA; excluded from visual parameter aggregation": "EMA update; no visual parameter aggregation",
    }
    metric_replacements = {
        "Average Gradient Conflict Angle (deg)": "Avg. Grad. Conflict Angle (deg)",
    }
    table4_replacements = {
        "Restricted Routing ": "Restricted Routing",
        "Global Representation Update": "Global Rep. Update",
        "C without global representation update": "C without global rep. update",
    }

    tbl = set_size_9pt(tbl)
    tbl = set_vertical_center(tbl)

    if idx == 1:
        tbl = replace_cell_texts(tbl, table1_replacements)
        tbl = set_table_widths(tbl, [1050, 1550, 1450, 1280, 1724])
        tbl = align_columns(tbl, 5, {4})
    elif idx == 3:
        tbl = replace_cell_texts(tbl, metric_replacements)
        tbl = set_table_widths(tbl, [650, 1150, 1250, 880, 880, 1020, 1224])
        tbl = align_columns(tbl, 7, {1, 4, 5, 6, 7})
    elif idx == 4:
        tbl = replace_cell_texts(tbl, table4_replacements)
        tbl = set_table_widths(tbl, [1800, 1100, 1150, 950, 950, 1100])
        tbl = align_columns(tbl, 6, {2, 3, 4, 5, 6})
    elif idx == 5:
        tbl = replace_cell_texts(tbl, metric_replacements)
        tbl = set_table_widths(tbl, [650, 800, 1050, 1600, 1950])
        tbl = align_columns(tbl, 5, {1, 2, 3, 4, 5})

    return tbl


def patch_docm(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.docm")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "word/document.xml":
                xml = data.decode("utf-8")
                tables = list(TABLE_RE.finditer(xml))
                if len(tables) != 5:
                    raise RuntimeError(f"Expected 5 tables, found {len(tables)}")
                parts: list[str] = []
                pos = 0
                for idx, match in enumerate(tables, start=1):
                    parts.append(xml[pos : match.start()])
                    parts.append(patch_table(match.group(0), idx))
                    pos = match.end()
                parts.append(xml[pos:])
                xml = "".join(parts)
                xml = keep_table4_together(xml)
                data = xml.encode("utf-8")
            out = zipfile.ZipInfo(info.filename, info.date_time)
            out.compress_type = zipfile.ZIP_DEFLATED
            out.external_attr = info.external_attr
            out.comment = info.comment
            zout.writestr(out, data)
    os.replace(tmp, dst)


def add_page_break_before(para: str) -> str:
    if "<w:pageBreakBefore" in para:
        return para
    if "<w:pPr>" in para:
        return para.replace("<w:pPr>", "<w:pPr><w:pageBreakBefore/>", 1)
    start = re.match(r"(<w:p\b[^>]*>)", para)
    if not start:
        return para
    return para[: start.end()] + "<w:pPr><w:pageBreakBefore/></w:pPr>" + para[start.end() :]


def keep_table4_together(xml: str) -> str:
    def patch_para(match: re.Match[str]) -> str:
        para = match.group(0)
        if extract_text(para).startswith("Table 4."):
            return add_page_break_before(para)
        return para

    return re.sub(r"<w:p\b[^>]*>.*?</w:p>", patch_para, xml, flags=re.DOTALL)


if __name__ == "__main__":
    patch_docm(Path(sys.argv[1]), Path(sys.argv[2]))
    print("PATCHED")
