from __future__ import annotations

import html
import os
import re
import sys
import zipfile
from pathlib import Path


PARA_RE = re.compile(r"<w:p\b[^>]*>.*?</w:p>", re.DOTALL)
TABLE_RE = re.compile(r"<w:tbl\b[^>]*>.*?</w:tbl>", re.DOTALL)
CELL_RE = re.compile(r"<w:tc\b[^>]*>.*?</w:tc>", re.DOTALL)
TEXT_RE = re.compile(r"<(?:w|m):t(?:\s[^>]*)?>(.*?)</(?:w|m):t>", re.DOTALL)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def extract_text(xml: str) -> str:
    return html.unescape("".join(TEXT_RE.findall(xml))).strip()


def split_para(para: str) -> tuple[str, str]:
    start = re.match(r"(<w:p\b[^>]*>)", para)
    if not start:
        raise RuntimeError("Invalid paragraph")
    rest = para[start.end() :]
    ppr = re.match(r"(<w:pPr>.*?</w:pPr>)", rest, re.DOTALL)
    return start.group(1), ppr.group(1) if ppr else ""


def para_with_text(para: str, text: str) -> str:
    start, ppr = split_para(para)
    preserve = "preserve" if text.startswith(" ") or text.endswith(" ") else "default"
    return f'{start}{ppr}<w:r><w:t xml:space="{preserve}">{esc(text)}</w:t></w:r></w:p>'


def replace_text_values(xml: str, replacements: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        full = match.group(0)
        text = html.unescape(match.group(1))
        if text not in replacements:
            return full
        return full.replace(match.group(1), esc(replacements[text]), 1)

    return TEXT_RE.sub(repl, xml)


def patch_caption_parts(para: str, replacements: dict[str, str]) -> str:
    return replace_text_values(para, replacements)


def patch_paragraphs(xml: str) -> str:
    full_para_replacements = {
        "Under missing-modality heterogeneity, different clients reliably update different parameter subspaces. FedSAM3-Hetero therefore partitions parameters, defines eligible clients for each group, and admits only valid updates.": "Under missing-modality heterogeneity, clients update different parameter subspaces. FedSAM3-Hetero partitions parameters, defines eligible clients for each group, and admits only valid updates.",
        "Let S_g denote clients allowed to aggregate the g-th parameter group. Its update after round t is written as Eq. (5):": "Let S_g denote clients eligible for group g. Its round-t update is Eq. (5):",
        "If a parameter group has no eligible uploader, the server retains its previous global value (Eq. (6)):": "If a group has no eligible uploader, the server retains its previous global value (Eq. (6)):",
        "This rule avoids invalid perturbations without valid updates and lets routing adapt to missing updates across clients and rounds.": "This avoids invalid perturbations and adapts routing to missing updates across clients and rounds.",
        "Beyond restricted routing, the server maintains cross-round global representations: aggregation constrains update eligibility, while representation statistics retain semantic references.": "Beyond restricted routing, the server maintains cross-round representations: aggregation constrains update eligibility, while statistics retain semantic references.",
        "After each round, the server updates global image and text representations from uploads using EMA (Eq. (7)):": "After each round, the server updates image and text representations using EMA (Eq. (7)):",
        "The overhead is limited: routing performs server-side whitelist filtering, while global representation update adds one 768-dimensional vector per client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload.": "Overhead is limited: routing uses server-side whitelist filtering, while global representation update adds one 768-dimensional vector per client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload.",
        "FedProx outperforms the internal protocols in Dice and HD95, which indicates that client-drift suppression is a strong competing mechanism. Therefore, this paper claims multimodal gains and lightweight routing benefits, not overall superiority over robust federated methods.": "FedProx outperforms internal protocols in Dice and HD95, indicating strong client-drift suppression; we therefore claim multimodal gains and lightweight routing benefits, not overall superiority over robust federated methods.",
        "Fig. 2 supports restricted routing as a stabilization mechanism rather than a strong optimization intervention. Groups B and C remain below 90∘ for most rounds, and Group C shows smoother HD95 trends, consistent with better boundary quality.": "Fig. 2 supports restricted routing as stabilization rather than strong optimization: B/C remain below 90∘ for most rounds, and C shows smoother HD95 trends.",
        "Table 5 compares Final Dice, Final HD95, and the average gradient conflict angle for Group C after 30 rounds under different λcream. Increasing λcream from 0.02 to 0.10 and 0.20 does not yield monotonic improvement.": "Table 5 compares Final Dice, Final HD95, and average gradient conflict for Group C after 30 rounds under λcream values of 0.02, 0.10, and 0.20, which do not yield monotonic improvement.",
        "Distillation strength does not dominate the difference between Groups B and C. The close metrics should not be attributed simply to a small λcream, nor should they be expected to improve by increasing it. Section 4.4 instead points to parameter mixing boundary constraints.": "Thus, distillation strength does not dominate the B-C difference; Section 4.4 instead points to parameter mixing boundary constraints.",
        "Overall, multimodal participation improves the image-only baseline, while restricted routing mainly constrains mixing boundaries and yields limited boundary-quality and stability gains.": "Multimodal participation improves the image-only baseline, while restricted routing yields limited boundary-quality and stability gains.",
        "Global representation update and distillation-weight ablation do not separately explain the B-C difference; the evidence instead points to parameter mixing boundary constraints.": "Global representation update and distillation-weight ablation remain auxiliary, pointing to parameter mixing constraints.",
        "Fig. 3 compares validation slices: Cases 0043 and 0044 support restricted routing, Case 0046 is marginal, and Case 0049 favors Group B, indicating sample-dependent improvement.": "Fig. 3 compares validation slices: Cases 0043/0044 support restricted routing, 0046 is marginal, and 0049 favors Group B, indicating sample-dependent gains.",
        "Fig. 4 summarizes endpoint performance: multimodal participation improves over Group A, while restricted routing mainly reduces HD95 slightly.": "Fig. 4 shows endpoint performance: multimodal participation improves over A, while routing mainly reduces HD95 slightly.",
        "The evidence suggests that constraining parameter mixing boundaries improves stability and interpretability and provides a reusable baseline for missing-modality federated segmentation.": "Constraining parameter mixing improves stability and interpretability, offering a reusable baseline for missing-modality federated segmentation.",
        "Several limitations remain. The main experiments include only image-only and multimodal clients; text-only clients are retained as an extensible setting but excluded from quantitative experiments. Thus, the conclusions mainly apply to image-client and image-text-client settings.": "The main experiments include only image-only and multimodal clients; text-only clients remain extensible but are excluded from quantitative experiments, so conclusions mainly apply to image and image-text clients.",
        "Second, the BraTS-style MRI task requires validation on more medical segmentation tasks, client compositions, and larger multi-center datasets.": "The BraTS-style MRI task also requires validation on more segmentation tasks, client compositions, and larger multi-center datasets.",
        "Third, restricted routing yields only limited boundary-quality and stability gains, while global representation update remains auxiliary in the current setting.": "Finally, routing gains are limited, and global representation update remains auxiliary in this setting.",
    }

    caption_replacements = {
        " Training dynamics of Groups A, B, and C: validation Dice, training loss, validation HD95, and gradient conflict. Solid lines denote the mean over three random seeds, and shaded regions indicate ±1 standard deviation.": " Training dynamics of Groups A-C. Solid lines denote the mean over three random seeds, and shaded regions indicate ±1 standard deviation.",
        " Qualitative comparison. A is the image-only baseline, B is direct FedAvg with multimodal clients, and C is restricted routing.": " Qualitative comparison: A image-only, B direct multimodal FedAvg, C restricted routing.",
    }

    inline_replacements_by_text = {
        "Here, θk(t+1) denotes the parameters of the k-th client after local training, and wk is the normalized aggregation weight. FedAvg assumes that clients have similar update capability over the same parameter space.": {
            " denotes the parameters of the ": " denotes parameters from client ",
            "-th client after local training, and ": " after local training, and ",
            " is the normalized aggregation weight. FedAvg assumes that clients have similar update capability over the same parameter space.": " is the normalized aggregation weight. FedAvg assumes similar update capability over the same parameter space.",
        },
        "Here, θk,g(t+1) is the update of client k for group g after round t, and αk,g is the normalized weight. Aggregation is therefore grouped by parameter semantics and client capability.": {
            " is the update of client ": " is client ",
            " for group ": "'s group ",
            " after round ": " update after round ",
            " is the normalized weight. Aggregation is therefore grouped by parameter semantics and client capability.": " is the normalized weight. Aggregation follows parameter semantics and client capability.",
        },
        "Here, μ∈[0,1] is EMA momentum, and rimgt and rtxtt are image/text representation statistics at round t. This path does not determine routing and is evaluated in component analysis.": {
            " are image/text representation statistics at round ": " are image/text statistics at round ",
            ". This path does not determine routing and is evaluated in component analysis.": ".",
        },
        "FedSAM3-Hetero initializes global parameters and image/text representations. In each round, clients optimize segmentation alone or segmentation with cross-modal alignment. The server applies restricted routing, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict. After T rounds, the final global model is obtained.": {
            "FedSAM3-Hetero initializes global parameters and image/text representations. In each round, clients optimize segmentation alone or segmentation with cross-modal alignment. The server applies restricted routing, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict. After ": "FedSAM3-Hetero initializes global parameters and image/text representations. Each round, clients optimize segmentation alone or with cross-modal alignment; the server applies restricted routing, updates EMA representations, evaluates Dice/HD95, and computes gradient conflict. After ",
        },
        "Table 5 compares Final Dice, Final HD95, and the average gradient conflict angle for Group C after 30 rounds under different λcream. Increasing λcream from 0.02 to 0.10 and 0.20 does not yield monotonic improvement.": {
            "Table 5 compares Final Dice, Final HD95, and the average gradient conflict angle for Group C after 30 rounds under different ": "Table 5 compares Final Dice, Final HD95, and average gradient conflict for Group C after 30 rounds under different ",
            ". Increasing ": "; ",
            " from 0.02 to 0.10 and 0.20 does not yield monotonic improvement.": " values of 0.02, 0.10, and 0.20 do not yield monotonic improvement.",
        },
        "Distillation strength does not dominate the difference between Groups B and C. The close metrics should not be attributed simply to a small λcream, nor should they be expected to improve by increasing it. Section 4.4 instead points to parameter mixing boundary constraints.": {
            "Distillation strength does not dominate the difference between Groups B and C. The close metrics should not be attributed simply to a small ": "Thus, distillation strength does not dominate the B-C difference; ",
            ", nor should they be expected to improve by increasing it. Section 4.4 instead points to parameter mixing boundary constraints.": " is not the driver, and Section 4.4 instead points to parameter mixing boundary constraints.",
        },
        "Future work may combine routing with FedProx [11], FedAdam [14], and larger heterogeneous multi-center settings.": {
            ", and larger heterogeneous multi-center settings.": ", and larger multi-center settings.",
        },
        "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve the image-only baseline. Under the same client composition, routing does not substantially enlarge the Dice gap but offers limited improvements in Final HD95 and stability-related metrics. Component analysis and minimal ablation support a stabilizing decoupled-collaboration interpretation centered on parameter mixing boundaries, not large main-metric gains. Because FedProx performs better overall, the method is not comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, client compositions, and multi-center datasets.": {
            "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve the image-only baseline. Under the same client composition, routing does not substantially enlarge the Dice gap but offers limited improvements in Final HD95 and stability-related metrics. Component analysis and minimal ablation support a stabilizing decoupled-collaboration interpretation centered on parameter mixing boundaries, not large main-metric gains. Because FedProx performs better overall, the method is not comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, client compositions, and multi-center datasets.": "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation with parameter-level restricted routing as its stable core and global representation update as auxiliary. Groups A-C show that multimodal clients improve the image-only baseline; under the same client composition, routing gives limited HD95 and stability gains rather than a larger Dice gap. Component analysis and minimal ablation support a decoupled-collaboration interpretation centered on parameter mixing boundaries, not large main-metric gains. FedProx remains stronger overall, so the method is not comprehensively superior to robust federated learning. Future work should validate generalization across tasks, client compositions, and multi-center datasets.",
        },
    }

    def repl(match: re.Match[str]) -> str:
        para = match.group(0)
        text = extract_text(para)
        if text in full_para_replacements:
            return para_with_text(para, full_para_replacements[text])
        if text in inline_replacements_by_text:
            return replace_text_values(para, inline_replacements_by_text[text])
        if text.startswith("Fig. 2.") or text.startswith("Fig. 3."):
            return patch_caption_parts(para, caption_replacements)
        if text == "Fig. 4. Endpoint summary of A/B/C/D. Bars show Final Dice and Final HD95 with standard deviation.":
            return para_with_text(para, "Fig. 4. Endpoint Dice and HD95 for A/B/C/D with standard deviation.")
        return para

    return PARA_RE.sub(repl, xml)


def tighten_equation_number_spacing(xml: str) -> str:
    def patch_para(match: re.Match[str]) -> str:
        para = match.group(0)
        text = extract_text(para)
        formula_numbered = (
            text.startswith("Limg=Lseg")
            or text.startswith("wk≥0")
            or text.startswith("αk,g≥0")
            or text.startswith("rimg(t+1)=")
            or text.startswith("Cimg∪Cmm")
            or text.startswith("θt=")
        )
        if not formula_numbered:
            return para

        def patch_text(tmatch: re.Match[str]) -> str:
            full = tmatch.group(0)
            raw = html.unescape(tmatch.group(1))
            if raw.strip() == "" and len(raw) > 1:
                new = ""
            else:
                new = re.sub(r"^\s+\((\d)\)$", r" (\1)", raw)
            return full.replace(tmatch.group(1), esc(new), 1)

        return TEXT_RE.sub(patch_text, para)

    return PARA_RE.sub(patch_para, xml)


def remove_blank_paras_before_table1(xml: str) -> str:
    paras = list(PARA_RE.finditer(xml))
    start = end = None
    for i, match in enumerate(paras):
        text = extract_text(match.group(0))
        if text.startswith("This avoids invalid perturbations"):
            start = i
        if text.startswith("Table 1."):
            end = i
            break
    if start is None or end is None or end <= start:
        return xml

    remove_ranges = [
        (m.start(), m.end())
        for m in paras[start + 1 : end]
        if extract_text(m.group(0)) == ""
    ]
    if not remove_ranges:
        return xml

    parts: list[str] = []
    pos = 0
    for a, b in remove_ranges:
        parts.append(xml[pos:a])
        pos = b
    parts.append(xml[pos:])
    return "".join(parts)


def set_cell_text(cell: str, text: str) -> str:
    para_match = re.search(r"(<w:p\b[^>]*>)(.*?)(</w:p>)", cell, re.DOTALL)
    if not para_match:
        return cell
    p_start, p_body, p_end = para_match.groups()
    ppr_match = re.match(r"(<w:pPr>.*?</w:pPr>)", p_body, re.DOTALL)
    ppr = ppr_match.group(1) if ppr_match else ""
    new_para = (
        f'{p_start}{ppr}<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        f"<w:t>{esc(text)}</w:t></w:r>{p_end}"
    )
    return cell[: para_match.start()] + new_para + cell[para_match.end() :]


def patch_table1(tbl: str) -> str:
    replacements = {
        "Shared visual representation and general segmentation backbone layers": "Shared visual and segmentation backbone layers",
        "Weighted aggregation among eligible clients": "Eligible-client weighted aggregation",
        "Visual encoding and segmentation modules": "Visual encoding and segmentation",
        "Image-text interaction, text adaptation, alignment modules": "Image-text interaction, adaptation, alignment modules",
        "Text semantic injection and consistency modeling": "Text semantics and consistency modeling",
        "Maintain cross-round image semantic anchors": "Cross-round image semantic anchors",
        "Maintain cross-round text semantic anchors": "Cross-round text semantic anchors",
    }

    def repl(match: re.Match[str]) -> str:
        cell = match.group(0)
        text = extract_text(cell)
        if text in replacements:
            return set_cell_text(cell, replacements[text])
        return cell

    tbl = CELL_RE.sub(repl, tbl)
    first_row = re.search(r"<w:tr\b[^>]*>.*?</w:tr>", tbl, re.DOTALL)
    if first_row:
        row = first_row.group(0)
        if "<w:tblHeader" not in row:
            if "<w:trPr>" in row:
                new_row = row.replace("<w:trPr>", "<w:trPr><w:tblHeader/>", 1)
            else:
                start = re.match(r"(<w:tr\b[^>]*>)", row)
                new_row = row
                if start:
                    new_row = row[: start.end()] + "<w:trPr><w:tblHeader/></w:trPr>" + row[start.end() :]
            tbl = tbl[: first_row.start()] + new_row + tbl[first_row.end() :]
    return tbl


def patch_tables(xml: str) -> str:
    table_index = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal table_index
        table_index += 1
        tbl = match.group(0)
        if table_index == 1:
            return patch_table1(tbl)
        return tbl

    return TABLE_RE.sub(repl, xml)


def patch_docm(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.docm")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml = patch_tables(xml)
                xml = patch_paragraphs(xml)
                xml = tighten_equation_number_spacing(xml)
                xml = remove_blank_paras_before_table1(xml)
                data = xml.encode("utf-8")
            out = zipfile.ZipInfo(info.filename, info.date_time)
            out.compress_type = zipfile.ZIP_DEFLATED
            out.external_attr = info.external_attr
            out.comment = info.comment
            zout.writestr(out, data)
    os.replace(tmp, dst)


if __name__ == "__main__":
    patch_docm(Path(sys.argv[1]), Path(sys.argv[2]))
    print("PATCHED")
