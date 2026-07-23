from __future__ import annotations

import html
import os
import re
import sys
import zipfile
from pathlib import Path


W_PARA_RE = re.compile(r"<w:p\b[^>]*>.*?</w:p>", re.DOTALL)
TEXT_RE = re.compile(r"<(?:w|m):t(?:\s[^>]*)?>(.*?)</(?:w|m):t>", re.DOTALL)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def mr(text: str, *, roman: bool = False, script: bool = False) -> str:
    props = ""
    if roman:
        props = "<m:rPr><m:sty m:val=\"p\"/></m:rPr>"
    elif script:
        props = "<m:rPr><m:scr m:val=\"script\"/></m:rPr>"
    else:
        props = "<m:rPr/>"
    return f"<m:r>{props}<m:t>{esc(text)}</m:t></m:r>"


def sub(base: str, subscript: str) -> str:
    return (
        "<m:sSub><m:sSubPr/>"
        f"<m:e>{base}</m:e>"
        f"<m:sub>{subscript}</m:sub>"
        "</m:sSub>"
    )


def sup(base: str, superscript: str) -> str:
    return (
        "<m:sSup><m:sSupPr/>"
        f"<m:e>{base}</m:e>"
        f"<m:sup>{superscript}</m:sup>"
        "</m:sSup>"
    )


def subsup(base: str, subscript: str, superscript: str) -> str:
    return (
        "<m:sSubSup><m:sSubSupPr/>"
        f"<m:e>{base}</m:e>"
        f"<m:sub>{subscript}</m:sub>"
        f"<m:sup>{superscript}</m:sup>"
        "</m:sSubSup>"
    )


def nary(subscript: str, superscript: str | None, body: str) -> str:
    sup_part = f"<m:sup>{superscript}</m:sup>" if superscript is not None else ""
    sup_hide = "<m:supHide m:val=\"1\"/>" if superscript is None else ""
    return (
        "<m:nary><m:naryPr><m:chr m:val=\"∑\"/><m:limLoc m:val=\"subSup\"/>"
        f"{sup_hide}</m:naryPr>"
        f"<m:sub>{subscript}</m:sub>"
        f"{sup_part}"
        f"<m:e>{body}</m:e>"
        "</m:nary>"
    )


def bar(expr: str) -> str:
    return f"<m:bar><m:barPr/><m:e>{expr}</m:e></m:bar>"


def omath(children: str) -> str:
    return f"<m:oMath>{children}</m:oMath>"


def wr(text: str, *, preserve: bool = True) -> str:
    space = " xml:space=\"preserve\"" if preserve else ""
    return f"<w:r><w:t{space}>{esc(text)}</w:t></w:r>"


def extract_text(para_xml: str) -> str:
    return "".join(html.unescape(m.group(1)) for m in TEXT_RE.finditer(para_xml)).strip()


def get_prefix_and_ppr(para_xml: str) -> tuple[str, str]:
    start_match = re.match(r"(<w:p\b[^>]*>)", para_xml)
    if not start_match:
        raise RuntimeError("Invalid paragraph XML")
    start = start_match.group(1)
    rest = para_xml[start_match.end() :]
    ppr_match = re.match(r"(<w:pPr>.*?</w:pPr>)", rest, re.DOTALL)
    ppr = ppr_match.group(1) if ppr_match else ""
    return start, ppr


def replace_para_content(para_xml: str, content: str) -> str:
    start, ppr = get_prefix_and_ppr(para_xml)
    return f"{start}{ppr}{content}</w:p>"


def replace_para_with_contents(para_xml: str, contents: list[str]) -> str:
    start, ppr = get_prefix_and_ppr(para_xml)
    return "".join(f"{start}{ppr}{content}</w:p>" for content in contents)


def eq_run(num: int, spaces: int = 6) -> str:
    return wr(f"{' ' * spaces}({num})")


CAL_C = mr("C", script=True)
THETA = mr("θ")


def theta_subsup(name: str, time: str = "t") -> str:
    return subsup(THETA, mr(name, roman=True), mr(time))


def r_subsup(name: str, time: str, *, barred: bool = False) -> str:
    base = mr("r")
    if barred:
        base = bar(base)
    return subsup(base, mr(name, roman=True), mr(time))


def l_sub(name: str) -> str:
    return sub(mr("L"), mr(name, roman=True))


def lambda_sub(name: str) -> str:
    return sub(mr("λ"), mr(name, roman=True))


def equation_1_line1() -> str:
    c_img = sub(CAL_C, mr("img", roman=True))
    c_mm = sub(CAL_C, mr("mm", roman=True))
    c_i = sub(mr("c"), mr("i"))
    m_i = sub(mr("m"), mr("i"))
    set_img = c_img + mr("={") + c_i + mr("∈") + CAL_C + mr("|") + m_i + mr("=") + mr("img", roman=True) + mr("}")
    set_mm = c_mm + mr("={") + c_i + mr("∈") + CAL_C + mr("|") + m_i + mr("=") + mr("mm", roman=True) + mr("}")
    return omath(set_img + mr(", ") + set_mm + mr(","))


def equation_1_line2() -> str:
    c_img = sub(CAL_C, mr("img", roman=True))
    c_mm = sub(CAL_C, mr("mm", roman=True))
    disjoint = c_img + mr("∪") + c_mm + mr("⊆") + CAL_C + mr(", ") + c_img + mr("∩") + c_mm + mr("=∅.")
    return omath(disjoint) + eq_run(1)


def equation_2() -> str:
    formula = (
        sup(THETA, mr("t"))
        + mr("={")
        + theta_subsup("shared")
        + mr(", ")
        + theta_subsup("img")
        + mr(", ")
        + theta_subsup("mm")
        + mr("}.")
    )
    return omath(formula) + eq_run(2)


def equation_3() -> str:
    formula = (
        l_sub("img")
        + mr("=")
        + l_sub("seg")
        + mr(", ")
        + l_sub("mm")
        + mr("=")
        + l_sub("seg")
        + mr("+")
        + lambda_sub("cream")
        + l_sub("cream")
        + mr("+")
        + lambda_sub("align")
        + l_sub("align")
        + mr(". (3)")
    )
    return omath(formula)


def equation_4_lines() -> tuple[str, str]:
    term = sub(mr("w"), mr("k")) + sup(sub(THETA, mr("k")), mr("(t+1)"))
    norm = sub(mr("w"), mr("k")) + mr("≥0, ") + nary(mr("k=1"), mr("N"), sub(mr("w"), mr("k"))) + mr("=1.")
    formula = sup(THETA, mr("(t+1)")) + mr("=") + nary(mr("k=1"), mr("N"), term) + mr(",")
    return formula, norm


def equation_5_lines() -> tuple[str, str]:
    alpha = sub(mr("α"), mr("k,g"))
    # Use a direct subscripted/superscripted theta for the term to avoid nested empty superscripts.
    term = alpha + subsup(THETA, mr("k,g"), mr("(t+1)"))
    norm = alpha + mr("≥0, ") + nary(mr("k∈") + sub(mr("S"), mr("g")), None, alpha) + mr("=1.")
    formula = subsup(THETA, mr("g"), mr("(t+1)")) + mr("=") + nary(mr("k∈") + sub(mr("S"), mr("g")), None, term) + mr(",")
    return formula, norm


def equation_7() -> str:
    mu = mr("μ")
    img_update = (
        r_subsup("img", "(t+1)")
        + mr("=")
        + mu
        + r_subsup("img", "t")
        + mr("+(1−")
        + mu
        + mr(")")
        + r_subsup("img", "t", barred=True)
    )
    txt_update = (
        r_subsup("txt", "(t+1)")
        + mr("=")
        + mu
        + r_subsup("txt", "t")
        + mr("+(1−")
        + mu
        + mr(")")
        + r_subsup("txt", "t", barred=True)
    )
    formula = img_update + mr(", ") + txt_update + mr(", ") + mu + mr("∈[0,1].")
    return omath(formula) + eq_run(7, spaces=1)


def explanation_3(para_xml: str) -> str:
    start, ppr = get_prefix_and_ppr(para_xml)
    content = (
        wr("Here, ")
        + omath(l_sub("seg"))
        + wr(" is segmentation loss, ")
        + omath(l_sub("cream"))
        + wr(" is the Cream consistency loss, ")
        + omath(l_sub("align"))
        + wr(" is image-text alignment loss, and ")
        + omath(lambda_sub("cream"))
        + wr(" and ")
        + omath(lambda_sub("align"))
        + wr(" are balancing coefficients. Thus, heterogeneous clients differ in local objectives and reliable parameter influence.")
    )
    return f"{start}{ppr}{content}</w:p>"


def explanation_7(para_xml: str) -> str:
    start, ppr = get_prefix_and_ppr(para_xml)
    content = (
        wr("Here, ")
        + omath(mr("μ∈[0,1]"))
        + wr(" is EMA momentum, and ")
        + omath(r_subsup("img", "t"))
        + wr(" and ")
        + omath(r_subsup("txt", "t"))
        + wr(" are image/text representation statistics at round ")
        + omath(mr("t"))
        + wr(". This path does not determine routing and is evaluated in component analysis.")
    )
    return f"{start}{ppr}{content}</w:p>"


def patch_document_xml(xml: str) -> str:
    replacements = 0

    intro_end_match = re.search(r"satisfying Eq\. \(1\):</w:t></w:r></w:p>", xml)
    if not intro_end_match:
        raise RuntimeError("Could not find Eq. (1) intro endpoint")
    eq1_start = intro_end_match.end()
    tail = xml[eq1_start:]
    next_match = None
    for candidate in W_PARA_RE.finditer(tail):
        if extract_text(candidate.group(0)).startswith("Let the global model parameters at communication round"):
            next_match = candidate
            break
    if next_match is None:
        raise RuntimeError("Could not find paragraph after Eq. (1)")
    eq1_end = eq1_start + next_match.start()
    old_block = xml[eq1_start:eq1_end]
    ppr_match = re.search(r"<w:pPr>.*?</w:pPr>", old_block, re.DOTALL)
    ppr = ppr_match.group(0) if ppr_match else '<w:pPr><w:pStyle w:val="3"/></w:pPr>'
    line1 = f"<w:p>{ppr}{equation_1_line1()}</w:p>"
    line2 = f"<w:p>{ppr}{equation_1_line2()}</w:p>"
    xml = xml[:eq1_start] + line1 + line2 + xml[eq1_end:]
    replacements += 1

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        para = match.group(0)
        text = extract_text(para)
        if text.startswith("θt=θsharedt") and "(2)" in text:
            replacements += 1
            return replace_para_content(para, equation_2())
        if text.startswith("Limg=Lseg") and "(3)" in text:
            replacements += 1
            return replace_para_content(para, equation_3())
        if text.startswith("Here, Lseg is segmentation loss") and "λc" in text:
            replacements += 1
            return explanation_3(para)
        if text.startswith("θ(t+1)=") and "(4)" in text:
            replacements += 1
            main, norm = equation_4_lines()
            return replace_para_with_contents(para, [omath(main), omath(norm) + eq_run(4)])
        if text.startswith("θg(t+1)=") and "(5)" in text:
            replacements += 1
            main, norm = equation_5_lines()
            return replace_para_with_contents(para, [omath(main), omath(norm) + eq_run(5)])
        if text.startswith("rimg(t+1)=") and "(7)" in text:
            replacements += 1
            return replace_para_content(para, equation_7())
        if text.startswith("Here, m is EMA momentum"):
            replacements += 1
            return explanation_7(para)
        return para

    xml = W_PARA_RE.sub(repl, xml)
    if replacements != 8:
        raise RuntimeError(f"Expected 8 replacements, applied {replacements}")
    return xml


def patch_docm(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.docm")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "word/document.xml":
                data = patch_document_xml(data.decode("utf-8")).encode("utf-8")
            out = zipfile.ZipInfo(info.filename, info.date_time)
            out.compress_type = zipfile.ZIP_DEFLATED
            out.external_attr = info.external_attr
            out.comment = info.comment
            zout.writestr(out, data)
    os.replace(tmp, dst)


if __name__ == "__main__":
    patch_docm(Path(sys.argv[1]), Path(sys.argv[2]))
    print("PATCHED")
