from __future__ import annotations

import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


SRC = Path(r"H:\FedSAM3-Cream存档\FedSAM3-Cream26.1.存档\1.0_anon_compressed_work.docm")
DST = Path(r"H:\FedSAM3-Cream存档\FedSAM3-Cream26.1.存档\1.0_anon_compressed.docm")

EMU_EXTENT_CROPS = {
    # PNG rid, SVG rid, original SVG height, top crop in SVG units.
    ("rId6", "rId7"): 40 / 850,
    ("rId8", "rId9"): 28 / 707.801,
    ("rId12", "rId13"): 34 / 293.354,
}

PNG_CROPS = {
    "word/media/image1.png": 38,
    "word/media/image3.png": 32,
    "word/media/image7.png": 93,
}

SVG_CROPS = {
}

SVG_PATHS = {
    "word/media/image2.svg",
    "word/media/image4.svg",
    "word/media/image6.svg",
    "word/media/image8.svg",
}

TEXT_REPLACEMENTS = [
    (
        "Standard FedAvg uniformly averages uploaded client parameters on the server, as shown in Eq. (4):",
        "Standard FedAvg averages uploaded client parameters on the server (Eq. (4)):",
    ),
    (
        "This assumption fails under missing-modality heterogeneity because different clients can update different parameter subspaces reliably. FedSAM3-Hetero applies restricted routing: the server partitions parameters, defines eligible clients for each group, and admits only valid updates. Table 1 maps modules, parameter groups, eligible clients, and server-side processing.",
        "Under missing-modality heterogeneity, different clients reliably update different parameter subspaces. FedSAM3-Hetero therefore partitions parameters, defines eligible clients for each group, and admits only valid updates.",
    ),
    (
        "Let S_g denote the set of clients that are allowed to aggregate the g-th parameter group. The update of this group after round t is written as Eq. (5):",
        "Let S_g denote clients allowed to aggregate the g-th parameter group. Its update after round t is written as Eq. (5):",
    ),
    (
        "If no eligible uploader exists for a parameter group, the server retains the previous global value of that group, as shown in Eq. (6):",
        "If a parameter group has no eligible uploader, the server retains its previous global value (Eq. (6)):",
    ),
    (
        "This rule avoids invalid perturbations when no valid update is available and allows restricted routing to adapt to missing updates across clients and rounds. Table 1 summarizes components, eligible clients, and server-side processing.",
        "This rule avoids invalid perturbations without valid updates and lets routing adapt to missing updates across clients and rounds.",
    ),
    (
        " Mapping among components, parameter groups, eligible clients, and server-side processing.",
        " Components, parameter groups, eligible clients, and server-side processing.",
    ),
    (
        "Shared visual representation layers and general segmentation foundation layers in the backbone",
        "Shared visual representation and general segmentation layers in the backbone",
    ),
    (
        "Provide basic representation capacity shared across clients",
        "Provide shared representation capacity",
    ),
    (
        "Visual encoding and segmentation-related modules",
        "Visual encoding and segmentation modules",
    ),
    (
        "Image representation extraction and segmentation-supervised learning",
        "Image representation extraction and segmentation supervision",
    ),
    (
        "Only clients with visual supervision are allowed to participate in aggregation",
        "Only visually supervised clients aggregate",
    ),
    (
        "Image-text interaction, text adaptation, and cross-modal alignment modules",
        "Image-text interaction, text adaptation, and alignment modules",
    ),
    (
        "Textual semantic injection and cross-modal consistency modeling",
        "Textual semantic injection and consistency modeling",
    ),
    (
        "Only clients with joint image-text optimization capability are allowed to participate in aggregation",
        "Only joint image-text clients aggregate",
    ),
    (
        "Maintains cross-round image semantic anchors",
        "Maintain cross-round image semantic anchors",
    ),
    (
        "Updated through EMA; not included in direct client-side averaging",
        "Updated by EMA; excluded from client-side averaging",
    ),
    (
        "Maintains cross-round text semantic anchors",
        "Maintain cross-round text semantic anchors",
    ),
    (
        "Updated through EMA; not included in global visual parameter aggregation",
        "Updated by EMA; excluded from visual parameter aggregation",
    ),
    (
        "Beyond restricted routing, the server maintains a cross-round global representation path. Parameter aggregation constrains update eligibility, whereas representation statistics maintain semantic references across communication rounds.",
        "Beyond restricted routing, the server maintains cross-round global representations: aggregation constrains update eligibility, while representation statistics retain semantic references.",
    ),
    (
        "After each round, the server updates global image and text representations from uploaded representations using EMA, as written in Eq. (7):",
        "After each round, the server updates global image and text representations from uploads using EMA (Eq. (7)):",
    ),
    (
        "This path does not determine routing; its contribution is evaluated in the component analysis.",
        "This path does not determine routing and is evaluated in component analysis.",
    ),
    (
        "The additional overhead is limited. Restricted routing performs server-side whitelist filtering, while global representation update adds one 768-dimensional vector per client, which is about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload.",
        "The overhead is limited: routing performs server-side whitelist filtering, while global representation update adds one 768-dimensional vector per client, about 3 KB per round, or 0.049% to 0.099% of the 5.95 MB trainable upload.",
    ),
    (
        "FedSAM3-Hetero starts by initializing global parameters and image and text representations. In each communication round, clients optimize either the segmentation objective or the combination of segmentation and cross-modal alignment. The server then applies restricted routing, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict for analysis. After ",
        "FedSAM3-Hetero initializes global parameters and image/text representations. In each round, clients optimize segmentation alone or segmentation with cross-modal alignment. The server applies restricted routing, updates representations through EMA, evaluates Dice and HD95, and computes gradient conflict. After ",
    ),
    (
        " Training dynamics of Groups A, B, and C, including validation Dice, training loss, validation HD95, and gradient conflict.",
        " Training dynamics of Groups A, B, and C: validation Dice, training loss, validation HD95, and gradient conflict.",
    ),
    (
        "Fig. 2 supports the interpretation of restricted routing as a stabilization mechanism rather than a strong optimization intervention. The gradient conflict angles of Groups B and C remain below ",
        "Fig. 2 supports restricted routing as a stabilization mechanism rather than a strong optimization intervention. Groups B and C remain below ",
    ),
    (
        " for most rounds, and Group C shows smoother HD95 trends, which is consistent with better boundary quality.",
        " for most rounds, and Group C shows smoother HD95 trends, consistent with better boundary quality.",
    ),
    (
        "This observation does not prove optimization reshaping or general superiority over robust baselines. It only supports limited improvements in stability and boundary quality under the current setting.",
        "This observation does not prove optimization reshaping or superiority over robust baselines, only limited stability and boundary-quality gains in the current setting.",
    ),
    (
        "Overall, multimodal participation improves the image-only baseline, while restricted routing mainly constrains mixing boundaries and yields limited gains in boundary quality and training stability.",
        "Overall, multimodal participation improves the image-only baseline, while restricted routing mainly constrains mixing boundaries and yields limited boundary-quality and stability gains.",
    ),
    (
        "Global representation update and the ablation of the distillation weight do not separately explain the difference between Groups B and C. The current evidence instead supports parameter mixing boundary constraints as the main explanatory factor.",
        "Global representation update and distillation-weight ablation do not separately explain the B-C difference; the evidence instead points to parameter mixing boundary constraints.",
    ),
    (
        "Fig. 3 compares validation slices. Cases 0043 and 0044 support restricted routing, Case 0046 is marginal, and Case 0049 favors Group B. Therefore, the improvement is sample dependent.",
        "Fig. 3 compares validation slices: Cases 0043 and 0044 support restricted routing, Case 0046 is marginal, and Case 0049 favors Group B, indicating sample-dependent improvement.",
    ),
    (
        "Fig. 4 summarizes endpoint performance. Multimodal participation improves over Group A, while restricted routing mainly reduces HD95 slightly.",
        "Fig. 4 summarizes endpoint performance: multimodal participation improves over Group A, while restricted routing mainly reduces HD95 slightly.",
    ),
    (
        "Fig. 4. Endpoint summary of the A/B/C/D protocols. Bars show Final Dice and Final HD95 with standard deviation error bars.",
        "Fig. 4. Endpoint summary of A/B/C/D. Bars show Final Dice and Final HD95 with standard deviation.",
    ),
    (
        "The current evidence suggests that the method improves stability and interpretability by constraining parameter mixing boundaries. It also provides a reusable baseline for future studies of missing-modality federated segmentation.",
        "The evidence suggests that constraining parameter mixing boundaries improves stability and interpretability and provides a reusable baseline for missing-modality federated segmentation.",
    ),
    (
        "Several limitations remain. The main experiments include only image-only and multimodal clients. Text-only clients are retained as an extensible setting but are not included in quantitative experiments. Therefore, the conclusions mainly apply to settings with image clients and image-text clients.",
        "Several limitations remain. The main experiments include only image-only and multimodal clients; text-only clients are retained as an extensible setting but excluded from quantitative experiments. Thus, the conclusions mainly apply to image-client and image-text-client settings.",
    ),
    (
        "Second, the BraTS-style MRI task requires further validation on more medical segmentation tasks, more client compositions, and larger multi-center datasets.",
        "Second, the BraTS-style MRI task requires validation on more medical segmentation tasks, client compositions, and larger multi-center datasets.",
    ),
    (
        "Third, restricted routing provides only limited gains in boundary quality and training stability, while global representation update remains an auxiliary component under the current setting.",
        "Third, restricted routing yields only limited boundary-quality and stability gains, while global representation update remains auxiliary in the current setting.",
    ),
    (
        "Future work may combine restricted routing with FedProx ",
        "Future work may combine routing with FedProx ",
    ),
    (
        "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation. The more stable core of the framework is parameter-level restricted routing, while global representation update remains auxiliary under the current setting. Results from Groups A-C show that multimodal clients improve the image-only baseline. Under the same client composition, routing does not substantially enlarge the Dice gap, but it provides limited improvements in Final HD95 and training stability-related metrics. Component analysis and the minimal ablation support a stabilizing decoupled-collaboration interpretation centered on parameter mixing boundaries, rather than an interpretation based on large gains in the main metrics. Because FedProx performs better overall, the method is not comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, client compositions, and multi-center datasets.",
        "FedSAM3-Hetero addresses missing-modality heterogeneous federated brain tumor segmentation. Its stable core is parameter-level restricted routing, while global representation update remains auxiliary. Groups A-C show that multimodal clients improve the image-only baseline. Under the same client composition, routing does not substantially enlarge the Dice gap but offers limited improvements in Final HD95 and stability-related metrics. Component analysis and minimal ablation support a stabilizing decoupled-collaboration interpretation centered on parameter mixing boundaries, not large main-metric gains. Because FedProx performs better overall, the method is not comprehensively superior to classical robust federated learning. Future work should validate stability and generalization on more tasks, client compositions, and multi-center datasets.",
    ),
]


def crop_png(data: bytes, top_px: int) -> bytes:
    with Image.open(BytesIO(data)) as img:
        cropped = img.crop((0, top_px, img.width, img.height))
        out = BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()


def crop_svg(data: bytes, top_units: float) -> bytes:
    text = data.decode("utf-8")
    match = re.search(r'viewBox="([^"]+)"', text)
    if not match:
        raise ValueError("SVG viewBox not found")
    parts = [float(p) for p in match.group(1).replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError(f"Unexpected viewBox: {match.group(1)}")
    x, y, width, height = parts
    new_y = y + top_units
    new_height = height - top_units

    def fmt(v: float) -> str:
        return f"{v:.3f}".rstrip("0").rstrip(".")

    new_viewbox = f'{fmt(x)} {fmt(new_y)} {fmt(width)} {fmt(new_height)}'
    text = text[: match.start(1)] + new_viewbox + text[match.end(1) :]
    text = re.sub(r'height="[^"]+"', f'height="{fmt(new_height)}"', text, count=1)
    return text.encode("utf-8")


def update_extents(xml: str) -> str:
    tree = ET.ElementTree(ET.fromstring(xml))
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
    }
    for blip in tree.findall(".//a:blip", ns):
        png_rid = blip.attrib.get(f"{{{ns['r']}}}embed")
        svg_blip = blip.find(".//asvg:svgBlip", ns)
        svg_rid = svg_blip.attrib.get(f"{{{ns['r']}}}embed") if svg_blip is not None else None
        crop_ratio = EMU_EXTENT_CROPS.get((png_rid, svg_rid))
        if crop_ratio is None:
            continue
        node = blip
        while node is not None and not node.tag.endswith("}drawing"):
            node = node.getparent() if hasattr(node, "getparent") else None
        # xml.etree has no parent pointers; use a regex fallback below.
    return xml


def update_extent_regex(xml: str) -> str:
    for (png_rid, svg_rid), crop_ratio in EMU_EXTENT_CROPS.items():
        pattern = re.compile(
            rf'(?P<prefix><a:blip r:embed="{png_rid}">.*?<asvg:svgBlip [^>]*r:embed="{svg_rid}"[^>]*/>.*?</a:blip>.*?<a:ext cx="(?P<cx>\d+)" cy=")(?P<cy>\d+)(?P<suffix>")',
            re.DOTALL,
        )

        def repl(match: re.Match[str]) -> str:
            cy = int(match.group("cy"))
            new_cy = round(cy * (1 - crop_ratio))
            return f"{match.group('prefix')}{new_cy}{match.group('suffix')}"

        xml, count = pattern.subn(repl, xml, count=1)
        if count != 1:
            raise RuntimeError(f"Could not update extent for {png_rid}/{svg_rid}")
    return xml


def remove_svg_extensions(xml: str) -> str:
    return re.sub(
        r'<a:extLst><a:ext uri="\{96DAC541-7B7A-43D3-8B79-37D633B846F1\}"><asvg:svgBlip [^>]*/></a:ext></a:extLst>',
        "",
        xml,
    )


def apply_text_replacements(xml: str) -> str:
    missing: list[str] = []
    for old, new in TEXT_REPLACEMENTS:
        if old not in xml:
            missing.append(old)
            continue
        xml = xml.replace(old, new, 1)
    if missing:
        sample = "\n- ".join(missing[:8])
        raise RuntimeError(f"Missing {len(missing)} replacement targets:\n- {sample}")
    return xml


def patch_docm(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename in SVG_PATHS:
                continue
            data = zin.read(info.filename)
            if info.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml = apply_text_replacements(xml)
                xml = update_extent_regex(xml)
                xml = remove_svg_extensions(xml)
                data = xml.encode("utf-8")
            elif info.filename == "word/_rels/document.xml.rels":
                xml = data.decode("utf-8")
                xml = re.sub(r'<Relationship [^>]*Target="media/image[2468]\.svg"[^>]*/>', "", xml)
                data = xml.encode("utf-8")
            elif info.filename == "[Content_Types].xml":
                xml = data.decode("utf-8")
                xml = re.sub(r'<Default Extension="svg" ContentType="image/svg\+xml"\s*/>', "", xml)
                data = xml.encode("utf-8")
            elif info.filename in PNG_CROPS:
                data = crop_png(data, PNG_CROPS[info.filename])
            elif info.filename in SVG_CROPS:
                data = crop_svg(data, SVG_CROPS[info.filename])
            out_info = zipfile.ZipInfo(info.filename, info.date_time)
            out_info.compress_type = zipfile.ZIP_DEFLATED
            out_info.external_attr = info.external_attr
            out_info.comment = info.comment
            zout.writestr(out_info, data)
    shutil.copystat(src, dst, follow_symlinks=True)


if __name__ == "__main__":
    patch_docm(SRC, DST)
    print(DST)
