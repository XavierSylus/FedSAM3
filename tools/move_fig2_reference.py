from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from lxml import etree


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v": "urn:schemas-microsoft-com:vml",
}


def paragraph_text(p: etree._Element) -> str:
    parts = p.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS)
    return "".join(parts)


def has_image(p: etree._Element) -> bool:
    return bool(
        p.xpath(
            ".//w:drawing | .//w:pict | .//v:imagedata",
            namespaces=NS,
        )
    )


def move_reference(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    with zipfile.ZipFile(src, "r") as zf:
        xml_bytes = zf.read("word/document.xml")
        parser = etree.XMLParser(remove_blank_text=False, recover=False)
        root = etree.fromstring(xml_bytes, parser)
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("word/document.xml has no w:body")

        paragraphs = body.findall("w:p", NS)
        heading_idx = next(
            i
            for i, p in enumerate(paragraphs)
            if paragraph_text(p).strip() == "Training Dynamics Analysis"
        )
        image_idx = next(
            i for i in range(heading_idx + 1, len(paragraphs)) if has_image(paragraphs[i])
        )
        image_p = paragraphs[image_idx]
        ref_matches = [
            i
            for i, p in enumerate(paragraphs)
            if paragraph_text(p).startswith("Fig. 2 supports restricted routing")
        ]
        if len(ref_matches) != 1:
            raise RuntimeError(f"Expected one Fig. 2 reference paragraph, found {len(ref_matches)}")

        ref_idx = ref_matches[0]
        if ref_idx < image_idx:
            raise RuntimeError("Fig. 2 reference already appears before the figure")

        ref_p = paragraphs[ref_idx]
        body.remove(ref_p)
        paragraphs = body.findall("w:p", NS)
        heading_idx = next(
            i
            for i, p in enumerate(paragraphs)
            if paragraph_text(p).strip() == "Training Dynamics Analysis"
        )
        image_idx = next(
            i for i in range(heading_idx + 1, len(paragraphs)) if has_image(paragraphs[i])
        )
        image_p = paragraphs[image_idx]
        body.insert(body.index(image_p), ref_p)

        updated = etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )
        entries = [(info, zf.read(info.filename)) for info in zf.infolist()]

    with zipfile.ZipFile(dst, "w") as zf:
        for info, data in entries:
            if info.filename == "word/document.xml":
                data = updated
            zf.writestr(info, data)

    with zipfile.ZipFile(dst, "r") as zf:
        xml_bytes = zf.read("word/document.xml")
        root = etree.fromstring(xml_bytes)
        body = root.find("w:body", NS)
        paragraphs = body.findall("w:p", NS)
        texts = [paragraph_text(p) for p in paragraphs]
        ref_positions = [
            i for i, text in enumerate(texts) if text.startswith("Fig. 2 supports restricted routing")
        ]
        if len(ref_positions) != 1:
            raise RuntimeError(f"Post-check found {len(ref_positions)} Fig. 2 reference paragraphs")
        ref_idx = ref_positions[0]
        heading_idx = next(i for i, text in enumerate(texts) if text.strip() == "Training Dynamics Analysis")
        image_idx = next(
            i
            for i in range(heading_idx + 1, len(paragraphs))
            if has_image(paragraphs[i])
        )
        if not (heading_idx < ref_idx < image_idx):
            raise RuntimeError("Post-check failed: Fig. 2 reference is not between the heading and figure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    move_reference(args.src, args.dst)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
