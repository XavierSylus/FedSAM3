from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path


def patch_document_xml(xml: str) -> str:
    replacements = [
        (
            "TextBraTS:",
            "Springer Nature Switzerland (2025).",
            "Springer Nature Switzerland (2025). doi:10.1007/978-3-032-04978-0_61.",
        ),
        (
            "SAM 3: Segment Anything with Concepts.",
            "International Conference on Learning Representations (2026).",
            "International Conference on Learning Representations (2026). arXiv preprint arXiv:2511.16719.",
        ),
    ]

    for marker, old, new in replacements:
        pattern = re.compile(r"(<w:p\b[^>]*>.*?" + re.escape(marker) + r".*?</w:p>)", re.DOTALL)
        match = pattern.search(xml)
        if not match:
            raise RuntimeError(f"Could not find reference paragraph containing {marker!r}")
        paragraph = match.group(1)
        if new in paragraph:
            continue
        if old not in paragraph:
            raise RuntimeError(f"Could not find target text {old!r} in paragraph {marker!r}")
        patched = paragraph.replace(old, new, 1)
        xml = xml[: match.start(1)] + patched + xml[match.end(1) :]
    return xml


def patch_docm(path: Path) -> None:
    tmp = path.with_suffix(".tmp.docm")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "word/document.xml":
                data = patch_document_xml(data.decode("utf-8")).encode("utf-8")
            out = zipfile.ZipInfo(info.filename, info.date_time)
            out.compress_type = zipfile.ZIP_DEFLATED
            out.external_attr = info.external_attr
            out.comment = info.comment
            zout.writestr(out, data)
    os.replace(tmp, path)


if __name__ == "__main__":
    patch_docm(Path(sys.argv[1]))
    print("PATCHED")
