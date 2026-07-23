from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path


SENTENCE = (
    " Solid lines denote the mean over three random seeds, "
    "and shaded regions indicate ±1 standard deviation."
)


def patch_document_xml(xml: str) -> str:
    sentence_run = f'<w:r><w:t xml:space="preserve">{SENTENCE}</w:t></w:r>'
    # Remove an earlier misplaced insertion, if present.
    xml = xml.replace(sentence_run, "")

    if SENTENCE.strip() in xml:
        return xml

    target = (
        '<w:t xml:space="preserve"> Training dynamics of Groups A, B, and C: '
        'validation Dice, training loss, validation HD95, and gradient conflict.</w:t>'
    )
    replacement = (
        '<w:t xml:space="preserve"> Training dynamics of Groups A, B, and C: '
        f'validation Dice, training loss, validation HD95, and gradient conflict.{SENTENCE}</w:t>'
    )
    if target not in xml:
        raise RuntimeError("Could not find Fig. 2 caption text")
    return xml.replace(target, replacement, 1)


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
