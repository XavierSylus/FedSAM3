#!/usr/bin/env python3
"""Build traceable ICASSP 2027 figure assets from the approved source document.

The source document is treated as an immutable Office Open XML archive. This
script never opens or executes its VBA project; it reads only named media parts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "paper_submission" / "icassp2027_fedsam3_hetero" / "figure_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the manuscript figure PNG, SVG, and PDF assets from approved static sources."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figure", help="Generate one configured figure, producing exactly PNG, SVG, and PDF.")
    parser.add_argument("--write-manifest", action="store_true", help="Write the manifest after all figure assets exist.")
    return parser.parse_args()


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(output_dir: Path, targets: list[Path]) -> None:
    if any(target.exists() for target in targets):
        existing = next(target for target in targets if target.exists())
        raise FileExistsError(f"Refusing to overwrite existing figure asset: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_bitmap_container_svg(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def write_raster_pdf(png_path: Path, pdf_path: Path, dpi: int) -> None:
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=float(dpi))


def extract_media(document_path: Path, media_path: str, output_path: Path) -> None:
    with zipfile.ZipFile(document_path) as archive:
        try:
            data = archive.read(media_path)
        except KeyError as error:
            raise FileNotFoundError(f"Missing media part {media_path} in {document_path}") from error
    output_path.write_bytes(data)


def build_figure(config: dict[str, Any], config_path: Path, figure: dict[str, Any]) -> None:
    source_document = resolve_path(config["source_document"], config_path)
    if not source_document.is_file():
        raise FileNotFoundError(f"Missing immutable source document: {source_document}")

    output_dir = resolve_path(config["output_dir"], config_path)
    dpi = int(config["raster_pdf_dpi"])
    stem = figure["stem"]
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    prepare_output(output_dir, [png_path, svg_path, pdf_path])
    extract_media(source_document, figure["document_media"], png_path)

    if figure["svg_mode"] == "pure_vector_copy":
        vector_source = resolve_path(figure["vector_source"], config_path)
        if not vector_source.is_file():
            raise FileNotFoundError(f"Missing vector source for {stem}: {vector_source}")
        shutil.copyfile(vector_source, svg_path)
    elif figure["svg_mode"] == "bitmap_container":
        write_bitmap_container_svg(png_path, svg_path)
    else:
        raise ValueError(f"Unsupported svg_mode for {stem}: {figure['svg_mode']}")
    write_raster_pdf(png_path, pdf_path, dpi)


def build_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    source_document = resolve_path(config["source_document"], config_path)
    output_dir = resolve_path(config["output_dir"], config_path)
    dpi = int(config["raster_pdf_dpi"])
    records: list[dict[str, Any]] = []
    for figure in config["figures"]:
        stem = figure["stem"]
        png_path = output_dir / f"{stem}.png"
        svg_path = output_dir / f"{stem}.svg"
        pdf_path = output_dir / f"{stem}.pdf"
        if not all(path.is_file() for path in (png_path, svg_path, pdf_path)):
            raise FileNotFoundError(f"Cannot write manifest until all assets exist for {stem}")
        svg_source = (
            str(resolve_path(figure["vector_source"], config_path))
            if figure["svg_mode"] == "pure_vector_copy"
            else str(source_document)
        )
        records.append(
            {
                "stem": stem,
                "caption_role": figure["caption_role"],
                "source_document_media": figure["document_media"],
                "png": {"path": str(png_path), "sha256": sha256(png_path)},
                "svg": {
                    "path": str(svg_path),
                    "sha256": sha256(svg_path),
                    "kind": "pure-vector" if figure["svg_mode"] == "pure_vector_copy" else "bitmap-container",
                    "source": svg_source,
                },
                "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            }
        )
    return {
        "status": "generated",
        "source_document": str(source_document),
        "source_document_sha256": sha256(source_document),
        "macro_execution": "not performed; ZIP media parts were read statically",
        "raster_pdf_dpi": dpi,
        "figures": records,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = {figure["stem"]: figure for figure in config["figures"]}
    if args.figure:
        if args.figure not in configured:
            raise ValueError(f"Unknown figure: {args.figure}. Choose one of {sorted(configured)}")
        build_figure(config, config_path, configured[args.figure])
        print(args.figure)
    elif args.write_manifest:
        manifest_path = resolve_path(config["manifest_path"], config_path)
        manifest = build_manifest(config, config_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(manifest_path)
    else:
        raise ValueError("Specify exactly one of --figure or --write-manifest")


if __name__ == "__main__":
    main()
