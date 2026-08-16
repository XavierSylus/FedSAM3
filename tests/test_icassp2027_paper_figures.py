from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data_processing" / "build_icassp2027_paper_figures.py"


def load_module():
    spec = importlib.util.spec_from_file_location("icassp2027_paper_figures", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperFigureManifestTests(unittest.TestCase):
    def test_build_manifest_reads_pdf_dpi_from_config(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source_document = tmp_path / "source.docm"
            source_document.write_bytes(b"immutable source")
            output_dir = tmp_path / "figures"
            output_dir.mkdir()
            for suffix in ("png", "svg", "pdf"):
                (output_dir / f"figure.{suffix}").write_bytes(suffix.encode("ascii"))

            config = {
                "source_document": str(source_document),
                "output_dir": str(output_dir),
                "raster_pdf_dpi": 300,
                "figures": [
                    {
                        "stem": "figure",
                        "caption_role": "fixture",
                        "document_media": "word/media/image.png",
                        "svg_mode": "bitmap_container",
                    }
                ],
            }

            manifest = module.build_manifest(config, tmp_path / "figure_config.json")

        self.assertEqual(manifest["raster_pdf_dpi"], 300)
        self.assertEqual(manifest["figures"][0]["svg"]["kind"], "bitmap-container")
