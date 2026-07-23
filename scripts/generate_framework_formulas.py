from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "framework_formulas_config.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def render_formula(formula: dict[str, str], render: dict[str, Any]) -> Path:
    output_path = resolve_path(Path(formula["output"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["svg.fonttype"] = render["svg_fonttype"]
    fig = plt.figure(
        figsize=(render["figure_width_inches"], render["figure_height_inches"]),
        dpi=render["dpi"],
    )
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        formula["latex"],
        ha="center",
        va="center",
        fontsize=render["font_size"],
        color=render["text_color"],
    )
    fig.savefig(
        output_path,
        format="svg",
        transparent=render["background"] == "none",
        bbox_inches="tight",
        pad_inches=render["padding_inches"],
        metadata={"Date": None},
    )
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone SVG formula assets for the framework figure.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    for formula in config["formulas"]:
        print(render_formula(formula, config["render"]))


if __name__ == "__main__":
    main()
