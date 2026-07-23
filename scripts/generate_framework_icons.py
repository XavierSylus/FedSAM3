from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "framework_icons_config.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def attrs(**values: object) -> str:
    rendered = []
    for key, value in values.items():
        if value is None:
            continue
        rendered.append(f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"')
    return " ".join(rendered)


def tag(name: str, /, **values: object) -> str:
    return f"<{name} {attrs(**values)}/>"


def snowflake_icon(colors: dict[str, str], stroke: dict[str, float]) -> list[str]:
    main = colors["snowflake"]
    lines = [
        tag("circle", cx=50, cy=50, r=46, fill=colors["snowflake_background"], stroke=colors["snowflake_border"], stroke_width=1.6),
        tag("line", x1=50, y1=15, x2=50, y2=85, stroke=main, stroke_width=stroke["snowflake"], stroke_linecap="round"),
        tag("line", x1=19.7, y1=32.5, x2=80.3, y2=67.5, stroke=main, stroke_width=stroke["snowflake"], stroke_linecap="round"),
        tag("line", x1=19.7, y1=67.5, x2=80.3, y2=32.5, stroke=main, stroke_width=stroke["snowflake"], stroke_linecap="round"),
        tag("circle", cx=50, cy=50, r=5.8, fill=main),
    ]
    branch_specs = [
        (50, 25, 38, 18),
        (50, 25, 62, 18),
        (50, 75, 38, 82),
        (50, 75, 62, 82),
        (29, 38, 25, 25),
        (29, 38, 16, 42),
        (71, 62, 84, 58),
        (71, 62, 75, 75),
        (29, 62, 16, 58),
        (29, 62, 25, 75),
        (71, 38, 75, 25),
        (71, 38, 84, 42),
    ]
    for x1, y1, x2, y2 in branch_specs:
        lines.append(tag("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=main, stroke_width=stroke["snowflake_branch"], stroke_linecap="round"))
    return lines


def flame_icon(colors: dict[str, str]) -> list[str]:
    return [
        tag("circle", cx=50, cy=50, r=46, fill="#fff7ed", stroke=colors["flame"], stroke_width=1.6),
        tag(
            "path",
            d=(
                "M51 83 "
                "C34 77 28 64 32 50 "
                "C35 39 43 34 43 22 "
                "C54 30 58 39 57 48 "
                "C64 42 67 35 66 27 "
                "C79 43 82 59 73 72 "
                "C68 79 61 83 51 83 Z"
            ),
            fill=colors["flame"],
        ),
        tag(
            "path",
            d=(
                "M49 78 "
                "C39 73 36 64 39 56 "
                "C41 50 47 46 47 37 "
                "C55 45 58 52 55 60 "
                "C60 56 63 51 63 45 "
                "C70 56 68 69 60 75 "
                "C57 78 53 79 49 78 Z"
            ),
            fill=colors["flame_deep"],
        ),
        tag(
            "path",
            d="M48 73 C43 69 43 62 48 57 C51 63 57 66 54 73 C52 76 50 76 48 73 Z",
            fill=colors["flame_light"],
        ),
    ]


def mri_slice_icon(colors: dict[str, str], stroke: dict[str, float]) -> list[str]:
    detail = stroke["mri_detail"]
    return [
        tag("rect", x=18, y=10, width=64, height=80, rx=8, fill=colors["mri_background"]),
        tag("ellipse", cx=50, cy=50, rx=24, ry=33, fill=colors["mri_brain_outer"]),
        tag("ellipse", cx=39, cy=50, rx=13.5, ry=26, fill=colors["mri_brain_inner"]),
        tag("ellipse", cx=61, cy=50, rx=13.5, ry=26, fill=colors["mri_brain_inner"]),
        tag("path", d="M50 18 C44 30 44 42 50 50 C56 58 56 70 50 82", fill="none", stroke=colors["mri_shadow"], stroke_width=2.4, stroke_linecap="round"),
        tag("path", d="M38 30 C45 36 35 44 43 51 C47 55 43 65 36 70", fill="none", stroke=colors["mri_shadow"], stroke_width=detail, stroke_linecap="round"),
        tag("path", d="M62 30 C55 37 65 45 57 52 C53 57 57 66 64 70", fill="none", stroke=colors["mri_shadow"], stroke_width=detail, stroke_linecap="round"),
        tag("path", d="M32 43 C39 41 42 46 39 52 C36 57 39 62 45 64", fill="none", stroke=colors["mri_line"], stroke_width=detail, stroke_linecap="round"),
        tag("path", d="M68 43 C61 41 58 46 61 52 C64 57 61 62 55 64", fill="none", stroke=colors["mri_line"], stroke_width=detail, stroke_linecap="round"),
        tag("path", d="M36 22 C29 30 28 42 31 53 C34 66 34 74 30 82", fill="none", stroke="#5a5a5a", stroke_width=1.1, stroke_linecap="round", opacity=0.75),
        tag("path", d="M64 22 C71 30 72 42 69 53 C66 66 66 74 70 82", fill="none", stroke="#5a5a5a", stroke_width=1.1, stroke_linecap="round", opacity=0.75),
    ]


def mask_icon(colors: dict[str, str]) -> list[str]:
    return [
        tag("rect", x=18, y=10, width=64, height=80, rx=8, fill=colors["mask_background"]),
        tag(
            "path",
            d=(
                "M52 82 "
                "C40 80 32 71 33 60 "
                "C34 52 40 47 42 40 "
                "C45 30 43 23 51 17 "
                "C55 26 62 31 65 41 "
                "C68 52 62 59 64 66 "
                "C66 75 61 82 52 82 Z"
            ),
            fill=colors["mask_region"],
        ),
        tag(
            "path",
            d="M45 39 C50 48 42 56 47 66 C51 73 57 75 60 70",
            fill="none",
            stroke="#d9d9d9",
            stroke_width=1.2,
            stroke_linecap="round",
            opacity=0.55,
        ),
    ]


def server_icon(colors: dict[str, str]) -> list[str]:
    rows = []
    for index, y in enumerate([23, 39, 55, 71]):
        rows.extend(
            [
                tag("rect", x=26, y=y, width=48, height=13, rx=3, fill=colors["server_shadow"], opacity=0.28),
                tag("rect", x=24, y=y - 2, width=52, height=13, rx=3, fill=colors["server_body"]),
                tag("rect", x=30, y=y + 2, width=16, height=3, rx=1.5, fill=colors["server_face"]),
                tag("circle", cx=64, cy=y + 4.5, r=2, fill=colors["server_light_green"] if index % 2 == 0 else colors["server_light_orange"]),
                tag("circle", cx=70, cy=y + 4.5, r=2, fill="#d1d5db"),
            ]
        )
    return [
        tag("circle", cx=50, cy=50, r=46, fill="#f8fafc", stroke="#94a3b8", stroke_width=1.6),
        *rows,
        tag("path", d="M50 84 L50 91", fill="none", stroke=colors["server_shadow"], stroke_width=4, stroke_linecap="round"),
        tag("path", d="M38 91 L62 91", fill="none", stroke=colors["server_shadow"], stroke_width=4, stroke_linecap="round"),
    ]


def local_client_icon(colors: dict[str, str], accent: str, background: str) -> list[str]:
    return [
        tag("circle", cx=50, cy=50, r=46, fill=background, stroke=accent, stroke_width=1.6),
        tag("rect", x=25, y=29, width=50, height=34, rx=5, fill=colors["client_screen"], stroke=accent, stroke_width=3),
        tag("rect", x=31, y=35, width=38, height=20, rx=2, fill=background, stroke=accent, stroke_width=1.6, opacity=0.92),
        tag("path", d="M45 65 L55 65 L57 75 L43 75 Z", fill=accent, opacity=0.9),
        tag("path", d="M35 78 L65 78", fill="none", stroke=accent, stroke_width=4, stroke_linecap="round"),
        tag("circle", cx=50, cy=44, r=6, fill=accent),
        tag("path", d="M38 56 C41 50 59 50 62 56", fill="none", stroke=accent, stroke_width=3.2, stroke_linecap="round"),
    ]


def icon_registry(config: dict[str, Any]) -> dict[str, dict[str, object]]:
    colors = config["colors"]
    stroke = config["stroke"]
    return {
        "snowflake": {
            "title": "Frozen snowflake icon",
            "desc": "Original vector snowflake icon for frozen parameter groups.",
            "body": snowflake_icon(colors, stroke),
        },
        "flame": {
            "title": "Learnable flame icon",
            "desc": "Original vector flame icon for learnable parameter groups.",
            "body": flame_icon(colors),
        },
        "mri_slice": {
            "title": "MRI slice icon",
            "desc": "Original vector MRI slice icon drawn from abstract geometry, not from patient data.",
            "body": mri_slice_icon(colors, stroke),
        },
        "mask": {
            "title": "Segmentation mask icon",
            "desc": "Original vector segmentation mask icon drawn from abstract geometry, not from a model output.",
            "body": mask_icon(colors),
        },
        "server": {
            "title": "Server icon",
            "desc": "Original vector server icon for aggregation and routing diagrams.",
            "body": server_icon(colors),
        },
        "local_client_blue": {
            "title": "Blue local client icon",
            "desc": "Original vector local client icon in blue.",
            "body": local_client_icon(colors, colors["client_blue"], colors["client_blue_background"]),
        },
        "local_client_green": {
            "title": "Green local client icon",
            "desc": "Original vector local client icon in green.",
            "body": local_client_icon(colors, colors["client_green"], colors["client_green_background"]),
        },
        "local_client_light_gray": {
            "title": "Light gray local client icon",
            "desc": "Original vector local client icon in light gray.",
            "body": local_client_icon(colors, colors["client_gray"], colors["client_gray_background"]),
        },
    }


def group(name: str, x: float, y: float, scale: float, body: list[str]) -> str:
    content = "\n    ".join(body)
    return f'  <g id="{name}" transform="translate({x:g} {y:g}) scale({scale:g})">\n    {content}\n  </g>'


def build_svg(config: dict[str, Any]) -> str:
    canvas = config["canvas"]
    icon_size = float(canvas["icon_size"])
    padding = float(canvas["padding"])
    gap = float(canvas["gap"])
    scale = icon_size / 100.0
    width = int((padding * 2) + (icon_size * 3) + (gap * 2))
    height = int((padding * 2) + icon_size)
    positions = [padding, padding + icon_size + gap, padding + (icon_size + gap) * 2]
    registry = icon_registry(config)
    body = [
        group("icon-snowflake", positions[0], padding, scale, registry["snowflake"]["body"]),
        group("icon-flame", positions[1], padding, scale, registry["flame"]["body"]),
        group("icon-mri-slice", positions[2], padding, scale, registry["mri_slice"]["body"]),
    ]
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            "  <title id=\"title\">FedSAM3 framework icons</title>",
            "  <desc id=\"desc\">Original vector icons for frozen parameters, learnable parameters, and an MRI slice. The artwork uses only generated SVG primitives and contains no external assets.</desc>",
            *body,
            "</svg>",
            "",
        ]
    )


def build_standalone_svg(title: str, desc: str, body: list[str]) -> str:
    content = "\n  ".join(body)
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100" role="img" aria-labelledby="title desc">',
            f'  <title id="title">{escape(title)}</title>',
            f'  <desc id="desc">{escape(desc)}</desc>',
            f"  {content}",
            "</svg>",
            "",
        ]
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate original SVG icons for FedSAM3 framework diagrams.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--standalone-only", action="store_true", help="Generate only standalone icon SVG files.")
    parser.add_argument("--sheet-only", action="store_true", help="Generate only the combined icon sheet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    written = []
    if args.sheet_only and args.standalone_only:
        raise ValueError("--sheet-only and --standalone-only cannot be used together.")
    if not args.standalone_only:
        output = args.output or Path(config["output"]["svg_path"])
        output_path = resolve_output(output)
        write_text(output_path, build_svg(config))
        written.append(output_path)
    if not args.sheet_only:
        registry = icon_registry(config)
        for name, raw_path in config["output"]["standalone"].items():
            icon = registry[name]
            output_path = resolve_output(Path(raw_path))
            write_text(
                output_path,
                build_standalone_svg(
                    str(icon["title"]),
                    str(icon["desc"]),
                    icon["body"],
                ),
            )
            written.append(output_path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
