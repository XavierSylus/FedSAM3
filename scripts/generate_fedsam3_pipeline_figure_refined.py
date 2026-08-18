from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fedsam3_pipeline_figure_refined_config.json"


def configure_matplotlib(config: dict[str, Any]) -> None:
    style = config["style"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": style["font_family"],
            "font.size": style["font_size"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def setup_canvas(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 6.0,
    weight: str = "normal",
    color: str = "#20252B",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.12,
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        family="sans-serif",
        zorder=10,
    )


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    edge: str,
    lw: float = 0.75,
    radius: float = 0.009,
    dashed: bool = False,
    zorder: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        mutation_aspect=1,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 0.85,
    head: float = 8.0,
    dashed: bool = False,
    connection: str = "arc3",
    zorder: float = 3.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=lw,
            color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def panel(
    ax: plt.Axes,
    box: tuple[float, float, float, float],
    title: str,
    *,
    edge: str,
    fill: str,
    accent: str,
) -> None:
    x, y, width, height = box
    rounded(ax, x, y, width, height, fill=fill, edge=edge, lw=0.8, radius=0.012, zorder=0.5)
    ax.plot(
        [x + 0.012, x + width - 0.012],
        [y + height - 0.052, y + height - 0.052],
        color=accent,
        lw=1.4,
        solid_capstyle="round",
        zorder=2,
    )
    label(ax, x + 0.014, y + height - 0.027, title, size=8.1, weight="bold")


def draw_model_icon(ax: plt.Axes, x: float, y: float, width: float, height: float, *, accent: str, edge: str) -> None:
    for offset in (0.010, 0.005, 0.000):
        rounded(
            ax,
            x + offset,
            y + offset,
            width - 0.010,
            height - 0.010,
            fill="white",
            edge=edge,
            lw=0.55,
            radius=0.004,
            zorder=2,
        )
    nodes = [
        (x + width * 0.28, y + height * 0.32),
        (x + width * 0.28, y + height * 0.68),
        (x + width * 0.53, y + height * 0.50),
        (x + width * 0.77, y + height * 0.32),
        (x + width * 0.77, y + height * 0.68),
    ]
    for left in nodes[:2]:
        for right in nodes[2:3]:
            ax.plot([left[0], right[0]], [left[1], right[1]], color=accent, lw=0.55, zorder=3)
    for left in nodes[2:3]:
        for right in nodes[3:]:
            ax.plot([left[0], right[0]], [left[1], right[1]], color=accent, lw=0.55, zorder=3)
    for node_x, node_y in nodes:
        ax.add_patch(Circle((node_x, node_y), width * 0.055, facecolor="white", edgecolor=accent, linewidth=0.6, zorder=4))


def draw_document_icon(ax: plt.Axes, x: float, y: float, width: float, height: float, *, accent: str, edge: str) -> None:
    for index in range(3):
        offset = index * width * 0.12
        rounded(
            ax,
            x + offset,
            y + offset * 0.55,
            width * 0.72,
            height * 0.78,
            fill="white",
            edge=edge,
            lw=0.5,
            radius=0.003,
            zorder=2 + index,
        )
    front_x = x + width * 0.24
    front_y = y + width * 0.13
    for fraction in (0.66, 0.48, 0.30):
        ax.plot(
            [front_x + width * 0.12, front_x + width * 0.52],
            [front_y + height * fraction, front_y + height * fraction],
            color=accent,
            lw=1.0,
            solid_capstyle="round",
            zorder=6,
        )


def draw_mri_icon(ax: plt.Axes, x: float, y: float, width: float, height: float, *, accent: str, edge: str) -> None:
    for index in range(3):
        offset = index * width * 0.10
        rounded(
            ax,
            x + offset,
            y + offset * 0.55,
            width * 0.74,
            height * 0.78,
            fill="#F4F5F6",
            edge=edge,
            lw=0.5,
            radius=0.003,
            zorder=2 + index,
        )
    cx = x + width * 0.56
    cy = y + height * 0.48
    ax.add_patch(Ellipse((cx, cy), width * 0.44, height * 0.46, facecolor="#D5D8DB", edgecolor="#747B82", linewidth=0.5, zorder=6))
    ax.add_patch(Ellipse((cx, cy), width * 0.24, height * 0.25, facecolor="#BFC4C9", edgecolor="#8B9299", linewidth=0.35, zorder=7))
    ax.plot([cx, cx], [cy - height * 0.19, cy + height * 0.19], color="#858C93", lw=0.45, zorder=8)
    ax.add_patch(Circle((cx + width * 0.11, cy - height * 0.05), width * 0.055, facecolor=accent, edgecolor="white", linewidth=0.35, zorder=9))


def draw_multimodal_icon(ax: plt.Axes, x: float, y: float, width: float, height: float, *, accent: str, edge: str) -> None:
    draw_mri_icon(ax, x, y, width * 0.58, height, accent=accent, edge=edge)
    draw_document_icon(ax, x + width * 0.47, y + height * 0.05, width * 0.53, height * 0.90, accent=accent, edge=edge)


def draw_module_stack(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    modules: list[str],
    *,
    fill: str,
    accent: str,
    edge: str,
    ink: str,
) -> None:
    depth = width * 0.045
    for index in range(3):
        offset = (2 - index) * depth
        points = [
            (x + offset, y + offset * 0.42),
            (x + width + offset, y + offset * 0.42),
            (x + width + offset, y + height + offset * 0.42),
            (x + offset, y + height + offset * 0.42),
        ]
        ax.add_patch(Polygon(points, closed=True, facecolor=fill if index == 2 else "white", edgecolor=accent if index == 2 else edge, linewidth=0.55, zorder=2 + index))
    label(ax, x + width * 0.50, y + height * 0.64, modules[0], size=5.2, weight="bold", color=ink, ha="center")
    label(ax, x + width * 0.50, y + height * 0.30, modules[1], size=5.2, color=ink, ha="center")


def draw_parameter_matrix(ax: plt.Axes, x: float, y: float, width: float, height: float, *, accent: str) -> None:
    rows = 3
    columns = 5
    cell_width = width / columns
    cell_height = height / rows
    active = {(0, 0), (0, 3), (1, 1), (2, 4)}
    for row in range(rows):
        for column in range(columns):
            fill = accent if (row, column) in active else "white"
            ax.add_patch(
                Rectangle(
                    (x + column * cell_width, y + row * cell_height),
                    cell_width,
                    cell_height,
                    facecolor=fill,
                    edgecolor="#9AA1A8",
                    linewidth=0.28,
                    zorder=3,
                )
            )


def draw_shared_state(ax: plt.Axes, config: dict[str, Any], box: tuple[float, float, float, float]) -> None:
    style = config["style"]
    shared = config["protocol"]["shared_state"]
    x, y, width, height = box
    rounded(ax, x, y, width, height, fill=style["server"], edge=style["edge"], lw=0.75, radius=0.010)
    draw_model_icon(ax, x + 0.012, y + 0.013, 0.050, height - 0.025, accent=style["blue_accent"], edge=style["edge"])
    label(ax, x + 0.073, y + height * 0.66, shared["title"], size=6.2, weight="bold", color=style["ink"])
    label(ax, x + 0.073, y + height * 0.32, shared["subtitle"], size=5.1, color=style["muted"])
    arrow(ax, (x + width * 0.50, y + height * 0.50), (x + width * 0.56, y + height * 0.50), color=style["arrow"], lw=0.75)
    proxy_x = x + width * 0.58
    proxy_width = width * 0.39
    rounded(ax, proxy_x, y + 0.010, proxy_width, height - 0.020, fill=style["cream"], edge=style["cream_accent"], lw=0.75, radius=0.008, dashed=True)
    label(ax, proxy_x + proxy_width * 0.50, y + height * 0.64, shared["proxy_title"], size=5.3, weight="bold", ha="center")
    label(ax, proxy_x + proxy_width * 0.50, y + height * 0.31, shared["proxy_subtitle"], size=5.0, color=style["muted"], ha="center")


def draw_client_card(
    ax: plt.Axes,
    config: dict[str, Any],
    client: dict[str, Any],
    box: tuple[float, float, float, float],
) -> None:
    style = config["style"]
    x, y, width, height = box
    fill = style[client["fill_key"]]
    accent = style[client["accent_key"]]
    rounded(ax, x, y, width, height, fill="white", edge=accent, lw=0.85, radius=0.010)
    rounded(ax, x + 0.008, y + height - 0.061, 0.040, 0.038, fill=fill, edge=accent, lw=0.65, radius=0.006)
    label(ax, x + 0.028, y + height - 0.042, client["id"], size=6.2, weight="bold", ha="center")
    label(ax, x + 0.056, y + height - 0.042, client["name"], size=5.9, weight="bold")

    icon_x = x + width * 0.32
    icon_y = y + height * 0.690
    icon_width = width * 0.36
    icon_height = height * 0.145
    if client["icon"] == "text":
        draw_document_icon(ax, icon_x, icon_y, icon_width, icon_height, accent=accent, edge=style["edge"])
    elif client["icon"] == "mri":
        draw_mri_icon(ax, icon_x, icon_y, icon_width, icon_height, accent=accent, edge=style["edge"])
    else:
        draw_multimodal_icon(ax, icon_x - width * 0.03, icon_y, icon_width * 1.10, icon_height, accent=accent, edge=style["edge"])
    label(ax, x + width * 0.50, y + height * 0.645, client["input"], size=5.0, color=style["muted"], ha="center")
    arrow(ax, (x + width * 0.50, y + height * 0.610), (x + width * 0.50, y + height * 0.550), color=style["arrow"], lw=0.7, head=6.5)

    draw_module_stack(
        ax,
        x + width * 0.18,
        y + height * 0.420,
        width * 0.60,
        height * 0.115,
        client["modules"],
        fill=fill,
        accent=accent,
        edge=style["edge"],
        ink=style["ink"],
    )
    arrow(ax, (x + width * 0.50, y + height * 0.405), (x + width * 0.50, y + height * 0.350), color=style["arrow"], lw=0.7, head=6.5)

    loss_width = width * 0.58
    rounded(
        ax,
        x + width * 0.21,
        y + height * 0.270,
        loss_width,
        height * 0.070,
        fill=fill,
        edge=accent,
        lw=0.7,
        radius=0.006,
    )
    label(ax, x + width * 0.50, y + height * 0.305, client["loss"], size=5.7, weight="bold", ha="center")
    arrow(ax, (x + width * 0.50, y + height * 0.260), (x + width * 0.50, y + height * 0.215), color=style["arrow"], lw=0.7, head=6.5)

    matrix_x = x + width * 0.09
    matrix_y = y + height * 0.070
    draw_parameter_matrix(ax, matrix_x, matrix_y, width * 0.22, height * 0.115, accent=accent)
    rounded(
        ax,
        x + width * 0.37,
        y + height * 0.055,
        width * 0.54,
        height * 0.145,
        fill=fill,
        edge=accent,
        lw=0.7,
        radius=0.006,
        dashed=True,
    )
    label(
        ax,
        x + width * 0.64,
        y + height * 0.128,
        "\n".join(client["upload_lines"]),
        size=5.0,
        weight="bold",
        ha="center",
        linespacing=1.08,
    )


def draw_panel_a(ax: plt.Axes, config: dict[str, Any]) -> None:
    style = config["style"]
    box = (0.025, 0.075, 0.620, 0.850)
    panel(ax, box, "(a) Heterogeneous local learning", edge="#AAB8C6", fill=style["panel_fill"], accent=style["blue_accent"])
    draw_shared_state(ax, config, (0.043, 0.785, 0.584, 0.085))

    card_boxes = [
        (0.043, 0.214, 0.180, 0.510),
        (0.245, 0.214, 0.180, 0.510),
        (0.447, 0.214, 0.180, 0.510),
    ]
    for client, client_box in zip(config["protocol"]["clients"], card_boxes):
        center_x = client_box[0] + client_box[2] * 0.50
        arrow(ax, (center_x, 0.785), (center_x, 0.735), color=style["arrow"], lw=0.65, head=6.5, dashed=True)
        draw_client_card(ax, config, client, client_box)

    rounded(ax, 0.043, 0.112, 0.584, 0.062, fill=style["neutral"], edge=style["edge"], lw=0.65, radius=0.007)
    label(ax, 0.335, 0.150, "optimizer-scoped uploads  Δθ(k,p)", size=5.6, weight="bold", ha="center")
    label(ax, 0.335, 0.127, "only locally optimized parameter groups are non-zero", size=5.0, color=style["muted"], ha="center")
    for client_box in card_boxes:
        center_x = client_box[0] + client_box[2] * 0.50
        arrow(ax, (center_x, 0.207), (center_x, 0.174), color=style["arrow"], lw=0.65, head=6.2)


def draw_route_row(
    ax: plt.Axes,
    config: dict[str, Any],
    route: dict[str, Any],
    box: tuple[float, float, float, float],
) -> None:
    style = config["style"]
    x, y, width, height = box
    rounded(ax, x, y, width, height, fill=style[route["fill_key"]], edge=style["edge"], lw=0.65, radius=0.006)
    label(ax, x + 0.010, y + height * 0.64, "\n".join(route["label_lines"]), size=5.0, weight="bold", linespacing=1.0)
    label(ax, x + 0.010, y + height * 0.20, route["eligible"], size=5.0, color=style["muted"])


def draw_panel_b(ax: plt.Axes, config: dict[str, Any]) -> None:
    style = config["style"]
    box = (0.670, 0.505, 0.305, 0.420)
    panel(ax, box, "(b) Compatible server aggregation", edge="#B6AAC2", fill=style["panel_fill"], accent=style["lilac_accent"])

    rounded(ax, 0.690, 0.824, 0.265, 0.046, fill=style["neutral"], edge=style["edge"], lw=0.6, radius=0.006)
    label(ax, 0.8225, 0.847, "optimizer-scoped Δθ(k,p)  ·  private-case weight n_k", size=5.0, weight="bold", ha="center")

    route_boxes = [
        (0.690, 0.746, 0.145, 0.060),
        (0.690, 0.676, 0.145, 0.060),
        (0.690, 0.606, 0.145, 0.060),
    ]
    for route, route_box in zip(config["protocol"]["routes"], route_boxes):
        draw_route_row(ax, config, route, route_box)
        arrow(
            ax,
            (route_box[0] + route_box[2] + 0.004, route_box[1] + route_box[3] * 0.50),
            (0.846, route_box[1] + route_box[3] * 0.50),
            color=style["arrow"],
            lw=0.68,
            head=6.5,
        )

    rounded(ax, 0.849, 0.606, 0.052, 0.200, fill=style["server"], edge=style["blue_accent"], lw=0.8, radius=0.008)
    label(ax, 0.875, 0.723, "Parameter-", size=5.0, weight="bold", ha="center")
    label(ax, 0.875, 0.700, "wise", size=5.0, weight="bold", ha="center")
    label(ax, 0.875, 0.662, "FedAvg", size=5.3, weight="bold", ha="center")
    arrow(ax, (0.905, 0.706), (0.915, 0.706), color=style["arrow"], lw=0.72, head=6.5)
    rounded(ax, 0.918, 0.646, 0.045, 0.120, fill=style["green"], edge=style["green_accent"], lw=0.75, radius=0.006)
    label(ax, 0.9405, 0.724, "w", size=5.5, weight="bold", ha="center")
    label(ax, 0.9405, 0.700, "(t+1)", size=5.0, weight="bold", ha="center")
    label(ax, 0.9405, 0.674, "next", size=5.0, color=style["muted"], ha="center")

    rules = config["protocol"]["aggregation_rules"]
    rule_y = [0.574, 0.548, 0.522]
    rule_colors = [style["server"], style["cream"], style["neutral"]]
    for (tag, rule), y, tag_fill in zip(rules, rule_y, rule_colors):
        rounded(ax, 0.690, y - 0.010, 0.030, 0.021, fill=tag_fill, edge=style["edge"], lw=0.45, radius=0.004)
        label(ax, 0.705, y, tag, size=5.0, weight="bold", ha="center")
        label(ax, 0.726, y, rule, size=5.0, color=style["ink"] if tag != "buffers" else style["muted"])


def draw_panel_c(ax: plt.Axes, config: dict[str, Any]) -> None:
    style = config["style"]
    box = (0.670, 0.075, 0.305, 0.385)
    panel(ax, box, "(c) Controlled 2 × 2 study", edge="#B6AAC2", fill=style["panel_fill"], accent=style["lilac_accent"])
    label(ax, 0.846, 0.368, "Local optimizer", size=5.3, weight="bold", ha="center")
    label(ax, 0.792, 0.337, "FedAvg", size=5.2, weight="bold", ha="center")
    label(ax, 0.904, 0.337, "FedProx", size=5.2, weight="bold", ha="center")
    label(ax, 0.704, 0.294, "routing", size=5.0, color=style["muted"], ha="center")
    label(ax, 0.704, 0.258, "U", size=6.3, weight="bold", ha="center")
    label(ax, 0.704, 0.191, "R", size=6.3, weight="bold", ha="center")

    cells = [
        (0.736, 0.231, "U-FedAvg", style["server"], style["blue_accent"]),
        (0.850, 0.231, "U-FedProx", style["server"], style["lilac_accent"]),
        (0.736, 0.164, "R-FedAvg", style["cream"], style["cream_accent"]),
        (0.850, 0.164, "R-FedProx", style["cream"], style["lilac_accent"]),
    ]
    for x, y, text_value, fill, edge in cells:
        rounded(ax, x, y, 0.104, 0.050, fill=fill, edge=edge, lw=0.7, radius=0.006)
        label(ax, x + 0.052, y + 0.025, text_value, size=5.1, weight="bold", ha="center")

    rounded(ax, 0.692, 0.095, 0.261, 0.048, fill=style["neutral"], edge=style["edge"], lw=0.55, radius=0.006, dashed=True)
    controls = config["protocol"]["controls"]
    label(ax, 0.8225, 0.126, controls[0], size=5.0, ha="center")
    label(ax, 0.8225, 0.108, controls[1], size=5.0, weight="bold", color=style["muted"], ha="center")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(config: dict[str, Any], config_path: Path, output_dir: Path, script_path: Path) -> Path:
    figure = config["figure"]
    output_files = [output_dir / f"{figure['id']}.{suffix}" for suffix in figure["formats"]]
    missing = [path.name for path in output_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot write manifest; missing deliverables: {', '.join(missing)}")
    manifest = {
        "artifact": figure["id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
        },
        "reproducibility": {
            "deterministic": figure["deterministic"],
            "random_seed": figure["random_seed"],
            "script": script_path.relative_to(REPO_ROOT).as_posix(),
            "script_sha256": sha256(script_path),
            "config": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": sha256(config_path),
        },
        "contract": config["contract"],
        "source_trace": config["source_trace"],
        "delivery": {
            "primary": "Pure-vector SVG with editable text",
            "secondary": ["PDF", "PNG"],
            "dpi": figure["dpi"],
            "final_size_mm": config["contract"]["final_size_mm"],
            "rendering_backend": "Python matplotlib",
            "wps_office_profile": "Basic paths, lines, polygons, solid fills, and editable text; no gradients, filters, masks, or embedded raster images in SVG.",
            "patient_or_prediction_imagery": False,
            "performance_values": False,
        },
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def render(config: dict[str, Any]) -> plt.Figure:
    figure = config["figure"]
    configure_matplotlib(config)
    fig = plt.figure(
        figsize=(figure["width_inches"], figure["height_inches"]),
        facecolor="white",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    setup_canvas(ax)
    draw_panel_a(ax, config)
    draw_panel_b(ax, config)
    draw_panel_c(ax, config)
    return fig


def save_formats(
    fig: plt.Figure,
    config: dict[str, Any],
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    figure = config["figure"]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in formats:
        suffix = suffix.lower()
        if suffix not in {"svg", "pdf", "png"}:
            raise ValueError(f"Unsupported format: {suffix}")
        output_path = output_dir / f"{figure['id']}.{suffix}"
        kwargs: dict[str, Any] = {"format": suffix, "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = figure["dpi"]
        fig.savefig(output_path, **kwargs)
        saved.append(output_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the refined FedSAM3-Cream pipeline figure.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--formats", nargs="+", choices=["svg", "pdf", "png"])
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    formats = args.formats or config["figure"]["formats"]
    output_dir = REPO_ROOT / config["figure"]["output_dir"]
    fig = render(config)
    try:
        for path in save_formats(fig, config, output_dir, formats):
            print(path)
    finally:
        plt.close(fig)
    if args.write_manifest:
        print(write_manifest(config, config_path, output_dir, Path(__file__).resolve()))


if __name__ == "__main__":
    main()
