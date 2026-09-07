"""Floorplan rendering: which tile holds which stage of which model instance."""

from __future__ import annotations

import math
from collections import Counter

from .mapper import Plan

KIND_ORDER = ["stem", "encoder", "embed", "decoder", "head", "spare", "unused"]
KIND_COLORS = {
    "stem": "#f4a261",
    "encoder": "#2a9d8f",
    "embed": "#a8dadc",
    "decoder": "#e76f51",
    "head": "#8d99ae",
    "spare": "#ffe8a3",
    "unused": "#e6e6e6",
}
KIND_CHARS = {
    "stem": "S",
    "encoder": "E",
    "embed": "M",
    "decoder": "D",
    "head": "H",
    "spare": "+",
    "unused": ".",
}


def tile_assignments(plan: Plan) -> list[tuple[str, int]]:
    """(kind, instance) for every physical tile in raster order.

    Each instance occupies a rectangular block (``plan.instance_block``); stages
    fill the block in dataflow order. Tiles left over inside blocks and outside
    the block grid are spares first, then unused. Real hardware would spread
    spares evenly; the layout here is for visualisation and counting only.
    """
    cols, rows = plan.grid
    grid: list[tuple[str, int]] = [("unused", -1)] * (cols * rows)
    bw, bh = plan.instance_block
    across = plan.block_grid[0]
    stage_seq: list[str] = []
    for p in plan.placements:
        stage_seq.extend([p.stage.kind] * p.tiles)
    for inst in range(plan.instances):
        bx, by = (inst % across) * bw, (inst // across) * bh
        for k, kind in enumerate(stage_seq):
            x, y = bx + k % bw, by + k // bw
            grid[y * cols + x] = (kind, inst)
    spares = plan.yield_report.spares_reserved
    for i, (kind, _) in enumerate(grid):
        if spares == 0:
            break
        if kind == "unused":
            grid[i] = ("spare", -1)
            spares -= 1
    return grid


def render_ascii(plan: Plan, width: int = 72) -> str:
    cols, rows = plan.grid
    assign = tile_assignments(plan)
    bw = max(1, math.ceil(cols / width))
    bh = bw * 2  # terminal characters are about twice as tall as wide
    lines: list[str] = []
    for y0 in range(0, rows, bh):
        line = []
        for x0 in range(0, cols, bw):
            block = Counter()
            for y in range(y0, min(rows, y0 + bh)):
                for x in range(x0, min(cols, x0 + bw)):
                    block[assign[y * cols + x]] += 1
            (kind, inst), _ = block.most_common(1)[0]
            ch = KIND_CHARS[kind]
            if inst >= 0 and inst % 2 == 1:
                ch = ch.lower()
            line.append(ch)
        lines.append("".join(line))
    legend = "  ".join(f"{KIND_CHARS[k]}={k}" for k in KIND_ORDER)
    scale = f"each character = {bw}x{bh} tiles of {plan.tile.area_mm2:g} mm²; lowercase = odd-numbered instance"
    return "\n".join(lines + ["", legend, scale])


def render_svg(plan: Plan, size_px: int = 900) -> str:
    cols, rows = plan.grid
    assign = tile_assignments(plan)
    target = plan.target
    is_wafer = target.kind == "wafer"
    diameter = target.diameter_mm if is_wafer else max(target.width_mm, target.height_mm) * 1.1  # type: ignore[attr-defined]
    margin = 20
    scale = (size_px - 2 * margin) / diameter
    cx = cy = margin + diameter * scale / 2
    fw, fh = target.width_mm * scale, target.height_mm * scale
    fx, fy = cx - fw / 2, cy - fh / 2
    ts = plan.tile.side_mm * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px + 120}" '
        f'viewBox="0 0 {size_px} {size_px + 120}" font-family="system-ui, sans-serif" font-size="13" shape-rendering="crispEdges">',
        f'<rect width="100%" height="100%" fill="white"/>',
    ]
    if is_wafer:
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{diameter * scale / 2:.1f}" fill="#f7f7f7" stroke="#999" stroke-width="1.5"/>'
        )
    # tiles, run-length merged per row
    for y in range(rows):
        x = 0
        while x < cols:
            kind, inst = assign[y * cols + x]
            x1 = x
            while x1 < cols and assign[y * cols + x1] == (kind, inst):
                x1 += 1
            color = KIND_COLORS[kind]
            opacity = 1.0 if inst < 0 or inst % 2 == 0 else 0.7
            parts.append(
                f'<rect x="{fx + x * ts:.2f}" y="{fy + y * ts:.2f}" width="{(x1 - x) * ts:.2f}" height="{ts:.2f}" '
                f'fill="{color}" fill-opacity="{opacity}"/>'
            )
            x = x1
    # reticle grid on wafers
    if is_wafer:
        rw, rh = target.reticle_w_mm * scale, target.reticle_h_mm * scale  # type: ignore[attr-defined]
        gc, gr = target.reticle_grid  # type: ignore[attr-defined]
        for i in range(gc + 1):
            parts.append(
                f'<line x1="{fx + i * rw:.1f}" y1="{fy:.1f}" x2="{fx + i * rw:.1f}" y2="{fy + fh:.1f}" stroke="#333" stroke-width="0.6" stroke-opacity="0.5"/>'
            )
        for j in range(gr + 1):
            parts.append(
                f'<line x1="{fx:.1f}" y1="{fy + j * rh:.1f}" x2="{fx + fw:.1f}" y2="{fy + j * rh:.1f}" stroke="#333" stroke-width="0.6" stroke-opacity="0.5"/>'
            )
    parts.append(f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{fw:.1f}" height="{fh:.1f}" fill="none" stroke="#222" stroke-width="1.2"/>')

    # legend and title
    ly = size_px + 10
    parts.append(
        f'<text x="{margin}" y="{ly}" font-size="15" font-weight="600">{plan.model.name} on {plan.process.name} '
        f'{target.kind}: {plan.instances} instance(s), {plan.tiles_per_instance} tiles each, '
        f'mux={plan.tile.mux_factor}, {plan.tile.weight_bits}-bit weights</text>'
    )
    x = margin
    for k in KIND_ORDER:
        parts.append(f'<rect x="{x}" y="{ly + 14}" width="14" height="14" fill="{KIND_COLORS[k]}" stroke="#666" stroke-width="0.5"/>')
        parts.append(f'<text x="{x + 18}" y="{ly + 26}">{k}</text>')
        x += 24 + 8 * len(k) + 20
    p = plan.perf
    line2 = (
        f"single-stream decode {p.decode_latency_us:.1f} µs/token ({p.decode_tokens_per_s_single_stream:,.0f} tok/s); "
        f"pipelined {p.sustained_tokens_per_s_total:,.0f} tok/s total"
    )
    if plan.model.has_encoder:
        line2 += f"; {p.sustained_audio_hours_per_hour_total:,.0f} audio-hours/hour"
    parts.append(f'<text x="{margin}" y="{ly + 52}">{line2}</text>')
    parts.append(
        f'<text x="{margin}" y="{ly + 72}">peak power {plan.power.total_w_peak:,.0f} W of {plan.power.budget_w:,.0f} W budget; '
        f'tile yield {plan.yield_report.tile_yield:.4f}, {plan.yield_report.spares_reserved} spares of {plan.tiles_physical} tiles</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
