"""Command-line interface: ``wispr-on-chip plan|sweep|models|processes``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .fabric import DieTarget, TileConfig, WaferTarget
from .floorplan import render_ascii, render_svg
from .mapper import Plan, auto_mux, plan
from .models import MODELS, get_model
from .process import PROCESSES, get_process


def _fmt_si(x: float, unit: str = "") -> str:
    for div, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:,.2f} {suffix}{unit}"
    return f"{x:,.2f} {unit}"


def format_plan(p: Plan, show_stages: bool = True) -> str:
    m, t, tg, b, y, pf, pw = p.model, p.tile, p.target, p.budget, p.yield_report, p.perf, p.power
    L: list[str] = []
    L.append(f"== {m.name}: {m.total_params / 1e6:,.0f} M params, {t.weight_bits}-bit weights, {t.act_bits}-bit activations")
    if tg.kind == "wafer":
        w: WaferTarget = tg  # type: ignore[assignment]
        L.append(
            f"target: {w.diameter_mm:g} mm wafer, {w.reticle_grid[0]}x{w.reticle_grid[1]} stitched reticle fields"
            f" ({w.width_mm:g} x {w.height_mm:g} mm = {w.area_mm2:,.0f} mm²), {w.power_budget_w:,.0f} W budget"
        )
    else:
        L.append(f"target: single die {tg.area_mm2:,.0f} mm², {tg.power_budget_w:,.0f} W budget")
    L.append(
        f"process: {p.process.name} @ {p.process.clock_ghz:g} GHz; tile {t.area_mm2:g} mm², mux={t.mux_factor},"
        f" {t.sram_kib} KiB SRAM, {p.grid[0]}x{p.grid[1]} = {p.tiles_physical:,} tiles"
    )
    bd = b.breakdown()
    L.append(
        f"tile budget: {b.params_per_tile / 1e6:,.2f} M weights + {b.macs_per_tile:,} MACs per tile"
        f" (area: weights {bd['weights']:.2f}, macs {bd['macs']:.2f}, sram {bd['sram']:.2f}, overhead {bd['overhead']:.2f} mm²)"
    )
    L.append(
        f"yield: tile yield {y.tile_yield:.4f} at D0={y.defect_density_per_cm2:g}/cm²; expect {y.expected_dead:.1f} dead,"
        f" reserve {y.spares_reserved} spares -> {y.tiles_usable:,} usable tiles (P[success]={y.success_probability:.4f})"
    )
    L.append("")
    fit = "FITS" if p.fits else "DOES NOT FIT"
    L.append(
        f"mapping: {p.tiles_per_instance:,} tiles ({p.area_per_instance_mm2:,.0f} mm²) per instance, packed as"
        f" {p.instance_block[0]}x{p.instance_block[1]} blocks -> {p.instances} instance(s) [{fit}, {p.fit_fraction:.2f}x],"
        f" fabric {p.fabric_utilization:.0%} used"
    )
    if show_stages:
        L.append("")
        L.append(f"  {'stage':<14}{'kind':<9}{'params':>10}{'tiles':>7}{'bound':>8}{'MACs':>10}{'cyc/item':>10}{'cyc/window':>12}{'sram%':>7}")
        for s in _compress_stages(p):
            L.append(s)
    L.append("")
    L.append("performance (per instance unless noted):")
    L.append(
        f"  decode: {pf.decode_latency_us:.1f} µs/token single stream = {pf.decode_tokens_per_s_single_stream:,.0f} tok/s;"
        f" bottleneck stage {pf.decode_bottleneck_stage}"
    )
    L.append(
        f"  decode pipelined: {pf.decode_tokens_per_s_per_instance:,.0f} tok/s with {pf.streams} streams"
        f" ({pf.decode_pipeline_stages} stages, fill {pf.pipeline_fill:.0%})"
    )
    if m.has_encoder:
        L.append(
            f"  encoder: {pf.encoder_window_latency_us / 1e3:.2f} ms per {m.window_seconds:g} s window;"
            f" {pf.encoder_windows_per_s_per_instance:,.1f} windows/s pipelined; bottleneck {pf.encoder_bottleneck_stage}"
        )
        L.append(
            f"  single stream: {pf.single_stream_window_latency_us / 1e3:.2f} ms per window"
            f" = {pf.realtime_factor_single_stream:,.0f}x real time"
        )
        L.append(
            f"  total ({p.instances} inst.): {pf.windows_per_s_total:,.0f} windows/s ="
            f" {pf.audio_hours_per_hour_total:,.0f} audio-hours per hour peak,"
            f" {pf.sustained_audio_hours_per_hour_total:,.0f} sustained in power budget"
        )
    L.append(
        f"  total ({p.instances} inst.): {pf.decode_tokens_per_s_total:,.0f} tok/s peak,"
        f" {pf.sustained_tokens_per_s_total:,.0f} sustained in power budget"
    )
    L.append("")
    L.append("power:")
    L.append(f"  {pw.energy_per_token_nj:,.0f} nJ per decoded token" + (f"; {pw.energy_per_window_total_nj / 1e3:,.1f} µJ per window" if m.has_encoder else ""))
    L.append(
        f"  peak dynamic {pw.dynamic_w_total_peak:,.0f} W + leakage {pw.leakage_w:,.0f} W = {pw.total_w_peak:,.0f} W"
        f" vs budget {pw.budget_w:,.0f} W -> throughput scale {pw.power_scale:.2f}"
    )
    return "\n".join(L)


def _compress_stages(p: Plan) -> list[str]:
    """Collapse runs of identical stages (enc0..enc31) into one row."""
    rows: list[str] = []
    i = 0
    pls = p.placements
    while i < len(pls):
        j = i
        while j + 1 < len(pls) and pls[j + 1].stage.kind == pls[i].stage.kind and pls[j + 1].tiles == pls[i].tiles:
            j += 1
        s = pls[i]
        name = s.stage.name if i == j else f"{pls[i].stage.name}..{pls[j].stage.name}"
        rows.append(
            f"  {name:<14}{s.stage.kind:<9}{_fmt_si(s.stage.params):>10}{s.tiles:>7}{s.bound:>8}{_fmt_si(s.macs):>10}"
            f"{s.cycles_per_item:>10,.0f}{s.cycles_per_window:>12,.0f}{s.sram_utilization:>7.0%}"
        )
        i = j + 1
    return rows


def _build(args: argparse.Namespace, model_name: str) -> Plan:
    model = get_model(model_name)
    proc = get_process(args.process)
    if args.clock:
        proc = proc.with_(clock_ghz=args.clock)
    tile = TileConfig(
        area_mm2=args.tile_area,
        mux_factor=args.mux,
        sram_kib=args.sram_kib,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        kv_bits=args.kv_bits,
        streams=args.streams,
    )
    if args.target == "wafer":
        target = WaferTarget(fabric_side_mm=args.fabric_side, power_budget_w=args.power_budget or 20_000.0, defect_density_per_cm2=args.defect_density)
    else:
        target = DieTarget(die_area_mm2=args.die_area, power_budget_w=args.power_budget or 350.0, defect_density_per_cm2=args.defect_density)
    if args.auto_mux:
        return auto_mux(model, proc, tile, target, instances=args.instances)
    return plan(model, proc, tile, target, max_instances=args.max_instances)


def _add_fabric_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--process", default="n5", choices=sorted(PROCESSES))
    sp.add_argument("--clock", type=float, default=None, help="override clock (GHz)")
    sp.add_argument("--target", default="wafer", choices=["wafer", "die"])
    sp.add_argument("--die-area", type=float, default=815.0, help="die area in mm² for --target die")
    sp.add_argument("--fabric-side", type=float, default=None, help="wafer fabric side in mm (default: inscribed square)")
    sp.add_argument("--tile-area", type=float, default=1.0, help="tile area in mm²")
    sp.add_argument("--mux", type=int, default=1024, help="weights per MAC (time-multiplexing factor)")
    sp.add_argument("--auto-mux", action="store_true", help="pick the smallest mux at which --instances copies fit")
    sp.add_argument("--instances", type=int, default=1, help="instances required for --auto-mux")
    sp.add_argument("--max-instances", type=int, default=None, help="cap instances placed (default: fill the fabric)")
    sp.add_argument("--sram-kib", type=int, default=1024)
    sp.add_argument("--weight-bits", type=int, default=4)
    sp.add_argument("--act-bits", type=int, default=8)
    sp.add_argument("--kv-bits", type=int, default=None, help="KV-cache width (default: act bits)")
    sp.add_argument("--streams", type=int, default=None, help="decode streams per instance (default: fill the pipeline)")
    sp.add_argument("--defect-density", type=float, default=0.1, help="defects per cm²")
    sp.add_argument("--power-budget", type=float, default=None, help="watts (default 20 kW wafer / 350 W die)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wispr-on-chip", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list model presets")
    sub.add_parser("processes", help="list process presets")

    sp = sub.add_parser("plan", help="map one model onto a fabric and report")
    sp.add_argument("--model", default="whisper-large-v3")
    _add_fabric_args(sp)
    sp.add_argument("--json", action="store_true", help="emit the plan as JSON")
    sp.add_argument("--ascii", action="store_true", help="print an ASCII floorplan")
    sp.add_argument("--svg", metavar="PATH", help="write an SVG floorplan")

    sw = sub.add_parser("sweep", help="compare all Whisper sizes (or --models) on one fabric")
    sw.add_argument("--models", nargs="*", default=None, help="model names (default: all whisper presets)")
    _add_fabric_args(sw)
    return ap


def _sweep(args: argparse.Namespace) -> str:
    names = args.models or [n for n in MODELS if n.startswith("whisper")]
    hdr = (
        f"{'model':<24}{'params':>10}{'tiles/inst':>11}{'inst':>6}{'µs/tok':>8}{'tok/s 1-str':>12}"
        f"{'x realtime':>11}{'audio-h/h':>12}{'peak W':>9}{'sust. W':>9}"
    )
    rows = [hdr, "-" * len(hdr)]
    for n in names:
        p = _build(args, n)
        pf = p.perf
        pw = p.power
        sustained_w = pw.leakage_w + pw.dynamic_w_total_peak * pw.power_scale
        rows.append(
            f"{p.model.name:<24}{_fmt_si(p.model.total_params):>10}{p.tiles_per_instance:>11,}"
            f"{p.instances:>6}{pf.decode_latency_us:>8.1f}{pf.decode_tokens_per_s_single_stream:>12,.0f}"
            f"{pf.realtime_factor_single_stream:>11,.0f}{pf.sustained_audio_hours_per_hour_total:>12,.0f}"
            f"{pw.total_w_peak:>9,.0f}{sustained_w:>9,.0f}"
        )
    rows.append("audio-h/h = hours of audio transcribed per hour, all instances, throttled to the power budget")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "models":
        for m in MODELS.values():
            print(m.describe())
        return 0
    if args.cmd == "processes":
        for pr in PROCESSES.values():
            print(
                f"{pr.name}: logic {pr.logic_density_mtr_mm2:g} MTr/mm², sram {pr.sram_density_mbit_mm2:g} Mbit/mm²,"
                f" weight cells {pr.weight_cell_density_mbit_mm2:g} Mbit/mm², int8 MAC {pr.mac_energy_pj_int8:g} pJ,"
                f" {pr.clock_ghz:g} GHz"
            )
        return 0
    if args.cmd == "sweep":
        print(_sweep(args))
        return 0

    try:
        p = _build(args, args.model)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(p.to_dict(), indent=2, default=str))
    else:
        print(format_plan(p))
    if args.ascii:
        print()
        print(render_ascii(p))
    if args.svg:
        with open(args.svg, "w", encoding="utf-8") as f:
            f.write(render_svg(p))
        print(f"\nwrote floorplan to {args.svg}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
