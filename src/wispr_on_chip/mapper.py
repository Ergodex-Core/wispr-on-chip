"""Place a model's stages onto tiles and derive latency, throughput, power and yield.

The mapping is a straight pipeline: every stage gets a contiguous block of
tiles, weights are spread evenly across the block, and items (frames or
tokens) flow from stage to stage. Within a stage all tiles run in lock step,
so the time to process one item is the time to sweep the stage's weights once
through its MAC arrays plus attention, communication and fixed overheads.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .fabric import DieTarget, Target, TileBudget, TileConfig, WaferTarget, YieldReport, tile_budget, yield_model
from .models import ModelSpec, Stage
from .process import Process


@dataclass(frozen=True)
class StagePlacement:
    stage: Stage
    tiles: int
    tiles_for_weights: int
    tiles_for_sram: int
    macs: int
    streams_held: int
    sram_bits_needed: int
    sram_bits_available: int
    sweep_cycles: float
    attn_cycles: float
    comm_cycles: float
    window_cycles: float
    items_per_window: int
    energy_per_item_pj: float
    energy_per_window_pj: float

    @property
    def cycles_per_item(self) -> float:
        return self.sweep_cycles + self.attn_cycles + self.comm_cycles

    @property
    def cycles_per_window(self) -> float:
        return self.items_per_window * self.cycles_per_item + self.window_cycles

    # populated by the mapper so utilization can be computed without the budget
    _params_per_tile: int = field(default=1, repr=False, compare=False)

    @property
    def weight_utilization(self) -> float:
        """Fraction of the stage's weight-cell capacity actually programmed."""
        return self.stage.params / (self.tiles * self._params_per_tile) if self.tiles else 0.0

    @property
    def sram_utilization(self) -> float:
        return self.sram_bits_needed / self.sram_bits_available if self.sram_bits_available else 0.0

    @property
    def bound(self) -> str:
        if self.tiles_for_sram > self.tiles_for_weights:
            return "sram"
        return "weights"


@dataclass(frozen=True)
class PerfReport:
    clock_hz: float
    streams: int
    decode_pipeline_stages: int
    pipeline_fill: float
    decode_latency_us: float
    """Single-stream time per generated token."""
    decode_tokens_per_s_single_stream: float
    decode_tokens_per_s_per_instance: float
    """With enough streams in flight to fill the decoder pipeline."""
    decode_bottleneck_stage: str
    window_seconds: float
    encoder_window_latency_us: float
    encoder_windows_per_s_per_instance: float
    encoder_bottleneck_stage: str
    decoder_windows_per_s_per_instance: float
    windows_per_s_per_instance: float
    single_stream_window_latency_us: float
    realtime_factor_single_stream: float
    """Audio seconds transcribed per wall-clock second by one stream (30 s / latency)."""
    instances: int
    decode_tokens_per_s_total: float
    windows_per_s_total: float
    audio_hours_per_hour_total: float
    """Hours of audio transcribed per hour of wall-clock, all instances, pipelines full."""
    power_scale: float = 1.0
    """Fraction of peak throughput sustainable inside the power budget."""

    @property
    def sustained_tokens_per_s_total(self) -> float:
        return self.decode_tokens_per_s_total * self.power_scale

    @property
    def sustained_audio_hours_per_hour_total(self) -> float:
        return self.audio_hours_per_hour_total * self.power_scale


@dataclass(frozen=True)
class PowerReport:
    energy_per_token_nj: float
    """Decoder-side energy per generated token (one instance)."""
    energy_per_window_encoder_nj: float
    energy_per_window_total_nj: float
    """Encoder + per-window decoder one-offs + tokens_per_window decode tokens."""
    dynamic_w_per_instance_peak: float
    dynamic_w_total_peak: float
    leakage_w: float
    total_w_peak: float
    budget_w: float
    power_scale: float


@dataclass(frozen=True)
class Plan:
    model: ModelSpec
    process: Process
    tile: TileConfig
    target: Target
    budget: TileBudget
    placements: list[StagePlacement]
    grid: tuple[int, int]
    tiles_physical: int
    yield_report: YieldReport
    tiles_per_instance: int
    instances: int
    instance_block: tuple[int, int]
    """Tiles (columns, rows) of the rectangular region each instance occupies."""
    block_grid: tuple[int, int]
    """How many instance blocks fit across and down the fabric."""
    perf: PerfReport
    power: PowerReport

    @property
    def fits(self) -> bool:
        return self.instances >= 1

    @property
    def tiles_used(self) -> int:
        return self.instances * self.tiles_per_instance

    @property
    def area_per_instance_mm2(self) -> float:
        return self.tiles_per_instance * self.tile.area_mm2

    @property
    def fabric_utilization(self) -> float:
        return self.tiles_used / self.tiles_physical if self.tiles_physical else 0.0

    @property
    def fit_fraction(self) -> float:
        """usable tiles / tiles needed for one instance (>= 1 means it fits)."""
        return self.yield_report.tiles_usable / self.tiles_per_instance if self.tiles_per_instance else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model.name,
            "model_params": self.model.total_params,
            "process": self.process.name,
            "tile": asdict(self.tile),
            "target": asdict(self.target),  # type: ignore[arg-type]
            "target_area_mm2": self.target.area_mm2,
            "budget": {
                "params_per_tile": self.budget.params_per_tile,
                "macs_per_tile": self.budget.macs_per_tile,
                "area_breakdown_mm2": self.budget.breakdown(),
            },
            "grid": list(self.grid),
            "tiles_physical": self.tiles_physical,
            "tiles_per_instance": self.tiles_per_instance,
            "instances": self.instances,
            "instance_block": list(self.instance_block),
            "block_grid": list(self.block_grid),
            "fits": self.fits,
            "fit_fraction": self.fit_fraction,
            "area_per_instance_mm2": self.area_per_instance_mm2,
            "fabric_utilization": self.fabric_utilization,
            "yield": asdict(self.yield_report),
            "perf": asdict(self.perf),
            "power": asdict(self.power),
            "stages": [
                {
                    "name": p.stage.name,
                    "kind": p.stage.kind,
                    "params": p.stage.params,
                    "tiles": p.tiles,
                    "tiles_for_weights": p.tiles_for_weights,
                    "tiles_for_sram": p.tiles_for_sram,
                    "bound": p.bound,
                    "macs": p.macs,
                    "cycles_per_item": p.cycles_per_item,
                    "cycles_per_window": p.cycles_per_window,
                    "sram_utilization": p.sram_utilization,
                    "energy_per_item_nj": p.energy_per_item_pj / 1e3,
                }
                for p in self.placements
            ],
        }
        d["perf"]["sustained_tokens_per_s_total"] = self.perf.sustained_tokens_per_s_total
        d["perf"]["sustained_audio_hours_per_hour_total"] = self.perf.sustained_audio_hours_per_hour_total
        return d


# --------------------------------------------------------------------------- placement


def _place_stage(model: ModelSpec, stage: Stage, budget: TileBudget, streams: int) -> StagePlacement:
    tile, proc = budget.tile, budget.process
    w, a, kvb = tile.weight_bits, tile.act_bits, tile.kv_bits_eff

    tiles_for_weights = math.ceil(stage.params / budget.params_per_tile) if stage.params else 0
    streams_held = streams if stage.decoder_side else 1
    kv_bits = stage.kv_elements_per_stream * kvb * streams_held
    act_bits = stage.act_elements_per_item * a * 2  # double-buffered working set
    sram_needed = kv_bits + act_bits
    tiles_for_sram = math.ceil(sram_needed / budget.sram_bits)
    tiles = max(1, tiles_for_weights, tiles_for_sram)

    has_macs = stage.linear_macs_per_item > 0 or stage.attn_macs_per_item > 0 or stage.window_macs > 0
    macs = tiles * budget.macs_per_tile if has_macs else 0

    sweep = math.ceil(stage.linear_macs_per_item / macs) if macs else 0.0
    attn = stage.attn_macs_per_item / macs if macs else 0.0
    comm = tile.hop_cycles * (2 * math.ceil(math.sqrt(tiles)) + 1) + tile.stage_overhead_cycles
    window = stage.window_macs / macs if macs else 0.0

    e_mac_w = proc.mac_energy_pj_int8 * (w * a) / 64.0
    e_mac_a = proc.mac_energy_pj_int8 * (a * a) / 64.0
    e_lin = stage.linear_macs_per_item * (e_mac_w + w * proc.weight_read_energy_pj_bit)
    e_attn = stage.attn_macs_per_item * (e_mac_a + kvb * proc.sram_read_energy_pj_bit)
    e_act = stage.act_elements_per_item * a * 2 * proc.sram_read_energy_pj_bit
    e_comm = stage.act_elements_per_item * a * tiles * proc.hop_energy_pj_bit
    e_item = (e_lin + e_attn + e_act + e_comm) * proc.activity_overhead
    e_window = stage.window_macs * (e_mac_w + w * proc.weight_read_energy_pj_bit) * proc.activity_overhead

    return StagePlacement(
        stage=stage,
        tiles=tiles,
        tiles_for_weights=tiles_for_weights,
        tiles_for_sram=tiles_for_sram,
        macs=macs,
        streams_held=streams_held,
        sram_bits_needed=sram_needed,
        sram_bits_available=tiles * budget.sram_bits,
        sweep_cycles=float(sweep),
        attn_cycles=attn,
        comm_cycles=float(comm),
        window_cycles=window,
        items_per_window=model.items_per_window(stage),
        energy_per_item_pj=e_item,
        energy_per_window_pj=e_window,
        _params_per_tile=budget.params_per_tile,
    )


# --------------------------------------------------------------------------- performance


def _perf(model: ModelSpec, proc: Process, placements: list[StagePlacement], streams: int, instances: int) -> PerfReport:
    clock = proc.clock_ghz * 1e9
    enc = [p for p in placements if p.stage.encoder_side]
    dec = [p for p in placements if p.stage.decoder_side]

    dec_latency = sum(p.cycles_per_item for p in dec)
    dec_bottleneck = max(dec, key=lambda p: p.cycles_per_item)
    fill = min(1.0, streams / len(dec))
    tok_single = clock / dec_latency
    tok_instance = clock / dec_bottleneck.cycles_per_item * fill

    if model.has_encoder:
        enc_latency = sum(p.cycles_per_window for p in enc)
        enc_bottleneck = max(enc, key=lambda p: p.cycles_per_window)
        enc_windows = clock / enc_bottleneck.cycles_per_window
        dec_window_bottleneck = max(p.cycles_per_window for p in dec)
        dec_windows = clock / dec_window_bottleneck * fill
        windows = min(enc_windows, dec_windows)
        single_window = (enc_latency + sum(p.cycles_per_window for p in dec)) / clock
        rtf = model.window_seconds / single_window
        enc_bottleneck_name = enc_bottleneck.stage.name
        enc_latency_us = enc_latency / clock * 1e6
        single_window_us = single_window * 1e6
    else:
        enc_windows = dec_windows = windows = 0.0
        rtf = 0.0
        enc_bottleneck_name = "-"
        enc_latency_us = single_window_us = 0.0

    return PerfReport(
        clock_hz=clock,
        streams=streams,
        decode_pipeline_stages=len(dec),
        pipeline_fill=fill,
        decode_latency_us=dec_latency / clock * 1e6,
        decode_tokens_per_s_single_stream=tok_single,
        decode_tokens_per_s_per_instance=tok_instance,
        decode_bottleneck_stage=dec_bottleneck.stage.name,
        window_seconds=model.window_seconds,
        encoder_window_latency_us=enc_latency_us,
        encoder_windows_per_s_per_instance=enc_windows,
        encoder_bottleneck_stage=enc_bottleneck_name,
        decoder_windows_per_s_per_instance=dec_windows,
        windows_per_s_per_instance=windows,
        single_stream_window_latency_us=single_window_us,
        realtime_factor_single_stream=rtf,
        instances=instances,
        decode_tokens_per_s_total=tok_instance * instances,
        windows_per_s_total=windows * instances,
        audio_hours_per_hour_total=windows * instances * model.window_seconds,
    )


def _power(
    model: ModelSpec, proc: Process, target: Target, placements: list[StagePlacement], perf: PerfReport, instances: int
) -> PowerReport:
    enc = [p for p in placements if p.stage.encoder_side]
    dec = [p for p in placements if p.stage.decoder_side]
    e_token = sum(p.energy_per_item_pj for p in dec)
    e_enc_window = sum(p.items_per_window * p.energy_per_item_pj for p in enc)
    e_dec_window_oneoff = sum(p.energy_per_window_pj for p in dec)
    e_window_total = e_enc_window + e_dec_window_oneoff + model.tokens_per_window * e_token

    if model.has_encoder:
        dyn_instance = perf.windows_per_s_per_instance * e_window_total * 1e-12
    else:
        dyn_instance = perf.decode_tokens_per_s_per_instance * e_token * 1e-12
    dyn_total = dyn_instance * instances
    leakage = target.area_mm2 * proc.leakage_mw_mm2 * 1e-3
    total = dyn_total + leakage
    headroom = target.power_budget_w - leakage
    scale = 1.0 if dyn_total <= headroom else max(0.0, headroom / dyn_total) if dyn_total > 0 else 1.0

    return PowerReport(
        energy_per_token_nj=e_token / 1e3,
        energy_per_window_encoder_nj=e_enc_window / 1e3,
        energy_per_window_total_nj=e_window_total / 1e3,
        dynamic_w_per_instance_peak=dyn_instance,
        dynamic_w_total_peak=dyn_total,
        leakage_w=leakage,
        total_w_peak=total,
        budget_w=target.power_budget_w,
        power_scale=scale,
    )


# --------------------------------------------------------------------------- packing


def pack_instances(
    cols: int, rows: int, tiles_per_instance: int, spares: int, max_instances: int | None = None
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Pack model instances as rectangular blocks on a ``cols`` x ``rows`` tile grid.

    Returns ``(instances, (block_w, block_h), (blocks_across, blocks_down))``.
    The block shape is chosen to maximise the number of whole blocks that fit;
    ties go to the squarest block (shortest broadcast/reduce paths). Spare
    tiles must fit in the slack: unused tiles inside blocks plus tiles outside
    the block grid. If they do not, instances are dropped until they do.
    """
    if tiles_per_instance <= 0 or cols <= 0 or rows <= 0:
        return 0, (0, 0), (0, 0)
    best: tuple[int, float, int, int] | None = None  # (capacity, -aspect_penalty, w, h)
    for w in range(1, cols + 1):
        h = math.ceil(tiles_per_instance / w)
        if h > rows:
            continue
        cap = (cols // w) * (rows // h)
        if cap == 0:
            continue
        penalty = abs(math.log(w / h))
        cand = (cap, -penalty, w, h)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return 0, (0, 0), (0, 0)
    cap, _, w, h = best
    instances = cap if max_instances is None else min(cap, max_instances)
    while instances > 0 and cols * rows - instances * tiles_per_instance < spares:
        instances -= 1
    return instances, (w, h), (cols // w, rows // h)


# --------------------------------------------------------------------------- stream balancing


def _auto_streams(
    model: ModelSpec, proc: Process, budget: TileBudget, stages: list[Stage], n_dec: int
) -> tuple[int, list[StagePlacement]]:
    """Pick the decode stream count.

    Decoder-only models: fill the pipeline (one stream per stage). Encoder-decoder
    models: the encoder holds one window per stage regardless of streams, so use
    the fewest streams at which the decoder's window rate matches the encoder's;
    more streams would only spend SRAM on KV caches the encoder cannot feed.
    """
    n_dec = max(1, n_dec)
    if not model.has_encoder:
        return n_dec, [_place_stage(model, s, budget, n_dec) for s in stages]
    enc_placements = [_place_stage(model, s, budget, 1) for s in stages if s.encoder_side]
    enc_window_cycles = max(p.cycles_per_window for p in enc_placements)
    best: tuple[int, list[StagePlacement]] | None = None
    for streams in range(1, n_dec + 1):
        dec_placements = [_place_stage(model, s, budget, streams) for s in stages if s.decoder_side]
        dec_window_cycles = max(p.cycles_per_window for p in dec_placements)
        best = (streams, enc_placements + dec_placements)
        # decoder window rate = clock / dec_cycles * streams / n_dec  >=  clock / enc_cycles
        if dec_window_cycles * n_dec <= enc_window_cycles * streams:
            break
    assert best is not None
    return best


# --------------------------------------------------------------------------- entry points


def plan(
    model: ModelSpec,
    process: Process,
    tile: TileConfig | None = None,
    target: Target | None = None,
    max_instances: int | None = None,
) -> Plan:
    """Map ``model`` onto ``target`` built from ``tile`` on ``process``."""
    tile = tile or TileConfig()
    target = target or WaferTarget()
    budget = tile_budget(process, tile)
    stages = model.stages()
    n_dec = sum(1 for s in stages if s.decoder_side)

    if tile.streams:
        streams = tile.streams
        placements = [_place_stage(model, s, budget, streams) for s in stages]
    else:
        streams, placements = _auto_streams(model, process, budget, stages, n_dec)
    tiles_per_instance = sum(p.tiles for p in placements)

    cols, rows = target.grid(tile.side_mm)
    tiles_physical = cols * rows
    yr = yield_model(tile, target, tiles_physical)
    instances, block, block_grid = pack_instances(cols, rows, tiles_per_instance, yr.spares_reserved, max_instances)

    perf = _perf(model, process, placements, streams, instances)
    power = _power(model, process, target, placements, perf, instances)
    perf = PerfReport(**{**asdict(perf), "power_scale": power.power_scale})

    return Plan(
        model=model,
        process=process,
        tile=tile,
        target=target,
        budget=budget,
        placements=placements,
        grid=(cols, rows),
        tiles_physical=tiles_physical,
        yield_report=yr,
        tiles_per_instance=tiles_per_instance,
        instances=instances,
        instance_block=block,
        block_grid=block_grid,
        perf=perf,
        power=power,
    )


def auto_mux(
    model: ModelSpec,
    process: Process,
    tile: TileConfig,
    target: Target,
    instances: int = 1,
    max_mux: int = 1 << 20,
) -> Plan:
    """Find the smallest ``mux_factor`` (most compute per weight) at which ``instances``
    copies of the model fit on the target, and return that plan.

    Smaller mux means more MACs per weight and lower latency; the area per
    weight is monotone non-increasing in mux, so a binary search suffices.
    """
    lo, hi = 1, max_mux
    best: Plan | None = None
    top = plan(model, process, tile.with_(mux_factor=hi), target)
    if top.instances < instances:
        raise ValueError(
            f"{model.name} does not fit {instances}x on {target.kind} even at mux={hi}: "
            f"needs {top.tiles_per_instance * instances} tiles, {top.yield_report.tiles_usable} usable"
        )
    while lo < hi:
        mid = (lo + hi) // 2
        p = plan(model, process, tile.with_(mux_factor=mid), target)
        if p.instances >= instances:
            best, hi = p, mid
        else:
            lo = mid + 1
    if best is None or best.tile.mux_factor != lo:
        best = plan(model, process, tile.with_(mux_factor=lo), target)
    # Packing makes fit slightly non-monotone in mux; probe a few smaller values.
    for mux in range(best.tile.mux_factor - 1, max(0, best.tile.mux_factor - 33), -1):
        cand = plan(model, process, tile.with_(mux_factor=mux), target)
        if cand.instances >= instances:
            best = cand
    return best


__all__ = [
    "DieTarget",
    "PerfReport",
    "Plan",
    "PowerReport",
    "StagePlacement",
    "WaferTarget",
    "auto_mux",
    "pack_instances",
    "plan",
]
