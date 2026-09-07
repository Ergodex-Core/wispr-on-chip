"""Tiles, targets (die or wafer), and the per-tile silicon budget.

The fabric is a regular 2-D mesh of identical tiles. A tile contains:

* a bank of hardwired weight cells (via/metal programmed; the per-model masks),
* a MAC array that sweeps the local weight bank ``mux_factor`` weights per MAC,
* tile-local SRAM for activations and KV cache,
* a router plus clock/power/redundancy overhead.

Only the weight bank changes between models; everything else is fixed base
silicon shared by every model on the same base layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from .process import Process


@dataclass(frozen=True)
class TileConfig:
    area_mm2: float = 1.0
    mux_factor: int = 1024
    """Weights per MAC unit: each MAC sweeps this many hardwired weights per item.
    1 = fully spatial (one multiplier per weight). Larger = less compute, more weights per mm²."""
    sram_kib: int = 1024
    overhead_frac: float = 0.15
    """Share of tile area for router, clock, power grid, redundancy muxes, stitching."""
    weight_bits: int = 4
    act_bits: int = 8
    kv_bits: int | None = None
    """KV-cache storage width; default = act_bits. 4 halves decoder SRAM at some accuracy cost."""
    accumulator_bits: int = 24
    hop_cycles: int = 4
    """Latency of one tile-to-tile hop."""
    stage_overhead_cycles: int = 64
    """Fixed per-item cost of LayerNorm, softmax, reductions and pipeline registers in a stage."""
    streams: int | None = None
    """Concurrent decode streams per model instance. Default: for encoder-decoder models, the
    fewest streams at which the decoder keeps up with the encoder; for decoder-only models,
    enough to fill the decoder pipeline."""

    @property
    def kv_bits_eff(self) -> int:
        return self.kv_bits or self.act_bits

    @property
    def transistors_per_mac(self) -> int:
        """Array multiplier (~30 transistors per partial-product cell) plus accumulator/register."""
        return 30 * self.weight_bits * self.act_bits + 48 * self.accumulator_bits

    @property
    def sram_bits(self) -> int:
        return self.sram_kib * 1024 * 8

    @property
    def side_mm(self) -> float:
        return math.sqrt(self.area_mm2)

    def with_(self, **changes) -> "TileConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class TileBudget:
    """How one tile's area is spent and what it holds."""

    tile: TileConfig
    process: Process
    params_per_tile: int
    macs_per_tile: int
    weight_area_mm2: float
    mac_area_mm2: float
    sram_area_mm2: float
    overhead_area_mm2: float

    @property
    def weight_bits_per_tile(self) -> int:
        return self.params_per_tile * self.tile.weight_bits

    @property
    def params_per_mm2(self) -> float:
        return self.params_per_tile / self.tile.area_mm2

    @property
    def sram_bits(self) -> int:
        return self.tile.sram_bits

    def breakdown(self) -> dict[str, float]:
        return {
            "weights": self.weight_area_mm2,
            "macs": self.mac_area_mm2,
            "sram": self.sram_area_mm2,
            "overhead": self.overhead_area_mm2,
        }


def tile_budget(process: Process, tile: TileConfig) -> TileBudget:
    """Solve for how many weights fit in a tile given the MAC:weight ratio.

    Area per parameter = weight cells + (1/mux) of a MAC unit. The SRAM and
    overhead are fixed, so the remaining area determines ``params_per_tile``.
    """
    if tile.mux_factor < 1:
        raise ValueError("mux_factor must be >= 1")
    overhead = tile.area_mm2 * tile.overhead_frac
    sram_area = tile.sram_bits / (process.sram_density_mbit_mm2 * 1e6)
    remaining = tile.area_mm2 - overhead - sram_area
    if remaining <= 0:
        raise ValueError(
            f"tile of {tile.area_mm2} mm² cannot hold {tile.sram_kib} KiB SRAM plus overhead on {process.name}"
        )
    weight_area_per_param = tile.weight_bits / (process.weight_cell_density_mbit_mm2 * 1e6)
    mac_area_per_param = tile.transistors_per_mac / (tile.mux_factor * process.logic_density_mtr_mm2 * 1e6)
    params = int(remaining / (weight_area_per_param + mac_area_per_param))
    macs = max(1, math.ceil(params / tile.mux_factor))
    return TileBudget(
        tile=tile,
        process=process,
        params_per_tile=params,
        macs_per_tile=macs,
        weight_area_mm2=params * weight_area_per_param,
        mac_area_mm2=macs * tile.transistors_per_mac / (process.logic_density_mtr_mm2 * 1e6),
        sram_area_mm2=sram_area,
        overhead_area_mm2=overhead,
    )


class Target(Protocol):
    kind: str
    power_budget_w: float
    defect_density_per_cm2: float

    @property
    def area_mm2(self) -> float: ...

    @property
    def width_mm(self) -> float: ...

    @property
    def height_mm(self) -> float: ...

    def grid(self, tile_side_mm: float) -> tuple[int, int]: ...


@dataclass(frozen=True)
class WaferTarget:
    """A stitched square fabric cut from a 300 mm wafer (Cerebras-style)."""

    kind: str = "wafer"
    diameter_mm: float = 300.0
    edge_exclusion_mm: float = 3.0
    reticle_w_mm: float = 26.0
    reticle_h_mm: float = 33.0
    fabric_side_mm: float | None = None
    """Override the inscribed-square side (e.g. 215 mm for a WSE-like 46,225 mm² fabric)."""
    power_budget_w: float = 20_000.0
    defect_density_per_cm2: float = 0.1

    @property
    def inscribed_side_mm(self) -> float:
        if self.fabric_side_mm is not None:
            return self.fabric_side_mm
        return (self.diameter_mm - 2 * self.edge_exclusion_mm) / math.sqrt(2)

    @property
    def reticle_grid(self) -> tuple[int, int]:
        side = self.inscribed_side_mm
        return int(side // self.reticle_w_mm), int(side // self.reticle_h_mm)

    @property
    def width_mm(self) -> float:
        return self.reticle_grid[0] * self.reticle_w_mm

    @property
    def height_mm(self) -> float:
        return self.reticle_grid[1] * self.reticle_h_mm

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def reticle_fields(self) -> int:
        c, r = self.reticle_grid
        return c * r

    def grid(self, tile_side_mm: float) -> tuple[int, int]:
        return int(self.width_mm // tile_side_mm), int(self.height_mm // tile_side_mm)

    def with_(self, **changes) -> "WaferTarget":
        return replace(self, **changes)


@dataclass(frozen=True)
class DieTarget:
    """A single (reticle-limited) die, for comparison with conventional hardwired-model chips."""

    kind: str = "die"
    die_area_mm2: float = 815.0
    power_budget_w: float = 350.0
    defect_density_per_cm2: float = 0.1
    aspect: float = 1.0
    """width / height."""

    @property
    def area_mm2(self) -> float:
        return self.die_area_mm2

    @property
    def width_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 * self.aspect)

    @property
    def height_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 / self.aspect)

    def grid(self, tile_side_mm: float) -> tuple[int, int]:
        return int(self.width_mm // tile_side_mm), int(self.height_mm // tile_side_mm)

    def with_(self, **changes) -> "DieTarget":
        return replace(self, **changes)


@dataclass(frozen=True)
class YieldReport:
    tile_area_cm2: float
    defect_density_per_cm2: float
    tile_yield: float
    tiles_physical: int
    expected_dead: float
    spares_reserved: int
    tiles_usable: int
    success_probability: float
    """Probability the fabric has no more dead tiles than spares."""


def _poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        total += math.exp(i * math.log(lam) - lam - math.lgamma(i + 1))
    return min(1.0, total)


def yield_model(tile: TileConfig, target: Target, tiles_physical: int, sigma: float = 4.0) -> YieldReport:
    """Poisson defect model with tile-level redundancy.

    Each tile survives with probability exp(-D0 * A). The fabric reserves enough
    spares to cover the expected dead tiles plus ``sigma`` standard deviations,
    so a wafer almost never fails outright; dead tiles are bypassed by the
    router and their work moves to spares.
    """
    area_cm2 = tile.area_mm2 / 100.0
    y = math.exp(-target.defect_density_per_cm2 * area_cm2)
    lam = tiles_physical * (1.0 - y)
    spares = int(math.ceil(lam + sigma * math.sqrt(lam))) if lam > 0 else 0
    spares = min(spares, tiles_physical)
    return YieldReport(
        tile_area_cm2=area_cm2,
        defect_density_per_cm2=target.defect_density_per_cm2,
        tile_yield=y,
        tiles_physical=tiles_physical,
        expected_dead=lam,
        spares_reserved=spares,
        tiles_usable=tiles_physical - spares,
        success_probability=_poisson_cdf(spares, lam),
    )
