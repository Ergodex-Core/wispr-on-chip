import math

import pytest

from wispr_on_chip.fabric import DieTarget, TileConfig, WaferTarget, tile_budget, yield_model
from wispr_on_chip.process import get_process


def test_tile_budget_conserves_area():
    proc = get_process("n5")
    tile = TileConfig(area_mm2=1.0, mux_factor=1024)
    b = tile_budget(proc, tile)
    total = sum(b.breakdown().values())
    assert total == pytest.approx(tile.area_mm2, rel=1e-3)
    assert b.params_per_tile > 1_000_000
    assert b.macs_per_tile == math.ceil(b.params_per_tile / tile.mux_factor)


def test_more_mux_means_more_weights_per_tile():
    proc = get_process("n5")
    prev = 0
    for mux in (1, 16, 256, 4096, 65536):
        p = tile_budget(proc, TileConfig(mux_factor=mux)).params_per_tile
        assert p > prev
        prev = p


def test_fully_spatial_tile_is_mac_dominated():
    proc = get_process("n5")
    b = tile_budget(proc, TileConfig(mux_factor=1))
    assert b.mac_area_mm2 > b.weight_area_mm2


def test_tile_too_small_for_sram_raises():
    with pytest.raises(ValueError):
        tile_budget(get_process("n5"), TileConfig(area_mm2=0.05, sram_kib=4096))


def test_wafer_geometry():
    w = WaferTarget()
    assert w.inscribed_side_mm == pytest.approx((300 - 6) / math.sqrt(2))
    cols, rows = w.reticle_grid
    assert cols * 26 <= w.inscribed_side_mm and rows * 33 <= w.inscribed_side_mm
    assert w.area_mm2 == pytest.approx(cols * 26 * rows * 33)
    assert 30_000 < w.area_mm2 < 50_000
    wse = WaferTarget(fabric_side_mm=215)
    assert wse.area_mm2 > w.area_mm2


def test_die_grid():
    d = DieTarget(die_area_mm2=815)
    c, r = d.grid(1.0)
    assert c == r == 28
    assert c * r <= 815


def test_yield_model_reserves_spares_and_is_safe():
    tile = TileConfig(area_mm2=1.0)
    y = yield_model(tile, WaferTarget(defect_density_per_cm2=0.1), tiles_physical=40_000)
    assert y.tile_yield == pytest.approx(math.exp(-0.001))
    assert y.expected_dead == pytest.approx(40_000 * (1 - y.tile_yield))
    assert y.spares_reserved > y.expected_dead
    assert y.tiles_usable == 40_000 - y.spares_reserved
    assert y.success_probability > 0.999


def test_zero_defects_needs_no_spares():
    y = yield_model(TileConfig(), DieTarget(defect_density_per_cm2=0.0), tiles_physical=100)
    assert y.spares_reserved == 0 and y.tiles_usable == 100 and y.success_probability == 1.0
