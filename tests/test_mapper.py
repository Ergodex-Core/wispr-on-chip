import json

import pytest

from wispr_on_chip import DieTarget, TileConfig, WaferTarget, auto_mux, get_model, get_process, plan
from wispr_on_chip.mapper import pack_instances
from wispr_on_chip.floorplan import render_ascii, render_svg, tile_assignments


def test_whisper_large_fits_many_times_on_a_wafer():
    p = plan(get_model("whisper-large-v3"), get_process("n5"), TileConfig(), WaferTarget())
    assert p.fits
    assert p.instances >= 10
    assert p.tiles_used <= p.yield_report.tiles_usable
    assert sum(pl.tiles for pl in p.placements) == p.tiles_per_instance
    assert p.perf.decode_tokens_per_s_single_stream > 5_000
    assert p.perf.realtime_factor_single_stream > 50
    assert p.power.total_w_peak > 0


def test_encoder_stages_are_sram_bound_at_default_sram():
    p = plan(get_model("whisper-large-v3"), get_process("n5"), TileConfig(), WaferTarget())
    enc = [pl for pl in p.placements if pl.stage.kind == "encoder"]
    assert all(pl.bound == "sram" for pl in enc)
    bigger = plan(get_model("whisper-large-v3"), get_process("n5"), TileConfig(area_mm2=4.0, sram_kib=8192), WaferTarget())
    assert bigger.area_per_instance_mm2 < p.area_per_instance_mm2


def test_kv_bits_shrink_decoder():
    m, pr = get_model("whisper-large-v3"), get_process("n5")
    eight = plan(m, pr, TileConfig(streams=4), WaferTarget())
    four = plan(m, pr, TileConfig(streams=4, kv_bits=4), WaferTarget())
    assert four.tiles_per_instance < eight.tiles_per_instance


def test_auto_streams_balances_decoder_with_encoder():
    p = plan(get_model("whisper-large-v3"), get_process("n5"), TileConfig(), WaferTarget())
    assert 1 <= p.perf.streams <= p.perf.decode_pipeline_stages
    assert p.perf.decoder_windows_per_s_per_instance >= p.perf.encoder_windows_per_s_per_instance * 0.999
    llm = plan(get_model("llama-3.1-8b"), get_process("n5"), TileConfig(), WaferTarget())
    assert llm.perf.streams == llm.perf.decode_pipeline_stages


def test_streams_scale_decoder_sram():
    m, pr = get_model("whisper-medium"), get_process("n5")
    one = plan(m, pr, TileConfig(streams=1), WaferTarget())
    many = plan(m, pr, TileConfig(streams=64), WaferTarget())
    assert many.tiles_per_instance > one.tiles_per_instance
    assert many.perf.pipeline_fill == 1.0
    assert one.perf.pipeline_fill < 1.0
    assert one.perf.decode_tokens_per_s_per_instance < many.perf.decode_tokens_per_s_per_instance


def test_single_die_calibration_against_taalas_class_chip():
    """An 815 mm² N6 die should hold Llama-8B at 3-bit and decode tens of thousands of tok/s."""
    p = auto_mux(get_model("llama-3.1-8b"), get_process("n6"), TileConfig(weight_bits=3, streams=1), DieTarget(815))
    assert p.fits and p.instances == 1
    assert 256 <= p.tile.mux_factor <= 4096
    tps = p.perf.decode_tokens_per_s_single_stream
    assert 5_000 < tps < 100_000


def test_auto_mux_is_minimal():
    m, pr, tg = get_model("whisper-small"), get_process("n5"), DieTarget(300)
    p = auto_mux(m, pr, TileConfig(streams=1), tg)
    assert p.fits
    smaller = plan(m, pr, TileConfig(streams=1, mux_factor=p.tile.mux_factor - 1), tg) if p.tile.mux_factor > 1 else None
    assert smaller is None or not smaller.fits


def test_auto_mux_raises_when_impossible():
    with pytest.raises(ValueError):
        auto_mux(get_model("llama-3.1-8b"), get_process("n5"), TileConfig(), DieTarget(50))


def test_does_not_fit_reports_fraction():
    p = plan(get_model("llama-3.1-8b"), get_process("n5"), TileConfig(), DieTarget(50))
    assert not p.fits and 0 < p.fit_fraction < 1 and p.instances == 0


def test_power_limit_scales_throughput():
    m, pr = get_model("whisper-large-v3"), get_process("n5")
    unlimited = plan(m, pr, TileConfig(), WaferTarget(power_budget_w=1e9))
    budget = unlimited.power.leakage_w + unlimited.power.dynamic_w_total_peak / 2
    limited = plan(m, pr, TileConfig(), WaferTarget(power_budget_w=budget))
    assert unlimited.power.power_scale == 1.0
    assert 0 < limited.power.power_scale < 1
    assert limited.perf.sustained_tokens_per_s_total < unlimited.perf.sustained_tokens_per_s_total


def test_json_roundtrip():
    p = plan(get_model("whisper-tiny"), get_process("n3"), TileConfig(), WaferTarget())
    d = json.loads(json.dumps(p.to_dict(), default=str))
    assert d["model"] == "whisper-tiny" and d["instances"] == p.instances
    assert len(d["stages"]) == len(p.placements)


def test_pack_instances_prefers_square_blocks_and_respects_spares():
    n, block, grid = pack_instances(100, 100, 100, spares=0)
    assert n == 100 and block == (10, 10) and grid == (10, 10)
    n, block, grid = pack_instances(100, 100, 100, spares=1)
    assert n == 99  # one block's worth of slack must be freed for the spare
    n, _, _ = pack_instances(100, 100, 100, spares=0, max_instances=3)
    assert n == 3
    assert pack_instances(10, 10, 101, spares=0)[0] == 0


def test_instances_respect_block_packing():
    p = plan(get_model("whisper-large-v3"), get_process("n5"), TileConfig(), WaferTarget())
    bw, bh = p.instance_block
    assert bw * bh >= p.tiles_per_instance
    assert p.instances == min(p.instances, p.block_grid[0] * p.block_grid[1])
    assert p.tiles_used + p.yield_report.spares_reserved <= p.tiles_physical


def test_floorplan_covers_every_tile():
    p = plan(get_model("whisper-base"), get_process("n5"), TileConfig(), WaferTarget())
    a = tile_assignments(p)
    assert len(a) == p.tiles_physical
    assert sum(1 for k, _ in a if k == "spare") == p.yield_report.spares_reserved
    assert sum(1 for k, i in a if i >= 0) == p.tiles_used
    assert sum(1 for k, i in a if k == "encoder") == p.instances * sum(pl.tiles for pl in p.placements if pl.stage.kind == "encoder")
    txt = render_ascii(p, width=40)
    assert "E=encoder" in txt and "E" in txt.splitlines()[0] + txt.splitlines()[1]
    svg = render_svg(p)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
