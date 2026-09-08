"""Sanity tests of every golden op against a float implementation, plus self-consistency tests.
These do not need the checkpoint (random weights / statistics)."""
import math

import numpy as np
import pytest

from golden import luts
from golden.fixedpoint import (I64, REQ_S2_BY_BITS, dyn_quant_rows, matmul_i8, requant, requant_wide, rowfac_eff,
                               rowfac_pack, rsr, sat, scaled_add)
from golden.quant import (D_MODEL, HEAD_DIM, KV_DIM, N_HEAD, N_KV_HEAD, AttnQ, make_attn, make_linear, make_rms,
                          make_rope, make_silu)
from golden.minicpm_int import attention, embed_rows, linear, rmsnorm, rope, silu_gate

rng = np.random.default_rng(1234)


def test_rsr_rounding():
    assert rsr(np.array([5, -5, 4, -4, 3, -3]), 1).tolist() == [3, -2, 2, -2, 2, -1]  # round half up
    assert rsr(np.array([7]), 0).tolist() == [7]
    assert sat(np.array([200, -200, 5]), 8).tolist() == [127, -128, 5]


def test_dyn_quant_int16_rows():
    x = (rng.standard_normal((16, 2048)) * 4000).astype(I64)
    x[3] = 0
    a8, rf = dyn_quant_rows(x)
    assert a8.dtype == np.int8 and rf.dtype == np.int32
    assert np.all(np.abs(a8).max(axis=1) <= 127)
    eff = rowfac_eff(rf)
    assert np.array_equal(eff[:3], np.abs(x[:3]).max(axis=1)) and eff[3] == 1 and np.all(rf >> 16 == 0)
    ideal = x * 127.0 / np.maximum(eff, 1)[:, None]
    assert np.abs(a8 - ideal).max() <= 0.6   # 0.5 rounding + reciprocal approximation


def test_dyn_quant_int32_rows():
    """The 32-bit rule: maxabs is normalised to 16 bits (m16 << b) and the shift grows by b."""
    x = (rng.standard_normal((8, 6144)) * 3e7).astype(I64)
    x[1] = (rng.standard_normal(6144) * 100).astype(I64)          # small row -> b = 0
    x[2, 5] = 2 ** 30 - 1
    a8, rf = dyn_quant_rows(x)
    eff = rowfac_eff(rf)
    b = rf >> 16
    assert b[1] == 0 and b[2] == 14 and np.all(eff <= np.abs(x).max(axis=1)) and np.all(eff >= np.abs(x).max(axis=1) - (1 << b))
    ideal = np.clip(x * 127.0 / eff[:, None], -128, 127)
    assert np.abs(a8 - ideal).max() <= 1.01     # m16 truncation adds up to one LSB at the row maximum
    assert np.array_equal(rowfac_pack(eff >> b, b), rf)


def test_requant_vs_float():
    M, N = 64, 32
    acc = rng.integers(-(1 << 24), 1 << 24, size=(M, N)).astype(np.int32)
    s_in, s_w = 1e-3, rng.uniform(1e-3, 3e-3, N)
    bias = np.zeros(N)
    for bits, s_out in ((8, 0.05), (16, 5e-4), (32, 2e-9)):
        s2 = REQ_S2_BY_BITS[bits]
        c = (s_in / 127) * s_w / s_out
        s1 = 30 - s2 - int(math.floor(math.log2(c.max())))
        assert s1 >= 0
        mult = np.round(c * 2.0 ** (s1 + s2)).astype(np.int32)
        rowmax = rng.integers(100, 1 << 30, M).astype(I64) if bits == 32 else rng.integers(100, 30000, M).astype(I64)
        bl = np.array([int(v).bit_length() for v in rowmax]); bb = np.maximum(bl - 16, 0)
        rf = rowfac_pack(rowmax >> bb, bb)
        y = requant(acc, mult, np.zeros(N, np.int32), s1, rf, bits)
        ref = (acc.astype(float) * rowfac_eff(rf)[:, None].astype(float) * (s_in / 127) * s_w[None, :] + bias[None, :]) / s_out
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        ref = np.clip(np.round(ref), lo, hi)
        tol = 1 if bits < 32 else 2 ** 20 * 1e-3 + 2   # 32-bit outputs: relative 1e-3 of full scale is plenty for the argmax path
        assert np.abs(y.astype(np.int64) - ref).max() <= tol, bits


def test_matmul_exact_and_order_invariant():
    a = rng.integers(-128, 128, size=(70, 2048)).astype(np.int8)
    w = rng.integers(-128, 128, size=(2048, 96)).astype(np.int8)
    acc = matmul_i8(a, w)
    ref = a.astype(np.int64) @ w.astype(np.int64)
    assert np.array_equal(acc, ref)
    # both implementations (torch int_mm / float64) agree, including M = 1
    assert np.array_equal(matmul_i8(a[:1], w), ref[:1])
    tiles = [a[:, k:k + 32].astype(np.int64) @ w[k:k + 32].astype(np.int64) for k in range(0, 2048, 32)]
    rng.shuffle(tiles)
    assert np.array_equal(sum(tiles), ref)
    # unsigned activations (P.V hi/lo split) also exact
    p = rng.integers(0, 32769, size=(8, 64)).astype(np.int64)
    v = rng.integers(-128, 128, size=(64, 32)).astype(np.int8)
    hi, lo = p >> 8, p & 255
    assert np.array_equal((matmul_i8(hi, v).astype(np.int64) << 8) + matmul_i8(lo, v), p @ v.astype(np.int64))


def test_rmsnorm_vs_float():
    s_in = 5300.0 * 2 / (2 ** 31 - 1)
    gamma = rng.uniform(0.2, 3.0, D_MODEL)
    nq = make_rms("t", gamma, s_in, 20.0)
    x = (rng.standard_normal((16, D_MODEL)) * 1.0 / s_in).astype(I64)
    x[5, 17] = int(5000 / s_in)      # massive activation
    x[6] = 3                         # tiny row -> eps path
    x[7] = (rng.standard_normal(D_MODEL) * 0.001 / s_in).astype(I64)
    y = rmsnorm(nq, x).astype(np.float64)
    xf = x * s_in
    ref = xf / np.sqrt((xf ** 2).mean(axis=1, keepdims=True) + 1e-6) * gamma
    ref_q = np.clip(ref / (2.0 ** -nq.F), -32768, 32767)     # the massive row saturates the int16 output
    err = np.abs(y - ref_q)
    assert err.max() <= 2.0


def test_rope_vs_float():
    table = luts.rope_table(64, HEAD_DIM, 5e6)
    s16 = np.full(N_HEAD, 12.0 * 2 / 32767); s8 = np.full(N_HEAD, 12.0 / 127)
    rq = make_rope("t", s16, s8)
    x = rng.integers(-16000, 16000, size=(5, N_HEAD * HEAD_DIM)).astype(np.int16)
    pos = np.array([0, 1, 7, 40, 63])
    y = rope(rq, x, pos, table).astype(np.float64)
    inv = 1.0 / (5e6 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    xh = (x * s16[0]).reshape(5, N_HEAD, HEAD_DIM)
    x1, x2 = xh[..., :64], xh[..., 64:]
    ref = np.concatenate([x1 * c[:, None] - x2 * s[:, None], x2 * c[:, None] + x1 * s[:, None]], axis=2).reshape(5, -1) / s8[0]
    assert np.abs(y - np.clip(np.round(ref), -128, 127)).max() <= 1


def test_silu_gate_vs_float():
    s_gate, s_up = 60.0 * 2 / 32767, 70.0 * 2 / 32767
    sq = make_silu("t", s_gate, s_up)
    g = rng.integers(-32768, 32768, size=(4, 6144)).astype(np.int16)
    u = rng.integers(-32768, 32768, size=(4, 6144)).astype(np.int16)
    h = silu_gate(sq, g, u).astype(np.float64)
    gf = g * s_gate
    ref = gf / (1 + np.exp(-gf)) * (u * s_up) / (s_gate * s_up)
    # sigmoid table: |x| clamps at 8 (sigmoid(8) = 0.99966), so silu16 is off by <= 3.5e-4 * |g| + 0.5, times |u|
    assert np.abs(h - ref).max() <= (3.5e-4 * 32768 + 0.5) * 32768 + 1


def test_sigmoid_and_exp_and_rsqrt_luts():
    xf = np.arange(-2048, 2048)
    sg = luts.sigmoid_lut(xf)
    assert np.all(np.diff(sg) >= 0) and abs(sg[2048] - 16384) <= 1
    assert np.abs(sg - 32768 / (1 + np.exp(-xf / 256))).max() < 4
    d = np.arange(4096)
    p = luts.exp_lut(d)
    assert p[0] == 32768 and p[-1] == 0 and np.all(np.diff(p) <= 0)
    assert np.abs(p - np.exp(-d / 256) * 32768).max() < 17
    m = rng.integers(1 << 28, 1 << 30, 10000).astype(I64)
    y1 = luts.rsqrt_newton(m)
    assert (np.abs(y1 - (1 << 30) / np.sqrt(m)) / y1).max() < 1e-4


def _attn_float(q8, k8, v8, at: AttnQ, n_keys, causal_offset=None):
    Mq = q8.shape[0]
    out = np.zeros((Mq, D_MODEL))
    for h in range(N_HEAD):
        hs = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        kv = h // (N_HEAD // N_KV_HEAD)
        ks = slice(kv * HEAD_DIM, (kv + 1) * HEAD_DIM)
        s = (q8[:, hs].astype(float) @ k8[:n_keys, ks].T.astype(float)) * at.s_q[h] * at.s_k[kv] / math.sqrt(HEAD_DIM)
        if causal_offset is not None:
            mask = np.arange(n_keys)[None, :] > (causal_offset + np.arange(Mq))[:, None]
            s = np.where(mask, -np.inf, s)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out[:, hs] = p @ v8[:n_keys, ks].astype(float) * 128   # out16 units = s_v/128
    return out


@pytest.mark.parametrize("n_keys,causal", [(200, None), (64, None), (1, None), (130, 0), (7, 0), (100, 60)])
def test_attention_vs_float(n_keys, causal):
    at = make_attn("t", rng.uniform(0.04, 0.12, N_HEAD), rng.uniform(0.03, 0.1, N_KV_HEAD), 0.05)
    Mq = 40 if causal is None else n_keys - causal
    q8 = rng.integers(-128, 128, size=(Mq, D_MODEL)).astype(np.int8)
    k8 = rng.integers(-128, 128, size=(n_keys, KV_DIM)).astype(np.int8)
    v8 = rng.integers(-128, 128, size=(n_keys, KV_DIM)).astype(np.int8)
    out = attention(at, q8, k8, v8, n_keys, causal_offset=causal).astype(np.float64)
    ref = _attn_float(q8, k8, v8, at, n_keys, causal)
    assert np.abs(out - ref).max() <= 64  # 0.5 v-LSB in out16 units (128 per v unit)


def test_attention_ignores_masked_garbage():
    at = make_attn("t", np.full(N_HEAD, 0.05), np.full(N_KV_HEAD, 0.05), 0.05)
    q8 = rng.integers(-128, 128, size=(3, D_MODEL)).astype(np.int8)
    k8 = rng.integers(-128, 128, size=(128, KV_DIM)).astype(np.int8)
    v8 = rng.integers(-128, 128, size=(128, KV_DIM)).astype(np.int8)
    a = attention(at, q8, k8, v8, 70)
    k8[70:], v8[70:] = 127, -128
    b = attention(at, q8, k8, v8, 70)
    assert np.array_equal(a, b)


def test_attention_prefill_equals_incremental():
    """Causal prefill over M rows == the same rows decoded one at a time (the chip runs both)."""
    at = make_attn("t", rng.uniform(0.04, 0.12, N_HEAD), rng.uniform(0.03, 0.1, N_KV_HEAD), 0.05)
    n = 70
    q8 = rng.integers(-128, 128, size=(n, D_MODEL)).astype(np.int8)
    k8 = rng.integers(-128, 128, size=(n, KV_DIM)).astype(np.int8)
    v8 = rng.integers(-128, 128, size=(n, KV_DIM)).astype(np.int8)
    full = attention(at, q8, k8, v8, n, causal_offset=0)
    for m in (0, 1, 63, 64, 69):
        one = attention(at, q8[m:m + 1], k8[:m + 1], v8[:m + 1], m + 1, causal_offset=m)
        assert np.array_equal(one[0], full[m]), m
    # chunked prefill (rows 32..69 with keys 0..69) is identical too
    part = attention(at, q8[32:], k8, v8, n, causal_offset=32)
    assert np.array_equal(part, full[32:])


def test_linear_dynamic_vs_float():
    W = rng.standard_normal((256, 2048)) * 0.05
    L = make_linear("t", W, None, 2.0 ** -9, True, 0.02, 16)
    x16 = (rng.standard_normal((50, 2048)) * 3000).astype(I64)
    a8, rf = dyn_quant_rows(x16)
    y = linear(L, a8, rf).astype(np.float64)
    ref = (x16 * 2.0 ** -9) @ W.T
    assert np.abs(y - np.clip(np.round(ref / 0.02), -32768, 32767)).max() <= 40   # int8 input quantisation dominates


def test_linear_int32_out_vs_float():
    W = rng.standard_normal((2048, 6144)) * 0.02
    s_out = 1500.0 * 2 / (2 ** 31 - 1)
    L = make_linear("t", W, None, 1e-7, True, s_out, 32)
    x = (rng.standard_normal((6, 6144)) * 2e7).astype(I64)       # real values ~2 with a 32-bit row
    a8, rf = dyn_quant_rows(x)
    y = linear(L, a8, rf).astype(np.float64) * s_out
    ref = (x * 1e-7) @ W.T
    assert np.abs(y - ref).max() / np.abs(ref).max() < 0.02


def test_requant_int32_folds_rowfac_exponent():
    """y = rsr(t*(m16<<b), 24) == rsr(t*m16, 24-b): the golden folds b into the shift; check against Python ints."""
    acc = np.array([[123456789, -987654321, 77, 2 ** 31 - 1]], dtype=np.int32)
    mult = np.array([2 ** 31 - 1, 1234567, 2 ** 30, 3], dtype=np.int32)
    for m16, b in ((65535, 15), (1, 0), (40000, 7)):
        rf = rowfac_pack(np.array([m16]), np.array([b]))
        y = requant(acc, mult, np.zeros(4, np.int32), 5, rf, 32)
        for j in range(4):
            t = (int(acc[0, j]) * int(mult[j]) + 16) >> 5
            t = max(-(1 << 47), min((1 << 47) - 1, t))
            u = t * (m16 << b)
            exp = max(-(1 << 31), min((1 << 31) - 1, (u + (1 << 23)) >> 24))
            assert int(y[0, j]) == exp, (m16, b, j)


def test_wide_mode_argmax():
    W = rng.standard_normal((4096, 2048)) * 0.05
    L = make_linear("lm", W, None, 2.0 ** -9, True, 1.0, 0)
    a8 = rng.integers(-128, 128, size=(3, 2048)).astype(np.int8)
    t = linear(L, a8)
    ref = a8.astype(float) @ (L.w8.astype(float) * L.s_w[None, :])
    assert t.shape == (3, 4096) and np.array_equal(np.argmax(t, axis=1), np.argmax(ref, axis=1))


def test_embed_and_scaled_add():
    e8 = rng.integers(-128, 128, size=(100, D_MODEL)).astype(np.int8)
    resmult = rng.integers(1000, 1 << 24, 100).astype(np.int32)
    x = embed_rows(e8, resmult, [3, 99, 0])
    assert x.dtype == np.int32 and np.array_equal(x[1], np.clip(e8[99].astype(np.int64) * resmult[99], -(1 << 31), (1 << 31) - 1))
    y = scaled_add(np.array([2 ** 31 - 1]), 1 << 16, np.array([2 ** 31 - 1]), 1 << 16, 32)
    assert y.tolist() == [2 ** 31 - 1]
    y = scaled_add(np.array([1000]), 40000, np.array([-7]), 65536, 32)
    assert y.tolist() == [round(1000 * 40000 / 65536) - 7]
