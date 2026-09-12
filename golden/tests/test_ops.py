"""Sanity tests of every golden op against a float implementation, plus self-consistency tests."""
import math

import numpy as np
import pytest

from golden import luts
from golden.fixedpoint import (I64, REQ_B_FRAC, REQ_S2_BY_BITS, dyn_quant_rows, matmul_i8, requant, rsr, sat,
                               scaled_add)
from golden.quant import D_MODEL, HEAD_DIM, N_HEAD, AttnQ, GeluQ, LayerNormQ, make_attn, make_gelu, make_linear, make_ln
from golden.whisper_int import IntWhisper, attention, gelu16, layernorm, linear

rng = np.random.default_rng(1234)


def test_rsr_rounding():
    assert rsr(np.array([5, -5, 4, -4, 3, -3]), 1).tolist() == [3, -2, 2, -2, 2, -1]  # round half up
    assert rsr(np.array([7]), 0).tolist() == [7]
    assert sat(np.array([200, -200, 5]), 8).tolist() == [127, -128, 5]


def test_dyn_quant():
    x = (rng.standard_normal((16, 384)) * 4000).astype(I64)
    x[3] = 0
    a8, mx = dyn_quant_rows(x)
    assert mx.dtype == np.uint16 and a8.dtype == np.int8
    assert np.all(np.abs(a8).max(axis=1) <= 127)
    ideal = x * 127.0 / np.maximum(mx, 1)[:, None]
    assert np.abs(a8 - ideal).max() <= 0.6   # 0.5 rounding + reciprocal approximation
    assert np.array_equal(mx[:3], np.abs(x[:3]).max(axis=1)) and mx[3] == 1


def test_requant_vs_float():
    M, N = 64, 32
    acc = rng.integers(-(1 << 24), 1 << 24, size=(M, N)).astype(np.int32)
    s_in, s_w, s_out = 1e-3, rng.uniform(1e-3, 3e-3, N), 0.05
    bias = rng.uniform(-2, 2, N)
    rowfac = rng.integers(100, 30000, M).astype(np.uint16)
    for bits in (8, 16):
        s2 = REQ_S2_BY_BITS[bits]
        c = (s_in / 127) * s_w / s_out
        s1 = 30 - s2 - int(math.floor(math.log2(c.max())))
        mult = np.round(c * 2.0 ** (s1 + s2)).astype(np.int32)
        b = np.round(bias / s_out * (1 << REQ_B_FRAC)).astype(np.int32)
        y = requant(acc, mult, b, s1, rowfac, bits)
        ref = (acc.astype(float) * rowfac[:, None].astype(float) * (s_in / 127) * s_w[None, :] + bias[None, :]) / s_out
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        ref = np.clip(np.round(ref), lo, hi)
        assert np.abs(y.astype(np.int64) - ref).max() <= 1


def test_matmul_exact_and_order_invariant():
    a = rng.integers(-128, 128, size=(70, 1536)).astype(np.int8)
    w = rng.integers(-128, 128, size=(1536, 96)).astype(np.int8)
    acc = matmul_i8(a, w)
    ref = a.astype(np.int64) @ w.astype(np.int64)
    assert np.array_equal(acc, ref)
    # k-tile order invariance (exact integer arithmetic)
    tiles = [a[:, k:k + 32].astype(np.int64) @ w[k:k + 32].astype(np.int64) for k in range(0, 1536, 32)]
    rng.shuffle(tiles)
    assert np.array_equal(sum(tiles), ref)
    # unsigned activations (P.V hi/lo split) also exact
    p = rng.integers(0, 32769, size=(8, 64)).astype(np.int64)
    v = rng.integers(-128, 128, size=(64, 32)).astype(np.int8)
    hi, lo = p >> 8, p & 255
    assert np.array_equal((matmul_i8(hi, v).astype(np.int64) << 8) + matmul_i8(lo, v), p @ v.astype(np.int64))


def test_layernorm_vs_float():
    s_in = 274.0 * 2 / 32767
    gamma = rng.uniform(0.2, 3.0, D_MODEL)
    beta = rng.uniform(-1, 1, D_MODEL)
    ln = make_ln("t", gamma, beta, s_in, 20.0)
    x = (rng.standard_normal((32, D_MODEL)) * 300).astype(I64)
    x[5, 17] = 3000   # dominant channel
    x[6] = 7          # constant row -> eps path
    y = layernorm(ln, x).astype(np.float64)
    xf = x * s_in
    mu = xf.mean(axis=1, keepdims=True)
    var = ((xf - mu) ** 2).mean(axis=1, keepdims=True)
    ref = (xf - mu) / np.sqrt(var + 1e-5) * gamma + beta
    ref_q = ref / (2.0 ** -ln.F)
    err = np.abs(y - ref_q)
    assert err[:6].max() <= 2.0 and err[7:].max() <= 2.0
    assert np.abs(y[6] - beta / (2.0 ** -ln.F)).max() <= 2.0


def test_gelu16_vs_float():
    s_in = 30.0 / 32767
    g = make_gelu("t", "phi16", s_in)
    h = rng.integers(-32768, 32768, size=(4, 1536)).astype(np.int16)
    y = gelu16(g, h).astype(np.float64)
    x = h * s_in
    ref = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))) / s_in
    assert np.abs(y - ref).max() <= 3.0   # int16 LSB units (Phi interpolation + rounding)


def test_gelu8_table():
    g = make_gelu("t", "lut8", 12.5 / 127, 12.5 / 127)
    t = np.array(g.table)
    assert t.shape == (256,) and t[128] == 0 and t[255] == 127 and t.min() >= -2


def test_exp_and_rsqrt_luts():
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
        s = (q8[:, hs].astype(float) @ k8[:n_keys, hs].T.astype(float)) * at.s_q[h] * at.s_k[h] / math.sqrt(HEAD_DIM)
        if causal_offset is not None:
            mask = np.arange(n_keys)[None, :] > (causal_offset + np.arange(Mq))[:, None]
            s = np.where(mask, -np.inf, s)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out[:, hs] = p @ v8[:n_keys, hs].astype(float) * 128   # out16 units = s_v/128
    return out


@pytest.mark.parametrize("n_keys,causal", [(200, None), (64, None), (1, None), (130, 0), (7, 0)])
def test_attention_vs_float(n_keys, causal):
    at = make_attn("t", rng.uniform(0.04, 0.08, N_HEAD), rng.uniform(0.03, 0.07, N_HEAD), 0.05)
    Mq = 40 if causal is None else n_keys
    q8 = rng.integers(-128, 128, size=(Mq, D_MODEL)).astype(np.int8)
    k8 = rng.integers(-128, 128, size=(n_keys, D_MODEL)).astype(np.int8)
    v8 = rng.integers(-128, 128, size=(n_keys, D_MODEL)).astype(np.int8)
    out = attention(at, q8, k8, v8, n_keys, causal_offset=causal).astype(np.float64)
    ref = _attn_float(q8, k8, v8, at, n_keys, causal)
    assert np.abs(out - ref).max() <= 64  # 0.5 v-LSB in out16 units (128 per v unit)


def test_attention_ignores_masked_garbage():
    """Keys beyond n_keys (or beyond the causal position) must not influence the result at all."""
    at = make_attn("t", np.full(N_HEAD, 0.05), np.full(N_HEAD, 0.05), 0.05)
    q8 = rng.integers(-128, 128, size=(3, D_MODEL)).astype(np.int8)
    k8 = rng.integers(-128, 128, size=(128, D_MODEL)).astype(np.int8)
    v8 = rng.integers(-128, 128, size=(128, D_MODEL)).astype(np.int8)
    a = attention(at, q8, k8, v8, 70)
    k8[70:], v8[70:] = 127, -128
    b = attention(at, q8, k8, v8, 70)
    assert np.array_equal(a, b)


def test_linear_static_vs_float():
    W = rng.standard_normal((384, 288)) * 0.1
    b = rng.standard_normal(384)
    L = make_linear("t", W, b, 0.0114, False, 0.0007, 16)
    a8 = rng.integers(-128, 128, size=(50, 288)).astype(np.int8)
    y = linear(L, a8).astype(np.float64)
    ref = (a8.astype(float) * 0.0114) @ (L.w8.astype(float) * L.s_w[None, :]) + b
    assert np.abs(y - np.clip(np.round(ref / 0.0007), -32768, 32767)).max() <= 1


def test_im2col_matches_conv(qmodel):
    import torch
    import torch.nn.functional as F
    m = IntWhisper(qmodel, dump=False)
    x = rng.integers(-128, 128, size=(64, 96)).astype(np.int8)
    rows = m.im2col(x, 1)
    assert rows.shape == (64, 288)
    xt = torch.from_numpy(x.astype(np.float32)).T[None]        # [1, C, T]
    for stride in (1, 2):
        rows = m.im2col(x, stride)
        w = torch.from_numpy(rng.standard_normal((5, 96, 3)).astype(np.float32))
        ref = F.conv1d(xt, w, stride=stride, padding=1)[0].T.numpy()  # [T/stride, 5]
        # our K index = tap*96 + bin  <->  w[n, bin, tap]
        Wmat = w.permute(0, 2, 1).reshape(5, 288).numpy()
        got = rows.astype(np.float32) @ Wmat.T
        assert np.allclose(got, ref, atol=1e-3)


def test_scaled_add_saturates():
    y = scaled_add(np.array([30000]), 1 << 16, np.array([30000]), 1 << 16, 16)
    assert y.tolist() == [32767]


def test_encoder_smoke(qmodel):
    m = IntWhisper(qmodel, dump=False)
    mel8 = rng.integers(-100, 100, size=(128, 96)).astype(np.int8)
    mel8[:, 80:] = 0
    enc8, rf = m.encoder(mel8)
    assert enc8.shape == (64, 384) and rf.shape == (64,) and enc8.dtype == np.int8
