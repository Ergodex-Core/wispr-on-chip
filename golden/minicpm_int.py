"""The integer golden model: exact fixed-point MiniCPM5-2B datapath. THIS IS THE SPEC FOR THE RTL.

Every op is a pure function on integer numpy arrays, tiled/ordered the way the hardware does it:
  * matmuls accumulate exactly (order-independent), requant at the tile edge
  * attention: per head, per 64-key tile, online softmax with running max/sum (tile order matters)
  * RMSNorm / RoPE / SiLU-gate / dynamic quant / embedding: per row
Prefill processes all prompt rows at once (row ops are independent, attention is causal), which is
bit-identical to feeding the tokens one at a time. Set dump=True to record every op's int tensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden import luts  # noqa: E402
from golden.data import EOS_IDS  # noqa: E402
from golden.fixedpoint import (I64, SADD_SHIFT, check_range, dyn_quant_rows, matmul_i8, requant, requant_wide,  # noqa: E402
                               rsr, sat, sat8, sat16)
from golden.quant import (D_MODEL, EMB_FRAC, HEAD_DIM, KV_DIM, LN_B_FRAC, LN_V_SHIFT, N_HEAD, N_KV_HEAD, N_VOCAB,  # noqa: E402
                          AttnQ, Linear, QModel, RmsNormQ, RopeQ, SiluQ)

KEY_TILE = 64
EXP_D_MAX = (1 << luts.EXP_D_BITS) - 1
ATT_RECIP_SHIFT = 40       # rl = floor((2^40 + l/2) / l)
ATT_OUT_SHIFT = 33         # out16 = rsr(O * rl, 33) == O/l * 2^7
GROUP = N_HEAD // N_KV_HEAD


class Dumper:
    def __init__(self, enabled: bool, layers=None):
        self.enabled = enabled
        self.layers = None if layers is None else set(layers)
        self.d: dict[str, np.ndarray] = {}

    def __call__(self, name: str, arr):
        if self.enabled:
            assert name not in self.d, name
            self.d[name] = np.array(arr, copy=True)

    def wants(self, layer: int) -> bool:
        return self.enabled and (self.layers is None or layer in self.layers)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.d)


# ---------------------------------------------------------------------------------------------- ops
def rmsnorm(nq: RmsNormQ, x) -> np.ndarray:
    """x [M,2048] int32 (residual, scale nq.s_in) -> y16 [M,2048] int16 with scale 2^-F.
    V = sum(x^2) + eps_q (up to 2^74: Python ints), m = V >> 2e in [2^28, 2^30), y1 ~ 2^30/sqrt(m),
    u = rsr(x*y1, 14+e) = x/sqrt(V) * 2^16, v = rsr(u*G, 12), y = sat16(rsr(v, 4))."""
    x = check_range(x, 32, "rmsnorm in")
    M = x.shape[0]
    out = np.empty((M, D_MODEL), dtype=I64)
    hi, lo = x >> I64(16), x & I64(0xFFFF)                   # x = hi*2^16 + lo ; every partial sum fits int64
    s_hh = (hi * hi).sum(axis=1); s_hl = (hi * lo).sum(axis=1); s_ll = (lo * lo).sum(axis=1)
    for i in range(M):
        V = (int(s_hh[i]) << 32) + (int(s_hl[i]) << 17) + int(s_ll[i]) + int(nq.eps_q)   # exact (> 64 bits possible)
        bl = V.bit_length()
        e = (bl - 29) // 2
        m = V >> (2 * e) if e >= 0 else V << (-2 * e)
        y1 = int(luts.rsqrt_newton(np.array([m], dtype=I64))[0])                 # ~2^30/sqrt(m), 17 bits
        sh = 14 + e
        assert sh >= 0, ("rmsnorm shift", sh)
        # x*y1 fits int64 (32 + 17 bits); u <= 2^16 in magnitude
        u = rsr(x[i] * I64(y1), sh)
        v = rsr(u * nq.g.astype(I64), LN_V_SHIFT)            # y * 2^4
        out[i] = sat16(rsr(v, LN_B_FRAC))
    return out.astype(np.int16)


def dynq(x):
    """Dynamic per-row quantisation of an int16 / int32 tensor -> (a8, rowfac packed as m16 | b<<16)."""
    return dyn_quant_rows(np.asarray(x, dtype=I64))


def linear(L: Linear, a8, rowfac=None):
    """a8 [M,K] int8 -> int8/int16 [M,N] (or int32 wide for out_bits == 0)."""
    acc = matmul_i8(np.asarray(a8, np.int8), L.w8)
    if L.out_bits == 0:
        return requant_wide(acc, L.mult, L.s1)
    rf = np.ones(acc.shape[0], dtype=I64) if rowfac is None else np.asarray(rowfac, dtype=I64)
    assert L.dynamic == (rowfac is not None), L.name
    return requant(acc, L.mult, L.bias, L.s1, rf, L.out_bits)


def rope(rq: RopeQ, x16, positions, table) -> np.ndarray:
    """x16 [M, heads*128] int16, positions [M] -> int8 [M, heads*128].
    Per head, per pair d < 64: r1 = x[d]*c - x[d+64]*s ; r2 = x[d+64]*c + x[d]*s (c/s Q15 from table[pos]);
    r16 = sat16(rsr(r, 15)) ; a8 = sat8(rsr(r16 * m_r, s_r))."""
    x = check_range(x16, 16, "rope in")
    M, W = x.shape
    H = W // HEAD_DIM
    assert H == rq.n_heads
    t = np.asarray(table, dtype=I64)[np.asarray(positions)]      # [M, 128] = cos[64] | sin[64]
    c, s = t[:, :HEAD_DIM // 2], t[:, HEAD_DIM // 2:]
    xh = x.reshape(M, H, HEAD_DIM)
    x1, x2 = xh[:, :, :HEAD_DIM // 2], xh[:, :, HEAD_DIM // 2:]
    r1 = x1 * c[:, None, :] - x2 * s[:, None, :]
    r2 = x2 * c[:, None, :] + x1 * s[:, None, :]
    r = np.concatenate([r1, r2], axis=2).reshape(M, W)
    r16 = sat16(rsr(r, 15))
    return sat8(rsr(r16 * I64(rq.m_r), rq.s_r)).astype(np.int8)


def silu_gate(sq: SiluQ, g16, u16) -> np.ndarray:
    """h32 = silu16(g) * up16 (exact, scale s_gate*s_up), silu16 = sat16(rsr(g16 * sigmoid_q(g), 15))."""
    g = check_range(g16, 16, "silu gate in")
    u = check_range(u16, 16, "silu up in")
    xf = np.clip(rsr(g * I64(sq.m_sig), sq.s_sig), -2048, 2047)
    sig = luts.sigmoid_lut(xf)
    s16 = sat16(rsr(g * sig, 15))
    return (s16 * u).astype(np.int32)                         # |h| < 2^30


def scaled_add(ma, a, mb, b, out_bits):
    a = np.asarray(a, dtype=I64)
    b = np.asarray(b, dtype=I64)
    return sat(rsr(a * I64(ma) + b * I64(mb), SADD_SHIFT), out_bits)


def embed_rows(e8_table, resmult, toks, out_bits=32):
    """x0[m] = sat_rb(rsr(e8[tok] * resmult[tok], EMB_FRAC))."""
    toks = np.asarray(toks)
    e = np.asarray(e8_table, dtype=I64)[toks]
    me = np.asarray(resmult, dtype=I64)[toks]
    return sat(rsr(e * me[:, None], EMB_FRAC), out_bits).astype(np.int32)


def attention(at: AttnQ, q8, k8, v8, n_keys: int, causal_offset: int | None = None, dump=None, tag=""):
    """q8 [Mq,2048] (16 heads), k8/v8 [n_keys_alloc,256] (2 kv heads) int8; head h uses kv head h//8.
    Row j valid iff j < n_keys and, if causal, j <= causal_offset+m. Returns out16 [Mq,2048] int16
    (scale s_v/128), heads processed independently, key tiles in ascending order."""
    q8 = np.asarray(q8, dtype=I64)
    k8 = np.asarray(k8, dtype=I64)
    v8 = np.asarray(v8, dtype=I64)
    Mq = q8.shape[0]
    n_tiles = (n_keys + KEY_TILE - 1) // KEY_TILE
    out = np.zeros((Mq, D_MODEL), dtype=I64)
    for h in range(N_HEAD):
        hs = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        kv = h // GROUP
        ks = slice(kv * HEAD_DIM, (kv + 1) * HEAD_DIM)
        qh = q8[:, hs]
        mq, sq = I64(int(at.mq[h])), int(at.sq[h])
        mx = np.zeros(Mq, dtype=I64)
        l = np.zeros(Mq, dtype=I64)
        O = np.zeros((Mq, HEAD_DIM), dtype=I64)
        started = np.zeros(Mq, dtype=bool)
        for t in range(n_tiles):
            j0 = t * KEY_TILE
            kt = k8[j0:j0 + KEY_TILE, ks]
            vt = v8[j0:j0 + KEY_TILE, ks]
            nt = kt.shape[0]
            s = (qh.astype(np.float64) @ kt.T.astype(np.float64)).astype(I64)   # exact, |s| <= 2^21
            jj = np.arange(j0, j0 + nt)
            valid = np.broadcast_to(jj < n_keys, (Mq, nt)).copy()
            if causal_offset is not None:
                valid &= jj[None, :] <= (causal_offset + np.arange(Mq))[:, None]
            any_valid = valid.any(axis=1)
            if not any_valid.any():
                continue
            s_masked = np.where(valid, s, np.iinfo(np.int64).min)
            mt = s_masked.max(axis=1)
            mnew = np.where(started, np.maximum(mx, mt), mt)
            grew = started & any_valid & (mnew > mx)
            if grew.any():
                da = np.clip(rsr((mnew - mx) * mq, sq), 0, EXP_D_MAX)
                alpha = luts.exp_lut(da)
                alpha = np.where(grew, alpha, 1 << 15)
                l = rsr(l * alpha, 15)
                O = rsr(O * alpha[:, None], 15)
            mx = np.where(any_valid, mnew, mx)
            d = np.clip(rsr((mx[:, None] - s) * mq, sq), 0, EXP_D_MAX)
            p = np.where(valid, luts.exp_lut(d), 0)
            l = l + p.sum(axis=1)
            O = O + (p.astype(np.float64) @ vt.astype(np.float64)).astype(I64)
            started |= any_valid
            if dump is not None:
                dump(f"{tag}.h{h}.t{t}.s", s)
                dump(f"{tag}.h{h}.t{t}.p", p)
        assert started.all(), "query with no valid key"
        assert l.max() < (1 << 28) and np.abs(O).max() < (1 << 35), "attention accumulators exceed the RTL widths"
        rl = ((I64(1) << I64(ATT_RECIP_SHIFT)) + (l >> I64(1))) // l
        out[:, hs] = sat16(rsr(O * rl[:, None], ATT_OUT_SHIFT))
        if dump is not None:
            dump(f"{tag}.h{h}.l", l)
            dump(f"{tag}.h{h}.O", O)
    return out.astype(np.int16)


# ---------------------------------------------------------------------------------------------- model
class IntMiniCPM:
    def __init__(self, qm: QModel, dump: bool = False, dump_layers=None, max_ctx: int | None = None,
                 n_layers: int | None = None, lm_cols: int | None = None):
        """n_layers / lm_cols run a prefix of the model and a prefix of the vocabulary: the configuration the
        layer-level Verilator test mirrors (docs/decisions.md #12). Constants stay those of the full model."""
        self.qm = qm
        self.cfg = qm.cfg
        self.dump = Dumper(dump, dump_layers)
        self.rb = qm.cfg.residual_bits
        self.n_layers = qm.cfg.n_layers if n_layers is None else n_layers
        self.lm_cols = lm_cols
        self.max_ctx = max_ctx or qm.tables["rope"].shape[0]
        self.reset_cache()

    def reset_cache(self):
        self.kcache = [np.zeros((self.max_ctx, KV_DIM), np.int8) for _ in range(self.n_layers)]
        self.vcache = [np.zeros((self.max_ctx, KV_DIM), np.int8) for _ in range(self.n_layers)]
        self.n_keys = 0

    # ---- helpers
    def _lin(self, name, a8, rowfac=None):
        return linear(self.qm.linears[name], a8, rowfac)

    def embed(self, toks):
        return embed_rows(self.qm.tables["embed"], self.qm.tables["embed.resmult"], toks, self.rb)

    def layer(self, l: int, x, positions, tag: str = ""):
        """One decoder layer over rows x [M,2048] int32 at `positions` (consecutive, = kv slots)."""
        qm, p = self.qm, f"L{l}"
        d = self.dump if self.dump.wants(l) else Dumper(False)
        tg = f"{tag}{p}"
        pos = np.asarray(positions)
        M = x.shape[0]
        assert np.array_equal(pos, np.arange(pos[0], pos[0] + M)), "rows must be consecutive positions"
        # -- attention
        y16 = rmsnorm(qm.norms[f"{p}.norm1"], x)
        a8, rf = dynq(y16)
        d(f"{tg}.norm1.out", y16); d(f"{tg}.attn.in", a8); d(f"{tg}.attn.rf", rf)
        q16 = self._lin(f"{p}.q", a8, rf)
        k16 = self._lin(f"{p}.k", a8, rf)
        v8 = self._lin(f"{p}.v", a8, rf)
        d(f"{tg}.q16", q16); d(f"{tg}.k16", k16); d(f"{tg}.v8", v8)
        q8 = rope(qm.ropes[f"{p}.rope_q"], q16, pos, qm.tables["rope"])
        k8 = rope(qm.ropes[f"{p}.rope_k"], k16, pos, qm.tables["rope"])
        d(f"{tg}.q8", q8); d(f"{tg}.k8", k8)
        self.kcache[l][pos[0]:pos[0] + M] = k8
        self.vcache[l][pos[0]:pos[0] + M] = v8
        n_keys = pos[0] + M
        att = attention(qm.attns[f"{p}.attn"], q8, self.kcache[l][:n_keys], self.vcache[l][:n_keys], n_keys,
                        causal_offset=int(pos[0]), dump=d if d.enabled else None, tag=f"{tg}.attn")
        d(f"{tg}.attn.out", att)
        a8, rf = dynq(att)
        d(f"{tg}.o.in", a8); d(f"{tg}.o.rf", rf)
        o16 = self._lin(f"{p}.o", a8, rf)                   # int32 (static scale)
        d(f"{tg}.o.out", o16)
        ad = qm.adds[f"{p}.add1"]
        x = scaled_add(ad.ma, x, ad.mb, o16, self.rb)
        d(f"{tg}.add1.out", x)
        # -- MLP
        y16 = rmsnorm(qm.norms[f"{p}.norm2"], x)
        a8, rf = dynq(y16)
        d(f"{tg}.norm2.out", y16); d(f"{tg}.mlp.in", a8); d(f"{tg}.mlp.rf", rf)
        g16 = self._lin(f"{p}.gate", a8, rf)
        u16 = self._lin(f"{p}.up", a8, rf)
        d(f"{tg}.gate.out", g16); d(f"{tg}.up.out", u16)
        h32 = silu_gate(qm.silus[f"{p}.silu"], g16, u16)
        d(f"{tg}.silu.out", h32)
        a8, rf = dynq(h32)
        d(f"{tg}.down.in", a8); d(f"{tg}.down.rf", rf)
        f16 = self._lin(f"{p}.down", a8, rf)                # int32 (static scale)
        d(f"{tg}.down.out", f16)
        ad = qm.adds[f"{p}.add2"]
        x = scaled_add(ad.ma, x, ad.mb, f16, self.rb)
        d(f"{tg}.add2.out", x)
        return x.astype(np.int32)

    def lm_head(self, x_rows, tag=""):
        """x_rows [M,2048] int32 -> int32 logits [M, N_VOCAB] (wide mode; argmax semantics only)."""
        d = self.dump
        y16 = rmsnorm(self.qm.norms["norm_f"], x_rows)
        a8, rf = dynq(y16)
        d(f"{tag}norm_f.out", y16); d(f"{tag}lm.in", a8); d(f"{tag}lm.rf", rf)
        t = self._lin("lm", a8)
        if self.lm_cols is not None:
            t = t[:, :self.lm_cols]
        d(f"{tag}logits", t)
        return t

    def forward(self, ids, pos0: int = 0, logits_rows: str = "last", tag: str = ""):
        """Run rows for tokens `ids` at positions pos0.. (appending to the KV cache).
        logits_rows: 'last' -> int32 [N_VOCAB] of the last row; 'all' -> [M, N_VOCAB]; None -> no LM head."""
        ids = list(ids)
        M = len(ids)
        assert pos0 == self.n_keys and pos0 + M <= self.max_ctx, (pos0, self.n_keys, M)
        positions = np.arange(pos0, pos0 + M)
        x = self.embed(ids)
        self.dump(f"{tag}x0", x)
        for l in range(self.n_layers):
            x = self.layer(l, x, positions, tag)
        self.n_keys = pos0 + M
        if logits_rows is None:
            return None
        rows = x if logits_rows == "all" else x[-1:]
        t = self.lm_head(rows, tag)
        return t if logits_rows == "all" else t[0]

    @staticmethod
    def argmax(logits) -> int:
        """Lowest index among maxima (the hardware sampler's rule)."""
        return int(np.argmax(np.asarray(logits)))

    def generate(self, prompt_ids, max_new: int, eos=EOS_IDS, verbose=False):
        """Greedy decoding. Returns the generated ids (the eos token, if reached, is not included)."""
        self.reset_cache()
        out = []
        t = self.forward(prompt_ids, 0, "last", tag="p.")
        nxt = self.argmax(t)
        while nxt not in eos and len(out) < max_new and self.n_keys < self.max_ctx:
            out.append(nxt)
            if verbose:
                print(nxt, end=" ", flush=True)
            t = self.forward([nxt], self.n_keys, "last", tag=f"d{len(out)}.")
            nxt = self.argmax(t)
        return out
