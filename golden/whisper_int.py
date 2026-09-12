"""The integer golden model: exact fixed-point Whisper-tiny datapath. THIS IS THE SPEC FOR THE RTL.

Every op is a pure function on integer numpy arrays, tiled/ordered the way the hardware does it:
  * matmuls accumulate exactly (order-independent), requant at the tile edge
  * attention: per head, per 64-key tile, online softmax with running max/sum (tile order matters)
  * LayerNorm / GELU / dynamic quant: per row
Set DUMP=1 (or IntWhisper(dump=...)) to record every op's int inputs/outputs for RTL tests.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden import luts  # noqa: E402
from golden.fixedpoint import (I64, SADD_SHIFT, check_range, dyn_quant_rows, matmul_i8, requant, requant_wide,  # noqa: E402
                               rsr, sat, sat8, sat16)
from golden.quant import (D_MODEL, HEAD_DIM, LN_B_FRAC, LN_V_SHIFT, LN_XC_FRAC, N_HEAD, N_MEL, N_MEL_PAD,  # noqa: E402
                          N_VOCAB, AttnQ, GeluQ, LayerNormQ, Linear, QModel)

KEY_TILE = 64
LN_C3 = 5592405            # round(2^24 / 3): mean_f = rsr(sum * C3, 23) == sum * 2^8 / 384
EXP_D_MAX = (1 << luts.EXP_D_BITS) - 1
ATT_RECIP_SHIFT = 40       # rl = floor((2^40 + l/2) / l)
ATT_OUT_SHIFT = 33         # out16 = rsr(O * rl, 33) == O/l * 2^7


class Dumper:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.d: dict[str, np.ndarray] = {}

    def __call__(self, name: str, arr):
        if self.enabled:
            assert name not in self.d, name
            self.d[name] = np.array(arr, copy=True)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.d)


# ---------------------------------------------------------------------------------------------- ops
def layernorm(ln: LayerNormQ, x) -> np.ndarray:
    """x [M,384] int (residual, scale ln.s_in) -> y16 [M,384] int16 with scale 2^-F."""
    x = np.asarray(x, dtype=I64)
    M = x.shape[0]
    s = x.sum(axis=1)                                        # int32
    mean_f = rsr(s * LN_C3, 23)                              # Q.8
    xc = (x << I64(LN_XC_FRAC)) - mean_f[:, None]            # Q.8, |xc| < 2^24
    V = (xc * xc).sum(axis=1) + I64(ln.eps_q)                # int64, > 0
    bl = np.array([int(v).bit_length() for v in V], dtype=I64)
    e = (bl - 29) // 2                                       # m = V >> 2e in [2^28, 2^30)
    m = np.array([int(v) >> (2 * int(ee)) if ee >= 0 else int(v) << (-2 * int(ee)) for v, ee in zip(V, e)], dtype=I64)
    y1 = luts.rsqrt_newton(m)                                # ~2^30/sqrt(m), 17 bits
    out = np.empty((M, D_MODEL), dtype=I64)
    for i in range(M):                                       # per-row shift amount
        sh = 14 + int(e[i])
        assert sh >= 0, ("LN shift", sh)
        u = rsr(xc[i] * y1[i], sh)                           # n/sqrt(384) * 2^16, |u| <= 2^16
        v = rsr(u * ln.g.astype(I64), LN_V_SHIFT)            # y * 2^4
        out[i] = sat16(rsr(v + ln.b.astype(I64), LN_B_FRAC))
    return out.astype(np.int16)


def dynq(x16, colmult=None):
    """Dynamic per-row quantisation, optional per-channel Q15 pre-multiplier (SmoothQuant on GELU16 out)."""
    x = np.asarray(x16, dtype=I64)
    if colmult is not None:
        x = sat16(rsr(x * np.asarray(colmult, dtype=I64)[None, :], 15))
    return dyn_quant_rows(x)


def linear(L: Linear, a8, rowfac=None):
    """a8 [M,K] int8 -> int8/int16 [M,N] (or int32 wide for out_bits == 0)."""
    acc = matmul_i8(a8, L.w8)
    if L.out_bits == 0:
        return requant_wide(acc, L.mult, L.s1)
    rf = np.ones(acc.shape[0], dtype=I64) if rowfac is None else np.asarray(rowfac, dtype=I64)
    assert L.dynamic == (rowfac is not None), L.name
    return requant(acc, L.mult, L.bias, L.s1, rf, L.out_bits)


_PHI = None


def phi_lut(xf):
    """xf: x in Q.8 clamped to [-2048, 2047]. Returns Phi(x) in Q15 (0..32768), linear interpolation."""
    global _PHI
    if _PHI is None:
        from golden.quant import phi_table
        _PHI = np.array(phi_table() + [32768], dtype=I64)
    xf = np.clip(np.asarray(xf, dtype=I64), -2048, 2047)
    i = (xf >> I64(4)) + I64(128)
    f = xf & I64(15)
    lo, hi = _PHI[i], _PHI[i + 1]
    return lo + rsr((hi - lo) * f, 4)


def gelu16(g: GeluQ, h16) -> np.ndarray:
    """int16 x*Phi(x): xf = rsr(h16*m_phi, s_phi) (Q.8), g16 = rsr(h16 * Phi_q, 15)."""
    h = check_range(h16, 16, "gelu16 in")
    xf = rsr(h * I64(g.m_phi), g.s_phi)
    phi = phi_lut(xf)
    return sat16(rsr(h * phi, 15)).astype(np.int16)


def gelu8(g: GeluQ, h8) -> np.ndarray:
    return luts.gelu_lut(h8, g.table)


def requant_static8(g: GeluQ, g16) -> np.ndarray:
    return sat8(rsr(np.asarray(g16, dtype=I64) * I64(g.req_mult), g.req_shift)).astype(np.int8)


def scaled_add(ma, a, mb, b, out_bits):
    a = np.asarray(a, dtype=I64)
    b = np.asarray(b, dtype=I64)
    return sat(rsr(a * I64(ma) + b * I64(mb), SADD_SHIFT), out_bits)


def attention(at: AttnQ, q8, k8, v8, n_keys: int, causal_offset: int | None = None, dump=None, tag=""):
    """q8 [Mq,384], k8/v8 [n_keys_alloc,384] int8 (row j valid iff j < n_keys and, if causal, j <= causal_offset+m).
    Returns out16 [Mq,384] int16 (scale s_v/128), heads processed independently, key tiles in order."""
    q8 = np.asarray(q8, dtype=I64)
    k8 = np.asarray(k8, dtype=I64)
    v8 = np.asarray(v8, dtype=I64)
    Mq = q8.shape[0]
    n_tiles = (n_keys + KEY_TILE - 1) // KEY_TILE
    out = np.zeros((Mq, D_MODEL), dtype=I64)
    for h in range(N_HEAD):
        hs = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        qh = q8[:, hs]
        mq, sq = I64(int(at.mq[h])), int(at.sq[h])
        mx = np.zeros(Mq, dtype=I64)
        l = np.zeros(Mq, dtype=I64)
        O = np.zeros((Mq, HEAD_DIM), dtype=I64)
        started = np.zeros(Mq, dtype=bool)
        for t in range(n_tiles):
            j0 = t * KEY_TILE
            kt = k8[j0:j0 + KEY_TILE, hs]
            vt = v8[j0:j0 + KEY_TILE, hs]
            nt = kt.shape[0]
            s = (qh.astype(np.float64) @ kt.T.astype(np.float64)).astype(I64)   # exact, |s| <= 2^20
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
            # rescale rows whose max increased (started rows only)
            grew = started & any_valid & (mnew > mx)
            if grew.any():
                da = np.clip(rsr((mnew - mx) * mq, sq), 0, EXP_D_MAX)
                alpha = luts.exp_lut(da)
                alpha = np.where(grew, alpha, 1 << 15)
                l = rsr(l * alpha, 15)
                O = rsr(O * alpha[:, None], 15)
            upd = any_valid
            mx = np.where(upd, mnew, mx)
            d = np.clip(rsr((mx[:, None] - s) * mq, sq), 0, EXP_D_MAX)
            p = np.where(valid, luts.exp_lut(d), 0)
            l = l + p.sum(axis=1)
            O = O + (p.astype(np.float64) @ vt.astype(np.float64)).astype(I64)
            started |= any_valid
            if dump is not None:
                dump(f"{tag}.h{h}.t{t}.s", s)
                dump(f"{tag}.h{h}.t{t}.p", p)
        assert started.all(), "query with no valid key"
        rl = ((I64(1) << I64(ATT_RECIP_SHIFT)) + (l >> I64(1))) // l
        out[:, hs] = sat16(rsr(O * rl[:, None], ATT_OUT_SHIFT))
        if dump is not None:
            dump(f"{tag}.h{h}.l", l)
            dump(f"{tag}.h{h}.O", O)
    return out.astype(np.int16)


# ---------------------------------------------------------------------------------------------- model
class IntWhisper:
    def __init__(self, qm: QModel, dump: bool | None = None):
        self.qm = qm
        self.cfg = qm.cfg
        self.dump = Dumper(bool(int(os.environ.get("DUMP", "0"))) if dump is None else dump)
        self.rb = qm.cfg.residual_bits
        self.sat_count = 0

    # ---- helpers
    def _lin(self, name, a8, rowfac=None):
        return linear(self.qm.linears[name], a8, rowfac)

    def _gelu(self, name, h):
        g = self.qm.gelus[name]
        return gelu8(g, h) if g.mode == "lut8" else gelu16(g, h)

    def _add(self, name, a, b):
        ad = self.qm.adds[name]
        return scaled_add(ad.ma, a, ad.mb, b, ad.out_bits)

    def _ln(self, name, x):
        return layernorm(self.qm.lns[name], x)

    def _ffn(self, p, x, add_key):
        d = self.dump
        y16 = self._ln(f"{p}.ln2", x)
        a8, rf = dynq(y16)
        d(f"{p}.ln2.out", y16); d(f"{p}.fc1.in", a8); d(f"{p}.fc1.rf", rf)
        h = self._lin(f"{p}.fc1", a8, rf)
        d(f"{p}.fc1.out", h)
        g = self._gelu(f"{p}.gelu", h)
        d(f"{p}.gelu.out", g)
        gq = self.qm.gelus[f"{p}.gelu"]
        if gq.mode == "lut8":
            f16 = self._lin(f"{p}.fc2", g)
        else:
            a8, rf = dynq(g, gq.smooth)
            d(f"{p}.fc2.in", a8); d(f"{p}.fc2.rf", rf)
            f16 = self._lin(f"{p}.fc2", a8, rf)
        d(f"{p}.fc2.out", f16)
        x = self._add(f"{p}.{add_key}", x, f16)
        d(f"{p}.{add_key}.out", x)
        return x

    def _self_attn(self, p, x, kcache=None, vcache=None, pos=None):
        """Encoder (kcache None): full self-attention over all rows. Decoder: single row at position pos."""
        d = self.dump
        y16 = self._ln(f"{p}.ln1", x)
        a8, rf = dynq(y16)
        d(f"{p}.ln1.out", y16); d(f"{p}.attn.in", a8); d(f"{p}.attn.rf", rf)
        q8 = self._lin(f"{p}.attn.q", a8, rf)
        k8 = self._lin(f"{p}.attn.k", a8, rf)
        v8 = self._lin(f"{p}.attn.v", a8, rf)
        d(f"{p}.attn.q", q8); d(f"{p}.attn.k", k8); d(f"{p}.attn.v", v8)
        if kcache is None:
            att = attention(self.qm.attns[f"{p}.attn"], q8, k8, v8, q8.shape[0], dump=d if d.enabled else None, tag=f"{p}.attn")
        else:
            kcache[pos] = k8[0]
            vcache[pos] = v8[0]
            att = attention(self.qm.attns[f"{p}.attn"], q8, kcache[: pos + 1], vcache[: pos + 1], pos + 1,
                            causal_offset=pos, dump=d if d.enabled else None, tag=f"{p}.attn.pos{pos}")
        d(f"{p}.attn.out", att)
        a8, rf = dynq(att)
        d(f"{p}.attn.o.in", a8); d(f"{p}.attn.o.rf", rf)
        o16 = self._lin(f"{p}.attn.o", a8, rf)
        d(f"{p}.attn.o.out", o16)
        x = self._add(f"{p}.add1", x, o16)
        d(f"{p}.add1.out", x)
        return x

    # ---- encoder
    def mel_quant(self, mel: np.ndarray, n_frames: int) -> np.ndarray:
        """mel float [80, T>=n_frames] (whisper log-mel of the 30 s window) -> int8 [n_frames, 96]."""
        s = self.qm.scalars["s_mel"]
        m8 = np.zeros((n_frames, N_MEL_PAD), dtype=np.int8)
        q = np.clip(np.round(mel[:, :n_frames].T / s), -127, 127)
        m8[:, :N_MEL] = q.astype(np.int8)
        return m8

    @staticmethod
    def im2col(x, stride: int) -> np.ndarray:
        """x [T, C] -> rows t: concat(x[stride*t-1], x[stride*t], x[stride*t+1]) with zero padding."""
        T, C = x.shape
        n_out = T // stride
        xp = np.zeros((T + 2, C), dtype=x.dtype)
        xp[1:T + 1] = x
        idx = stride * np.arange(n_out)[:, None] + np.arange(3)[None, :]       # frame index into xp
        return xp[idx].reshape(n_out, 3 * C)

    def encoder(self, mel8: np.ndarray):
        """mel8 [n_frames, 96] int8 -> (enc8 [n_ctx,384] int8, rowfac [n_ctx]) after ln_post + dynq."""
        d = self.dump
        n_frames = mel8.shape[0]
        assert n_frames % 2 == 0
        d("enc.mel8", mel8)
        c1 = self._lin("enc.conv1", self.im2col(mel8, 1))
        d("enc.conv1.out", c1)
        g1 = self._gelu("enc.conv1", c1)
        if self.qm.gelus["enc.conv1"].mode == "phi16":
            g1 = requant_static8(self.qm.gelus["enc.conv1"], g1)
        d("enc.conv1.gelu", g1)
        c2 = self._lin("enc.conv2", self.im2col(g1, 2))
        d("enc.conv2.out", c2)
        g2 = self._gelu("enc.conv2", c2)
        d("enc.conv2.gelu", g2)
        n_ctx = c2.shape[0]
        pos = self.qm.tables["enc.pos"][:n_ctx]
        x = self._add("enc.x0", g2, pos)
        d("enc.x0", x)
        for l in range(4):
            p = f"enc.{l}"
            x = self._self_attn(p, x)
            x = self._ffn(p, x, "add2")
        y16 = self._ln("enc.ln_post", x)
        enc8, rf = dynq(y16)
        d("enc.ln_post.out", y16); d("enc.out", enc8); d("enc.out.rf", rf)
        return enc8, rf

    # ---- decoder
    def cross_kv(self, enc8, rf):
        d = self.dump
        ks, vs = [], []
        for l in range(4):
            p = f"dec.{l}"
            k = self._lin(f"{p}.xattn.k", enc8, rf)
            v = self._lin(f"{p}.xattn.v", enc8, rf)
            d(f"{p}.xattn.k", k); d(f"{p}.xattn.v", v)
            ks.append(k)
            vs.append(v)
        return ks, vs

    def embed(self, tok: int, pos: int):
        L = self.qm.linears["dec.lm"]
        e8 = L.w8[:, tok].astype(I64)                  # embedding row (K = 384) of column tok
        me = I64(int(self.qm.tables["dec.emb.resmult"][tok]))
        ad = self.qm.adds["dec.x0"]
        p8 = self.qm.tables["dec.pos"][pos].astype(I64)
        return sat(rsr(e8 * me + p8 * I64(ad.mb), SADD_SHIFT), self.rb)[None, :]

    def decoder_step(self, tok: int, pos: int, state: dict, sample: bool, first_sample: bool, lang_meta: dict):
        d = self.dump
        tg = f"dec.pos{pos}"
        x = self.embed(tok, pos)
        d(f"{tg}.x0", x)
        for l in range(4):
            p = f"dec.{l}"
            pt = f"{tg}.{l}"
            # dumps inside _self_attn use the layer prefix; make them position-unique
            self.dump_prefix = pt
            x = self._self_attn_dec(p, pt, x, state["k"][l], state["v"][l], pos)
            y16 = self._ln(f"{p}.lnc", x)
            a8, rf = dynq(y16)
            d(f"{pt}.lnc.out", y16); d(f"{pt}.xattn.in", a8); d(f"{pt}.xattn.rf", rf)
            q8 = self._lin(f"{p}.xattn.q", a8, rf)
            d(f"{pt}.xattn.q", q8)
            att = attention(self.qm.attns[f"{p}.xattn"], q8, state["ck"][l], state["cv"][l], state["n_ctx"],
                            dump=d if d.enabled else None, tag=f"{pt}.xattn")
            d(f"{pt}.xattn.out", att)
            a8, rf = dynq(att)
            co = self._lin(f"{p}.xattn.o", a8, rf)
            d(f"{pt}.xattn.o.out", co)
            x = self._add(f"{p}.add2", x, co)
            d(f"{pt}.add2.out", x)
            x = self._ffn_tagged(p, pt, x, "add3")
        if not sample:
            return None, None
        y16 = self._ln("dec.ln", x)
        a8, rf = dynq(y16)
        d(f"{tg}.ln.out", y16); d(f"{tg}.lm.in", a8)
        t = self._lin("dec.lm", a8)[0]                    # int32 [51872]
        t = t.astype(I64)
        t[N_VOCAB:] = np.iinfo(np.int32).min
        t[lang_meta["suppress_tokens"]] = np.iinfo(np.int32).min
        if first_sample:
            t[lang_meta["suppress_blank"]] = np.iinfo(np.int32).min
        d(f"{tg}.logits", t)
        nxt = int(np.argmax(t))                            # first max index
        return nxt, t

    def _self_attn_dec(self, p, pt, x, kcache, vcache, pos):
        d = self.dump
        y16 = self._ln(f"{p}.ln1", x)
        a8, rf = dynq(y16)
        d(f"{pt}.ln1.out", y16); d(f"{pt}.attn.in", a8); d(f"{pt}.attn.rf", rf)
        q8 = self._lin(f"{p}.attn.q", a8, rf)
        k8 = self._lin(f"{p}.attn.k", a8, rf)
        v8 = self._lin(f"{p}.attn.v", a8, rf)
        d(f"{pt}.attn.q", q8); d(f"{pt}.attn.k", k8); d(f"{pt}.attn.v", v8)
        kcache[pos] = k8[0]
        vcache[pos] = v8[0]
        att = attention(self.qm.attns[f"{p}.attn"], q8, kcache[: pos + 1], vcache[: pos + 1], pos + 1,
                        causal_offset=pos, dump=d if d.enabled else None, tag=f"{pt}.attn")
        d(f"{pt}.attn.out", att)
        a8, rf = dynq(att)
        o16 = self._lin(f"{p}.attn.o", a8, rf)
        d(f"{pt}.attn.o.out", o16)
        x = self._add(f"{p}.add1", x, o16)
        d(f"{pt}.add1.out", x)
        return x

    def _ffn_tagged(self, p, pt, x, add_key):
        d = self.dump
        y16 = self._ln(f"{p}.ln2", x)
        a8, rf = dynq(y16)
        d(f"{pt}.ln2.out", y16); d(f"{pt}.fc1.in", a8); d(f"{pt}.fc1.rf", rf)
        h = self._lin(f"{p}.fc1", a8, rf)
        d(f"{pt}.fc1.out", h)
        g = self._gelu(f"{p}.gelu", h)
        d(f"{pt}.gelu.out", g)
        gq = self.qm.gelus[f"{p}.gelu"]
        if gq.mode == "lut8":
            f16 = self._lin(f"{p}.fc2", g)
        else:
            a8, rf = dynq(g, gq.smooth)
            d(f"{pt}.fc2.in", a8); d(f"{pt}.fc2.rf", rf)
            f16 = self._lin(f"{p}.fc2", a8, rf)
        d(f"{pt}.fc2.out", f16)
        x = self._add(f"{p}.{add_key}", x, f16)
        d(f"{pt}.{add_key}.out", x)
        return x

    def transcribe(self, mel: np.ndarray, n_frames: int, language: str = "en", max_new: int | None = None):
        """mel: whisper log-mel [80, 3000] float. Returns generated token ids (without prompt/eot)."""
        lm = self.qm.meta["decoding"][language]
        mel8 = self.mel_quant(mel, n_frames)
        enc8, rf = self.encoder(mel8)
        ck, cv = self.cross_kv(enc8, rf)
        n_ctx = enc8.shape[0]
        state = dict(n_ctx=n_ctx, ck=ck, cv=cv,
                     k=[np.zeros((448, D_MODEL), np.int8) for _ in range(4)],
                     v=[np.zeros((448, D_MODEL), np.int8) for _ in range(4)])
        prompt = list(lm["initial_tokens"])
        eot = lm["eot"]
        limit = lm["sample_len"] if max_new is None else max_new
        out = []
        pos = 0
        tok = prompt[0]
        while True:
            last_prompt = pos == len(prompt) - 1
            sample = pos >= len(prompt) - 1
            nxt, _ = self.decoder_step(tok, pos, state, sample, first_sample=last_prompt, lang_meta=lm)
            pos += 1
            if not sample:
                tok = prompt[pos]
                continue
            if nxt == eot or len(out) >= limit:
                break
            out.append(nxt)
            tok = nxt
            if len(out) >= limit or pos >= 448:
                break
        return out


def n_frames_for(n_samples: int, mode: str = "var", pad_frames: int = 0, min_frames: int = 0) -> int:
    """Host-side rule: frames = max(ceil(samples/160) + pad_frames, min_frames), rounded up to a multiple
    of 128 (so n_ctx is a multiple of 64), capped at 3000. mode "full" = always 3000 (whisper's window)."""
    if mode == "full":
        return 3000
    f = max((n_samples + 159) // 160 + pad_frames, min_frames)
    f = ((f + 127) // 128) * 128
    return min(f, 3000)
