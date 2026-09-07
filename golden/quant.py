"""Quantisation: builds the complete integer model (weights, scales, requant constants, LUTs) from the
fp32 checkpoint + calibration statistics. Every constant the RTL uses originates here.

Numerics summary (see docs/numerics.md for the full spec):
  * matmul/conv weights: int8 symmetric, per-output-channel scale s_w[n]
  * activations feeding matmuls: int8; per-token dynamic (maxabs row factor) when produced by LN /
    attention / GELU16, static when produced by conv1-GELU (im2col rows mix frames)
  * requant: t = sat32(rsr(acc*M[n], s1)); y = sat_w(rsr(t*rowfac[m] + B[n], s2)), s2 = 24 (int8) / 16 (int16)
  * residual stream: int16 (or int8, cfg.residual_bits) static per-point scale
  * LN: integer mean/var, rsqrt LUT+Newton, fused affine, int16 out (scale 2^-F) then dynamic quant
  * softmax: online, exp LUT (Q1.15), per-head static score multiplier
  * GELU: cfg.gelu_mode = "lut8" (int8->int8 table, prompt spec) or "phi16" (int16 x*Phi(x), Phi LUT)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden import luts  # noqa: E402
from golden.data import REPO, WHISPER_ROOT  # noqa: E402
from golden.fixedpoint import REQ_B_FRAC, REQ_S2_BY_BITS, SADD_SHIFT, mult_for  # noqa: E402

D_MODEL, N_HEAD, HEAD_DIM, D_FF = 384, 6, 64, 1536
N_VOCAB, N_VOCAB_PAD = 51865, 51872           # padded to a multiple of 32 output channels
N_MEL, N_MEL_PAD = 80, 96                      # mel bins padded to 3 tiles of 32
N_AUDIO_CTX, N_TEXT_CTX = 1500, 448
TILE = 32
LN_EPS = 1e-5
LN_XC_FRAC = 8           # xc_f = x*2^8 - mean_f
LN_U_FRAC = 16           # u = n_k/sqrt(384) * 2^16
LN_V_SHIFT = 12          # v = rsr(u*G, 12)  (= y * 2^4 in output units)
LN_B_FRAC = 4            # y16 = sat16(rsr(v + B, 4)),  B = beta * 2^(F+4)


@dataclass
class QConfig:
    residual_bits: int = 16
    gelu_mode: str = "phi16"          # "lut8" | "phi16"
    smooth_alpha: float | None = None  # SmoothQuant alpha for LN->matmul inputs (None = off)
    smooth_fc2: bool = False           # per-channel smoothing of GELU16 output into fc2 (phi16 only)
    margin_int8: float = 1.0           # static int8 scales = max*margin/127
    margin_int16: float = 2.0          # static int16 scales = max*margin/32767
    n_calib: int = 128

    def tag(self) -> str:
        return f"r{self.residual_bits}_{self.gelu_mode}_sa{self.smooth_alpha}_sf{int(self.smooth_fc2)}"


@dataclass
class Linear:
    """W[K,N] int8 (+ requant). rowfac semantics: dynamic -> S_in_eff = S_in/127, static -> S_in."""
    name: str
    w8: np.ndarray            # [K, N] int8
    s_w: np.ndarray           # [N] float
    mult: np.ndarray          # [N] int32
    bias: np.ndarray          # [N] int32
    s1: int
    out_bits: int             # 8 or 16 (0 = wide/raw mode for LM head)
    dynamic: bool
    s_in: float               # scale of the producer's int16/int8 tensor (S_in)
    s_out: np.ndarray         # [N] float (per channel, may be constant)


@dataclass
class LayerNormQ:
    name: str
    g: np.ndarray     # [384] int32
    b: np.ndarray     # [384] int32
    eps_q: int
    F: int            # output scale 2^-F
    s_in: float       # residual scale of the input


@dataclass
class AttnQ:
    name: str
    mq: np.ndarray    # [H] int32
    sq: np.ndarray    # [H] int
    s_q: np.ndarray   # [H] float (Q output scale per head)
    s_k: np.ndarray   # [H]
    s_v: float


@dataclass
class AddQ:
    name: str
    ma: int
    mb: int
    s_a: float
    s_b: float
    s_out: float
    out_bits: int


@dataclass
class GeluQ:
    name: str
    mode: str
    s_in: float                        # input scale (int8 for lut8, int16 for phi16)
    s_out: float                       # output scale (lut8: int8 out; phi16: same as s_in)
    table: list[int] | None = None     # lut8 table (256)
    m_phi: int = 0                     # phi16: xf = rsr(h16*m_phi, s_phi)
    s_phi: int = 0
    smooth: np.ndarray | None = None   # phi16: per-channel int32 multipliers (Q15) applied before dynq
    req_mult: int = 0                  # optional static int8 requant of the GELU16 output (conv1)
    req_shift: int = 0


@dataclass
class QModel:
    cfg: QConfig
    linears: dict[str, Linear] = field(default_factory=dict)
    lns: dict[str, LayerNormQ] = field(default_factory=dict)
    attns: dict[str, AttnQ] = field(default_factory=dict)
    adds: dict[str, AddQ] = field(default_factory=dict)
    gelus: dict[str, GeluQ] = field(default_factory=dict)
    tables: dict[str, np.ndarray] = field(default_factory=dict)   # pos-embs (int8), emb resmult, etc
    scalars: dict[str, float] = field(default_factory=dict)       # s_mel, s_res.*, ...
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------
def quant_per_channel(W: np.ndarray, axis_out: int = 0):
    """Symmetric per-output-channel int8. W float [N, K] -> (w8 [N,K], s [N])."""
    W = np.asarray(W, dtype=np.float64)
    amax = np.abs(W).max(axis=1)
    s = np.where(amax > 0, amax / 127.0, 1.0)
    w8 = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
    return w8, s


def make_linear(name: str, W: np.ndarray, b: np.ndarray | None, s_in: float, dynamic: bool,
                s_out, out_bits: int) -> Linear:
    """W float [N, K] (torch layout), b [N] or None. s_out: scalar or [N]."""
    N, K = W.shape
    assert K % TILE == 0, (name, K)
    w8, s_w = quant_per_channel(W)
    s_out = np.broadcast_to(np.asarray(s_out, dtype=np.float64), (N,)).copy()
    s_in_eff = s_in / 127.0 if dynamic else s_in
    if out_bits == 0:  # wide mode (argmax): M[n] = s_w[n]/max(s_w) * 2^30, s1 fixed by caller
        c = s_w / s_w.max()
        mult = np.round(c * (1 << 30)).astype(np.int64)
        s1 = 22
        bias = np.zeros(N, np.int64)
    else:
        s2 = REQ_S2_BY_BITS[out_bits]
        c = s_in_eff * s_w / s_out
        s1 = 30 - s2 - int(math.floor(math.log2(c.max())))
        assert s1 >= 0, (name, c.max(), s1)
        mult = np.round(c * float(2 ** (s1 + s2))).astype(np.int64)
        assert mult.max() < (1 << 31) and mult.min() >= 0
        bias = np.zeros(N, np.int64) if b is None else np.round(np.asarray(b, np.float64) / s_out * (1 << REQ_B_FRAC))
        bias = np.clip(bias, -(1 << 31), (1 << 31) - 1).astype(np.int64)
    return Linear(name, np.ascontiguousarray(w8.T), s_w, mult.astype(np.int32), bias.astype(np.int32),
                  s1, out_bits, dynamic, s_in, s_out)


def make_ln(name: str, gamma, beta, s_in: float, max_out: float) -> LayerNormQ:
    """LN on a residual of scale s_in; output int16 with scale 2^-F where F fits max_out."""
    F = int(math.floor(math.log2(32767.0 / max(max_out, 1e-3))))
    F = max(0, min(F, 14))
    s_ln = 2.0 ** (-F)
    # u = n_k/sqrt(384)*2^16 ; u*G = n*gamma*2^(F+16) ; v = rsr(u*G,12) = n*gamma*2^(F+4) ; y = rsr(v+B,4)
    g = np.round(math.sqrt(D_MODEL) * np.asarray(gamma, np.float64) / s_ln)
    b = np.round(np.asarray(beta, np.float64) / s_ln * (1 << LN_B_FRAC))
    for v in (g, b):
        assert np.abs(v).max() < (1 << 31), name
    eps_q = int(round(LN_EPS * D_MODEL * ((1 << LN_XC_FRAC) / s_in) ** 2))
    return LayerNormQ(name, g.astype(np.int32), b.astype(np.int32), eps_q, F, s_in)


def make_attn(name: str, s_q: np.ndarray, s_k: np.ndarray, s_v: float) -> AttnQ:
    c = s_q * s_k / math.sqrt(HEAD_DIM) * (1 << luts.EXP_FRAC_BITS)   # d/256 = (score_int * c/256)
    sq = np.array([30 - int(math.floor(math.log2(x))) for x in c], dtype=np.int64)
    mq = np.array([int(round(x * 2.0 ** s)) for x, s in zip(c, sq)], dtype=np.int64)
    assert mq.max() < (1 << 31) and sq.min() >= 0
    return AttnQ(name, mq.astype(np.int32), sq, s_q, s_k, s_v)


def make_add(name: str, s_a: float, s_b: float, s_out: float, out_bits: int) -> AddQ:
    return AddQ(name, mult_for(s_a / s_out, SADD_SHIFT), mult_for(s_b / s_out, SADD_SHIFT), s_a, s_b, s_out, out_bits)


def phi_table() -> list[int]:
    """Phi(x) = 0.5*(1+erf(x/sqrt2)) in Q15 at x = (i-128)/16, i=0..255; T[256] := 32768."""
    return [int(round(0.5 * (1.0 + math.erf(((i - 128) / 16.0) / math.sqrt(2.0))) * 32768)) for i in range(256)]


def make_gelu(name: str, mode: str, s_in: float, s_out: float | None = None, smooth=None,
              req_to_int8: float | None = None) -> GeluQ:
    if mode == "lut8":
        return GeluQ(name, mode, s_in, s_out, table=luts.gelu_table(s_in, s_out))
    # phi16: xf = rsr(h16 * m_phi, s_phi) is x in Q.8 (x = h16*s_in); m_phi = s_in*2^8*2^s_phi
    c = s_in * (1 << 8)
    s_phi = 30 - int(math.floor(math.log2(c)))
    m_phi = int(round(c * 2.0 ** s_phi))
    g = GeluQ(name, mode, s_in, s_in, m_phi=m_phi, s_phi=s_phi)
    if smooth is not None:
        g.smooth = np.round((1 << 15) / np.asarray(smooth, np.float64)).astype(np.int32)
    if req_to_int8 is not None:  # g8 = sat8(rsr(g16 * req_mult, req_shift)), g8 scale = req_to_int8
        c = s_in / req_to_int8
        sh = 30 - int(math.floor(math.log2(c)))
        g.req_mult, g.req_shift, g.s_out = int(round(c * 2.0 ** sh)), sh, req_to_int8
    return g


# ------------------------------------------------------------------------------------------------
def smooth_factors(x_chan_max: np.ndarray, W_list: list[np.ndarray], alpha: float) -> np.ndarray:
    """SmoothQuant: s_k = max|X_k|^a / max_n|W[n,k]|^(1-a); W torch layout [N,K]."""
    wmax = np.max([np.abs(W).max(axis=0) for W in W_list], axis=0)
    x = np.maximum(np.asarray(x_chan_max, np.float64), 1e-5)
    wmax = np.maximum(wmax, 1e-5)
    s = x ** alpha / wmax ** (1 - alpha)
    return np.clip(s, 1e-2, 1e2)


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> np.ndarray:
    """Whisper's sinusoidal positional embedding (encoder), fp64 replica of whisper.model.sinusoids."""
    log_ts = math.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-log_ts * np.arange(channels // 2))
    scaled = np.arange(length)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)


def quant_tensor_int8(x: np.ndarray, margin: float = 1.0):
    s = np.abs(x).max() * margin / 127.0
    return np.clip(np.round(x / s), -127, 127).astype(np.int8), float(s)


# ------------------------------------------------------------------------------------------------
def load_state_dict() -> dict[str, np.ndarray]:
    import torch

    ck = torch.load(WHISPER_ROOT / "tiny.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    out = {k: v.float().numpy().astype(np.float64) for k, v in sd.items()}
    # The checkpoint is fp16; openai-whisper loads those values into fp32 parameters, including the
    # sinusoidal positional-embedding buffer, so we use the stored (fp16-rounded) buffer, not a recompute.
    pe = sinusoids(N_AUDIO_CTX, D_MODEL)
    assert np.abs(pe - out["encoder.positional_embedding"]).max() < 1e-3
    return out


def build(cfg: QConfig, sd: dict[str, np.ndarray] | None = None, stats: dict | None = None) -> QModel:
    sd = sd or load_state_dict()
    stats = stats or json.load(open(REPO / "weights" / "calib_stats.json"))
    mx, chan, head = stats["max"], stats["chan"], stats["head"]
    m8, m16 = cfg.margin_int8, cfg.margin_int16
    rb = cfg.residual_bits
    rmax = 127.0 if rb == 8 else 32767.0
    rmargin = m8 if rb == 8 else m16
    qm = QModel(cfg)
    qm.meta = dict(cfg=cfg.__dict__, n_calib=stats["n_clips"])

    def s_res(key):  # residual static scale at a residual point
        return mx[key] * rmargin / rmax

    def s8(key):
        return mx[key] * m8 / 127.0

    def s16(key):
        return mx[key] * m16 / 32767.0

    # ---------------- encoder front-end ----------------
    s_mel = s8("enc.mel")
    qm.scalars["s_mel"] = s_mel
    W1 = np.zeros((D_MODEL, 3, N_MEL_PAD))               # [N, tap, bin] -> K = tap*96 + bin
    W1[:, :, :N_MEL] = sd["encoder.conv1.weight"].transpose(0, 2, 1)
    s_c1 = s16("enc.conv1")
    qm.linears["enc.conv1"] = make_linear("enc.conv1", W1.reshape(D_MODEL, 3 * N_MEL_PAD), sd["encoder.conv1.bias"],
                                          s_mel, False, s_c1, 16)
    s_g1 = s8("enc.conv1")  # GELU(conv1) <= max(conv1) ; int8 static (conv2 input, im2col)
    qm.gelus["enc.conv1"] = (make_gelu("enc.conv1", "phi16", s_c1, req_to_int8=s_g1) if cfg.gelu_mode == "phi16"
                             else make_gelu("enc.conv1", "lut8", s8("enc.conv1"), s_g1))
    if cfg.gelu_mode == "lut8":  # conv1 output must then be int8
        qm.linears["enc.conv1"] = make_linear("enc.conv1", W1.reshape(D_MODEL, 3 * N_MEL_PAD), sd["encoder.conv1.bias"],
                                              s_mel, False, s8("enc.conv1"), 8)
    W2 = sd["encoder.conv2.weight"].transpose(0, 2, 1).reshape(D_MODEL, 3 * D_MODEL)  # K = tap*384 + ch
    s_c2 = s16("enc.conv2") if cfg.gelu_mode == "phi16" else s8("enc.conv2")
    qm.linears["enc.conv2"] = make_linear("enc.conv2", W2, sd["encoder.conv2.bias"], s_g1, False, s_c2,
                                          16 if cfg.gelu_mode == "phi16" else 8)
    s_g2 = s8("enc.conv2")
    qm.gelus["enc.conv2"] = (make_gelu("enc.conv2", "phi16", s_c2) if cfg.gelu_mode == "phi16"
                             else make_gelu("enc.conv2", "lut8", s_c2, s_g2))
    pos8, s_pos = quant_tensor_int8(sd["encoder.positional_embedding"])
    qm.tables["enc.pos"] = pos8
    qm.scalars["s_enc_pos"] = s_pos
    s_x = s_res("enc.0.x_in")
    qm.adds["enc.x0"] = make_add("enc.x0", qm.gelus["enc.conv2"].s_out, s_pos, s_x, rb)

    # ---------------- encoder layers ----------------
    for l in range(4):
        p, sp = f"enc.{l}", f"encoder.blocks.{l}"
        s_x = _block_common(qm, cfg, sd, mx, chan, head, p, sp, s_x, s_res(f"{p}.x_mid"), s_res(f"{p}.x_out"), cross=False)
    qm.lns["enc.ln_post"] = make_ln("enc.ln_post", sd["encoder.ln_post.weight"], sd["encoder.ln_post.bias"], s_x,
                                    mx["enc.ln_post"])
    s_enc = 2.0 ** (-qm.lns["enc.ln_post"].F)
    qm.scalars["s_enc_out"] = s_enc

    # ---------------- decoder ----------------
    emb = sd["decoder.token_embedding.weight"]                # [51865, 384]
    embp = np.zeros((N_VOCAB_PAD, D_MODEL))
    embp[:N_VOCAB] = emb
    qm.linears["dec.lm"] = make_linear("dec.lm", embp, None, 0.0, True, 1.0, 0)   # wide mode
    s_emb = qm.linears["dec.lm"].s_w                           # per-row embedding scale
    dpos8, s_dpos = quant_tensor_int8(sd["decoder.positional_embedding"])
    qm.tables["dec.pos"] = dpos8
    qm.scalars["s_dec_pos"] = s_dpos
    s_x = s_res("dec.0.x_in")
    qm.tables["dec.emb.resmult"] = np.round(s_emb / s_x * (1 << SADD_SHIFT)).astype(np.int32)
    qm.adds["dec.x0"] = make_add("dec.x0", 1.0, s_dpos, s_x, rb)   # ma unused (per-token resmult)
    for l in range(4):
        p, sp = f"dec.{l}", f"decoder.blocks.{l}"
        s_x = _block_common(qm, cfg, sd, mx, chan, head, p, sp, s_x, s_res(f"{p}.x_mid1"), s_res(f"{p}.x_out"),
                            cross=True, s_mid2=s_res(f"{p}.x_mid2"), s_enc=s_enc)
    # final LN + LM head (optionally smoothed into the LN)
    g, b = sd["decoder.ln.weight"], sd["decoder.ln.bias"]
    xmax = np.asarray(chan["dec.ln"])
    if cfg.smooth_alpha is not None:
        s = smooth_factors(xmax, [emb], cfg.smooth_alpha)
        g, b, xmax = g / s, b / s, xmax / s
        embp = embp * s[None, :]
        qm.linears["dec.lm"] = make_linear("dec.lm", embp, None, 0.0, True, 1.0, 0)
        qm.tables["dec.lm.smooth"] = s
    qm.lns["dec.ln"] = make_ln("dec.ln", g, b, s_x, xmax.max())
    qm.linears["dec.lm"].s_in = 2.0 ** (-qm.lns["dec.ln"].F)
    # decoding rules
    meta = json.load(open(REPO / "data" / "ref" / "_meta.json"))
    qm.meta["decoding"] = meta["languages"]
    qm.meta["phi_table"] = phi_table()
    qm.meta["exp_table"] = luts.exp_table()
    qm.meta["rsqrt_table"] = luts.rsqrt_table()
    return qm


def _block_common(qm, cfg, sd, mx, chan, head, p, sp, s_x_in, s_x_mid, s_x_out, cross, s_mid2=None, s_enc=None):
    """One transformer block (encoder: attn+ffn; decoder: self-attn + cross-attn + ffn). Returns s_x_out."""
    m8, m16, rb = cfg.margin_int8, cfg.margin_int16, cfg.residual_bits
    H = N_HEAD

    def attn_block(prefix, ln_key, q_key, k_key, v_key, o_key, torch_attn, torch_ln, x_scale, xmax_key, wv_key,
                   kv_from_ln: bool, s_kv_in: float | None):
        """Self-attn (kv_from_ln): q/k/v all consume this block's LN output. Cross-attn: k/v consume the
        encoder output (scale s_kv_in, dynamic) and only q consumes the LN."""
        Wq, bq = sd[f"{torch_attn}.query.weight"], sd[f"{torch_attn}.query.bias"]
        Wk = sd[f"{torch_attn}.key.weight"]
        Wv, bv = sd[f"{torch_attn}.value.weight"], sd[f"{torch_attn}.value.bias"]
        Wo, bo = sd[f"{torch_attn}.out.weight"], sd[f"{torch_attn}.out.bias"]
        g, b = sd[f"{torch_ln}.weight"], sd[f"{torch_ln}.bias"]
        xmax = np.asarray(chan[xmax_key])
        if cfg.smooth_alpha is not None:
            s = smooth_factors(xmax, [Wq, Wk, Wv] if kv_from_ln else [Wq], cfg.smooth_alpha)
            g, b, xmax = g / s, b / s, xmax / s
            Wq = Wq * s[None, :]
            if kv_from_ln:
                Wk, Wv = Wk * s[None, :], Wv * s[None, :]
        qm.lns[ln_key] = make_ln(ln_key, g, b, x_scale, xmax.max())
        s_ln = 2.0 ** (-qm.lns[ln_key].F)
        s_kv = s_ln if kv_from_ln else s_kv_in
        s_q = np.asarray(head[q_key]) * m8 / 127.0
        s_k = np.asarray(head[k_key]) * m8 / 127.0
        s_v = float(np.max(head[v_key])) * m8 / 127.0
        # wv (attention output) smoothing folds into V columns / O rows
        if cfg.smooth_alpha is not None:
            sv = smooth_factors(np.asarray(chan[wv_key]), [Wo], cfg.smooth_alpha)
            Wv, bv, Wo = Wv / sv[:, None], bv / sv, Wo * sv[None, :]
            s_v = float(np.max(np.asarray(chan[wv_key]) / sv)) * m8 / 127.0
        qm.linears[f"{prefix}.q"] = make_linear(f"{prefix}.q", Wq, bq, s_ln, True, np.repeat(s_q, HEAD_DIM), 8)
        qm.linears[f"{prefix}.k"] = make_linear(f"{prefix}.k", Wk, None, s_kv, True, np.repeat(s_k, HEAD_DIM), 8)
        qm.linears[f"{prefix}.v"] = make_linear(f"{prefix}.v", Wv, bv, s_kv, True, s_v, 8)
        qm.attns[prefix] = make_attn(prefix, s_q, s_k, s_v)
        s_o = mx[o_key] * m16 / 32767.0
        qm.linears[f"{prefix}.o"] = make_linear(f"{prefix}.o", Wo, bo, s_v / 128.0, True, s_o, 16)
        return s_o

    # --- self attention ---
    s_o = attn_block(f"{p}.attn", f"{p}.ln1", f"{p}.q", f"{p}.k", f"{p}.v", f"{p}.o", f"{sp}.attn", f"{sp}.attn_ln",
                     s_x_in, f"{p}.ln1", f"{p}.wv", True, None)
    qm.adds[f"{p}.add1"] = make_add(f"{p}.add1", s_x_in, s_o, s_x_mid, rb)
    s_x = s_x_mid
    if cross:
        s_co = attn_block(f"{p}.xattn", f"{p}.lnc", f"{p}.cq", f"{p}.ck", f"{p}.cv", f"{p}.co", f"{sp}.cross_attn",
                          f"{sp}.cross_attn_ln", s_x, f"{p}.lnc", f"{p}.cwv", False, s_enc)
        qm.adds[f"{p}.add2"] = make_add(f"{p}.add2", s_x, s_co, s_mid2, rb)
        s_x = s_mid2
    # --- FFN ---
    g, b = sd[f"{sp}.mlp_ln.weight"], sd[f"{sp}.mlp_ln.bias"]
    W1, b1 = sd[f"{sp}.mlp.0.weight"], sd[f"{sp}.mlp.0.bias"]
    W2, b2 = sd[f"{sp}.mlp.2.weight"], sd[f"{sp}.mlp.2.bias"]
    xmax = np.asarray(chan[f"{p}.ln2"])
    if cfg.smooth_alpha is not None:
        s = smooth_factors(xmax, [W1], cfg.smooth_alpha)
        g, b, xmax, W1 = g / s, b / s, xmax / s, W1 * s[None, :]
    qm.lns[f"{p}.ln2"] = make_ln(f"{p}.ln2", g, b, s_x, xmax.max())
    s_ln2 = 2.0 ** (-qm.lns[f"{p}.ln2"].F)
    if cfg.gelu_mode == "lut8":
        s_h = mx[f"{p}.fc1"] * m8 / 127.0
        s_g = mx[f"{p}.gelu"] * m8 / 127.0
        qm.linears[f"{p}.fc1"] = make_linear(f"{p}.fc1", W1, b1, s_ln2, True, s_h, 8)
        qm.gelus[f"{p}.gelu"] = make_gelu(f"{p}.gelu", "lut8", s_h, s_g)
        qm.linears[f"{p}.fc2"] = make_linear(f"{p}.fc2", W2, b2, s_g, False, mx[f"{p}.fc2"] * m16 / 32767.0, 16)
    else:
        s_h = mx[f"{p}.fc1"] * m16 / 32767.0
        qm.linears[f"{p}.fc1"] = make_linear(f"{p}.fc1", W1, b1, s_ln2, True, s_h, 16)
        smooth = None
        if cfg.smooth_fc2:
            smooth = smooth_factors(np.asarray(chan[f"{p}.gelu"]), [W2], cfg.smooth_alpha or 0.5)
            W2 = W2 * smooth[None, :]
        qm.gelus[f"{p}.gelu"] = make_gelu(f"{p}.gelu", "phi16", s_h, smooth=smooth)
        qm.linears[f"{p}.fc2"] = make_linear(f"{p}.fc2", W2, b2, s_h, True, mx[f"{p}.fc2"] * m16 / 32767.0, 16)
    key = "add3" if cross else "add2"
    qm.adds[f"{p}.{key}"] = make_add(f"{p}.{key}", s_x, qm.linears[f"{p}.fc2"].s_out[0], s_x_out, rb)
    return s_x_out
