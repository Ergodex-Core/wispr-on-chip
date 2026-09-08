"""Quantisation: builds the complete integer model (weights, scales, requant constants, LUTs) of
MiniCPM5-2B (a 42-layer Llama: d = 2048, 16 query heads / 2 KV heads of 128, FFN 6144, vocab 130560)
from the bf16 checkpoint + calibration statistics. Every constant the RTL uses originates here.

Numerics summary (docs/numerics.md is the normative text):
  * matmul weights: int8 symmetric, per-output-channel scale s_w[n]; no biases (Llama)
  * matmul inputs: int8 per-token dynamic (maxabs row factor) from RMSNorm / attention / SiLU-gate outputs
  * requant: t = sat40(rsr(acc*M[n], s1)); y = sat_w(rsr(t*rowfac[m], s2)), s2 = 24 (int8) / 20 (int16) / 8 (int32)
  * residual stream: int32 static per-point scale (cfg.residual_bits; Llama's massive activations reach 5300
    while typical values are ~1, see docs/decisions.md); o_proj / down_proj outputs are int32 static too
  * RMSNorm: integer sum of squares, rsqrt LUT + Newton, gain folded into the output scale, int16 out
  * RoPE: Q/K projections are int16 (static per-head scale); rotation with Q15 cos/sin tables, then a
    static requant to int8 per head (constant ratio 254/32767 between the int16 and int8 scales)
  * attention: online softmax, exp LUT (Q1.15), per-head static score multiplier, GQA (head h -> kv h//8)
  * SiLU-gated MLP: silu(g) = g * sigmoid(g) with a 256-entry Q15 sigmoid table on int16; the exact 32-bit
    product with the int16 up projection is quantised per token (dynamic, 32-bit rule) for down_proj
  * embedding: int8 per row (token) with a per-token int32 multiplier into the int32 residual
  * LM head: wide mode (int32 logits, argmax only), split into column slices of LM_SLICE
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
from golden.data import CHECKPOINT, CONFIG, REPO  # noqa: E402
from golden.fixedpoint import REQ_B_FRAC, REQ_S2_BY_BITS, SADD_SHIFT, mult_for, mult_shift  # noqa: E402

D_MODEL, N_HEAD, N_KV_HEAD, HEAD_DIM, D_FF = 2048, 16, 2, 128, 6144
KV_DIM = N_KV_HEAD * HEAD_DIM                  # 256
N_VOCAB, N_LAYERS = 130560, 42
RMS_EPS = 1e-6
ROPE_THETA = 5_000_000.0
MAX_CTX = 2048                                 # positions (KV cache depth, RoPE table rows)
LM_SLICE = 8192                                # LM-head columns per weight tensor slice (16 slices)
EMB_SLICE = 8192                               # embedding rows per table slice (16 slices)
N_LM_SLICES = (N_VOCAB + LM_SLICE - 1) // LM_SLICE
TILE = 32
LN_U_FRAC = 16           # u = n_k/sqrt(N) * 2^16
LN_V_SHIFT = 12          # v = rsr(u*G, 12)  (= y * 2^4 in output units)
LN_B_FRAC = 4            # y16 = sat16(rsr(v, 4))
ROPE_MULT_BITS = 31
EMB_FRAC = 0             # x0 = e8 * resmult (resmult = round(s_emb / S_x0)); int32 residual needs no fraction bits


@dataclass
class QConfig:
    residual_bits: int = 32
    margin_int8: float = 1.0           # static int8 scales = max*margin/127
    margin_int16: float = 1.5          # static int16/int32 scales = max*margin/32767 (or /2^31-1)
    smooth_alpha: float | None = 0.5   # SmoothQuant folding into the RMSNorm gains (None = off)
    n_layers: int = N_LAYERS           # < 42 only for smoke tests (truncated model)

    def tag(self) -> str:
        return f"r{self.residual_bits}_m8{self.margin_int8}_m16{self.margin_int16}_sa{self.smooth_alpha}_L{self.n_layers}"


@dataclass
class Linear:
    """W[K,N] int8 (+ requant). rowfac semantics: dynamic -> S_in_eff = S_in/127, static -> S_in."""
    name: str
    w8: np.ndarray            # [K, N] int8
    s_w: np.ndarray           # [N] float
    mult: np.ndarray          # [N] int32
    bias: np.ndarray          # [N] int32 (all zero for MiniCPM)
    s1: int
    out_bits: int             # 8, 16 or 32 (0 = wide mode for the LM head)
    dynamic: bool
    s_in: float               # scale of the producer's int16 tensor (S_in)
    s_out: np.ndarray         # [N] float (per channel, may be constant)


@dataclass
class RmsNormQ:
    name: str
    g: np.ndarray     # [D_MODEL] int32
    eps_q: int
    F: int            # output scale 2^-F
    s_in: float       # residual scale of the input


@dataclass
class RopeQ:
    """int16 (scale s16 per head) -> rotate (Q15 cos/sin) -> int8 (scale s8 per head): a8 = sat8(rsr(r16 * m, s))."""
    name: str
    m_r: int
    s_r: int
    n_heads: int
    s16: np.ndarray   # [heads] int16 input scale per head
    s8: np.ndarray    # [heads] int8 output scale per head


@dataclass
class AttnQ:
    name: str
    mq: np.ndarray    # [N_HEAD] int32
    sq: np.ndarray    # [N_HEAD] int
    s_q: np.ndarray   # [N_HEAD] float (Q int8 scale per head)
    s_k: np.ndarray   # [N_KV_HEAD]
    s_v: float


@dataclass
class SiluQ:
    """h32 = silu16(g16) * up16 (exact); silu16 = sat16(rsr(g16 * SIG(xf), 15)), xf = rsr(g16*m_sig, s_sig) (x in Q.8).
    h32 has the static scale s_gate * s_up and is quantised per token for down_proj."""
    name: str
    m_sig: int
    s_sig: int
    s_gate: float
    s_up: float
    s_out: float


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
class QModel:
    cfg: QConfig
    linears: dict[str, Linear] = field(default_factory=dict)
    norms: dict[str, RmsNormQ] = field(default_factory=dict)
    ropes: dict[str, RopeQ] = field(default_factory=dict)
    attns: dict[str, AttnQ] = field(default_factory=dict)
    silus: dict[str, SiluQ] = field(default_factory=dict)
    adds: dict[str, AddQ] = field(default_factory=dict)
    tables: dict[str, np.ndarray] = field(default_factory=dict)   # embed (int8), embed.resmult (int32), rope (int16)
    scalars: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------
def quant_per_channel(W: np.ndarray):
    """Symmetric per-output-channel int8. W float [N, K] -> (w8 [N,K], s [N])."""
    W = np.asarray(W, dtype=np.float32)
    amax = np.abs(W).max(axis=1).astype(np.float64)
    s = np.where(amax > 0, amax / 127.0, 1.0)
    w8 = np.clip(np.round(W / s[:, None].astype(np.float32)), -127, 127).astype(np.int8)
    return w8, s


def make_linear(name: str, W: np.ndarray, b: np.ndarray | None, s_in: float, dynamic: bool,
                s_out, out_bits: int) -> Linear:
    """W float [N, K] (torch layout), b [N] or None. s_out: scalar or [N]."""
    N, K = W.shape
    assert K % TILE == 0 and N % TILE == 0, (name, K, N)
    w8, s_w = quant_per_channel(W)
    s_out = np.broadcast_to(np.asarray(s_out, dtype=np.float64), (N,)).copy()
    s_in_eff = s_in / 127.0 if dynamic else s_in
    if out_bits == 0:  # wide mode (argmax): M[n] = s_w[n]/max(s_w) * 2^30, s1 = 22
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


def make_rms(name: str, gamma, s_in: float, max_out: float) -> RmsNormQ:
    """RMSNorm on a residual of scale s_in; output int16 with scale 2^-F where F fits max_out."""
    F = int(math.floor(math.log2(32767.0 / max(max_out, 1e-3))))
    F = max(0, min(F, 14))
    s_ln = 2.0 ** (-F)
    # u = n_k/sqrt(N)*2^16 ; u*G = n*gamma*2^(F+16) ; v = rsr(u*G,12) = n*gamma*2^(F+4) ; y = rsr(v,4)
    g = np.round(math.sqrt(D_MODEL) * np.asarray(gamma, np.float64) / s_ln)
    assert np.abs(g).max() < (1 << 31), name
    eps_q = int(round(RMS_EPS * D_MODEL / (s_in ** 2)))
    assert eps_q >= 1, (name, eps_q)
    return RmsNormQ(name, g.astype(np.int32), eps_q, F, s_in)


def make_rope(name: str, s16: np.ndarray, s8: np.ndarray) -> RopeQ:
    ratio = np.asarray(s16, np.float64) / np.asarray(s8, np.float64)
    assert np.allclose(ratio, ratio[0]), (name, ratio)   # both scales derive from the same per-head max
    m, s = mult_shift(float(ratio[0]), ROPE_MULT_BITS)
    return RopeQ(name, m, s, len(s16), np.asarray(s16, np.float64), np.asarray(s8, np.float64))


def make_attn(name: str, s_q: np.ndarray, s_k: np.ndarray, s_v: float) -> AttnQ:
    s_k_h = np.asarray([s_k[h // (N_HEAD // N_KV_HEAD)] for h in range(N_HEAD)])
    c = np.asarray(s_q) * s_k_h / math.sqrt(HEAD_DIM) * (1 << luts.EXP_FRAC_BITS)   # d/256 = score_int * c/256
    sq = np.array([30 - int(math.floor(math.log2(x))) for x in c], dtype=np.int64)
    mq = np.array([int(round(x * 2.0 ** s)) for x, s in zip(c, sq)], dtype=np.int64)
    assert mq.max() < (1 << 31) and sq.min() >= 0 and sq.max() < 64
    return AttnQ(name, mq.astype(np.int32), sq, np.asarray(s_q, np.float64), np.asarray(s_k, np.float64), s_v)


def make_silu(name: str, s_gate: float, s_up: float) -> SiluQ:
    m_sig, s_sig = mult_shift(s_gate * (1 << 8))        # xf = g16 * s_gate in Q.8
    return SiluQ(name, m_sig, s_sig, s_gate, s_up, s_gate * s_up)


def make_add(name: str, s_a: float, s_b: float, s_out: float, out_bits: int) -> AddQ:
    return AddQ(name, mult_for(s_a / s_out, SADD_SHIFT), mult_for(s_b / s_out, SADD_SHIFT), s_a, s_b, s_out, out_bits)


def smooth_factors(x_chan_max: np.ndarray, W_list: list[np.ndarray], alpha: float) -> np.ndarray:
    """SmoothQuant per-input-channel factor s_k = max|X_k|^a / max_n|W[n,k]|^(1-a) (W in torch layout [N, K]).

    The factor is folded into the producing RMSNorm gain (gamma/s) and into the consuming weight columns
    (W*s), which is exact in real arithmetic and moves quantisation difficulty from the per-token int8
    activation to the per-output-channel int8 weight. Nothing in the datapath changes."""
    wmax = np.max([np.abs(np.asarray(W, np.float32)).max(axis=0) for W in W_list], axis=0).astype(np.float64)
    x = np.maximum(np.asarray(x_chan_max, np.float64), 1e-5)
    wmax = np.maximum(wmax, 1e-5)
    return np.clip(x ** alpha / wmax ** (1 - alpha), 1e-2, 1e2)


# ------------------------------------------------------------------------------------------------
class WeightSource:
    """Lazy float32 numpy view of the bf16 safetensors checkpoint."""

    def __init__(self, path=CHECKPOINT):
        from safetensors import safe_open
        self.f = safe_open(str(path), framework="pt")
        self.keys = set(self.f.keys())

    def __call__(self, name: str) -> np.ndarray:
        return self.f.get_tensor(name).float().numpy()


def default_stats_path() -> Path:
    return REPO / "weights" / "calib_stats.json"


def manifest_config(path: Path | None = None) -> QConfig:
    """The QConfig the weight images on disk were generated with. Vector generators use this so the golden
    model can never drift from the weights the RTL loads (fields absent from an older manifest take the
    value the code had when it was written)."""
    man = json.load(open(path or REPO / "weights" / "MANIFEST.json"))
    c = dict(man["config"])
    c.setdefault("smooth_alpha", None)
    unknown = set(c) - set(QConfig().__dict__)
    assert not unknown, f"manifest config has unknown fields {unknown}"
    return QConfig(**c)


def build(cfg: QConfig, sd: WeightSource | None = None, stats: dict | None = None, verbose: bool = False) -> QModel:
    sd = sd or WeightSource()
    stats = stats or json.load(open(default_stats_path()))
    mx, chan, head = stats["max"], stats["chan"], stats["head"]
    m8, m16 = cfg.margin_int8, cfg.margin_int16
    rb = cfg.residual_bits
    rmax = {8: 127.0, 16: 32767.0, 32: float(2 ** 31 - 1)}[rb]
    rmargin = m8 if rb == 8 else m16
    qm = QModel(cfg)
    qm.meta = dict(cfg=cfg.__dict__, n_calib_tokens=stats["n_tokens"], model_config=json.load(open(CONFIG)))
    hf = json.load(open(CONFIG))
    assert hf["hidden_size"] == D_MODEL and hf["num_attention_heads"] == N_HEAD and hf["num_key_value_heads"] == N_KV_HEAD
    assert hf["intermediate_size"] == D_FF and hf["vocab_size"] == N_VOCAB and hf["head_dim"] == HEAD_DIM
    assert hf["rms_norm_eps"] == RMS_EPS and hf["rope_theta"] == ROPE_THETA and hf["hidden_act"] == "silu"

    def s_res(l):  # residual static scale at the input of layer l (l = n_layers: final norm input)
        return mx[f"x_in.{l}"] * rmargin / rmax

    def s16(key):
        return mx[key] * m16 / 32767.0

    def s32(key):
        return mx[key] * m16 / float(2 ** 31 - 1)

    # ---------------- embedding: int8 per row, per-token multiplier into the int16 residual ----------------
    s_x0 = s_res(0)
    emb = sd("model.embed_tokens.weight")                    # [V, D] float32
    e8 = np.empty((N_VOCAB, D_MODEL), np.int8)
    s_emb = np.empty(N_VOCAB, np.float64)
    for r0 in range(0, N_VOCAB, EMB_SLICE):
        w8, s = quant_per_channel(emb[r0:r0 + EMB_SLICE])
        e8[r0:r0 + EMB_SLICE] = w8
        s_emb[r0:r0 + EMB_SLICE] = s
    del emb
    qm.tables["embed"] = e8
    # x0 = sat_rb(e8 * resmult[tok]) with resmult = round(s_emb[tok] / S_x0 * 2^EMB_FRAC) (EMB_FRAC = 0 for int32)
    resmult = np.round(s_emb / s_x0 * (1 << EMB_FRAC))
    assert resmult.max() < (1 << 31) and resmult.min() >= 0, (resmult.min(), resmult.max())
    qm.tables["embed.resmult"] = resmult.astype(np.int32)
    qm.scalars["s_x0"] = s_x0

    # ---------------- layers ----------------
    for l in range(cfg.n_layers):
        p, sp = f"L{l}", f"model.layers.{l}"
        s_x = s_res(l)
        # -- attention block
        g1 = sd(f"{sp}.input_layernorm.weight")
        Wq, Wk, Wv = sd(f"{sp}.self_attn.q_proj.weight"), sd(f"{sp}.self_attn.k_proj.weight"), sd(f"{sp}.self_attn.v_proj.weight")
        xmax1 = np.asarray(chan[f"norm1.{l}"], np.float64)
        if cfg.smooth_alpha is not None:
            sm = smooth_factors(xmax1, [Wq, Wk, Wv], cfg.smooth_alpha)
            g1, xmax1 = g1 / sm.astype(np.float32), xmax1 / sm
            Wq, Wk, Wv = Wq * sm.astype(np.float32)[None, :], Wk * sm.astype(np.float32)[None, :], Wv * sm.astype(np.float32)[None, :]
        qm.norms[f"{p}.norm1"] = make_rms(f"{p}.norm1", g1, s_x, float(np.max(xmax1)))
        s_ln1 = 2.0 ** (-qm.norms[f"{p}.norm1"].F)
        max_q = np.asarray(head[f"q.{l}"], np.float64)      # [16] max |q| over pre- and post-RoPE
        max_k = np.asarray(head[f"k.{l}"], np.float64)      # [2]
        s_q8, s_q16 = max_q * m8 / 127.0, max_q * m16 / 32767.0
        s_k8, s_k16 = max_k * m8 / 127.0, max_k * m16 / 32767.0
        s_v = mx[f"v.{l}"] * m8 / 127.0
        qm.linears[f"{p}.q"] = make_linear(f"{p}.q", Wq, None, s_ln1, True, np.repeat(s_q16, HEAD_DIM), 16)
        qm.linears[f"{p}.k"] = make_linear(f"{p}.k", Wk, None, s_ln1, True, np.repeat(s_k16, HEAD_DIM), 16)
        qm.linears[f"{p}.v"] = make_linear(f"{p}.v", Wv, None, s_ln1, True, s_v, 8)
        del Wq, Wk, Wv
        qm.ropes[f"{p}.rope_q"] = make_rope(f"{p}.rope_q", s_q16, s_q8)
        qm.ropes[f"{p}.rope_k"] = make_rope(f"{p}.rope_k", s_k16, s_k8)
        qm.attns[f"{p}.attn"] = make_attn(f"{p}.attn", s_q8, s_k8, s_v)
        s_o = s32(f"o.{l}")
        qm.linears[f"{p}.o"] = make_linear(f"{p}.o", sd(f"{sp}.self_attn.o_proj.weight"), None, s_v / 128.0, True, s_o, 32)
        s_mid = mx[f"x_mid.{l}"] * rmargin / rmax
        qm.adds[f"{p}.add1"] = make_add(f"{p}.add1", s_x, s_o, s_mid, rb)
        # -- MLP block
        g2 = sd(f"{sp}.post_attention_layernorm.weight")
        Wg, Wu = sd(f"{sp}.mlp.gate_proj.weight"), sd(f"{sp}.mlp.up_proj.weight")
        xmax2 = np.asarray(chan[f"norm2.{l}"], np.float64)
        if cfg.smooth_alpha is not None:
            sm = smooth_factors(xmax2, [Wg, Wu], cfg.smooth_alpha)
            g2, xmax2 = g2 / sm.astype(np.float32), xmax2 / sm
            Wg, Wu = Wg * sm.astype(np.float32)[None, :], Wu * sm.astype(np.float32)[None, :]
        qm.norms[f"{p}.norm2"] = make_rms(f"{p}.norm2", g2, s_mid, float(np.max(xmax2)))
        s_ln2 = 2.0 ** (-qm.norms[f"{p}.norm2"].F)
        s_gate, s_up = s16(f"gate.{l}"), s16(f"up.{l}")
        qm.linears[f"{p}.gate"] = make_linear(f"{p}.gate", Wg, None, s_ln2, True, s_gate, 16)
        qm.linears[f"{p}.up"] = make_linear(f"{p}.up", Wu, None, s_ln2, True, s_up, 16)
        del Wg, Wu
        qm.silus[f"{p}.silu"] = make_silu(f"{p}.silu", s_gate, s_up)
        s_down = s32(f"down.{l}")
        qm.linears[f"{p}.down"] = make_linear(f"{p}.down", sd(f"{sp}.mlp.down_proj.weight"), None, s_gate * s_up, True, s_down, 32)
        qm.adds[f"{p}.add2"] = make_add(f"{p}.add2", s_mid, s_down, s_res(l + 1), rb)
        if verbose:
            print(f"layer {l}: s_x={s_x:.3e} F1={qm.norms[f'{p}.norm1'].F} F2={qm.norms[f'{p}.norm2'].F} s_v={s_v:.3e} s_gate={s_gate:.3e}")

    # ---------------- final norm + LM head ----------------
    lf = cfg.n_layers
    gf = sd("model.norm.weight")
    Wlm = sd("lm_head.weight")
    xmaxf = np.asarray(chan["norm_f"], np.float64)
    if cfg.smooth_alpha is not None:
        sm = smooth_factors(xmaxf, [Wlm], cfg.smooth_alpha)
        gf, xmaxf = gf / sm.astype(np.float32), xmaxf / sm
        Wlm = Wlm * sm.astype(np.float32)[None, :]
    qm.norms["norm_f"] = make_rms("norm_f", gf, s_res(lf), float(np.max(xmaxf)))
    qm.linears["lm"] = make_linear("lm", Wlm, None, 2.0 ** (-qm.norms["norm_f"].F), True, 1.0, 0)
    del Wlm
    qm.tables["rope"] = luts.rope_table(MAX_CTX, HEAD_DIM, ROPE_THETA)
    qm.meta["exp_table"] = luts.exp_table()
    qm.meta["rsqrt_table"] = luts.rsqrt_table()
    qm.meta["sigmoid_table"] = luts.sigmoid_table() + [luts.sigmoid_last()]
    return qm
