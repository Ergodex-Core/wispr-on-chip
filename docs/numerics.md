# Numerics specification (frozen after calibration)

The golden model `golden/minicpm_int.py` (+ `fixedpoint.py`, `luts.py`, `quant.py`) is the executable form
of this document. RTL is correct iff bit-identical to it. All arithmetic is integer; the only floats are in
`quant.py` at weight-generation time and in the fp32 reference (`golden/reference_cpu.py`).

Model: openbmb/MiniCPM5-2B = Llama architecture, 42 layers, d = 2048, 16 query heads and 2 KV heads of 128,
FFN 6144 (SiLU gate), vocab 130560, RMSNorm eps 1e-6, RoPE theta 5·10⁶, no biases, untied embeddings.

## Conventions
* `rsr(x, s)` = `(x + 2^(s-1)) >> s` (arithmetic shift, round-half-up); `rsr(x, 0) = x`.
* `satW(x)` = clamp to `[-2^(W-1), 2^(W-1)-1]`.
* Tensors are `[rows, features]`; matmul weights are `W[K, N]` (K = reduction, N = output channel).
* Every static scale `S` is a float chosen at generation time from `weights/calib_stats.json`; it never
  appears in the datapath, only the derived integer multipliers/shifts do. Static int16 scales are
  `max_calib·m/32767`, static int32 scales `max_calib·m/(2^31−1)` with the margin `m = 1.5`
  (`QConfig.margin_int16`, decision #13), static int8 scales `max_calib/127`.
* **SmoothQuant folding** (`QConfig.smooth_alpha = 0.5`): before quantisation, the per-input-channel factor
  `s_k = max|X_k|^α / max_n|W[n,k]|^(1−α)` is divided into the producing RMSNorm gain and multiplied into
  the consuming weight columns (norm1 → q/k/v, norm2 → gate/up, norm_f → LM head). This is exact in real
  arithmetic and invisible to the datapath: it only changes the constants in `G[k]` and `w8[k,n]`.

## Activations
| Tensor | Format | Scale |
|---|---|---|
| residual stream `x` (per layer input, after attention, after MLP) | **int32 static** (`cfg.residual_bits`) | `max_calib·m/(2^31−1)` per residual point |
| embedding rows | int8 per token | `s_emb[tok] = max|row|/127` |
| RMSNorm out | int16, scale `2^-F` per norm (`F = floor(log2(32767/max_calib_channel))`), transient (row buffer) | |
| matmul inputs | int8 per-token dynamic (below) | real = `a8 · (m16<<b) · S_in / 127` |
| Q, K projections | int16 static per head (`max_head·m/32767`) | rotated and requantised by RoPE |
| Q, K after RoPE | int8 static per head (`max_head/127`) | |
| V | int8 static per tensor (`max_calib/127`) | |
| attention out | int16 `= rsr(O·rl, 33)` = `O/l·128`, scale `S_v/128`, transient → dynamic quant | |
| o_proj out, down_proj out | **int32 static** `max_calib·m/(2^31−1)` (feed the residual add) | |
| gate, up | int16 static | |
| gated hidden `h = silu(gate)·up` | **int32 exact product**, scale `S_gate·S_up`, transient → dynamic quant | |
| LM logits | int32 wide (argmax only) | |

Why 32 bits: MiniCPM5-2B has Llama's massive activations. On 836 calibration tokens the residual reaches
5330 from layer 8 on while typical elements are ~1; layer 7's MLP emits 4500 and layers 40/41 emit 1900/1500
into the residual. A static int16 residual (whisper-si, decision #5 there) would leave typical values at
0–3 LSBs. Int32 keeps 15+ bits for typical values everywhere (docs/decisions.md #3).

## Dynamic per-token quantisation (producer side; int16 or int32 rows)
```
maxabs = max(1, max_k |x[k]|)
b      = max(bitlen(maxabs) − 16, 0)            (0 for int16 rows)
m16    = maxabs >> b                             (uint16 ≥ 1)
recip  = floor((127·2^16 + m16/2) / m16)         (24 bits)
a8[k]  = sat8( rsr(x[k] · recip, 16 + b) )
rowfac = m16 | (b << 16)                          (side table, 24 bits; effective factor m16 << b)
```

## Matmul edge requant (fused in the engine drain, per output column n, per row m)
```
acc = Σ_k a8[m,k] · w8[k,n]                          int32 (exact)
t   = sat48( rsr(acc · M[n], s1) )                   M[n] int32 in [0, 2^31), s1 per op
u   = t · m16[m] + (B[n] << (s2 − 16))               (bias term only for int8/int16 outputs; MiniCPM has B = 0)
y   = satW( rsr(u, s2 − b[m]) )                      s2 = 24 (W = 8), 20 (W = 16), 24 (W = 32)
```
`M[n] = round(S_in'·s_w[n]/s_out[n]·2^(s1+s2))`, `S_in' = S_in/127`, `s1 = 30 − s2 − floor(log2 max_n c[n])`.
Folding the row factor's power of two `b` into the final shift keeps `u` below 2^64 (t < 2^47, m16 < 2^16)
while `t` keeps 24 fraction bits for int32 outputs (decision #4).
Wide mode (LM head): `t = sat32(rsr(acc·M[n], 22))`, `M[n] = round(s_w[n]/max(s_w)·2^30)`; the common
row factor of the single row is dropped (argmax-invariant).
Weights: int8 symmetric per output channel, `s_w[n] = max_k|W| / 127`, `w8 = round(W/s_w)`.

## Scaled add (residual, int32 operands)
`out = sat32( rsr(a·Ma + b·Mb, 16) )`, `Ma = round(S_a/S_out·2^16)`, `Mb = round(S_b/S_out·2^16)`.

## Embedding
`x0[k] = sat32( e8[tok][k] · resmult[tok] )`, `resmult[tok] = round(s_emb[tok] / S_x0)` (int32, `EMB_FRAC = 0`).

## RMSNorm (N = 2048, int32 input x, scale S_in)
```
V   = Σ x² + eps_q                              eps_q = round(1e-6 · 2048 / S_in²)   (V up to 2^75: 80-bit)
e   = (bitlen(V) − 29) // 2 ; m = V >> 2e        (m in [2^28, 2^30); e may be negative → left shift)
y0  = RSQRT[m >> 22]                             256-entry LUT, RSQRT[i] = round(2^30/sqrt((i+0.5)·2^22)), i in 64..255
t   = m·y0·y0 ; d = rsr(3·2^60 − t, 30) ; y1 = rsr(y0·d, 31)        (one Newton step, y1 ≈ 2^30/sqrt(m))
u   = rsr(x · y1, 14 + e)                        (= x/sqrt(V) · 2^16 = n_k/sqrt(N) · 2^16, |u| ≤ 2^16)
v   = rsr(u · G[k], 12)                          G[k] = round(sqrt(N) · gamma[k] · 2^F)
y16 = sat16( rsr(v, 4) )
```

## RoPE (per head, pairs (d, d+64), HF Llama rotate_half convention)
```
c[i] = round(cos(pos·θ_i)·32767), s[i] = round(sin(pos·θ_i)·32767)   θ_i = theta^(−2i/128) (fp32 like HF), table [maxCtx, 128] int16
r[d]    = x16[d]·c[d] − x16[d+64]·s[d]          r[d+64] = x16[d+64]·c[d] + x16[d]·s[d]
r16     = sat16( rsr(r, 15) )
a8      = sat8( rsr(r16 · m_r, s_r) )           m_r/s_r = mult_shift(S16/S8) = 254/32767 (same for every head)
```
The int16 and int8 scales of a head both derive from the head's calibrated maximum over pre- and
post-rotation values, so the ratio is a constant.

## Attention (per head h with kv head h//8; score `s = q8·k8`, |s| ≤ 2^21; key tiles of 64 in ascending order)
```
Mq[h] = round(S_q[h]·S_k[h//8]/sqrt(128) · 2^8 · 2^sq[h]),  sq chosen so Mq in [2^30, 2^31)
per key tile t (keys j0..j0+63; valid iff j < n_keys and (causal) j <= qPos0 + m):
  mt   = max over valid s[j]              (tile skipped for a query with no valid key)
  mnew = max(mx, mt)   (first tile: mnew = mt, no rescale)
  if mnew > mx:  da = min(4095, rsr((mnew − mx)·Mq, sq)); alpha = EXP(da); l = rsr(l·alpha, 15); O = rsr(O·alpha, 15)
  mx   = mnew
  d[j] = min(4095, rsr((mx − s[j])·Mq, sq)) ; p[j] = EXP(d[j]) (0 if invalid)      p in [0, 32768] (Q1.15)
  l   += Σ p[j] ;  O[d] += Σ_j p[j]·v8[j][d]                                       (O: 36-bit signed)
after the last tile:  rl = floor((2^40 + l/2) / l) ;  out16[d] = sat16( rsr(O[d]·rl, 33) )   (= O/l · 2^7)
EXP(d): i = d >> 4, f = d & 15, T[i] = round(exp(−i/16)·2^15) (i = 0..255), T[256] = 0, EXP = T[i] − rsr((T[i] − T[i+1])·f, 4)
```
Hardware detail with no numeric effect: `P·V` runs on the int8 array as two unsigned passes
(`hi = p >> 8`, `lo = p & 255`), `O += (acc_hi << 8) + acc_lo`. Chunked prefill (queries `[c0, c0+n)` against
keys `[0, c0+n)` with `qPos0 = c0`) is bit-identical to a full causal prefill and to one-row decoding.

## SiLU gate
```
xf     = clamp(rsr(g16 · m_sig, s_sig), −2048, 2047)      (= gate value in Q.8; m_sig/s_sig = mult_shift(S_gate·2^8))
σq     = SIG[i] + rsr((SIG[i+1] − SIG[i])·f, 4),  i = (xf >> 4) + 128, f = xf & 15
         SIG[i] = round(sigmoid((i−128)/16)·2^15) (256 entries), SIG[256] = round(sigmoid(8)·2^15) = 32757
silu16 = sat16( rsr(g16 · σq, 15) )                        (scale S_gate)
h32    = silu16 · up16                                     (exact, scale S_gate·S_up, |h| < 2^30) → dynamic quant
```

## Decoding rules
Prompt = the chat template `<s><|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
(thinking disabled) or raw text after `<s>`. Greedy: argmax = lowest index among maxima. Stop at `</s>` (1) or
`<|im_end|>` (130073), after `max_new` tokens, or when the token buffer (maxCtx = 2048) is full.
