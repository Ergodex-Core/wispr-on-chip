# Numerics specification (frozen at the end of Phase 1)

The golden model `golden/whisper_int.py` (+ `fixedpoint.py`, `luts.py`, `quant.py`) is the executable
form of this document. RTL is correct iff bit-identical to it. All arithmetic is integer; the only
floats are in `quant.py` at weight-generation time.

## Conventions
* `rsr(x, s)` = `(x + 2^(s-1)) >> s` (arithmetic shift, round-half-up); `rsr(x, 0) = x`.
* `satW(x)` = clamp to `[-2^(W-1), 2^(W-1)-1]`.
* Tensors are `[rows, features]`; matmul weights are `W[K, N]` (K = reduction, N = output channel).
* Every static scale `S` is a float chosen at generation time; it never appears in the datapath, only
  the derived integer multipliers/shifts do.

## Activations
| Tensor | Format | Scale |
|---|---|---|
| mel (host) | int8 `[n_frames, 96]`, bins 80..95 zero | `s_mel = max_calib * 1.0 / 127` (static) |
| conv1 out | int16 static | `max_calib*2/32767` |
| conv1 GELU out | int8 static (im2col input to conv2) | `max_calib/127` |
| conv2 out / GELU | int16 static | `max_calib*2/32767` |
| residual stream `x` | int16 static, one scale per residual point (`cfg.residual_bits`; int8 is a supported option, see decisions) | `max_calib*2/32767` |
| LN out | int16, scale `2^-F` per LN (`F = floor(log2(32767/max_calib))`), transient (row buffer) | |
| matmul inputs (from LN / attention / GELU16) | int8 per-token dynamic: `a8 = sat8(rsr(x16 * recip, 16))`, `recip = floor((127<<16 + maxabs/2)/maxabs)`, `maxabs = max(1, max|x16|)` (uint16) | real = `a8 * maxabs * S_in / 127` |
| Q, K | int8 static per head (`max_calib_head/127`) | |
| V | int8 static per tensor (`max_calib/127`) | |
| attention out | int16 `= rsr(O * rl, 33)` = `O/l * 128`, scale `S_v/128`, transient → dynamic quant | |
| out-proj / fc2 out | int16 static `max_calib*2/32767` (feeds residual add) | |
| fc1 out | int16 static (`phi16` GELU mode) | |
| LM logits | int32 wide (argmax only) | |

## Matmul edge requant (fused, per output column n, per row m)
```
acc  = sum_k a8[m,k] * w8[k,n]                     int32 (exact)
t    = sat40( rsr(acc * M[n], s1) )                 M[n] int32 in [0, 2^31), s1 per op
u    = t * rowfac[m] + (B[n] << (s2 - 16))          rowfac = maxabs[m] (dynamic) or 1 (static)
y    = satW( rsr(u, s2) )                           s2 = 24 (W=8), 20 (W=16)
```
`M[n] = round(S_in' * s_w[n] / s_out[n] * 2^(s1+s2))`, `S_in' = S_in/127` (dynamic) or `S_in` (static),
`s1 = 30 - s2 - floor(log2(max_n c[n]))`; `B[n] = round(bias[n]/s_out[n] * 2^16)`.
Wide mode (LM head): `t = sat32(rsr(acc * M[n], 22))`, `M[n] = round(s_emb[n]/max(s_emb) * 2^30)`.
Weights: int8 symmetric per output channel, `s_w[n] = max_k|W| / 127`, `w8 = round(W/s_w)`.

## Scaled add (residual, pos-emb, embedding)
`out = satW( rsr(a * Ma + b * Mb, 16) )`, `Ma = round(S_a/S_out * 2^16)`, `Mb = round(S_b/S_out * 2^16)`.
Decoder embedding: `a = emb8[tok]` (column `tok` of the LM-head matrix), `Ma = resmult[tok] = round(s_emb[tok]/S_x0 * 2^16)`,
`b = dpos8[pos]`.

## LayerNorm (N = 384)
```
sum    = Σ x                                    int32
mean_f = rsr(sum * 5592405, 23)                 (= sum * 2^8 / 384; 5592405 = round(2^24/3))  Q.8
xc     = (x << 8) - mean_f                      Q.8, |xc| < 2^24
V      = Σ xc^2 + eps_q                         int64; eps_q = round(1e-5 * 384 * (2^8/S_in)^2)
e      = (bitlen(V) - 29) // 2 ; m = V >> 2e   (m in [2^28, 2^30); e may be negative → left shift)
y0     = RSQRT[m >> 22]                          256-entry LUT, RSQRT[i] = round(2^30 / sqrt((i+0.5)*2^22)), i in 64..255
t      = m*y0*y0 ; d = rsr(3*2^60 - t, 30) ; y1 = rsr(y0*d, 31)        (one Newton step, y1 ≈ 2^30/sqrt(m))
u      = rsr(xc * y1, 14 + e)                   (= n_k/sqrt(384) * 2^16, |u| <= 2^16)
v      = rsr(u * G[k], 12)                      G[k] = round(sqrt(384) * gamma[k] * 2^F)
y16    = sat16( rsr(v + Bq[k], 4) )             Bq[k] = round(beta[k] * 2^(F+4))
```

## Attention (per head h; score `s = q8·k8`, |s| ≤ 2^20; key tiles of 64 in ascending order)
```
Mq[h] = round(S_q[h]*S_k[h]/8 * 2^8 * 2^sq[h]),  sq chosen so Mq in [2^30, 2^31)
per key tile t (keys j0..j0+63; valid iff j < n_keys and (causal) j <= pos):
  mt   = max over valid s[j]              (tile skipped for a query with no valid key)
  mnew = max(mx, mt)   (first tile: mnew = mt, no rescale)
  if mnew > mx:  da = min(4095, rsr((mnew - mx) * Mq, sq)); alpha = EXP(da); l = rsr(l*alpha, 15); O = rsr(O*alpha, 15)
  mx   = mnew
  d[j] = min(4095, rsr((mx - s[j]) * Mq, sq)) ; p[j] = EXP(d[j]) (0 if invalid)      p in [0, 32768] (Q1.15)
  l   += Σ p[j] ;  O[d] += Σ_j p[j] * v8[j][d]                                      (O: 36-bit signed)
after the last tile:  rl = floor((2^40 + l/2) / l) ;  out16[d] = sat16( rsr(O[d] * rl, 33) )   (= O/l * 2^7)
EXP(d): i = d >> 4, f = d & 15, T[i] = round(exp(-i/16) * 2^15) (i = 0..255), T[256] = 0,
        EXP = T[i] - rsr((T[i] - T[i+1]) * f, 4)
```
Hardware detail with no numeric effect: `P·V` runs on the int8 array as two unsigned passes
(`hi = p >> 8`, `lo = p & 255`), `O += (acc_hi << 8) + acc_lo` — identical integer result.

## GELU
* `phi16` (default): `xf = clamp(rsr(h16 * m_phi, s_phi), -2048, 2047)` (= x in Q.8, `m_phi = round(S_h*2^8*2^s_phi)`),
  `i = (xf >> 4) + 128`, `f = xf & 15`, `PHI[i] = round(Φ((i-128)/16) * 2^15)` (256 entries, `PHI[256] = 32768`),
  `Φq = PHI[i] + rsr((PHI[i+1] - PHI[i]) * f, 4)`, `g16 = sat16(rsr(h16 * Φq, 15))` (same scale as h16).
  conv1 output additionally requantised to int8: `g8 = sat8(rsr(g16 * req_mult, req_shift))`.
* `lut8` (prompt's original spec, kept as an option): `g8 = T[h8 + 128]`, `T[i+128] = sat8(round(gelu(i*S_h)/S_g))`.

## Decoding rules (replicating openai-whisper greedy, `without_timestamps=True`)
Prompt `<|sot|> <|lang|> <|transcribe|> <|notimestamps|>` at positions 0..3; sample from position 3.
Suppress: the 88 non-speech tokens + `<|sot|>`/task tokens (list in `data/ref/_meta.json`), padded
vocab columns ≥ 51865, and on the first sampled position also `" "` (220) and `<|eot|>`.
Argmax = lowest index among maxima. Stop at `<|eot|>` or after 224 generated tokens.

## Host-side front end
`mel = whisper.log_mel_spectrogram(pad_or_trim(audio))` (fp32, 30 s window, so the log-mel max
normalisation is identical to the reference), truncated to `n_frames` (see `n_frames_for` in
`whisper_int.py`: `max(ceil(samples/160) + pad, min) rounded up to a multiple of 128, ≤ 3000`), then
`mel8 = sat8(round(mel / s_mel))`.
