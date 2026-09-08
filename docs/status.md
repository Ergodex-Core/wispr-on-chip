# Status

Chip: `MiniCPMTop` (Chisel 7.15, package `minicpm`), model openbmb/MiniCPM5-2B (checkpoint sha256 in
`weights/MANIFEST.json`). Everything below was run on this branch on 2026-09-08 (4-core container,
Verilator 5.020). The chip runs on Verilator over real layers (§5); a 42-layer run is out of scope
(docs/decisions.md #8).

## 1. Calibration and quantisation

* `golden/calib.py`: fp32 reference (`golden/reference_cpu.py`, own torch Llama implementation reading the
  bf16 safetensors) over the 8 calibration prompts of `data/prompts.json` (4 raw texts, 4 chat prompts with
  24 greedy tokens each): 836 tokens, ~8 min. Output `weights/calib_stats.json` (committed).
* Residual maxima: 0.1 (layer 0 input, the embedding), 9.9 / 286 / 335 (layers 1–7), 4537–5333 from
  layer 8 on; layer 7 down_proj 4506, gated hidden 4206; layers 40/41 down_proj 1936 / 1534. Q/K/V heads
  ≤ 17, gate/up ≤ 70, o_proj ≤ 38 → the 32-bit residual path of decision #3.
* Frozen configuration (decision #13): W8A8, int32 residual, SmoothQuant α = 0.5 folded into the RMSNorm
  gains, static int16/int32 margin 1.5.
* `gen/dump_weights.py`: 2.52 G parameters → 2.52 GB of int8 images in 723 tensors (`weights/*.bin`,
  generated in ~2 min, not committed) + `MANIFEST.json` (403 KB, committed); `make weights-check` rebuilds
  in memory and verifies every sha256 (run: OK, 723 tensors, 2522 MB).
* `gen/emit_microcode.py`: 848 instructions for the full model (docs/microcode.txt) — 3 for the embedding
  loop, 20 per layer in a chunk loop, 4 for the final norm / LM head / sample / loop — plus the two small
  emissions the layer-level test runs. **Every emission is checked statically** against the activation
  banks, the KV cache and the three weight address spaces, for the worst-case prefill chunk and decode
  position, so the 42-layer program is validated even though it is not simulated.

## 2. Golden model accuracy

### 2.1 Configuration sweep (`make sweep`, 722 teacher-forced positions of four texts)

| configuration | top-1 agreement with fp32 | mean rank of the fp32 choice | max rank | requant saturation |
|---|---|---|---|---|
| no smoothing, margin 2.0 (first frozen set) | 93.35 % | 1.080 | 5 | 0 |
| smoothing α = 0.5, margin 2.0 | 93.35 % | 1.080 | 4 | 0 |
| no smoothing, margin 1.5 | 92.80 % | 1.091 | 6 | 0 |
| **smoothing α = 0.5, margin 1.5 (frozen)** | **94.60 %** | **1.058** | 4 | 0 |
| smoothing α = 0.5, margin 1.25 | 94.04 % | 1.066 | 4 | 0 |

Neither change helps alone; together they gain 1.25 points (decision #13). A first, narrower sweep over
184 positions had suggested +1.1 points from smoothing alone — that was noise, and the wider set corrected
it. Saturation is counted in the golden model (`fixedpoint.SAT`) and by the chip (register 10).

### 2.2 Eval prompts (`make eval`, `golden/eval_golden.py`)

_(pending: the verification run of this configuration is still in flight)_

## 3. Fixed-point op tests (`make golden-test`)

22 pytest cases in `golden/tests/test_ops.py`: rounding/saturation, int16 and int32 dynamic quantisation
(row-factor exponent), requant vs float for 8/16/32-bit outputs and the exact fold of the row-factor
exponent into the shift, exact matmul (torch `_int_mm` == float64, tile-order invariance, unsigned P·V
split), RMSNorm vs float incl. a massive-activation row, RoPE vs float, SiLU gate vs float,
sigmoid/exp/rsqrt tables, GQA attention vs float (6 shapes incl. causal), masked-key independence,
chunked prefill == incremental decode, dynamic and int32-output linears vs float, wide-mode argmax,
embedding and saturating adds. All pass (5 s).

## 4. RTL unit tests (`make test-rtl`), Verilator, bit-exact against golden vectors of the real model

Vectors: `tests/vectors/gen_all.py` — one golden prefill of a real 128-token prompt with dumps of layers
0, 20, 41 and the LM head (~1 min after the 1-min model build).

### MatmulEngineSpec — 22/22 bit-exact
| case | shape / mode | cycles | MAC utilisation |
|---|---|---|---|
| rand_1 … rand_10 (Sram store) | raw, int8/int16 static+dynamic, unsigned, wide, 2×32×32 … 256×256×256 | 16 … 25354 | up to 98.4 % (256×256×256) |
| rand_11 | 20×6144×64 int32 out, 32-bit rows (row-factor exponent b > 0) | 9226 | 83.2 % |
| rand_12 | 7×2048×32 int32 out, b = 0 | 714 | 62.7 % |
| real_L0_q, real_L0_o | 40×2048×2048 (int16 / int32 out), RomInit | 180234 | 90.9 % |
| real_L0_k, real_L20_k (int16), real_L0_v (int8) | 40×2048×256 | 22538 | 90.9 % |
| real_L0_gate, real_L0_up | 40×2048×6144 int16 | 540682 | 90.9 % |
| real_L0_down, real_L41_down | 40×6144×2048 int32 | 540682 | 90.9 % |
| real_lm00 | 1×2048×8192 wide (LM head slice 0, 16.7 MB ROM) | 81930 | 20.0 % (M = 1: weight-port bound) |

Per-job cost is `KT·NT·(M+4)+10` as in whisper-si (40 rows → 40/44 = 90.9 %).

### VectorUnitSpec — 14/14 bit-exact (outputs and row factors)
| case | op | rows × cols | cycles / row |
|---|---|---|---|
| rmsnorm_L0n1, rmsnorm_L41n2, rmsnorm_f_last | RMSNORM int32 → int8 + rowfac | 40 (1) × 2048 | 691 |
| dynq_attn_L0 | DYNQ int16 → int8 + rowfac | 40 × 2048 | 295 |
| silumul_L0, silumul_L41 (massive rows, b up to 14) | SILUMUL gate16·up16 → int8 + rowfac | 40 × 6144 | 807 |
| add1_L0, add2_L41 | ADD int32 residual | 40 × 2048 | 262 |
| embed_prompt (7 table slices), embed_special (ids 0, 1, 130072, 130073, 130559, …) | EMBED → int32 | 40 / 10 × 2048 | 139 |
| rope_q_L0, rope_q_L0_pos1000 | ROPE q16 → q8, positions 0.. / 1000.. | 40 × 2048 | 410 |
| rope_k_L20 | ROPE k16 → k8 to a bank | 40 × 256 | 74 |
| rope_k_L0_kv | ROPE k16 → 320 KV beats | 40 × 256 | 74 |

### AttentionSpec — 5/5 bit-exact
| case | queries × keys | cycles |
|---|---|---|
| L0_prefill128 (causal, qPos0 0) | 128 × 128 | 228960 |
| L0_chunk64 (causal, qPos0 64: second chunk of a prefill) | 64 × 128 | 119088 |
| L20_ragged100x70 (non-causal, partial tiles) | 100 × 70 | 188000 |
| L41_decode_pos40 (1 query, 41 keys) | 1 × 41 | 3392 |
| L41_decode_pos127 (1 query, 128 keys) | 1 × 128 | 6192 |

### KVWriteSpec — 1/1 bit-exact
Layer 0's real v_proj on 40 real rows through the engine (int8 beats, key-major) and the real rotated K
rows of layer 0 through the vector unit's RoPE beats (transposed by the cache); then activation row 5 as a
single key appended at slot 205 (decode-style, keyOff 200) on both paths: every written byte matches the
golden packed K^T / V layouts.

### SamplerSpec — 4/4
LM-head slice 0 (256 tiles, wide mode) streamed from the engine on the golden final-norm row picks the
golden argmax; three random int32 streams (50 / 17 / 300 tiles, planted ties) pick the lowest-index maximum,
and the stream whose maximum is token 1 (`</s>`) raises `isEos`.

### WeightStoreEquivalenceSpec — 16/16
RomLiteral, RomInit and Sram return identical data at every address for real tensors of every kind:
`L0.k.mult`, `L0.norm1.g`, `L20.gate.mult` (i32vec, all three backends), a 64 Kbit window of `L0.q.w`
(RomLiteral), all of `L0.k.w` (4 Mbit) and the RoPE table (i16mat, 4 Mbit) under RomInit, 256 sampled
rows of `embed.15` (last, partial embedding slice) and 512 sampled words of `L41.down.w` (100 Mbit),
`embed.resmult` (4 Mbit) under RomLiteral and RomInit, and the flat w8 address decode across layer 0.

### RomLiteralLayerSpec — 2/2
`real_L0_k` and `real_L0_v` (4 Mbit tensors each) re-run with literal `VecInit` ROMs: bit-exact.

## 5. The whole chip on Verilator (`make test-layer`, `LayerSpec`)

`MiniCPMTop` runs the generated micro-program end to end: prompt token ids in, embedding from the on-chip
token buffer, real transformer layers over a chunked prefill, final norm, LM head, sampler, the sampled
token written back into the token buffer, decode steps, tokens out. Two emissions of the *same* generated
program (decision #12); the golden mirror `tests/vectors/gen_layer.py` executes the identical sequence.

| run | program | what it adds | cycles (rate, engine busy) | tokens | residual bank |
|---|---|---|---|---|---|
| A | 48 instructions: 2 layers, maxCtx 64, chunk 8; 12-token prompt (chunks of 8 + 4), 2 decode steps | layer indexing (two KV regions), multi-chunk prefill, the token-buffer feedback path | 3,166,208 (38.9 k cycles/s, 96.0 % busy, 3 requant saturations) | 1313, 7360 — identical | all 14 rows identical |
| B | 28 instructions: 1 layer, maxCtx 2048, chunk 32; 70-token prompt (chunks 32 + 32 + 6), 1 decode step | full-scale addressing (16 MB residual bank, 20-bit word addresses), attention over two key tiles in situ | 4,534,272 (46.4 k cycles/s, 93.0 % busy, 0 saturations) | 52 — identical | all 71 rows identical |

Elaboration + Verilator build is ~175 s per configuration; the two runs load 94 MB and 47 MB of int8
weight images. Run A's three requant saturations are reproduced exactly by the golden model (the residual
is bit-identical), which is the tighter margin of decision #13 being exercised on real data.

What this covers that the unit specs cannot: the sequencer's control flow and chunk loop, the runtime
substitution of rows / row offsets / key counts / positions, the token-buffer feedback path, engine
arbitration between the sequencer and the attention unit, activation-bank read/write muxing and port
conflicts, the KV cache filled by both producers in situ and read back by attention across chunks, and
every RTL assertion (irrevocable handshakes, no sink stalls, address ranges, token ids in range).

## 6. Bugs the ladder caught (each fixed, each now covered)
* Vector unit: the RMSNorm bit-length register of an 80-bit sum needs 8 signed bits (7 wrapped 80 to −48).
* Attention: on a key tile with no valid key for a query (the first query block of a causal prefill against
  the second key tile) the skip path left the previous tile's probabilities in the P buffer, so the P·V pass
  added stale products. Whisper-si never hit this (its encoder is non-causal, its decoder has one query).
  Now the skipped query's two P words are zeroed. Found by `L0_prefill128`, invisible to `L0_chunk64`.
* KV cache: the base transposer assumes 32 consecutive keys of one d-tile (engine order); the RoPE op emits
  a row's eight d-tiles back to back. The transposer now holds all eight d-tile slots of a 32-key group per
  buffer and flushes the used slots. Found by `KVWriteSpec` (assertion "new key group while flushing").
* Vector unit: KV beats were tagged with the chunk-local row, so a single key written at an offset landed
  in the wrong slot. Found by `KVWriteSpec`'s single-key case.
* Golden: int32 requant with a fixed 8-bit fraction lost ~7 bits for large row factors (decision #4); the
  numpy requant overflowed int64 for extreme row factors (now the exponent is folded into the shift).
* Test harness: embedding cases must instantiate every table slice their tokens fall in; the `$readmemh`
  cache must be keyed on the weight digest, not the file size, or regenerated weights read a stale image.

## 7. Results table
| Level | What | Result |
|---|---|---|
| Ops | 22 fixed-point ops vs float | pass |
| Golden vs fp32 | 722 teacher-forced positions (sweep) | 94.60 % top-1, mean rank 1.058, no saturation |
| Golden vs fp32 | eval prompts, greedy generations | _(pending: the verification run of this configuration is still in flight)_ |
| Matmul engine | 12 random shapes + 10 real tensors (layers 0/20/41, LM head) | 22/22 bit-exact |
| Vector unit | 14 cases, every op, layers 0/20/41, final norm | 14/14 bit-exact |
| Attention | 5 cases (prefill, chunk, ragged, decode ×2) | 5/5 bit-exact |
| KV cache write path | V via engine, K via RoPE beats, single key at a slot | bit-exact |
| Sampler | LM-head slice 0 from the engine + 3 random streams (ties, eos) | 4/4 |
| Weight backends | RomLiteral / RomInit / Sram on real tensors of every kind | 16/16 identical |
| Literal ROM path | k and v projections of layer 0 as VecInit ROMs | 2/2 bit-exact |
| Micro-program | 848 instructions checked against every memory they address | pass |
| **Whole chip on Verilator** | **the generated program over real layers, prefill + sampling + decode** | ****2/2 bit-exact: every residual row and every emitted token (§5)**** |
| Whole chip | elaboration of the 42-layer configuration to SystemVerilog | 82 modules, 9.9 MB of SystemVerilog in 39 s (Sequencer with the 848-instruction program: 5.2 MB) |

Clean run of this configuration after the requantisation: `make golden-test` 22/22, `make test-rtl` 64/64 in 36 min, `make test-layer` 2/2 in 9 min, `make elab` in 47 s, `make weights-check` OK.

## 8. Analytic cycle model (`tests/cycle_model.py`)
Built only from the unit measurements above (engine `KT·NT·(M+4)+10`, the per-row vector costs, an
attention formula), then **validated against the measured full-chip runs** (`--validate`):

| run | predicted | measured | ratio |
|---|---|---|---|
| 2 layers, ctx 64, chunk 8, 12 prompt tokens, 2 decode steps | 3,144,081 | 3,166,208 | 0.993 |
| 1 layer, ctx 2048, chunk 32, 70 prompt tokens, 1 decode step | 4,491,979 | 4,534,272 | 0.991 |

Nothing was fitted to these runs: every formula comes from the unit specs.

At full scale it predicts 282 M cycles for a 128-token prefill (gate/up 48 %, down 24 %, q/k/v/o 18 %,
attention 2 %) and 11.4 M cycles per decode step (gate/up 45 %, down 23 %, LM head 11.5 %), i.e. the
2048-bit weight port bound of 4 cycles per 32×32 tile (21.6 % MAC utilisation at M = 1). At 1 GHz that is
282 ms prefill and 11.4 ms per generated token.

## 9. Not done / known limitations
* No 42-layer chip simulation (decision #8); the full program is statically checked and elaborated, and
  the same generated code is simulated with 1 and 2 layers.
* The weight images (2.5 GB) are regenerated, not committed; only their manifest is.
* Accuracy is 94.6 % top-1 agreement with fp32. The remaining gap is inherent to per-token W8A8 without
  per-channel activation scaling; the next levers are a finer V/attention path and per-channel smoothing
  of the gated hidden into down_proj (the vector unit would need a per-channel multiplier).
* Calibration used 836 tokens of hand-written prompts.
* No synthesis or timing.
