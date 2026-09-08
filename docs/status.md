# Status

Chip: `MiniCPMTop` (Chisel 7.15, package `minicpm`), model openbmb/MiniCPM5-2B (checkpoint sha256 in
`weights/MANIFEST.json`). Everything below was run on this branch on 2026-09-08 (4-core container, Verilator
5.020). The full-chip simulation was deliberately not run (docs/decisions.md #8).

## 1. Calibration and quantisation

* `golden/calib.py`: fp32 reference (`golden/reference_cpu.py`, own torch Llama implementation reading the
  bf16 safetensors) over the 8 calibration prompts of `data/prompts.json` (4 raw texts, 4 chat prompts with
  24 greedy tokens each): 836 tokens, ~8 min. Output `weights/calib_stats.json` (committed).
* Residual maxima: 0.1 (layer 0 input, the embedding), 9.9 / 286 / 335 (layers 1–7), 4537–5333 from
  layer 8 on; layer 7 down_proj 4506, gated hidden 4206; layers 40/41 down_proj 1936 / 1534. Q/K/V heads
  ≤ 17, gate/up ≤ 70, o_proj ≤ 38 → the 32-bit residual path of decision #3.
* `gen/dump_weights.py`: 2.52 G parameters → 2.52 GB of int8 images in 723 tensors (`weights/*.bin`,
  generated in ~5 min, not committed) + `MANIFEST.json` (403 KB, committed); `make weights-check` (run: OK, 723 tensors, 2522 MB) rebuilds
  in memory and verifies every sha256.
* `gen/emit_microcode.py`: 848 instructions (docs/microcode.txt): 3 for the embedding loop, 20 per layer
  in a chunk loop, 4 for the final norm / LM head / sample / loop.

## 2. Golden model accuracy (`make eval`, `golden/eval_golden.py`)

Integer golden vs fp32 reference, eval prompts of `data/prompts.json`, greedy, thinking disabled.

| prompt | fp32 | int golden | tokens identical |
|---|---|---|---|
| capital (chat) | `Paris` | `Paris` | all (1) |
| sum 17 + 26 (chat) | `43` | `43` | all (1) |
| haiku (chat, 12 tokens) | `Still water, / A breath of blue, / Peace in` | `Still water, / A mirror of the sky, / Peace` | first 5 |
| code_rev (chat, 12 tokens) | ```` ```python\ndef reverse_string(s):\n    """\n    Returns ```` | ```` ```python\ndef reverse_string(s):\n    """Return the ```` | first 9 |
| zh_greet (chat, 12 tokens) | `你好，我是MiniCPM系列模型，由面壁智能和` | `我是 MiniCPM 系列模型，由面壁智能（` | 0 (same content, different opening) |
| prose_eval (text, 89 positions) | teacher-forced top-1 agreement 79/89 = 88.8 %, fp32 argmax has int rank ≤ 2 everywhere (mean 1.11) | | |
| code_eval (text, 95 positions) | top-1 agreement 93/95 = 97.9 %, mean rank 1.02, max 2 | | |

Overall next-token top-1 agreement 172/184 = 93.5 %; wherever they disagree the fp32 choice is the int
model's second candidate. Greedy generations stay on-topic and grammatical and diverge only where two
continuations are near-equal (W8A8 with per-token int8 activations; no SmoothQuant-style smoothing yet).

## 3. Fixed-point op tests (`make golden-test`)

22 pytest cases in `golden/tests/test_ops.py`: rounding/saturation, int16 and int32 dynamic quantisation
(row-factor exponent), requant vs float for 8/16/32-bit outputs and the exact 72-bit fold
(`test_requant_int32_folds_rowfac_exponent`), exact matmul (torch `_int_mm` == float64, tile-order
invariance, unsigned P·V split), RMSNorm vs float incl. a massive-activation row, RoPE vs float, SiLU gate
vs float, sigmoid/exp/rsqrt tables, GQA attention vs float (6 shapes incl. causal), masked-key independence,
chunked prefill == incremental decode, dynamic and int32-output linears vs float, wide-mode argmax,
embedding and saturating adds. All pass (4 s).

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
`real_L0_k` and `real_L0_v` (4 Mbit tensors each) re-run with literal `VecInit` ROMs: bit-exact, 65–74 s
elaboration + Verilator build each.

### TopElabSpec — pass
`MiniCPMTop` with layer 0's weights, `norm_f`, `embed.00`, `embed.resmult`, the RoPE table and `lm.00`
instantiated (the rest of the weight space decodes to zero), 42-layer KV cache, 8 banks, the
848-instruction sequencer: 82 SystemVerilog modules, 9.9 MB (Sequencer 5.2 MB, KVCache 2.0 MB, Attention
1.1 MB, VectorUnit 1.0 MB), elaborated by firtool in 38 s.

## 5. Bugs the ladder caught (each fixed, each now covered)
* Vector unit: the RMSNorm bit-length of an 80-bit sum needs 8 signed bits (7 wrapped 80 to −48).
* Attention: on a key tile with no valid key for a query (the first query block of a causal prefill against
  the second key tile) the skip path left the previous tile's probabilities in the P buffer, so the P·V pass
  added stale products. Whisper-si never hit this (its encoder is non-causal, its decoder has one query).
  Now the skipped query's two P words are zeroed. Found by `L0_prefill128`, invisible to `L0_chunk64`.
* KV cache: the base transposer assumes 32 consecutive keys of one d-tile (engine order); the RoPE op emits
  a row's eight d-tiles back to back. The transposer now holds all eight d-tile slots of a 32-key group per
  buffer and flushes the used slots. Found by `KVWriteSpec` (assertion "new key group while flushing").
* Golden: int32 requant with a fixed 8-bit fraction lost ~7 bits for large row factors (decision #4); the
  numpy requant overflowed int64 for extreme row factors (now folded exactly).
* Test harness: embedding cases must instantiate every table slice their tokens fall in.

## 6. Results table
| Level | What | Result |
|---|---|---|
| Ops | 22 fixed-point ops vs float | pass |
| Golden vs fp32 | 184 teacher-forced positions, 5 greedy chats | 93.5 % top-1, 2/5 chats identical, rest diverge after 5–9 tokens |
| Matmul engine | 12 random shapes + 10 real tensors (layers 0/20/41, LM head) | 22/22 bit-exact |
| Vector unit | 14 cases, every op, layers 0/20/41, final norm | 14/14 bit-exact |
| Attention | 5 cases (prefill, chunk, ragged, decode ×2) | 5/5 bit-exact |
| KV cache write path | V via engine, K via RoPE beats, single key at a slot | bit-exact |
| Sampler | LM-head slice 0 from the engine + 3 random streams (ties, eos) | 4/4 |
| Weight backends | RomLiteral / RomInit / Sram on real tensors of every kind | 16/16 identical |
| Literal ROM path | k and v projections of layer 0 as VecInit ROMs | 2/2 bit-exact |
| Whole chip | elaboration to SystemVerilog with layer 0's weights | 82 modules, 9.9 MB, 38 s |

Final clean run of `make test-rtl` + `make elab` after the last RTL fix: `make test-rtl` 64/64 tests pass in 29 min (7 suites: WeightStore 16, Matmul 22, Vector 14, Attention 5, KVWrite 1, Sampler 4, RomLiteral 2); `make elab` 82 modules, 9.9 MB of SystemVerilog in 38 s (Sequencer with the 848-instruction program 5.2 MB).

## 7. Analytic cycle model (`tests/cycle_model.py`)
From the unit measurements above: prefill of 128 tokens 282 M cycles (gate/up 48 %, down 24 %, q/k/v/o
18 %, attention 2 %); one decode step 11.4 M cycles (gate/up 45 %, down 23 %, LM head 11.5 %), i.e. the
2048-bit weight port bound of 4 cycles per 32×32 tile (21.6 % MAC utilisation at M = 1). At 1 GHz that is
282 ms prefill and 11.4 ms per generated token.

## 8. Not done / known limitations
* No full-chip Verilator run (by request); the chip is elaborated, not simulated end to end.
* The weight images (2.5 GB) are regenerated, not committed; only their manifest is.
* Accuracy: 93.5 % top-1 agreement is W8A8 without smoothing; SmoothQuant-style per-channel folding into
  the RMSNorm gains (as in whisper-si's `smooth_alpha`) is the obvious next step.
* No synthesis or timing.
