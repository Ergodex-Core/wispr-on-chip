# minicpm-si: MiniCPM5-2B as a Chisel accelerator generator

Technical report, 2026-09-08. Repository `tibrewalrachit/wispr-on-chip`, branch
`claude/minicpm5-2b-chip-eval-mky3t0`. Base design: whisper-si (`claude/whisper-tiny-chisel-accelerator-03do1o`).

## 1. Summary

minicpm-si is a hardware implementation of openbmb/MiniCPM5-2B — a 42-layer Llama with d = 2048, 16 query
heads and 2 KV heads of 128, a 6144-wide SiLU-gated FFN and a 130560-token vocabulary (2.52 G parameters) —
written as a Chisel generator in the structure of the whisper-si base: one 32×32 int8 weight-stationary
matmul engine, a 16-lane vector unit, an attention unit with online softmax, a KV cache, a streaming argmax
sampler, and a sequencer executing an 848-instruction micro-program generated from the model graph. The
2.5 GB of int8 weight images are consumed at elaboration through the base's three interchangeable weight
backends; they are generated, not committed (their manifest with sha256s is).

Correctness is defined by an integer golden model in numpy (`golden/minicpm_int.py`): the RTL is correct
iff it produces the same bits. Every unit is verified against it on real activations of real layers, and
the **whole chip is run on Verilator** over real transformer layers — prompt tokens in, tokens out — with
the residual bank and the token stream compared bit for bit. What is out of scope is a 42-layer run
(decision #8): the same generated program is simulated with 1 and 2 layers, and the 42-layer emission is
checked statically against every memory it addresses.

| Level | What was compared | Result |
|---|---|---|
| Ops | every fixed-point op vs float | 22 pytest cases pass |
| Golden model vs fp32 MiniCPM5-2B | 722 teacher-forced positions | 94.60 % top-1 agreement, fp32 choice at mean rank 1.058 |
| Matmul engine | 12 random shapes + 10 real tensors (q/k/v/o/gate/up/down of layers 0/20/41, LM head slice) | 22/22 bit-exact, 90.9 % utilisation at M = 40, 98.4 % at 256 rows |
| Vector unit | RMSNorm, dynamic quant, SiLU gate, residual add, embedding, RoPE (q, k, k→KV), 14 cases | 14/14 bit-exact incl. row factors |
| Attention (GQA, online softmax) | prefill 128, chunk at qPos0 64, ragged 100×70, decode at 40 and 127 | 5/5 bit-exact |
| KV cache write path | V from the engine, K from RoPE beats, single key at an offset slot | bit-exact |
| Sampler | LM-head slice from the engine, random streams with ties and eos | 4/4 |
| Weight backends | RomLiteral / RomInit / Sram on real tensors of all four kinds | 16/16 identical |
| Micro-program | every instruction of the 42-layer program vs the memories it addresses | pass |
| **Whole chip on Verilator** | **the generated program over real layers: prefill, sampling, decode** | **both emissions bit-exact: residual bank and token stream** |
| Whole chip | elaboration of the 42-layer configuration to SystemVerilog | 82 modules, 9.9 MB, 39 s |

An analytic cycle model built only from unit measurements predicts the measured full-chip runs within
0.9 %, and gives 282 M cycles for a 128-token prefill and 11.4 M cycles per generated
token (weight-port bound: 2.5 GB of weights through a 256-byte port).

## 2. Method

The whisper-si method is kept: numerics are frozen in Python before any RTL exists; the golden model is
the vector generator; every deviation is a numbered decision (`docs/decisions.md`). Two things differ in
kind from the base. First, the weights do not fit in a repository: they are regenerated from the checkpoint
(`make weights`, ~2 min) and `make weights-check` proves the regenerated set identical to the committed
manifest. Second, accuracy is measured against the model's own fp32 forward (`golden/reference_cpu.py`,
a 150-line torch Llama reading the bf16 safetensors) on committed hand-written prompts (`data/prompts.json`)
by teacher-forced next-token agreement and greedy-generation comparison, since there is no WER equivalent.

## 3. Numerics: what a Llama changes

`docs/numerics.md` is the normative text. Relative to Whisper-tiny:

* **Massive activations force a 32-bit residual.** Calibration (836 tokens) shows the residual at 5330
  from layer 8 on, produced by layer 7's MLP (down_proj 4506, gated hidden 4206), while typical elements
  are of order 1. whisper-si's static int16 residual (its decision #5) would leave typical values at 0–3
  LSBs and delete layer 7's MLP output for ordinary tokens. The residual, the o/down projection outputs
  and the SiLU-gate product are therefore 32-bit; the bank word is 1024 bits so an int32 tile is written in
  one cycle; RMSNorm sums 80 bits of squares. Everything else (Q/K int16 → int8, V int8, gate/up int16) keeps
  the base's formats because their maxima are ≤ 70.
* **Dynamic quantisation of 32-bit rows.** The per-token factor becomes `m16 << b`; the engine folds `b`
  into the final shift (`y = rsr(t·m16, s2 − b)`) so `t` keeps 24 fraction bits at 48 bits width and the
  product stays below 2^64 — identical to the whisper rule when b = 0.
* **RMSNorm** is the base's LayerNorm without the mean (same rsqrt table and Newton step, gain folded into
  the output scale). **RoPE** rotates int16 projections with Q15 cos/sin tables and requantises to int8 with a
  constant per-head ratio; **SiLU** is a 256-entry sigmoid table on the int16 gate. **GQA**: head h reads the KV
  region of head h/8; head_dim 128 makes Q·Kᵀ a 4-k-tile job and P·V a 4-n-tile job.
* **Embedding**: int8 per token with a per-token int32 multiplier into the int32 residual (no positional
  table; positions enter through RoPE).
* **Quantisation choices are measured, not assumed** (decision #13): SmoothQuant α = 0.5 folded into the
  RMSNorm gains plus a static margin of 1.5 gains 1.25 points of top-1 agreement over the first frozen
  set, while either change alone is neutral or harmful. Both are weight-generation-time transformations:
  the datapath does not know about them.

## 4. Architecture

The units are the base's with the changes above. The vector unit gains four ops (RMSNORM, ROPE, SILUMUL,
EMBED with a token-buffer read port) and a KV-beat output; the attention unit loops 16 heads over 2 KV
regions with a 4-tile O accumulator; the KV cache transposer holds all eight d-tile slots of a 32-key group so
K beats may arrive row-major from the vector unit; the engine has an int32 output mode and 24-bit row
factors. The sequencer runs one program for prefill and decoding: every layer is a row-chunk loop
(chunk 512) whose bounds come from the phase (all prompt rows from 0, or one row at `pos`), chunk-local
operands use `rowOff = 0`, the residual uses the chunk base, attention uses `qPos0 = chunkBase` and
`nKeys = chunkBase + chunkRows`. The sampled token is written into the on-chip token buffer that the EMBED
op reads, so the host only streams the prompt and reads tokens back. Chunked causal prefill is bit-identical
to one-row decoding (test_ops), which is why the golden model can run the prompt as one batch.

The micro-program is a generator parameter, not a constant: `gen/emit_microcode.py --layers --max-ctx
--chunk --vocab-tiles` emits a `MicroProgram` object carrying the parameters it was emitted for, and
`MiniCPMTop` refuses a configuration that disagrees with it. That is what makes a real chip-level
simulation affordable (§5) while the shipped 42-layer program stays the same generated code.

Memory: 8 activation banks (X int32 for 2048 positions = 16 MB, chunk-local banks 24 MB), a 42-layer KV
cache of 42 MB (int8, 2048 keys), 2.5 GB of weights.

## 5. Verification and what it found

The ladder is the base's, plus a rung the base did not have:

1. Ops vs float (22 tests).
2. Golden dumps of a real 128-token prompt → engine, vector-unit, attention, KV and sampler vectors
   (`tests/vectors/gen_all.py`); one spec per unit (`make test-rtl`, 64 tests, ~30 min); the literal-ROM
   path on two real tensors.
3. **Static checking of the micro-program**: every instruction of every emission is checked against the
   activation banks, the KV cache and the weight address spaces for the worst-case prefill chunk and
   decode position. The 42-layer program is validated this way without being simulated.
4. **The whole chip on Verilator** (`make test-layer`): `MiniCPMTop` runs the generated program over real
   layers, and both the residual bank and the emitted token ids must equal the golden mirror's.
   Two emissions were run: 2 layers / maxCtx 64 / chunk 8 with a 12-token prompt and 2 decode steps (3.17 M cycles), and 1 layer / maxCtx 2048 / chunk 32 with a 70-token prompt (4.53 M cycles, full-scale addressing and attention over two key tiles). Both matched the golden mirror exactly — every residual row and every emitted token.
5. Elaboration of the 42-layer configuration to SystemVerilog.

Every RTL run carries the base's assertions (irrevocable handshakes, no sink stalls, address ranges, bank
port conflicts, token ids in range).

Bugs the ladder caught in this design (all fixed, all covered): the RMSNorm bit-length register wrapped for
80-bit sums; the attention skip path for a key tile with no valid key left stale probabilities in the P
buffer, corrupting the first query block of every causal prefill (invisible to single-query decode and to
chunks whose tiles always have a valid key — found by `L0_prefill128`, not by `L0_chunk64`); the KV
transposer assumed the engine's d-tile-major beat order, which the vector unit's RoPE output violates; the
vector unit tagged KV beats with the chunk-local row, so a key written at an offset landed in the wrong
slot; the int32 requant lost precision with a fixed 8-bit fraction. Details in `docs/status.md` §6.

## 6. Performance (`tests/cycle_model.py`)

Engine jobs cost `KT·NT·(M+4)+10` cycles; vector ops 74–807 cycles per row (RMSNorm 691, SiLU gate 807
over 6144, RoPE-q 410); attention 229 k cycles for 128×128 with 16 heads. The model was built from those
unit measurements alone and then checked against the full-chip runs (`--validate`):
| run | predicted | measured | ratio |
|---|---|---|---|
| 2 layers, ctx 64, chunk 8, 12 prompt tokens, 2 decode steps | 3,144,081 | 3,166,208 | 0.993 |
| 1 layer, ctx 2048, chunk 32, 70 prompt tokens, 1 decode step | 4,491,979 | 4,534,272 | 0.991 |

Nothing was fitted to these runs: the formulas come from the unit specs alone.

A 128-token prefill is 282 M cycles (gate/up 48 %, down 24 %, q/k/v/o 18 %, attention 2 %). A decode step
is 11.4 M cycles of which gate/up 45 %, down 23 %, LM head 11.5 %: at M = 1 the array is bound by the
2048-bit weight port (one 32×32 tile per 4 cycles, 21.6 % MAC utilisation), exactly as in whisper-si's
decoder. At 1 GHz: 282 ms prefill, 11.4 ms per token. The obvious lever is the one the base already named —
a wider weight port or a second engine — since the vector unit and attention are under 5 % of a decode step.

## 7. Deviations and limitations

1. No 42-layer chip simulation (decision #8). The program is statically checked and elaborated, and the
   same generated code is simulated at 1 and 2 layers with real weights.
2. Weights are generated, not committed (decision #2).
3. 32-bit residual path (decision #3) instead of the base's int16.
4. Accuracy is 94.6 % top-1 agreement with fp32 over 722 positions, with the fp32 token at mean rank
   1.058. The remaining gap is inherent to per-token W8A8; per-channel smoothing of the gated hidden into
   down_proj would need a per-channel multiplier in the vector unit's SILUMUL op.
5. Calibration used 836 tokens of hand-written prompts.
6. No synthesis or timing.

## 8. Reproducing

```
make setup && make model      # uv env; 5 GB checkpoint into $MINICPM_SI_DATA
make calib                    # weights/calib_stats.json (committed; ~8 min)
make weights                  # 2.5 GB of images + MANIFEST.json + generated Scala (~2 min)
make weights-check            # regenerate in memory, verify sha256s
make golden-test              # 22 op tests
make vectors                  # golden prefill -> out/vectors, and the layer-test mirrors (~5 min)
make test-rtl                 # 7 Verilator unit suites, 64 tests (~30 min)
make test-layer               # the whole chip on Verilator over real layers (~15 min)
make elab                     # the 42-layer configuration -> target/top-sv
make sweep                    # quantisation configurations vs fp32 (722 positions)
make eval                     # golden vs fp32 on the eval prompts
uv run python tests/cycle_model.py --validate
```
