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
iff it produces the same bits. As requested, the evaluation is per unit rather than a full-chip Verilator
run: every layer operation is simulated on golden activations of real layers (0, 20, 41) and the LM head
plus random shapes, and compared bit for bit; the whole chip is elaborated to SystemVerilog.

| Level | What was compared | Result |
|---|---|---|
| Ops | every fixed-point op vs float | 22 pytest cases pass |
| Golden model vs fp32 MiniCPM5-2B | 184 teacher-forced positions, 5 greedy chats | 93.5 % top-1 agreement; fp32 choice always rank ≤ 2; 2/5 chats token-identical |
| Matmul engine | 12 random shapes + 10 real tensors (q/k/v/o/gate/up/down of layers 0/20/41, LM head slice) | 22/22 bit-exact, 90.9 % utilisation at M = 40, 98.4 % at 256 rows |
| Vector unit | RMSNorm, dynamic quant, SiLU gate, residual add, embedding, RoPE (q, k, k→KV), 14 cases | 14/14 bit-exact incl. row factors |
| Attention (GQA, online softmax) | prefill 128, chunk at qPos0 64, ragged 100×70, decode at 40 and 127 | 5/5 bit-exact |
| KV cache write path | V from the engine, K from RoPE beats, single key at an offset slot | bit-exact |
| Sampler | LM-head slice from the engine, random streams with ties and eos | 4/4 |
| Weight backends | RomLiteral / RomInit / Sram on real tensors of all four kinds | 16/16 identical |
| Whole chip | MiniCPMTop with layer 0's weights → 82 SystemVerilog modules, 9.9 MB | elaborates in 38 s |

An analytic cycle model from the unit measurements gives 282 M cycles for a 128-token prefill and 11.4 M
cycles per generated token (weight-port bound: 2.5 GB of weights through a 256-byte port).

## 2. Method

The whisper-si method is kept: numerics are frozen in Python before any RTL exists; the golden model is
the vector generator; every deviation is a numbered decision (`docs/decisions.md`). Two things differ in
kind from the base. First, the weights do not fit in a repository: they are regenerated from the checkpoint
(`make weights`, ~5 min) and `make weights-check` proves the regenerated set identical to the committed
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

Memory: 8 activation banks (X int32 for 2048 positions = 16 MB, chunk-local banks 24 MB), a 42-layer KV
cache of 42 MB (int8, 2048 keys), 2.5 GB of weights. These are model-scale figures; unit tests instantiate
one layer's tensors and a 256-key cache.

## 5. Verification and what it found

The ladder is the base's: ops vs float (22 tests); golden dumps of a real 128-token prompt → engine,
vector-unit, attention, KV and sampler vectors (`tests/vectors/gen_all.py`); one spec per unit
(`make test-rtl`, 64 tests, 29 min); the literal-ROM path on two real tensors; elaboration of the whole chip.
Every RTL run carries the base's assertions (irrevocable handshakes, no sink stalls, address ranges, bank
port conflicts, token ids in range).

Bugs the ladder caught in this design (all fixed, all covered): the RMSNorm bit-length register wrapped for
80-bit sums; the attention skip path for a key tile with no valid key left stale probabilities in the P
buffer, corrupting the first query block of every causal prefill (invisible to single-query decode and to
chunks whose tiles always have a valid key — found by `L0_prefill128`, not by `L0_chunk64`); the KV
transposer assumed the engine's d-tile-major beat order; the vector unit tagged KV beats with the local
row; the int32 requant lost precision with a fixed 8-bit fraction. Details in `docs/status.md` §5.

## 6. Performance (`tests/cycle_model.py`)

Engine jobs cost `KT·NT·(M+4)+10` cycles; vector ops 74–807 cycles per row (RMSNorm 691, SiLU gate 807
over 6144, RoPE-q 410); attention 229 k cycles for 128×128 with 16 heads. A 128-token prefill is 282 M
cycles (gate/up 48 %, down 24 %, q/k/v/o 18 %, attention 2 %). A decode step is 11.4 M cycles of which
gate/up 45 %, down 23 %, LM head 11.5 %: at M = 1 the array is bound by the 2048-bit weight port (one
32×32 tile per 4 cycles, 21.6 % MAC utilisation), exactly as in whisper-si's decoder. At 1 GHz: 282 ms
prefill, 11.4 ms per token. The obvious lever is the one the base already named — a wider weight port or a
second engine — since the vector unit and attention are under 5 % of a decode step.

## 7. Deviations and limitations

1. No full-chip simulation (by request; decision #8). The chip is elaborated and its program checked
   against the units' command bundles, but the sequencer's control flow has not been simulated.
2. Weights are generated, not committed (decision #2).
3. 32-bit residual path (decision #3) instead of the base's int16.
4. Accuracy is 93.5 % top-1 agreement with fp32 on the eval texts, with the fp32 token always within the
   int model's top 2; two of five greedy generations are identical, the others diverge after 5–9 tokens
   into equally plausible continuations. No SmoothQuant folding was attempted; it is the next step.
5. Calibration used 836 tokens of hand-written prompts.
6. No synthesis or timing.

## 8. Reproducing

```
make setup && make model      # uv env; 5 GB checkpoint into $MINICPM_SI_DATA
make calib                    # weights/calib_stats.json (committed; ~8 min)
make weights                  # 2.5 GB of images + MANIFEST.json + generated Scala (~5 min)
make weights-check            # regenerate in memory, verify sha256s
make golden-test              # 22 op tests
make vectors                  # golden prefill -> out/vectors (~2 min)
make test-rtl                 # 7 Verilator suites, 64 tests (~30 min)
make elab                     # whole chip -> target/top-sv
make eval                     # golden vs fp32 (~25 min)
uv run python tests/cycle_model.py
```
