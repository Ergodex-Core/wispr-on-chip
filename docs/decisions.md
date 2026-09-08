# Design decisions

Numbered, dated, one paragraph each. Anything that deviates from the whisper-si base design
(`claude/whisper-tiny-chisel-accelerator-03do1o`) or that the task left open is recorded here.

1. **2026-09-08 — Toolchain and repository layout.** Scala 2.13.18, Chisel 7.15.0, sbt 1.10.7 (launcher jar
   from Maven Central, repositories restricted to Maven Central because the remote proxy rate-limits the
   default multi-repository resolution), Verilator 5.020 (apt), Python 3.11 via uv with torch 2.14 CPU
   wheels, safetensors, tokenizers. Same repository skeleton as whisper-si (golden/, gen/, src/, tests/,
   docs/), Scala package `minicpm`. The checkpoint (bf16 safetensors, 5.03 GB) lives in
   `$MINICPM_SI_DATA/minicpm5-2b`, not in the repository.

2. **2026-09-08 — Weight images are binary and not committed.** MiniCPM5-2B is 2.52 G parameters; the int8
   images are 2.5 GB (whisper-si committed 86.6 MB of hex). `gen/dump_weights.py` writes one little-endian
   `.bin` per tensor (the RomInit backend converts only the instantiated tensors to `$readmemh` text at
   elaboration), `weights/MANIFEST.json` with a sha256 per file *is* committed, and `make weights-check`
   regenerates everything in memory and proves it identical. The LM head and the embedding table are stored
   as 16 slices each (docs/tiling.md) so unit tests can instantiate one 16.7 MB slice.

3. **2026-09-08 — Residual stream, o_proj/down_proj outputs and the gated hidden product are 32-bit.**
   Calibration (836 tokens, 8 prompts) shows Llama-style massive activations: the residual reaches 5330 from
   layer 8, layer 7's down_proj emits 4506 and its gated hidden 4206, layers 40/41 emit 1936/1534, while
   typical elements are of order 1. With whisper-si's static int16 residual, typical values would be 0–3
   LSBs and layer 7's MLP output would vanish for ordinary tokens. The residual bank therefore holds int32
   rows (physical bank word widened to 1024 bits so the engine writes an int32 tile per cycle), o/down are
   requantised to int32, RMSNorm reads int32 (80-bit sum of squares), and the SiLU gate keeps its exact
   32-bit product and quantises it per token. gate/up/Q/K stay int16, V int8 — their maxima are ≤ 70.

4. **2026-09-08 — Requant keeps 24 fraction bits for int32 outputs by folding the row factor's exponent
   into the final shift.** The per-token factor of a 32-bit row is `m16 << b` (b ≤ 15). With whisper-si's
   `y = rsr(t·rowfac, s2)` and a 40-bit `t`, an 8-bit `s2` would be forced (overflow otherwise) and `t` would
   carry only ~7 significant bits for typical rows of layer 7. Instead `t` is 48 bits, `u = t·m16` (< 2^64)
   and `y = rsr(u, s2 − b)`, which is exactly `rsr(t·(m16<<b), s2)` and identical to the whisper rule when
   b = 0. `golden/tests/test_ops.py::test_requant_int32_folds_rowfac_exponent` pins this against Python ints.

5. **2026-09-08 — RoPE is applied by the vector unit on int16 projections.** Q and K come out of the engine as
   int16 with a static per-head scale, are rotated with Q15 cos/sin tables (4 Mbit ROM for 2048 positions,
   HF's fp32 angle computation reproduced in `luts.rope_table`) and requantised to int8 with the constant
   ratio 254/32767 (both scales derive from the same per-head maximum over pre- and post-rotation values).
   Rotating int8 values would double-quantise. K reaches the KV cache as beats from the vector unit through
   the same transposer the engine's V beats do not need (V is key-major).

6. **2026-09-08 — GQA in the attention unit.** 16 query heads loop over the KV region of head h/8; head_dim
   128 makes Q·Kᵀ a 4-k-tile job and P·V a 4-n-tile job, the O accumulator is 4 × 64 rows and the final
   pass writes 8 words per query. The online softmax, the exp table and the 41-bit reciprocal are unchanged
   from whisper-si.

7. **2026-09-08 — One micro-program for prefill and decoding.** Every layer is a row-chunk loop (chunk =
   512 rows); `CHUNK_BEGIN` sizes it from the phase: prefill = all prompt rows from position 0, decoding =
   one row at `pos`. Chunk-local operands use `rowOff = 0`, the residual (all positions) uses
   `rowOff = chunkBase`, attention uses `qPos0 = chunkBase`, `nKeys = chunkBase + chunkRows`. Chunked causal
   prefill is bit-identical to one-row decoding (test_ops), so the golden model runs prefill as one batch.
   848 instructions for 42 layers (docs/microcode.txt); the sampled token is written back into the on-chip
   token buffer that the EMBED op reads.

8. **2026-09-08 — The chip runs on Verilator at layer granularity, not at 42 layers.** Correctness is
   established per unit (every op of the datapath on golden activations of real layers 0, 20, 41 and the
   LM head, plus random shapes) *and* on the whole chip running the generated micro-program over real
   layers (decision #12). A 42-layer run is what is skipped: it would need the full 2.5 GB of ROM images
   (≈5 GB of `$readmemh` text) and ≈11 M cycles per generated token, and it would add no coverage that the
   layer-level runs do not already give — the program is the same generated code with a different layer
   count, and `gen/emit_microcode.py` checks the 42-layer emission statically against every memory it
   addresses.

9. **2026-09-08 — Accuracy gate.** The integer golden model is compared with the fp32 reference on
   data/prompts.json: teacher-forced next-token top-1 agreement over the eval texts and greedy
   generations on the chat prompts (`golden/eval_golden.py`, results in docs/status.md). Calibration prompts
   are hand-written texts committed in the repository (no external datasets), 836 tokens.

10. **2026-09-08 — SyncReadMem timing and masked memories** follow whisper-si decisions #11 and #13: read
    data is valid exactly one cycle after an enabled read (the RomLiteral backend mirrors it with
    `RegEnable`), memory randomisation is disabled in simulation, and every write port of the KV cache
    carries a data-dependent byte mask (the test-only load port takes its mask from IO).

12. **2026-09-08 — The micro-program is parameterised, statically checked, and run on the whole chip.**
    `gen/emit_microcode.py` takes `--layers / --max-ctx / --chunk / --vocab-tiles` and emits a
    `MicroProgram` object carrying the parameters it was emitted for; `MiniCPMTop` refuses a configuration
    that disagrees. Every emission is checked instruction by instruction against the activation banks, the
    KV cache and the weight address spaces for the worst-case prefill chunk and decode position
    (`check_program`), so the 42-layer program is validated even though it is not simulated. `LayerSpec`
    then runs two emissions of the same generated code on `MiniCPMTop` under Verilator: (a) 2 layers,
    maxCtx 64, chunk 8, a 12-token prompt and 2 decode steps — layer indexing, multi-chunk prefill, the
    token-buffer feedback path; (b) 1 layer, maxCtx 2048, chunk 32, a 70-token prompt — full-scale
    addressing (16 MB residual bank) and attention over two key tiles in situ. `tests/vectors/gen_layer.py`
    mirrors exactly that sequence in the golden model, and the residual bank and the emitted token ids must
    match bit for bit. The LM head is restricted to the first vocabulary slice (8192 columns) so the test
    loads 81 MB of weight images instead of 2.5 GB; sampled ids therefore stay inside the instantiated
    embedding slice. `$readmemh` images are cached under `target/readmemh/<name>.<sha8>.mem`, so
    regenerated weights can never be read from a stale image.

13. **2026-09-08 — Quantisation: SmoothQuant α = 0.5 into the RMSNorm gains and a static margin of 1.5.**
    Measured as teacher-forced top-1 agreement with the fp32 reference over 722 positions of four texts
    (`make sweep`, `golden/sweep_quant.py`):

    | configuration | top-1 | mean rank of the fp32 choice | requant saturation |
    |---|---|---|---|
    | no smoothing, margin 2.0 (the first frozen set) | 93.35 % | 1.080 | 0 |
    | smoothing α = 0.5, margin 2.0 | 93.35 % | 1.080 | 0 |
    | no smoothing, margin 1.5 | 92.80 % | 1.091 | 0 |
    | **smoothing α = 0.5, margin 1.5** | **94.60 %** | **1.058** | 0 |
    | smoothing α = 0.5, margin 1.25 | 94.04 % | 1.066 | 0 |

    Neither change helps on its own — smoothing alone is exactly neutral, and tightening the margin alone
    makes things worse — but together they gain 1.25 points and the best tail (the fp32 token is never
    below rank 4). That is the expected interaction: the margin only buys precision if the outlier channels
    it has to cover have been flattened first. No requant or residual-add saturation was observed at
    margin 1.5 on any of the 722 positions, and the chip counts saturations at run time in register 10.

14. **2026-09-08 — Exact int8 matmuls in the golden model use `torch._int_mm`.** Exact int32 accumulation
    on CPU (14× faster than the float64 path for a 2 B model); the float64 path remains as fallback and both
    are checked equal (test_ops). Prefill of 128 tokens through all 42 layers takes ~1 min.
