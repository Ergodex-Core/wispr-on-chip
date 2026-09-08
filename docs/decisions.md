# Design decisions

Numbered, dated, one paragraph each. Anything that deviates from the task prompt, or that the prompt
left open, is recorded here.

1. **2026-09-07 — Toolchain pins.** Scala 2.13.18, Chisel 7.15.0 (latest stable on Maven Central; the
   2.13.16 compiler is rejected by sbt because chisel 7.15 pulls scala-library 2.13.18), sbt 1.10.7,
   Verilator 5.020 (Ubuntu 24.04 apt package; ≥ 5.0 as required), firtool resolved automatically by
   chisel's firtool-resolver. Simulation through `chisel3.simulator` (svsim) — proven end to end in
   `SmokeSpec`, so no chiseltest fallback is needed. Python 3.11 via uv with torch 2.14 CPU wheels
   (PyTorch CPU index) to avoid CUDA downloads.

2. **2026-09-07 — Reference decoding = one 30 s window, greedy, whisper defaults.** The fp32 reference
   is `whisper.decode` on a single padded/trimmed 30 s log-mel window with `without_timestamps=True`,
   `temperature=0`, `fp16=False`, language supplied per clip. It uses whisper's default logit filters
   (SuppressBlank at the first sampled position, SuppressTokens with the model's non-speech list) and
   whisper's default `sample_len = n_text_ctx // 2 = 224` generated tokens. The golden model and RTL
   replicate exactly these rules (mask ROM for suppressed tokens, blank suppression on step 0, stop at
   `<|endoftext|>` or 224 generated tokens) so that token streams are comparable. This is a narrower
   stop rule than the prompt's "448 tokens"; positions still support the full 448 context.

3. **2026-09-07 — Non-English clips come from FLEURS (google/fleurs on the HF hub).** Common Voice
   requires gated access and MLS tarballs are multi-GB; FLEURS `de_de`/`fr_fr` test tarballs are
   reachable and small. One German and one French utterance (first by filename with 4–8 s duration)
   are committed under `data/varied/`. One FLEURS raw transcript had a stray trailing character which
   is corrected in `gen/make_varied.py` (`TEXT_FIX`).

4. **2026-09-07 — WER normalisation.** English WER uses whisper's `EnglishTextNormalizer` on both
   reference and hypothesis (the standard LibriSpeech reporting convention for Whisper); non-English
   uses `BasicTextNormalizer`. Deterministic references use single-threaded torch per worker process
   so `make refs-check` is byte-identical across runs.

5. **2026-09-07 — Residual stream is int16; residual-feeding matmul outputs are int16.** Calibration on
   128 dev-clean utterances shows encoder residual max-abs 274 (layer 3) with a channel median of 5.8,
   and decoder residual max 118: an int8 static residual would leave typical values 0–2 LSBs. The
   residual width is a `QConfig`/`WhisperConfig` parameter; the int8 variant is still evaluated by the
   golden model for the record (see status.md), but the frozen datapath uses int16 for the residual and
   for the out-proj / fc2 outputs that are added into it (`s2 = 20` requant path).

6. **2026-09-07 — GELU is computed as x·Φ(x) on int16 with a 256-entry Φ LUT (`phi16`), not as an
   int8→int8 table.** fc1 outputs reach max-abs 326 (enc.3) / 73 (dec.1) while their channel medians
   are 14 / 1: an int8 per-tensor static input to a 256-entry GELU table would give the nonlinearity
   2–4 levels for typical values. `phi16` keeps fc1's output in int16, interpolates Φ linearly from a
   256-entry Q15 table (max error 1.2e-4), then quantises the result per token dynamically for fc2.
   The prompt's `lut8` form remains implemented and selectable (`QConfig.gelu_mode = "lut8"`) and is
   reported in status.md. conv1's GELU output must be int8 static (im2col rows mix frames), so it uses
   `phi16` followed by a static int8 requant; conv2's GELU is fused into the pos-emb add.

7. **2026-09-07 — Q/K use per-head static output scales, V per-tensor; scores use a per-head
   multiplier.** Per-token dynamic scales on K or V cannot factor out of the softmax / P·V sums, so K
   and V are static. Q is static per head (its output is a matmul output). Attention output is int16
   `O/l·2^7` in units of `S_v/128`, then per-token dynamic quantisation feeds the out-projection.

8. **2026-09-07 — Weight ROM word = 2048 bits (one 8-row tile group); a 32×32 tile loads in 4 cycles.**
   With a 32-byte/cycle weight port the decoder (M = 1) would run at 1/32 utilisation (≈1 M cycles per
   token). A 2048-bit word gives 4 cycles per tile (25 % utilisation at M = 1, ≈130 k cycles/token) and
   ≥ 99 % at M ≥ 64. The committed hex files keep the prompt's one-32-bit-word-per-line format; the
   `$readmemh` image (one memory word per line) is derived from them at elaboration into
   `target/readmemh/` after the sha256 check, so nothing but the committed files is ever loaded.

9. **2026-09-07 — Array organisation: 4 pipeline stages of 8 combinational MAC rows; activations
   broadcast along rows, partial sums flow down.** A per-row-skewed systolic array needs either a
   2048-bit × 32-stage weight-load chain or a 32-cycle tile switch; grouping 8 rows per stage matches the
   ROM word to one stage, keeps the 4-cycle tile switch, and cuts latency to 5 cycles
   (`WhisperConfig.rowsPerStage`, 1 = fully systolic).

10. **2026-09-07 — Reference stop rule and context.** The int model stops at `<|eot|>` or 224 generated
    tokens, like the reference. Whisper needs trailing (silent) context to emit `<|eot|>`: with the
    context truncated to the speech length the model repeats the phrase until the limit, and even
    5 s of padding is not always enough for a 2 s clip. The variable-context rule used for RTL runs is
    chosen from the measurements in status.md (`n_frames_for` in `golden/whisper_int.py`).

11. **2026-09-07 — `SyncReadMem` read data is valid exactly one cycle after an enabled read** (firtool
    lowers a disabled read to X). Every consumer registers read data on that cycle; the RomLiteral
    backend mirrors the timing with `RegEnable`. Memory randomisation is disabled in simulation
    (`-disable-mem-randomization`), otherwise firtool's init loop overwrites `$readmemh` contents.

12. **2026-09-07 — RTL accuracy runs use the full 30 s window (`n_frames = 3000`); the variable-frame
    path stays supported but is not the accuracy configuration.** Measured with the int golden on the
    short RTL set (rtl_20) / varied set: context = speech + 5.12 s → WER 179 % / 85 %; ≥ 10.24 s →
    175 % / 83 %; ≥ 15.36 s → 5.1 % / 76 %; full 30 s → 7.6 % / 10.8 %. Whisper-tiny hallucinates
    repetitions whenever the encoder context is shorter than it was trained on, independent of our
    numerics (the fp32 model behaves the same). The hardware keeps `n_frames ≤ 3000` variable (every
    op is sized from it, the sequencer substitutes it), so short clips *can* pay less, but the WER gate
    and the e2e token comparison are run at 3000 frames. Cost: every clip costs the full encoder
    (see status.md cycle table).

13. **2026-09-07 — No memory may have a write port with a constant all-true (or absent) mask if any
    other port of it uses byte masks.** firtool 1.x (via Chisel 7.15) silently lowers such a memory
    without masks on *every* port, so masked writes clobber whole words. The KV cache therefore has only
    data-dependent-mask ports in the chip, and its test-only load port takes its mask from IO
    (`KVCache(debugPort = true)`). Found by `KVWriteSpec` (engine → cache path), which the unit tests
    of Phase 3 did not cover because they loaded the cache directly.

14. **2026-09-07 — FastSim and where the e2e sets run.** The `FastSim` requirement is implemented in the
    harness, not the RTL: tokens accumulate in a 256-deep on-chip queue and the host only polls `done`
    every 4096 cycles (`E2E_POLL`), so no compute is skipped or altered. Verilator throughput of the full
    chip is ~12–13 k cycles/s on 4 threads (this container) and a 30 s-context clip costs ~42 M cycles,
    i.e. ~55 min per clip; the default set (13 clips) and the long set (21 clips) are therefore run as
    one container per clip on Modal (`tests/e2e/modal/modal_e2e.py`, `make e2e-modal`) with the
    identical image (Debian, Verilator 5.006, sbt 1.10.7, committed weights). Locally, `make e2e`
    still runs the same sets serially.

15. **2026-09-08 — The chip never fabricates mel rows.** The effective frame count is the value of
    register 2 if non-zero, else the number of frames streamed; it must be a non-zero multiple of 128,
    at most 3000, and at most the number of frames streamed. A start that violates this is refused and
    flagged in register 0 bit 3 (`frameErr`). Previously the auto path rounded the streamed count up to
    a multiple of 128, which would have read unwritten or stale rows of the mel bank (found in review of
    PR #1; `FrameCheckSpec` covers the rule). The host rule (`n_frames_for`) already produced multiples of
    128 from the 30 s mel, so no existing run is affected.

16. **2026-09-08 — Reductions are balanced trees; the sampler argmax is pipelined.** ASAP7 logic
    synthesis (Yosys/ABC, 1 ns target; see `docs/paper`) showed the sampler's lowest-index
    argmax as a 31-deep serial compare chain (9.5 ns) and serial 16-lane sums / maxima in the vector
    unit and attention. All are now `reduceTree`s (integer results identical), and the sampler tree is
    split across two register stages with the sequencer waiting for the sampler's `done` before it
    samples. `SamplerSpec` (full 1621-beat rows, ties, masks, bubbles) and the smoke e2e (tokens and
    cycle count unchanged: 10,153,984 at 1024 frames) cover the change; sampler critical path 977 ps.

