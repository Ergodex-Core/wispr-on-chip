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
