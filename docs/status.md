# Status

Numbers are measured unless marked "est.". Commands are relative to the repo root; `sbt` is at
`/opt/sbt/bin/sbt`, Python runs through `uv run`.

## Phase 0 — environment and references (2026-09-07) — PASS

| Item | Result | Reproduce |
|---|---|---|
| Toolchain | Scala 2.13.18, Chisel 7.15.0, sbt 1.10.7, Verilator 5.020, firtool via resolver; svsim smoke test green (17 s incl. compile) | `sbt "testOnly whisper.SmokeSpec"` |
| Python | uv, torch 2.14 (CPU), openai-whisper 20250625, transformers 5.16, numpy 2.4 | `make setup` |
| Checkpoint | `tiny.pt` sha256 `65147644…22b9` (openai-whisper multilingual tiny), facts verified: n_vocab 51865, d 384, 6 heads, 4+4 layers, 37.18 M params (prompt said ≈37.8 M; the checkpoint has 37,184,640), key projections have no bias | `golden/reference_cpu.py` writes `data/ref/_meta.json` |
| HF mirror cross-check | `openai/whisper-tiny` safetensors weights identical to `tiny.pt` (max-abs diff 0.0); fp32 greedy tokens identical on 60 clips (50 test-clean + 10 varied), WER between the two = 0.0 | `uv run python golden/crosscheck_hf.py --n 50` |
| Data | LibriSpeech test-clean (2620 utts) fetched to `$WHISPER_SI_DATA`; `data/testclean_200.txt` = sorted first 200 ids; `data/rtl_20.txt` / `data/rtl_default.txt` = first 20 / 10 of those with duration ≤ 6 s; `data/varied/` = 10 committed clips (2/5/12/29 s, M/F, white & babble noise at 10 dB SNR seeded, 8 s inserted silence, German + French from FLEURS) | `make varied` |
| fp32 references | `data/ref/*.json` (210 clips), byte-identical on regeneration | `make refs-check` → "refs byte-identical" |

Baseline fp32 WER (openai-whisper tiny, greedy, one 30 s window, whisper English normalizer):

| Set | n | WER |
|---|---|---|
| testclean_200 | 200 | 5.54 % |
| rtl_20 | 20 | 8.08 % |
| rtl_default | 10 | 12.75 % |
| varied | 10 | 10.8 % |

`uv run python golden/wer.py --hyp data/ref --sets testclean_200,rtl_20,rtl_default,varied`

Deviations: see `docs/decisions.md` #2 (224-token stop rule follows whisper's default), #3 (FLEURS for
non-English).

Next-phase risks: int8 residual stream is expected to miss the WER gate because of Whisper's outlier
channels (int16 fallback is planned as a parameter from the start); calibration uses dev-clean, not
test-clean, so static scales may be exceeded on the varied/noisy clips (saturation counters will be
reported).

## Phase 1 — golden int model and numerics freeze (2026-09-07) — PASS (gate met)

Frozen numerics: `docs/numerics.md`; tiling: `docs/tiling.md`; constants in `weights/MANIFEST.json`
(`params`). Weights: 247 tensors, 86.6 MB of hex, committed; `make weights-check` → OK.

| Model (full 30 s context) | Set | n | WER | token identity vs fp32 | text WER vs fp32 |
|---|---|---|---|---|---|
| fp32 CPU reference | testclean_200 | 200 | 5.54 % | — | — |
| int golden (r16 / phi16 / no smooth) | testclean_200 | 200 | **5.52 %** | 61 % | 1.6 % |
| fp32 | varied | 10 | 10.8 % | | |
| int golden | varied | 10 | 10.8 % | 60 % | 3.6 % |
| fp32 / int | rtl_20 | 20 | 8.08 % / 7.58 % | 65 % | 3.5 % |

Gate: int WER ≤ fp32 + 0.5 → 5.52 ≤ 6.04 ✓.  
Reproduce: `uv run python golden/run_golden.py --set testclean_200,varied --frames full` then
`uv run python golden/wer.py --hyp out/golden_r16_phi16_saNone_sf0_full`.
Unit tests: `uv run pytest -q golden/tests` (18 tests: every op vs float, tile-order invariance,
masked-key independence, im2col vs torch conv).
Variants (int8 residual, int8 GELU LUT, SmoothQuant) and the variable-context rule: measurements are
appended below when the batch finishes (`/tmp/golden_batch.sh`).

## Phase 2 — MatmulEngine + WeightStore (2026-09-07) — PASS

* `WeightStoreEquivalenceSpec`: 13/13 — RomLiteral, RomInit ($readmemh) and Sram (after loading)
  return identical data for every address on real tensors of all three kinds, and the flat address
  decoder is checked on one encoder layer. `sbt "testOnly whisper.WeightStoreEquivalenceSpec"`.
* `MatmulEngineSpec`: 19/19 bit-exact (`uv run python tests/vectors/gen_matmul.py` then
  `sbt "testOnly whisper.MatmulEngineSpec"`): 10 random cases (int8/int16/raw/wide, dynamic/static,
  unsigned activations, M = 1..1500) + real tensors with real activations for every distinct shape
  (conv1 im2col 288→384, conv2 stride-2 im2col 1152→384, q/k/o 384→384, fc1 384→1536, fc2 1536→384,
  cross-K, LM head 384→51872 wide).

| Job | cycles | MAC utilisation |
|---|---|---|
| 1500×384×384 int8 dynamic (random) | 216,586 | **99.7 %** (gate ≥ 90 %) |
| real fc1 128×384×1536 | 76,042 | 97.0 % |
| real conv2 128×1152×384 (im2col) | 57,034 | 97.0 % |
| M = 1 LM head 384×51872 (wide) | 97,270 | 20.0 % (weight-port bound, 4 cycles/tile) |
| M = 1 384×1536 int16 | 2,890 | 19.9 % |
