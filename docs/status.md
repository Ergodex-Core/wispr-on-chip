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
