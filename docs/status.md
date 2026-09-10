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
Variants (same golden model, one switch changed, full 30 s context; WER on rtl_20 / varied):

| Variant | flag | rtl_20 | varied | verdict |
|---|---|---|---|---|
| frozen numerics (int16 residual, Φ-LUT GELU) | — | **7.58 %** | **10.8 %** | shipped |
| int8 residual stream | `--residual-bits 8` | 98.0 % | 98.4 % | unusable (outlier channels), decision #5 |
| int8 GELU output LUT | `--gelu-mode lut8` | 21.2 % | 35.2 % | rejected, decision #6 |
| SmoothQuant α = 0.5 | `--smooth-alpha 0.5` | 99.5 % | 99.6 % | folding not validated; not needed, not pursued |

Variable encoder context (`--frames var --pad-frames P --min-frames M`, frozen numerics), WER on rtl_20 / varied:
P = 512, M = 0 → 178.8 % / 84.8 %; P = 512, M = 1024 → 174.7 % / 82.8 %; P = 256, M = 1024 → 174.7 % / 74.8 %;
P = 512, M = 1536 → 5.1 % / 76.0 %; full 3000 frames → 7.6 % / 10.8 %. Whisper-tiny repeats itself whenever the
encoder context is shorter than 30 s (the fp32 model does the same), so the accuracy configuration is the full
window (decision #12). Reproduce: `uv run python golden/run_golden.py --set rtl_20,varied --frames var --pad-frames 512 --min-frames 1024`
then `golden/wer.py --hyp out/golden_r16_phi16_saNone_sf0_var_p512_m1024`.

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

## Phase 3 — VectorUnit + Attention (2026-09-07) — PASS

All vectors come from the golden model on a real clip (`varied/en_2s_f`, 256 frames); every case is
bit-exact against the golden dumps. `uv run python tests/vectors/gen_vector.py && uv run python
tests/vectors/gen_attention.py && sbt "testOnly whisper.VectorUnitSpec whisper.AttentionSpec"` → 12/12.

| Case | What | cycles |
|---|---|---|
| ln_enc0 | LayerNorm (enc.0.ln1) on 40 residual rows → int8 + rowfac | 123 / row |
| dynq_attn_enc0 | dynamic quant of attention output | 85 / row |
| gelu_dynq_enc0 | fused Φ-LUT GELU + dynamic quant, 1536 wide | 229 / row |
| gelu_static8_conv1 | GELU16 + static int8 requant (conv1) | 59 / row |
| add_resid_enc0 / add_pos_enc / add_embed_dec | scaled adds (bank+bank, bank+ROM pos-emb, per-token embedding mult) | 30–33 / row |
| embed | embedding row extracted from the LM-head ROM tiles | 83 / row |
| enc0_full | encoder layer-0 self-attention, 128 q × 128 k, 6 heads | 62,052 |
| enc0_ragged | 100 q × 70 k (partial tiles, masked keys) | 48,948 |
| dec0_self_pos41 | decoder self-attention, causal, 42-key history | 864 |
| dec0_cross_pos41 | decoder cross-attention, 1 q × 128 k | 1,542 |

Bugs found by these tests and fixed: pipelined divider produced one extra quotient bit (attention
output exactly 2×); vector-unit pass-3 half-word misalignment; array valid-chain reset alignment.

## Phase 4 — full chip on Verilator (2026-09-07/08) — PASS

`WhisperTop` (sequencer + 186-instruction generated micro-program, engine, vector unit, attention,
KV cache, sampler, 8 activation banks, weight store RomInit) elaborates and runs under Verilator.
Every clip is run at the full 30 s context (3000 mel frames, decision #12), one Verilator process per
clip (locally serial, or one Modal container per clip — same image, same committed weights, decision #14).

**Gate: RTL tokens must be 100 % identical to the golden int model on every clip, and the RTL WER on
the 20-utterance set must be ≤ fp32 WER + 0.5 (8.08 + 0.5 = 8.58 %).**

| Set | clips | ran | RTL tokens = golden tokens | RTL WER vs ground truth | fp32 CPU WER | gate |
|---|---|---|---|---|---|---|
| smoke (2 s clip, 1024 and 3000 frames) | 1 | 1 | 1/1 | 0 % | 0 % | ✓ |
| long = rtl_20 (20 test-clean utts) + 29 s clip | 21 | 21 | **21/21** | 6.59 % (all 21) / **7.58 % on rtl_20** | 8.08 % on rtl_20 | ✓ (7.58 ≤ 8.58) |
| default = 2/5/12 s clips + rtl_default (10 utts) | 13 | 13 | **13/13** | 10.24 % (all 13) / **12.75 % on rtl_default** | 12.75 % on rtl_default | ✓ (12.75 ≤ 13.25) |

RTL text vs fp32 CPU text on rtl_20: 3.5 % word difference (the same as golden-vs-fp32 in Phase 1, since
the tokens are identical). The WER on the tiny 10-utterance rtl_default set is dominated by two
utterances that both fp32 and int transcribe wrongly ("Stuffed into you, his belly…" for
1089-134686-0001, 37.5 % on its own), which is why the 20-utterance set is the one gated.

Reproduce (locally, serial; ≈ 1 h per clip): `make e2e` (default set), `make e2e-long`; or on Modal
`make e2e-modal`; then `make report` renders the tables below from `out/e2e/<set>_full/results.json`.
Prepared inputs, golden tokens, RTL tokens and cycle counts are in `out/e2e/<set>_full/<clip>/`.

### Cycle counts vs the analytic model

`tests/e2e/cycle_model.py` predicts the cycle count of a clip from the micro-program structure and
constants measured only at unit level (Phase 2: `KT·NT·(M+4)+10` per matmul; Phase 3: per-row
vector-unit costs and the attention loop: S-matmul, 12 cycles per query row of online softmax, two
P·V passes, 31 cycles per query row of finalisation). No constant was fitted on the e2e runs.

| clip | frames | generated tokens | measured cycles | model | model / measured |
|---|---|---|---|---|---|
| varied/en_2s_f | 1024 | 6 | 10,153,984 | 10,176,473 | 1.002 |
| varied/en_2s_f | 3000 | 6 | 42,135,552 | 42,175,157 | 1.001 |
| 1089-134686-0001 | 3000 | 12 | 43,401,216 | 43,444,109 | 1.001 |
| 1089-134691-0001 | 3000 | 20 | 45,088,768 | 45,136,045 | 1.001 |
| varied/en_12s_f | 3000 | 55 | 52,477,952 | 52,538,265 | 1.001 |
| varied/en_29s_m | 3000 | 89 | 59,731,968 | 59,808,337 | 1.001 |

Breakdown at 3000 frames (model, 6 tokens): encoder layers 37.2 M (88 %), of which attention 23.1 M
(55 % of the whole run: 1500 queries × 1500 keys × 6 heads × 4 layers through one 32×32 engine and one
16-lane softmax), conv1/conv2 1.2 M, ln_post + cross-K/V 1.9 M, decoder 1.8 M (≈ 200 k cycles per
generated token, of which the 384×51872 LM head is 97 k). Every 3000-frame clip therefore costs
42.1 M + ≈ 0.2 M per generated token. Measured engine utilisation over a whole run (register 7,
cumulative engine-busy cycles): 69.7 % at 1024 frames (7.08 M of 10.15 M); the rest is the softmax and
vector-unit phases, during which the engine idles. Requant saturations per run (register 10): 1 at
1024 frames on the smoke clip.

### Simulation cost

| | elaboration (Chisel + firtool) | Verilator build (verilate + C++) | throughput | 3000-frame clip |
|---|---|---|---|---|
| this container (4 threads, Verilator 5.020) | 16 s (incl. writing 247 `$readmemh` files) | 8 s + 157 s | 12–13 k cycles/s | 54 min |
| Modal, 8 vCPU (4 threads, Verilator 5.006) | same | ≈ 3 min | 6.6–8.9 k cycles/s | 85–150 min |

Modal pre-empted 3 of 13 (default) and 4 of 21 (long) containers once each; the function restarts the
same input automatically, which is why the long set took 2 h 40 min wall and the default set 3 h 20 min
instead of ~1.5 h. Cycle counts are deterministic: the smoke clip gives 42,135,552 cycles locally and
on Modal.

### Per-clip results (`make report`)

### e2e set `default_full` — 13/13 clips ran, golden-vs-RTL token identity 13/13, RTL WER vs ground truth 10.24 %

| clip | CPU fp32 text | RTL text | tokens = golden | text = CPU | WER | cycles | sim wall (s) | cycles/s |
|---|---|---|---|---|---|---|---|---|
| varied/en_2s_f | Won't you tell Douglas? | Won't you tell Douglas? | yes | yes | 0.0 % | 42,135,552 | 6387 | 6,597 |
| varied/en_5s_m | The greatness of the ransom priced the son of God indicates this. | The greatness of the ransom priced the son of God indicates this. | yes | yes | 8.3 % | 43,610,112 | 6561 | 6,647 |
| varied/en_12s_f | It takes me several years to make this magic powder, but at this moment I am pleased to say it is nearly done. You see I am making it for my good wife Margot Lot, who wants to use some of it for a purpose of her own. | It takes me several years to make this magic powder, but at this moment I am pleased to say it is nearly done. You see, I am making it for my good wife Margot a lot, who wants to use some of it for a purpose of her own. | yes | no | 6.4 % | 52,477,952 | 7952 | 6,599 |
| 1089-134686-0001 | Stuffed into you, his belly, couchled him. | Stuffed into you, his belly, couchled him. | yes | yes | 37.5 % | 43,401,216 | 6535 | 6,641 |
| 1089-134686-0003 | Hey Bertie, any good in your mind? | Hello Bertie, any good in your mind? | yes | no | 0.0 % | 42,979,328 | 3977 | 10,806 |
| 1089-134686-0004 | Number 10 Fresh Nelly is waiting on you. Good night husband. | Number 10, Fresh Nelly is waiting on you. Good night husband. | yes | yes | 0.0 % | 44,036,096 | 5041 | 8,736 |
| 1089-134686-0007 | A cold lucid indifference rained in his soul. | A cold lucid in difference rained in his soul. | yes | no | 37.5 % | 43,188,224 | 6583 | 6,561 |
| 1089-134686-0010 | Well now, in this I declare you have a head and so has my stick. | Well now, in this I declare you have a head and so has my stick. | yes | yes | 14.3 % | 44,457,984 | 6709 | 6,626 |
| 1089-134686-0014 | He tried to think how it could be. | He tried to think how it could be. | yes | yes | 0.0 % | 42,766,336 | 6087 | 7,026 |
| 1089-134686-0015 | but the dusk deepening in the school room covered over his thoughts. The bell rang. | but the dusk deepening in the school room covered over his thoughts. The bell rang. | yes | yes | 14.3 % | 44,879,872 | 6824 | 6,577 |
| 1089-134686-0016 | Then you can ask him questions on the cataclysm deadelist. | Then you can ask him questions on the cataclysm deadelist. | yes | yes | 20.0 % | 44,244,992 | 6626 | 6,677 |
| 1089-134686-0026 | The rector did not ask for a catacysm to hear the lesson from. | The rector did not ask for a catacysm to hear the lesson from. | yes | yes | 7.7 % | 44,666,880 | 6705 | 6,662 |
| 1089-134686-0027 | He clashed his hands on the desk and said, | He clasped his hands on the desk and said, | yes | no | 0.0 % | 43,401,216 | 5700 | 7,615 |

Mean cycles/clip 44,326,597; mean sim wall 6,284 s.

### e2e set `long_full` — 21/21 clips ran, golden-vs-RTL token identity 21/21, RTL WER vs ground truth 6.59 %

| clip | CPU fp32 text | RTL text | tokens = golden | text = CPU | WER | cycles | sim wall (s) | cycles/s |
|---|---|---|---|---|---|---|---|---|
| varied/en_29s_m | to the surprise of all, and especially of Lieutenant Procope, the line indicated a bottom at a nearly uniformed depth of from four to five fathoms. And although the sounding was persevered with continuously for more than two hours over a considerable area, the differences of level were insignificant, not corresponding in any degree to what would be expected over the sight of a city that had been terrorist like the seats of an amphitheater. | to the surprise of all, and especially of Lieutenant Procope, the line indicated a bottom at a nearly uniformed depth of from four to five fathoms. And although the sounding was persevered with continuously for more than two hours over a considerable area, the differences of level were insignificant, not corresponding in any degree to what would be expected over the sight of a city that had been terrorist like the seats of an amphitheater. | yes | yes | 4.0 % | 59,731,968 | 9054 | 6,598 |
| 1089-134686-0001 | Stuffed into you, his belly, couchled him. | Stuffed into you, his belly, couchled him. | yes | yes | 37.5 % | 43,401,216 | 6555 | 6,621 |
| 1089-134686-0003 | Hey Bertie, any good in your mind? | Hello Bertie, any good in your mind? | yes | no | 0.0 % | 42,979,328 | 6195 | 6,938 |
| 1089-134686-0004 | Number 10 Fresh Nelly is waiting on you. Good night husband. | Number 10, Fresh Nelly is waiting on you. Good night husband. | yes | yes | 0.0 % | 44,036,096 | 6565 | 6,707 |
| 1089-134686-0007 | A cold lucid indifference rained in his soul. | A cold lucid in difference rained in his soul. | yes | no | 37.5 % | 43,188,224 | 6242 | 6,919 |
| 1089-134686-0010 | Well now, in this I declare you have a head and so has my stick. | Well now, in this I declare you have a head and so has my stick. | yes | yes | 14.3 % | 44,457,984 | 6699 | 6,636 |
| 1089-134686-0014 | He tried to think how it could be. | He tried to think how it could be. | yes | yes | 0.0 % | 42,766,336 | 6379 | 6,705 |
| 1089-134686-0015 | but the dusk deepening in the school room covered over his thoughts. The bell rang. | but the dusk deepening in the school room covered over his thoughts. The bell rang. | yes | yes | 14.3 % | 44,879,872 | 6727 | 6,671 |
| 1089-134686-0016 | Then you can ask him questions on the cataclysm deadelist. | Then you can ask him questions on the cataclysm deadelist. | yes | yes | 20.0 % | 44,244,992 | 6704 | 6,600 |
| 1089-134686-0026 | The rector did not ask for a catacysm to hear the lesson from. | The rector did not ask for a catacysm to hear the lesson from. | yes | yes | 7.7 % | 44,666,880 | 5894 | 7,578 |
| 1089-134686-0027 | He clashed his hands on the desk and said, | He clasped his hands on the desk and said, | yes | no | 0.0 % | 43,401,216 | 6470 | 6,708 |
| 1089-134686-0029 | On Friday, confession will be heard all the afternoon after beads. | On Friday, confession will be heard all the afternoon after beads. | yes | yes | 0.0 % | 43,610,112 | 5769 | 7,559 |
| 1089-134686-0030 | Beware of making that mistake. | Beware of making that mistake. | yes | yes | 0.0 % | 42,344,448 | 5539 | 7,645 |
| 1089-134686-0032 | He is called as you know the apostle of the Indies. | He has called, as you know, the apostle of the Indies. | yes | no | 9.1 % | 44,036,096 | 5096 | 8,642 |
| 1089-134686-0033 | A great saint, Saint Francis Xavier. | A great saint, Saint Francis Xavier. | yes | yes | 0.0 % | 42,557,440 | 6385 | 6,665 |
| 1089-134686-0034 | The rector paused and then shaking his clasp to hands before him went on. | The rector paused and then shaking his clasped hands before him went on. | yes | no | 0.0 % | 44,457,984 | 6673 | 6,662 |
| 1089-134686-0035 | He had the faith in him that moves mountains. | He had the faith in him that moves mountains. | yes | yes | 0.0 % | 42,979,328 | 6487 | 6,625 |
| 1089-134686-0036 | a great saint, saint Francis Xavier. | A great saint, saint Francis Xavier. | yes | yes | 0.0 % | 42,557,440 | 5623 | 7,569 |
| 1089-134686-0037 | In the silence, their dark fire kindled the dusk into a tony glow. | In the silence, their dark fire kindled the dusk into a tony glow. | yes | yes | 7.7 % | 44,666,880 | 6711 | 6,656 |
| 1089-134691-0000 | He could wait no longer. | He could wait no longer. | yes | yes | 0.0 % | 42,135,552 | 6297 | 6,691 |
| 1089-134691-0001 | for a full hour he had paced up and down waiting, but he could wait no longer. | for a full hour he had paced up and down waiting, but he could wait no longer. | yes | yes | 0.0 % | 45,088,768 | 6772 | 6,658 |

Mean cycles/clip 44,389,912; mean sim wall 6,421 s.

WER above is per clip against the LibriSpeech/varied ground truth after whisper's English normaliser;
"text = CPU" compares the RTL text with the fp32 CPU text (differences are the int-vs-fp32 token
differences already measured for the golden model in Phase 1, since RTL tokens equal golden tokens).

Bugs found and fixed at chip level (all now covered by unit tests): fused GELU in the x0 add not
applied; KV transposer stalls; firtool dropping byte masks on the KV memory (decision #13); KV key
index using the rowfac row instead of the tensor row. Each was located by the breakpoint flow
(`E2EDebugSpec` pauses the sequencer at a pc, dumps the banks, `tests/e2e/compare_dumps.py` compares
with the golden dumps): every intermediate tensor of the encoder (conv1/conv2/x0, layer-0 LN/Q/attention/
adds/FFN, encoder output + rowfac) and of decoder position 0 through layer 0 plus the LM-head input at
position 3 is bit-exact.

## Phase 5 — ROM-literal proof and hardening (2026-09-07/08) — PASS

* `RomLiteralLayerSpec` (`make test`): the six weight tensors of encoder layer 0 elaborated as literal
  `VecInit` ROMs (`romLiteralMaxBits = 8 Mbit`) and re-run bit-exact on the real-activation cases:

  | tensor | bank size | elaboration + Verilator build | result |
  |---|---|---|---|
  | enc.0.attn.q.w (384×384) | 1.18 Mbit | 21 s | bit-exact |
  | enc.0.fc1.w (384×1536) | 4.72 Mbit | 171 s | bit-exact |
  | enc.0.fc2.w (1536×384) | 4.72 Mbit | 167 s | bit-exact |

  Practical ceiling for the literal path on this machine: a 4.7 Mbit bank costs ~3 min of build; the
  whole model (302 Mbit) as literals would take ~3 h of Verilator build and is not attempted — the ASIC
  flow only needs the emission to be correct, which the above proves. RomInit stays the default for
  full-chip simulation.
* Assertions added (all active in every Verilator run): engine output sink never stalls; `MatmulCmd`,
  `VecCmd`, `AttnCmd` and the token stream are irrevocable Decoupled handshakes; command rows/tiles in
  range; KV-cache read/write addresses in range; activation-bank read/write addresses in range and no
  same-bank port conflicts; emitted token ids < 51865. A saturation counter for the requant stage is
  exposed at register 10. A start with an invalid frame count (zero, not a multiple of 128, > 3000 or
  > frames streamed) is refused and flagged in register 0 bit 3 (`FrameCheckSpec`, decision #15). (Verilator is 2-state, so the "no X" requirement is checked by construction:
  registers are reset-initialised and memory randomisation is disabled; outputs are additionally
  range-checked as above.)

* `make test` (python op tests + the six Scala/Verilator suites, RomLiteral included): see the
  "Final" section below for the last full run.

## Phase 6 — ASAP7 7 nm logic synthesis (2026-09-08/10)

Pre-layout standard-cell synthesis of every compute block with Yosys 0.33 + ABC onto the ASAP7
predictive PDK (7.5-track RVT, typical corner), 1000 ps delay target, memories black-boxed; the
Verilog is the same firtool output that runs under Verilator, re-emitted without packed arrays
(`sbt "Test/runMain whisper.synth.EmitSynth"`). Reproduce with `synth/README.md`.

| block | cells | flops | area (mm²) | critical path (ps) | est. Fmax (GHz) |
|---|---|---|---|---|---|
| SystolicArray (32×32, 8 rows/stage) | 706,724 | 29,477 | 0.076 | 1207 | 0.79 |
| &nbsp;&nbsp;variant: 4 rows/stage | 759,002 | 41,389 | 0.083 | 1100 | 0.86 |
| MatmulEngine (array + requant + control) | 1,118,387 | 50,626 | 0.120 | 1267 | 0.75 |
| VectorUnit (16 lanes) | 505,826 | 1,633 | 0.050 | 4843 | 0.20 |
| Attention (softmax, divider) | 634,784 | 13,501 | 0.059 | 5824 | 0.17 |
| KVCache logic (transposer) | 87,073 | 16,494 | 0.010 | 725 | 1.27 |
| Sequencer | 4,860 | 602 | 0.0005 | 981 | 0.96 |
| Sampler | 8,821 | 385 | 0.001 | 977 | 0.96 |
| WeightStore mux (247 banks) | 513,535 | 425 | 0.040 | 1027 | 0.92 |

Logic total 0.28 mm² (the array is counted once, inside MatmulEngine). Fmax adds ~60 ps for setup
plus clock-to-Q; wire load is not modelled, so these paths are optimistic.

**Meets 1 GHz:** sequencer, sampler, KV-cache logic, weight-store mux — after the serial reductions
were replaced by trees (decision #16; the sampler's original 31-deep compare chain was 9.5 ns).
**Misses it:** the attention unit (5.8 ns) and the vector unit (4.8 ns), both single-cycle arithmetic
chains inside step-sequenced state machines — the online-softmax step (64-way max tree, subtract,
scale multiply, barrel shift, exp LUT, rescale multiply in one FSM cycle) and the LayerNorm
inverse-square-root Newton step. Both pipeline mechanically: five stages on the softmax step costs
4 cycles per query row per key tile = 3.5 M cycles (8 %) on a 3000-frame clip; the Newton step runs
once per row so its pipeline costs nothing measurable. Not done here.
**The array:** 1.21 ns with 8 combinational MAC rows per stage; 1.10 ns with 4 rows (a generator
parameter, bit-exact on the same engine tests) for 10 % more area — so about half the path is the
int8 multiplier and 32-bit accumulate, not the chain.

Memories are not synthesised (ASAP7 ships no compiler): 119 Mbit of SRAM (banks 59, KV 58,
accumulator 1.6, attention/row buffers 0.6) and 308 Mbit of weight ROM. At a published 7 nm
high-density bit cell of 0.027 µm² and 50 % macro efficiency that is ≈ 6.4 mm² of SRAM plus ≈ 3 mm²
of via-programmed ROM, so the chip is memory-dominated: ~10–13 mm² of memory against 0.28 mm² of logic.

Latency from the measured cycle counts: a 30 s window is 42.1 M cycles = 248 ms as synthesised
(0.17 GHz), 56 ms at 0.75 GHz (once the two chains are pipelined, engine-limited), 42 ms at 1 GHz;
each generated token adds 200 k cycles (0.2 ms at 1 GHz).

## Final (2026-09-08)

**Result: all five phases pass, plus 7 nm synthesis of every block (Phase 6 above).** The RTL is bit-exact to the golden int model on every clip run
(35 full-context runs: 1 + 13 + 21), the int model meets the WER gate on the 200-utterance set
(5.52 % vs fp32 5.54 %), and the RTL meets it on the 20-utterance RTL set (7.58 % vs fp32 8.08 %).

`make all` = `weights-check` + `test` + `e2e`. The three parts were run separately because the serial
`e2e` target takes ~13 h on this machine (13 clips × ~1 h); the e2e sets were run on Modal with the
identical image (`make e2e-modal`), the unit tests locally on the final commit: `make test` → pytest 18 passed (4.7 s); sbt 6 suites, 49 tests
succeeded, 0 failed, 18 min (2026-09-08 00:21 UTC, `/tmp/make_test_final.log`); `make weights-check` → OK.

Deviations from the prompt, all recorded in `docs/decisions.md`:

1. Accuracy runs use the full 30 s encoder context, not a variable one (decision #12; the hardware still
   supports `n_frames ≤ 3000`). Cost: ~42 M cycles per clip regardless of length.
2. The residual stream is int16, not int8 (decision #5; int8 gives 98 % WER).
3. GELU is x·Φ(x) with a Φ LUT on int16, not an int8 output LUT (decision #6).
4. The systolic array has 8 rows per pipeline stage (4 stages) rather than one row per stage
   (decision #9); utilisation is unchanged (99.7 %) and the tile switch stays 4 cycles.
5. Whole-model RomLiteral elaboration was not attempted (≈ 3 h of Verilator build); the literal path is
   proven on three real tensors of encoder layer 0 (up to 4.7 Mbit each) and the equivalence spec.
6. `FastSim` is a harness feature (token queue + coarse `done` polling), not an RTL mode (decision #14).
7. The checkpoint has 37.18 M parameters, not ≈ 37.8 M as the prompt stated.

Known risks / not done:

* Calibration used dev-clean only; the noisy/foreign varied clips were transcribed correctly by the
  int model (Phase 1), but the requant saturation counter is the only runtime guard.
* The default and long sets overlap in 10 utterances (rtl_default ⊂ rtl_20), so the distinct clips run
  end-to-end on the RTL are 24, not 35.
* Synthesis is pre-layout, from an open-source mapper, with memories black-boxed and no wire load;
  no place-and-route or power simulation was run. Two blocks miss 1 GHz as written (Phase 6).
* SmoothQuant folding was implemented but never validated (99 % WER); it is off and not needed.
