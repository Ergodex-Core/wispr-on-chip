# 51-case Arcilator / Verilator comparison

Finalized: 2026-09-12T17:37:06.569033+00:00

All **51 planned unit-test replay pairs have finished**: 102 simulator records are present and the bulk runner has written its completion marker. Completion includes failed and timed-out cases; it does not mean every case passed.

| Test group | Passed | Expected | Other outcomes |
|---|---:|---:|---|
| Original Scala reference cases | 51 | 51 | none |
| Verilator standalone replays | 51 | 51 | none |
| Arcilator standalone replays | 9 | 51 | jit_or_startup_timeout: 39; simulation_failed: 3 |
| Python golden-model cases | 18 | 18 | passed |

The completion marker contains exit code `1`. Exit 1 is expected when one or more benchmark cases fail. Python results are software validation and have no hardware simulation cycles/second metric. Its JUnit-reported wall time is 3.141 s.

The **34 named E2E clip executions (24 unique clips) are outside this completed unit benchmark** and have no paired runtime results in these artifacts. This table makes no E2E transcription claim.

## Measurement setup

Both engines ran on the same EC2 **m7i.xlarge (4 vCPU, 16 GiB)**. Recorded CPU: **Intel(R) Xeon(R) Platinum 8488C**; the provenance reports 4 logical CPUs. Repository commit: `42847240bad0d496b47093e0a4f7f1a706d447b8`.

Arcilator: CIRCT 04824a080, LLVM 23.0.0git. Verilator 5.020 2024-01-01 rev (Debian 5.020-1).

The benchmark used fresh frontend/build outputs and sequential engine executions, with 2 Verilator C++ build jobs and 1 simulation thread. Recorded timeout limit per frontend/build or runtime stage: **300 s**. Arcilator runs that time out before BEGIN therefore exhaust the runtime-stage limit without yielding a completed startup or simulation measurement.

Arcilator used the compiler's default internal MLIR/LLVM threading, with no explicit compiler job/thread cap. `OMP_NUM_THREADS=1` does not cap all compiler threads. **Build parallelism was not an equal-thread controlled comparison**: Verilator used two C++ build jobs; Arcilator used its default compiler threading. The simulation thread setting is separate from compiler parallelism.

Arcilator ran with **`ARC_DESEQ_DISABLE=1`**, the supplied package's validated mode with clock promotion disabled. These results characterize that mode, rather than establishing the performance of every Arcilator optimization configuration.

Fifty of the 51 cases already target components or subsystems; only FrameCheckSpec targets the full WhisperTop. Some harnesses still include large configured weight selections. No measurement here proves which unused logic either compiler pruned. See the [methodology review](METHODOLOGY_REVIEW.md) for the tested DUTs, weight footprints and limits of the comparison.

Recorded compiler-stage CPU utilization is `(user CPU seconds + system CPU seconds) / wall seconds`. It estimates average busy cores, not the number of OS threads. The rows cover different compiler stages and are not a direct compiler-speed ratio.

| Recorded stage | Stages with CPU data | Median busy-core estimate | Range |
|---|---:|---:|---|
| Arcilator SV frontend | 51 | 1.160 | 0.827–2.016 |
| Verilator translation + C++ build | 51 | 1.778 | 1.063–1.928 |

Runtime CPU counters cover each whole process, including startup/JIT and simulation. They do not isolate the JIT phase or establish a general JIT thread count.

## Timing definitions

Each simulator timing cell is **frontend / startup / build estimate / simulation**, in wall-clock seconds:

- **Frontend:** Arcilator SystemVerilog-to-MLIR, or Verilator translation plus C++ executable compilation. These are different compiler stages.
- **Startup:** runtime-process launch to the unique valid BEGIN marker. Arcilator includes LLVM optimization, JIT/native compilation, loading and initialization; Verilator includes loading and initialization.
- **Build estimate:** frontend plus startup. This includes initialization and is not compiler-exclusive time. A failed run can still have a valid startup/build estimate if it reached BEGIN.
- **Simulation:** host-observed BEGIN-to-END region, including instruction-file parsing, replay stimulus, reset, DUT evaluation and captured-observation checks. It excludes Scala IPC. It is reported only when the run passes validation.
- **Cycles/s:** median of each successful process's measured region cycles divided by region wall time. Failed runs and intervals below 0.1 s have no throughput claim. A dagger marks a short but otherwise passing interval.

Startup, simulation and build estimates use medians across the completed process repetitions shown. A JIT/startup timeout without BEGIN provides no completed startup estimate; its attempted process duration remains in the raw CSV. These runs compare two-state semantics and execute one simulation thread. No scalar speedup is inferred from failed or short measurements.

The clock column is the **planned clocks per process execution**: one original replay's cycles × internal replay iterations. Successful results validate their actual clock count. Failures may stop before the planned clock count. Process repetitions and internal iterations do not increase the 51 distinct original-case count.

## All 51 cases

| # | Original case | Planned clocks (base × iterations) | Process repetitions V/A | Verilator front/start/build/sim (s) | Arcilator front/start/build/sim (s) | Cycles/s V/A | Status V/A |
|---:|---|---:|---:|---|---|---:|---|
| 1 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.attn.q.mult under RomLiteral | 13 × 1 | 1/1 | 3.771 / 0.002366 / 3.773 / 0.000452† | 0.034365 / 8.080 / 8.114 / 0.000333† | short† / short† | passed / passed |
| 2 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.attn.q.mult under RomInit | 13 × 1 | 1/1 | 3.754 / 0.002491 / 3.757 / 0.000433† | 0.033713 / 6.811 / 6.845 / 0.000460† | short† / short† | passed / passed |
| 3 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.attn.q.mult under Sram | 397 × 1 | 1/1 | 3.761 / 0.002328 / 3.763 / 0.002858† | 0.037911 / 8.125 / 8.163 / — | short† / — | passed / simulation_failed |
| 4 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.ln1.g under RomLiteral | 13 × 1 | 1/1 | 3.761 / 0.002355 / 3.764 / 0.000418† | 0.034905 / 8.127 / 8.162 / 0.000450† | short† / short† | passed / passed |
| 5 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.ln1.g under RomInit | 13 × 1 | 1/1 | 3.754 / 0.002494 / 3.757 / 0.000398† | 0.033449 / 6.766 / 6.799 / 0.000447† | short† / short† | passed / passed |
| 6 | WeightStoreEquivalenceSpec: return the committed contents of enc.0.ln1.g under Sram | 397 × 1 | 1/1 | 3.752 / 0.002357 / 3.754 / 0.002859† | 0.043512 / 8.140 / 8.183 / — | short† / — | passed / simulation_failed |
| 7 | WeightStoreEquivalenceSpec: return the committed contents of dec.emb.resmult under RomLiteral | 1,622 × 1 | 1/1 | 35.866 / 0.002284 / 35.868 / 0.053053† | 0.177 / 7.466 / 7.643 / 0.522 | short† / 3,108 | passed / passed |
| 8 | WeightStoreEquivalenceSpec: return the committed contents of dec.emb.resmult under RomInit | 1,622 × 1 | 1/1 | 3.788 / 0.019661 / 3.808 / 0.051549† | 0.036257 / 219.199 / 219.235 / 0.661 | short† / 2,455 | passed / passed |
| 9 | WeightStoreEquivalenceSpec: return the committed contents of dec.emb.resmult under Sram | 53,494 × 1 | 1/1 | 3.766 / 0.002662 / 3.768 / 0.382 | 0.038613 / 8.180 / 8.218 / — | 139,880 / — | passed / simulation_failed |
| 10 | WeightStoreEquivalenceSpec: match on a 64 Kbit window of enc.0.attn.q.w under RomLiteral | 33 × 1 | 1/1 | 3.802 / 0.002444 / 3.804 / 0.001629† | 0.041817 / 3.633 / 3.675 / 0.001768† | short† / short† | passed / passed |
| 11 | WeightStoreEquivalenceSpec: match the whole of enc.0.attn.q.w under RomInit | 577 × 1 | 1/1 | 3.786 / 0.018003 / 3.804 / 0.028738† | 0.034538 / 134.567 / 134.601 / 0.149 | short† / 3,865 | passed / passed |
| 12 | WeightStoreEquivalenceSpec: match the whole of enc.pos under RomInit | 18,001 × 1 | 1/1 | 3.754 / 0.044192 / 3.798 / 0.345 | 0.034394 / — / — / — | 52,212 / — | passed / jit_or_startup_timeout |
| 13 | WeightStoreEquivalenceSpec: decode the flat w8 address space correctly (RomInit, one layer) | 25 × 1 | 1/1 | 9.961 / 0.190 / 10.151 / 0.001155† | 0.055268 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 14 | MatmulEngineSpec: be bit-exact on rand_1_5x32x32_raw | 358 × 1 | 1/1 | 21.326 / 0.006154 / 21.332 / 0.004988† | 2.746 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 15 | MatmulEngineSpec: be bit-exact on rand_2_70x96x64_int8_dyn | 2,542 × 1 | 1/1 | 21.407 / 0.006490 / 21.413 / 0.035978† | 2.772 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 16 | MatmulEngineSpec: be bit-exact on rand_3_33x384x384_int8 | 43,799 × 1 | 1/1 | 21.273 / 0.006298 / 21.279 / 0.563 | 2.794 / — / — / — | 77,849 / — | passed / jit_or_startup_timeout |
| 17 | MatmulEngineSpec: be bit-exact on rand_4_40x1536x384_int16_dyn | 176,502 × 1 | 1/1 | 21.491 / 0.006087 / 21.497 / 2.264 | 2.746 / — / — / — | 77,961 / — | passed / jit_or_startup_timeout |
| 18 | MatmulEngineSpec: be bit-exact on rand_5_1x384x1536_int16_dyn | 153,531 × 1 | 1/1 | 21.401 / 0.006126 / 21.407 / 1.965 | 2.749 / — / — / — | 78,144 / — | passed / jit_or_startup_timeout |
| 19 | MatmulEngineSpec: be bit-exact on rand_6_3x64x64_raw_u8 | 1,209 × 1 | 1/1 | 21.421 / 0.005999 / 21.427 / 0.016337† | 2.746 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 20 | MatmulEngineSpec: be bit-exact on rand_7_64x64x96_wide | 2,534 × 1 | 1/1 | 21.487 / 0.006175 / 21.494 / 0.037566† | 2.767 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 21 | MatmulEngineSpec: be bit-exact on rand_8_130x288x384_int8_dyn | 45,762 × 1 | 1/1 | 21.475 / 0.006237 / 21.481 / 0.598 | 2.770 / — / — / — | 76,507 / — | passed / jit_or_startup_timeout |
| 22 | MatmulEngineSpec: be bit-exact on rand_9_1500x384x384_int8_dyn | 291,722 × 1 | 1/1 | 21.356 / 0.006179 / 21.362 / 3.979 | 2.762 / — / — / — | 73,324 / — | passed / jit_or_startup_timeout |
| 23 | MatmulEngineSpec: be bit-exact on rand_10_2x32x32_int8_dyn | 346 × 1 | 1/1 | 21.238 / 0.006248 / 21.244 / 0.004825† | 2.753 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 24 | MatmulEngineSpec: be bit-exact on real_conv1 | 35,262 × 1 | 1/1 | 12.368 / 0.016569 / 12.385 / 0.386 | 2.668 / — / — / — | 91,251 / — | passed / jit_or_startup_timeout |
| 25 | MatmulEngineSpec: be bit-exact on real_conv2 | 63,310 × 1 | 1/1 | 12.488 / 0.052218 / 12.540 / 0.628 | 2.686 / — / — / — | 100,867 / — | passed / jit_or_startup_timeout |
| 26 | MatmulEngineSpec: be bit-exact on real_enc0_q | 22,222 × 1 | 1/1 | 12.663 / 0.020912 / 12.684 / 0.229 | 2.699 / — / — / — | 96,854 / — | passed / jit_or_startup_timeout |
| 27 | MatmulEngineSpec: be bit-exact on real_enc0_k | 22,222 × 1 | 1/1 | 12.465 / 0.020349 / 12.485 / 0.228 | 2.697 / — / — / — | 97,298 / — | passed / jit_or_startup_timeout |
| 28 | MatmulEngineSpec: be bit-exact on real_enc0_o | 23,758 × 1 | 1/1 | 12.378 / 0.021057 / 12.399 / 0.259 | 2.677 / — / — / — | 91,585 / — | passed / jit_or_startup_timeout |
| 29 | MatmulEngineSpec: be bit-exact on real_enc0_fc1 | 89,998 × 1 | 1/1 | 12.388 / 0.069239 / 12.457 / 0.941 | 2.698 / — / — / — | 95,633 / — | passed / jit_or_startup_timeout |
| 30 | MatmulEngineSpec: be bit-exact on real_enc0_fc2 | 85,390 × 1 | 1/1 | 12.586 / 0.068387 / 12.655 / 0.857 | 2.685 / — / — / — | 99,609 / — | passed / jit_or_startup_timeout |
| 31 | MatmulEngineSpec: be bit-exact on real_dec0_xk | 22,222 × 1 | 1/1 | 12.583 / 0.021480 / 12.605 / 0.230 | 2.698 / — / — / — | 96,817 / — | passed / jit_or_startup_timeout |
| 32 | MatmulEngineSpec: be bit-exact on real_lm | 98,908 × 1 | 1/1 | 12.184 / 1.948 / 14.132 / 0.916 | 2.697 / — / — / — | 107,992 / — | passed / jit_or_startup_timeout |
| 33 | VectorUnitSpec: be bit-exact on ln_enc0 | 6,404 × 1 | 1/1 | 8.705 / 1.988 / 10.693 / 0.052200† | 0.344 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 34 | VectorUnitSpec: be bit-exact on gelu_dynq_enc0 | 14,964 × 1 | 1/1 | 8.733 / 1.986 / 10.719 / 0.137 | 0.347 / — / — / — | 109,486 / — | passed / jit_or_startup_timeout |
| 35 | VectorUnitSpec: be bit-exact on dynq_attn_enc0 | 4,884 × 1 | 1/1 | 8.699 / 1.984 / 10.683 / 0.042269† | 0.338 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 36 | VectorUnitSpec: be bit-exact on gelu_static8_conv1 | 3,804 × 1 | 1/1 | 8.928 / 1.989 / 10.916 / 0.034746† | 0.338 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 37 | VectorUnitSpec: be bit-exact on add_resid_enc0 | 4,084 × 1 | 1/1 | 8.946 / 1.984 / 10.931 / 0.045099† | 0.340 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 38 | VectorUnitSpec: be bit-exact on add_pos_enc | 3,124 × 1 | 1/1 | 8.868 / 1.987 / 10.854 / 0.033928† | 0.340 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 39 | VectorUnitSpec: be bit-exact on add_gelu_pos_enc | 3,124 × 1 | 1/1 | 8.778 / 1.986 / 10.764 / 0.034186† | 0.342 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 40 | VectorUnitSpec: be bit-exact on embed | 759 × 1 | 1/1 | 8.868 / 1.979 / 10.847 / 0.006620† | 0.335 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 41 | VectorUnitSpec: be bit-exact on add_embed_dec | 493 × 1 | 1/1 | 8.962 / 1.985 / 10.947 / 0.005403† | 0.342 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 42 | AttentionSpec: be bit-exact on enc0_full | 71,272 × 1 | 1/1 | 21.312 / 0.006641 / 21.319 / 0.988 | 2.910 / — / — / — | 72,122 / — | passed / jit_or_startup_timeout |
| 43 | AttentionSpec: be bit-exact on enc0_ragged | 57,160 × 1 | 1/1 | 21.521 / 0.006668 / 21.527 / 0.825 | 2.924 / — / — / — | 69,257 / — | passed / jit_or_startup_timeout |
| 44 | AttentionSpec: be bit-exact on dec0_self_pos41 | 2,248 × 1 | 1/1 | 21.627 / 0.006320 / 21.633 / 0.069526† | 2.930 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 45 | AttentionSpec: be bit-exact on dec0_cross_pos41 | 6,190 × 1 | 1/1 | 21.620 / 0.006372 / 21.626 / 0.219 | 2.917 / — / — / — | 28,313 / — | passed / jit_or_startup_timeout |
| 46 | KVWriteSpec: write K (transposed) and V through the engine bit-exactly | 42,361 × 1 | 1/1 | 24.427 / 0.039389 / 24.467 / 0.535 | 3.040 / — / — / — | 79,121 / — | passed / jit_or_startup_timeout |
| 47 | FrameCheckSpec: refuse zero, odd, oversized and unstreamed frame counts, and accept a valid one | 442 × 1 | 1/1 | 189.389 / 4.048 / 193.438 / 0.015628† | 4.637 / — / — / — | short† / — | passed / jit_or_startup_timeout |
| 48 | SamplerSpec: pick the lowest-index maximum with suppression, and pulse done | 15,594 × 1 | 3/3 | 5.824 / 0.002536 / 5.826 / 0.321 | 0.106 / 3.279 / 3.385 / 17.946 | 48,569 / 869 | passed / passed |
| 49 | RomLiteralLayerSpec: be bit-exact on real_enc0_q | 22,222 × 1 | 1/1 | 21.933 / 0.004971 / 21.938 / 0.230 | 2.914 / — / — / — | 96,623 / — | passed / jit_or_startup_timeout |
| 50 | RomLiteralLayerSpec: be bit-exact on real_enc0_fc1 | 89,998 × 1 | 1/1 | 167.882 / 0.004949 / 167.887 / 0.943 | 3.535 / — / — / — | 95,424 / — | passed / jit_or_startup_timeout |
| 51 | RomLiteralLayerSpec: be bit-exact on real_enc0_fc2 | 85,390 × 1 | 1/1 | 165.014 / 0.004997 / 165.019 / 0.861 | 3.533 / — / — / — | 99,171 / — | passed / jit_or_startup_timeout |

## Eligible timing comparisons

A ratio is shown only when both engines pass, their captured sources and cycle counts match, their original Scala reference case passed, and both measured simulation regions are at least 0.1 s. Ratio = Verilator simulation time / Arcilator simulation time; values above 1 favor Arcilator.

| Case | V / A simulation-time ratio | Observation |
|---|---:|---|
| SamplerSpec-01 | 0.01789× | Verilator is 55.89× faster in this replay region |

These eligible rows are individual replay measurements. **No suite-wide speedup is claimed**: failed, timed-out and short measurements do not supply comparable throughput, and E2E remains unmeasured.

## Failure and artifact notes

`frontend_timeout` and `compile_timeout` identify frontend/C++ build limits; `jit_or_startup_timeout` means runtime timed out before BEGIN; `simulation_timeout` means it timed out after BEGIN. `simulation_failed`/`validation_failed` do not by themselves identify a hardware defect versus a replay or simulator issue. `killed_signal_9` is not automatically labeled out-of-memory without kernel evidence.

The original Scala golden assertions passed or failed as recorded in the reference artifacts. Standalone replays check captured observations from those executions. Replay success is a separate result, not a fresh execution of Scala's golden assertions.

The [audit archive](evidence-final-001.tar.gz) was verified against its SHA-256 sidecar and per-file inventory: **1,004 files**, **208 referenced stage logs**, and **0 omissions**. See the [verification result](evidence-verification.json). Archive SHA-256: `ea3037caa329b276297759c92f5d0495bf135eab1dc0da84928bc1d18b2f1cb0`.

Supporting files: [numeric comparison CSV](COMPARISON.csv), [all simulator-stage metrics](COMPARISON_RAW.csv), [coverage CSV](COMPARISON_COVERAGE.csv), [complete report data](COMPARISON.json), [unit raw records](unit-results.json), [Sampler raw records](sampler-results.json), [unit provenance](unit-provenance.json), and [Sampler provenance](sampler-provenance.json). The prior `CURRENT_COMPARISON.*` snapshot is preserved.
