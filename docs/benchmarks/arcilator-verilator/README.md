# Arcilator / Verilator comparison

All **51 planned unit-test replay pairs are complete** (102 simulator records),
against source commit `42847240bad0d496b47093e0a4f7f1a706d447b8`.

Read the [final 51-case report](COMPARISON.md), [comparison CSV](COMPARISON.csv),
and [methodology review](METHODOLOGY_REVIEW.md). Completion includes failures
and timeouts; it does not mean every case passed.

| Validation | Passed | Other outcomes |
|---|---:|---|
| Original Scala hardware tests with Verilator | 51/51 | None |
| Python golden-model tests | 18/18 | None |
| Verilator standalone replays | 51/51 | None |
| Arcilator standalone replays | 9/51 | 3 SRAM readback failures; 39 JIT/startup timeouts |

Every Arcilator timeout occurred before the BEGIN marker, at the 300-second
runtime-stage limit. The three SRAM cases reached simulation and failed
readback checks. Those failures do not establish whether their cause is
simulator lowering, scheduling or the replay interface. The completed batch
[exited 1](completed.exit) because some cases did not pass.

The 34 named E2E clip executions (24 unique clips) have no paired simulation
results in this benchmark. These unit-replay results make no transcription
claim. Python tests are separate software validation and have no hardware
cycles-per-second metric.

## Measurement setup and limits

- Same EC2 `m7i.xlarge`: 4 vCPUs, 16 GiB RAM, Intel Xeon Platinum 8488C.
- Supplied Arcilator/CIRCT `04824a080`, LLVM `23.0.0git`; Verilator `5.020`.
- Both engines use two-state semantics, equivalent generated SystemVerilog
  replays and one simulation thread. Engines ran sequentially with fresh
  frontend/build outputs and no waveform output.
- Verilator used two C++ build jobs. Arcilator used its default internal
  compiler threading, with no explicit compiler job/thread cap.
  `OMP_NUM_THREADS=1` does not cap every MLIR/LLVM compiler thread; this was not
  an equal-thread controlled compilation comparison.
- Arcilator used **`ARC_DESEQ_DISABLE=1`**, the supplied package's validated
  mode with clock promotion disabled. The runner applied a 300-second limit
  to each frontend/build or runtime stage.
- Sampler used three process repetitions; the other cases used one. Repeats
  and internal replay iterations do not increase the 51 distinct-case count.

Fifty cases already instantiate components or subsystems; only FrameCheck
instantiates the full WhisperTop. Some wrappers include broad weight
selections. The [methodology review](METHODOLOGY_REVIEW.md) identifies the
hardware and explains why these results do not prove which logic either
compiler pruned.

The final table separates frontend, startup, build estimate and simulation
wall time. Arcilator startup includes LLVM/JIT, loading and initialization;
Verilator's frontend/build includes translation and C++ compilation. Their
build estimates are not compiler-exclusive measurements. A valid BEGIN still
provides startup timing for a later failed run.

Simulation timing includes text replay parsing, dispatch, settling, reset and
checks. The original Scala assertions passed against golden vectors; the
standalone replays check captured observations from those executions. Original
source defines, including `VERILATOR`, are preserved. The compensated 1 ps
settling interval is the same for both engines.

Failed measurements and intervals below 0.1 seconds have no throughput claim.
Sampler is the only pair with two passing, sufficiently long intervals:
Verilator ran 15,594 replay cycles in 0.321 s versus Arcilator's 17.946 s,
approximately 55.9 times faster for this replay. This is not pure Sampler
hardware-model evaluation speed. **No suite-wide speedup is claimed.**

## Final evidence

- [All 51 numeric comparisons](COMPARISON.csv), [102 timing records](COMPARISON_RAW.csv),
  [coverage](COMPARISON_COVERAGE.csv), and [complete report data](COMPARISON.json).
- [Original hardware reference](reference-report.json),
  [Sampler JUnit](TEST-whisper.SamplerSpec.xml), and [Python JUnit](python.xml).
- [Unit raw records](unit-results.json), [unit summary](unit-summary.json),
  [Sampler raw records](sampler-results.json), and [Sampler summary](sampler-summary.json).
- [Unit provenance](unit-provenance.json), [Sampler provenance](sampler-provenance.json),
  [all-case manifest](replay-manifest.json), and [exact bulk manifest](bulk-manifest.json).
- [Verified audit archive](evidence-final-001.tar.gz),
  [SHA-256 sidecar](evidence-final-001.tar.gz.sha256),
  [per-file inventory](evidence-final-001.tar.gz.inventory.json), and
  [verification result](evidence-verification.json).
- SRAM readback logs: [case 03](logs/weight-sram-03.log),
  [case 06](logs/weight-sram-06.log), [case 09](logs/weight-sram-09.log).

The archive contains **1,004 files**, including **208 referenced stage logs**,
with **zero omissions**. Its SHA-256 is
`ea3037caa329b276297759c92f5d0495bf135eab1dc0da84928bc1d18b2f1cb0`.
Archive integrity, every included file's hash/size and all 51 paired cases were
verified. Final raw records, summaries and provenances exactly match that
archive. It excludes generated RTL, replay payloads, models and native compiler
products. Recorded filesystem paths identify the original measurement or
verification environment, rather than files included at those paths in Git.

## Preserved interim snapshot

[CURRENT_COMPARISON.md](CURRENT_COMPARISON.md) and
[CURRENT_COMPARISON.csv](CURRENT_COMPARISON.csv) retain the **35-pair snapshot
from 2026-09-12 16:05 UTC**. Its original supporting records remain in
[commit fbd9476](https://github.com/Ergodex-Core/wispr-on-chip/tree/fbd94760baf816be36e40af5343ecfcbc6eb1ad2/docs/benchmarks/arcilator-verilator).
The raw files at this branch's current revision describe the final 51-pair
batch and should not be interpreted as that earlier snapshot's records.
