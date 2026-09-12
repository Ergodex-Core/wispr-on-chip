# Arcilator / Verilator comparison

Snapshot recorded on 2026-09-12 at 14:48 UTC, against source commit
`42847240bad0d496b47093e0a4f7f1a706d447b8`.

**The batch is still running. This is an interim report.**
The [full 51-case table](CURRENT_COMPARISON.md) and
[comparison CSV](CURRENT_COMPARISON.csv) distinguish completed measurements
from pending cases.

Read [the methodology review](METHODOLOGY_REVIEW.md) for the exact DUT scope,
compiler-optimization caveats and distinction between replay execution and
native hardware-model throughput.

| Validation | Passed | Other outcomes |
|---|---:|---|
| Original Scala hardware tests with Verilator | 51/51 | None |
| Python golden-model tests | 18/18 | None |
| Paired Verilator replays | 21/51 | 30 pending |
| Paired Arcilator replays | 9/51 | 3 readback mismatches, 9 JIT/startup timeouts, 30 pending |

Completed pairs cover all 13 weight-store cases, the first seven Matmul cases,
and Sampler. The separately invoked 34 end-to-end clip executions have inputs
and golden outputs prepared, but have no paired simulation results here.

## Measurement setup

- Same EC2 `m7i.xlarge`: 4 vCPUs, 16 GiB RAM, Intel Xeon Platinum 8488C.
- Supplied Arcilator/CIRCT binary `04824a080`, LLVM `23.0.0git`.
- Arcilator uses `ARC_DESEQ_DISABLE=1`, the supplied package's validated
  configuration with clock promotion disabled.
- Verilator `5.020`, GCC 13; two native build jobs, one simulation thread.
- Both engines use two-state semantics and the same generated SystemVerilog
  replay. Engines run sequentially, with fresh builds and no waveform output.
- The runner applies a 300-second limit to each compiler/runtime stage.
  Sampler uses three process repetitions; other cases use one.

The original Scala assertions ran against golden vectors with Verilator.
Standalone replays reproduce the recorded input protocol and check the
captured observations. The replay clock and explicit zero-time evaluations
use a compensated 1 ps settling interval in both engines. Original generated
source defines, including `VERILATOR`, are preserved. A successful replay is
evidence that the engine reproduces those observations under that stimulus.

## Interpreting the table

Each timing cell is **build plus startup / simulation**, in wall-clock seconds.
Arcilator's build estimate includes its SystemVerilog frontend, LLVM/JIT
compilation, loading and initialization. Verilator's estimate includes its
translation, C++ compilation, loading and initialization. These are estimates
of preparation time, not compiler-exclusive measurements.

Simulation time spans flushed BEGIN/END markers and includes replay-file
parsing, stimulus, reset, DUT execution and checks. Cycles per second is the
measured region's cycle count divided by its wall time. This is not pure DUT
evaluation throughput. Intervals below 0.1 seconds are marked with a dagger;
their throughput is withheld. The clock column specifies the planned cycles
per execution. Passing runs validate that count; failed runs may stop earlier.

Timeouts before BEGIN have no completed simulation measurement. Their
`≥300` entry describes the attempted JIT/startup duration, not a measured
completed build. The three SRAM cases failed readback checks; these results
do not establish whether the cause is simulator lowering, scheduling or the
replay interface.

Sampler is currently the only passing pair with both simulation intervals
above 0.1 seconds: Verilator simulated 15,594 cycles in 0.321 s versus
Arcilator's 17.946 s (approximately 55.9 times faster for this replay).
Arcilator's build-plus-startup estimate was 3.385 s versus 5.826 s.
No suite-wide performance ratio is claimed.

## Evidence

- [Original hardware reference report](reference-report.json),
  [separately captured Sampler JUnit](TEST-whisper.SamplerSpec.xml), and
  [Python JUnit](python.xml).
- [Unit raw stage/process records](unit-results.json) and
  [unit summary](unit-summary.json).
- [Sampler raw stage/process records](sampler-results.json) and
  [Sampler summary](sampler-summary.json).
- [Unit provenance](unit-provenance.json) and
  [Sampler provenance](sampler-provenance.json), including tool versions,
  CPU metadata and source SHA-256 hashes.
- [All-case replay manifest](replay-manifest.json) and the exact
  [bulk manifest](bulk-manifest.json) referenced by the unit provenance.
- SRAM mismatch logs: [case 03](logs/weight-sram-03.log),
  [case 06](logs/weight-sram-06.log), and [case 09](logs/weight-sram-09.log).

Raw records retain the executed commands and original EC2 filesystem paths.
Those paths identify the measurement environment; they are not links to
files in this Git checkout. Generated RTL, replay payloads, model binaries
and compiler products are not part of this results snapshot.
