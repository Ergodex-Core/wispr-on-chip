# What this comparison measures

This comparison measures the supplied Arcilator package and Verilator executing
the same standalone SystemVerilog replay. It is useful for replay compatibility,
time to prepare a simulator, and total replay execution time. It does **not**
isolate native hardware-model evaluation speed or establish a general ranking
of the two simulators.

## Hardware included in the original cases

At source commit `42847240bad0d496b47093e0a4f7f1a706d447b8`, 50 of the 51
hardware cases already instantiate components or subsystems. Only the frame
check instantiates the full `WhisperTop`.

| Original suite | Cases | Instantiated hardware |
|---|---:|---|
| WeightStoreEquivalenceSpec | 13 | Individual WeightBank, or a WeightSpace for the decoder case |
| MatmulEngineSpec | 19 | MatmulEngine, selected WeightStore, two activation banks, row-factor table and output memory |
| VectorUnitSpec | 9 | VectorUnit, selected WeightStore, three activation banks and row-factor table |
| AttentionSpec | 4 | Attention, MatmulEngine, KVCache and two activation banks; no WeightStore |
| KVWriteSpec | 1 | MatmulEngine, WeightStore, KVCache, activation bank and row-factor table |
| SamplerSpec | 1 | Sampler directly |
| RomLiteralLayerSpec | 3 | MatmulTestbench with selected literal-ROM tensors |
| FrameCheckSpec | 1 | Full WhisperTop |

Source examples: [MatmulTestbench](../../../src/test/scala/whisper/MatmulTestbench.scala),
[VectorTestbench](../../../src/test/scala/whisper/VectorTestbench.scala),
[AttentionTestbench](../../../src/test/scala/whisper/AttentionTestbench.scala),
[SamplerSpec](../../../src/test/scala/whisper/SamplerSpec.scala), and
[FrameCheckSpec](../../../src/test/scala/whisper/FrameCheckSpec.scala).

There is still room to reduce some component wrappers. Random Matmul cases
select all `enc.0` tensors, then exercise one weight bank and two parameter
banks. Vector cases share a tensor selection containing layer-normalization,
position, embedding and `dec.lm.w` data, even when a particular operation needs
fewer tensors. The `dec.lm.w` bank alone has 77,808 words of 2,048 bits, about
19 MiB of logical storage. The selection also names `enc.0.gelu`, which matches
no tensor in this revision's generated table. Real Matmul cases already select
their operation's tensor prefix. These selections are visible in
[MatmulEngineSpec](../../../src/test/scala/whisper/MatmulEngineSpec.scala) and
[VectorUnitSpec](../../../src/test/scala/whisper/VectorUnitSpec.scala), with bank
dimensions in [WeightTables](../../../src/main/scala/whisper/generated/WeightTables.scala).

This is evidence of a broader elaborated model for some cases. It does not
prove which logic either compiler removes or how much time that logic costs.
Establishing that requires inspection of generated models and profiling.

## Optimizations and testbench overhead

Verilator compiles an optimized model; eliminating unused or constant logic is
a normal compiler optimization. A performance comparison may allow both tools
to optimize while requiring equivalent inputs, observable behavior and timing.
Different internal instruction counts do not alone invalidate that comparison.
See the [Verilator project description](https://github.com/verilator/verilator)
and [unused-signal documentation](https://verilator.org/guide/latest/warnings.html).

These runs enable Verilator `--timing --assert`; they do not intentionally
disable timing or checks. Both replays use the same generated sources, defines,
stimulus and captured-observation checks. Passing runs validate the captured
clock count. That does not prove equal internal work or exhaustive equivalence.

The measured simulation region includes `$fscanf`, replay dispatch, clock
settling, reset and checks. In particular, the 55.9-times Sampler ratio is a
**replay execution ratio**, not a pure Sampler evaluation-speed result. Sampler
already contains no other SoC blocks to remove, so full-SoC isolation alone
cannot explain that particular measurement.

Arcilator runs with `ARC_DESEQ_DISABLE=1`, recorded in each runtime record's
`environment_overrides`. The supplied package's README requires that setting
for its validated configuration and says clock-promotion-enabled behavior and
some same-time JIT event-settling cases remain under investigation. This
configuration must not be presented as a separately validated, optimized native
cycle-stepping implementation. Removing that setting would require fresh
correctness validation before reporting performance.

## A separate DUT-throughput benchmark

A useful follow-up would measure the same smallest sufficient DUT on both
engines, using an equivalent native driver and preloaded input vectors. For
each operation, keep only required tensor banks while preserving memory
latency, handshakes, address mapping, reset behavior and checked outputs.
Changes to DUT boundaries or memory models define a new benchmark and should
be identified separately from the existing test-suite replay.

Measure vector loading, reset/setup, active simulation, verification and build
stages separately. Compile once per DUT, perform sufficiently long repeated
active runs, and preserve all correctness checks outside or explicitly within
the stated timing region. Inspect generated model sizes and profiles before
attributing differences to pruning or scheduling. A procedural testbench with
no text replay parser is a useful intermediate measurement, but still measures
testbench scheduling as well as hardware execution.

This approach could improve the comparison's focus and either engine's
performance. It does not guarantee an Arcilator speedup; the measured result
must determine the conclusion.
