# Wafer-scale fabric with hardwired model weights

This document describes the hardware concept that `wispr-on-chip` plans for,
and the assumptions baked into the planner. The short version: take the
"model etched into the chip" idea that Taalas demonstrated on a single
reticle-limited die, and scale it to a whole 300 mm wafer the way Cerebras
scales general-purpose compute. Whisper-class speech models are the first
workload because they are small enough that a wafer holds dozens of copies.

## 1. Why hardwire weights

Inference on a GPU is a memory-bandwidth problem: every generated token
streams the entire weight set out of HBM. Two ways out exist.

* **Keep the weights on-die in SRAM** (Cerebras, Groq). SRAM is fast but
  costs about 0.03 mm² per Mbit at N5 and leaks; a 1.5 B-parameter model at
  4 bits needs ~190 mm² of SRAM and still has to be loaded.
* **Etch the weights into the silicon** (Taalas). Each weight becomes a
  constant in a via/metal-programmed cell sitting next to the multiplier
  that consumes it. There is no weight traffic at all. The model is fixed
  at tape-out of the via layers; changing it means new via masks, not a new
  chip design.

The second approach turns a model into a *mask set*. The base layers
(transistors, SRAM, routers, clock, power) are common to every model. Only
the few via/metal layers that program the weight cells are model-specific,
which keeps the per-model NRE to a fraction of a full mask set and the
turnaround to weeks rather than a year.

## 2. Fabric anatomy

The fabric is a uniform 2-D mesh of identical tiles. Uniformity is what
makes reticle stitching and redundancy tractable.

```
+------------------------------------------------------------+
|  tile (1 mm² default)                                       |
|                                                             |
|   +--------------------------+   +---------------------+    |
|   | hardwired weight bank    |   | MAC array           |    |
|   | (via-programmed cells,   |==>| params / mux units  |    |
|   |  ~10 M x 4-bit at N5)    |   | int4 x int8 -> 24b  |    |
|   +--------------------------+   +---------------------+    |
|                                          |                  |
|   +--------------------------+   +-------v-------------+    |
|   | tile SRAM (1 MiB)        |<=>| vector unit:        |    |
|   | activations, KV cache    |   | softmax, LayerNorm, |    |
|   +--------------------------+   | GELU, reductions    |    |
|                                  +---------------------+    |
|   router: 4 neighbours + stage broadcast/reduce trees        |
+------------------------------------------------------------+
```

**Weight bank.** One access device per bit, programmed by via presence. The
planner assumes 110 Mbit/mm² usable at N5, roughly 3.5x SRAM density, because
there is no latch, no write path and the array can be wide and shallow.

**MAC array and the mux factor.** Fully spatial (one multiplier per weight)
would make MACs, not weights, dominate area: a 4x8 multiplier plus a 24-bit
accumulator is ~2,100 transistors, versus 4 bits of via cells. The
`mux_factor` sets how many hardwired weights each MAC sweeps per item. At
mux=1024 a tile holds ~10 M weights and ~10 k MACs; the whole bank is swept
once per token in 1024 cycles, about 1 µs at 1.1 GHz. Lower mux buys latency
with area; `--auto-mux` finds the lowest mux at which the model still fits.

**SRAM.** Holds working activations and, for decoder stages, the KV cache of
every stream in flight. For Whisper this, not the weights, sizes the decoder
(see section 4).

**Router.** Nearest-neighbour links plus per-stage broadcast (input vector
to every tile of the stage) and reduce (partial sums back) trees. Dead tiles
are bypassed by the router, which is the redundancy mechanism.

## 3. Mapping a model

A model is a pipeline of *stages*: the conv stem, each encoder layer, each
decoder layer, and the LM head. Each stage owns a contiguous block of
tiles; its weight matrices are striped across the block so every tile's MAC
array participates. Items (audio frames, then text tokens) flow stage to
stage.

Per item, a stage costs

```
cycles = ceil(linear MACs / MACs in stage)      # weight sweep
       + attention MACs / MACs in stage         # QK^T and PV against the KV cache
       + hop_cycles * (2*ceil(sqrt(tiles)) + 1) # broadcast + reduce across the block
       + fixed overhead                         # LayerNorm, softmax, registers
```

Because stages are physically separate, they pipeline: the decoder admits a
new token every `max(stage cycles)` cycles as long as enough independent
streams are in flight to fill it. Single-stream latency is the sum over the
decoder path; there is no batching penalty and no batching benefit.

Encoder layers work at window granularity. Self-attention needs the whole
window's K and V before it can start, so a layer processes all 1,500 frames
of a window, then hands the window to the next layer and takes the next one.
With 32 encoder layers there are 32 windows in flight per instance.

## 4. What the planner says about Whisper

Running `wispr-on-chip plan --model whisper-large-v3` on the default N5
wafer (7x6 stitched 26x33 mm fields, 36,036 mm², 1 mm² tiles):

* **Weights are cheap.** 1.54 B weights at 4 bits occupy ~155 tiles' worth
  of via cells. A single instance would fit in about 160 mm² if weights were
  the only constraint.
* **KV cache is not.** Each decoder layer must hold, per stream, its own
  self-attention cache (448 x 1280 x 2) *and* the cross-attention K/V for the
  whole 30 s window (1,500 x 1280 x 2). At 8 bits that is ~5 MB per layer
  per stream, so decoder layers are SRAM-bound at ~20 tiles each for 4
  streams. Encoder layers hold one window of K/V and are SRAM-bound at 4
  tiles. The instance ends up at 776 tiles, of which weights fill under a
  fifth.
* **The encoder is the throughput limit.** Each encoder layer needs
  ~1 M cycles per window (1,500 frames x ~690 cycles), about 1,060 windows/s
  per instance. The decoder keeps up with only 4 streams, so the planner
  stops there by default rather than spending SRAM on idle KV caches.
* **The wafer holds 45 instances** packed as 36x22-tile blocks, and at full
  rate transcribes ~1.4 million hours of audio per hour inside an 18 kW
  envelope, at ~920x real time per stream and ~8 µs per decoded token.

Knobs that move these numbers: `--kv-bits 4` halves decoder SRAM; larger
`--sram-kib` with a larger `--tile-area` rebalances the tile toward state;
lower `--mux` trades area for latency. The sweep below compares Whisper sizes
on the same fabric:

```
wispr-on-chip sweep
```

## 5. Wafer-scale specifics

**Reticle stitching.** The fabric is the largest grid of whole 26x33 mm
reticle fields that fits inside the wafer's inscribed square (7x6 on a 300 mm
wafer with 3 mm edge exclusion). Cross-scribe wires connect neighbouring
tiles across field boundaries; the tile mesh does not know where fields end.
Pass `--fabric-side 215` to model a WSE-like 46,225 mm² fabric.

**Redundancy and yield.** Tiles fail independently with probability
`1 - exp(-D0 * A)`. At D0 = 0.1/cm² and 1 mm² tiles that is 0.1 %, so a
36k-tile wafer expects ~36 dead tiles. The planner reserves the expected
number plus four standard deviations (61 tiles) as spares, giving a
wafer-level success probability above 99.99 %. Dead tiles are mapped out by
the router and their stage slice moves to a spare; because instances are
independent, one bad tile never costs more than one instance's worth of
re-routing.

The via layers are model-specific, so a defect in a *weight cell* cannot be
repaired by swapping in a spare tile with different vias. Two mitigations
exist and the planner assumes the first: (a) place each stage's weights
with a small over-provision so any single tile in the block can be dropped
and its columns recomputed elsewhere in the block; (b) accept that a weight
defect damages one column of one matrix, which quantised transformers are
empirically tolerant of.

**Power and cooling.** Peak dynamic power is energy per window times windows
per second, plus leakage at 50 mW/mm² (1.8 kW for the wafer). The default
20 kW budget matches the class of cold-plate systems that already host
wafer-scale parts. When the fabric would exceed the budget the planner
reports the fraction of peak throughput that is sustainable and uses it for
the "sustained" figures; it does not model per-instance power gating.

**I/O.** Audio enters as log-mel spectrograms: 3,000 x 128 x 8 bits = 3 Mbit
per window. At 49k windows/s that is ~150 Gbit/s in, and a few Gbit/s of
tokens out. A handful of SerDes lanes on the wafer edge suffice; there is no
HBM, no weight loading and no inter-wafer traffic.

**Clocking.** Each instance is its own clock domain; stages within an
instance are mesochronous with FIFOs at the stage boundaries. The fabric-wide
grid clock only has to reach the routers.

## 6. Calibration against a public data point

Taalas's first hardwired chip has been reported as a ~815 mm² TSMC N6 die
running Llama 3.1 8B at 3-bit weights at roughly 17,000 tokens/s for a
single user. The planner reproduces that class of result from first
principles:

```
wispr-on-chip plan --model llama-3.1-8b --process n6 --target die \
    --weight-bits 3 --auto-mux --streams 1
```

gives mux ≈ 1,700, 776 of 784 tiles used, and ~14,000 tokens/s single-stream
at 1 GHz. The agreement is within the uncertainty of the process constants
(all of which are order-of-magnitude public estimates in `process.py`), which
is the point: the same constants then scale to the wafer.

## 7. Economics

A wafer is one unit. Its cost is the base-layer wafer (shared across every
model on the platform), the model-specific via masks (a handful of layers,
amortised over every wafer of that model), stitching, and the cold plate
and power delivery. Per instance of Whisper large-v3, at 45 instances per
wafer, the silicon cost is ~800 mm² of N5, comparable to one GPU die, while
delivering ~30,000 hours of audio per hour per instance with no HBM stack.

The trade is flexibility. A wafer serves exactly the model whose vias it
carries. That fits speech recognition well: the model changes a few times a
year, the workload is enormous and uniform, and latency per stream matters.

## 8. Limitations of the planner and next steps

* No RTL. The tile is a set of area and energy constants, not a design.
  The next milestone is a synthesisable tile (weight bank + MAC array +
  vector unit) to replace the constants with measured numbers.
* Homogeneous tiles. Encoder stages want more MACs and less SRAM than
  decoder stages. A fabric with two tile flavours, or an SRAM-rich tile
  variant on alternate rows, would raise encoder throughput at no weight
  cost.
* Decoder-only models are modelled for decode only; prompt prefill is not
  included in the throughput.
* Quantisation accuracy is out of scope. 4-bit weights and 8-bit KV are the
  defaults because they are routine for Whisper; verifying WER on a chosen
  checkpoint is a prerequisite to taping out its vias.
* Thermal modelling stops at a wafer-level power budget; hot-spot analysis
  of dense decoder blocks is future work.
