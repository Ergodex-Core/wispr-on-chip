# minicpm-si — MiniCPM5-2B as a Chisel accelerator generator

A from-scratch hardware implementation of [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
(a 42-layer Llama: d = 2048, 16 query / 2 KV heads of 128, FFN 6144, vocab 130560) in the style of the
whisper-si base design (branch `claude/whisper-tiny-chisel-accelerator-03do1o`): a 32×32 int8
weight-stationary matmul engine, a 16-lane vector unit (RMSNorm, RoPE, SiLU gate, dynamic quantisation,
residual adds, embedding), an attention unit with online softmax and grouped-query heads, a KV cache, a
streaming argmax sampler and a sequencer running an 848-instruction micro-program generated from the model.

Correctness is defined by an integer golden model in numpy (`golden/minicpm_int.py`); every unit of the
RTL is verified bit-exactly against it on real activations of real layers. The full-chip Verilator run is
deliberately not part of the evaluation (docs/decisions.md #8).

* `docs/numerics.md` — the frozen fixed-point specification
* `docs/tiling.md` — memory layouts and tiling
* `docs/decisions.md` — dated design decisions (what differs from whisper-si and why)
* `docs/status.md` — what was run, results, cycle counts
* `docs/report.md` — technical report

```
make setup            # uv env
make model            # fetch the bf16 checkpoint (5 GB) into $MINICPM_SI_DATA/minicpm5-2b
make calib            # fp32 activation statistics on data/prompts.json (~8 min CPU)
make weights          # 2.4 GB of int8 images + MANIFEST.json + generated Scala tables / micro-program
make golden-test      # fixed-point op tests
make vectors          # RTL unit-test vectors from one golden prefill
make test-rtl         # Verilator unit tests, every unit bit-exact vs golden
make elab             # elaborate the whole chip to SystemVerilog
make eval             # integer golden vs fp32 on the eval prompts
```
