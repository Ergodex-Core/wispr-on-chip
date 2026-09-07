# whisper-si build entry points. All Python runs through uv; Scala through sbt (PATH must include sbt).
SHELL := /bin/bash
UV ?= uv run
SBT ?= sbt -batch
export WHISPER_SI_DATA ?= /home/user/data-cache

.PHONY: all setup refs refs-check varied wer weights weights-check golden test e2e e2e-long clean

setup:                      ## create venv, install python deps
	uv sync

varied:                     ## rebuild data/varied from cached corpora (deterministic)
	$(UV) python gen/make_varied.py

refs:                       ## fp32 CPU references -> data/ref (deterministic, single-thread torch per worker)
	$(UV) python golden/reference_cpu.py --set all

refs-check:                 ## regenerate refs into a temp dir and diff byte-for-byte against data/ref
	rm -rf out/refs-check && mkdir -p out/refs-check
	$(UV) python golden/reference_cpu.py --set all --out out/refs-check > /dev/null
	diff -r out/refs-check data/ref && echo "refs byte-identical"

wer:                        ## WER of the fp32 references against ground truth
	$(UV) python golden/wer.py --hyp data/ref

weights:                    ## regenerate weights/*.hex + MANIFEST.json + generated Scala tables from the checkpoint
	$(UV) python gen/dump_weights.py
	$(UV) python gen/emit_weights.py
	$(UV) python gen/emit_luts.py
	$(UV) python gen/emit_microcode.py

weights-check:              ## verify committed weights match the manifest sha256s
	$(UV) python gen/dump_weights.py --check

golden:                     ## run the int golden model over the eval sets and report WER
	$(UV) python golden/run_golden.py --set testclean_200,varied
	$(UV) python golden/wer.py --hyp out/golden_int

test:                       ## python unit tests + scala/verilator unit tests
	$(UV) pytest -q
	$(SBT) test

e2e:                        ## default Verilator e2e set (short clips)
	$(UV) python tests/e2e/run_e2e.py --set default

e2e-long:                   ## long e2e set (29 s clip + 20-utterance RTL set)
	$(UV) python tests/e2e/run_e2e.py --set long

all: weights-check test e2e

clean:
	rm -rf target project/target out test_run_dir
