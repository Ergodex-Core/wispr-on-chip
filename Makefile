# minicpm-si build entry points. All Python runs through uv; Scala through sbt (PATH must include sbt).
SHELL := /bin/bash
UV ?= uv run
SBT ?= sbt
export MINICPM_SI_DATA ?= /home/user/data-cache

.PHONY: all setup model calib weights weights-check tables golden-test vectors test test-rtl elab eval clean

setup:                      ## create venv, install python deps
	uv sync

model:                      ## fetch the bf16 checkpoint + tokenizer into $(MINICPM_SI_DATA)/minicpm5-2b
	mkdir -p $(MINICPM_SI_DATA)/minicpm5-2b
	cd $(MINICPM_SI_DATA)/minicpm5-2b && for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
	  special_tokens_map.json model.safetensors.index.json model-00000-of-00001.safetensors; do \
	  [ -f $$f ] || curl -sSL -o $$f https://huggingface.co/openbmb/MiniCPM5-2B/resolve/main/$$f; done

calib:                      ## activation statistics of the fp32 reference on data/prompts.json -> weights/calib_stats.json
	$(UV) python golden/calib.py

weights:                    ## regenerate weights/*.bin (2.5 GB) + MANIFEST.json + generated Scala tables and micro-program
	$(UV) python gen/dump_weights.py
	$(UV) python gen/emit_weights.py
	$(UV) python gen/emit_luts.py
	$(UV) python gen/emit_microcode.py

weights-check:              ## rebuild the quantised model in memory and verify every weight image's sha256 + the manifest
	$(UV) python gen/dump_weights.py --check

golden-test:                ## fixed-point op tests (no checkpoint needed)
	$(UV) pytest -q golden/tests

vectors:                    ## RTL unit-test vectors from one golden prefill (layers 0 / 20 / 41, LM head)
	$(UV) python tests/vectors/gen_all.py

test-rtl:                   ## Verilator unit tests: every unit bit-exact against the golden vectors
	$(SBT) "testOnly minicpm.WeightStoreEquivalenceSpec minicpm.MatmulEngineSpec minicpm.VectorUnitSpec minicpm.AttentionSpec minicpm.KVWriteSpec minicpm.SamplerSpec minicpm.RomLiteralLayerSpec"

elab:                       ## elaborate the whole chip (sequencer + micro-program + all units) to SystemVerilog
	$(SBT) "testOnly minicpm.TopElabSpec"

test: golden-test test-rtl elab

eval:                       ## integer golden vs fp32 reference on data/prompts.json (top-1 agreement, greedy generations)
	$(UV) python golden/eval_golden.py

clean:
	rm -rf target project/target out test_run_dir
