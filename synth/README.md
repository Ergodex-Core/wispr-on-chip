# ASAP7 logic synthesis

Pre-layout standard-cell synthesis of the compute blocks with Yosys + ABC on the ASAP7
predictive 7 nm PDK (7.5-track RVT cells, typical corner), memories black-boxed. Results are in
`docs/paper/` (Table V) and `docs/status.md`.

## Reproduce

```
# 1. libraries: fetch the ASAP7 NLDM liberty files and concatenate their cells into one library
#    (OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM: AO, INVBUF, OA, SIMPLE, SEQ, RVT TT)
#    -> /home/user/asap7/asap7_merged_TT.lib
# 2. RTL: firtool output without packed arrays (Yosys cannot parse them)
sbt "Test/runMain whisper.synth.EmitSynth /home/user/synth/rtl"       # default 8 rows/stage
sbt "Test/runMain whisper.synth.EmitSynth /home/user/synth/rtl4 4"    # 4 rows/stage variant
# 3. synthesise one block (writes stat.txt + yosys.log under /home/user/synth/<TOP>)
./synth.sh MatmulEngine
SRC=/home/user/synth/rtl4 OUTNAME=SystolicArray4 ./synth.sh SystolicArray
# 4. collect into the paper's numbers + a markdown table
python3 fill_numbers.py discussion.tex
```

Notes:
* `synth -noabc` then an explicit `abc` call: the two largest blocks need ABC's `&nf` mapper
  (the older `map` runs out of memory on ~1 M cells); everything else uses the faster `map`.
* The reported critical path is ABC's post-mapping register-to-register delay; add roughly 60 ps
  for setup plus clock-to-Q in ASAP7 before converting to a frequency. Wire load is not modelled.
