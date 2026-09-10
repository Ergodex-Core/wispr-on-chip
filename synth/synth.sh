#!/bin/bash
# usage: synth.sh TOP [extra yosys opts]; generic ASAP7 synthesis of one firtool-generated module with memories blackboxed
TOP=$1
SRC=${SRC:-/home/user/synth/rtl}
LIB=/home/user/asap7/asap7_merged_TT.lib
OUT=/home/user/synth/${OUTNAME:-$TOP}; [ -e $OUT/stat.txt ] && { echo "skip $TOP (done)"; exit 0; }; [ -e $OUT/running ] && { echo "skip $TOP (running)"; exit 0; }; mkdir -p $OUT; touch $OUT/running; trap 'rm -f $OUT/running' EXIT
MEMS=$(ls $SRC | grep -E "^(mem_|accMem|oMem|pMem|ram_|rowBuf|scoreMem)" | sed "s|^|$SRC/|" | tr '\n' ' ')
LOGIC=$(ls $SRC | grep -v -E "^(mem_|accMem|oMem|pMem|ram_|rowBuf|scoreMem|verification|layers)" | grep "\.sv$" | sed "s|^|$SRC/|" | tr '\n' ' ')
# big netlists: ABC's old "map" runs out of memory -> use the &nf mapper; otherwise the fast delay-driven map
case $TOP in SystolicArray*|MatmulEngine|VectorUnit) ABC="strash;&get -n;&nf {D};&put;topo;buffer;upsize {D};dnsize {D};topo;stime -p";;
  *) ABC="strash;dretime;map {D};topo;buffer;upsize {D};dnsize {D};topo;stime -p";; esac
cat > $OUT/synth.ys <<EOS
read_verilog -lib -sv $MEMS
read_verilog -sv -defer $LOGIC
hierarchy -check -top $TOP
synth -top $TOP -flatten -noabc
opt -full
dfflibmap -liberty $LIB
abc -D 1000 -liberty $LIB -script "+$ABC"
opt_clean -purge
tee -o $OUT/stat.txt stat -liberty $LIB
write_verilog -noattr $OUT/netlist.v
EOS
SECONDS=0; yosys -q -l $OUT/yosys.log $OUT/synth.ys > $OUT/stdout.txt 2>&1
echo "RC=$? TOP=$TOP"
grep -E "Chip area|Number of cells|\\\$_" $OUT/stat.txt | head -5
grep -a -E "Delay =" $OUT/yosys.log | tail -2
echo "wall ${SECONDS}s"
