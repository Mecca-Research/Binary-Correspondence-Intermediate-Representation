#!/bin/bash
# BWD-T's kernels on cache-resident operands, interleaved: eight repeats, each a pass over every configuration.
# usage: run_micro.sh SRC_DIR OUT_DIR   (SRC_DIR holds dp_micro.c and tr_micro.c)
SRC=$1; O=$2; mkdir -p $O/bin
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
for cc in clang gcc; do
  $cc $CF $SRC/dp_micro.c -o $O/bin/dp_micro_$cc -lm || exit 1
  $cc $CF $SRC/tr_micro.c -o $O/bin/tr_micro_$cc -lm || exit 1
done
: > $O/micro.txt
for rep in 1 2 3 4 5 6 7 8; do
  for cc in clang gcc; do
    for ldm in 64 1024 2048 2064; do echo "rep $rep $cc tr $($O/bin/tr_micro_$cc $ldm)" >> $O/micro.txt; done
    for dl in "32 64" "32 2048" "32 2064" "64 64" "64 2048"; do echo "rep $rep $cc dp $($O/bin/dp_micro_$cc $dl)" >> $O/micro.txt; done
  done
done
wc -l < $O/micro.txt
