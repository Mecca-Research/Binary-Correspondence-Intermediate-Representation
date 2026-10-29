#!/bin/bash
# The records' two GCC readings that stand apart from their neighbours, run again: rope.json's table rotation on four
# threads at T256 (8.94 us on the first commit's unit, 13.04 on the tree after's, where GCC's other rows read
# x0.82-1.04) and bound_cause.c's first form on one thread at T256 (26.63 us, the other four forms 20.77-21.73). The
# binaries measure_table2.sh built, with its arguments: rope_lent_bench on the first commit's unit (F) and the tree
# after's (B), four threads, T256 to T2048, and bound_cause on one thread and four at T256 and T1024; six runs, the
# two units' order alternating from run to run. After measure_table2.sh.
set -o pipefail
S=<SCRATCH>
C=$S/rms/table; O=$C/final
: > $O/recheck.jsonl
for k in 1 2 3 4 5 6; do
  if [ $((k % 2)) = 1 ]; then order="F B"; else order="B F"; fi
  for tt in 256 512 1024 2048; do
    for side in $order; do
      $O/bin/rope_lent_bench_gcc_$side 4 4 $tt 4 32 100 9 | sed "s/^{/{\"what\":\"rope\",\"compiler\":\"gcc\",\"unit\":\"$side\",\"run\":$k,/" >> $O/recheck.jsonl || echo "rope_lent_bench gcc $side T$tt FAILED"
    done
  done
  for tt in 256 1024; do
    for th in 1 4; do
      $O/bin/bound_cause_gcc $th $tt 100 9 | sed "s/^{/{\"what\":\"cause\",\"compiler\":\"gcc\",\"run\":$k,/" >> $O/recheck.jsonl || echo "bound_cause gcc $th T$tt FAILED"
    done
  done
done
cat $O/recheck.jsonl
