#!/bin/bash
# ROPE-BOUND's mechanism: bound_cause.c with its row's heads 1, 2 and 4 (its batch 4, positions T256, one thread),
# clang 18 and clang 23. The vector loop is entered past an overlap test clang takes once for all of a row's heads;
# with one head a row the halves cannot interleave across heads and every form should run alike. After
# measure_table2.sh, whose sources it builds against.
set -o pipefail
S=<SCRATCH>
C=$S/rms/table; O=$C/final; T=$S/rms/tools
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
mkdir -p $O/bin
sed -e 's/size_t b=4,t,h=4,dim=32,n,i;/size_t b=4,t,h=HEADS,dim=32,n,i;/' $T/bound_cause.c > $O/bin/bound_heads.c
grep -q "h=HEADS" $O/bin/bound_heads.c || { echo "the heads did not become a parameter"; exit 1; }
: > $O/bound-heads.jsonl
for cc in clang <CLANG23>; do
  tag=$(basename $cc)
  for hh in 1 2 4; do
    $cc $CF -DHEADS=$hh -I $C/srcB $O/bin/bound_heads.c $C/srcB/bcir_tensor.c -o $O/bin/bound_heads_${tag}_$hh -lm || exit 1
    $O/bin/bound_heads_${tag}_$hh 1 256 100 9 | sed "s/^{/{\"compiler\":\"$tag\",\"heads\":$hh,/" >> $O/bound-heads.jsonl || echo "bound_heads $tag $hh FAILED"
  done
done
cat $O/bound-heads.jsonl
