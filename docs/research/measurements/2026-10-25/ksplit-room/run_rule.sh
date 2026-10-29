#!/bin/bash
# KSPLIT-ROWS-0's GEMM rows: wadj_ab.py on HEAD's unit (ks4) and the prototype's (rows128, rows256), split vs nest
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python; W=$R/docs/research/measurements/2026-10-25/ksplit/wadj_ab.py; O=$S/next4/out
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
CORPUS="128,128,2048;64,128,2048;128,344,2048;344,128,2048;1024,128,2048"
GEO="128,128,2048;128,256,2048;256,128,2048;260,128,2048;128,128,4096;260,128,4096;128,128,8192;256,128,8192;128,256,8192;260,128,8192"
DEC="1024,384,4096;1024,384,8192;384,1024,8192;384,384,8192;768,384,8192"
: > $O/rule.jsonl
for rep in 1 2; do for cc in clang gcc; do for th in 4 1; do for lib in ks4 rule; do
  for set in CORPUS GEO DEC; do
    shapes=${!set}
    (cd $S/next4/cwd && $PY -I $W $S/next4/bin/lib_${cc}_$lib.so $th "$shapes") | \
      sed "s/^{/{\"rep\":$rep,\"compiler\":\"$cc\",\"lib\":\"$lib\",\"set\":\"$set\",/" >> $O/rule.jsonl || echo "FAILED $cc $th $lib $set"
  done
done; done; done; done
echo done
