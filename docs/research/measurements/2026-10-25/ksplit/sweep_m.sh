#!/bin/bash
# KSPLIT's records' third run, after measure_ksplit.sh and measure_ksplit2.sh (their libraries), serially: the weight
# adjoint's width m at the 9.84M decoder's other width (n 384) and 8,192 rows -- 256 to 1,024, where a K block's panel
# of A (448 x m floats) grows from 0.44 to 1.75 MiB against a core's 2 MiB of L2 -- wadj_ab.py on one thread and four,
# clang and gcc, two runs each, torch.mm beside them under clang
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next3/meas; O=$C/final; T=$R/docs/research/measurements/2026-10-25/ksplit
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
echo "== $(date -u +%H:%M:%S) tree=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
SHAPES="256,384,8192;384,384,8192;512,384,8192;640,384,8192;768,384,8192;896,384,8192;1024,384,8192"
: > $O/gemm-m.jsonl
for run in 1 2; do for th in 1 4; do for cc in clang gcc; do
  torch=""; [ $cc = clang ] && torch="--torch"
  (cd $O/cwd && $PY -I $T/wadj_ab.py $O/bin/libdec_${cc}_B.so $th "$SHAPES" $torch) | \
    sed "s/^{/{\"compiler\":\"$cc\",\"run\":$run,/" >> $O/gemm-m.jsonl || echo "wadj_ab $cc $th run $run FAILED"
done; done; done
cat $O/gemm-m.jsonl
echo "== $(date -u +%H:%M:%S) done"
