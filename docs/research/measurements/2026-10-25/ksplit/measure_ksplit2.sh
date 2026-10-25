#!/bin/bash
# KSPLIT's records' second run, after measure_ksplit.sh (its libraries and sources), serially, two workers at most:
#  0 splits -- need_ab.py's copy of step_ab (two rounds) over split_probe.py's library: every weight adjoint the
#              split rule weighs in the step lent the workspace the Python trainer lends -- the exact need at four
#              threads (native.memory_plan: bcir_decoder_workspace_need) -- where step_ab lends the workspace for
#              any pool (bcir_decoder_workspace_bytes)
#  1 step   -- the same copy between the two trees: the GEO dimensions T1024 and T2048 and the 9.84M decoder T1024,
#              clang, beside the A/A control, every round's weights the same bytes
#  2 again  -- step_ab at T2048 again, 64 rounds, clang and gcc twice each: measure_ksplit.sh's 24 rounds there
#              swung by a tenth round to round and its clang A/A control excluded 1
#  3 one    -- step_ab on one thread (the workspace for any pool, so every weight adjoint four K blocks deep splits):
#              the GEO dimensions T1024 and the 9.84M decoder T1024, clang and gcc -- the one thread where
#              measure_ksplit.sh's GEMM rows found the 9.84M decoder's widest weight adjoint (1,024 x 384) slower
#  4 recheck -- wadj_ab.py on one thread at that weight adjoint and its neighbours, three runs, clang and gcc
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next3/meas; O=$C/final; T=$R/docs/research/measurements/2026-10-25/ksplit; D=$R/docs/research/measurements
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
echo "== $(date -u +%H:%M:%S) tree=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
python3 -I $T/need_ab.py $D/2026-10-22/attn-rate/step_ab.c $C/step_need_ab.c || exit 1
clang -std=c11 -O2 -I $C/srcB $C/step_need_ab.c -o $O/bin/step_need_ab -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) 0 splits at the need"
: > $O/splits-need.txt
for args in "260 128 4 4 3 256 4 1024 14 4 2" "260 128 4 4 3 256 4 2048 14 4 2" "260 384 8 4 6 1024 8 1024 46 4 2"; do
  echo "== step_need_ab $args" >> $O/splits-need.txt
  $O/bin/step_need_ab $O/bin/libdec_split.so $O/bin/libdec_split.so $args 2>&1 | grep "^KSPLIT" | sort | uniq -c >> $O/splits-need.txt
done
cat $O/splits-need.txt
echo "== $(date -u +%H:%M:%S) 1 the training step at the need"
: > $O/ab-step-need.jsonl
for spec in "clang|geo-dims T1024 kblock+exp+reduce, step at the need|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims T2048 kblock+exp+reduce, step at the need|260 128 4 4 3 256 4 2048 14 4 64" \
            "clang|decoder-9.84M T1024 kblock+exp+reduce+attn, step at the need|260 384 8 4 6 1024 8 1024 46 4 16"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_need_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step-need.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 2 the training step at T2048 again, 64 rounds"
: > $O/ab-step-again.jsonl
for spec in "clang|geo-dims T2048 kblock+exp+reduce, step, run 2|260 128 4 4 3 256 4 2048 14 4 64" \
            "gcc|geo-dims T2048 kblock+exp+reduce, step, run 2|260 128 4 4 3 256 4 2048 14 4 64" \
            "clang|geo-dims T2048 kblock+exp+reduce, step, run 3|260 128 4 4 3 256 4 2048 14 4 64" \
            "gcc|geo-dims T2048 kblock+exp+reduce, step, run 3|260 128 4 4 3 256 4 2048 14 4 64"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step-again.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 3 the training step on one thread"
: > $O/ab-step-one.jsonl
for spec in "clang|geo-dims T1024 kblock+exp+reduce, step on one thread|260 128 4 4 3 256 4 1024 14 1 24" \
            "gcc|geo-dims T1024 kblock+exp+reduce, step on one thread|260 128 4 4 3 256 4 1024 14 1 24" \
            "clang|decoder-9.84M T1024 kblock+exp+reduce+attn, step on one thread|260 384 8 4 6 1024 8 1024 46 1 12" \
            "gcc|decoder-9.84M T1024 kblock+exp+reduce+attn, step on one thread|260 384 8 4 6 1024 8 1024 46 1 12"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step-one.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 4 the widest weight adjoint on one thread, three runs"
: > $O/gemm-recheck.jsonl
for run in 1 2 3; do for cc in clang gcc; do
  (cd $O/cwd && $PY -I $T/wadj_ab.py $O/bin/libdec_${cc}_B.so 1 "1024,384,4096;1024,384,8192;384,1024,8192;384,384,8192") | \
    sed "s/^{/{\"compiler\":\"$cc\",\"run\":$run,/" >> $O/gemm-recheck.jsonl || echo "wadj_ab $cc run $run FAILED"
done; done
cat $O/gemm-recheck.jsonl
echo "== $(date -u +%H:%M:%S) done"
