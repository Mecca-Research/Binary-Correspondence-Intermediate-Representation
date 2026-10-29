#!/bin/bash
# KSPLIT's records on the committed tree (HEAD) against the tree before it (HEAD~1), serially, two workers at
# most:
#  0 sizes   -- ws_sizes.c: the state's workspace and the largest group's partials at each shape below; and
#               split_probe.py's copy of HEAD's unit under step_ab (two rounds): every weight adjoint the split rule
#               weighs in the step, its partials beside what its job's room had left, split or kept
#  1 values  -- the C harness on each tree and compiler: HEAD's output the tree before's line for line but the new
#               KSPLIT line; every timed arm below holds its weights or outputs equal byte for byte
#  2 gemm    -- wadj_ab.py on HEAD's unit: the weight adjoint (C = A^T B) lent a room for its partials against lent
#               none, the split's bytes the nest's, at the GEO step's shapes over 2,048 to 8,192 rows and the 9.84M
#               decoder's over 1,024 to 8,192, one thread and four, clang and gcc; torch.mm (MKL) beside them
#  3 parts   -- ../../2026-10-24/gp-lanes/probe_jobs.py and probe_step.py on each tree: the GEO step's pool jobs a
#               worker at a time, T1024 and T2048 (the contexts whose state lends a workspace)
#  4 step    -- ../../2026-10-22/attn-rate/step_ab.c: the training step, both trees in one process with an A/A
#               control, the GEO dimensions T512 to T2048 and the 9.84M decoder T128 to T1024, clang and gcc
#  5 forward -- ../../2026-10-24/pool-affinity/fwd_ab.c: the inference forward the same way (no weight adjoint:
#               the entries' new shape alone), B1 T256 and B4 T1024
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next3/meas; O=$C/final; T=$R/docs/research/measurements/2026-10-25/ksplit; D=$R/docs/research/measurements
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
H=$(git rev-parse HEAD); P=$(git rev-parse HEAD~1)
echo "== $(date -u +%H:%M:%S) tree=$H before=$P dirty=$(git status --short | wc -l)"
rm -rf $C; mkdir -p $C/srcA $C/srcB $O/bin $O/cwd
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h test_decoder_train.c; do
  git show $P:runtime/c/$f > $C/srcA/$f; git show $H:runtime/c/$f > $C/srcB/$f
done
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
echo "== $(date -u +%H:%M:%S) 0 sizes: the workspace and the partials, and the calls the split rule weighs"
clang $CF -I $C/srcB $T/ws_sizes.c -o $O/bin/ws_sizes -ldl || exit 1
$O/bin/ws_sizes $O/bin/libdec_clang_B.so | tee $O/sizes.txt
mkdir -p $C/srcS; cp $C/srcB/* $C/srcS/
python3 -I $T/split_probe.py $C/srcB/bcir_tensor.c $C/srcS/bcir_tensor.c || exit 1
clang $CF -fPIC -shared -I $C/srcS $C/srcS/bcir_tensor.c $C/srcS/bcir_decoder_train.c -o $O/bin/libdec_split.so -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $D/2026-10-22/attn-rate/step_ab.c -o $O/bin/step_ab -ldl -lm || exit 1
: > $O/splits.txt
for args in "260 128 4 4 3 256 4 512 14 4 2" "260 128 4 4 3 256 4 1024 14 4 2" "260 128 4 4 3 256 4 2048 14 4 2" \
            "260 384 8 4 6 1024 8 128 46 4 2" "260 384 8 4 6 1024 8 512 46 4 2" "260 384 8 4 6 1024 8 1024 46 4 2"; do
  echo "== step_ab $args" >> $O/splits.txt
  $O/bin/step_ab $O/bin/libdec_split.so $O/bin/libdec_split.so $args 2>&1 | grep "^KSPLIT" | sort | uniq -c >> $O/splits.txt
done
cat $O/splits.txt
echo "== $(date -u +%H:%M:%S) 1 values: the C harness on each tree and compiler"
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc -std=c2x -O3 -ffp-contract=off -fno-math-errno -pthread -I $src $src/test_decoder_train.c $src/bcir_tensor.c \
    $src/bcir_decoder_train.c -o $O/bin/harness_${cc}_$side -lm || exit 1
  $O/bin/harness_${cc}_$side > $O/harness-$cc-$side.txt 2>&1; echo "harness $cc $side exit $?"
done
  if diff <(grep -v "(KSPLIT)" $O/harness-$cc-B.txt) $O/harness-$cc-A.txt > /dev/null; then echo "harness $cc: the same output but the new line"; else echo "harness $cc: OUTPUT DIFFERS"; fi
  grep "(KSPLIT)" $O/harness-$cc-B.txt
done
echo "== $(date -u +%H:%M:%S) 2 the weight adjoint alone, HEAD's unit, lent a room and lent none"
GEO="128,128,2048;256,128,2048;128,256,2048;260,128,2048;128,128,4096;256,128,4096;128,256,4096;260,128,4096;128,128,8192;256,128,8192;128,256,8192;260,128,8192"
DEC="384,384,1024;1024,384,1024;384,1024,1024;384,384,4096;192,384,4096;1024,384,4096;384,1024,4096;384,384,8192;192,384,8192;1024,384,8192"
: > $O/gemm.jsonl
for cc in clang gcc; do for th in 4 1; do
  torch=""; [ $cc = clang ] && torch="--torch"
  for set in GEO DEC; do
    shapes=$GEO; [ $set = DEC ] && shapes=$DEC
    (cd $O/cwd && $PY -I $T/wadj_ab.py $O/bin/libdec_${cc}_B.so $th "$shapes" $torch) | \
      sed "s/^{/{\"compiler\":\"$cc\",\"set\":\"$set\",/" >> $O/gemm.jsonl || echo "wadj_ab $cc $th $set FAILED"
  done
done; done
cat $O/gemm.jsonl
echo "== $(date -u +%H:%M:%S) 3 the GEO step's pool jobs a worker at a time, each tree"
for side in A B; do
  rev=$P; [ $side = B ] && rev=$H
  git worktree add -q --detach $C/wt-$side $rev || exit 1
  python3 -I $D/2026-10-24/gp-lanes/probe_jobs.py $C/src$side $C/wt-$side/runtime/c > /dev/null || exit 1
done
for tt in 1024 2048; do for side in A B; do
  (cd $O/cwd && $PY -I $D/2026-10-24/gp-lanes/probe_step.py $C/wt-$side $tt kblock+exp+reduce 12 $O/probe-$side-T$tt.json 4 > $O/probe-$side-T$tt.txt 2>&1); echo "probe $side T$tt exit $?"
done; done
for side in A B; do git worktree remove --force $C/wt-$side; done; git worktree prune
echo "== $(date -u +%H:%M:%S) 4 the training step in one process"
clang -std=c11 -O2 -I $C/srcB $D/2026-10-24/pool-affinity/fwd_ab.c -o $O/bin/fwd_ab -ldl -lm || exit 1
: > $O/ab-step.jsonl
for spec in "clang|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T1024 kblock+exp+reduce, step|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 24" \
            "gcc|geo-dims T1024 kblock+exp+reduce, step|260 128 4 4 3 256 4 1024 14 4 48" \
            "gcc|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 24" \
            "clang|geo-dims T1024 v1 (no split: v1's GEMM), step|260 128 4 4 3 256 4 1024 0 4 24" \
            "clang|decoder-9.84M kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 128 46 4 48" \
            "clang|decoder-9.84M T512 kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 512 46 4 16" \
            "clang|decoder-9.84M T1024 kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 1024 46 4 12"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 5 the inference forward in one process"
: > $O/ab-forward.jsonl
for spec in "clang|geo B1 T256 forward|260 128 4 4 3 256 1 256 14 4 64" \
            "clang|geo B4 T1024 forward|260 128 4 4 3 256 4 1024 14 4 32"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/fwd_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-forward.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) done"
