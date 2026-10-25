#!/bin/bash
# ROPE-TABLE's records on the committed tree (HEAD, ROPE-BOUND) against the tree before the slice (HEAD~2), with
# ROPE-TABLE's first commit (HEAD~1) where ROPE-BOUND's own effect is read; serially, two workers at most:
#  1 values  -- the C harness on each compiler (its ROPE-TABLE witness and every trainer witness, the output the
#               tree before's line for line but the new one, and the first commit's byte for byte); every timed arm
#               below holds its weights or logits equal byte for byte, round by round
#  2 rope    -- rope_lent_bench.c on HEAD's unit and on the first commit's: bcir_tensor_rope_pool against
#               bcir_tensor_rope_lent_pool reading a table bcir_tensor_rope_table_pool filled once, every value
#               checked first, the GEO shapes (4 x T positions, 4 heads of 32) T256 to T2048, clang and gcc, four
#               threads and one, and the fill's own time; then bound_cause.c, the five forms of the table's
#               rotation, under clang 18, clang 23 and gcc
#  3 parts   -- ../gp-lanes/probe_jobs.py and probe_step.py on each tree: the GEO step's pool jobs a worker at a
#               time (the rotations and the fill among them), T256, T512, T1024
#  4 step    -- ../../2026-10-22/attn-rate/step_ab.c: the training step, both trees in one process with an A/A
#               control, the GEO dimensions T128 to T2048 and the 9.84M decoder, clang and gcc
#  5 forward -- ../pool-affinity/fwd_ab.c: the inference forward the same way, B4 T1024 and T2048, B1 T256; then
#               ../pool-affinity/fwd_one.c and step_one.c, one library a process, alternating, six runs each:
#               the forward at B1 and B4 T256, the step at T128 and T256
#  6 rms     -- RMS-LANES-0, measured and not taken, on the tree before: rms_bench.c (the RMSNorm's jobs alone
#               against proto_rms_lanes.py's AVX-512 rows, 1,024 to 4,096 rows of 128, four threads, clang and
#               gcc) and proto_rms_lanes.py's unit in the step (../pool-affinity/step_one.c, one library a
#               process, alternating, six runs each, T128 to T1024)
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/rms/table; O=$C/final; T=$S/rms/tools; D=$R/docs/research/measurements
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
H=$(git rev-parse HEAD); F=$(git rev-parse HEAD~1); P=$(git rev-parse HEAD~2)
git log -1 --format=%s $H | grep -q "^ROPE-BOUND: " || { echo "HEAD is not ROPE-BOUND"; exit 2; }
git log -1 --format=%s $F | grep -q "^ROPE-TABLE: " || { echo "HEAD~1 is not ROPE-TABLE"; exit 2; }
echo "== $(date -u +%H:%M:%S) tree=$H first=$F before=$P dirty=$(git status --short | wc -l)"
rm -rf $C; mkdir -p $C/srcA $C/srcB $C/srcF $O/bin $O/cwd
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h test_decoder_train.c; do
  git show $P:runtime/c/$f > $C/srcA/$f; git show $H:runtime/c/$f > $C/srcB/$f; git show $F:runtime/c/$f > $C/srcF/$f
done
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
echo "== $(date -u +%H:%M:%S) 1 values: the C harness on each tree and compiler"
for cc in clang gcc; do for side in A B F; do
  src=$C/src$side
  $cc -std=c2x -O3 -ffp-contract=off -fno-math-errno -pthread -I $src $src/test_decoder_train.c $src/bcir_tensor.c \
    $src/bcir_decoder_train.c -o $O/bin/harness_${cc}_$side -lm || exit 1
  $O/bin/harness_${cc}_$side > $O/harness-$cc-$side.txt 2>&1; echo "harness $cc $side exit $?"
done
  if diff <(grep -v "ROPE-TABLE" $O/harness-$cc-B.txt) $O/harness-$cc-A.txt > /dev/null; then echo "harness $cc: the same output but the new line"; else echo "harness $cc: OUTPUT DIFFERS"; fi
  grep "ROPE-TABLE" $O/harness-$cc-B.txt
  if cmp -s $O/harness-$cc-B.txt $O/harness-$cc-F.txt; then echo "harness $cc: the first commit's output byte for byte"; else echo "harness $cc: OUTPUT DIFFERS from the first commit's"; fi
done
echo "== $(date -u +%H:%M:%S) 2 the rotation alone, HEAD's unit and the first commit's"
: > $O/rope.jsonl
CC23=<CLANG23>
for cc in clang gcc; do
  for side in B F; do
    $cc $CF -I $C/src$side $T/rope_lent_bench.c $C/src$side/bcir_tensor.c -o $O/bin/rope_lent_bench_${cc}_$side -lm || exit 1
  done
  for tt in 256 512 1024 2048; do for th in 4 1; do for side in F B; do
    $O/bin/rope_lent_bench_${cc}_$side $th 4 $tt 4 32 100 9 | sed "s/^{/{\"compiler\":\"$cc\",\"unit\":\"$side\",/" >> $O/rope.jsonl || echo "rope_lent_bench $cc $side $tt $th FAILED"
  done; done; done
done
cat $O/rope.jsonl
: > $O/bound.jsonl
for cc in clang $CC23 gcc; do
  tag=$(basename $cc)
  $cc $CF -Wall -Wextra -I $C/srcB $T/bound_cause.c $C/srcB/bcir_tensor.c -o $O/bin/bound_cause_$tag -lm || exit 1
  for tt in 256 1024; do for th in 1 4; do
    $O/bin/bound_cause_$tag $th $tt 100 9 | sed "s/^{/{\"compiler\":\"$tag\",/" >> $O/bound.jsonl || echo "bound_cause $tag $tt $th FAILED"
  done; done
done
cat $O/bound.jsonl
echo "== $(date -u +%H:%M:%S) 3 the GEO step's pool jobs a worker at a time, each tree"
for side in A B; do
  rev=$P; [ $side = B ] && rev=$H
  git worktree add -q --detach $C/wt-$side $rev || exit 1
  python3 -I $D/2026-10-24/gp-lanes/probe_jobs.py $C/src$side $C/wt-$side/runtime/c > /dev/null || exit 1
done
for tt in 256 512 1024; do for side in A B; do
  (cd $O/cwd && $PY -I $D/2026-10-24/gp-lanes/probe_step.py $C/wt-$side $tt kblock+exp+reduce 12 $O/probe-$side-T$tt.json 4 > $O/probe-$side-T$tt.txt 2>&1); echo "probe $side T$tt exit $?"
done; done
for side in A B; do git worktree remove --force $C/wt-$side; done; git worktree prune
echo "== $(date -u +%H:%M:%S) 4 the training step in one process"
clang -std=c11 -O2 -I $C/srcB $D/2026-10-22/attn-rate/step_ab.c -o $O/bin/step_ab -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $D/2026-10-24/pool-affinity/fwd_ab.c -o $O/bin/fwd_ab -ldl -lm || exit 1
: > $O/ab-step.jsonl
for spec in "clang|geo-dims T128 kblock+exp+reduce, step|260 128 4 4 3 256 4 128 14 4 64" \
            "clang|geo-dims T256 kblock+exp+reduce, step|260 128 4 4 3 256 4 256 14 4 64" \
            "clang|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T1024 kblock+exp+reduce, step|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 24" \
            "gcc|geo-dims T256 kblock+exp+reduce, step|260 128 4 4 3 256 4 256 14 4 64" \
            "gcc|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T256 v1 (the libm softmax), step|260 128 4 4 3 256 4 256 0 4 64" \
            "clang|decoder-9.84M kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 128 46 4 48"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 5 the inference forward in one process"
: > $O/ab-forward.jsonl
for spec in "clang|geo-dims B4 T1024, forward|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims B4 T2048, forward|260 128 4 4 3 256 4 2048 14 4 32" \
            "gcc|geo-dims B4 T2048, forward|260 128 4 4 3 256 4 2048 14 4 32" \
            "clang|geo-dims B1 T256, forward|260 128 4 4 3 256 1 256 14 4 64"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/fwd_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-forward.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 5b the forward and the step alone in their own processes"
clang -std=c11 -O2 -I $C/srcB $D/2026-10-24/pool-affinity/fwd_one.c -o $O/bin/fwd_one -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $D/2026-10-24/pool-affinity/step_one.c -o $O/bin/step_one_B -ldl -lm || exit 1
: > $O/one.jsonl
for k in 1 2 3 4 5 6; do for v in A B; do
  for bt in "1 256" "4 256"; do
    set -- $bt
    $O/bin/fwd_one $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 $1 $2 14 4 64 4 | sed "s/^{/{\"what\":\"forward\",\"batch\":$1,\"build\":\"$v\",\"run\":$k,/" >> $O/one.jsonl || echo "fwd_one $v B$1 T$2 FAILED"
  done
  for tt in 128 256; do
    $O/bin/step_one_B $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 4 $tt 14 4 40 4 | sed "s/^{/{\"what\":\"step\",\"batch\":4,\"build\":\"$v\",\"run\":$k,/" >> $O/one.jsonl || echo "step_one $v T$tt FAILED"
  done
done; done
wc -l $O/one.jsonl
echo "== $(date -u +%H:%M:%S) 6 RMS-LANES-0 on the tree before"
mkdir -p $C/srcR; cp $C/srcA/* $C/srcR/
python3 -I $T/proto_rms_lanes.py $C/srcA/bcir_tensor.c $C/srcR/bcir_tensor.c || exit 1
: > $O/rms-bench.jsonl
for cc in clang gcc; do
  $cc $CF -Wall -Wextra -I $C/srcA $T/rms_bench.c -o $O/bin/rms_bench_$cc -lm || exit 1
  for rows in 1024 2048 4096; do
    $O/bin/rms_bench_$cc 4 $rows 128 200 9 > $O/rms-bench-$cc-$rows.txt 2>&1; echo "rms_bench $cc $rows exit $?"
    grep "^{" $O/rms-bench-$cc-$rows.txt | sed "s/^{/{\"compiler\":\"$cc\",/" >> $O/rms-bench.jsonl
    grep -v "^{" $O/rms-bench-$cc-$rows.txt
  done
done
clang $CF -fPIC -shared -I $C/srcR $C/srcR/bcir_tensor.c $C/srcR/bcir_decoder_train.c -o $O/bin/libdec_clang_R.so -lm || exit 1
clang -std=c11 -O2 -I $C/srcA $D/2026-10-24/pool-affinity/step_one.c -o $O/bin/step_one -ldl -lm || exit 1
: > $O/rms-steady.jsonl
for tt in 128 256 512 1024; do st=40; [ $tt = 1024 ] && st=20; for k in 1 2 3 4 5 6; do for v in A R; do
  $O/bin/step_one $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 4 $tt 14 4 $st 4 | sed "s/^{/{\"build\":\"$v\",\"run\":$k,/" >> $O/rms-steady.jsonl || echo "step_one $v $tt FAILED"
done; done; done
wc -l $O/rms-steady.jsonl
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
