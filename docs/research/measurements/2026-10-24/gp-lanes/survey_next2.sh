#!/bin/bash
# NEXT-2's survey, the measurements that placed the slice after LOSS-ONCE, on the committed tree (HEAD), serially, two
# workers at most:
#  1 peak     -- fma_peak.c: the core's AVX-512 FMA rate with every operand in registers, on one, two and four threads
#  2 ceiling  -- gemm_ceiling.c: the T256 step's GEMM shapes on four threads each with operands of its own (the
#                kernel's all-core ceiling), against one thread and the pool's one GEMM over four, twice
#  3 jobs     -- gemm_jobs.c: the step's GEMM jobs at T256 as the trainer issues them (alone, pairs, groups, chains)
#                against their own four-copy ceiling
#  4 pool     -- pool_rtt.c: a job of trivial tasks on the pool, back to back and after a pause
#  5 probe    -- probe_jobs.py's copies of the unit and the trainer (every pool job's launch, tasks, wall cycles and,
#                a worker at a time, entry, busy cycles and finish) in a worktree of HEAD; probe_step.py at T256 (by
#                claim and task count), T512 and T1024; probe_order.py, one T256 step's jobs in order
#  6 not taken -- GROUP-ROWS-0 (proto_group_rows.py: the forward's group jobs a row range at a time) and COMMIT-BLOCKS-0
#                (the update's commit in chunks of 16,384 where the tree takes 65,536): ../pool-affinity/step_one.c,
#                one library a process, alternating, six runs each, every run's weights the same digest; the commit
#                job itself by probe_step.py, the two chunkings alternating
#  7 rope     -- rope_bench.c (ROPE-TABLE-0): bcir_tensor_rope_pool against the rotation reading a table of the same
#                cosines and sines, every value byte for byte, clang and gcc
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next2; O=$C/survey; T=$C/tools; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
H=$(git rev-parse HEAD)
echo "== $(date -u +%H:%M:%S) tree=$H dirty=$(git status --short | wc -l)"
rm -rf $O; mkdir -p $O/bin $O/cwd $O/src $O/gr $O/cb
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do git show $H:runtime/c/$f > $O/src/$f; done
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
LF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic -Werror -DBCIR_DT_BUILD_SHARED -shared -fPIC -pthread"
gcc -O2 -mavx512f -pthread $T/fma_peak.c -o $O/bin/fma_peak || exit 1
for t in gemm_ceiling gemm_jobs pool_rtt; do clang $CF -I $O/src $T/$t.c $O/src/bcir_tensor.c -o $O/bin/$t -lm || exit 1; done
for cc in clang gcc; do $cc $CF -I $O/src $T/rope_bench.c $O/src/bcir_tensor.c -o $O/bin/rope_bench_$cc -lm || exit 1; done
clang -std=c11 -O2 -I $O/src $D4/pool-affinity/step_one.c -o $O/bin/step_one -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) 1 the FMA peak"
for w in 1 2 4 1 2 4; do $O/bin/fma_peak $w 200000000; done | tee $O/fma-peak.jsonl
echo "== $(date -u +%H:%M:%S) 2 the GEMM's all-core ceiling, the T256 shapes"
: > $O/ceiling.jsonl
for rep in 1 2; do for shape in "0 1 1024 128 128" "0 1 1024 256 128" "0 1 1024 128 256" "0 1 1024 260 128" "0 0 1024 128 128" \
    "0 0 1024 256 128" "0 0 1024 128 256" "0 0 1024 128 260" "1 0 128 128 1024" "1 0 256 128 1024" "1 0 128 256 1024" "1 0 260 128 1024"; do
  $O/bin/gemm_ceiling 4 $shape 200 11 | sed "s/^{/{\"rep\":$rep,/" >> $O/ceiling.jsonl || echo "gemm_ceiling $shape FAILED"
done; done
cut -c1-220 $O/ceiling.jsonl
echo "== $(date -u +%H:%M:%S) 3 the step's GEMM jobs against their ceiling"
for rep in 1 2; do $O/bin/gemm_jobs 4 1024 200 11 | sed "s/^{/{\"rep\":$rep,/"; done | tee $O/jobs.jsonl | cut -c1-220
echo "== $(date -u +%H:%M:%S) 4 the pool's cost a job"
for tasks in 4 16 64; do for gap in 20 150; do $O/bin/pool_rtt 4 $tasks 2000 9 $gap; done; done | tee $O/pool-rtt.jsonl
echo "== $(date -u +%H:%M:%S) 5 the step's jobs a worker at a time"
rm -rf $C/wt-probe; git worktree add --detach $C/wt-probe $H > /dev/null 2>&1 || { echo "worktree FAILED"; exit 1; }
python3 -I $T/probe_jobs.py $C/wt-probe/runtime/c $O/probe-src > /dev/null || exit 1
cp $O/probe-src/bcir_tensor.c $O/probe-src/bcir_decoder_train.c $C/wt-probe/runtime/c/ || exit 1
(cd $O/cwd && $PY -I $T/probe_step.py $C/wt-probe 256 kblock+exp+reduce 12 $O/probe-T256.json 4 split) > $O/probe-T256.txt 2>&1; echo "probe T256 exit $?"
(cd $O/cwd && $PY -I $T/probe_step.py $C/wt-probe 512 kblock+exp+reduce 8 $O/probe-T512.json 4) > $O/probe-T512.txt 2>&1; echo "probe T512 exit $?"
(cd $O/cwd && $PY -I $T/probe_step.py $C/wt-probe 1024 kblock+exp+reduce 8 $O/probe-T1024.json 4) > $O/probe-T1024.txt 2>&1; echo "probe T1024 exit $?"
(cd $O/cwd && $PY -I $T/probe_order.py $C/wt-probe 256 kblock+exp+reduce) > $O/probe-order-T256.txt 2>&1; echo "probe order exit $?"
head -n 8 $O/probe-T256.txt $O/probe-T512.txt $O/probe-T1024.txt | cut -c1-200
echo "== $(date -u +%H:%M:%S) 6 measured and not taken: GROUP-ROWS-0 and COMMIT-BLOCKS-0"
cp $O/src/bcir_tensor.h $O/src/bcir_decoder_train.h $O/src/bcir_decoder_train.c $O/gr/ && python3 -I $T/proto_group_rows.py $O/src/bcir_tensor.c $O/gr/bcir_tensor.c > /dev/null || exit 1
cp $O/src/bcir_tensor.h $O/src/bcir_tensor.c $O/src/bcir_decoder_train.h $O/cb/ || exit 1
sed 's/^#define BCIR_DT_CHUNK 65536$/#define BCIR_DT_CHUNK 16384/' $O/src/bcir_decoder_train.c > $O/cb/bcir_decoder_train.c
grep -q "^#define BCIR_DT_CHUNK 16384$" $O/cb/bcir_decoder_train.c || { echo "COMMIT-BLOCKS-0 edit FAILED"; exit 1; }
for v in src gr cb; do clang $LF -I $O/$v $O/$v/bcir_tensor.c $O/$v/bcir_decoder_train.c -o $O/bin/lib_$v.so -lm || exit 1; done
: > $O/not-taken.jsonl
for spec in "group-rows gr 128" "group-rows gr 256" "group-rows gr 512" "commit-blocks cb 128" "commit-blocks cb 256"; do
  set -- $spec
  for k in 1 2 3 4 5 6; do for v in src $2; do
    $O/bin/step_one $O/bin/lib_$v.so 260 128 4 4 3 256 4 $3 14 4 40 4 \
      | sed "s/^{/{\"case\":\"$1\",\"build\":\"$v\",\"run\":$k,/" >> $O/not-taken.jsonl || echo "step_one $v $3 FAILED"
  done; done
done
for k in 1 2; do for v in tree blocks; do
  if [ $v = blocks ]; then sed 's/^#define BCIR_DT_CHUNK 65536$/#define BCIR_DT_CHUNK 16384/' $O/probe-src/bcir_decoder_train.c > $C/wt-probe/runtime/c/bcir_decoder_train.c
  else cp $O/probe-src/bcir_decoder_train.c $C/wt-probe/runtime/c/bcir_decoder_train.c; fi
  (cd $O/cwd && $PY -I $T/probe_step.py $C/wt-probe 256 kblock+exp+reduce 16 $O/probe-commit-$v-$k.json 4 split) > $O/probe-commit-$v-$k.txt 2>&1; echo "probe commit $v $k exit $?"
done; done
git worktree remove --force $C/wt-probe; git worktree prune
grep -h "train.update" $O/probe-commit-*.txt | cut -c1-160
echo "== $(date -u +%H:%M:%S) 7 ROPE-TABLE-0"
: > $O/rope.jsonl
for cc in clang gcc; do for tt in 256 512 1024 2048; do
  $O/bin/rope_bench_$cc 4 4 $tt 4 32 100 9 | sed "s/^{/{\"compiler\":\"$cc\",/" >> $O/rope.jsonl || echo "rope_bench $cc $tt FAILED"
done; done
cat $O/rope.jsonl
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
