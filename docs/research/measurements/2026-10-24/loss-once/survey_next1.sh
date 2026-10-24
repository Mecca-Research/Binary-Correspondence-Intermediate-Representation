#!/bin/bash
# NEXT-1's survey, the measurements that placed LOSS-ONCE, on the tree before it (HEAD~1; its tensor unit is the tree
# after's byte for byte), serially, two workers at most; after measure_once.sh, whose sources and study builds it reads:
#  1 gemm    -- ../pool-affinity/gemm_b2b.c: the GEO step's T256 GEMM shapes back to back and after a 1 ms pause, on
#               one thread and four
#  2 tasks   -- probe_tasks.py's copy of the unit (TSC counters a worker at a time: when it entered each job after its
#               dispatch, its cycles inside tasks, when it left; the job's wall cycles at the caller; the CPU each
#               worker ran each job on) and gemm_tasks.c: the 1,024 x 128 x 128 product (nt and nn), the 2,048-row one
#               and the 128 x 128 weight adjoint over 1,024 rows on four threads, three repetitions each
#  3 sleeps  -- count_sleeps.py's copy of the unit (each time a worker leaves its spins and yields for the condition
#               variable, each job, each time the caller waits on done) and ../pool-affinity/step_one.c: the GEO step
#               at T128, T256 and T512, 4 untimed steps and 40 timed
#  4 wall    -- step_wall.py from a checkout of the tree before: the GEO step's wall time a step against its jobs'
#               clocks in the same run, T256, 4 steps a call, 12 rounds
#  5 profile -- sampler.c (SIGPROF every 100 us of the process's CPU time, preloaded; each sample's instruction
#               pointer resolved offline by resolve_samples.py against the libraries' symbols): BCIR's own step at
#               T256 (step_one.c, 200 steps), and the study's models (prof_study.py) at level 2 -- global's training at
#               T256 and T512 and its inference at T1024, query_full's training at T512 and its inference at T1024 --
#               and the attention model at level 3, training at T512 and inference at T2048
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next1/once; O=$C/survey; T=$S/next1/tools; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
A=$(git rev-parse HEAD~1)
echo "== $(date -u +%H:%M:%S) before=$A after=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do
  git show $A:runtime/c/$f | cmp -s - $C/srcA/$f || { echo "$C/srcA/$f is not HEAD~1's (measure_once.sh writes it)"; exit 1; }
done
for w in l2 l3; do [ -f $C/$w/build/attention.so ] || { echo "missing $C/$w (measure_once.sh builds it)"; exit 1; }; done
rm -rf $O; mkdir -p $O/bin $O/src_tasks $O/src_count $O/cwd
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
echo "== $(date -u +%H:%M:%S) 1 the T256 GEMM shapes, one thread and four"
clang $CF -I $C/srcA $D4/pool-affinity/gemm_b2b.c $C/srcA/bcir_tensor.c -o $O/bin/gemm_b2b -lm || exit 1
: > $O/gemm-b2b.txt
for shape in "0 1 1024 128 128" "0 1 1024 256 128" "0 1 1024 128 256" "0 0 1024 128 128" "0 0 1024 256 128" \
             "1 0 128 128 1024" "1 0 256 128 1024"; do
  for th in 1 4; do $O/bin/gemm_b2b $th $shape 50 9 1000 >> $O/gemm-b2b.txt || echo "gemm_b2b $th $shape FAILED"; done
done
cat $O/gemm-b2b.txt
echo "== $(date -u +%H:%M:%S) 2 a GEMM job a worker at a time"
cp $C/srcA/bcir_tensor.h $O/src_tasks/ || exit 1
python3 -I $T/probe_tasks.py $C/srcA/bcir_tensor.c $O/src_tasks/bcir_tensor.c > /dev/null || exit 1
clang $CF -I $O/src_tasks $T/gemm_tasks.c $O/src_tasks/bcir_tensor.c -o $O/bin/gemm_tasks -lm || exit 1
: > $O/gemm-tasks.txt
for shape in "0 1 1024 128 128" "0 0 1024 128 128" "0 1 2048 128 128" "1 0 128 128 1024"; do
  for rep in 1 2 3; do echo "rep $rep" >> $O/gemm-tasks.txt; $O/bin/gemm_tasks 4 $shape 50 9 >> $O/gemm-tasks.txt || echo "gemm_tasks $shape FAILED"; done
done
grep -E "^rep|threads" $O/gemm-tasks.txt
echo "== $(date -u +%H:%M:%S) 3 the pool's sleeps in the step"
cp $C/srcA/bcir_tensor.h $C/srcA/bcir_decoder_train.c $C/srcA/bcir_decoder_train.h $O/src_count/ || exit 1
python3 -I $T/count_sleeps.py $C/srcA/bcir_tensor.c $O/src_count/bcir_tensor.c > /dev/null || exit 1
clang $CF -fPIC -shared -I $O/src_count $O/src_count/bcir_tensor.c $O/src_count/bcir_decoder_train.c -o $O/bin/libdec_count.so -lm || exit 1
clang -std=c11 -O2 -I $O/src_count $D4/pool-affinity/step_one.c -o $O/bin/step_one -ldl -lm || exit 1
: > $O/sleeps.txt
for tt in 128 256 512; do
  echo "T$tt $($O/bin/step_one $O/bin/libdec_count.so 260 128 4 4 3 256 4 $tt 14 4 40 4 2>&1 | tr '\n' ' ')" >> $O/sleeps.txt
done
cat $O/sleeps.txt | cut -c1-240
echo "== $(date -u +%H:%M:%S) 4 the step's time outside its jobs"
rm -rf $C/wt-survey; git worktree add --detach $C/wt-survey $A > /dev/null 2>&1 || { echo "worktree FAILED"; exit 1; }
(cd $O/cwd && python3 -I $T/step_wall.py $C/wt-survey 256 12 4 4) > $O/step-wall.txt 2>&1; echo "step_wall exit $?"
git worktree remove --force $C/wt-survey; git worktree prune
tail -1 $O/step-wall.txt
echo "== $(date -u +%H:%M:%S) 5 the sampled profiles"
gcc -O2 -fPIC -shared $T/sampler.c -o $O/bin/sampler.so || exit 1
clang $CF -fPIC -shared -I $C/srcA $C/srcA/bcir_tensor.c $C/srcA/bcir_decoder_train.c -o $O/bin/libdec_before.so -lm || exit 1
SAMPLER_OUT=$O/prof-step256.raw SAMPLER_US=100 LD_PRELOAD=$O/bin/sampler.so $O/bin/step_one $O/bin/libdec_before.so 260 128 4 4 3 256 4 256 14 4 200 4 > /dev/null || echo "profile step FAILED"
for spec in "l2 global train 256 60" "l2 global train 512 30" "l2 global infer 1024 120" "l2 query_full train 512 30" \
            "l2 query_full infer 1024 120" "l3 attention train 512 20" "l3 attention infer 2048 12"; do
  set -- $spec; tag=$2-$3$4
  (cd $O/cwd && SAMPLER_OUT=$O/prof-$tag.raw LD_PRELOAD=$O/bin/sampler.so $PY -I $T/prof_study.py $C/$1 $2 $3 $4 $5) || echo "profile $tag FAILED"
done
for raw in $O/prof-*.raw; do
  python3 -I $T/resolve_samples.py $raw 40 > ${raw%.raw}.txt || echo "resolve $raw FAILED"
  rm -f $raw  # the raw file holds the host's maps; the resolved counts are the record
done
head -n 16 $O/prof-*.txt | cut -c1-160
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
