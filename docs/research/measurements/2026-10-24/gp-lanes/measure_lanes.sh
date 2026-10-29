#!/bin/bash
# GP-LANES' records on the committed tree (HEAD, the runtime LOSS-ONCE's; the overlay is the study's alone), serially,
# two workers at most:
#  1 values  -- gp_lanes_check.c: the lane kernels (gp_lanes_pair.inc, gp_lanes_query.inc) against the study fork's
#               scalar kernels (gp_extract.py: copied verbatim from the level-2 build's query_guard_compact.c), every
#               output byte for byte over six kinds of row (uniform; near parallel, so the FP64 fallback; tiny; zero;
#               large; sparse), clang and gcc, then each kernel's time a row on 1,008 rows (L2-resident) and 24,576
#               (streaming); gp_parts.c, the lane forward's parts on one L1-resident group; gp_pf_bench.c (GP-PF-0),
#               the lane forward on cold outputs with and without the next group's rows prefetched
#  2 build   -- the PR818 study's level-2 build (../geo-amx/build_amx.py, HEAD's unit, GCC as the study builds), then
#               ../loss-once/once_level3.py (LOSS-ONCE and GATE-SPAN: the level the lanes are measured against), then
#               gp_lanes_overlay.py on a copy; gp_tasks_check.c, the two builds' geometric-head entries side by side
#               (row counts 1 to 1,024, serial and pooled), every output byte for byte; red_lanes.py, five faults
#               in the overlay each caught by a witness, three equivalent mutations recorded with their reason
#  3 study   -- ../geo-now/geo_ab.py: global and query_full with the lanes against without, in one process, 48 rounds,
#               every trained state and logit equal; training T128, T256, T512, inference T1024, T2048
#  4 pytorch -- ../compile-limit/geo_compare_static.py: the lanes build against PyTorch eager and Inductor
#  5 profile -- ../loss-once/sampler.c and resolve_samples.py: global and query_full without the lanes and with them,
#               training at T256 (global) and T512 (query_full), inference at T1024
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next2/lanes; O=$C/final; T=$S/next2/tools; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
H=$(git rev-parse HEAD)
echo "== $(date -u +%H:%M:%S) tree=$H dirty=$(git status --short | wc -l)"
rm -rf $C/src $O; mkdir -p $C/src $O/bin $O/cwd $O/gpl
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do git show $H:runtime/c/$f > $C/src/$f; done
echo "== $(date -u +%H:%M:%S) 1 values: the lane kernels against the fork's scalar kernels"
for w in l2 l2-once l2-lanes; do rm -rf $C/$w; done
mkdir -p $C/l2
tar -C $S/geo2/work/level2 --exclude=./build --exclude=./baseline --exclude=./bf16 --exclude=./checkpoints --exclude=./results --exclude=__pycache__ -cf - . | tar -C $C/l2 -xf - || exit 1
python3 -I $D4/geo-amx/build_amx.py $S/geo2/work/level2 $C/l2 $C/src $S/geo2/src/main > $O/study-build-l2.log 2>&1 || { echo "build_amx FAILED"; tail -5 $O/study-build-l2.log; exit 1; }
python3 -I $T/gp_extract.py $C/l2/build/query_guard_compact.c $O/gpl/gp_scalar.inc || exit 1
python3 -I - $C/l2/build/guard_compact.c $C/l2/build/query_guard_compact.c <<'PY' || exit 1  # one pair kernel in both forks
import sys
a, b = (open(f, encoding="utf-8").read() for f in sys.argv[1:3])
cut = lambda s: s[s.index("static int wedge_ill_conditioned"):s.index("static void gp_forward_task")]
sys.exit(0 if cut(a) == cut(b) else "the forks' pair kernels differ")
PY
cp $T/gp_lanes_pair.inc $T/gp_lanes_query.inc $T/gp_lanes_check.c $T/gp_parts.c $O/gpl/ || exit 1
for cc in clang gcc; do
  $cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -I $O/gpl $O/gpl/gp_lanes_check.c -o $O/bin/gp_lanes_check_$cc -lm || exit 1
  $cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -I $O/gpl $O/gpl/gp_parts.c -o $O/bin/gp_parts_$cc -lm || exit 1
  $cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -I $O/gpl $T/gp_pf_bench.c -o $O/bin/gp_pf_bench_$cc -lm || exit 1
done
gcc -std=c11 -O2 $T/gp_tasks_check.c -o $O/bin/gp_tasks_check -ldl || exit 1
: > $O/gp-lanes.jsonl
for cc in clang gcc; do for rows in 170 4096; do
  $O/bin/gp_lanes_check_$cc $rows 20 | sed "s/^{/{\"compiler\":\"$cc\",/" >> $O/gp-lanes.jsonl || echo "gp_lanes_check $cc $rows FAILED"
done; echo "$cc $($O/bin/gp_parts_$cc | tr '\n' ';')" >> $O/gp-parts.txt; done
: > $O/gp-pf.jsonl
for cc in clang gcc; do for rows in 1024 4096 8192; do
  $O/bin/gp_pf_bench_$cc $rows | sed "s/^{/{\"compiler\":\"$cc\",/" >> $O/gp-pf.jsonl || echo "gp_pf_bench $cc $rows FAILED"
done; done
cat $O/gp-lanes.jsonl; cat $O/gp-parts.txt; cat $O/gp-pf.jsonl
echo "== $(date -u +%H:%M:%S) 2 the study's builds: level 2, once_level3.py, then the lanes"
cp -a $C/l2 $C/l2-once || exit 1
python3 -I $D4/loss-once/once_level3.py $C/l2-once guard_compact query_guard_compact attention > $O/study-build-once.log 2>&1 || { echo "once FAILED"; cat $O/study-build-once.log; exit 1; }
cp -a $C/l2-once $C/l2-lanes || exit 1
python3 -I $T/gp_lanes_overlay.py $C/l2-lanes guard_compact query_guard_compact > $O/study-build-lanes.log 2>&1 || { echo "lanes FAILED"; cat $O/study-build-lanes.log; exit 1; }
cat $O/study-build-once.log $O/study-build-lanes.log
: > $O/gp-tasks.jsonl
for spec in "guard_compact 0" "query_guard_compact 1"; do
  set -- $spec
  $O/bin/gp_tasks_check $C/l2-once/build/$1.so $C/l2-lanes/build/$1.so $2 | sed "s/^{/{\"library\":\"$1\",/" >> $O/gp-tasks.jsonl || echo "gp_tasks_check $1 FAILED"
done
cat $O/gp-tasks.jsonl
python3 -I $T/red_lanes.py $T $C/l2-once $O/gpl/gp_scalar.inc $O/bin/gp_tasks_check $O/red $O/red.json || echo "red_lanes FAILED"
echo "== $(date -u +%H:%M:%S) 3 the study: the geometric heads with the lanes against without"
(cd $O/cwd && $PY -I $D4/geo-now/geo_ab.py $C/l2-once $C/l2-lanes $O/study-ab.json --rounds 48 --steps 3 --models global,query_full --train-t 128,256,512 --infer-t 1024,2048) > $O/study-ab.txt 2>&1; echo "geo_ab exit $?"
grep "^{" $O/study-ab.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) 4 the lanes build against PyTorch"
(cd $O/cwd && $PY -I $D4/compile-limit/geo_compare_static.py $C/l2-lanes $O/study-vs-pytorch.json --rounds 12 --steps 3 --models global,query_full) > $O/study-vs-pytorch.txt 2>&1; echo "geo_compare_static exit $?"
tail -n 12 $O/study-vs-pytorch.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) 5 the sampled profiles with the lanes"
gcc -O2 -fPIC -shared $D4/loss-once/sampler.c -o $O/bin/sampler.so || exit 1
for w in once lanes; do for spec in "global train 256 60" "global infer 1024 120" "query_full train 512 30" "query_full infer 1024 120"; do
  set -- $spec; tag=$w-$1-$2$3
  (cd $O/cwd && SAMPLER_OUT=$O/prof-$tag.raw SAMPLER_US=100 LD_PRELOAD=$O/bin/sampler.so $PY -I $D4/loss-once/prof_study.py $C/l2-$w $1 $2 $3 $4) > /dev/null 2>&1 || echo "profile $tag FAILED"
  python3 -I $D4/loss-once/resolve_samples.py $O/prof-$tag.raw 40 > $O/prof-$tag.txt || echo "resolve $tag FAILED"
  rm -f $O/prof-$tag.raw  # the raw file holds the host's maps; the resolved counts are the record
done; done
head -n 14 $O/prof-*.txt | cut -c1-160
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
