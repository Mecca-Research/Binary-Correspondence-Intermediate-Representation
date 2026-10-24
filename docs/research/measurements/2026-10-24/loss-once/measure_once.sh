#!/bin/bash
# LOSS-ONCE's records on the committed tree (HEAD) against the tree before it (HEAD~1), serially, two workers at most:
#  1 values  -- exp_span_check.c: the tensor unit's span exponential (bcir_tensor_exp, every instance) against
#               bcir_tensor_expf on all 2^32 inputs and on spans of every length to 80 over the special values, clang
#               and gcc -- the premise of once_level3.py's gate edit; every timed arm below holds its outputs equal
#  2 loss    -- loss_bench.c (mk_loss_bench.py: each tree's loss task copied from its own source) on one thread: the
#               GEO step's rows (1,024, 2,048, 4,096, 8,192) of 260 logits and 512 rows of 1,500 (past the kept
#               1,024), every adjoint and term equal, clang and gcc
#  3 parts   -- ../../2026-10-23/bwd-keep/step_parts.py at T256 and T512, each tree from its own checkout (the tree
#               before in a detached worktree), twice, interleaved: the step's train.loss job and the rest
#  4 steady  -- ../pool-affinity/step_one.c, one library a process, the libraries alternating run by run, six runs each:
#               the training step at the GEO dimensions, T128 to T1024, and the 9.84M decoder; every run's weights the
#               same digest on both trees
#  5 study   -- once_level3.py on the PR818 study's level-2 build (../geo-amx/build_amx.py, the tree after's unit,
#               GCC as the study builds) and on the attention model's level-3 build (P-BANDS's chain:
#               ../../2026-10-22/infer-attn/build_level3.py --scratch, ../../2026-10-23/att-contig/contig_level3.py,
#               ../../2026-10-23/p-lower/lower_level3.py, ../p-bands/bands_level3.py); ../geo-now/geo_ab.py with each
#               build before and after the overlay in one process, every trained state and logit equal
#  6 pytorch -- the overlaid builds against PyTorch eager and Inductor at fixed shapes:
#               ../compile-limit/geo_compare_static.py (global, query_full at level 2) and
#               ../compile-limit/static_compare.py (the attention model at level 3, training and inference)
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next1/once; O=$C/final; T=$S/next1/tools
D=$R/docs/research/measurements/2026-10-22; D3=$R/docs/research/measurements/2026-10-23; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $R || exit 2
A=$(git rev-parse HEAD~1); H=$(git rev-parse HEAD)
echo "== $(date -u +%H:%M:%S) before=$A after=$H dirty=$(git status --short | wc -l)"
rm -rf $C/srcA $C/srcB $O; mkdir -p $C/srcA $C/srcB $O/bin $O/cwd
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do
  git show $A:runtime/c/$f > $C/srcA/$f; git show $H:runtime/c/$f > $C/srcB/$f
done
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
clang -std=c11 -O2 -I $C/srcB $D4/pool-affinity/step_one.c -o $O/bin/step_one -ldl -lm || exit 1
python3 -I $T/mk_loss_bench.py $C/srcA/bcir_decoder_train.c $C/srcB/bcir_decoder_train.c $O/loss_bench.c > /dev/null || exit 1
for cc in clang gcc; do
  $cc -std=c11 -O3 -ffp-contract=off -fno-math-errno $O/loss_bench.c -o $O/bin/loss_bench_$cc -lm || exit 1
  $cc $CF -I $C/srcB $T/exp_span_check.c $C/srcB/bcir_tensor.c -o $O/bin/exp_span_check_$cc -lm || exit 1
done
echo "== $(date -u +%H:%M:%S) 1 values: the span exponential against the scalar one on every input"
for cc in clang gcc; do echo "$cc $($O/bin/exp_span_check_$cc)" | tee -a $O/exp-span.txt; done
echo "== $(date -u +%H:%M:%S) 2 the loss task alone, one thread"
: > $O/loss.jsonl
for cc in clang gcc; do for spec in "1024 260 31" "2048 260 21" "4096 260 15" "8192 260 11" "512 1500 21"; do
  set -- $spec
  $O/bin/loss_bench_$cc $1 $2 $3 | sed "s/^{/{\"compiler\":\"$cc\",/" >> $O/loss.jsonl || echo "loss_bench $cc $spec FAILED"
done; done
cat $O/loss.jsonl | cut -c1-240
echo "== $(date -u +%H:%M:%S) 3 the GEO step a part at a time (the job clock), both trees"
rm -rf $C/wt-before; git worktree add --detach $C/wt-before $A > /dev/null 2>&1 || { echo "worktree FAILED"; exit 1; }
for rep in 1 2; do for tt in 256 512; do
  python3 -I $C/wt-before/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py $tt kblock+exp+reduce 16 $O/step-parts-T$tt-before-$rep.json 4 > $O/step-parts-T$tt-before-$rep.txt 2>&1; echo "step_parts T$tt before $rep exit $?"
  python3 -I $R/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py $tt kblock+exp+reduce 16 $O/step-parts-T$tt-after-$rep.json 4 > $O/step-parts-T$tt-after-$rep.txt 2>&1; echo "step_parts T$tt after $rep exit $?"
done; done
git worktree remove --force $C/wt-before; git worktree prune
echo "== $(date -u +%H:%M:%S) 4 the training step, one library a process, alternating"
: > $O/steady.jsonl
for spec in "geo 260 128 4 4 3 256 4 128 14 4 40 4" "geo 260 128 4 4 3 256 4 256 14 4 40 4" \
            "geo 260 128 4 4 3 256 4 512 14 4 40 4" "geo 260 128 4 4 3 256 4 1024 14 4 20 4" \
            "decoder-9.84M 260 384 8 4 6 1024 8 128 46 4 20 4"; do
  set -- $spec; tag=$1; shift
  for k in 1 2 3 4 5 6; do for v in A B; do
    $O/bin/step_one $O/bin/libdec_clang_$v.so "$@" \
      | sed "s/^{/{\"case\":\"$tag\",\"build\":\"$v\",\"run\":$k,/" >> $O/steady.jsonl || echo "step_one $v $tag $* FAILED"
  done; done
done
echo "== $(date -u +%H:%M:%S) 5 the study: level 2 and the attention model's level 3, each before and after once_level3.py"
for w in l2 l2-once l3 l3-once; do rm -rf $C/$w; done
mkdir -p $C/l2 $C/l3
for w in l2 l3; do
  tar -C $S/geo2/work/level2 --exclude=./build --exclude=./baseline --exclude=./bf16 --exclude=./checkpoints --exclude=./results --exclude=__pycache__ -cf - . | tar -C $C/$w -xf - || exit 1
done
python3 -I $D4/geo-amx/build_amx.py $S/geo2/work/level2 $C/l2 $C/srcB $S/geo2/src/main > $O/study-build-l2.log 2>&1 || { echo "build_amx FAILED"; tail -5 $O/study-build-l2.log; exit 1; }
python3 -I $D/infer-attn/build_level3.py $S/geo2/work/level2 $C/l3 $C/srcB $S/geo2/src/main --scratch > $O/study-build-l3.log 2>&1 || { echo "level3 FAILED"; tail -5 $O/study-build-l3.log; exit 1; }
python3 -I $D3/att-contig/contig_level3.py $C/l3 $C/srcB >> $O/study-build-l3.log 2>&1 || { echo "contig FAILED"; exit 1; }
python3 -I $D3/p-lower/lower_level3.py $C/l3 >> $O/study-build-l3.log 2>&1 || { echo "lower FAILED"; exit 1; }
python3 -I $D4/p-bands/bands_level3.py $C/l3 >> $O/study-build-l3.log 2>&1 || { echo "bands FAILED"; exit 1; }
cp -a $C/l2 $C/l2-once && cp -a $C/l3 $C/l3-once || exit 1
python3 -I $T/once_level3.py $C/l2-once guard_compact query_guard_compact attention > $O/study-build-once.log 2>&1 || { echo "once l2 FAILED"; cat $O/study-build-once.log; exit 1; }
python3 -I $T/once_level3.py $C/l3-once attention >> $O/study-build-once.log 2>&1 || { echo "once l3 FAILED"; cat $O/study-build-once.log; exit 1; }
(cd $O/cwd && $PY -I $D4/geo-now/geo_ab.py $C/l2 $C/l2-once $O/study-ab.json --rounds 16 --steps 3 --models global,query_full --train-t 128,256,512 --infer-t 1024,2048) > $O/study-ab.txt 2>&1; echo "geo_ab level 2 exit $?"
grep "^{" $O/study-ab.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/geo-now/geo_ab.py $C/l3 $C/l3-once $O/level3-ab.json --rounds 16 --steps 3 --models attention --train-t 512,1024,2048 --infer-t 1024,2048) > $O/level3-ab.txt 2>&1; echo "geo_ab level 3 exit $?"
grep "^{" $O/level3-ab.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) 6 the overlaid builds against PyTorch"
(cd $O/cwd && $PY -I $D4/compile-limit/geo_compare_static.py $C/l2-once $O/study-vs-pytorch.json --rounds 12 --steps 3 --models global,query_full) > $O/study-vs-pytorch.txt 2>&1; echo "geo_compare_static exit $?"
tail -n 12 $O/study-vs-pytorch.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/compile-limit/static_compare.py $C/l3-once $O/level3-vs-pytorch-train.json --mode train --rounds 24 --steps 3 --times 512,1024,2048) > $O/level3-vs-pytorch-train.txt 2>&1; echo "static_compare train exit $?"
tail -n 4 $O/level3-vs-pytorch-train.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/compile-limit/static_compare.py $C/l3-once $O/level3-vs-pytorch-infer.json --mode infer --rounds 24 --steps 3 --times 1024,2048) > $O/level3-vs-pytorch-infer.txt 2>&1; echo "static_compare infer exit $?"
tail -n 3 $O/level3-vs-pytorch-infer.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
