#!/bin/bash
# POOL-AFFINITY's records on the committed tree (HEAD) against the tree before it (HEAD~1), serially, two workers at most:
#  1 values    -- ../../2026-10-23/mm-rows/bitcheck.c (every k-blocked entry: serial instances, the pool at four and two
#                 threads and through each instance, pair jobs, group jobs) and P-BANDS's five attention hash programs,
#                 before == after under clang and gcc; every timed arm below holds its outputs equal as it runs
#  2 gemm      -- ../../2026-10-23/mm-rows/ab_gemm.c: the step's GEMM jobs at 1024, 2048 and 8192 rows, both units in one
#                 process beside an A/A control
#  3 step      -- ../../2026-10-22/attn-rate/step_ab.c: the training step at the GEO dimensions, T128 to T2048 (clang),
#                 T256 and T512 (gcc), T256 and T512 on two threads, the v1 contract, the 9.84M decoder; the
#                 weights equal every round
#  4 forward   -- fwd_ab.c: the inference forward (bcir_decoder_forward) at B4 T256/T1024/T2048 and B1 T128/T256; the
#                 logits equal every round
#  5 attention -- ab_bands_own.c (../p-bands/ab_bands.c with each arm's pool made by its own library: the original made
#                 every arm's pool with the first library's, and the tree after's pool is laid out apart): the attention
#                 alone (EXP|LOWER|BANDS) at B4/T2048 and B4/T256, outputs equal
#  6 parts     -- ../../2026-10-23/bwd-keep/step_parts.py at T256 and T512, each tree from its own checkout (the tree
#                 before in a detached worktree), twice, interleaved
#  7 design    -- the committed pool against itself without one part of its design (proto_aff_variants.py: nohead,
#                 the pair and group jobs naming no head; nosteal, no worker taking another block's leftovers):
#                 ab_gemm.c on the pair, group and single jobs, step_ab.c at T128, T256 and T512
#  8 elementwise -- ew_cold.c on both units: the RMSNorm, its VJP and the residual add back to back, cold, and
#                 after a pool job wrote their operands, at 1,024, 2,048 and 8,192 rows on one thread and four
#  9 study     -- the PR818 study's three models (../geo-amx/build_amx.py on each unit, GCC as the study builds):
#                 ../geo-now/geo_ab.py before/after with every trained state and logit equal, and
#                 ../compile-limit/geo_compare_static.py, the tree after against PyTorch eager and Inductor
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next0/aff; O=$C/final; T=$S/next0/tools
D=$R/docs/research/measurements/2026-10-22; D3=$R/docs/research/measurements/2026-10-23; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE
cd $R || exit 2
A=$(git rev-parse HEAD~1); H=$(git rev-parse HEAD)
echo "== $(date -u +%H:%M:%S) before=$A after=$H dirty=$(git status --short | wc -l)"
rm -rf $C/srcA $C/srcB $O; mkdir -p $C/srcA $C/srcB $O/bin
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do
  git show $A:runtime/c/$f > $C/srcA/$f; git show $H:runtime/c/$f > $C/srcB/$f
done
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc $CF -I $src $D3/mm-rows/bitcheck.c $src/bcir_tensor.c -o $O/bin/bitcheck_${cc}_$side -lm || exit 1
  $cc $CF -I $src $D3/scores-t/scratch_hash.c $src/bcir_tensor.c -o $O/bin/strict_${cc}_$side -lm || exit 1
  for hsh in "$D3/fwd-strip/fwd_hash.c" "$D3/p-lower/lower_hash.c" "$D/attn-rate/attn_hash.c" "$D/bwd-long/band_hash.c"; do
    $cc $CF -I $src $hsh $src/bcir_tensor.c -o $O/bin/$(basename $hsh .c)_${cc}_$side -lm || exit 1
  done
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c -o $O/bin/libtensor_${cc}_$side.so -lm || exit 1
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
clang -std=c11 -O2 $D3/mm-rows/ab_gemm.c -o $O/bin/ab_gemm -ldl -lm || exit 1
clang -std=c11 -O2 $T/ab_bands_own.c -o $O/bin/ab_bands -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $D/attn-rate/step_ab.c -o $O/bin/step_ab -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $T/fwd_ab.c -o $O/bin/fwd_ab -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) 1 values"
for cc in clang gcc; do
  for hb in bitcheck strict fwd_hash lower_hash attn_hash band_hash; do
    arg=""; [ $hb = bitcheck ] && arg=700
    $O/bin/${hb}_${cc}_A $arg > $O/${hb}_${cc}_A.txt && $O/bin/${hb}_${cc}_B $arg > $O/${hb}_${cc}_B.txt
    cmp -s $O/${hb}_${cc}_A.txt $O/${hb}_${cc}_B.txt && echo "$hb $cc: $(wc -l < $O/${hb}_${cc}_B.txt) lines before == after" || echo "$hb $cc: DIFFER"
  done
done
for hb in bitcheck strict fwd_hash lower_hash attn_hash band_hash; do
  cmp -s $O/${hb}_clang_B.txt $O/${hb}_gcc_B.txt && echo "$hb: clang's lines gcc's" || echo "$hb clang vs gcc: DIFFER"
done
echo "== $(date -u +%H:%M:%S) 2 the step's GEMM jobs in one process"
: > $O/ab-gemm.jsonl
for M in 1024 2048 8192; do
  $O/bin/ab_gemm $O/bin/libtensor_clang_A.so $O/bin/libtensor_clang_B.so 32 4 5 \
    g:0,1,$M,128,128 g:0,1,$M,256,128 g:0,1,$M,128,256 g:0,1,$M,260,128 \
    p:0,0,$M,128,128:1,0,128,128,$M p:0,0,$M,256,128:1,0,128,256,$M p:0,0,$M,128,260:1,0,260,128,$M \
    c3:0,0,$M,128,128:3x1,0,128,128,$M c2:0,0,$M,128,256:2x1,0,256,128,$M \
    g:1,0,128,128,$M g:0,0,$M,128,128 >> $O/ab-gemm.jsonl || echo "ab_gemm $M FAILED"
done
echo "== $(date -u +%H:%M:%S) 3 the training step in one process"
: > $O/ab-step.jsonl
for spec in "clang|geo-dims T128 kblock+exp+reduce, step|260 128 4 4 3 256 4 128 14 4 48" \
            "clang|geo-dims T256 kblock+exp+reduce, step|260 128 4 4 3 256 4 256 14 4 48" \
            "clang|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T1024 kblock+exp+reduce, step|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 24" \
            "gcc|geo-dims T256 kblock+exp+reduce, step|260 128 4 4 3 256 4 256 14 4 48" \
            "gcc|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T256 two threads kblock+exp+reduce, step|260 128 4 4 3 256 4 256 14 2 48" \
            "clang|geo-dims T512 two threads kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 2 48" \
            "clang|geo-dims T256 v1 (the libm softmax), step|260 128 4 4 3 256 4 256 0 4 48" \
            "clang|decoder-9.84M kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 128 46 4 48"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 4 the inference forward in one process"
: > $O/ab-forward.jsonl
for spec in "clang|geo-dims B4 T256, forward|260 128 4 4 3 256 4 256 14 4 48" \
            "clang|geo-dims B4 T1024, forward|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims B4 T2048, forward|260 128 4 4 3 256 4 2048 14 4 24" \
            "clang|geo-dims B1 T128, forward|260 128 4 4 3 256 1 128 14 4 48" \
            "clang|geo-dims B1 T256, forward|260 128 4 4 3 256 1 256 14 4 48"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/fwd_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-forward.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 5 the attention alone in one process (EXP|LOWER|BANDS: 49)"
: > $O/ab-attention.jsonl
for spec in "48 4 2048 4 4 32 4 49" "48 4 256 4 4 32 4 49"; do
  $O/bin/ab_bands $O/bin/libtensor_clang_A.so $O/bin/libtensor_clang_B.so $spec | sed 's/^{/{"compiler":"clang",/' >> $O/ab-attention.jsonl || echo "ab_bands $spec FAILED"
done
echo "== $(date -u +%H:%M:%S) 6 the GEO step a part at a time (the job clock), both trees"
rm -rf $C/wt-before; git worktree add --detach $C/wt-before $A > /dev/null 2>&1 || { echo "worktree FAILED"; exit 1; }
for rep in 1 2; do for tt in 256 512; do
  python3 -I $C/wt-before/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py $tt kblock+exp+reduce 16 $O/step-parts-T$tt-before-$rep.json 4 > $O/step-parts-T$tt-before-$rep.txt 2>&1; echo "step_parts T$tt before $rep exit $?"
  python3 -I $R/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py $tt kblock+exp+reduce 16 $O/step-parts-T$tt-after-$rep.json 4 > $O/step-parts-T$tt-after-$rep.txt 2>&1; echo "step_parts T$tt after $rep exit $?"
done; done
git worktree remove --force $C/wt-before; git worktree prune
echo "== $(date -u +%H:%M:%S) 7 the design a part at a time: the committed pool without the led head, without stealing"
for v in nohead nosteal; do
  rm -rf $C/src_$v; mkdir -p $C/src_$v
  cp $C/srcB/bcir_tensor.h $C/srcB/bcir_decoder_train.c $C/srcB/bcir_decoder_train.h $C/src_$v/ || exit 1
done
python3 -I $T/proto_aff_variants.py $C/srcB/bcir_tensor.c $C/src_nohead/bcir_tensor.c $C/src_nosteal/bcir_tensor.c > /dev/null || exit 1
for v in nohead nosteal; do
  clang $CF -fPIC -shared -I $C/src_$v $C/src_$v/bcir_tensor.c -o $O/bin/libtensor_clang_$v.so -lm || exit 1
  clang $CF -fPIC -shared -I $C/src_$v $C/src_$v/bcir_tensor.c $C/src_$v/bcir_decoder_train.c -o $O/bin/libdec_clang_$v.so -lm || exit 1
done
: > $O/design-gemm.jsonl
for v in nohead nosteal; do for M in 1024 2048 8192; do
  $O/bin/ab_gemm $O/bin/libtensor_clang_B.so $O/bin/libtensor_clang_$v.so 32 4 5 \
    p:0,0,$M,128,128:1,0,128,128,$M p:0,0,$M,256,128:1,0,128,256,$M p:0,0,$M,128,260:1,0,260,128,$M \
    c3:0,0,$M,128,128:3x1,0,128,128,$M c2:0,0,$M,128,256:2x1,0,256,128,$M \
    g:0,1,$M,128,128 g:1,0,128,128,$M | sed "s/^{/{\"variant\":\"$v\",/" >> $O/design-gemm.jsonl || echo "ab_gemm $v $M FAILED"
done; done
: > $O/design-step.jsonl
for v in nohead nosteal; do for tt in 128 256 512; do
  out=$($O/bin/step_ab $O/bin/libdec_clang_B.so $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 4 $tt 14 4 48 2>&1 | tail -1)
  echo "{\"variant\":\"$v\",\"case\":\"geo-dims T$tt kblock+exp+reduce, step\",\"run\":$out}" >> $O/design-step.jsonl
  echo "$v T$tt: $(echo "$out" | cut -c1-260)"
done; done
echo "== $(date -u +%H:%M:%S) 8 the elementwise jobs warm, cold and after a pool job wrote their operands, both units"
for side in A B; do clang $CF -I $C/src$side $T/ew_cold.c $C/src$side/bcir_tensor.c -o $O/bin/ew_cold_$side -lm || exit 1; done
: > $O/elementwise.txt
for rep in 1 2 3; do for side in A B; do for th in 1 4; do for rows in 1024 2048 8192; do
  echo "side $side rep $rep $($O/bin/ew_cold_$side $th $rows 128 15)" >> $O/elementwise.txt
done; done; done; done
grep " rep 1 " $O/elementwise.txt
echo "== $(date -u +%H:%M:%S) 9 the study's three models on both units (build_amx.py, GCC)"
for side in before after; do
  rm -rf $C/geo-$side; mkdir -p $C/geo-$side
  tar -C $S/geo2/work/level2 --exclude=./build --exclude=./baseline --exclude=./bf16 --exclude=./checkpoints --exclude=./results --exclude=__pycache__ -cf - . | tar -C $C/geo-$side -xf - || exit 1
  src=$C/srcA; [ $side = after ] && src=$C/srcB
  python3 -I $D4/geo-amx/build_amx.py $S/geo2/work/level2 $C/geo-$side $src $S/geo2/src/main > $O/study-build-$side.log 2>&1 || { echo "build_amx $side FAILED"; tail -5 $O/study-build-$side.log; exit 1; }
done
mkdir -p $O/cwd
(cd $O/cwd && $PY -I $D4/geo-now/geo_ab.py $C/geo-before $C/geo-after $O/study-ab.json --rounds 16 --steps 3 --models global,query_full,attention --train-t 128,256,512 --infer-t 256,1024,2048) > $O/study-ab.txt 2>&1; echo "geo_ab exit $?"
grep "^{" $O/study-ab.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/compile-limit/geo_compare_static.py $C/geo-after $O/study-vs-pytorch.json --rounds 12 --steps 3) > $O/study-vs-pytorch.txt 2>&1; echo "geo_compare_static exit $?"
tail -n 20 $O/study-vs-pytorch.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
