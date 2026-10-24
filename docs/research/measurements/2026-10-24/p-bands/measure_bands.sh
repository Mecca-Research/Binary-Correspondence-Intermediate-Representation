#!/bin/bash
# P-BANDS's records on the committed tree (HEAD) against the tree before it (HEAD~1), serially, two workers at most:
#  1 hashes     -- the attention's row paths (scratch, fwd, lower, attn, band hashes: no BANDS) before == after, clang and gcc
#  1b bands     -- bands_check.c: the after unit under BANDS against the before unit, every value at its place, every instance
#                  pair, clang and gcc and across them
#  2 attention  -- ab_bands.c (ab_keep.c with every arm's outputs checked): both trees under the trainer's word (EXP|LOWER|BANDS: the tree before
#                  ignores the bit), forward, backward and both, in one process with an A/A control
#  3 phases     -- the backward a phase at a time on both trees (instrument_bwd4.py, instrument_bands.py)
#  4 step       -- ../../2026-10-22/attn-rate/step_ab.c: the training step, both trainers (the after one runs BANDS)
#  4b parts     -- ../../2026-10-23/bwd-keep/step_parts.py: the GEO step at T2048 a part at a time, each tree from its own
#                  checkout (the tree before in a detached worktree), twice
#  5 level 3    -- the GEO attention model built on both units (build_level3, contig, lower, bands_level3): geo_ab.py's
#                  before/after with every trained state and logit equal, and ../compile-limit/static_compare.py against PyTorch
#  5b sdpa      -- ../../2026-10-23/fwd-strip/attn_vs_sdpa.py: the attention as the trainer runs it against PyTorch's own
#  6 placement  -- run_place.sh: the host's bandwidths, the row-major walk's band/prefetch variants, the banded walks
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/bands; O=$C/final; T=$C/tools
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
  $cc $CF -I $src $D3/scores-t/scratch_hash.c $src/bcir_tensor.c -o $O/bin/strict_${cc}_$side -lm || exit 1
  for hsh in "$D3/fwd-strip/fwd_hash.c" "$D3/p-lower/lower_hash.c" "$D/attn-rate/attn_hash.c" "$D/bwd-long/band_hash.c"; do
    $cc $CF -I $src $hsh $src/bcir_tensor.c -o $O/bin/$(basename $hsh .c)_${cc}_$side -lm || exit 1
  done
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c -o $O/bin/libtensor_${cc}_$side.so -lm || exit 1
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
clang -std=c11 -O2 $T/ab_bands.c -o $O/bin/ab_bands -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $D/attn-rate/step_ab.c -o $O/bin/step_ab -ldl -lm || exit 1
clang $CF $T/bands_check.c -o $O/bin/bands_check -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) 1 hashes"
for cc in clang gcc; do
  for hb in strict fwd_hash lower_hash attn_hash band_hash; do
    $O/bin/${hb}_${cc}_A > $O/${hb}_${cc}_A.txt && $O/bin/${hb}_${cc}_B > $O/${hb}_${cc}_B.txt
    cmp -s $O/${hb}_${cc}_A.txt $O/${hb}_${cc}_B.txt && echo "$hb $cc: $(wc -l < $O/${hb}_${cc}_B.txt) lines equal" || echo "$hb $cc: DIFFER"
  done
done
for hb in strict fwd_hash lower_hash attn_hash band_hash; do
  cmp -s $O/${hb}_clang_B.txt $O/${hb}_gcc_B.txt && echo "$hb: clang's lines gcc's" || echo "$hb clang vs gcc: DIFFER"
done
echo "== $(date -u +%H:%M:%S) 1b bands_check"
: > $O/bands-check.txt
for run in "clang clang 1 0 0 0" "clang clang 1 4 4 0" "clang clang 0 1 1 1" "clang clang 0 2 2 1" "clang clang 0 4 1 1" \
           "clang clang 0 1 4 1" "clang clang 0 2 4 1" "clang clang 0 4 2 1" "clang clang 0 1 2 1" "gcc gcc 1 0 0 0" \
           "gcc gcc 0 1 4 1" "clang gcc 1 0 0 0" "gcc clang 0 2 1 1"; do
  set -- $run
  echo "before $1, after $2, heavy $3, forward instance $4, backward instance $5, small $6: $($O/bin/bands_check $O/bin/libtensor_$1_A.so $O/bin/libtensor_$2_B.so $3 $4 $5 $6 | tail -1)" | tee -a $O/bands-check.txt
done
echo "== $(date -u +%H:%M:%S) 2 attention in one process (both under EXP|LOWER|BANDS: 49; FUSED adds 2)"
: > $O/ab-attention.jsonl
for spec in "48 4 2048 4 4 32 4 49" "24 4 2048 4 4 32 1 49" "48 4 1024 4 4 32 4 49" "48 4 512 4 4 32 4 49" \
            "48 4 256 4 4 32 4 49" "48 4 128 4 4 32 4 49" "48 4 2048 4 4 32 4 51" "48 4 2048 4 4 32 4 48" \
            "48 4 2048 4 4 32 4 50" "48 1 2048 8 8 64 4 49" "48 1 1024 4 4 128 4 49" "48 2 1024 8 2 48 4 49" \
            "48 1 4096 8 8 64 4 49" "48 8 128 8 4 48 4 51" \
            "48 4 1000 4 4 32 4 49"; do
  $O/bin/ab_bands $O/bin/libtensor_clang_A.so $O/bin/libtensor_clang_B.so $spec | sed 's/^{/{"compiler":"clang",/' >> $O/ab-attention.jsonl || echo "ab_bands $spec FAILED"
done
$O/bin/ab_bands $O/bin/libtensor_gcc_A.so $O/bin/libtensor_gcc_B.so 48 4 2048 4 4 32 4 49 | sed 's/^{/{"compiler":"gcc",/' >> $O/ab-attention.jsonl
python3 -I $T/fmt_ab.py $O/ab-attention.jsonl
echo "== $(date -u +%H:%M:%S) 3 phases"
python3 -I $T/instrument_bwd4.py $C/srcA/bcir_tensor.c $O/instA.c > /dev/null && python3 -I $T/instrument_bands.py $C/srcB/bcir_tensor.c $O/instB.c > /dev/null || exit 1
clang $CF -I $C/srcA $T/bwd_bench4.c $O/instA.c -o $O/bin/phasesA -lm && clang $CF -I $C/srcB $T/bwd_bench4.c $O/instB.c -o $O/bin/phasesB -lm || exit 1
: > $O/phases.jsonl
for rep in 1 2; do for side in A B; do for a in "4 2048 4 4 32 4 8 49 0" "4 2048 4 4 32 4 8 49 1" "4 2048 4 4 32 1 3 49 0" "4 1024 4 4 32 4 12 49 0"; do
  $O/bin/phases$side $a | sed "s/^{/{\"tree\":\"$side\",/" >> $O/phases.jsonl
done; done; done
echo "== $(date -u +%H:%M:%S) 4 the training step in one process"
: > $O/ab-step.jsonl
for spec in "clang|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 48" \
            "clang|geo-dims T1024 kblock+exp+reduce, step|260 128 4 4 3 256 4 1024 14 4 48" \
            "clang|geo-dims T512 kblock+exp+reduce, step|260 128 4 4 3 256 4 512 14 4 48" \
            "clang|geo-dims T2048 kblock+exp+reduce, 1 thread, step|260 128 4 4 3 256 4 2048 14 1 24" \
            "clang|decoder-9.84M kblock+exp+reduce+attn, step|260 384 8 4 6 1024 8 128 46 4 48" \
            "gcc|geo-dims T2048 kblock+exp+reduce, step|260 128 4 4 3 256 4 2048 14 4 48" \
            "clang|geo-dims T2048 v1 (the libm softmax), step|260 128 4 4 3 256 4 2048 0 4 48"; do
  IFS='|' read -r cc name args <<< "$spec"
  out=$($O/bin/step_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
  echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> $O/ab-step.jsonl
  echo "$cc $name: $(echo "$out" | cut -c1-260)"
done
echo "== $(date -u +%H:%M:%S) 4b the GEO step a part at a time (the job clock), both trees"
rm -rf $C/wt-before; git worktree add --detach $C/wt-before $A > /dev/null 2>&1 || { echo "worktree FAILED"; exit 1; }
for rep in 1 2; do
  python3 -I $C/wt-before/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py 2048 kblock+exp+reduce 8 $O/step-parts-before-$rep.json 4 > $O/step-parts-before-$rep.txt 2>&1; echo "step_parts before $rep exit $?"
  python3 -I $R/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py 2048 kblock+exp+reduce 8 $O/step-parts-after-$rep.json 4 > $O/step-parts-after-$rep.txt 2>&1; echo "step_parts after $rep exit $?"
done
git worktree remove --force $C/wt-before; git worktree prune
echo "== $(date -u +%H:%M:%S) 5 level 3: work dirs and builds (gcc, as the study builds)"
for side in before after; do
  rm -rf $C/geo-$side; mkdir -p $C/geo-$side
  tar -C $S/geo2/work/level2 --exclude=./build --exclude=./baseline --exclude=./bf16 --exclude=./checkpoints --exclude=./results --exclude=__pycache__ -cf - . | tar -C $C/geo-$side -xf - || exit 1
done
for side in before after; do
  src=$C/srcA; [ $side = after ] && src=$C/srcB
  python3 -I $D/infer-attn/build_level3.py $S/geo2/work/level2 $C/geo-$side $src $S/geo2/src/main --scratch > $O/level3-$side.log 2>&1 || { echo "level3 $side FAILED"; tail -5 $O/level3-$side.log; exit 1; }
  python3 -I $D3/att-contig/contig_level3.py $C/geo-$side $src >> $O/level3-$side.log 2>&1 || { echo "contig $side FAILED"; exit 1; }
  python3 -I $D3/p-lower/lower_level3.py $C/geo-$side >> $O/level3-$side.log 2>&1 || { echo "lower $side FAILED"; exit 1; }
  python3 -I $T/bands_level3.py $C/geo-$side >> $O/level3-$side.log 2>&1 || { echo "bands $side FAILED"; exit 1; }
done
mkdir -p $O/cwd
(cd $O/cwd && $PY -I $D4/geo-now/geo_ab.py $C/geo-before $C/geo-after $O/geo-ab.json --rounds 16 --steps 3 --models attention --train-t 512,1024,2048 --infer-t 1024,2048) > $O/geo-ab.txt 2>&1; echo "geo_ab exit $?"
grep "^{" $O/geo-ab.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/compile-limit/static_compare.py $C/geo-after $O/level3-train.json --mode train --rounds 24 --steps 3 --times 512,1024,2048) > $O/level3-train.txt 2>&1; echo "static_compare train exit $?"
tail -n 6 $O/level3-train.txt | cut -c1-300
(cd $O/cwd && $PY -I $D4/compile-limit/static_compare.py $C/geo-after $O/level3-train-confirm.json --mode train --rounds 32 --steps 3 --times 2048) > $O/level3-train-confirm.txt 2>&1; echo "static_compare confirm exit $?"
tail -n 3 $O/level3-train-confirm.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) 5b the attention against PyTorch's scaled_dot_product_attention (the tree after, contract 49)"
for tt in 2048 1024; do
  (cd $O/cwd && $PY -I $D3/fwd-strip/attn_vs_sdpa.py $O/bin/libtensor_clang_B.so $O/sdpa-T$tt.json --t $tt --contract 49 --rounds 24) > $O/sdpa-T$tt.txt 2>&1; echo "sdpa T$tt exit $?"
  tail -n 4 $O/sdpa-T$tt.txt | cut -c1-240
done
echo "== $(date -u +%H:%M:%S) 6 placement on the after unit (membw, the row walk's variants, the banded walks)"
bash $T/run_place.sh $C/srcB $T $O/place || echo "run_place FAILED"
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
