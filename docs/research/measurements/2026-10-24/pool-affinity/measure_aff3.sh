#!/bin/bash
# POOL-AFFINITY's level-3 reading, declared (2026-10-24, 15:36Z) after measure_aff.sh's study section read the GEO
# attention model only through build_amx.py's level-2 overlays, before any of this ran: the attention model built as
# P-BANDS's records build it -- ../../2026-10-22/infer-attn/build_level3.py --scratch, ../../2026-10-23/att-contig/
# contig_level3.py, ../../2026-10-23/p-lower/lower_level3.py and ../p-bands/bands_level3.py (GCC, as the study builds)
# -- on the tree before's unit and on the tree after's (measure_aff.sh's srcA and srcB); ../geo-now/geo_ab.py before
# and after in one process (training T512, T1024 and T2048, inference T1024 and T2048), every trained state and logit
# the same; and ../compile-limit/static_compare.py, the tree after against PyTorch eager and Inductor at each case's
# fixed shapes (training and inference, 24 rounds).
set -o pipefail
S=<SCRATCH>
R=<REPO>
PY=$S/r814/mlenv/bin/python
C=$S/next0/aff; O=$C/final
D=$R/docs/research/measurements/2026-10-22; D3=$R/docs/research/measurements/2026-10-23; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE
cd $R || exit 2
echo "== $(date -u +%H:%M:%S) before=$(git rev-parse HEAD~1) after=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
for side in A B; do [ -f $C/src$side/bcir_tensor.c ] || { echo "missing $C/src$side (measure_aff.sh writes it)"; exit 1; }; done
echo "== $(date -u +%H:%M:%S) level 3: work dirs and builds (gcc, as the study builds)"
for side in before after; do
  rm -rf $C/l3-$side; mkdir -p $C/l3-$side
  tar -C $S/geo2/work/level2 --exclude=./build --exclude=./baseline --exclude=./bf16 --exclude=./checkpoints --exclude=./results --exclude=__pycache__ -cf - . | tar -C $C/l3-$side -xf - || exit 1
done
for side in before after; do
  src=$C/srcA; [ $side = after ] && src=$C/srcB
  python3 -I $D/infer-attn/build_level3.py $S/geo2/work/level2 $C/l3-$side $src $S/geo2/src/main --scratch > $O/l3-build-$side.log 2>&1 || { echo "level3 $side FAILED"; tail -5 $O/l3-build-$side.log; exit 1; }
  python3 -I $D3/att-contig/contig_level3.py $C/l3-$side $src >> $O/l3-build-$side.log 2>&1 || { echo "contig $side FAILED"; exit 1; }
  python3 -I $D3/p-lower/lower_level3.py $C/l3-$side >> $O/l3-build-$side.log 2>&1 || { echo "lower $side FAILED"; exit 1; }
  python3 -I $D4/p-bands/bands_level3.py $C/l3-$side >> $O/l3-build-$side.log 2>&1 || { echo "bands $side FAILED"; exit 1; }
done
mkdir -p $O/cwd3
echo "== $(date -u +%H:%M:%S) level 3: before and after in one process"
(cd $O/cwd3 && $PY -I $D4/geo-now/geo_ab.py $C/l3-before $C/l3-after $O/l3-ab.json --rounds 16 --steps 3 --models attention --train-t 512,1024,2048 --infer-t 1024,2048) > $O/l3-ab.txt 2>&1; echo "geo_ab level3 exit $?"
grep "^{" $O/l3-ab.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) level 3: the tree after against PyTorch"
(cd $O/cwd3 && $PY -I $D4/compile-limit/static_compare.py $C/l3-after $O/l3-vs-pytorch-train.json --mode train --rounds 24 --steps 3 --times 512,1024,2048) > $O/l3-vs-pytorch-train.txt 2>&1; echo "static_compare train exit $?"
tail -n 4 $O/l3-vs-pytorch-train.txt | cut -c1-300
(cd $O/cwd3 && $PY -I $D4/compile-limit/static_compare.py $C/l3-after $O/l3-vs-pytorch-infer.json --mode infer --rounds 24 --steps 3 --times 1024,2048) > $O/l3-vs-pytorch-infer.txt 2>&1; echo "static_compare infer exit $?"
tail -n 3 $O/l3-vs-pytorch-infer.txt | cut -c1-300
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
