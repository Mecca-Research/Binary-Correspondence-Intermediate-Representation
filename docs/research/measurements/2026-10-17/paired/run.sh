#!/bin/bash
# LM-PAIRED (2026-10-17): the 1,000-step language-model run at four threads, BCIR under kblock+exp+reduce
# against torch.compile with fused AdamW, five seeds, one pair a seed, the pair's order alternating seed by
# seed (BCIR first on the first, third and fifth). Resumable: a finished run is kept.
#
#   bash run.sh OUT TORCH_PYTHON      (from the repository root; TORCH_PYTHON has NumPy and CPU PyTorch)
#
# Writes OUT/data-<seed>/ (train_comparison.py prepare, --seed for all but the default) and
# OUT/runs/{bcir,torch-compiled}-<seed>.json (the train stage's records), then: python3 paired.py OUT/runs
set -u
OUT=$1; ML=$2
H=docs/research/measurements/2026-10-11/train_comparison.py
O=$OUT/runs; mkdir -p "$O"
echo "=== head $(git rev-parse --short HEAD) $(date -u +%H:%M:%S), runtime/c against it: $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"
i=0
for seed in 20261011 1001 2002 3003 4004; do
  D=$OUT/data-$seed
  if [ ! -f "$D/prepare.json" ]; then
    echo "=== prepare seed $seed $(date -u +%H:%M:%S)"
    if [ $seed = 20261011 ]; then python3 $H prepare --output "$D" > "$O/prepare-$seed.log" 2>&1
    else python3 $H prepare --output "$D" --seed $seed > "$O/prepare-$seed.log" 2>&1; fi
    echo "exit $?"
  fi
  if [ $((i % 2)) = 0 ]; then order="bcir torch-compiled"; else order="torch-compiled bcir"; fi
  i=$((i+1))
  for b in $order; do
    if [ -f "$O/$b-$seed.json" ]; then echo "=== $b seed $seed: kept"; continue; fi
    echo "=== $b seed $seed $(date -u +%H:%M:%S)"
    if [ $b = bcir ]; then
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract kblock+exp+reduce --eval-every 100 --lr-scale 0.25 > "$O/$b-$seed.log" 2>&1; echo "exit $?"
      cp "$D/runs/lm-bcir-kblock+exp+reduce-t4.json" "$O/$b-$seed.json"
    else
      "$ML" $H train --data "$D" --task lm --backend $b --threads 4 --eval-every 100 --lr-scale 0.25 > "$O/$b-$seed.log" 2>&1; echo "exit $?"
      cp "$D/runs/lm-$b-t4.json" "$O/$b-$seed.json"
    fi
    tail -1 "$O/$b-$seed.log" | cut -c1-200
  done
done
echo "PAIRED-DONE $(date -u +%H:%M:%S)"
