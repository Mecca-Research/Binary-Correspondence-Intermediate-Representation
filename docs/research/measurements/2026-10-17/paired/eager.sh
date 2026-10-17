#!/bin/bash
# LM-PAIRED's check added after the runs: eager PyTorch (torch.optim.AdamW defaults) on two of the paired seeds, in the
# same session, so the comparison can be read against the faster of PyTorch's two arms where both ran.
#
#   bash eager.sh OUT TORCH_PYTHON      (after run.sh OUT TORCH_PYTHON; then python3 paired.py OUT/runs)
set -u
OUT=$1; ML=$2
H=docs/research/measurements/2026-10-11/train_comparison.py
O=$OUT/runs
echo "=== head $(git rev-parse --short HEAD) $(date -u +%H:%M:%S)"
for seed in 20261011 1001; do
  D=$OUT/data-$seed
  echo "=== torch-eager seed $seed $(date -u +%H:%M:%S)"
  "$ML" $H train --data "$D" --task lm --backend torch-eager --threads 4 --eval-every 100 --lr-scale 0.25 > "$O/torch-eager-$seed.log" 2>&1; echo "exit $?"
  cp "$D/runs/lm-torch-eager-t4.json" "$O/torch-eager-$seed.json"
  tail -1 "$O/torch-eager-$seed.log" | cut -c1-160
done
echo "EAGER-DONE $(date -u +%H:%M:%S)"
