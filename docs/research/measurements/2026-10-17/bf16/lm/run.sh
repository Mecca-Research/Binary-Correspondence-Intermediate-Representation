#!/bin/bash
# LM-BF16 (2026-10-17): the 1,000-step language-model run at four threads, four arms a seed -- BCIR
# under bf16+exp+reduce and under kblock+exp+reduce, torch.compile with fused AdamW under CPU autocast
# to bfloat16 and in FP32 -- five seeds (LM-PAIRED's), one session, the arms' order rotated seed by seed;
# then the check fixed beside the analysis (paired_bf16.py): eager PyTorch under autocast on two seeds.
# Resumable: a finished run is kept.
#
#   bash run.sh OUT TORCH_PYTHON      (from the repository root; TORCH_PYTHON has NumPy and CPU PyTorch)
#
# Writes OUT/data-<seed>/ (train_comparison.py prepare, --seed for all but the default) and
# OUT/runs/<arm>-<seed>.json (the train stage's records), then: python3 paired_bf16.py OUT/runs
set -u
OUT=$1; ML=$2
H=docs/research/measurements/2026-10-11/train_comparison.py
O=$OUT/runs; mkdir -p "$O"
echo "=== head $(git rev-parse --short HEAD) $(date -u +%H:%M:%S), runtime/c against it: $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"
ARMS=(bcir-bf16 torch-compiled-bf16 bcir torch-compiled)
run() {  # arm seed
  local arm=$1 seed=$2 D=$OUT/data-$2 name
  if [ -f "$O/$arm-$seed.json" ]; then echo "=== $arm seed $seed: kept"; return; fi
  echo "=== $arm seed $seed $(date -u +%H:%M:%S)"
  case $arm in  # the train command last in each arm, so the case's status is its own
    bcir-bf16) name=lm-bcir-bf16+exp+reduce-t4
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract bf16+exp+reduce \
        --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    bcir) name=lm-bcir-kblock+exp+reduce-t4
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract kblock+exp+reduce \
        --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    *) name=lm-$arm-t4
      "$ML" $H train --data "$D" --task lm --backend $arm --threads 4 --eval-every 100 --lr-scale 0.25 \
        > "$O/$arm-$seed.log" 2>&1;;
  esac
  echo "exit $?"
  cp "$D/runs/$name.json" "$O/$arm-$seed.json"
  tail -1 "$O/$arm-$seed.log" | cut -c1-200
}
i=0
for seed in 20261011 1001 2002 3003 4004; do
  D=$OUT/data-$seed
  if [ ! -f "$D/prepare.json" ]; then
    echo "=== prepare seed $seed $(date -u +%H:%M:%S)"
    if [ $seed = 20261011 ]; then python3 $H prepare --output "$D" > "$O/prepare-$seed.log" 2>&1
    else python3 $H prepare --output "$D" --seed $seed > "$O/prepare-$seed.log" 2>&1; fi
    echo "exit $?"
  fi
  for j in 0 1 2 3; do run "${ARMS[$(( (i + j) % 4 ))]}" $seed; done
  i=$((i+1))
done
for seed in 20261011 1001; do run torch-eager-bf16 $seed; done
echo "LM-BF16-DONE $(date -u +%H:%M:%S)"
