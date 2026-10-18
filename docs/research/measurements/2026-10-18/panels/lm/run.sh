#!/bin/bash
# LM-PANELS (2026-10-18): the 1,000-step language-model run at four threads, three arms a seed -- BCIR
# under bf16+exp+reduce+attn with BF16-KERN and BF16-PANELS-1 (HEAD_TREE) and before them (BASE_TREE,
# d5bc457), and torch.compile with fused AdamW under CPU autocast to bfloat16 -- LM-PAIRED's five
# seeds, one session, the arms' order rotated seed by seed. The analysis is paired_panels.py, fixed
# before the runs. Resumable: a finished run is kept.
#
#   bash run.sh OUT TORCH_PYTHON HEAD_TREE BASE_TREE [DATA_SRC]
#
# HEAD_TREE and BASE_TREE are checkouts of the two commits (git worktree); each BCIR arm runs that
# tree's comparison harness, which builds the trainer from that tree's runtime/c. DATA_SRC, if given,
# holds prepared data-<seed>/ directories (train_comparison.py prepare: deterministic for a seed),
# whose prepared files are copied; otherwise each seed is prepared here.
set -u
OUT=$1; ML=$2; HEAD_TREE=$3; BASE_TREE=$4; SRC=${5:-}
H=docs/research/measurements/2026-10-11/train_comparison.py
O=$OUT/runs; mkdir -p "$O"
for t in "$HEAD_TREE" "$BASE_TREE"; do
  echo "=== tree $t: commit $(git -C "$t" rev-parse --short HEAD), runtime/c against it: $(git -C "$t" diff HEAD -- runtime/c | sha256sum | cut -c1-16)"
done
echo "=== host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs, $(date -u +%H:%M:%S)"
ARMS=(bcir-panels torch-compiled-bf16 bcir-base)
run() {  # arm seed
  local arm=$1 seed=$2 D=$OUT/data-$2 name rc
  if [ -f "$O/$arm-$seed.json" ]; then echo "=== $arm seed $seed: kept"; return; fi
  echo "=== $arm seed $seed $(date -u +%H:%M:%S)"
  # the two BCIR arms write one record name: each run's record is its own or there is none
  case $arm in bcir-panels|bcir-base) name=lm-bcir-bf16+exp+reduce+attn-t4;; *) name=lm-$arm-t4;; esac
  rm -f "$D/runs/$name.json"
  case $arm in  # the train command last in each arm, so the case's status is its own
    bcir-panels) name=lm-bcir-bf16+exp+reduce+attn-t4
      python3 "$HEAD_TREE/$H" train --data "$D" --task lm --backend bcir --threads 4 \
        --contract bf16+exp+reduce+attn --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    bcir-base) name=lm-bcir-bf16+exp+reduce+attn-t4
      python3 "$BASE_TREE/$H" train --data "$D" --task lm --backend bcir --threads 4 \
        --contract bf16+exp+reduce+attn --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    *) name=lm-$arm-t4
      "$ML" "$HEAD_TREE/$H" train --data "$D" --task lm --backend $arm --threads 4 --eval-every 100 \
        --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
  esac
  rc=$?
  echo "exit $rc"
  if [ $rc -ne 0 ] || [ ! -f "$D/runs/$name.json" ]; then echo "STOP: $arm seed $seed left no record"; exit 1; fi
  cp "$D/runs/$name.json" "$O/$arm-$seed.json"
  tail -1 "$O/$arm-$seed.log" | cut -c1-200
}
i=0
for seed in 20261011 1001 2002 3003 4004; do
  D=$OUT/data-$seed
  if [ ! -f "$D/prepare.json" ]; then
    if [ -n "$SRC" ] && [ -f "$SRC/data-$seed/prepare.json" ]; then
      echo "=== copy the prepared seed $seed from $SRC"
      mkdir -p "$D"; find "$SRC/data-$seed" -maxdepth 1 -type f -exec cp {} "$D/" \;
    else
      echo "=== prepare seed $seed $(date -u +%H:%M:%S)"
      if [ $seed = 20261011 ]; then python3 "$HEAD_TREE/$H" prepare --output "$D" > "$O/prepare-$seed.log" 2>&1
      else python3 "$HEAD_TREE/$H" prepare --output "$D" --seed $seed > "$O/prepare-$seed.log" 2>&1; fi
      echo "exit $?"
    fi
  fi
  for j in 0 1 2; do run "${ARMS[$(( (i + j) % 3 ))]}" $seed; done
  i=$((i+1))
done
echo "LM-PANELS-DONE $(date -u +%H:%M:%S)"
