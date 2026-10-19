#!/bin/bash
# LM-MULTI (2026-10-19): the 1,000-step language-model run at four threads, four arms a seed -- BCIR
# under bf16+exp+reduce+attnbf16 a hundred steps a run call, under bf16+exp+reduce+attn a hundred
# steps a call and one step a call, and torch.compile with fused AdamW under CPU autocast to bfloat16
# -- LM-PAIRED's five seeds, one session, the arms' order rotated seed by seed. The analysis is
# paired_multi.py, fixed before the runs. Resumable: a finished run is kept.
#
#   bash run.sh OUT TORCH_PYTHON [DATA_SRC]   (from the repository root)
#
# Every BCIR arm runs this tree's comparison harness, which builds the trainer from this tree's
# runtime/c. A call of a hundred steps never crosses an evaluation (every hundred steps), so each
# call is the run's steps between two evaluations. DATA_SRC, if given, holds prepared data-<seed>/
# directories (train_comparison.py prepare: deterministic for a seed), whose prepared files are
# copied; otherwise each seed is prepared here.
set -u
OUT=$1; ML=$2; SRC=${3:-}
H=docs/research/measurements/2026-10-11/train_comparison.py
O=$OUT/runs; mkdir -p "$O"
echo "=== tree: commit $(git rev-parse --short HEAD), runtime/c against it: $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"
echo "=== host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs, $(date -u +%H:%M:%S)"
ARMS=(bcir-attnbf16-k100 torch-compiled-bf16 bcir-attn-k100 bcir-attn-k1)
run() {  # arm seed
  local arm=$1 seed=$2 D=$OUT/data-$2 name rc
  if [ -f "$O/$arm-$seed.json" ]; then echo "=== $arm seed $seed: kept"; return; fi
  echo "=== $arm seed $seed $(date -u +%H:%M:%S)"
  case $arm in
    bcir-attnbf16-k100) name=lm-bcir-bf16+exp+reduce+attnbf16-t4-k100;;
    bcir-attn-k100) name=lm-bcir-bf16+exp+reduce+attn-t4-k100;;
    bcir-attn-k1) name=lm-bcir-bf16+exp+reduce+attn-t4;;
    *) name=lm-$arm-t4;;
  esac
  rm -f "$D/runs/$name.json"
  case $arm in  # the train command last in each arm, so the case's status is its own
    bcir-attnbf16-k100)
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract bf16+exp+reduce+attnbf16 \
        --steps-per-call 100 --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    bcir-attn-k100)
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract bf16+exp+reduce+attn \
        --steps-per-call 100 --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    bcir-attn-k1)
      python3 $H train --data "$D" --task lm --backend bcir --threads 4 --contract bf16+exp+reduce+attn \
        --eval-every 100 --lr-scale 0.25 > "$O/$arm-$seed.log" 2>&1;;
    *)
      "$ML" $H train --data "$D" --task lm --backend $arm --threads 4 --eval-every 100 \
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
      if [ $seed = 20261011 ]; then python3 $H prepare --output "$D" > "$O/prepare-$seed.log" 2>&1
      else python3 $H prepare --output "$D" --seed $seed > "$O/prepare-$seed.log" 2>&1; fi
      echo "exit $?"
    fi
  fi
  for j in 0 1 2 3; do run "${ARMS[$(( (i + j) % 4 ))]}" $seed; done
  i=$((i+1))
done
echo "LM-MULTI-DONE $(date -u +%H:%M:%S)"
