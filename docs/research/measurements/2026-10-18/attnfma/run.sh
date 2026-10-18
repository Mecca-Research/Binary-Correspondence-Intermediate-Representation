#!/bin/bash
# ATTN-FMA (2026-10-18): the records of this directory, in order, one at a time (README.json).
#
#   bash run.sh TMP DATA [STAGE...]    (from the repository root; TMP an empty scratch directory,
#                                       DATA a comparison data directory:
#                                       train_comparison.py prepare --output DATA)
#
# STAGE is any of attention step attribution identity (default: all, in that order).
# attention-*.txt, step-*.txt and attribution-*.json are indicative wall times; identity-*.json are
# the comparison harness's identity stage.
set -u
T=$1; DATA=$2; shift 2
STAGES=${*:-attention step attribution identity}
R=docs/research/measurements/2026-10-18/attnfma
C=runtime/c
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if want attention; then
  gcc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -pthread -I $C $R/attention.c $C/bcir_tensor.c -lm -o "$T/attention"
  { host; "$T/attention" 31 4 128; } > $R/attention-t4.txt 2>&1; echo "attention t4 exit $?"
  { host; "$T/attention" 15 1 128; } > $R/attention-t1.txt 2>&1; echo "attention t1 exit $?"
  { host; "$T/attention" 11 4 512; } > $R/attention-t4-512.txt 2>&1; echo "attention t4 512 exit $?"
fi
if want step; then
  { host; python3 $R/step.py . 4 8 3 bf16+exp+reduce bf16+exp+reduce+attn; } > $R/step-t4.txt 2>&1; echo "step t4 exit $?"
  { host; python3 $R/step.py . 4 8 3 kblock+exp+reduce kblock+exp+reduce+attn; } > $R/step-t4-kblock.txt 2>&1; echo "step t4 kblock exit $?"
  { host; python3 $R/step.py . 1 4 2 bf16+exp+reduce bf16+exp+reduce+attn; } > $R/step-t1.txt 2>&1; echo "step t1 exit $?"
fi
if want attribution; then
  # the step in place, statement by statement (NDT-SCAN's instrument), the two contracts alternated
  A=docs/research/measurements/2026-10-13/scan/attribution.py
  for round in 1 2 3 4 5; do
    for c in bf16+exp+reduce bf16+exp+reduce+attn; do
      name=$( [ "$c" = bf16+exp+reduce ] && echo bf16 || echo attn )
      python3 $A --src $C --data "$DATA" --threads 4 --steps 8 --contract "$c" \
        --out $R/attribution-t4-$name-$round.json > "$T/attribution-$name-$round.log" 2>&1
      echo "attribution $name $round exit $?"
    done
  done
fi
if want identity; then
  # six steps of the comparison's decoder through the four builds and a repeat, at one and four
  # threads: one state for each contract with the attention's products fused (the AVX-512
  # blocks, the AVX2+FMA kernel and the portable fmaf loops alike)
  for c in bf16+exp+reduce+attn kblock+exp+reduce+attn; do
    name=$( [ "$c" = bf16+exp+reduce+attn ] && echo bf16 || echo kblock )
    rm -f "$DATA/identity.json"
    python3 docs/research/measurements/2026-10-11/train_comparison.py identity --data "$DATA" \
      --steps 6 --cc gcc --threads 1,4 --contract "$c" \
      --variants widest,avx512,avx2,portable > "$T/identity-$name.log" 2>&1
    echo "identity $name exit $?"; cp "$DATA/identity.json" $R/identity-$name.json
  done
  # and the contracts without the bit, unchanged: the 20-step kblock state every slice since NDT-SCAN
  # has kept (4bf4e057...) and NDT-BF16's six-step bf16 state (e91ae701...)
  rm -f "$DATA/identity.json"
  python3 docs/research/measurements/2026-10-11/train_comparison.py identity --data "$DATA" \
    --steps 20 --cc gcc --threads 1,4 --contract kblock+exp+reduce --variants widest,avx2 \
    > "$T/identity-unchanged-kblock.log" 2>&1
  echo "identity unchanged kblock exit $?"; cp "$DATA/identity.json" $R/identity-unchanged-kblock.json
  rm -f "$DATA/identity.json"
  python3 docs/research/measurements/2026-10-11/train_comparison.py identity --data "$DATA" \
    --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce --variants widest,avx512 \
    > "$T/identity-unchanged-bf16.log" 2>&1
  echo "identity unchanged bf16 exit $?"; cp "$DATA/identity.json" $R/identity-unchanged-bf16.json
fi
echo "RUN-DONE $(date -u +%H:%M:%S)"
