#!/bin/bash
# PACK-ATTR and GEMM-OVERLAP (2026-10-18): the bf16 GEMMs' packs inside the 9.84M decoder's training
# step at four threads, by role and operand; the untransposed B pack's loops swapped; and the packs
# overlapped with the compute in one job a group, per call and per strip. Each comparison two
# libraries alternated step by step in one process, the losses equal every round.
#   bash run.sh OUT     (from the repository root, on a quiet host)
set -eu
OUT=$1; D=docs/research/measurements/2026-10-18/panels/packs; mkdir -p "$OUT"
python3 -I $D/patches.py runtime/c "$OUT"
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD)"
python3 $D/pair.py "$PWD" "$OUT" 4 40 probe swap > "$OUT/attribution-t4.json"
python3 $D/pair.py "$PWD" "$OUT" 4 60 base fused > "$OUT/fused-t4.json"
python3 $D/pair.py "$PWD" "$OUT" 4 60 base fine > "$OUT/fine-t4.json"
