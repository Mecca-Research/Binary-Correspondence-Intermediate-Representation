#!/bin/bash
# PANELS-WRITER (2026-10-18): the SwiGLU panel writers' candidates against the shipped writer, then the
# shipped writer's time split, at the decoder's shape (1024 x 1024) on huge pages, each in one process
# with its arms rotated round by round; every candidate's panels compared with the shipped writer's byte
# for byte every round. Scratch copies only: the tree is not edited.
#   bash run.sh OUT     (from the repository root, on a quiet host)
set -eu
OUT=$1; D=docs/research/measurements/2026-10-18/panels/writer; mkdir -p "$OUT"
cp runtime/c/bcir_tensor.h "$OUT/"
python3 -I $D/patch_v4.py runtime/c/bcir_tensor.c "$OUT/bcir_tensor_v4.c"
python3 -I $D/split.py "$OUT/bcir_tensor_v4.c" "$OUT/bcir_tensor_split.c"
F="-std=c11 -O3 -ffp-contract=off -fno-math-errno -Wall -Wextra"
gcc $F -I "$OUT" $D/pw.c "$OUT/bcir_tensor_v4.c" -o "$OUT/pw" -lm -pthread
gcc $F -I "$OUT" $D/pw_split.c "$OUT/bcir_tensor_split.c" -o "$OUT/pw_split" -lm -pthread
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD)"
for t in 4 1; do
  r=41; [ $t = 1 ] && r=21
  "$OUT/pw" $t $r > "$OUT/screen-t$t.txt"; "$OUT/pw_split" $t $r > "$OUT/split-t$t.txt"
done
