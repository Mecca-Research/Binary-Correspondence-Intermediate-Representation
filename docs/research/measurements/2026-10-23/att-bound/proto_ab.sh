#!/bin/bash
# ATT-BOUND's prototype runs: the tree at REPO's HEAD (A) against a prototype's edit of it (P) -- the values first,
# then the time, both units in one process beside an A/A control.
#   usage: proto_ab.sh REPO OUT PROTO.py stored|scratch
#   stored:  ../p-lower/lower_hash.c (208 cases), ../../2026-10-22/attn-rate/attn_hash.c (88) and ../fwd-strip/fwd_hash.c
#            (288), A against P, clang and gcc, and P built with copies and strips from 16 positions; ../bwd-keep/ab_keep.c
#            at four shapes, 32 rounds; ../../2026-10-22/attn-rate/step_ab.c, the GEO step at T2048 and T1024 (clang)
#            and T2048 (gcc), 32 rounds
#   scratch: attn_hash.c (the scratch forward's y among its cases), likewise; ab_keep.c's forward under contract 9
#            (EXP + SCRATCH) at three shapes, 32 rounds (its backward column reads what the scratch forward left)
set -u
R=$(cd "$1" && pwd); O=$2; PROTO=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); MODE=$4
H=$(cd "$(dirname "$0")" && pwd); D=$H/../../2026-10-22; D3=$H/..
cd "$R"; T=$(git rev-parse HEAD)
echo "== $(date -u +%H:%M:%S) tree=$T dirty=$(git status --short | wc -l) prototype=$(basename "$PROTO") $MODE"
rm -rf "$O"; mkdir -p "$O/srcA" "$O/srcP" "$O/bin"
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do git show "$T:runtime/c/$f" > "$O/srcA/$f"; done
cp "$O/srcA/bcir_tensor.h" "$O/srcA/bcir_decoder_train.c" "$O/srcA/bcir_decoder_train.h" "$O/srcP/"
python3 -I "$PROTO" "$O/srcA/bcir_tensor.c" "$O/srcP/bcir_tensor.c" || exit 1
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
if [ "$MODE" = stored ]; then HASHES="$D3/p-lower/lower_hash.c $D/attn-rate/attn_hash.c $D3/fwd-strip/fwd_hash.c"
else HASHES="$D/attn-rate/attn_hash.c"; fi
for cc in clang gcc; do
  for side in A P; do
    for hc in $HASHES; do $cc $CF -I "$O/src$side" "$hc" "$O/src$side/bcir_tensor.c" -o "$O/bin/$(basename "$hc" .c)_${cc}_$side" -lm || exit 1; done
    $cc $CF -fPIC -shared -I "$O/src$side" "$O/src$side/bcir_tensor.c" -o "$O/bin/libtensor_${cc}_$side.so" -lm || exit 1
    $cc $CF -fPIC -shared -I "$O/src$side" "$O/src$side/bcir_tensor.c" "$O/src$side/bcir_decoder_train.c" -o "$O/bin/libdec_${cc}_$side.so" -lm || exit 1
  done
  for hc in $HASHES; do $cc $CF -DBCIR_ATT_CONTIG_FROM=16 -DBCIR_ATT_STRIP_FROM=16 -I "$O/srcP" "$hc" "$O/srcP/bcir_tensor.c" -o "$O/bin/$(basename "$hc" .c)_${cc}_P16" -lm || exit 1; done
done
clang -std=c11 -O2 "$D3/bwd-keep/ab_keep.c" -o "$O/bin/ab_keep" -ldl -lm || exit 1
clang -std=c11 -O2 -I "$O/srcA" "$D/attn-rate/step_ab.c" -o "$O/bin/step_ab" -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) values"
for cc in clang gcc; do for hc in $HASHES; do k=$(basename "$hc" .c)
  for side in A P P16; do "$O/bin/${k}_${cc}_$side" > "$O/${k}_${cc}_$side.txt" || echo "$k $cc $side FAILED"; done
  cmp -s "$O/${k}_${cc}_A.txt" "$O/${k}_${cc}_P.txt" && echo "$k $cc A == P ($(wc -l < "$O/${k}_${cc}_P.txt") lines)" || echo "$k $cc A != P"
  cmp -s "$O/${k}_${cc}_A.txt" "$O/${k}_${cc}_P16.txt" && echo "$k $cc A == P16" || echo "$k $cc A != P16"
done; done
echo "== $(date -u +%H:%M:%S) the attention, A, P and A again in one process, 32 rounds"
if [ "$MODE" = stored ]; then SPECS=("4 2048 4 4 32 4 17" "4 2048 4 4 32 1 17" "4 1024 4 4 32 4 17" "1 4096 8 8 64 4 18")
else SPECS=("4 2048 4 4 32 4 9" "4 2048 4 4 32 1 9" "4 1024 4 4 32 4 9"); fi
for spec in "${SPECS[@]}"; do
  out=$("$O/bin/ab_keep" "$O/bin/libtensor_clang_A.so" "$O/bin/libtensor_clang_P.so" 32 $spec) || { echo "ab_keep $spec FAILED"; continue; }
  echo "$out" > "$O/ab_$(echo $spec | tr ' ' _).json"; echo "ab_keep $spec done"
done
if [ "$MODE" = stored ]; then
  echo "== $(date -u +%H:%M:%S) the training step at the GEO dimensions, A, P and P again, 32 rounds"
  for spec in "clang|T2048|260 128 4 4 3 256 4 2048 14 4" "clang|T1024|260 128 4 4 3 256 4 1024 14 4" "gcc|T2048|260 128 4 4 3 256 4 2048 14 4"; do
    IFS='|' read -r cc name args <<< "$spec"
    out=$("$O/bin/step_ab" "$O/bin/libdec_${cc}_A.so" "$O/bin/libdec_${cc}_P.so" $args 32 2>&1 | tail -1)
    echo "{\"compiler\":\"$cc\",\"case\":\"$name\",\"run\":$out}" >> "$O/step.jsonl"; echo "step $cc $name done"
  done
fi
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
