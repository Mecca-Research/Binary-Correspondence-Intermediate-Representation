#!/bin/bash
# BF16-ATTR, BF16-KERN and BF16-PANELS-1 (2026-10-18): the records of this directory, in order, one at
# a time (README.json).
#
#   bash run.sh TMP DATA ML [STAGE...]   (from the repository root; TMP an empty scratch directory,
#                                         DATA a comparison data directory:
#                                         train_comparison.py prepare --output DATA; ML a Python
#                                         with PyTorch, for the oneDNN reference)
#
# STAGE is any of phases kern onednn block steppair ab identity (default: all, in that order). The
# base is d5bc457, this commit's parent: phases times the base's GEMM phases and this commit's back
# to back, kern times its candidates on the base's packs against the base's kernel, steppair runs
# the base, this commit without its panels (BCIR_DT_NO_PANELS) and this commit a step each in turn
# in one process, and ab runs the base, the base with this commit's AMX kernel (kern.patch) and
# this commit in place, statement by statement, rotated round by round. Every record but
# identity-*.json is an indicative wall time; identity-*.json are the comparison harness's identity
# stage.
set -u
T=$1; DATA=$2; ML=$3; shift 3
STAGES=${*:-phases kern onednn block steppair ab identity}
R=docs/research/measurements/2026-10-18/panels
C=runtime/c
BASE=d5bc457
CC="gcc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -pthread"
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
trees() {  # $T/base: the base's runtime/c; $T/kern: the base with this commit's AMX kernel
  if [ ! -d "$T/base/runtime/c" ]; then
    mkdir -p "$T/base" "$T/kern"
    git archive $BASE runtime/c | tar -x -C "$T/base"
    git archive $BASE runtime/c | tar -x -C "$T/kern"
    (cd "$T/kern" && git apply "$OLDPWD/$R/kern.patch") || { echo "kern.patch does not apply"; exit 1; }
  fi
}

if want phases; then
  # the bf16 GEMM's phases at the 9.84M decoder's 13 shapes (A pack, B pack, compute, the whole entry)
  # and the tile unit's peak, the base's tensor unit and this commit's back to back
  trees
  $CC -I "$T/base/runtime/c" $R/phases.c -lm -o "$T/phases-base"
  $CC -I $C $R/phases.c -lm -o "$T/phases"
  { host; "$T/phases-base" 4 31; } > $R/phases-base-t4.txt 2>&1; echo "phases base t4 exit $?"
  { host; "$T/phases" 4 31; } > $R/phases-t4.txt 2>&1; echo "phases t4 exit $?"
  { host; "$T/phases-base" 1 21; } > $R/phases-base-t1.txt 2>&1; echo "phases base t1 exit $?"
  { host; "$T/phases" 1 21; } > $R/phases-t1.txt 2>&1; echo "phases t1 exit $?"
fi
if want kern; then
  # the AMX kernel's candidates against the kernel before BF16-KERN, on the base's packs
  trees
  $CC -I "$T/base/runtime/c" $R/kern.c -lm -o "$T/kern-cand"
  { host; "$T/kern-cand" 4 21; } > $R/kern-t4.txt 2>&1; echo "kern t4 exit $?"
  { host; "$T/kern-cand" 1 11; } > $R/kern-t1.txt 2>&1; echo "kern t1 exit $?"
fi
if want onednn; then
  # oneDNN's bf16 matmul through PyTorch on the same 13 shapes
  { host; "$ML" $R/mm.py 4 31; } > $R/onednn-t4.txt 2>&1; echo "onednn t4 exit $?"
  { host; "$ML" $R/mm.py 1 21; } > $R/onednn-t1.txt 2>&1; echo "onednn t1 exit $?"
fi
if want block; then
  # one layer's SwiGLU block with binary32 tensors and packs and with the panels, alternated
  $CC -I $C $R/block.c $C/bcir_tensor.c -lm -o "$T/block"
  { host; "$T/block" 4 21; } > $R/block-t4.txt 2>&1; echo "block t4 exit $?"
  { host; "$T/block" 1 11; } > $R/block-t1.txt 2>&1; echo "block t1 exit $?"
fi
if want steppair; then
  # the 9.84M decoder's step: the base, this commit without its panels and this commit, a step each in
  # turn in one process (their three losses equal every round)
  trees
  { host; python3 $R/steppair.py . "$T/base/runtime/c" 4 150; } > $R/steppair-t4.txt 2>&1; echo "steppair t4 exit $?"
  { host; python3 $R/steppair.py . "$T/base/runtime/c" 1 40; } > $R/steppair-t1.txt 2>&1; echo "steppair t1 exit $?"
fi
if want ab; then
  # the step in place, statement by statement (NDT-SCAN's instrument), three trees rotated
  trees
  A=docs/research/measurements/2026-10-13/scan/attribution.py
  mkdir -p $R/ab
  for round in 1 2 3 4 5 6 7 8 9 10; do
    case $((round % 3)) in 1) order="base kern panels";; 2) order="kern panels base";; *) order="panels base kern";; esac
    for arm in $order; do
      case $arm in base) src="$T/base/runtime/c";; kern) src="$T/kern/runtime/c";; *) src=$C;; esac
      python3 $A --src "$src" --data "$DATA" --threads 4 --steps 8 --contract bf16+exp+reduce+attn \
        --out $R/ab/attribution-t4-$arm-$round.json > "$T/ab-$arm-$round.log" 2>&1
      echo "ab $arm $round exit $?"
    done
  done
  python3 $R/ab_summary.py $R/ab 10 > $R/ab/summary.json; echo "ab summary exit $?"
fi
if want identity; then
  # six steps of the comparison's decoder -- every SwiGLU block of it on the panels -- through the four
  # builds and a repeat, at one and four threads: the state of bf16+exp+reduce+attn before this commit
  # (08da09e8...); bf16+exp+reduce's (e91ae701...); and kblock's 20-step state (4bf4e057...), which no
  # panel reaches
  H=docs/research/measurements/2026-10-11/train_comparison.py
  rm -f "$DATA/identity.json"
  python3 $H identity --data "$DATA" --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce+attn \
    --variants widest,avx512,avx2,portable > "$T/identity-bf16-attn.log" 2>&1
  echo "identity bf16+attn exit $?"; cp "$DATA/identity.json" $R/identity-bf16-attn.json
  rm -f "$DATA/identity.json"
  python3 $H identity --data "$DATA" --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce \
    --variants widest,avx512 > "$T/identity-bf16.log" 2>&1
  echo "identity bf16 exit $?"; cp "$DATA/identity.json" $R/identity-bf16.json
  rm -f "$DATA/identity.json"
  python3 $H identity --data "$DATA" --steps 20 --cc gcc --threads 1,4 --contract kblock+exp+reduce \
    --variants widest > "$T/identity-kblock.log" 2>&1
  echo "identity kblock exit $?"; cp "$DATA/identity.json" $R/identity-kblock.json
fi
