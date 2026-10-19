#!/bin/bash
# BF16-ATTN and ATTN-SOFT (2026-10-19): the records of this directory, in order, one at a time
# (README.json).
#
#   bash run.sh TMP DATA [STAGE...]   (from the repository root; TMP an empty scratch directory, DATA
#                                      a comparison data directory: train_comparison.py prepare
#                                      --output DATA)
#
# STAGE is any of attention steppair identity soft gemm controls (default: all, in that order). The
# base is 9a702a4, this commit's parent. attention times one layer's attention alone under the fused
# binary32 contract and the bf16 one, alternated round by round; steppair runs the base and this tree
# under bf16+exp+reduce+attn and this tree under bf16+exp+reduce+attnbf16 a step each in turn in one
# process (the first two arms' losses equal every round); identity is the comparison harness's
# identity stage. soft times the fused binary32 attention alone, the base's trainer library against
# this tree's in one process (ATTN-SOFT: the same bits); gemm the bf16 GEMM alone over five of the
# decoder's shapes, the same two libraries (whether splitting the AMX kernel into begin, blocks and end
# costs a GEMM anything); controls the A/A controls those library-against-library ratios are read
# against -- gemm over two copies of one library, gemm with its arms' slots swapped, and steppair with
# this tree as its own base (its base/soft an A/A) -- what two libraries' placement alone moves a
# ratio -- and procs.py's gemm pairs over eight processes, the slots swapped every other one.
# attention-*.txt, steppair-*.txt, soft-*.txt, gemm-*.txt and controls-*.txt are indicative wall
# times; identity-*.json are states, bit for bit.
set -u
T=$1; DATA=$2; shift 2
STAGES=${*:-attention steppair identity soft gemm controls}
R=docs/research/measurements/2026-10-19/attnbf16
C=runtime/c
BASE=9a702a4
CC="gcc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -pthread"
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
base() {  # $T/base: the base's runtime/c
  if [ ! -d "$T/base/runtime/c" ]; then mkdir -p "$T/base"; git archive $BASE runtime/c | tar -x -C "$T/base"; fi
}
F="-std=c11 -O3 -ffp-contract=off -fno-math-errno -DBCIR_DT_BUILD_SHARED -shared -fPIC -pthread"
libs() {  # $T/base.so, $T/this.so: the base's and this tree's trainer libraries, the comparison's build
  base
  [ -f "$T/base.so" ] || gcc $F -I "$T/base/$C" "$T/base/$C/bcir_tensor.c" "$T/base/$C/bcir_decoder_train.c" \
    -o "$T/base.so" -lm
  [ -f "$T/this.so" ] || gcc $F -I $C $C/bcir_tensor.c $C/bcir_decoder_train.c -o "$T/this.so" -lm
}
driver() {  # $T/NAME from $R/NAME.c: a driver that dlopens the two libraries
  [ -x "$T/$1" ] || gcc -std=c11 -O2 -Wall -Wextra -Werror $R/$1.c -ldl -lm -o "$T/$1"
}

if want attention; then
  # one layer's attention, forward and backward, fused binary32 against bf16, alternated round by round
  $CC -I $C $R/attention.c $C/bcir_tensor.c -lm -o "$T/attention"
  { host; "$T/attention" 4 61; } > $R/attention-t4.txt 2>&1; echo "attention t4 exit $?"
  { host; "$T/attention" 1 31; } > $R/attention-t1.txt 2>&1; echo "attention t1 exit $?"
fi
if want steppair; then
  # the 9.84M decoder's step: the base and this tree under the fused binary32 attention (their losses
  # equal every round) and this tree under the bf16 attention, a step each in turn in one process
  base
  { host; python3 $R/steppair.py . "$T/base/runtime/c" 4 150; } > $R/steppair-t4.txt 2>&1; echo "steppair t4 exit $?"
  { host; python3 $R/steppair.py . "$T/base/runtime/c" 1 40; } > $R/steppair-t1.txt 2>&1; echo "steppair t1 exit $?"
fi
if want identity; then
  # six steps of the comparison's decoder through the builds and a repeat, at one and four threads:
  # bf16+exp+reduce+attnbf16 one state on every instance of the bf16 attention (AMX, AVX-512 and the
  # portable loops: widest, avx512, avx2 and portable builds); bf16+exp+reduce+attn the state it had
  # before this commit (08da09e8..., the BF16-PANELS record), ATTN-SOFT being bit-identical
  H=docs/research/measurements/2026-10-11/train_comparison.py
  rm -f "$DATA/identity.json"
  python3 $H identity --data "$DATA" --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce+attnbf16 \
    --variants widest,avx512,avx2,portable > "$T/identity-attnbf16.log" 2>&1
  echo "identity attnbf16 exit $?"; cp "$DATA/identity.json" $R/identity-attnbf16.json
  rm -f "$DATA/identity.json"
  python3 $H identity --data "$DATA" --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce+attn \
    --variants widest > "$T/identity-attn.log" 2>&1
  echo "identity attn exit $?"; cp "$DATA/identity.json" $R/identity-attn.json
fi
if want soft; then
  # one layer's fused binary32 attention, forward and backward: the base's library against this tree's
  libs; driver soft
  { host; "$T/soft" "$T/base.so" "$T/this.so" 4 61; } > $R/soft-t4.txt 2>&1; echo "soft t4 exit $?"
  { host; "$T/soft" "$T/base.so" "$T/this.so" 1 31; } > $R/soft-t1.txt 2>&1; echo "soft t1 exit $?"
fi
if want gemm; then
  # the bf16 GEMM alone over five of the decoder's shapes: the base's library against this tree's
  libs; driver gemm
  { host; "$T/gemm" "$T/base.so" "$T/this.so" 4 401; } > $R/gemm-t4.txt 2>&1; echo "gemm t4 exit $?"
  { host; "$T/gemm" "$T/base.so" "$T/this.so" 1 201; } > $R/gemm-t1.txt 2>&1; echo "gemm t1 exit $?"
fi
if want controls; then
  # the A/A controls: gemm over two copies of one library (the same code at two places), with its
  # slots swapped, and repeated; steppair with this tree as its own base, so its base/soft is an A/A
  libs; driver gemm
  cp "$T/base.so" "$T/base-copy.so"; cp "$T/this.so" "$T/this-copy.so"
  rc=0
  {
    host
    for run in "base.so base-copy.so 1 201" "this.so this-copy.so 1 201" "this.so base.so 1 201" \
               "base.so this.so 1 201" "base.so base-copy.so 4 401" "this.so this-copy.so 4 401"; do
      set -- $run
      echo "# gemm $run"; "$T/gemm" "$T/$1" "$T/$2" $3 $4 || rc=1
    done
  } > $R/controls-gemm.txt 2>&1; echo "controls gemm exit $rc"
  # the same pairs over eight processes each, the slots swapped every other process (procs.py): the
  # mean of the processes' ratios, which no one placement decides, beside the A/A's
  { host
    for th in "1 201" "4 401"; do
      set -- $th
      echo "# procs.py gemm base.so this.so $1 $2 8"; python3 $R/procs.py "$T/gemm" "$T/base.so" "$T/this.so" $1 $2 8
      echo "# procs.py gemm base.so base-copy.so $1 $2 8 (A/A)"
      python3 $R/procs.py "$T/gemm" "$T/base.so" "$T/base-copy.so" $1 $2 8
    done; } > $R/controls-procs.txt 2>&1; echo "controls procs exit $?"
  { host; echo "# steppair.py . runtime/c 4 150 (this tree as its own base: base/soft an A/A)"
    python3 $R/steppair.py . $C 4 150; } > $R/controls-steppair-t4.txt 2>&1; echo "controls steppair exit $?"
fi
