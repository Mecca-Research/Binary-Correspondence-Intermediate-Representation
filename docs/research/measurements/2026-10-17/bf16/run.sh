#!/bin/bash
# NDT-BF16 (2026-10-17): the records of this directory, in order, one at a time (README.json).
#
#   bash run.sh TMP CLANG23 DATA [STAGE...]    (from the repository root; TMP an empty scratch
#                                               directory, DATA a comparison data directory:
#                                               train_comparison.py prepare --output DATA)
#
# STAGE is any of semantics tininess cvt2 check cols gemm step identity (default: all, in that order).
# semantics.txt, tininess.txt and cvt2.txt need an AMX-BF16 host (they execute TDPBF16PS,
# VDPBF16PS and VCVTNE2PS2BF16); check.txt builds the unit five ways; gemm-*.txt and step-*.txt are
# indicative wall times; identity.json is the comparison harness's identity stage.
set -u
T=$1; CLANG23=$2; DATA=$3; shift 3
STAGES=${*:-semantics tininess cvt2 check cols gemm step identity}
R=docs/research/measurements/2026-10-17/bf16
C=runtime/c
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if want semantics; then
  gcc -O2 -ffp-contract=off -mamx-tile -mamx-bf16 -mavx512f -mavx512bf16 $R/semantics.c -lm -o "$T/semantics"
  { host; "$T/semantics" 200 1; } > $R/semantics.txt 2>&1; echo "semantics exit $?"
fi
if want tininess; then
  gcc -O2 -ffp-contract=off -mamx-tile -mamx-bf16 -mavx512f $R/tininess.c -lm -o "$T/tininess"
  { host; "$T/tininess"; } > $R/tininess.txt 2>&1; echo "tininess exit $?"
fi
if want cvt2; then
  gcc -O2 $R/cvt2.c -o "$T/cvt2"
  { host; "$T/cvt2"; } > $R/cvt2.txt 2>&1; echo "cvt2 exit $?"
fi
if want check; then
  {
    host
    for build in "gcc-13 -O3" "clang-18 -O3" "$CLANG23 -O3" "gcc-13 -O3 -DBCIR_TENSOR_MAX_VARIANT=4" "gcc-13 -O3 -DBCIR_TENSOR_NO_DISPATCH"; do
      set -- $build
      cc=$1; shift
      echo "== $($cc --version | head -1) $*"
      $cc -std=gnu11 "$@" -ffp-contract=off -fno-math-errno -Wall -Wextra -I $C $R/check.c $C/bcir_tensor.c -o "$T/check" -lm -pthread
      "$T/check" 60
      echo "exit $?"
    done
  } > $R/check.txt 2>&1; echo "check done"
fi
if want cols; then
  gcc -O2 -std=c11 -Wall -Wextra $R/cols_check.c -o "$T/cols_check"
  { host; "$T/cols_check"; } > $R/cols_check.txt 2>&1; echo "cols exit $?"
fi
if want gemm; then
  gcc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -I $C $R/gemm.c $C/bcir_tensor.c -o "$T/gemm" -lm -pthread
  { host; "$T/gemm" 4 15; } > $R/gemm-t4.txt 2>&1; echo "gemm t4 exit $?"
  { host; "$T/gemm" 1 9; } > $R/gemm-t1.txt 2>&1; echo "gemm t1 exit $?"
  { host; "$T/gemm" 1 9 4; } > $R/gemm-v4.txt 2>&1; echo "gemm v4 exit $?"
fi
if want step; then
  { host; python3 $R/step.py . 4 8 3; } > $R/step-t4.txt 2>&1; echo "step t4 exit $?"
  { host; python3 $R/step.py . 1 4 2; } > $R/step-t1.txt 2>&1; echo "step t1 exit $?"
fi
if want identity; then
  # six steps of the comparison's decoder through the four builds and a repeat, at one and four
  # threads: one state for the bf16 contract (the tile, AVX-512 and portable instances alike)
  python3 docs/research/measurements/2026-10-11/train_comparison.py identity --data "$DATA" \
    --steps 6 --cc gcc --threads 1,4 --contract bf16+exp+reduce \
    --variants widest,avx512,avx2,portable > "$T/identity.log" 2>&1
  echo "identity exit $?"; cp "$DATA/identity.json" $R/identity.json
fi
echo "RUN-DONE $(date -u +%H:%M:%S)"
