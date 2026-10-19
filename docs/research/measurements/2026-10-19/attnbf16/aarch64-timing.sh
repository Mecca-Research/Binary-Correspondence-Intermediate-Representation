#!/bin/bash
# BF16-ATTN (2026-10-19): the C harness's run on a host without the x86 instances, timed against the
# caps test_native_decoder.py gives it -- each harness run 60 s, the UBSan run 300 s. The aarch64 CI
# job builds the trainer with no x86 dispatch (BCIR_MM_DISPATCH is x86-64 only), so every bf16
# GEMM, every bf16 attention and the references run as the portable loops; BCIR_TENSOR_NO_DISPATCH
# builds the same code here, as the harness test builds it (its probes, its warnings, -Werror),
# under clang and GCC. A time on this host is not the aarch64 runner's: it says how near the caps
# the portable run is, which the CI job then measures. Wall time: indicative, never a gate.
#
#   bash aarch64-timing.sh TMP   (from the repository root; TMP an empty scratch directory)
set -u
T=$1; C=runtime/c
P="-DBCIR_DT_DIVISION_PROBE -DBCIR_DT_WORKSPACE_PROBE -DBCIR_DT_ATTENTION_PROBE -DBCIR_DT_PANELS_PROBE -DBCIR_DT_SCAN_PROBE"
U="$C/test_decoder_train.c $C/bcir_tensor.c $C/bcir_decoder_train.c"
now() { date +%s.%N; }
span() { python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$2"; }
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"
for cc in clang gcc; do
  echo "# $($cc --version | head -1)"
  t0=$(now)
  $cc -std=c2x $P -DBCIR_TENSOR_NO_DISPATCH -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -I $C $U \
    -o "$T/portable-$cc" -lm || exit 1
  t1=$(now); "$T/portable-$cc" > "$T/portable-$cc.out" 2>&1; rc=$?; t2=$(now)
  echo "$cc portable: build $(span $t0 $t1) s, run $(span $t1 $t2) s (cap 60), exit $rc, PASS $(grep -c PASS "$T/portable-$cc.out")"
  t0=$(now)
  $cc -std=c11 -O1 -g -ffp-contract=off -fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all $P \
    -DBCIR_TENSOR_NO_DISPATCH -I $C $U -o "$T/ubsan-$cc" -lm || exit 1
  t1=$(now); "$T/ubsan-$cc" > "$T/ubsan-$cc.out" 2>&1; rc=$?; t2=$(now)
  echo "$cc portable UBSan: build $(span $t0 $t1) s, run $(span $t1 $t2) s (cap 300), exit $rc, PASS $(grep -c PASS "$T/ubsan-$cc.out"), runtime errors $(grep -c 'runtime error' "$T/ubsan-$cc.out")"
done
