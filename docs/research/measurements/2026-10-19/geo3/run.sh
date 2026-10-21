#!/bin/bash
# GEO-3 (2026-10-19): the records of this directory, in order, one at a time (README.json).
#
#   bash run.sh TMP [STAGE...]   (from the repository root; TMP an empty scratch directory)
#
# STAGE is any of inventory parent identity frontier scenario harness (default: all, in that order). The
# parent is 12aa6dc, this commit's parent. inventory prints, for each shape and contract, the arenas,
# a layer's cache, the bf16 workspace's need, panels and floor at one worker, four and any pool, and
# the state's bytes under each recompute set at each workspace tier the memory planner walks -- from
# this tree's trainer and from a build without the SwiGLU panels (BCIR_DT_NO_PANELS); parent the
# parent's plan figure and what its harness probe recorded over one step; identity every (recompute
# set, workspace tier) against the all-kept state at the need, with the workspace and panels probes,
# at one and four threads, and the library build without them at four; frontier ABBA rounds of the
# all-kept state at the need against each candidate; scenario the memory planner's choice inside a
# budget against the full micro-batch kept and two accumulated half micro-batches; harness the C
# harness built as the aarch64 CI job builds it (no x86 dispatch: every bf16 product the portable
# loops), timed against the caps test_native_decoder.py gives it, 60 s a run and 300 s under UBSan, and the
# parent's harness built and timed alike after it.
# inventory-*.txt, parent.txt and identity-*.txt are exact; frontier-*.txt, scenario-*.txt and
# harness.txt are indicative wall times on one host, one process at a time.
set -u
T=$1; shift
STAGES=${*:-inventory parent identity frontier scenario harness}
R=docs/research/measurements/2026-10-19/geo3
C=runtime/c
PARENT=12aa6dc
F="-std=c11 -O3 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic -pthread"
PROBES="-DBCIR_DT_WORKSPACE_PROBE -DBCIR_DT_PANELS_PROBE"
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- $C | sha256sum | cut -c1-16)"; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
build() {  # the driver over this tree's runtime/c: the library build, with the probes, without panels
  [ -x "$T/geo3" ] || gcc $F -I $C $R/geo3.c $C/bcir_decoder_train.c $C/bcir_tensor.c -lm -o "$T/geo3"
  [ -x "$T/geo3_probe" ] || gcc $F $PROBES -I $C $R/geo3.c $C/bcir_decoder_train.c $C/bcir_tensor.c -lm \
    -o "$T/geo3_probe"
  [ -x "$T/geo3_nopanels" ] || gcc $F -DBCIR_DT_NO_PANELS -I $C $R/geo3.c $C/bcir_decoder_train.c \
    $C/bcir_tensor.c -lm -o "$T/geo3_nopanels"
}

if want inventory; then
  build
  { host; "$T/geo3" inventory; } > $R/inventory.txt 2>&1; echo "inventory exit $?"
  { host; "$T/geo3_nopanels" inventory; } > $R/inventory-nopanels.txt 2>&1; echo "inventory nopanels exit $?"
fi
if want parent; then
  # the parent's bcir_decoder_workspace_bytes and its harness probe (BCIR_DT_WORKSPACE_PROBE) over a step
  [ -d "$T/parent/$C" ] || { mkdir -p "$T/parent"; git archive $PARENT $C | tar -x -C "$T/parent"; }
  gcc $F -DBCIR_DT_WORKSPACE_PROBE -I "$T/parent/$C" $R/parent.c "$T/parent/$C/bcir_decoder_train.c" \
    "$T/parent/$C/bcir_tensor.c" -lm -o "$T/parent_probe"
  { host; "$T/parent_probe"; } > $R/parent.txt 2>&1; echo "parent exit $?"
fi
if want identity; then
  build
  { host; "$T/geo3_probe" identity 1; } > $R/identity-t1.txt 2>&1; echo "identity t1 exit $?"
  { host; "$T/geo3_probe" identity 4; } > $R/identity-t4.txt 2>&1; echo "identity t4 exit $?"
  { host; "$T/geo3" identity 4; } > $R/identity-t4-library.txt 2>&1; echo "identity t4 library exit $?"
fi
if want frontier; then
  # the 9.84M decoder under the bf16 attention (the fastest contract) at four threads and one, under
  # the blocked binary32 contract at four; Codex's attention model under bf16 and binary32 at four
  build
  { host; "$T/geo3" frontier 9.84M bf16+exp+reduce+attnbf16 4 24 2; } > $R/frontier-9.84M-bf16attn-t4.txt 2>&1
  echo "frontier 9.84M bf16attn t4 exit $?"
  { host; "$T/geo3" frontier 9.84M kblock+exp+reduce+attn 4 16 2; } > $R/frontier-9.84M-fp32-t4.txt 2>&1
  echo "frontier 9.84M fp32 t4 exit $?"
  { host; "$T/geo3" frontier codex-attn bf16+exp+reduce+attn 4 24 3; } > $R/frontier-codex-bf16-t4.txt 2>&1
  echo "frontier codex bf16 t4 exit $?"
  { host; "$T/geo3" frontier codex-attn kblock+exp+reduce+attn 4 24 3; } > $R/frontier-codex-fp32-t4.txt 2>&1
  echo "frontier codex fp32 t4 exit $?"
  { host; "$T/geo3" frontier 9.84M bf16+exp+reduce+attnbf16 1 12 2; } > $R/frontier-9.84M-bf16attn-t1.txt 2>&1
  echo "frontier 9.84M bf16attn t1 exit $?"
fi
if want scenario; then
  # Codex's attention model in its 32 MiB envelope under each contract; the 9.84M decoder in 256 MiB
  build
  for c in kblock+exp+reduce+attn bf16+exp+reduce+attn bf16+exp+reduce+attnbf16; do
    { host; "$T/geo3" scenario codex-attn $c 4 64 6 33554432; } > $R/scenario-codex-$c-t4.txt 2>&1
    echo "scenario codex $c exit $?"
  done
  { host; "$T/geo3" scenario 9.84M bf16+exp+reduce+attnbf16 4 64 3 268435456; } \
    > $R/scenario-9.84M-256MiB-t4.txt 2>&1
  echo "scenario 9.84M exit $?"
fi
if want harness; then
  # this tree's harness and then the parent's, built and timed alike, back to back
  P="-DBCIR_DT_DIVISION_PROBE -DBCIR_DT_WORKSPACE_PROBE -DBCIR_DT_ATTENTION_PROBE -DBCIR_DT_PANELS_PROBE -DBCIR_DT_SCAN_PROBE"
  now() { date +%s.%N; }
  span() { python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$2"; }
  [ -d "$T/parent/$C" ] || { mkdir -p "$T/parent"; git archive $PARENT $C | tar -x -C "$T/parent"; }
  {
    host
    for tree in this parent; do
      if [ $tree = this ]; then D=$C; echo "# this tree's runtime/c"; else D=$T/parent/$C; echo "# the parent $PARENT's runtime/c"; fi
      U="$D/test_decoder_train.c $D/bcir_tensor.c $D/bcir_decoder_train.c"
      for cc in clang gcc; do
        echo "# $($cc --version | head -1)"
        $cc -std=c2x $P -DBCIR_TENSOR_NO_DISPATCH -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -I $D $U \
          -o "$T/portable-$tree-$cc" -lm -pthread || exit 1
        t1=$(now); "$T/portable-$tree-$cc" > "$T/portable-$tree-$cc.out" 2>&1; rc=$?; t2=$(now)
        echo "$tree, $cc portable: run $(span $t1 $t2) s (cap 60), exit $rc, PASS $(grep -c PASS "$T/portable-$tree-$cc.out")"
        $cc -std=c11 -O1 -g -ffp-contract=off -fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all $P \
          -DBCIR_TENSOR_NO_DISPATCH -I $D $U -o "$T/ubsan-$tree-$cc" -lm -pthread || exit 1
        t1=$(now); "$T/ubsan-$tree-$cc" > "$T/ubsan-$tree-$cc.out" 2>&1; rc=$?; t2=$(now)
        echo "$tree, $cc portable UBSan: run $(span $t1 $t2) s (cap 300), exit $rc, PASS $(grep -c PASS "$T/ubsan-$tree-$cc.out"), runtime errors $(grep -c 'runtime error' "$T/ubsan-$tree-$cc.out")"
      done
    done
  } > $R/harness.txt 2>&1
  echo "harness exit $?"
fi
