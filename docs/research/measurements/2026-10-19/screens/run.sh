#!/bin/bash
# Screens (2026-10-19): the records of this directory (README.json), one at a time.
#
#   bash run.sh TMP   (from the repository root; TMP an empty scratch directory)
#
# firstpass: the ceiling of FACTS-PASS -- a run call's later steps taking the weights' and moments'
# maxima and finiteness from the previous commit instead of reading w, m1 and m2 in the update's
# first pass. firstpass.patch points the first pass's w, m1 and m2 at an untouched read-only
# anonymous mapping (one zero page, so the reads cost no memory traffic); every step of the bench
# still skips the preflight by adamw_bounded's proof, so the patched trainer computes the tree's
# state and pair.py holds the two arms' losses equal. aa: pair.py over two builds of this tree's
# runtime/c, an A/A -- what two libraries' placement alone moves the ratio on this host, the band a
# screen's ratio is read against. Wall time: indicative, never a gate.
set -u
T=$1
R=docs/research/measurements/2026-10-19/screens
host() { echo "# host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) vCPUs; head $(git rev-parse --short HEAD), runtime/c against it $(git diff HEAD -- runtime/c | sha256sum | cut -c1-16)"; }
mkdir -p "$T/firstpass"
cp runtime/c/bcir_tensor.c runtime/c/bcir_tensor.h runtime/c/bcir_decoder_train.h runtime/c/bcir_decoder_train.c "$T/firstpass/"
(cd "$T/firstpass" && patch -s -p3 < "$OLDPWD/$R/firstpass.patch") || { echo "firstpass.patch does not apply"; exit 1; }
{ host; python3 $R/pair.py . runtime/c "$T/firstpass" 4 80 bf16+exp+reduce+attn; } > $R/firstpass-t4.txt 2>&1; echo "firstpass t4 exit $?"
{ host; python3 $R/pair.py . runtime/c "$T/firstpass" 1 24 bf16+exp+reduce+attn; } > $R/firstpass-t1.txt 2>&1; echo "firstpass t1 exit $?"
{ host; python3 $R/pair.py . runtime/c runtime/c 4 80 bf16+exp+reduce+attn; } > $R/aa-t4.txt 2>&1; echo "aa t4 exit $?"
{ host; python3 $R/pair.py . runtime/c runtime/c 1 24 bf16+exp+reduce+attn; } > $R/aa-t1.txt 2>&1; echo "aa t1 exit $?"
