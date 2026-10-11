#!/usr/bin/env bash
# The same native decoder laws over manifest-built harnesses: C11 and C23, and the C23 harness
# rebuilt with the tensor kernels held to their portable C and AVX2-capped variants (the default
# build runs the widest variant the host has), so each variant is held to the scalar references.
set -euo pipefail
[ "$#" -eq 4 ] || { echo "usage: decoder_train.sh <C11> <C23> <C23 portable> <C23 AVX2-capped>" >&2; exit 2; }
for harness in "$@"; do "$harness"; done
