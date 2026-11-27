#!/usr/bin/env bash
# Native ABI 6 MIXER-EXT (GQA, SWA, DeltaNet, KDA, MLA, GDN-2 + G1/QK-Norm/Canon/mHC/online-softmax
# + Muon/Q8-Muon/Adafactor/Q8-BF16Grad) verified under C11 and C23.
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: mixer_ext.sh <C11> <C23>" >&2; exit 2; }
for harness in "$@"; do "$harness"; done
