#!/usr/bin/env bash
# The same native decoder laws over manifest-built C11 and C23 harnesses.
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: decoder_train.sh <C11 harness> <C23 harness>" >&2; exit 2; }
"$1"
"$2"
