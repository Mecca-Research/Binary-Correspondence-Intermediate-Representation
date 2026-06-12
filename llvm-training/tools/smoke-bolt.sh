#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${TMPDIR:-/tmp}/llvm-training-bolt-smoke

# This CI-safe smoke leg always preserves a baseline and records why the rewrite
# was skipped when llvm-bolt or an explicit profile is unavailable.
exec "$SCRIPT_DIR/run-bolt-experiment.sh" --output "$OUT_DIR"
