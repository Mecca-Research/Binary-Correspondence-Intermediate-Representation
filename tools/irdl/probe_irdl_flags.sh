#!/usr/bin/env bash
# Discover how the installed mlir-opt exposes IRDL loading. Different LLVM builds
# expose slightly different command surfaces; do not bake assumptions -- probe.
set -euo pipefail

if ! command -v mlir-opt >/dev/null 2>&1; then
  echo "mlir-opt not found on PATH; IRDL projection cannot be validated on this host." >&2
  exit 0
fi

echo "mlir-opt: $(command -v mlir-opt)"
mlir-opt --version | head -1 || true
echo "--- IRDL-related flags ---"
mlir-opt --help 2>&1 | grep -i irdl || echo "(no flag matched 'irdl' -- check this LLVM version's IRDL support)"
