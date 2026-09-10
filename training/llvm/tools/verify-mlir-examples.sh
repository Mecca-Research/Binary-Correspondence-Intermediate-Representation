#!/usr/bin/env bash
# Registry-driven tiered MLIR validation. Missing optional tools are reported as
# reduced coverage; pass --require-tools in the MLIR-specific CI rail.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/verify-mlir-lowering.py" "$@"
