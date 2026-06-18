#!/usr/bin/env bash
# Set up a TRUE MLIR 22 toolchain LOCALLY, for sandboxes where apt.llvm.org (the usual
# MLIR-22 source) is blocked by the network policy but conda-forge IS reachable. The
# stock Ubuntu archive only ships MLIR up to 18, so an 18 build (build_mlir.sh) validates
# the C++/pass logic but not the 22-only rules (IRDL named-operand syntax, the Symbol
# verifier tightenings, 22 deprecations). conda-forge ships real mlir=22.1.x dev libs +
# an ABI-matched compiler, which closes that gap.
#
# Installs micromamba + an 'm22' env (mlir + llvmdev + gxx_linux-64 + ninja + cmake).
# Idempotent. After this, run tools/local/check_rail22.sh to build bcir-opt against 22
# and run the full rail. CI still uses apt.llvm.org (the authoritative gate); this is a
# local convenience only.
set -euo pipefail

PREFIX="${MAMBA_ROOT_PREFIX:-/tmp/mamba}"
MM="${MICROMAMBA:-/tmp/micromamba}"
export MAMBA_ROOT_PREFIX="${PREFIX}"

# Arch-select the micromamba asset + the conda compiler package (so this builds natively on a
# Raspberry Pi 5 / aarch64, not only x86_64). conda-forge publishes mlir=22 for linux-aarch64.
case "$(uname -m)" in
  aarch64|arm64) MM_ASSET="micromamba-linux-aarch64"; GXX_PKG="gxx_linux-aarch64" ;;
  *)             MM_ASSET="micromamba-linux-64";       GXX_PKG="gxx_linux-64" ;;
esac

if [ ! -x "${MM}" ]; then
  echo "[setup_mlir22] fetching micromamba (${MM_ASSET})..."
  curl -fsSL "https://github.com/mamba-org/micromamba-releases/releases/latest/download/${MM_ASSET}" \
    -o "${MM}"
  chmod +x "${MM}"
fi

if [ ! -x "${PREFIX}/envs/m22/bin/mlir-opt" ]; then
  echo "[setup_mlir22] creating the MLIR 22 env from conda-forge (~250 MB)..."
  "${MM}" create -y -n m22 -c conda-forge mlir=22 llvmdev=22 "${GXX_PKG}" ninja cmake
fi

echo "[setup_mlir22] toolchain ready at ${PREFIX}/envs/m22"
LD_LIBRARY_PATH="${PREFIX}/envs/m22/lib" "${PREFIX}/envs/m22/bin/mlir-opt" --version | grep -i "LLVM version"
echo "[setup_mlir22] now run: bash tools/local/check_rail22.sh"
