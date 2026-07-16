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

# Arch-select the micromamba asset + the conda compiler package (so this builds natively on a
# Raspberry Pi 5 / aarch64, not only x86_64). conda-forge publishes mlir=22 for linux-aarch64.
case "$(uname -m)" in
  aarch64|arm64)
    MM_ASSET="micromamba-linux-aarch64"
    MM_SHA256="e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a"
    GXX_PKG="gxx_linux-aarch64"
    ;;
  *)
    MM_ASSET="micromamba-linux-64"
    MM_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
    GXX_PKG="gxx_linux-64"
    ;;
esac
MM_VERSION="2.8.1-0"
BCIR_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/bcir"
PREFIX="${MAMBA_ROOT_PREFIX:-${BCIR_CACHE_ROOT}/mamba}"
MM="${MICROMAMBA:-${BCIR_CACHE_ROOT}/tools/${MM_ASSET}-${MM_VERSION}}"
MM_DIR="$(dirname "${MM}")"
export MAMBA_ROOT_PREFIX="${PREFIX}"

# Never execute a bootstrap binary from a shared temporary directory.  In
# particular, the former /tmp/micromamba default let another local user place
# an executable before a privileged invocation.  The default is now a
# user-owned cache; explicit overrides must meet the same ownership/mode rule.
umask 077
mkdir -p -- "${MM_DIR}" "${PREFIX}"
for directory in "${MM_DIR}" "${PREFIX}"; do
  owner="$(stat -c '%u' -- "${directory}")"
  mode="$(stat -c '%a' -- "${directory}")"
  if [ "${owner}" != "${EUID}" ] || (( (8#${mode} & 0022) != 0 )); then
    echo "[setup_mlir22] refusing non-private directory: ${directory}" >&2
    exit 1
  fi
done

if [ -e "${MM}" ]; then
  if [ -L "${MM}" ] || [ ! -f "${MM}" ] || [ "$(stat -c '%u' -- "${MM}")" != "${EUID}" ]; then
    echo "[setup_mlir22] refusing unowned, non-regular, or symlinked bootstrap: ${MM}" >&2
    exit 1
  fi
  actual="$(sha256sum -- "${MM}" | awk '{print $1}')"
  if [ "${actual}" != "${MM_SHA256}" ]; then
    echo "[setup_mlir22] refusing micromamba with unexpected SHA-256: ${MM}" >&2
    exit 1
  fi
else
  echo "[setup_mlir22] fetching pinned micromamba ${MM_VERSION} (${MM_ASSET})..."
  part="$(mktemp "${MM}.part.XXXXXXXX")"
  trap 'rm -f -- "${part:-}"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    "https://github.com/mamba-org/micromamba-releases/releases/download/${MM_VERSION}/${MM_ASSET}" \
    -o "${part}"
  actual="$(sha256sum -- "${part}" | awk '{print $1}')"
  if [ "${actual}" != "${MM_SHA256}" ]; then
    echo "[setup_mlir22] downloaded micromamba failed SHA-256 verification" >&2
    exit 1
  fi
  chmod 0700 "${part}"
  mv -- "${part}" "${MM}"
  part=""
fi
chmod 0700 "${MM}"

if [ ! -x "${PREFIX}/envs/m22/bin/mlir-opt" ]; then
  echo "[setup_mlir22] creating the MLIR 22 env from conda-forge (~250 MB)..."
  "${MM}" create -y -n m22 -c conda-forge mlir=22 llvmdev=22 "${GXX_PKG}" ninja cmake
fi

echo "[setup_mlir22] toolchain ready at ${PREFIX}/envs/m22"
LD_LIBRARY_PATH="${PREFIX}/envs/m22/lib" "${PREFIX}/envs/m22/bin/mlir-opt" --version | grep -i "LLVM version"
echo "[setup_mlir22] now run: bash tools/local/check_rail22.sh"
