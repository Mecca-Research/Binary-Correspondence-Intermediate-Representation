#!/usr/bin/env bash
# Set up a TRUE MLIR toolchain of the rail's major LOCALLY, for sandboxes where apt.llvm.org
# (the usual source) is blocked by the network policy but conda-forge IS reachable. The
# stock Ubuntu archive only ships MLIR up to 18, so an 18 build (build_mlir.sh) validates
# the C++/pass logic but not the newer-major-only rules (IRDL named-operand syntax, the
# Symbol verifier tightenings, current deprecations). conda-forge ships real mlir dev libs +
# an ABI-matched compiler, which closes that gap.
#
# MLIR_MAJOR selects the major (default 23, the major CI tracks; 22 is still in the CI
# matrix and is valid here too). Installs micromamba + an 'm<MAJOR>' env (mlir + llvmdev +
# gxx_linux-64 + ninja + cmake). Idempotent. After this, run tools/local/check_rail.sh to
# build bcir-opt against that major and run the full rail. CI still uses apt.llvm.org (the
# authoritative gate); this is a local convenience only.
set -euo pipefail
MLIR_MAJOR="${MLIR_MAJOR:-23}"
if ! [[ "${MLIR_MAJOR}" =~ ^[0-9]{2}$ ]]; then
  echo "[setup_mlir] MLIR_MAJOR must be a two-digit LLVM major (e.g. 23)" >&2
  exit 2
fi
ENV_NAME="m${MLIR_MAJOR}"

# Arch-select the micromamba asset + the conda compiler package (so this builds natively on a
# Raspberry Pi 5 / aarch64, not only x86_64). conda-forge publishes mlir for linux-aarch64.
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
    echo "[setup_mlir] refusing non-private directory: ${directory}" >&2
    exit 1
  fi
done

if [ -e "${MM}" ]; then
  if [ -L "${MM}" ] || [ ! -f "${MM}" ] || [ "$(stat -c '%u' -- "${MM}")" != "${EUID}" ]; then
    echo "[setup_mlir] refusing unowned, non-regular, or symlinked bootstrap: ${MM}" >&2
    exit 1
  fi
  actual="$(sha256sum -- "${MM}" | awk '{print $1}')"
  if [ "${actual}" != "${MM_SHA256}" ]; then
    echo "[setup_mlir] refusing micromamba with unexpected SHA-256: ${MM}" >&2
    exit 1
  fi
else
  echo "[setup_mlir] fetching pinned micromamba ${MM_VERSION} (${MM_ASSET})..."
  part="$(mktemp "${MM}.part.XXXXXXXX")"
  trap 'rm -f -- "${part:-}"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    "https://github.com/mamba-org/micromamba-releases/releases/download/${MM_VERSION}/${MM_ASSET}" \
    -o "${part}"
  actual="$(sha256sum -- "${part}" | awk '{print $1}')"
  if [ "${actual}" != "${MM_SHA256}" ]; then
    echo "[setup_mlir] downloaded micromamba failed SHA-256 verification" >&2
    exit 1
  fi
  chmod 0700 "${part}"
  mv -- "${part}" "${MM}"
  part=""
fi
chmod 0700 "${MM}"

# BCIR_LOCAL_FULL=1 adds the whole lowering toolset of the same major -- clang, clang++, lld (wasm-ld),
# the llvm tools (lli / llc / opt / llvm-as) and compiler-rt (ASan / UBSan / TSan / libFuzzer) -- so the
# thorough Python tier, the C rails and the security campaigns run on that major too, as the
# `oracle-llvm-latest` and `c-rails-llvm-latest` CI jobs do (TC23). The env is then ~4.6 GB. The `nocfg`
# clang variant behaves like a distribution clang (the host's headers and libc), which is what every gate
# assumes; `LLVM_BIN="${PREFIX}/envs/${ENV_NAME}/bin"` and that directory first on PATH select it.
FULL_PKGS=()
if [ "${BCIR_LOCAL_FULL:-0}" = "1" ]; then
  FULL_PKGS=("clang=${MLIR_MAJOR}=default_nocfg*" "clangxx=${MLIR_MAJOR}=default_nocfg*" "clangdev=${MLIR_MAJOR}"
             "llvm-tools=${MLIR_MAJOR}" "lld=${MLIR_MAJOR}" "compiler-rt=${MLIR_MAJOR}")
fi
if [ ! -x "${PREFIX}/envs/${ENV_NAME}/bin/mlir-opt" ]; then
  echo "[setup_mlir] creating the MLIR ${MLIR_MAJOR} env from conda-forge (~250 MB; ~4.6 GB with BCIR_LOCAL_FULL=1)..."
  "${MM}" create -y -n "${ENV_NAME}" -c conda-forge "mlir=${MLIR_MAJOR}" "llvmdev=${MLIR_MAJOR}" "${GXX_PKG}" ninja cmake "${FULL_PKGS[@]}"
elif [ "${#FULL_PKGS[@]}" -gt 0 ] && [ ! -x "${PREFIX}/envs/${ENV_NAME}/bin/clang" ]; then
  echo "[setup_mlir] adding the lowering toolset of major ${MLIR_MAJOR} to the existing env..."
  "${MM}" install -y -n "${ENV_NAME}" -c conda-forge "${FULL_PKGS[@]}"
fi

echo "[setup_mlir] toolchain ready at ${PREFIX}/envs/${ENV_NAME}"
LD_LIBRARY_PATH="${PREFIX}/envs/${ENV_NAME}/lib" "${PREFIX}/envs/${ENV_NAME}/bin/mlir-opt" --version | grep -i "LLVM version"
echo "[setup_mlir] now run: MLIR_MAJOR=${MLIR_MAJOR} bash tools/local/check_rail.sh"
