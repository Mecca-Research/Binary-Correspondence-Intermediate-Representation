# Source this to point the rail scripts at the local conda-forge MLIR toolchain that
# tools/local/setup_mlir.sh installed:  MLIR_MAJOR=23 source tools/local/env_mlir.sh
# (MLIR_MAJOR defaults to 23, the major CI tracks; 22 is still in the CI matrix.)
# Sets the tool/include overrides the tools/wsl/* and tools/irdl/* scripts already honor
# (BCIR_OPT / MLIR_OPT / MLIR_TBLGEN / MLIR_INCLUDE), plus LD_LIBRARY_PATH for the conda
# shared libs and the ABI-matched compiler the build must use (system gcc vs conda's MLIR
# mismatches the MLIR TypeID statics and segfaults at startup; conda's g++ matches).  LLVM_BIN
# also pins every backend utility to this distribution: a system llc of the same major is not
# necessarily ABI-compatible with the conda libraries on LD_LIBRARY_PATH.
MLIR_MAJOR="${MLIR_MAJOR:-23}"
_BCIR_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/bcir"
_ENV="${MAMBA_ROOT_PREFIX:-${_BCIR_CACHE_ROOT}/mamba}/envs/m${MLIR_MAJOR}"
export MLIR_MAJOR
export MLIR_LOCAL_ENV="${_ENV}"
export LD_LIBRARY_PATH="${_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLIR_TBLGEN="${_ENV}/bin/mlir-tblgen"
export MLIR_INCLUDE="${_ENV}/include"
export MLIR_OPT="${_ENV}/bin/mlir-opt"
export LLVM_BIN="${_ENV}/bin"
# The conda compiler triple is arch-specific (so this resolves on aarch64 / Pi 5 too).
case "$(uname -m)" in
  aarch64|arm64) _CONDA_TRIPLE="aarch64-conda-linux-gnu" ;;
  *)             _CONDA_TRIPLE="x86_64-conda-linux-gnu" ;;
esac
export MLIR_LOCAL_CC="${_ENV}/bin/${_CONDA_TRIPLE}-gcc"
export MLIR_LOCAL_CXX="${_ENV}/bin/${_CONDA_TRIPLE}-g++"
