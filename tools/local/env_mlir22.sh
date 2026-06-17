# Source this to point the rail scripts at the local conda-forge MLIR 22 toolchain that
# tools/local/setup_mlir22.sh installed:  source tools/local/env_mlir22.sh
# Sets the tool/include overrides the tools/wsl/* and tools/irdl/* scripts already honor
# (BCIR_OPT / MLIR_OPT / MLIR_TBLGEN / MLIR_INCLUDE), plus LD_LIBRARY_PATH for the conda
# shared libs and the ABI-matched compiler the build must use (system gcc vs conda's MLIR
# mismatches the MLIR TypeID statics and segfaults at startup; conda's g++ matches).
_ENV="${MAMBA_ROOT_PREFIX:-/tmp/mamba}/envs/m22"
export MLIR22_ENV="${_ENV}"
export LD_LIBRARY_PATH="${_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLIR_TBLGEN="${_ENV}/bin/mlir-tblgen"
export MLIR_INCLUDE="${_ENV}/include"
export MLIR_OPT="${_ENV}/bin/mlir-opt"
export MLIR22_CC="${_ENV}/bin/x86_64-conda-linux-gnu-gcc"
export MLIR22_CXX="${_ENV}/bin/x86_64-conda-linux-gnu-g++"
