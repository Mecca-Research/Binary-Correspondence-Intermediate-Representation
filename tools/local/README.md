# Local true-MLIR validation (conda-forge)

CI validates the MLIR rail against **LLVM/MLIR 23** (with 22 kept in the matrix for one
release cycle) installed from `apt.llvm.org`. In
sandboxes whose network policy **blocks `apt.llvm.org`** (e.g. Claude Code on the web with
a restrictive egress allowlist), that source is unreachable — and the stock Ubuntu archive
only ships MLIR up to **18**. An 18 build (`tools/wsl/build_mlir.sh`) validates the
C++/pass logic, but not the newer-major rules: the IRDL named-operand syntax, the `Symbol`
verifier tightenings, and the current API deprecations.

**conda-forge is usually still reachable** and ships real `mlir=23.1.x` (and `22.1.x`) dev
libs plus an ABI-matched compiler, which closes the gap entirely — a true MLIR 23 `bcir-opt`
built and run locally.

## Usage

```bash
bash tools/local/setup_mlir.sh                 # micromamba + conda-forge mlir=23 env (~250 MB, idempotent)
bash tools/local/check_rail.sh                 # build bcir-opt vs 23 + run the WHOLE rail on 23
MLIR_MAJOR=22 bash tools/local/setup_mlir.sh   # the other major still in the CI matrix
MLIR_MAJOR=22 bash tools/local/check_rail.sh
```

`check_rail.sh` runs tblgen, the R1–R25 / GEM / optimizer pass suite, the ODS examples,
the bytecode round-trip, and the **IRDL named-syntax corpus** (the check an 18 build cannot
do) — all against the true selected major.

## Notes

- **ABI:** conda's MLIR is built with conda's GCC. Building `bcir-opt` with the *system*
  compiler links incompatible MLIR `TypeID` statics and segfaults at startup; the scripts
  build with conda's `gxx_linux-64` (`x86_64-conda-linux-gnu-g++`) to match.
- **Private cache:** the toolchain defaults to
  `${XDG_CACHE_HOME:-$HOME/.cache}/bcir/mamba`; the pinned micromamba bootstrap is kept
  in the adjacent private `tools/` directory. The setup script rejects symlinked,
  unowned, or group/world-writable bootstrap locations. `MAMBA_ROOT_PREFIX` and
  `MICROMAMBA` may override these paths only when the same ownership rules hold.
- **Authority:** CI (`apt.llvm.org`) remains the gating check. This is a fast local mirror
  so major-specific failures are caught before pushing, not just in CI.
- **The clean alternative** is to allow `apt.llvm.org` in the environment's network policy
  (see https://code.claude.com/docs/en/claude-code-on-the-web); then the CI install path
  works locally verbatim and conda is unnecessary.
