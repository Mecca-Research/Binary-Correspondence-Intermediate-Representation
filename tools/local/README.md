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
BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh   # + clang/lld/llvm-tools/compiler-rt of the same major (~4.6 GB)
```

With `BCIR_LOCAL_FULL=1` the env also carries the lowering toolset of that major, so the gates the
`oracle-llvm-latest` and `c-rails-llvm-latest` CI jobs run on apt.llvm.org's LLVM 23 run here too
(TC23): put `${XDG_CACHE_HOME:-$HOME/.cache}/bcir/mamba/envs/m23/bin` first on PATH, export
`LLVM_BIN` to it, and run `BCIR_THOROUGH=1 BCIR_REQUIRE_LLVM=1 python -m bcir.tests.run_all -j 2`,
`bash tools/c/check_runtime.sh`, `CLANG=clang bash tools/c/sanitize_cfront.sh`, and the rest of the
C and security rails as CI spells them. The conda binaries need no `LD_LIBRARY_PATH`.

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

## The newest toolchain, for every change

Every BCIR change is judged on the newest LLVM/Clang/MLIR 23, Node 24 and Python 3.15 as well
as on the CI-default Clang 18 / GCC 13 / Python 3.12 (`.claude/skills/bcir-latest-toolchain/SKILL.md`):

```bash
python3 tools/local/latest_toolchain.py status        # installed vs the newest llvmorg-23 tag / Node 24 release / v3.15 tag
python3 tools/local/latest_toolchain.py install-node  # the newest Node 24, signature-verified, into the BCIR cache
uv python install 3.15                                # or: micromamba create -n py315 -c conda-forge python=3.15
eval "$(python3 tools/local/latest_toolchain.py env)" # LLVM 23 and Node 24 first on PATH; BCIR_PYTHON_LATEST the 3.15
bash tools/local/check_latest.sh                      # status, confirm, python (quick tier + security gates on 3.15),
                                                      # cmake (clang23 preset), fuzz, analyze, gate, thorough (on 3.15,
                                                      # BCIR_REQUIRE_LLVM=1) and mlir legs on them
```

`status` exits 0 only when every tool is the newest release; a source it cannot reach is
UNKNOWN (exit 2), never current. `check_latest.sh` runs its legs one after another at two workers
and prints one verdict line per leg; `--legs` picks a subset and `--allow-outdated` records a
toolchain behind upstream instead of failing on it.
