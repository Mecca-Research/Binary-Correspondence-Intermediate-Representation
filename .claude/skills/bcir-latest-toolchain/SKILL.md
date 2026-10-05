---
name: bcir-latest-toolchain
description: >-
  Judge every BCIR change on the newest LLVM/Clang/MLIR 23 and Node 24, not only on the CI-default
  Clang 18 / GCC 13. Load this skill before committing, pushing or updating a PR that touches any
  rail (`bcir/`, `mlir/`, `runtime/c`, `runtime/cpp`, `tools/`, the CMake build, workflows), and
  whenever work involves a compiler or runtime: clang/llc/lli/opt/wasm-ld, MLIR, sanitizers,
  CMake presets, generated C, the WASM tests or Node -- even when the request only says "test it",
  "validate", "is it ready" or "open the PR". Also load it when the request mentions LLVM 23,
  Clang 23, Node 24, "the latest toolchain" or a compiler upgrade. Pairs with bcir-cicd (the
  pipeline) and bcir-systems-engineer (the method).
---

# BCIR on the newest toolchain

## The rule

Every change is validated on two toolchains before it is published:

1. **the CI default** -- Ubuntu's Clang 18 and GCC 13: the quick tier, `tools/c/check_runtime.sh`,
   the `gcc`/`clang` CMake trees (AGENTS.md, CONTRIBUTING.md); and
2. **the newest release of the majors the law rail and the WASM tests track** -- LLVM/Clang/MLIR
   **23** (the highest `llvmorg-23.x.y` tag) and Node **24** (the highest v24.x of nodejs.org's
   index): `bash tools/local/check_latest.sh`.

A pass on Clang 18 alone is not evidence about 23. BUILD-2c is the worked example: the generated
Q8 table header used C23 `#embed` whenever the toolchain had it; Clang 18 and GCC 13 do not, so
every Clang-18 gate was green, while Clang 23.1.2 offers `#embed` to C11 as an extension and the
`-Werror` C11 build failed. CI's CMake job had no Clang 23 cell, so it would have gone out green.
Newer compilers find new things: extensions offered to older standards, new default warnings,
removed or renamed APIs (`applyPatternsAndFoldGreedily`, `builder.create<Op>` -- PR #752),
stricter sanitizers.

## 1. Is the newest toolchain installed?

```bash
python3 tools/local/latest_toolchain.py status    # exit 0 CURRENT, 1 BEHIND/MISSING, 2 UNKNOWN
```

It reads the installed `clang`, `llvm-config`, `mlir-opt` (from `LLVM_BIN`, the
`tools/local/setup_mlir.sh` env, or apt.llvm.org's `/usr/lib/llvm-23`) and the highest Node 24
it can find, and compares them with the newest releases read from the sources themselves.

- **LLVM 23 missing or behind:** `BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh` (conda-forge,
  which tracks the upstream point releases; about 4.6 GB). To move an existing env to a new point
  release, install the same package set at the new version into `m23` (the list is in
  `setup_mlir.sh`'s `FULL_PKGS`). Where the network policy allows apt.llvm.org, its
  `clang-23 lld-23 llvm-23 libclang-rt-23-dev libmlir-23-dev mlir-23-tools` work too.
- **Node 24 missing or behind:** `python3 tools/local/latest_toolchain.py install-node` -- the
  tarball is checked against nodejs.org's checksum list, and the list's signature against the Node
  release keys, before anything is extracted. `nvm install 24` is fine where nvm is set up.
- **UNKNOWN** (an upstream source unreachable, e.g. a network policy): say so in the PR body and
  run the check on what is installed with `--allow-outdated`. Never report UNKNOWN as current.

## 2. Run the check

```bash
bash tools/local/check_latest.sh                     # every leg, about 35 minutes
bash tools/local/check_latest.sh --legs cmake,gate   # while iterating
```

| Leg | What it runs on LLVM 23 + Node 24 | CI owner |
|---|---|---|
| `status` | the installed toolchain is the newest release | -- |
| `confirm` | the oracle's resolver finds one coherent LLVM 23; node is 24; clang 23's fuzzer, ASan/UBSan and TSan runtimes link and run | every `*-llvm-latest` job's confirmation step |
| `cmake` | a fresh `clang23` preset tree with `BCIR_REQUIRE_TSAN=ON`: build (`-Werror`), `ctest -L section`, `ctest -L build` (section and build parity) | `cmake-build` (clang23 cell) |
| `fuzz` | the `fuzzer` preset on clang 23, `ctest -L fuzz` | `c-rails-llvm-latest` (analysis) |
| `gate` | `tools/c/check_runtime.sh` on clang 23 with `BCIR_REQUIRE_TSAN=1` (`--cfront-sanitize` adds the cfront ASan/UBSan sweep) | `c-rails-llvm-latest` (runtime) |
| `thorough` | the thorough tier with LLVM 23 first on PATH and Node 24, `BCIR_REQUIRE_LLVM=1` | `oracle-llvm-latest` |
| `mlir` | the `mlir` preset on MLIR 23: `bcir-opt` under the top-level project, `ctest -L mlir` | `mlir-rail-validate` (LLVM 23) |

The legs run one after another at two workers (AGENTS.md). Run the check after the default-toolchain
gates, never alongside them; each leg's log stays in `--out` (default: a new temp directory).

## 3. Read the result

- **A leg that fails on 23 and passes on 18 is a finding, not noise.** Root-cause it in the source:
  never pin a target back to 18, never relax `-Werror` for a whole target, never add a
  compiler-version special case where one portable form exists. Then prove it both ways (the old
  code fails on 23, the new code passes on 23, 18 and GCC) and record it in the PR's "Found while
  building".
- **A warning only 23 prints** in code the change touches: fix it. In code it does not touch:
  record it in the PR and open a follow-up -- do not widen the PR.
- **Record the evidence:** the PR body's Verification section names the versions
  (`latest_toolchain.py status` prints them) and every leg's verdict line from the summary.

## 4. What CI covers, and what nobody covers

CI judges 23 remotely: `oracle-llvm-latest` (the thorough shards, apt.llvm.org's clang/lld/llvm 23,
Node 24), `c-rails-llvm-latest` (runtime / analysis / ubsan-1 / ubsan-2 on clang 23),
`mlir-rail-validate` (LLVM 22 and 23), `llvm-training` (23), and `cmake-build`'s `clang23` cell. The
oracle jobs run Node 24 through `actions/setup-node` (`node-version: 24`, the newest 24.x), and
apt.llvm.org's `clang-23` is the newest 23.x build. The local check exists so that what CI would
find on 23 is found before the push -- CI stays the authority (`bcir-cicd`).

Not covered by either: LLVM 23 on native aarch64 (CI's ARM jobs run Ubuntu's Clang 18). Say so in
the PR when a change is architecture-sensitive.

## 5. When the tracked majors move

When the law rail moves to LLVM 24, or Node's next LTS line replaces 24, change in one PR
(laws.md L14): `LLVM_MAJOR`/`NODE_MAJOR` in `tools/local/latest_toolchain.py`, the `clang23` preset
(and its CI cell), the `*-llvm-latest` jobs' packages and `node-version`, `setup_mlir.sh`'s default
major, `check_latest.sh`'s legs, `bcir/tests/test_latest_toolchain.py`, and this skill.
