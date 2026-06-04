# LLVM Training Tools

This directory contains repository-maintenance scripts for the `llvm-training/`
corpus. They are intentionally small shell scripts so CI and local agents can run
the same checks without a build-system dependency.

## Scripts

| Script | Purpose | Required tools |
| --- | --- | --- |
| `verify-examples.sh` | Builds the known-good standalone `.ll` manifest from chapter-local `examples/` directories, assembles each file with `llvm-as`, and runs `opt -passes=verify`. It also checks the broken `.ll.txt` sentinel so intentionally invalid examples do not drift into the manifest. | `llvm-as`, `opt` |
| `smoke-lli.sh` | Runs only curated examples that have a safe no-argument entry point under `lli`. Most training snippets are library-style IR and should stay out of this list. | `lli` |
| `smoke-llc.sh` | Lowers curated examples with `llc` to catch target-codegen regressions without treating every IR snippet as a runnable program. | `llc` |
| `smoke-bolt.sh` | Builds the BOLT layout demo fixture, records baseline symbol/disassembly text, and exits with a clean skip when `llvm-bolt` is not installed. A full profile-driven rewrite still requires host support for `perf2bolt`/`perf`; see the walkthrough. | Optional `llvm-bolt`; `clang` and `llvm-objdump` when BOLT is present |
| `demo-mem2reg.sh` | Demonstrates `mem2reg` on the checked-in diamond example, first verifying the fixture and then printing the promoted SSA form to stdout. | `opt` |
| `demo-o2.sh` | Runs `default<O2>` on the O2 pipeline inspection fixture, writes the optimized IR to a temporary file, prints it, and optionally smoke-checks the result with `llc` when available. | `opt`; optional `llc` |
| `demo-vectorize.sh` | Shows loop-vectorization remarks from `clang` on the C fixture, then forces a visible loop-vectorizer experiment over the checked-in IR and prints the transformed IR. | `opt`, `clang` |
| `demo-debug-pipeline.sh` | Captures `-debug-pass-manager` output for `default<O2>` into a temporary log, then prints the pass schedule for inspection. | `opt` |

## Skip-list rationale

`verify-examples.sh` is a positive manifest rather than a global `*.ll*` sweep:

- files ending in `.ll.txt` are reserved for intentionally invalid parser,
  verifier, or migration examples;
- files with `invalid` in the filename are also reserved for expected-failure
  examples;
- only files below an `examples/` directory are considered standalone training
  modules.

`smoke-bolt.sh` is also a positive, guarded check: it validates the documented
BOLT fixture only when the host has `llvm-bolt`, and otherwise reports an
intentional skip so minimal CI images are not forced to install BOLT packages.

The sentinel `../examples/broken-example.ll.txt` is deliberately malformed. The
verifier script asserts that LLVM rejects it while the script as a whole still
succeeds, which catches future changes that accidentally include invalid fixtures
in the known-good example manifest.

## Adding a script

When adding a new script:

1. keep it executable and runnable from the repository root;
2. document required external tools and any intentional skips in this README;
3. print the command being demonstrated before executing it;
4. write demo output either to stdout or to a clearly named file under `${TMPDIR:-/tmp}`;
5. make it fail closed when a required fixture disappears;
6. wire it into `.github/workflows/ci.yml` when it guards repository health.
