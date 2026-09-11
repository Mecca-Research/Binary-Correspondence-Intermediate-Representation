# LLVM Version Compatibility Policy

The training corpus targets **LLVM 18 or newer**, and tracks **LLVM 23** as the
release its material is re-derived against.

## Why 18, and why it moved

The baseline was LLVM 15 until Phase 1.7, chosen because opaque pointers are the
default from 15 onward and the examples use `ptr` rather than `i32*`. That reason
still holds — every release at or above 18 has opaque pointers — but the number
was not true in the sense that matters.

**Nothing assembled at 15.** The floor is enforced by
[`tools/verify-frontend-lowering.py`](tools/verify-frontend-lowering.py), which
assembles each normalized snapshot with the *oldest* `llvm-as` on `PATH` at or
above the declared baseline. CI's training job installs Ubuntu's default LLVM,
which is 18; no CI job had an `llvm-as-15`. So the search resolved to 18 every
time and the corpus's LLVM 15 claim was tested on no machine that gates anything.
A declared floor that nothing assembles is not a floor — it is a number in a
document.

18 is the floor the pipeline can actually hold: it is what CI installs, it is the
current Ubuntu LTS default and therefore what a reader most likely has, and it
keeps the property the floor existed for.

**The claim now has an owner.** The CI job that installs the baseline toolchain
runs the gate with `--require-baseline`, which fails when no `llvm-as` at the
declared major is present, rather than silently checking against whatever it
found. On a developer host without that assembler the check still says nothing
and moves on — a local skip is honest; a silent skip in the job that is supposed
to provide the tool is not.

## Policy

- Standalone `.ll` examples should assemble and verify with LLVM 18+ unless the
  example explicitly declares a newer requirement.
- An example needing a construct newer than the baseline carries
  `; REQUIRES: llvm >= N` on its first lines.
  [`tools/verify-examples.sh`](tools/verify-examples.sh) skips it, with the
  reason printed, on an older assembler and verifies it normally once the
  assembler is new enough. The skip has an owner: CI's `LLVM training corpus
  (LLVM 23)` job installs 23 and verifies every version-gated example there.
  `23-version-movement/examples/ptrtoaddr-vs-ptrtoint.ll` is the first user —
  `ptrtoaddr` is a syntax error before LLVM 23.
- Prefer IR syntax that remains stable across supported LLVM versions.
- Use opaque pointers (`ptr`) in examples, exercises, and solutions.
- Keep intentionally invalid examples as `.ll.txt` or with `invalid` in the file
  name so version-specific parser failures do not enter the known-good manifest.
- If an example needs a newer intrinsic, pass spelling, verifier behavior,
  backend feature, or MLIR feature, state that requirement near the example and
  in the prompt or README that tells users how to run it.

## Local tool names

Many systems install versioned binaries such as `llvm-as-18`, `opt-18`,
`llc-18`, `mlir-opt-18`, or `mlir-translate-18`. The commands in this corpus use
unversioned names for readability. If your environment only has versioned names,
substitute the matching binary from the same LLVM/MLIR installation.

## Core tools

Core standalone `.ll` verification requires only `llvm-as` and `opt`. These are
the only tools assumed by the known-good LLVM IR manifest.

## Optional MLIR tools

MLIR examples under `14-mlir-bridge/examples/*.mlir` are optional review
artifacts unless the local environment provides matching MLIR tools. When MLIR
verification is available, use `./training/llvm/tools/verify-mlir-examples.sh`
or chapter-local commands with tools such as `mlir-opt` and `mlir-translate`.

MLIR syntax and dialect availability can shift across LLVM releases more quickly
than core textual LLVM IR. Chapters that depend on a particular MLIR dialect,
conversion pass, or translation flag must document the required LLVM/MLIR
version and should skip cleanly when the tool is absent.

## Optional advanced LLVM tools

Advanced walkthroughs for optimization, backend/JIT, binary analysis, and
post-link review may reference tools such as `clang`, `llvm-profdata`, `llc`,
`lli`, `llvm-objdump`, `llvm-readobj`, `llvm-nm`, `llvm-bolt`, `perf2bolt`,
`llvm-mc`, or target-specific backend tools.

When an advanced-tool command is not part of the CI guarantee, the chapter must
mark it as schematic or optional so readers can distinguish required
verification steps from environment-dependent demonstrations. CI jobs and local
scripts should also skip cleanly when these optional tools are unavailable
instead of failing the core standalone `.ll` verification path.

## CI expectation

CI runs the corpus twice against different toolchains, and the two jobs own
different claims. `LLVM training corpus` installs Ubuntu's default (18) and owns
the **baseline**: it passes `--require-baseline`, and the frontend snapshots are
stamped `Produced by clang 18`, so byte-identity is enforced there. `LLVM training
corpus (LLVM 23)` installs 23 from apt.llvm.org and owns the **ceiling**: it
re-runs every toolchain-consuming gate against the latest release, verifies the
version-gated examples, and reports snapshot deltas rather than enforcing them.

CI runs `training/llvm/tools/verify-examples.sh` when `llvm-as` and `opt` are on
`PATH`. That script verifies known-good standalone examples and checks the
broken `.ll.txt` sentinel used to keep invalid examples out of the manifest.
Additional scripts such as `verify-exercises.sh`, `verify-invalid-fixtures.sh`,
`verify-mlir-examples.sh`, `smoke-llc.sh`, `smoke-lli.sh`, and `smoke-bolt.sh`
cover optional families and must skip cleanly when their tools are unavailable.
