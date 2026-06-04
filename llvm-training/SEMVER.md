# LLVM Version Compatibility Policy

The training corpus targets **LLVM 15 or newer**. LLVM 15 is the baseline
because opaque pointers are the default model and the examples intentionally use
`ptr` instead of typed pointers such as `i32*`.

## Policy

- Standalone `.ll` examples should assemble and verify with LLVM 15+ unless a
  chapter explicitly documents a newer requirement.
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
verification is available, use `./llvm-training/tools/verify-mlir-examples.sh`
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

CI runs `llvm-training/tools/verify-examples.sh` when `llvm-as` and `opt` are on
`PATH`. That script verifies known-good standalone examples and checks the
broken `.ll.txt` sentinel used to keep invalid examples out of the manifest.
Additional scripts such as `verify-exercises.sh`, `verify-invalid-fixtures.sh`,
`verify-mlir-examples.sh`, `smoke-llc.sh`, `smoke-lli.sh`, and `smoke-bolt.sh`
cover optional families and must skip cleanly when their tools are unavailable.
