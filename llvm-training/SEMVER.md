# LLVM Version Compatibility Policy

The training corpus targets **LLVM 15 or newer**. LLVM 15 is the baseline because
opaque pointers are the default model and the examples intentionally use `ptr`
instead of typed pointers such as `i32*`.

## Policy

- Standalone `.ll` examples should assemble and verify with LLVM 15+ unless a
  chapter explicitly documents a newer requirement.
- Prefer IR syntax that remains stable across supported LLVM versions.
- Use opaque pointers (`ptr`) in examples, exercises, and solutions.
- Keep intentionally invalid examples as `.ll.txt` or with `invalid` in the file
  name so version-specific parser failures do not enter the known-good manifest.
- If an example needs a newer intrinsic, pass spelling, or verifier behavior,
  state that requirement near the example and in the prompt or README that tells
  users how to run it.

## Local tool names

Many systems install versioned binaries such as `llvm-as-18`, `opt-18`, or
`llc-18`. The commands in this corpus use unversioned names for readability. If
your environment only has versioned names, substitute the matching binary.

## Optional advanced tools

Core standalone `.ll` verification requires only `llvm-as` and `opt`. Advanced
walkthroughs for PGO, LTO, or BOLT may additionally reference tools such as
`clang`, `llvm-profdata`, `llc`, `llvm-objdump`, `llvm-bolt`, or `perf2bolt`.

When an advanced-tool command is not part of the CI guarantee, the chapter must
mark it as schematic or optional so readers can distinguish required
verification steps from environment-dependent demonstrations. CI jobs and local
scripts should also skip cleanly when these optional tools are unavailable
instead of failing the core standalone `.ll` verification path.

## CI expectation

CI runs `llvm-training/tools/verify-examples.sh` when `llvm-as` and `opt` are on
`PATH`. That script verifies the known-good standalone examples and also checks
the broken `.ll.txt` sentinel used to keep invalid examples out of the manifest.
