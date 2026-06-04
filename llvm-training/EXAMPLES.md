# LLVM Training Example and Exercise Conventions

This document defines the naming, validity, and verification conventions for
LLVM IR examples and exercises in `llvm-training/`. The goal is to make it clear
which files are runnable, which files intentionally demonstrate failures, and
which commands maintainers and agents should run before shipping changes.

## Standalone `.ll` examples

Standalone LLVM IR examples are files that are meant to assemble on their own,
without extracting surrounding prose or adding missing declarations.

- Standalone examples should live in a chapter-local `examples/` directory.
- Standalone examples use the `.ll` extension.
- Every standalone `.ll` example must assemble with LLVM >= 15, where opaque
  pointers are the default.
- Use `ptr` for pointer-typed values instead of typed pointers such as `i32*`.
- Prefer examples that are small enough to diagnose quickly with `llvm-as` and
  `opt -passes=verify`.

## Intentionally invalid examples

Examples that intentionally demonstrate parser, verifier, or migration failures
must not be confused with known-good standalone examples.

Use one of these conventions for intentionally invalid examples:

- Store the example as `.ll.txt`, for example `typed-pointer-before.ll.txt`; or
- Include `invalid` in the filename, for example `phi-invalid-predecessor.ll`.

Invalid examples should also describe the expected failure in nearby prose or a
local `examples/README.md`, including whether the failure is expected from the
parser, assembler, verifier, or a specific pass.

## Pass-output example names

When a chapter checks in before/after IR for an LLVM pass or optimization level,
name files so the pass relationship is obvious from the filename.

Recommended pattern:

- `foo-before.ll`
- `foo-after-mem2reg.ll`
- `foo-after-o2.ll`

Use lowercase pass names and optimization levels matching the command when
possible. For example, if the command is `opt -passes=mem2reg`, prefer
`*-after-mem2reg.ll`; if the command is `opt -passes='default<O2>'`, prefer
`*-after-o2.ll`.

## Chapter-local command documentation

Each chapter that contains runnable examples should include one of the following
near the examples:

- A local `examples/README.md`; or
- A short chapter section that lists the commands needed to assemble, verify,
  optimize, or otherwise inspect the examples.

The command list should distinguish known-good standalone examples from invalid
or illustrative examples. If a chapter includes pass-output examples, document
how to regenerate or compare them.

## Exercise conventions

Each exercise should include the following pieces of information:

- **Prompt**: the task the learner should complete.
- **Expected command**: the exact command to assemble, verify, run, optimize, or
  otherwise check the learner's answer.
- **Expected observation**: the result the learner should observe, such as a
  successful assembler exit, a specific optimized instruction pattern, or a
  known verifier diagnostic.
- **Optional solution file**: a checked-in solution such as
  `001-add.solution.ll` when the exercise benefits from a reference answer.

Solutions that are checked in as `.ll` files are known-good standalone examples
and must follow the LLVM >= 15 opaque-pointer convention.

## Top-level known-good verification

From the repository root, validate known-good standalone `.ll` examples with:

```bash
find llvm-training -path '*/examples/*.ll' ! -iname '*invalid*.ll' -print0 | sort -z | while IFS= read -r -d '' f; do
  llvm-as "$f" -o /dev/null || exit 1
done
```

This command intentionally skips `.ll.txt` files and any `.ll` file with
`invalid` in its name. Use targeted commands from the relevant chapter when you
want to demonstrate or test an expected failure.
