# LLVM Training Example and Exercise Conventions

This document defines the naming, validity, and verification conventions for
LLVM IR examples and exercises in `llvm-training/`. The goal is to make it clear
which files are runnable, which files intentionally demonstrate failures, and
which commands maintainers and agents should run before shipping changes. See
[`SEMVER.md`](SEMVER.md) for the LLVM-version compatibility policy.

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

## Data artifacts and schemas

Some chapter `examples/` directories include data artifacts that document input
or output schemas rather than standalone LLVM IR. For example, the following
files in `llvm-training/15-binary-analysis/examples/` are schema examples, not
LLVM IR examples:

- `dynamic-trace-sample.csv`
- `perf-counter-sample.csv`
- `bcsa-feature-sample.csv`

These CSV files should not be included in `llvm-as` verification loops. Keep
assembler verification scoped to known-good standalone `.ll` files, and use
chapter-specific checks or prose review for data artifacts.

Nearby chapter prose must describe each data artifact's fields, intended
interpretation, and limitations, especially for hardware-specific or
profile-specific data such as performance counters, dynamic traces, or features
derived from a particular binary-analysis workflow.

## Before/after examples

Before/after examples are teaching pairs, not a promise that every LLVM version
will produce byte-identical IR. Use them to show structural intent.

- Name inputs `topic-before.ll` or `topic.input.ll` when the learner should run
  a command.
- Name expected snapshots `topic-after-<pass>.ll`, `topic.after-<pass>.ll`, or
  `topic-after-<level>.ll`, using the exact pass or optimization-level spelling
  where practical.
- Document the regenerating command in chapter prose, an `examples/README.md`,
  or the exercise prompt.
- State which observations are stable: added `phi` nodes, removed blocks, vector
  types, runtime wrapper calls, metadata preservation, or diagnostic shape.
- Do not make fragile value names or pass-manager cleanup details part of the
  exercise unless the chapter explicitly pins an LLVM version.

## `.invalid.ll.txt` fixtures

Use `.invalid.ll.txt` for broken IR that must remain outside the known-good
manifest while still being available for repair drills or diagnostic examples.

- Pair repair prompts with `NNN-topic.invalid.ll.txt` and, when useful, a fixed
  `NNN-topic.solution.ll`.
- The prompt or adjacent prose must name the expected parser, assembler,
  verifier, or pass-level failure.
- Validate intentional failures with
  `./llvm-training/tools/verify-invalid-fixtures.sh` instead of broad `llvm-as`
  loops.
- Never rename an invalid fixture to plain `.ll` just to simplify linking; that
  would enroll it in known-good verification.

## MLIR examples

MLIR examples are not LLVM IR examples even when they use the LLVM dialect.

- Store MLIR examples as `.mlir` under chapter-local `examples/` directories.
- Document whether an example is a custom dialect sketch, an LLVM-dialect shape,
  or an illustrative lowering boundary.
- Use `./llvm-training/tools/verify-mlir-examples.sh` only when the local
  environment has the required MLIR tools; otherwise treat the file as a review
  artifact.
- Do not include `.mlir` artifacts in `llvm-as` or `opt -passes=verify` loops.

## CSV and data artifacts

CSV and similar data artifacts document evidence schemas, not runnable IR.

- Keep `.csv`, `.json`, trace, counter, and feature samples under the relevant
  chapter's `examples/` directory with prose describing every column or field.
- State whether values are synthetic, normalized, hardware-specific,
  profile-specific, or copied from a real run.
- Prefer tiny samples that clarify schema shape over large benchmark dumps.
- Exclude data artifacts from LLVM assembler, optimizer, and exercise-solution
  verification.

## Generated BCIR mapping outputs

BCIR mapping outputs are examples of generated LLVM IR and must be easy to
compare against the source-domain intent.

- Keep generated or expected IR under `bcir-mapping/examples/` unless a chapter
  has a more specific examples directory.
- Use names that encode the lowering pattern, such as
  `claim-resource-lookup.ll`, `bcir-op-runtime-wrapper.ll`,
  `graph-fragment-struct-gep.ll`, or `diagnostic-metadata-preservation.ll`.
- Include nearby prose linking the output to the BCIR mapping chapter that owns
  the rule.
- Generated `.ll` outputs are known-good standalone examples unless explicitly
  marked invalid, so they must assemble with opaque pointers and verify with the
  top-level example verifier.
- Diagnostic metadata in generated examples may explain provenance, but required
  execution semantics must remain in instructions, operands, calls, or ABI data.

## Intentionally invalid examples

Examples that intentionally demonstrate parser, verifier, or migration failures
must not be confused with known-good standalone examples. The repository keeps
`llvm-training/examples/broken-example.ll.txt` as a deliberately malformed
trip-wire fixture; `llvm-training/tools/verify-examples.sh` checks that LLVM
rejects it while still excluding it from the known-good manifest.

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
and must follow the LLVM >= 15 opaque-pointer convention. Intentionally broken
exercise inputs must use `.ll.txt` or include `invalid` in the filename, even if
the prompt asks the learner to run LLVM and observe the diagnostic.

## Exercise families

Exercises are broader than standalone IR-writing drills. Use the filename and
verification conventions below so learners and verification scripts know what is
expected.

- **Standalone IR writing** exercises use `NNN-topic.prompt.md` and, when a
  reference answer is useful, `NNN-topic.solution.ll`. Checked-in `.solution.ll`
  files should assemble as complete modules.
- **Repair** exercises pair a prompt with an intentionally broken input named
  `NNN-topic.invalid.ll.txt` or another filename containing `invalid`. The
  broken input should be rejected by LLVM, while any checked-in fixed solution
  should use `.solution.ll` and assemble normally.
- **Optimization pass reasoning** exercises may include `NNN-topic.input.ll` and
  optional `NNN-topic.after-<pass>.ll` snapshots. Inputs should assemble before
  the pass is run. After-pass files are teaching snapshots for structural
  comparison; exact value names, attributes, and cleanup can differ by LLVM
  version.
- **Language-agnostic review** exercises use prompts without requiring a checked
  in `.ll` solution when the answer is a review checklist or written diagnosis.
  Add these before asking learners to implement real passes.
- **Pass implementation skeletons**, if added later, should live in a clearly
  named non-verified exercise family with local build instructions. Do not mix
  C++ skeletons into known-good LLVM IR verification loops.

## Top-level known-good verification

From the repository root, validate known-good standalone `.ll` examples with
the canonical example verification script:

```bash
./llvm-training/tools/verify-examples.sh
```

Maintainers should prefer this script over ad-hoc `find ... llvm-as` loops. It
assembles every known-good standalone example with `llvm-as` and then runs
`opt -passes=verify` on the same file, so it catches both parser/assembler
errors and verifier failures. It also validates
`llvm-training/examples/broken-example.ll.txt` as the invalid-example tripwire:
the fixture must remain outside the known-good manifest and must continue to be
rejected by LLVM.

Do not copy ad-hoc `find ... llvm-as` loops into CI or docs as a substitute;
they miss verifier-only failures and the invalid-example tripwire.
A manual loop can still be useful as an illustrative fallback when inspecting a
minimal environment, but it is weaker than `verify-examples.sh` because it does
not run `opt -passes=verify` and does not check the invalid-example tripwire:

```bash
find llvm-training -path '*/examples/*.ll' -type f \
  ! -iname '*.ll.txt' ! -iname '*invalid*' -print0 |
  sort -z |
  xargs -0 -n1 llvm-as -o /dev/null
```

Validate checked-in exercise solutions separately with their maintained script:

```bash
./llvm-training/tools/verify-exercises.sh
```

After configuring the repository, the same batch checks are available through
CMake targets that skip cleanly when the relevant optional LLVM tools are not on
`PATH`:

```bash
cmake --build build --target llvm-training-verify-examples
cmake --build build --target llvm-training-verify-exercises
cmake --build build --target llvm-training-smoke-llc
cmake --build build --target llvm-training-smoke-lli
```

`verify-examples.sh` intentionally skips `.ll.txt` files and any `.ll` file with
`invalid` in its name, while `verify-exercises.sh` checks every
`llvm-training/exercises/*.solution.ll` reference answer. Use targeted commands
from the relevant chapter when you want to demonstrate or test an expected
failure.
