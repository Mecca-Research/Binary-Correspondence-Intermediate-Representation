# LLVM training autograder

This directory defines the deterministic grading contract for **external learner
or agent artifacts**. It does not modify prompts or checked-in reference
solutions, and it does not replace `../tools/verify-exercises.sh`. That existing
script remains the trusted reference-artifact verifier; the autograder handles
untrusted copies in isolated temporary working directories.

The harness evaluates the artifact produced by a learner or model. It does **not**
invoke, benchmark, authenticate, or score the external model-generation call.
Model invocation belongs in an adapter outside the deterministic core. An adapter
may write the attempt layout below and then invoke the grader.

## Attempt format

An attempt root contains one directory per stable exercise ID. The answer
filename is fixed by answer kind:

```text
attempts/
├── 001/answer.ll
├── 025/answer.md
├── 032/answer.md
└── 039/answer.ll
```

The complete filename mapping is:

| Answer kind | Attempt filename | Intended artifact |
| --- | --- | --- |
| `llvm-ir` | `answer.ll` | Standalone LLVM IR module |
| `mlir` | `answer.mlir` | Standalone MLIR artifact |
| `markdown-review` | `answer.md` | Explanatory review |
| `pass-output` | `answer.ll` | Predicted/recorded textual pass result |
| `diagnostic` | `answer.md` | Written diagnosis and repair rationale |

The declarative registry is [`exercises.json`](exercises.json). Every registered
entry supplies a stable ID, prompt and trusted reference paths, answer kind,
required tools, timeout, scoring dimensions, structural assertions, and (where
needed) a comparison strategy or execution vectors. Paths in the registry are
repository-relative. The initial registry covers representative construction,
indexing, review, MLIR-boundary, and metadata exercises; entries can be added
without changing the core grader.

The complete graded-set metadata lives in per-exercise manifests under
[`manifests/`](manifests/) and is governed by
[`schema/exercise.schema.json`](schema/exercise.schema.json). These manifests
cover every numbered exercise and make points, tool requirements, minimum tool
versions, tool-absence policy, determinism, and timeouts reviewable without
replacing the human-readable prompts. Validate them with:

```sh
python3 llvm-training/tools/verify-exercise-manifests.py
```

`exercises.json` remains the executable grader's smaller compatibility registry
until the grader consumes the new check vocabulary directly. A manifest is
required for every numbered graded exercise even when that exercise is not yet
registered with the executable grader.

## Usage

From the repository root:

```sh
python3 llvm-training/tools/grade-exercises.py --attempts /path/to/attempts
python3 llvm-training/tools/grade-exercises.py --exercise 013 --answer answer.ll
python3 llvm-training/tools/grade-exercises.py --attempts /path/to/attempts \
  --format json --output score.json
python3 llvm-training/tools/grade-exercises.py --self-test
```

`grade-exercises.sh` is only a discovery wrapper around the Python program:

```sh
llvm-training/tools/grade-exercises.sh --exercise 001 --answer answer.ll
```

When `--exercise` is omitted with `--attempts`, all registered exercises are
graded. Repeat `--exercise NNN` to select several. A missing registered answer is
reported as a normal failed artifact check rather than crashing the run. The
process exits nonzero when an executed check fails. `--self-test` grades the
checked-in references and requires every executed check to meet the registry's
expected percentage; checks needing unavailable optional tools remain explicit
skips.

## Grading model

### LLVM IR

LLVM IR grading is layered:

1. Require an existing, non-empty answer.
2. Assemble it with `llvm-as`.
3. verify it with `opt -passes=verify`.
4. Normalize accepted textual IR with `opt -S` when available.
5. Check declarative structure: functions, declarations, instructions,
   attributes, metadata, minimum counts, and prohibited legacy/unsafe forms.
6. For registered deterministic vectors, synthesize a small `@main` harness and
   run it with `lli` under the exercise timeout.

The grader never awards correctness by raw diffing against a `*.solution.ll`.
Equivalent SSA names, block labels, instruction ordering permitted by LLVM, and
other canonical forms may differ. The trusted solution path supports provenance
and self-testing; structural assertions and deterministic behavior are the
actual grading contract.

### Optimization predictions and pass output

`pass-output` entries should register structural observations, such as a removed
load, a surviving call, a folded return, or the presence/absence of a loop.
Byte-for-byte pass output is deliberately unsupported as a correctness oracle:
LLVM versions can rename blocks, reorder instructions, and select different
canonical forms. A registry entry may still request `opt -S` normalization.

### Markdown reviews and diagnostics

Written answers receive deterministic **rubric coverage**, not semantic proof.
Registry checks can require:

- concepts that all must appear or one of several accepted formulations;
- diagnostic vocabulary and repair consequences;
- absence of explicitly prohibited claims;
- references to prompt artifacts, operations, symbols, or ABI boundaries; and
- a minimum explanation length.

These checks are intentionally transparent and gameable. They are useful for
repeatable partial credit and omission reporting, but a passing markdown score
must not be represented as proof that the prose is technically correct.

### Points and skips

Each check belongs to a scoring dimension. The dimension's points are divided
evenly among its registered checks. Reports include points earned and available,
per-check status, skipped checks with reasons, and both exercise and aggregate
percentages. The ordinary score retains skipped points in the denominator so a
report never silently claims full tool-backed confidence. Reference self-tests
also calculate an executed-check percentage, allowing repositories without an
LLVM installation to validate deterministic structural/rubric behavior while
still reporting tool skips.

Suggested interpretation:

| Score | Interpretation |
| --- | --- |
| 90–100% | Strong artifact coverage; inspect skips before claiming tool-backed correctness |
| 70–89% | Substantial partial credit; one or more required properties are missing |
| 1–69% | Incomplete or structurally incorrect artifact |
| 0% | Missing/empty artifact or no satisfied checks |

## Tool requirements and discovery

The grader itself requires only Python 3 and the standard library—no package
installation is needed. Individual registry entries declare external tools:

- `llvm-as` for LLVM parsing/assembly;
- `opt` for verifier runs and textual normalization;
- `lli` only for deterministic execution vectors;
- `mlir-opt` for registered MLIR parsing.

Tools are discovered as the unsuffixed name, with `$LLVM_SUFFIX`, or as common
versioned names such as `opt-18`. Reports record paths and the first line of each
available tool's `--version` output. A missing tool skips only its dependent
checks and records the reason; it never turns a parser/verifier check into a
regex-only pass.

## Output contract

Text output is intended for humans. JSON output has top-level
`schema_version: "1.0"` and includes:

- registry path and SHA-256 digest;
- UTC generation timestamp;
- discovered toolchain versions;
- every check's pass/fail/skip state, dimension, message, command, bounded
  stdout/stderr, and points;
- exercise totals, skipped-check count, comparison strategy, and rubric-only
  label; and
- aggregate earned/available points and percentage.

Consumers should reject unknown major schema versions rather than guessing at
field semantics.

## Security boundaries

Submissions are untrusted. The harness reduces accidental exposure by using a
fresh temporary working directory, a minimal environment, bounded subprocess
output in reports, and a per-exercise timeout. It invokes tools with argument
arrays rather than a shell and never writes into the reference exercise tree.

This is **not a hardened sandbox**. In particular, `lli`, LLVM passes, parsers,
and native libraries may contain vulnerabilities; IR can consume CPU or memory;
and a process may access resources allowed to the grader's OS account. For
hostile submissions, run the whole grader in an external container/VM with:

- no secrets or credentials;
- no network;
- a read-only repository and disposable writable temp volume;
- unprivileged UID, syscall filtering, process limits, and strict CPU/memory/file
  quotas; and
- a trusted, patched LLVM toolchain.

The grader does not execute arbitrary commands from the registry or submission.
Only a trusted registry may be used. `--registry` is an administrator/testing
feature and must not point to learner-controlled data.

## Fixtures and reference self-tests

`fixtures/incomplete/attempts/` intentionally omits required behavior and prose
coverage. It verifies partial-credit and failure reporting; these files are not
solutions. Run the standard-library test suite with:

```sh
python3 -m unittest discover -s llvm-training/autograder/tests -v
```

Reference self-tests use the registry's checked-in solution paths and do not
copy answers over learner attempts or modify any reference artifact.
