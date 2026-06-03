# Pitfalls — Real-World LLVM IR Bugs

A checklist of common mistakes when writing or generating LLVM IR.
Most entries come from the sibling **BCIR** project, where each was
caught by `llvm-as` or `llvm-link` and fixed with a referenced commit.

Read these *before* writing IR. They're the cheapest insurance you'll
buy.

## Index

| # | File | One-liner |
|---|---|---|
| 01 | `01-nested-instruction-expressions.md` | `or i1 (xor i1 %x, true), %y` — invalid; can't nest instructions as expressions |
| 02 | `02-phi-predecessor-mismatch.md` | "PHI node entries do not match predecessors" |
| 03 | `03-duplicate-block-labels.md` | Two basic blocks with the same label name in one function |
| 04 | `04-duplicate-symbols.md` | Two modules `define` the same `@symbol` |
| 05 | `05-type-schema-drift.md` | `%T = type { i32, i32 }` in module A; `{ i32, i32, i32 }` in module B |
| 06 | `06-immarg-violation.md` | `call void @llvm.foo(i32 %dynamic)` where `i32` arg is declared `immarg` |

## How to use this list

When you write IR:
1. Skim the one-liner column.
2. If any look relevant to what you're writing, open that file.
3. Each pitfall page includes: the exact verifier message, a minimal
   reproducer, the fix pattern, and (where applicable) the BCIR
   commit that fixed the real instance.

When you debug failing IR:
1. Copy the `llvm-as` / `llvm-link` / `opt -passes=verify` error
   message.
2. Grep this directory for distinctive words from the message.
3. Each pitfall page documents the exact text the verifier emits.

## What this list *does not* cover

- **Optimization correctness bugs** (e.g., "my pass is producing
  poison"). Those are in pass-design territory.
- **Backend / codegen ICEs.** Use `llc -mtriple=... -O0` and report
  to LLVM.
- **Linker errors not caused by IR issues** (missing libraries,
  wrong target).

## Pattern recognition

Many of these bugs share a root cause: **a generator that synthesizes
IR algorithmically and doesn't validate against `llvm-as` after every
emit**.

If you write an IR generator (compiler, JIT, DSL frontend):

1. Pipe its output through `llvm-as -o /dev/null` in test.
2. Pipe through `opt -passes=verify -o /dev/null` to catch semantic
   bugs the assembler accepts.
3. Add the BCIR-style validate scripts (`runtime/llvm/validate_*.sh`
   in the parent repo) to CI.

Most of the pitfalls below would have been caught the first time the
generator was wired into CI.
