# Exercise 014: Attach HAM hint metadata to a lowered memory operation

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR concept being modeled

Model a BCIR hardware-affinity / hazard-affinity memory (HAM) hint that survives
lowering as LLVM metadata on the concrete memory operation. The hint should not
change the value semantics of the load; it should annotate the lowered access
for later analysis or code generation passes.

Write a standalone LLVM IR module that defines:

```llvm
define i32 @bcir.exercise.load_with_ham_hint(ptr %base, i64 %offset)
```

The function should compute `base + offset`, load an `i32`, and attach custom
metadata named `!bcir.ham.hint` to the load. Also add named module metadata that
describes the hint vocabulary.

## Required LLVM constructs

- `getelementptr i8` for byte-addressed pointer movement.
- A scalar `load i32` with an instruction metadata attachment.
- At least one metadata node containing strings and integer fields that describe
  the BCIR hint, such as domain, locality, lane, or hazard mode.
- A named metadata root that makes the hint discoverable at module scope.

## Expected verification command

```sh
llvm-as -disable-output llvm-training/exercises/014-ham-hint-metadata.solution.ll
```

## Expected observation

The module assembles successfully. The learner should observe that BCIR lowering
can preserve non-semantic scheduling or hardware hints as LLVM metadata while the
actual memory operation remains a normal load.

## Optional runtime reference

Compare this with schema metadata patterns in `runtime/llvm/bcir_claim_schema.ll`
and `runtime/llvm/bcir_schedule_schema.ll`.
