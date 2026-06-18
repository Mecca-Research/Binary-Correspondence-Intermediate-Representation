# Exercise 012: Wrap an LLVM intrinsic behind a BCIR operation

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR concept being modeled

Model the BCIR lowering pattern where a high-level BCIR operation is represented
as a small named wrapper around a target-independent LLVM intrinsic. The wrapper
keeps the BCIR operation name stable while the body uses canonical LLVM IR.

Write a standalone LLVM IR module that declares `@llvm.uadd.with.overflow.i32`
and defines:

```llvm
define i32 @bcir.exercise.uadd_checked.i32(i32 %lhs, i32 %rhs, ptr %overflow_out)
```

The function should call the intrinsic, extract the sum and overflow flag, store
the overflow flag to `%overflow_out`, and return the sum.

## Required LLVM constructs

- An intrinsic declaration returning `{ i32, i1 }`.
- A wrapper function with a BCIR-style symbol name.
- A `call` to the intrinsic.
- `extractvalue` instructions for both aggregate fields.
- A `store i1` to communicate side-band status.

## Expected verification command

```sh
llvm-as -disable-output llvm-training/exercises/012-custom-intrinsic-wrapper.solution.ll
```

## Expected observation

The module assembles successfully. The learner should observe that BCIR wrappers
can preserve operation vocabulary while delegating arithmetic semantics to LLVM
intrinsics.

## Optional runtime reference

Compare this wrapper shape with the BCIR operation wrappers in
`runtime/llvm/bcir_ops.ll`.
