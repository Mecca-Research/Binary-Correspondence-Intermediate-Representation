# Pitfall 06 — `immarg` Parameter Violation on Intrinsics

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_prefetch_profiles.ll` | `1f62e86` | `opt -passes=verify runtime/llvm/bcir_prefetch_profiles.ll -o /dev/null` | Replace SSA operands to `immarg` intrinsic parameters with literal constants. | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md); [`reference/intrinsics.md`](../reference/intrinsics.md); [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |

## The error

```
llvm-as: assembly parsed, but does not verify as correct!
immarg operand has non-immediate parameter
i32 %rw
```

## What's happening

Some LLVM intrinsics declare certain parameters as `immarg`. An
`immarg` parameter **must be a compile-time constant**, never an SSA
value. The verifier rejects any non-constant argument.

Common offenders:

- `llvm.prefetch(ptr %p, i32 <rw>, i32 <locality>, i32 <cache_type>)`
  — all three `i32` args are `immarg`.
- `llvm.lifetime.start.p0(i64 <size>, ptr <ptr>)` — size is `immarg`.
- `llvm.memcpy.inline.*` — count is `immarg`.
- `llvm.assume(i1)` — argument is just an i1, not immarg per se, but
  has its own constraints.
- `llvm.experimental.constrained.*` — rounding mode and exception
  behavior args are `immarg`.

## Minimal reproducer

```llvm
declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @bad(ptr %p, i32 %rw, i32 %loc) {
  call void @llvm.prefetch(ptr %p, i32 %rw, i32 %loc, i32 1)   ; ❌
  ret void
}
```

```
$ opt -passes=verify bad.ll -o /dev/null
immarg operand has non-immediate parameter
i32 %rw
```

## Fix

Pass literal constants:

```llvm
define void @good(ptr %p) {
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)        ; ✓
  ret void
}
```

If you need runtime selection between a few constant variants, branch
on the runtime value and call the intrinsic with constants in each
arm:

```llvm
define void @runtime_dispatch(ptr %p, i32 %rw) {
  switch i32 %rw, label %default [
    i32 0, label %read
    i32 1, label %write
  ]
read:
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
  ret void
write:
  call void @llvm.prefetch(ptr %p, i32 1, i32 3, i32 1)
  ret void
default:
  ret void
}
```

This emits one call site per constant variant, satisfying `immarg`.

## The real BCIR instance

`runtime/llvm/bcir_prefetch_profiles.ll` had:

```llvm
define void @bcir.op.prefetch.linear(ptr %base, i64 %offset, i32 %locality, i32 %rw) {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  call void @llvm.prefetch(ptr %p, i32 %rw, i32 %locality, i32 1)   ; ❌
  ret void
}
```

Both `%rw` and `%locality` are runtime SSA values passed to `immarg`
parameters. Fixed in commit `1f62e86` by replacing the SSA args with
literal `i32 0, i32 3`:

```llvm
call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)   ; ✓
```

(The wrapper function signature kept `%rw`/`%locality` but no longer
uses them — they're effectively dead args. A cleaner fix would also
remove them.)

## How to detect early

When wrapping an intrinsic that has `immarg` parameters, **inline
the constants** in your wrapper rather than threading runtime values
through. If the wrapper must support runtime selection, use the
switch pattern above.

## Where to check `immarg` declarations

Search the LLVM source for `immarg`:

```bash
grep -rn 'ImmArg' llvm/include/llvm/IR/Intrinsics*.td
```

Or, for any intrinsic you're using, check the declaration in
`llvm/include/llvm/IR/Intrinsics.td` (or one of the
target-specific `Intrinsics*.td`). `immarg` is annotated on each
parameter that requires a constant.

## See also

- [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) — operand kinds
- [`../reference/intrinsics.md`](../reference/intrinsics.md) — common intrinsics
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — `call` instruction
