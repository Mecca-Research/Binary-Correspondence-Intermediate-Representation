# Intrinsics — Common LLVM Built-ins

LLVM intrinsics are `declare`d functions with the `llvm.` name
prefix. The compiler knows them by name and lowers them specially
(to specific instructions, libcalls, or assembly).

## Calling an intrinsic

```llvm
declare double @llvm.sqrt.f64(double)
declare i32 @llvm.ctlz.i32(i32, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias, ptr noalias, i64, i1)

define double @demo() {
  %r = call double @llvm.sqrt.f64(double 2.0)
  ret double %r
}
```

Many intrinsics are **overloaded**: the type is encoded in the
mangled name. `llvm.sqrt.f64` takes a `double`; `llvm.sqrt.f32`
takes a `float`. You must declare the exact mangled name you call.
For a compact table of common families, signatures, attributes, and `immarg`
constraints, start with [`intrinsics-quickref.md`](intrinsics-quickref.md). For
target-specific namespaces and token/special types, see
[`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md)
and [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md).

## Lookup dispatcher

| Need | Read |
|---|---|
| How to declare and call an intrinsic | This file |
| Category quick reference for math, bit operations, overflow, memory, lifetime, atomics, reductions, constrained FP, target-specific namespaces, coroutines, EH, and debug info | [`intrinsics-quickref.md`](intrinsics-quickref.md) |
| Runnable common intrinsic examples | [`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) |
| Target-specific namespace and feature requirements | [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) |
| `immarg` pitfall | [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |

## Category summary

| Category | Examples | Details |
|---|---|---|
| Math and constrained FP | `llvm.sqrt.*`, `llvm.fma.*`, `llvm.experimental.constrained.*` | [`intrinsics-quickref.md#math`](intrinsics-quickref.md#math), [`intrinsics-quickref.md#constrained-floating-point-intrinsics`](intrinsics-quickref.md#constrained-floating-point-intrinsics) |
| Bit manipulation | `llvm.ctlz.*`, `llvm.cttz.*`, `llvm.ctpop.*`, `llvm.bswap.*` | [`intrinsics-quickref.md#bit-manipulation`](intrinsics-quickref.md#bit-manipulation) |
| Overflow-checked arithmetic | `llvm.uadd.with.overflow.*`, `llvm.smul.with.overflow.*` | [`intrinsics-quickref.md#overflow-checked-arithmetic`](intrinsics-quickref.md#overflow-checked-arithmetic) |
| Memory and lifetime | `llvm.memcpy.*`, `llvm.memmove.*`, `llvm.memset.*`, `llvm.lifetime.start.*` | [`intrinsics-quickref.md#memory-intrinsics`](intrinsics-quickref.md#memory-intrinsics), [`intrinsics-quickref.md#lifetime-intrinsics`](intrinsics-quickref.md#lifetime-intrinsics) |
| Atomics, reductions, prefetch | `llvm.*atomic*`, `llvm.vector.reduce.*`, `llvm.prefetch` | [`intrinsics-quickref.md#atomic-primitives-and-optimizer-hints`](intrinsics-quickref.md#atomic-primitives-and-optimizer-hints), [`intrinsics-quickref.md#vector-reduction-intrinsics`](intrinsics-quickref.md#vector-reduction-intrinsics), [`intrinsics-quickref.md#prefetch-and-cache-related-intrinsics`](intrinsics-quickref.md#prefetch-and-cache-related-intrinsics) |
| Target-specific, coroutine, EH, debug | `llvm.x86.*`, `llvm.coro.*`, `llvm.eh.*`, `llvm.dbg.*` | [`intrinsics-quickref.md#target-specific-intrinsic-naming-patterns`](intrinsics-quickref.md#target-specific-intrinsic-naming-patterns), [`intrinsics-quickref.md#coroutine-intrinsics`](intrinsics-quickref.md#coroutine-intrinsics), [`intrinsics-quickref.md#exception-handling`](intrinsics-quickref.md#exception-handling), [`intrinsics-quickref.md#debug-info`](intrinsics-quickref.md#debug-info) |

## How to find the canonical declaration

Look in LLVM's source:
- `llvm/include/llvm/IR/Intrinsics.td` — generic
- `llvm/include/llvm/IR/IntrinsicsX86.td`, `IntrinsicsAArch64.td`,
  `IntrinsicsAMDGPU.td`, etc. — target-specific

Each declaration tells you:
- The mangled name pattern
- The argument types (overloaded vs fixed)
- Which arguments are `immarg`
- Function attributes (`nounwind`, `readnone`, `readonly`, etc.)

## Pitfalls

- **`immarg` violation.** See [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md).

- **Wrong name mangling.** `llvm.sqrt.f64(float %x)` won't link — the
  name says `f64`, the argument is `f32`. Use `llvm.sqrt.f32` (or
  cast).

- **Calling target-specific intrinsics on the wrong target.** They
  will fail at codegen, not at IR parse. See
  [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md).

- **Volatile is not atomic.** Volatile memory intrinsics preserve observable
  access behavior but do not create synchronization. See
  [`../08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md).

- **Forgetting that intrinsics are `declare`, not `define`.** You
  don't write the body.

## See also

- [`intrinsics-quickref.md`](intrinsics-quickref.md) — category quick reference for common intrinsic families
- [`instruction-quickref.md`](instruction-quickref.md) — instructions that pair with intrinsic results, including aggregate extraction, vector operations, atomics, EH pads, and `freeze`
- [`../09-vectorization/README.md`](../09-vectorization/README.md) — vectorizer overview and examples
- [`../11-concurrency/`](../11-concurrency/) — atomic orderings and volatile-vs-atomic
- [`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) — advanced common intrinsic signatures, overloaded names, and examples
- [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) — target-specific intrinsic namespaces, feature requirements, and portability
- [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) — `token`, `metadata`, `half`, `bfloat`, `x86_amx`, and scalable vectors
- [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) — `call` instruction
- [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) — `immarg` constraint
- [`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md) — missed vectorization due to memory dependence uncertainty
- LLVM LangRef: https://llvm.org/docs/LangRef.html#intrinsic-functions
