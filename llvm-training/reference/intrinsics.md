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

## Common categories

### Math

| Intrinsic | Lowers to |
|---|---|
| `llvm.sqrt.<T>` | `sqrt` |
| `llvm.sin.<T>`, `llvm.cos.<T>`, `llvm.tan.<T>` | libm calls |
| `llvm.exp.<T>`, `llvm.exp2.<T>`, `llvm.log.<T>`, `llvm.log2.<T>`, `llvm.log10.<T>` | libm |
| `llvm.pow.<T>(x, y)` | `pow` |
| `llvm.powi.<T>(x, i)` | Integer-power form |
| `llvm.fabs.<T>` | `fabs` |
| `llvm.copysign.<T>` | `copysign` |
| `llvm.floor.<T>`, `llvm.ceil.<T>`, `llvm.trunc.<T>`, `llvm.round.<T>`, `llvm.rint.<T>`, `llvm.nearbyint.<T>` | Rounding modes |
| `llvm.fma.<T>(a, b, c)` | Fused multiply-add |
| `llvm.fmuladd.<T>` | Multiply-add that may fuse |
| `llvm.minnum.<T>`, `llvm.maxnum.<T>` | IEEE 754 min/max |
| `llvm.minimum.<T>`, `llvm.maximum.<T>` | NaN-propagating min/max |

### Bit manipulation

| Intrinsic | Effect |
|---|---|
| `llvm.ctlz.<T>(v, is_zero_poison)` | Count leading zeros |
| `llvm.cttz.<T>(v, is_zero_poison)` | Count trailing zeros |
| `llvm.ctpop.<T>(v)` | Population count |
| `llvm.bswap.<T>(v)` | Byte swap |
| `llvm.bitreverse.<T>(v)` | Bit reverse |
| `llvm.fshl.<T>(a, b, n)` | Funnel shift left |
| `llvm.fshr.<T>(a, b, n)` | Funnel shift right |

The `is_zero_poison` argument to `ctlz`/`cttz` is `immarg`.

### Overflow-checked arithmetic

Each returns a `{T, i1}` struct: `{result, overflow_flag}`.

| Intrinsic | Op |
|---|---|
| `llvm.sadd.with.overflow.<T>` | Signed add |
| `llvm.uadd.with.overflow.<T>` | Unsigned add |
| `llvm.ssub.with.overflow.<T>`, `llvm.usub.with.overflow.<T>` | Sub |
| `llvm.smul.with.overflow.<T>`, `llvm.umul.with.overflow.<T>` | Mul |

```llvm
%r = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
%sum = extractvalue { i32, i1 } %r, 0
%ovf = extractvalue { i32, i1 } %r, 1
```

### Memory

| Intrinsic | Effect |
|---|---|
| `llvm.memcpy.p0.p0.<size>(dst, src, n, isvolatile)` | Memory copy |
| `llvm.memmove.p0.p0.<size>(dst, src, n, isvolatile)` | Overlap-safe copy |
| `llvm.memset.p0.<size>(dst, val, n, isvolatile)` | Memory fill |
| `llvm.lifetime.start.p0(size, ptr)` | Object lifetime begins |
| `llvm.lifetime.end.p0(size, ptr)` | Object lifetime ends |
| `llvm.invariant.start.p0(size, ptr)` | Memory immutable from here |
| `llvm.invariant.end.p0(start, size, ptr)` | Pairs with invariant.start |
| `llvm.prefetch(ptr, rw, locality, cache)` | Prefetch hint |

**Caveat:** all `immarg` parameters (sizes for `memcpy.inline` and
`lifetime.*`, all args of `prefetch`) must be compile-time constants.
See [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md).

### Atomic primitives

Almost always you'll use the `atomicrmw`, `cmpxchg`, and `fence`
instructions directly rather than intrinsics, but a few helpers exist:

| Intrinsic | Effect |
|---|---|
| `llvm.assume(i1)` | Assume the condition is true |
| `llvm.expect.<T>(val, expected)` | Branch-probability hint |
| `llvm.trap()` | Unconditional trap |
| `llvm.debugtrap()` | Like `trap` but reserves for debugger |

### Vector reductions

| Intrinsic | Effect |
|---|---|
| `llvm.vector.reduce.add.<T>(vec)` | Sum lanes |
| `llvm.vector.reduce.mul.<T>(vec)` | Product of lanes |
| `llvm.vector.reduce.and.<T>(vec)` | Bitwise AND across lanes |
| `llvm.vector.reduce.or.<T>(vec)` | Bitwise OR across lanes |
| `llvm.vector.reduce.xor.<T>(vec)` | Bitwise XOR across lanes |
| `llvm.vector.reduce.smin.<T>`, `umax`, `smin`, `smax` | Per-lane min/max |
| `llvm.vector.reduce.fadd.<T>(start, vec)` | FP sum with starting value |
| `llvm.vector.reduce.fmax.<T>`, `fmin` | FP min/max |
| `llvm.vector.reduce.fmaximum`, `fminimum` | NaN-propagating |

### Coroutine intrinsics

`llvm.coro.id`, `llvm.coro.begin`, `llvm.coro.suspend`,
`llvm.coro.resume`, etc. — used by `clang -fcoroutines-ts` and
similar frontends. Rarely written by hand.

### Exception handling

`llvm.eh.typeid.for`, `llvm.eh.exceptionpointer`,
`llvm.eh.exceptioncode`, `llvm.eh.sjlj.setjmp`,
`llvm.eh.sjlj.longjmp`, etc.

### Debug info

| Intrinsic | Effect |
|---|---|
| `llvm.dbg.declare(metadata, !var, !expr)` | Variable lives at the pointed address |
| `llvm.dbg.value(metadata, !var, !expr)` | Variable has this value here |
| `llvm.dbg.addr(metadata, !var, !expr)` | Variable address (DI v2) |

These are erased before codegen; they only affect debug output.

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
  will fail at codegen, not at IR parse.

- **Forgetting that intrinsics are `declare`, not `define`.** You
  don't write the body.

## See also

- [`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) — `call` instruction
- [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) — `immarg` constraint
- LLVM LangRef: https://llvm.org/docs/LangRef.html#intrinsic-functions
