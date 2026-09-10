# Comparisons and `select` (`icmp`, `fcmp`, and branchless choice)

## TL;DR

```llvm
%c  = icmp slt i32 %a, %b        ; -> i1
%f  = fcmp olt double %x, %y     ; -> i1, ordered (false if either is NaN)
%m  = icmp sgt <4 x i32> %v, %w  ; -> <4 x i1>, one lane per element
%r  = select i1 %c, i32 %a, i32 %b
br i1 %c, label %then, label %else
```

Comparisons produce `i1` (or a vector of `i1`), never a target flag register and
never an integer 0/1. `select` is the value-level counterpart to a branch: same
choice, no new basic blocks, no `phi`.

## `icmp` predicates

| Predicate | Meaning |
| --- | --- |
| `eq`, `ne` | Equality; sign-agnostic |
| `ugt`, `uge`, `ult`, `ule` | Unsigned ordering |
| `sgt`, `sge`, `slt`, `sle` | Signed ordering |

The operands may be integers, pointers, or vectors of either. Both operands must
have the same type.

Signedness lives in the **predicate**, not in the type: `i32` is neither signed
nor unsigned. `icmp slt i32 %a, %b` and `icmp ult i32 %a, %b` compare the same
bits and disagree whenever the sign bit differs. Choosing the wrong one is the
most common frontend-lowering bug in this family, and it survives every type
check.

### Comparing pointers

```llvm
%same = icmp eq ptr %p, %q
%below = icmp ult ptr %p, %q
```

Equality on pointers is always meaningful. Relational comparison is only
meaningful for pointers into the same allocation; across allocations the result
is not specified by the language and must not be used to derive a fact about
memory layout. Both pointers must be in the same address space.

## `fcmp` predicates: ordered versus unordered

A float comparison has to answer a second question first: is either operand
`NaN`? "Ordered" means *neither operand is NaN*; "unordered" means *at least one
is*.

| Ordered | Unordered | Relation |
| --- | --- | --- |
| `oeq` | `ueq` | equal |
| `one` | `une` | not equal |
| `ogt` | `ugt` | greater than |
| `oge` | `uge` | greater or equal |
| `olt` | `ult` | less than |
| `ole` | `ule` | less or equal |
| `ord` | `uno` | (no relation — just "neither is NaN" / "one is NaN") |
| `true` | `false` | constant results |

Read the leading letter as the NaN answer: **o**rdered predicates are `false`
when a NaN is present; **u**nordered predicates are `true` when a NaN is
present.

Two consequences worth memorising:

- `fcmp oeq double %x, %x` is **not** a tautology — it is `false` when `%x` is
  NaN. It is the idiomatic "is not NaN" test, and `fcmp uno double %x, %x` is
  the idiomatic "is NaN" test.
- C's `x != y` lowers to `fcmp une`, not `fcmp one`. A frontend that emits the
  ordered form changes the NaN behaviour of every comparison it lowers.

### Fast-math flags change the answer

```llvm
%c = fcmp nnan olt double %x, %y
```

`nnan` tells the optimizer no operand is NaN; `ninf` says the same about
infinities. Under those flags the ordered/unordered distinction collapses and
the optimizer will freely rewrite between the two. That is correct *only* if the
promise holds — otherwise the comparison silently changes meaning on exactly the
inputs the distinction existed for. See
[`../13-advanced-ir/06-fast-math-flags.md`](../13-advanced-ir/06-fast-math-flags.md).

## Vector comparisons

Comparing vectors yields a vector of `i1` — a mask, one lane per element:

```llvm
%mask = icmp sgt <4 x i32> %v, %w      ; <4 x i1>
%sel  = select <4 x i1> %mask, <4 x i32> %v, <4 x i32> %w
```

A `<N x i1>` mask cannot feed `br`; `br` needs a scalar `i1`. Reduce it first:

```llvm
%any = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %mask)
br i1 %any, label %some, label %none
```

Masks are also the currency of masked loads/stores and interleaved access; see
[`../09-vectorization/07-masked-and-interleaved-access.md`](../09-vectorization/07-masked-and-interleaved-access.md).

## `select`

```
<result> = select [fast-math flags] <cond-ty> <cond>, <ty> <true-val>, <ty> <false-val>
```

- `<cond-ty>` is `i1` for a scalar select, or `<N x i1>` for a lane-wise select.
- Both arms have the same type, and that type is the result type.
- With a vector condition, both arms must be vectors of the same element count.

### `select` evaluates both arms

This is the property that decides when `select` is legal at all. Unlike a
branch, both operands are ordinary SSA values that already exist. Two rules
follow:

1. **`select` does not make a side effect conditional.** Replacing a guarded
   `load` with a `select` over two loads introduces a load that the original
   program never performed. If the guarded pointer could be null or
   out-of-bounds, that is a new fault. This is why `SimplifyCFG` only speculates
   instructions it can prove are safe to execute unconditionally.
2. **`select` does not block poison.** If the not-taken arm is `poison`, the
   `select` result is poison — a branch would never have evaluated it. Repair
   this with `freeze` on the value whose poison must not propagate:

```llvm
%safe = freeze i32 %maybe_poison
%r    = select i1 %c, i32 %safe, i32 0
```

See [`../13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md).

### `select` versus `phi`

| | `select` | `phi` |
| --- | --- | --- |
| Blocks | none added | one per incoming edge |
| Side effects | both arms already evaluated | only the taken path executes |
| Cost model | branchless; no misprediction | branch; may be cheaper when strongly biased |
| Codegen | `cmov`/predication, or a branch anyway | branch |

`SimplifyCFG` converts small diamonds to `select`; the backend may convert a
`select` back into a branch. Neither direction is guaranteed, so do not build a
correctness argument on which one survives. Build it on the semantics above:
if either arm must *not* be evaluated, you need a branch.

## Idioms worth recognising

```llvm
; boolean not
%not = xor i1 %c, true

; clamp to a lower bound (branchless)
%lt   = icmp slt i32 %x, 0
%clmp = select i1 %lt, i32 0, i32 %x

; sign-extend an i1 to a full mask of ones/zeros
%m = sext i1 %c to i32

; zero-extend a predicate to a counter increment
%inc = zext i1 %c to i32
%n   = add i32 %acc, %inc

; three-way compare without branches
%gt  = icmp sgt i32 %a, %b
%lt2 = icmp slt i32 %a, %b
%hi  = select i1 %gt, i32 1, i32 0
%cmp = select i1 %lt2, i32 -1, i32 %hi
```

## Checked example

[`examples/comparisons-and-select.ll`](examples/comparisons-and-select.ll) is a
standalone module covering signed/unsigned predicates, ordered/unordered float
comparisons, pointer equality, a vector mask with a reduction into `br`, and a
`freeze`-guarded `select`.

```bash
llvm-as training/llvm/05-control-flow/examples/comparisons-and-select.ll -o /dev/null
opt -passes=verify -S training/llvm/05-control-flow/examples/comparisons-and-select.ll -o /dev/null
```

## Pitfalls

- **Signed predicate on unsigned data (or the reverse).** Compiles, verifies,
  and produces the wrong answer for exactly half the input domain.
- **`fcmp one` where the source language means `!=`.** Changes NaN behaviour.
- **Feeding a `<N x i1>` to `br`.** The verifier rejects it; reduce the mask
  first.
- **Comparing values of different types.** `icmp eq i32 %a, %b` where `%b` is
  `i64` is a verifier error — extend or truncate explicitly.
- **Treating `i1` as if it were an 8-bit boolean in memory.** `i1` in registers
  is one bit; the ABI's `_Bool` is a byte. Cross the boundary with an explicit
  `zext`/`trunc`, not by assuming a layout.
- **`select` used to speculate a faulting operation.** Both arms are evaluated;
  a speculated load or division is a new fault, not an optimization.
- **Assuming `select` filters poison.** It does not. `freeze` does.
- **Nested comparisons inside a branch condition.** `br i1 (icmp ...)` is not
  IR syntax — name the comparison first.

## BCIR notes

- Legality predicates lowered from BCIR verifier laws must not be relaxed by
  fast-math flags. A comparison that decides a *legality* question belongs in
  the strict, flag-free form, because the two-truth separation forbids letting a
  measured or learned relaxation change a verdict.
- Lane-wise masks are the natural lowering for BCIR lane-typed predicates. Keep
  the mask width and the lane type in agreement with the claim's declared lanes;
  a silently widened mask is a mapping drift, not a performance choice.
- When a comparison result becomes a diagnostic fact (a refusal reason), keep it
  as an explicit named value so the diagnostic metadata can point at it. See
  [`../bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md).

## See also

- [`02-conditional-br.md`](02-conditional-br.md) — consuming the `i1`
- [`05-call-and-ret.md`](05-call-and-ret.md) — the other half of the dedicated instruction chapters
- [`../13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md) — why `freeze` is needed
- [`../09-vectorization/README.md`](../09-vectorization/README.md) — masks in vector code
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — one-line syntax reference
