# Profile and Optimization Metadata

## TL;DR

Profile and optimization metadata guides transformations without changing
LLVM IR language semantics. It tells LLVM which paths are likely hot,
which loops are good candidates for vectorization or unrolling, and which
facts the optimizer may use when they are true.

Official references:

- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
- [LLVM LangRef — Branch Weight Metadata](https://llvm.org/docs/LangRef.html#branch-weight-metadata)
- [LLVM LangRef — `llvm.loop` metadata](https://llvm.org/docs/LangRef.html#llvm-loop)

See the standalone example: [`examples/loop-metadata.ll`](examples/loop-metadata.ll).

## `!prof` and branch weights

Branch-weight metadata is attached with `!prof`:

```llvm
br i1 %cmp, label %hot, label %cold, !prof !0
!0 = !{!"branch_weights", i32 99, i32 1}
```

The weights are relative counts or probabilities. They do not need to
sum to 100. A `99:1` pair means the first successor is expected to be
about 99 times as frequent as the second successor.

Common uses:

- Block layout and code placement.
- Inlining and call-site profitability decisions.
- Choosing between equivalent transformations when one favors hot code.
- Preserving measured or estimated profile-guided optimization (PGO)
  information.

For multi-way terminators such as `switch`, the metadata has one weight
for each destination in the order documented by the LangRef.

## Function and call profiling metadata

`!prof` can also represent other profiling concepts such as function
entry counts and value profiling, depending on the IR construct and node
shape. Always check the LangRef for the exact tuple layout before
emitting it.

The key rule is the same as branch weights: profile metadata is guidance.
It can affect optimization decisions and layout, but it cannot make a
source-level behavior disappear.

## `!llvm.loop`

Loop metadata is usually attached to the loop latch branch, often the
conditional branch that jumps back to the loop header:

```llvm
br i1 %again, label %loop, label %exit, !llvm.loop !2

!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.unroll.count", i32 4}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
```

The first operand is conventionally a self-reference to the loop ID.
That is why loop metadata is commonly `distinct`.

Common loop metadata families include:

| Metadata | Intent |
|---|---|
| `llvm.loop.unroll.*` | Request, disable, or parameterize loop unrolling. |
| `llvm.loop.vectorize.*` | Request, disable, or parameterize vectorization. |
| `llvm.loop.interleave.*` | Control interleaving decisions. |
| `llvm.loop.mustprogress` | Record progress requirements where valid. |

Loop metadata is not a magic override. If vectorization would violate
memory dependencies, poison rules, target constraints, or required
semantics, the optimizer must not vectorize merely because metadata asks
for it.

## Other optimization-related tags

| Tag | Example | Optimizer use |
|---|---|---|
| `!tbaa` | `load i32, ptr %p, !tbaa !5` | Refines alias analysis with type-based facts. |
| `!range` | `load i32, ptr %p, !range !6` | Narrows possible loaded/call-result values. |
| `!nonnull` | `load ptr, ptr %slot, !nonnull !7` | Says a loaded pointer is non-null. |
| `!prof` | `br ..., !prof !8` | Carries branch weights and other profile data. |
| `!llvm.loop` | latch branch attachment | Carries loop transformation hints and attributes. |

Example range metadata:

```llvm
%small = load i32, ptr %p, align 4, !range !0
!0 = !{i32 0, i32 10} ; valid values are in [0, 10)
```

## How metadata guides optimization without changing semantics

Think of metadata as one of two categories:

1. **Facts** — `!range`, `!nonnull`, `!tbaa`, alias metadata. These must
   be true when attached. Optimizers may use them to simplify or reorder
   code. If the fact is false, the IR producer has created misleading IR.
2. **Preferences/measurements** — `!prof`, many `!llvm.loop.*` nodes.
   These guide profitability and strategy. They do not introduce new
   behavior, and they do not permit unsound transformations.

For example, branch weights can make LLVM lay out the hot successor
fall-through, but both successors still exist. Loop unroll metadata can
make unrolling more likely, but the unrolled loop must compute the same
values and obey the same memory rules.

## Pitfalls

- **Attaching loop metadata to the wrong branch.** `!llvm.loop` belongs
  on the loop latch terminator associated with the loop ID, not on an
  arbitrary branch nearby.
- **Forgetting the self-reference.** Many loop metadata examples use
  `!2 = distinct !{!2, ...}`. Omitting it can confuse loop-ID handling.
- **Stale profile data.** After CFG rewrites, old branch weights may no
  longer match successor order or path frequency.
- **False optimization facts.** Incorrect `!range`, `!nonnull`, `!tbaa`,
  alias scope, or noalias metadata can enable invalid transformations.
- **Metadata that optimizers may legally drop.** If a pass cannot update
  metadata precisely, it may remove it. Do not use metadata as the only
  place where required program behavior is represented.

## See also

- [`examples/loop-metadata.ll`](examples/loop-metadata.ll)
- [`01-metadata-basics.md`](01-metadata-basics.md)
- [`../05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md) — branch weights on `br`
- [LLVM LangRef — Branch Weight Metadata](https://llvm.org/docs/LangRef.html#branch-weight-metadata)
- [LLVM LangRef — `llvm.loop` metadata](https://llvm.org/docs/LangRef.html#llvm-loop)
