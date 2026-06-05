# Profile and Optimization Metadata

## TL;DR

Profile and optimization metadata guides transformations without changing
LLVM IR language semantics. It tells LLVM which paths are likely hot,
which functions and call sites were frequently executed, which indirect-call
targets were common in a training run, which loops are good candidates for
vectorization or unrolling, and which facts the optimizer may use when they are
true.

Official references:

- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
- [LLVM LangRef — `!prof` metadata](https://llvm.org/docs/LangRef.html#prof-metadata)
- [LLVM LangRef — Branch Weight Metadata](https://llvm.org/docs/LangRef.html#branch-weight-metadata)
- [LLVM LangRef — `llvm.loop` metadata](https://llvm.org/docs/LangRef.html#llvm-loop)

See the standalone examples:

- [`examples/profile-branch-weights.ll`](examples/profile-branch-weights.ll)
- [`examples/profile-entry-count.ll`](examples/profile-entry-count.ll)
- [`examples/profile-value-indirect-call.ll`](examples/profile-value-indirect-call.ll)
- [`examples/loop-metadata.ll`](examples/loop-metadata.ll)

For how profile metadata participates in full production optimization pipelines,
see [`../07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md).

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

## `function_entry_count`

Function entry-count metadata records how many times profile collection observed
a function being entered:

```llvm
define i32 @hot_worker(i32 %x) !prof !0 {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 10000}
```

The attachment is on the function definition, not on an instruction in the
function body. LLVM combines the function entry count with block-frequency
information (BFI) to estimate basic-block execution counts inside the function.
Those estimated block counts feed optimization heuristics even when an
individual branch does not have explicit `branch_weights` metadata.

Typical optimizer effects:

- A high entry count marks the function as hot, increasing the benefit side of
  inlining decisions for calls into or inside the function.
- A low entry count can make a function a candidate for cold code placement,
  reduced inlining, or less aggressive code-size growth.
- Entry counts help normalize branch weights into absolute hotness, so a hot
  function with a `99:1` branch may be treated differently from a tiny cold
  helper with the same ratio.

See [`examples/profile-entry-count.ll`](examples/profile-entry-count.ll) for a
minimal function-level profile example.

## Value profiling metadata for indirect-call targets

Value-profile (`VP`) metadata records hot runtime values at an instruction. For
indirect calls, the profiled values are hashes of likely callee function names:

```llvm
%result = call i32 %callee(i32 %x), !prof !0

!0 = !{!"VP", i32 0, i64 1600,
       i64 7651369219802541373, i64 1030,
       i64 -4377547752858689819, i64 410}
```

Read this tuple as:

1. `!"VP"` — this is value-profile metadata.
2. `i32 0` — value-profile kind 0, meaning indirect-call targets.
3. `i64 1600` — total dynamic executions of the indirect call.
4. Pairs of `i64 <target-name-hash>, i64 <count>` for the hottest observed
   targets.

The per-target counts do not have to sum to the total execution count because
PGO usually records only the hottest values. The target hashes are profile-data
identifiers rather than IR-level function references, so hand-written examples
should treat them as illustrative profile payloads, not as a stable way to name a
callee in source code.

This metadata is the input that makes indirect-call promotion profitable: LLVM
can clone or split the call site so the hottest target is checked and called
directly, while preserving a fallback indirect call for all other targets. The
transformation still has to preserve semantics; profile metadata only says which
runtime values were likely in the training workload.

See [`examples/profile-value-indirect-call.ll`](examples/profile-value-indirect-call.ll)
for a compact indirect-call value-profile example.

## Function and call profiling metadata

`!prof` can represent several profiling concepts depending on the IR construct
and node shape:

| Profile metadata | Where it appears | What it describes |
|---|---|---|
| `branch_weights` | `br`, `switch`, `select`, and some call sites | Relative hotness of successors or call outcomes. |
| `function_entry_count` | Function definitions | Dynamic function entry count from profile data. |
| `VP` kind `0` | Indirect calls | Hottest indirect-call target hashes and counts. |
| `VP` kind `1` | Memory intrinsics | Hottest observed memory lengths. |

Always check the LangRef for the exact tuple layout before emitting metadata.
The key rule is the same as branch weights: profile metadata is guidance. It can
affect optimization decisions and layout, but it cannot make source-level
behavior disappear.

## How PGO metadata influences optimization

PGO metadata connects training-workload measurements to LLVM's cost models:

- **Inlining.** Hot function entry counts and hot call-site/block frequencies
  make call overhead more important and make exposed scalar simplifications more
  valuable. Cold callees or cold call sites are less likely to be inlined when
  doing so would grow code size.
- **Indirect-call promotion.** `VP` metadata on an indirect call identifies the
  hottest targets. LLVM can add target checks, direct calls, and fallback paths,
  which can then enable normal inlining and devirtualization-like cleanups on the
  promoted direct edge.
- **Block layout.** Branch weights and derived block frequencies let layout
  passes place hot successors close together, make likely paths fall through, and
  outline or separate cold blocks when profitable.
- **Branch probabilities.** Branch-weight metadata becomes `BranchProbabilityInfo`
  and `BlockFrequencyInfo`, giving later passes a common view of which paths are
  likely. A pass may choose a transformation that benefits the hot path even if
  it leaves a colder path slightly larger or slower.

These effects are pipeline-level behavior: PGO metadata may be consumed before,
during, or after LTO/ThinLTO import, and post-link tools such as BOLT can apply
another profile-driven layout pass after IR has been compiled. For the broader
workflow, see [`../07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md).

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
values and obey the same memory rules. Value profiling can turn a hot indirect
callee into a guarded direct-call fast path, but the fallback indirect call must
still handle other legal targets.

## Pitfalls

- **Attaching loop metadata to the wrong branch.** `!llvm.loop` belongs
  on the loop latch terminator associated with the loop ID, not on an
  arbitrary branch nearby.
- **Forgetting the self-reference.** Many loop metadata examples use
  `!2 = distinct !{!2, ...}`. Omitting it can confuse loop-ID handling.
- **Stale profile data.** After CFG rewrites, old branch weights may no
  longer match successor order, path frequency, or the current deployment
  workload.
- **Mismatched value-profile payloads.** Duplicate VP target hashes are invalid,
  and target counts should describe the same call site and training profile as
  the total count.
- **False optimization facts.** Incorrect `!range`, `!nonnull`, `!tbaa`,
  alias scope, or noalias metadata can enable invalid transformations.
- **Metadata that optimizers may legally drop.** If a pass cannot update
  metadata precisely, it may remove it. Do not use metadata as the only
  place where required program behavior is represented.

## See also

- [`examples/profile-branch-weights.ll`](examples/profile-branch-weights.ll)
- [`examples/profile-entry-count.ll`](examples/profile-entry-count.ll)
- [`examples/profile-value-indirect-call.ll`](examples/profile-value-indirect-call.ll)
- [`examples/loop-metadata.ll`](examples/loop-metadata.ll)
- [`01-metadata-basics.md`](01-metadata-basics.md)
- [`../05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md) — branch weights on `br`
- [`../07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) — PGO, LTO, ThinLTO, and BOLT pipeline context
- [LLVM LangRef — `!prof` Metadata](https://llvm.org/docs/LangRef.html#prof-metadata)
- [LLVM LangRef — Branch Weight Metadata](https://llvm.org/docs/LangRef.html#branch-weight-metadata)
- [LLVM LangRef — `llvm.loop` metadata](https://llvm.org/docs/LangRef.html#llvm-loop)
