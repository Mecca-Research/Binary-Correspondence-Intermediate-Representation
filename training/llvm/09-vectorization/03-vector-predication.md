# Vector Predication and Masked Execution

**"Vector predication" is a specific thing in LLVM, not a general description.** It names
the `llvm.vp.*` intrinsic family, every member of which carries a mask *and* an explicit
vector length.

<!-- generated: vp-family -->
In LLVM 23 the family is **90** intrinsics spelled `llvm.vp.*`, **4** still staged as
`llvm.experimental.vp.*`, and the length helper `llvm.experimental.get.vector.length` —
**95** names, out of 524 target-independent intrinsics in the whole language.
<!-- /generated -->

If you have met the word only as a synonym for "masking", the family will look like
unfamiliar territory when you open real RISC-V or SVE output. It is not; it is the
mechanism the term was coined for.

This chapter covers both mechanisms and, more importantly, when each one is the answer.

## Two questions, not one

A masked operation asks one question per lane: **is this lane active?** A vector-predicated
operation asks two, and they are independent:

| Operand | Question | Shape |
| --- | --- | --- |
| `%mask` | Which lanes are active, lane by lane? | `<vscale x 4 x i1>` |
| `%evl` | How many lanes are live at all? | `i32` |

Every `llvm.vp.*` intrinsic ends with exactly those two operands:

```llvm
%r = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %a,
                                                  <vscale x 4 x i32> %b,
                                                  <vscale x 4 x i1> %mask, i32 %evl)
```

Lanes at or beyond `%evl` take no part in the operation and their results are poison.
Lanes below `%evl` behave as the mask says.

**Neither operand substitutes for the other.** A mask cannot shorten the vector — it still
describes a full-width operation. A length cannot express "every lane except the third".
That separation is the whole design: `%mask` carries a *conditional*, `%evl` carries a
*tail*, and a loop usually has both.

## Why this beats a scalar remainder loop

The classic vectorization problem is a trip count that is not a multiple of the vector
width. The traditional answer is a scalar remainder loop; the cost is a second copy of the
body and a branch.

With VP, the final iteration simply passes a smaller `%evl`:

```llvm
%remaining = sub i64 %n, %i
%evl = call i32 @llvm.experimental.get.vector.length.i64(i64 %remaining, i32 4, i1 true)
```

`llvm.experimental.get.vector.length` asks the target how many elements it is willing to
process, given the remaining trip count, the vectorization factor, and whether the type is
scalable. One loop body handles every iteration including the last.

## `%evl` is not a software fiction

On RISC-V's vector extension it becomes `vsetvli`, the instruction that configures the
active vector length. Compiling
[`examples/vector-predication-evl.ll`](examples/vector-predication-evl.ll) with
`llc -mtriple=riscv64 -mattr=+v` produces:

```asm
	vsetvli	a5, a5, e32, m2, ta, ma
	vle32.v	v8, (a7)
	vadd.vv	v8, v8, v10
	vse32.v	v8, (a6)
```

The value computed by `get.vector.length` is the operand `vsetvli` is configured with. When
the mask is not all-true the loads and stores carry RVV's mask register as `v0.t` — both
gates reaching the hardware separately, exactly as the IR describes them.

That is why VP exists rather than being folded into masking: on a target with a hardware
active-length register, the length is *cheaper* than a mask and expresses the tail
directly.

## How this differs from `llvm.masked.*`

The corpus teaches `llvm.masked.load`/`store`/`gather`/`scatter` elsewhere, and they are
not the same tool:

| | `llvm.masked.load` | `llvm.vp.load` |
| --- | --- | --- |
| Mask | Yes | Yes |
| Length | **No** | **Yes** (`%evl`) |
| Inactive lanes | Come from an explicit **passthru** operand | **Poison** |
| Expresses a tail | No | Yes |

`masked.load` needs a passthru because it must produce *something* for an inactive lane.
VP needs none: lanes beyond `%evl` are poison by definition, so there is nothing to merge.

Reach for the masked family when you have a per-lane condition and a full-width vector.
Reach for VP when the *length* varies — which, in a loop over a dynamic trip count, it
does.

## The other mechanism: mask and select

Predication without either intrinsic family is still legitimate and still common:

```llvm
%mask = icmp ult <4 x i32> %idxs, %limit
%old = load <4 x i32>, ptr %fallback
%new = add <4 x i32> %a, %b
%merged = select <4 x i1> %mask, <4 x i32> %new, <4 x i32> %old
```

This is the right shape when the operation is cheap and safe to perform on every lane, and
you only need to choose the result. It is the wrong shape when performing the operation on
an inactive lane would fault, trap, or cost more than the select saves — which is exactly
when you want a genuinely masked or predicated operation instead.

## Core ideas

| Idea | Meaning |
| --- | --- |
| Mask | A vector of `i1` values saying which lanes are active. |
| Explicit vector length (`%evl`) | How many leading lanes participate at all; the rest are poison. |
| Predicated operation | Executes or commits only for lanes the mask *and* the length admit. |
| Tail handling | Processing leftover elements when the trip count is not a multiple of the vector width. |
| Scalable vector | `<vscale x N x T>`, where hardware chooses the runtime multiple. |

## When to care

- Trip counts are dynamic and scalar remainder loops dominate runtime.
- The target has a hardware active-length register (RVV) or native predication
  (SVE-style scalable vectors).
- Branches are lane-local and can be represented as masks instead of CFG splits.
- BCIR lanes have enable bits or validity masks that naturally map to vector predicates —
  and BCIR lane *counts* that map to `%evl` rather than to a mask.

## Pipeline placement

For BCIR, predication should cross the optimization boundary as explicit masks until the
pass that can legally lower them. The advanced optimization chapter places masked vector
lowering after semantic BCIR lowering, SCCP poison repair, and loop canonicalization; see
[`../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline`](../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline).
The New Pass Manager chapter explains how to spell these stages as `opt -passes=...`
pipelines; see
[`../17-new-pass-manager/01-passbuilder-and-pipelines.md`](../17-new-pass-manager/01-passbuilder-and-pipelines.md).

## Pitfalls

- Do not read "vector predication" as a synonym for masking. In LLVM it names `llvm.vp.*`,
  and a search for the term that lands on `select` will leave you unable to read RVV
  output.
- A mask is not a bounds proof for inactive lanes unless the operation is truly masked; an
  ordinary vector load may still touch every lane's address.
- `select` prevents a value from being chosen, but both operands must already be safe to
  compute.
- Do not use a mask to express a tail. It is the wrong operand: it says which lanes are
  active in a full-width vector, not how many lanes there are.
- Lanes at or beyond `%evl` are **poison**, not zero and not the previous value. If you
  need a defined value there, VP is not what you want — merge explicitly.
- Scalable vectors are not arrays with a compile-time element count. Avoid code that
  assumes `vscale` is known during IR construction.
- The `llvm.experimental.vp.*` names (`reverse`, `splice`, `strided.load`,
  `strided.store`) are still in the staging namespace and may be renamed without a
  deprecation period — see
  [`../23-version-movement/01-reading-the-delta.md`](../23-version-movement/01-reading-the-delta.md)
  for what graduation does to a name.

## BCIR lowering advice

If BCIR already has lane masks, preserve them as explicit `i1` vectors or mask values long
enough for vector passes and target lowering to see them. Do not scalarize masks into
unrelated branches unless later passes need scalar CFG. Where BCIR carries a lane *count*
rather than a per-lane predicate, lower it to `%evl` and not to a mask — the two are
different facts, and collapsing them loses the one a hardware length register can use.

## Checks

[`examples/vector-predication-evl.ll`](examples/vector-predication-evl.ll) is assembled and
verified by [`../tools/verify-examples.sh`](../tools/verify-examples.sh) on every host: the
VP family predates the corpus baseline, so it needs no version directive. The RVV assembly
quoted above is reproducible with `llc -mtriple=riscv64 -mattr=+v` on any LLVM new enough
to have the RISC-V vector backend; the corpus's `llc` smoke gate is host-target only, so
that particular command is documented rather than gated, and this chapter says so rather
than implying a check that does not exist.
