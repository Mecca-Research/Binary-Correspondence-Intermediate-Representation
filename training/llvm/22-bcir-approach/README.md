# BCIR's Own Approach

Every other chapter in this corpus teaches a compiler you can install. This one
teaches the IR this repository *is*: a **correspondence IR** that decides whether
a plan is legal before it prices it, and prices it against a twelve-axis cost
vector rather than a heuristic buried in a pass.

The subject exists because the corpus could not previously answer the question a
reader arrives with — *what is BCIR, as an IR?* — from anything it taught.
`K_BCIR` appeared in the whole `training/` tree exactly once, in an index row,
and no chapter named a single verifier law.

## Key takeaways

- A correspondence IR holds the **claim** and its **realization** as two linked
  objects. LLVM IR is one of the renderings of the second; the first does not
  survive into it, which is why `llvm-as` accepting a module says nothing about
  whether the plan behind it was legal.
- **Legality precedes cost, and it is not a tie-breaker.** A budget that makes
  the cheap plan illegal forces a *more expensive* legal one; tighten it further
  and the planner raises `Infeasible` rather than returning a plan that does not
  fit.
- The cost vector is **twelve declared axes**, not twelve live ones. Which axes
  are produced, and which are merely weighted, is a fact about this tree — and
  this subject reports it from the tree rather than describing it.
- The same program, unchanged, chooses a different lane width on each substrate
  and under a different live state Θ. Nothing about the program says `16`.

## A word that means two different things here

`14-mlir-bridge/` and `18-mlir-lowering-to-llvm/` use **legality** in MLIR's
sense: a `ConversionTarget` declares which operations may *remain* after a
conversion, and "illegal" means "must be rewritten before this boundary".

This subject uses **legality** in BCIR's sense: a verifier law (R1–R25) refuses a
plan outright, before `K_BCIR` is allowed to price it. The two senses are not in
tension — they are checks at different boundaries — but a reader who carries the
first into this subject will misread every page of it.

## Division of labour with `bcir-mapping/`

[`bcir-mapping/`](../bcir-mapping) lowers BCIR concepts *into LLVM IR*: vertices,
strides, ABI structs, metadata. In particular
[`bcir-mapping/11-normal-forms-and-verification.md`](../bcir-mapping/11-normal-forms-and-verification.md)
owns the LLVM-side stage contract — what a *hypothetical* `bcir-verify` pass
would enforce on a lowered module.

This subject owns the rails that ship: the executable oracle under
[`bcir/`](../../../bcir) and the MLIR law rail under [`mlir/`](../../../mlir),
and the cost model that sits between them.

## Lesson map

| Page | Teaches |
|---|---|
| [`01-correspondence-versus-structure.md`](01-correspondence-versus-structure.md) | What "correspondence" buys that structure does not: the same program, four substrates, two live states |
| [`02-what-k-bcir-prices.md`](02-what-k-bcir-prices.md) | The twelve axes, which of them anything actually produces, and what each policy weights |
| [`03-legality-before-cost.md`](03-legality-before-cost.md) | The budget ladder, `Infeasible` as a verdict, and the laws on both rails |

## Cross-links

- [`indexes/bcir-crossrefs.md`](../indexes/bcir-crossrefs.md) — where each BCIR concept lives in the tree.
- [`17-new-pass-manager/04-adaptive-bcir-pipelines.md`](../17-new-pass-manager/04-adaptive-bcir-pipelines.md) — driving a pass pipeline from a BCIR plan.
- [`19-hardware-aware/03-calibration-governor.md`](../19-hardware-aware/03-calibration-governor.md) — where the substrate profile H comes from.

## Checks

Run [`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py) after
editing any page here. Every table in this subject marked
`<!-- generated: … -->` is recomputed from `bcir/` and compared byte for byte, so
a number that drifts from the code fails rather than ages; the same gate holds
the twelve axis names identical across the oracle, the MLIR attribute and
`docs/PARITY.md`, and refuses any `opt` pass name the installed toolchain does
not have.
