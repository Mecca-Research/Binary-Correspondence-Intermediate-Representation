# What K_BCIR Prices

## Key takeaways

- The cost vector has **twelve axes in a fixed order**, and that order is the
  same list in three places: the oracle, the MLIR attribute, and the prose in
  `docs/PARITY.md`. The gate for this subject holds all three equal.
- Twelve axes are **declared**, not twelve produced. One axis has no producer
  anywhere in non-test `bcir/` — and it is the one the `safe` policy weights.
- One axis has four producers and is weighted **zero** by every shipped policy.
  Declared-and-unpriced sits next to produced-and-unweighted, in the same table.
- A survey of an interface is not a survey of its callers: the planner's
  innermost constructor builds cost vectors **positionally**, so a keyword scan
  of the codebase cannot see it. The table below says so rather than hiding it.

## The twelve axes, as the tree has them

The `files that produce it` column counts files under non-test `bcir/` that pass
a nonzero value for that axis **by keyword**. The remaining columns are the
weight each shipped policy gives the axis (`bcir/kbcir/weights.py`).

<!-- generated: axes -->
| # | axis | files that produce it | latency | throughput | energy | safe |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `compute` | 10 | 2 | 2 | 1 | 1 |
| 1 | `memory` | 11 | 2 | 3 | 1 | 1 |
| 2 | `fabric` | 1 | 1 | 1 | 1 | 1 |
| 3 | `sync` | 2 | 1 | 1 | 1 | 2 |
| 4 | `compile` | 0 | 0 | 0 | 0 | 1 |
| 5 | `thermal` | 14 | 0 | 0 | 2 | 1 |
| 6 | `power` | 8 | 0 | 0 | 3 | 1 |
| 7 | `reliability` | 1 | 1 | 1 | 1 | 3 |
| 8 | `security` | 0 | 0 | 0 | 0 | 1 |
| 9 | `accuracy` | 4 | 0 | 0 | 0 | 0 |
| 10 | `contention` | 6 | 1 | 2 | 1 | 1 |
| 11 | `verification` | 1 | 1 | 1 | 1 | 2 |
<!-- /generated -->

Read the two extreme rows together, because they are the lesson:

- **`security`** — produced by nothing. `perf`, `throughput` and `energy` weight
  it `0`; `safe` weights it `1`. So the one policy that would price it is the one
  policy for which nothing supplies it. It is a declared axis with a reserved
  meaning, and this corpus labels it that way rather than implying a live signal.
- **`accuracy`** — produced by four modules, and weighted `0` by all four
  policies. The signal exists and no shipped policy currently pays for it.

Neither is a defect. Both are the difference between an IR that *declares* a
resource model and one that has *implemented* every axis of it, and a corpus that
cannot tell you which is which is not teaching the IR — it is advertising it.

### Why the producer count is keyword-only, and says so

`bcir/kbcir/realize.py` builds the planner's innermost cost vector positionally,
with a comment saying it does so deliberately (resolving names through a dict on
every call is the wrong cost in the hot constructor). A keyword scan therefore
under-counts: `compile` shows a producer count that misses that constructor.

The check is left keyword-only *and documented*, rather than quietly extended
into a half-parser, because the honest boundary of a scan is part of what it
reports. A checker that claimed to survey producers and silently missed the
planner's own would be worse than one that states its reach.

## Fixed point, stated narrowly

"All arithmetic is integer" is a determinism claim, and it is true. "Everything
is Q8" is a format claim, and it is not.

- The **coupling factor** is Q8: `256` is ×1.0, applied as `(c * f) >> 8` — an
  arithmetic shift, so coupling truncates and is lossy in a defined direction.
- The **MLIR attribute** declares its components as Q16 fixed point
  (`BCIR_CostVectorAttr`, `mlir/include/BCIR/BCIRAttrs.td`).
- The **policy weights** are plain integers, multiplied with no shift at all.

Three different fixed-point conventions in one pipeline is exactly the kind of
detail that a summary flattens and a debugger then pays for.

## The axis order lives in three places

The order is not documentation, it is an interface: the oracle indexes the vector
positionally, the MLIR attribute names its parameters in that order, and
`docs/PARITY.md` states it in prose for readers. Three copies drift unless
something holds them equal, so the gate for this subject parses all three and
requires them identical, name for name and position for position.

That is this repository's own law about repeated definitions applied to itself —
see `docs/security/laws.md` L14: one predicate per repeated defect, and make it
total.

## Pitfalls checklist

- Do not describe the cost vector as "twelve measured resources". Twelve
  declared; check the table for which are produced.
- Do not read a weight of `0` as "this axis does not matter"; read it as "no
  shipped policy prices it today".
- Do not assume the Q8 coupling factor makes every component Q8.
- Do not add a fourth copy of the axis order without adding it to the gate.

## Checks

[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py)
regenerates the table above from `bcir/kbcir/cost.py` and
`bcir/kbcir/weights.py`, and separately requires the axis order to be identical
in the oracle, `mlir/include/BCIR/BCIRAttrs.td` and `docs/PARITY.md`.
