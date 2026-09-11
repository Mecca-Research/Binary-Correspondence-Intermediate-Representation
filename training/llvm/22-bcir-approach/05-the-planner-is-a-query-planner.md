# 05 — The planner is a query planner

A database optimiser, an instruction selector and BCIR's `K_BCIR` are the same
algorithm wearing three vocabularies. Each enumerates the plans that are *allowed*,
prices them, and returns the cheapest. Each is wrong in the same way when it gets the
order backwards.

This chapter is the corpus's bridge to the database subject, which
[`../../data/README.md`](../../data/README.md) has scoped and not yet opened. It is here
rather than there because the claim is checkable here: BCIR ships a working cost-governed
selector, and every number below is recomputed from it.

## The same three words

| query planner | instruction selection | BCIR |
| --- | --- | --- |
| the plan must return the right rows | the pattern must cover the DAG | `R(π, Θ) ⪯ B(H, Θ)`, laws R1–R25 |
| statistics and cardinality estimates | pattern costs, scheduling model | the twelve-axis cost vector `M(π, Θ)` |
| index, sort order, uniqueness | register class, addressing mode | lane type, phase, residency |
| "seq scan vs index scan" | "which pattern covers this" | `min_π M(π, Θ)` |

[`03-legality-before-cost.md`](03-legality-before-cost.md) is where the rule itself is
taught. What this chapter adds is a second, independent implementation of it, at a
different level of the stack, with different units — which is the only way to tell a
principle from a coincidence.

## A selector you can run

`bcir/asn1/selection.py` answers the question the whole ASN.1 portfolio exists to raise:
*given an abstract value, which encoding rule should carry it?* X.692 defines a notation
for **describing** encodings and says nothing about **choosing** among them. Choosing is an
optimiser's job.

Its `select` applies three laws in a fixed order, and the order is the entire content:

1. **Legality first, without reference to any number.** A candidate that cannot represent
   the value is not an expensive candidate; it is not a candidate. The measurement carries
   a `refusal` string rather than raising, so one unrepresentable value cannot abort the
   comparison.
2. **A property, not a cost.** A rule is *canonical* when one abstract value has exactly
   one encoding under it. A rule without that may be decoded but never selected for
   emission, because two conforming senders would produce two different digests for the
   same value.
3. **Only what survives both is priced.**

A query planner's equivalent of law 2 is the one people forget. A plan that returns the
right rows in an order the query did not pin is legal for `SELECT *` and illegal under
`ORDER BY`; the property, not the cost, decides whether it is a candidate at all.

## What the selector actually returns

The table is recomputed from `bcir.asn1.selection` by
[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py). The value is the
same in both columns — a `SEQUENCE` holding the integer `200`. Only the *schema* differs:
the right-hand column constrains that integer to `0..255`.

<!-- generated: plan-selection -->
| candidate | canonical | octets, no constraint | octets, `(0..255)` |
| --- | :---: | ---: | ---: |
| `DER` | yes | 6 | 6 |
| `CANONICAL-PER-UNALIGNED` | yes | 3 | 1 |
| `CANONICAL-PER-ALIGNED` | yes | 3 | 1 |
| `COER` | yes | 3 | 1 |
| `JER-BCIR-CANONICAL` | yes | 9 | 9 |
| `BER` | no | 6 | 6 |
| `BASIC-PER-UNALIGNED` | no | 3 | 1 |
| `BASIC-PER-ALIGNED` | no | 3 | 1 |
| `BASIC-OER` | no | 3 | 1 |
| `JER` | no | 9 | 9 |

| objective | decided by | selects, no constraint | selects, `(0..255)` |
| --- | --- | --- | --- |
| `none` | definition | `DER` | `DER` |
| `memory` | arithmetic | `CANONICAL-PER-UNALIGNED` | `CANONICAL-PER-UNALIGNED` |
<!-- /generated -->

Two things in that table are worth more than the rest of this chapter.

**The constraint moves only the candidates that can read it.** Adding `(0..255)` takes
PER and OER from three octets to one, and leaves DER and JER exactly where they were.
Nothing about the value changed; a *declaration* changed, and only the rules that consult
declarations benefited.

That is the database lesson in its smallest form. Declaring a column `TINYINT`, adding a
`CHECK`, or marking it `NOT NULL` does not change a single row — it changes what the
planner is allowed to assume. An engine that reads the constraint plans differently; an
engine that does not, or a query written so the constraint cannot apply, pays the
unconstrained price. "The index made no difference" is almost always this.

**`none` is a named objective, not an absent one.** With no budget the selector returns
DER — today's answer, unchanged — because "we did not ask for anything" has to be a
decision the system can state and reproduce, not a gap where a decision should be. A
planner with no cost model does not become neutral; it becomes arbitrary, and nobody can
tell the two apart from the outside.

## Two kinds of truth, in the units they arrive in

The table above carries no timings, and that is deliberate.

`octets` is **exact**: deterministic arithmetic, the same on every host, forever. So are
legality, canonicality, and the `none` and `memory` objectives decided from them — which
is why a generated block can pin them byte for byte.

`Objective.ENCODE_LATENCY` and `Objective.DECODE_LATENCY` are **measured**, by timing the
Python oracle on whatever machine is running. On the host this text was written they both
select `COER`; on another host, or another day, they might not. The corpus does not pin
them, and the module itself says why: a calibrated cost table for a real decision lives in
`kbcir/microbench.py` and the frozen-table machinery, not in a timing loop.

A database has exactly this split and loses track of it constantly. The schema is exact:
this column is four bytes, this index exists, this join key is unique. The statistics are
estimates: this predicate selects 3% of rows, this table has a million rows. A plan chosen
from estimates is a *guess priced honestly*; a plan chosen from the schema is arithmetic.
The failure is not estimating — it is reporting an estimate in the same voice as a fact,
which is what BCIR's two-truth rule exists to forbid: a measured cost may rank candidates
and must never become a legality verdict.

## Pitfalls checklist

- Do not treat a cheaper plan as a legal one. Legality is settled first, with no number in
  hand, or the cheapest illegal plan wins every time.
- Do not let a property be priced. Canonicality, sort order and uniqueness decide
  *candidacy*; costing them invites the optimiser to trade away a guarantee for a
  microsecond.
- Do not read "no objective" as "no decision". `Objective.NONE` returns DER by definition
  and says so; an optimiser with an unnamed default returns whatever its code happens to
  do first.
- Do not pin a measured number as though it were an exact one. Wire size is arithmetic;
  latency is a measurement with a host attached.
- Do not conclude a constraint is useless because the plan did not change. Check whether
  the rule you are using reads constraints at all — DER never will.

## Checks

[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py) recomputes the
`plan-selection` block from `bcir.asn1.selection` and compares it byte for byte, so the
table cannot drift from the selector it describes: a candidate added or removed, a
canonicality flag flipped, or an encoding whose length changed all fail this chapter
rather than silently making it wrong.

The block deliberately contains only exact quantities. Regenerate it with:

```sh
python3 training/llvm/tools/verify-bcir-approach.py --update
```

The claim that the ASN.1 selector obeys the same three laws is not re-proved here — it is
BCIR's own, and `bcir/tests/test_asn1_selection.py` holds it, including the case this
chapter is built on: a constraint that moves only the candidates that can read it.
