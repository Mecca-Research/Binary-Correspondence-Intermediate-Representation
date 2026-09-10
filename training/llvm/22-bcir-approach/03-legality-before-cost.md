# Legality Before Cost

## Key takeaways

- Legality is decided **before** pricing, and the ordering is observable: a
  budget that makes the cheapest plan illegal produces a plan that costs *more*.
- When no legal realization fits the budget, the planner raises `Infeasible`.
  A refusal is a verdict, not a fallback to the nearest plan that does not fit.
- The verifier laws R1–R25 exist on **two rails** — the executable oracle under
  `bcir/` and the MLIR law rail under `mlir/`. Same laws, two mechanisms.
- Two of those laws are enforced by the oracle at a *different moment* than by
  the law rail, in a different module. "Which rail implements it" is the wrong
  question; "at which moment does each rail refuse" is the right one.

## The ladder

One module (`vector_add(1024)`), one substrate (`x86-64-avx512`), one live state
(cool). Only the thermal cap moves.

<!-- generated: thermal-ladder -->
```text
thermal cap   chosen    score   thermal used
     (none)   vec16      7808           1088
       2000   vec16      7808           1088
       1088   vec16      7808           1088
       1000   vec8       9472            640
        700   vec8       9472            640
        500   --           --   Infeasible
        300   --           --   Infeasible
```
<!-- /generated -->

Three regions, and the middle one is the whole lesson:

1. **Cap above what the best plan uses.** The budget is slack; the argmin is
   unchanged. Legality has no work to do.
2. **Cap below the best plan and above the next.** The cheap wide plan is now
   *illegal*, so it is not priced at all — and the planner returns a plan whose
   score is strictly worse. A cost-first engine cannot produce this row: it would
   have to notice afterwards that its winner does not fit, and then repair.
3. **Cap below every candidate.** `Infeasible`, naming the claim and the cap it
   could not satisfy. Not the closest plan, not the cheapest violation.

`K_BCIR(G | H, Θ) = min over π in Legal(G, H) of M(π, Θ)` subject to
`R(π, Θ) ⪯ B(H, Θ)`. Region 2 is what the `Legal(…)` in that expression buys, and
region 3 is what happens when the set is empty.

## Why this ordering is a design decision, not an optimization

A planner that prices first and checks later has to answer "what do I do with a
cheap illegal plan?" — and every answer is bad: repair it (and the price you
quoted is now wrong), return it with a warning (and the budget was decoration),
or search again (and the cost model was consulted for nothing).

Deciding legality first makes the refusal total. It also makes the plan's
resource vector auditable *before* anything is emitted, which is the property a
deployed system needs: `feasible(result, Θ, budget)` is documented in
`bcir/kbcir/` as the correctness property a budget-unaware compiler has no
concept of — that a plan provably fits its thermal, power and bandwidth caps.

## The laws, on both rails

The verifier laws R1–R25 are the legality half of that expression. They exist
twice on purpose:

| Rail | Mechanism | When it refuses |
|---|---|---|
| Oracle (`bcir/`) | executable Python checks | at plan time, and — for the encoding laws — at encode time |
| Law rail (`mlir/`) | verifier passes over the dialect | at `bcir-opt` verify time |

The two-rail arrangement is this repository's central discipline: one semantic
truth, two realizations, and a differential that proves they agree. Neither rail
is the specification; `docs/BCIR_LANGREF.md` is, and both rails are checked
against it.

### The case that teaches the most: R24 and R25

It is tempting to say the ASN.1 laws (R24, X.680–X.697) and the ECN laws (R25,
X.692) are "law-rail only", because `bcir/verify/` does not contain them.

That is false, and the way it is false is the lesson. The oracle *does* enforce
them — at **encode time**, in `bcir/asn1/`, where an encoder refuses to emit
octets that violate the rule. The law rail enforces the same rules at **verify
time**, rejecting a module that carries such an encoding.
`bcir/tests/test_asn1_law_parity.py` pins one against the other and says so
directly: the same rules, checked at two different moments, which is the whole
point of having a law rail *and* an executable oracle.

A reader — or a checker — that looks for a law in one directory and concludes the
rail lacks it has mistaken a module boundary for a rail asymmetry. The gate for
this subject resolves oracle laws across `bcir/verify/` **and** `bcir/asn1/` for
exactly this reason.

## Pitfalls checklist

- Do not describe legality as a filter applied to the cheapest plan. It is
  applied to the candidate set, before pricing.
- Do not treat `Infeasible` as an error to be caught and worked around; it is the
  answer to a question whose other answers would be false.
- Do not conclude a rail lacks a law because one directory does not mention it.
- Do not cite a law number range for the oracle without checking the tree:
  `tools/docs/check_law_range.py` exists because those ranges go stale in prose.

## Checks

[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py)
regenerates the ladder above by re-running the planner, and requires every R-law
named on this page to be findable on both rails — the oracle side resolved across
`bcir/verify/` and `bcir/asn1/`.
