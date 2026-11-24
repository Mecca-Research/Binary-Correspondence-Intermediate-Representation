# BCIR numerical contracts and precision propagation (2026-11-23)

**Status:** opt-in reference implementation; no production FP/BFP/SR or Posit lowering enabled.
**Implementation:** `bcir/kbcir/precision_contracts.py`. **Tests:** `bcir/tests/test_precision_contracts.py`.
**Prior foundations:** Smart Numbers (learned correction, not arithmetic semantics),
MPAT (backward error requirements and numerical reformulation), AEDACI
(conditional monitoring and safe fallback), BCIR Precision Policy P1-P6.

## Decision

No universal format wins. Select by *role*: storage, compute, accumulation,
rounding, range, tolerance, verified error and hardware cost. **Never silently
convert measured model-quality retention into a mathematical error proof.**
This module makes the first narrow MPAT-style step: named absolute-error
contracts on an acyclic graph, conservative backwards requirement propagation,
then a forward recomputation of the candidate error bounds. It is not an ML
accuracy simulator or a universal precision optimizer.

| Technique | Eligible first BCIR role | Claim that must NOT be made |
| --- | --- | --- |
| Wide fixed-point accumulator | Q8/integer dot products; width proven before native execution | Exact in a finite C register without overflow proof |
| Residual/error feedback | Explicit stateful quantizers and calibration *experiments* | Eliminates bias for every signal or restores clipping losses |
| Stochastic rounding | Seeded, counter-addressed low-precision training experiment | Automatically O(sqrt(N)) total error, universally stable training |
| BFP and OCP MX | Separate tensor storage/compute candidates, hardware qualified | Generic MXFP is integer arithmetic or always handles outliers |
| Fixed-point intervals | Static range/overflow and local error bounds | Interval width itself proves actual error |
| Quire/superaccumulator | Cold reference; qualified boundary reduction where worthwhile | A general wide accumulator is automatically a Posit Quire |
| Arbitrary/ball precision | High-confidence reference, constant evaluation, proof path | Default hot-loop arithmetic or always a faster alternative |

## Correct numerical semantics (P1/P2)

- An **exact** result is defined by its representation and overflow semantics. For
  symmetric signed quantizer codes, the minimum *sufficient* accumulator width
  can be derived from max code magnitude and count using
  `signed_dot_accumulator_bits(bits_a,bits_b,count)`. Native code must still
  implement and test that width. A Python unbounded integer is only the oracle.
- A **bounded** result has an absolute error amount in explicit units. A
  quantizer grid of `2**scale_exp` has at most half-step roundoff for nearest,
  or less than one step for truncation, **only when the input range is proved
  nonsaturating** and additional conversion errors are accounted for. There is
  no universal one-Q8-ULP bound for a different group grid. The reference helper
  converts the bound to conservative Q8 ticks.
- An **empirical** result, such as checkpoint-specific perplexity retention at
  8-bit or sub-1-bit weight storage, is not an arithmetic losslessness proof.
- A **predicted** result or learned Smart Numbers correction is a proposal.
  Only the independent verifier/qualification process may promote it.

The pre-existing `bcir/kbcir/precision.py` preserves compatibility with the
R17 opt-in *grid-ULP* score (and a fixed-point Q8 reduction score); the two
units are **not** interchangeable. The new reference contract is stricter:
mixed grids must first be converted into the same physical/numerical error
unit and the quantization/input sensitivity accounted for. Do not feed
unqualified `accuracy_bound` summaries into the strict planner as proofs.

## Backward requirements: the deliberately narrow proof scope (P3/P4)

For a topologically ordered acyclic graph, a node `v` has a *supplied* local
absolute-error bound `e_v` and nonnegative conservative error-amplification
coefficients `L_v,u` for each input `u`. All error amounts share one unit.
The verifier's forward recurrence is:

```text
E(v) <= e(v) + sum(L(v,u) * E(u) for each input u)
```

A requested output `y` has tolerance `tau_y`. Backward propagation sums
**all paths**, including multiple paths through a shared input, to compute
error-amplification coefficients `A_y,v`. Each contributor receives the
conservative budget `tau_y / (M_y * A_y,v)`, where `M_y` counts positive
contributors; multiple outputs use the tightest budget.

Then choose the least-cost *qualified* option **per node** within its budget,
and recompute the output bound. This is a feasible heuristic, **not a globally
minimum-cost assignment**, and it may reject feasible assignments because
the equal-share budget is deliberately conservative.

The proof is conditional on correct input range, local-error and sensitivity
bounds. Arbitrary nonlinear operators, different physical units, data-dependent
conditioning, unproved overflow safety, experimental stochastic bounds, and
untrusted empirical options are **rejected or excluded**, not guessed.

`PrecisionOption.assurance` distinguishes exact / bounded / empirical /
predicted. A nonempty evidence ID is required for exact or bounded candidates.
An ID is a provenance handle, **not** validation of the referenced proof
by itself; production registration must bind it to independently checked
artifacts.

## OCP MX (P1)

For the **OCP MX v1 concrete formats**, the metadata contract requires 32
elements per block and one E8M0 scale: MXFP8 (E4M3 or E5M2), MXFP6
(E2M3 or E3M2), MXFP4 (E2M1) and MXINT8. The reference planner validates
block-size and scale metadata; it does **not** implement bit-exact OCP MX
conversion, scaling, rounding or native silicon lowering.

In particular, MXFP4's element values are floating-point encoded, not
integer values. Outlier range conflicts, rounding, saturation and propagation
require separate analysis and a measured compiler/hardware gate.

Reference: OCP Microscaling Formats (MX) Specification v1.0,
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

## Stochastic rounding and residual feedback (P3/P5)

Unbiased stochastic rounding between adjacent representable points has
`E[SR(x) | x] = x` only under exact probabilities and a nonsaturating range.
A square-root-N cumulative-error statement depends on martingale/mean
independence assumptions, operation and error model. Nonlinear feedback and
limited-resolution PRNGs can introduce dependencies and bias; seeded replay
is not a numerical proof. Record PRNG algorithm/version, seed, operation ID,
logical element index and step so thread scheduling cannot change the
stream. No SR runtime conversion is shipped in this patch. The reference
planner deliberately excludes stochastic candidates from its deterministic
proof selection even when a local worst-case bound is supplied.

For error-feedback quantizers, preserve residual state and reset/version
semantics explicitly. Compensation may reduce long-run bias but cannot undo
saturation or prove downstream model accuracy. It must be qualified separately
from the already-proven integer Q8 residual-carry summation path.

References: Croci et al., *Stochastic Rounding: Implementation, Error
Analysis, and Applications* (2022); SIAM probabilistic rounding research.
https://pmc.ncbi.nlm.nih.gov/articles/PMC8905452/

## Qualification boundaries (P4/P5/P6)

1. Run exact-encoding, source-range, format and accumulator-width tests.
2. Compare numerical kernels against an independent high-precision reference
   with adversarial conditioning, overflow, saturating, underflow and outlier
   fixtures. Do not confuse local roundoff bounds with full-model guarantees.
3. For AI candidates, evaluate several frozen checkpoints/seeds and learning
   curves; include loss, CE/PPL, attention association/counterfactuals and
   equal-token **and** equal-wall-time comparisons.
4. Record hardware backend, compilation flags, storage/compute/accumulate
   format, PRNG/reduction-order contract and source SHA for every candidate.
5. Permit dynamic numerical fallback only at an isolated, replayable operation
   boundary and only to a pre-qualified implementation. No asynchronous
   patching of arithmetic and no kernel-wide numerical monitor.

The 50M study in BCIR PR #833 and Modular-AI PR #228 supplies preliminary
empirical comparisons. BF16/FP8/MXFP4 model-quality gates are **not**
converted into analytic per-operation error certificates by this module.

## Adoption sequence

- **Phase NC-1 (this patch):** pure Python typed contracts; no implicit
  lowering or change to legacy K_BCIR scores; conservative graph propagation;
  quantizer unit-conversion and accumulator-width helpers; explicit refusals.
- **Phase NC-2:** bind proof IDs to verifier-checked artifacts and use exact
  interval/affine or specialized FP error analysis to derive local bounds.
  Add support for heterogeneous physical units by typed sensitivity.
- **Phase NC-3:** supply measured hardware cost candidates and use Pareto or
  constrained global optimization, not the equal-budget heuristic.
- **Phase NC-4:** qualify native C kernels and selectively lower graph regions
  to BF16, FP8, OCP MX or a guarded FP64 fallback; add validated seeded SR
  as a separate experiment, not a silently enabled default.

No benchmark or broad CI result is claimed by this document; new regression
tests verify reference-planner behavior and explicit refusal contracts.
