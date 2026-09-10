# Significance, Effect Size, and What "No Difference" Means

## Why the usual tools are the wrong ones

The t-test, the mean, and the standard deviation all assume a roughly symmetric,
roughly normal distribution. Timing data is none of those things:

- **Bounded below.** There is a fastest possible execution. Nothing is faster.
- **Unbounded above.** Any interference — an interrupt, a page fault, a
  neighbour — adds time. Nothing subtracts it.
- **Frequently multimodal.** Cold and warm cache, taken and not-taken branch,
  fast and slow path: distinct modes, not noise around one centre.

The consequence is that the mean sits between the modes where no execution ever
lands, and the standard deviation describes a spread the data does not have. Use
statistics that assume nothing about the shape.

## The four numbers, and what each answers

| Statistic | Question | Robust to |
| --- | --- | --- |
| Median ratio | how much faster? | outliers, skew |
| Bootstrap CI on that ratio | how sure are we of "how much"? | any distribution shape |
| Mann-Whitney U p-value | is there a difference at all? | non-normality; not ties (corrected) |
| Cliff's delta | how separated are the distributions? | shape, scale, outliers |

They answer different questions, and reporting only one of them is where most
reports go wrong.

### Mann-Whitney U

Ranks all observations from both series and asks whether one series
systematically occupies the higher ranks. It assumes no distribution shape. The
implementation in
[`../tools/analyze-benchmark-samples.py`](../tools/analyze-benchmark-samples.py)
uses average ranks for ties, applies the tie correction to the variance, applies
a continuity correction, and — importantly — **returns no p-value at all** below
8 samples per group, where the normal approximation is not supported.

A tool that prints a p-value it cannot justify is worse than one that prints
nothing: the number gets quoted.

### Cliff's delta

```
delta = (#(a > b) - #(a < b)) / (n1 * n2)
```

In [-1, +1]. It measures *stochastic dominance*: the probability that a random
sample from one series beats a random sample from the other. The conventional
magnitude bands are negligible (<0.147), small (<0.33), medium (<0.474), and
large above that.

Delta is not a speedup. Two distributions can be perfectly separated —
delta = +1 — while differing by 0.3%. That is exactly the
`significant-but-trivial.json` fixture, and it is why delta alone cannot
authorize a claim.

### The bootstrap confidence interval

Resample both series with replacement, recompute the median ratio, repeat, and
take percentiles of the resulting distribution. No distributional assumption,
and it answers the question people actually care about: *what range of speedups
is consistent with this data?*

The implementation is seeded, so the interval is reproducible: the same input
file always yields the same interval, which makes it a regression test as well
as a statistic.

### The rig's resolution

Before any of the above means anything, the measurement has to be able to see
the effect. Split the baseline into two interleaved halves and bootstrap *their*
ratio: the half-width of that interval is the smallest difference this rig can
distinguish from nothing. See
[`03-repeated-trials-and-noise-control.md`](03-repeated-trials-and-noise-control.md).

## The decision ladder

The analysis tool applies these in order, and every rung can answer
"inconclusive":

1. **Enough samples?** Below 20 per arm: `inconclusive`.
2. **Is either series drifting?** A monotone trend in run order:
   `inconclusive` — the comparison would attribute the drift to the change.
3. **Is the effect above the rig's resolution?** If not:
   `no-detectable-difference`.
4. **Is `p < alpha`?** If not: `no-detectable-difference`, reported *together
   with* the resolution, so the reader knows what could have been detected.
5. **Is Cliff's delta above negligible?** If not: `inconclusive` — significance
   without separation means the sample count did the work.
6. **Does the confidence interval exclude 1.0?** If not: `inconclusive` — the
   point estimate is not supported by its own interval.
7. **Is the effect above the declared practical threshold?** If not:
   `detectable-but-immaterial`.
8. Otherwise: `improvement` or `regression`.

Five of the eight rungs are refusals. That is not pessimism; it is the base rate.

## "No difference" is a claim, and it needs a bound

"We measured no difference" is only meaningful with the resolution attached:

> No difference detected (p = 0.62). With 40 samples per arm this measurement
> could resolve differences above 2.1%; a smaller regression would not have
> been visible.

Without that second sentence the statement is compatible with a 5% regression
the measurement was too coarse to see. The tool always prints the resolution
alongside a null result, for exactly this reason.

## Multiple comparisons

Testing twenty workloads at alpha = 0.05 gives you roughly one false positive
by construction. Options, in order of preference:

1. **Pre-register the primary workload.** One test, one claim. Everything else
   is exploratory and labelled as such.
2. **Control the family-wise error rate** (Bonferroni: divide alpha by the
   number of tests) when a small set of tests must all hold.
3. **Control the false discovery rate** (Benjamini-Hochberg) when many tests are
   screened and some false positives are acceptable.

What is not acceptable is running twenty tests, reporting the one that reached
significance, and describing it as the result.

## Worked example

`clean-improvement.json`, 40 samples per arm:

```
rig resolution: 1.08%  (smallest difference this measurement can distinguish from itself)
median ratio  : 0.8839  CI [0.8795, 0.8868]
HL shift      : -11.6229 ms
Mann-Whitney p: 0.0000
Cliff's delta : +1.000 (large)

VERDICT: improvement
  11.61% faster (median ratio 0.8839, CI [0.8795, 0.8868], p = 0.0000,
  Cliff's delta +1.000 = large)
```

Every rung passes: enough samples, no drift, an effect ten times the rig's
resolution, a tiny p-value, complete separation, an interval well clear of 1.0,
and a magnitude far above the 1% threshold. That is what a defensible claim
looks like — and note that all six numbers are needed to see it.

## Pitfalls

- **Reporting a p-value alone.** Says nothing about magnitude.
- **Reporting a speedup alone.** Says nothing about whether it is real.
- **Using the mean of a skewed distribution.** Reports a value nothing produced.
- **Treating a large Cliff's delta as a large speedup.** It measures separation.
- **Taking more samples until p < 0.05.** Optional stopping guarantees
  eventual significance regardless of truth.
- **Choosing the threshold after seeing the effect.** Declare it first.
- **Reading "not significant" as "no difference".** It means "not detected, at
  this resolution".
- **Ignoring the confidence interval when it contains 1.0.** The interval is the
  claim; the point estimate is a summary of it.

## BCIR notes

- Learned or measured data may **rank and calibrate**; it never becomes a
  legality verdict. A statistical result is an input to a decision, never the
  decision itself where correctness is at stake.
- Metric classes decide what a row may do: `exact` gates within a 2% band,
  `ratio` within a wide 25% band, `wall` never gates. The analysis tool refuses
  `--gate` on a `wall`-class row rather than letting a caller opt into it.
- The per-slice analysis protocol in this repository: on a gain, report the
  magnitude, the remaining headroom, and what the residual gap is made of; on no
  change, determine whether the effect was mis-assigned, cancelled, or already
  at the bound — *a proved bound retires the row, and that is a success*; on a
  regression, the slice does not land until it is explained.

## See also

- [`05-reporting-and-claim-discipline.md`](05-reporting-and-claim-discipline.md) — turning this into a claim
- [`examples/README.md`](examples/README.md) — the fixture set and its schema
- [`../tools/analyze-benchmark-samples.py`](../tools/analyze-benchmark-samples.py) — the implementation
