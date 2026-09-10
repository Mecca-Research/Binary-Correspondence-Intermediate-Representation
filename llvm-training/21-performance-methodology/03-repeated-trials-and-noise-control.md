# Repeated Trials and Noise Control

## Why repetition, and what it buys

A single timing is a draw from a distribution you have not seen. Repetition
gives you the distribution's shape, and the shape is what tells you whether a
difference between two builds is distinguishable from the difference between a
build and itself.

The quantity that matters is not "how long did it take" but **how wide is the
interval this rig can resolve**. The analysis tool measures that directly, by
splitting the baseline into two interleaved halves and bootstrapping the ratio
of their medians:

```
rig resolution: 6.49%  (smallest difference this measurement can distinguish from itself)
VERDICT: no-detectable-difference
  the observed 3.36% difference is within this rig's 6.49% resolution;
  the measurement cannot distinguish it from noise
```

That is the `below-resolution.json` fixture: a real 1.5% effect, measured on a
noisy rig, correctly refused. The arithmetic in the report was never wrong. The
measurement simply could not see it.

## Run order is the design, not a detail

Running all baseline samples, then all candidate samples, confounds the
comparison with everything that changed in between — temperature, frequency,
page cache, another tenant arriving. The `thermal-drift.json` fixture is exactly
this, and the analysis refuses it:

```
VERDICT: inconclusive
  the baseline series drifts monotonically across its run order
  (Spearman rho +0.98); the comparison would attribute that drift to the change.
  Interleave the two builds, warm up first, and re-measure
```

**Interleave.** Alternate A, B, A, B, or randomize the order and record it.
Interleaving does not remove drift — it makes drift affect both series equally,
which is what turns it from a bias into noise.

Then check for drift anyway. A rank correlation between sample index and value
costs nothing and catches thermal throttling, a filling page cache, a leaking
buffer, and a neighbour starting a build.

## Warm-up, and how much to discard

Warm-up is real: page faults, lazy symbol binding, JIT compilation, cold
caches, and CPU frequency ramp all make early iterations slower.

The discipline is:

1. **Run warm-up iterations and discard them** — but decide the count *before*
   the measurement, and record it.
2. **Never discard on the basis of the values.** "Discard the first N until the
   numbers look stable" is choosing your data after seeing it, and it will
   produce a stable-looking series from pure noise.
3. **Report the warm-up count** as part of the method. A comparison where the
   two arms had different warm-ups is not a comparison.

If you do not know how much warm-up is needed, measure it: plot value against
index for a long run and find where the drift flattens. That is a separate
experiment, run once, whose *conclusion* becomes the fixed policy.

## How many samples

There is no universal number; there is a procedure:

1. Take a pilot run of the baseline — 30 samples is usually enough to start.
2. Compute the rig's resolution from it.
3. Compare that resolution against the effect size you care about.
4. If the resolution is larger than the effect, **do not** simply take more
   samples first — reduce the noise. Sampling more from a noisy rig improves
   the resolution slowly (roughly with the square root of the count) while
   every source of interference improves it immediately.

Below about 20 samples per arm the analysis tool refuses outright, and below 8
it will not report a p-value at all: the normal approximation behind the rank
test has nothing to stand on there, and a number printed anyway would be
decoration.

## Reducing the noise itself

Ordered by how much they usually buy, most first:

| Action | Effect |
| --- | --- |
| Stop sharing the machine | Usually the dominant term; a shared CI runner cannot be quieted |
| Pin the frequency (disable turbo, set a fixed governor) | Removes the largest systematic drift |
| Pin the process to a core, and isolate that core | Removes scheduler migration and neighbour interference |
| Disable address-space randomization for the measurement | Removes layout-driven variance between runs |
| Pre-fault and pre-allocate | Removes first-touch and allocator noise from the timed region |
| Fix the working set and cache regime | Removes the largest input-driven variance |
| Increase the sample count | Slowest lever; use last |

Two cautions:

- **Every one of these changes the thing measured.** A pinned, isolated,
  turbo-disabled core is not the deployment environment. That is an acceptable
  and normal trade — measure in a controlled environment, then confirm the
  direction in a realistic one — but the report must say which one it was.
- **A virtualized or shared host cannot be fixed by any of them.** The
  interference is outside the guest. On such a host, prefer `exact`-class
  evidence (counts, digests, artifact sizes) and treat timings as indicative.

## Outliers

Outliers are evidence. The order is: **record, explain, then decide.**

- A single enormous sample is usually interference — another process, a page
  fault storm, an interrupt. If you can explain it, say so and exclude it *by
  the stated rule*, not by hand.
- A cluster of slow samples is usually structure, not noise: a second mode
  (cold cache, a periodic GC, a different code path). Trimming it hides the
  most interesting thing in the data.
- Declare the rule up front — "Tukey fences, computed per series" — and apply
  it identically to both arms. A trimming rule applied to one arm is a thumb on
  the scale.

The analysis tool reports the outlier fraction and warns above 10% rather than
trimming anything itself. Removing data is a decision that belongs to the person
who can explain it.

## The self-check that catches the most mistakes

**Measure the baseline against itself and confirm the answer is "no
difference".** Build the same commit twice, or split one baseline run in half,
and run the whole comparison. If that reports an improvement, the pipeline is
broken, and every result it has produced is suspect.

`noise-only.json` is that check in fixture form:

```
VERDICT: no-detectable-difference
  the observed 0.14% difference is within this rig's 2.10% resolution
```

Run it after any change to the harness, the host, or the analysis. It is the
performance equivalent of a negative test, and it is the one step that most
benchmarking setups skip.

## Pitfalls

- **All-A-then-all-B ordering.** Confounds the change with everything else.
- **Discarding warm-up by eye.** Manufactures stability.
- **Different warm-up per arm.** Not a comparison.
- **Adding samples to fix a noisy rig.** Slowest possible lever.
- **Trimming outliers without a declared rule.** Especially per-arm.
- **Reporting the minimum as "the real time".** The minimum is one order
  statistic with no dispersion; it is also the most sensitive to sample count.
- **Quieting the machine and forgetting to say so.** The number is real and the
  environment is not the deployment one.
- **Never running the A-against-A check.** The cheapest validation available.

## BCIR notes

- Local hosts in this repository are bounded to two workers, with heavy gates
  serialized and never run concurrently. That is a measurement decision as much
  as a resource one: concurrent gates make every timing on the host meaningless.
- A shared or virtualized runner cannot produce a frozen calibration table here,
  and the calibration code refuses to accept one. A refusal predicate that costs
  something is what makes the accepted certificates worth anything.

## See also

- [`04-significance-and-effect-size.md`](04-significance-and-effect-size.md) — the tests applied to these samples
- [`05-reporting-and-claim-discipline.md`](05-reporting-and-claim-discipline.md) — writing it down
- [`examples/README.md`](examples/README.md) — the fixtures quoted above
