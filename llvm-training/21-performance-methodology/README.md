# Performance Methodology: From Timings to a Defensible Claim

## Key takeaways

- **Establish the rig's resolution before reporting an effect.** Compare the
  baseline against itself; the width of that interval is the smallest
  difference the measurement can distinguish from nothing. Any claimed effect
  below it is a claim about noise.
- **Timing distributions are not normal.** They are bounded below by the
  fastest possible execution, unbounded above by interference, and often
  multimodal. Means and standard deviations describe a shape the data does not
  have; use ranks, medians, and the bootstrap.
- **Significance is not magnitude, and magnitude is not importance.** Enough
  samples make any difference significant. Declare the threshold that matters
  *before* looking at the result.
- **A drifting series is not a measurement of the build.** It is a measurement
  of the machine changing underneath it. Interleave, warm up, and re-measure.
- **Classify every row before you use it.** `exact` rows gate; `ratio` rows gate
  with a wide band; `wall` rows are indicative and never gate.
- **An unreadable counter is `null`, never `0`.** Privilege is not capability.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Choosing what to measure, and why a microbenchmark may answer nothing | [`01-workload-selection.md`](01-workload-selection.md) |
| Getting numbers out of a program: timers, quanta, profiles, sampling bias | [`02-measurement-and-profile-collection.md`](02-measurement-and-profile-collection.md) |
| How many trials, in what order, on what machine | [`03-repeated-trials-and-noise-control.md`](03-repeated-trials-and-noise-control.md) |
| Which test, which effect size, and what "no difference" means | [`04-significance-and-effect-size.md`](04-significance-and-effect-size.md) |
| Writing the claim so it survives review | [`05-reporting-and-claim-discipline.md`](05-reporting-and-claim-discipline.md) |
| Hardware counters, and refusing honestly when there is no PMU | [`06-hardware-counter-harness.md`](06-hardware-counter-harness.md) |

## The tools

Both are dependency-free (standard library only) and deterministic.

```bash
# Grade a pair of sample series into a verdict -- or into a refusal.
python3 llvm-training/tools/analyze-benchmark-samples.py \
  llvm-training/21-performance-methodology/examples/clean-improvement.json

# Machine-readable, and usable as a CI gate on a gateable metric class.
python3 llvm-training/tools/analyze-benchmark-samples.py samples.json --format json
python3 llvm-training/tools/analyze-benchmark-samples.py samples.json --gate

# Ask the host whether hardware counters exist. Record the answer either way.
python3 llvm-training/tools/probe-hardware-counters.py

# Self-test: statistical kernels plus every fixture's expected verdict.
python3 llvm-training/tools/verify-benchmark-analysis.py
```

The analysis tool's verdicts are:

| Verdict | Meaning |
| --- | --- |
| `improvement` / `regression` | a real difference, larger than the declared threshold |
| `detectable-but-immaterial` | real, and smaller than the threshold that was declared up front |
| `no-detectable-difference` | below the rig's resolution, or `p >= alpha` |
| `inconclusive` | the data cannot answer: too few samples, a drifting series, or an interval containing 1.0 |

Three of the five are refusals. That ratio is the point of the chapter.

## Examples

[`examples/`](examples) holds six sample files, one per *shape of wrong answer*
this methodology exists to catch. Each is graded with a pinned expected verdict
by [`../tools/verify-benchmark-analysis.py`](../tools/verify-benchmark-analysis.py):

| Fixture | The mistake it represents | Verdict |
| --- | --- | --- |
| `clean-improvement.json` | (none — a real, resolvable win) | `improvement` |
| `noise-only.json` | reporting a difference between a build and itself | `no-detectable-difference` |
| `below-resolution.json` | a small effect on a rig that cannot resolve it | `no-detectable-difference` |
| `thermal-drift.json` | measuring the baseline first and heating the machine | `inconclusive` |
| `significant-but-trivial.json` | a huge sample count making a trivial effect significant | `detectable-but-immaterial` |
| `wall-clock-indicative.json` | gating on an absolute wall-clock row | refuses to gate |

The sample values are **synthetic** and generated deterministically. They exist
to exercise the analysis, not to describe any hardware. No claim in this chapter
is a measurement of a real machine, and the fixtures say so in their own
`environment` block.

## Verification boundary

- The statistics are checked: the analysis tool's kernels are verified against
  hand-computable values, and each fixture's verdict is pinned.
- The **methodology** — workload selection, warm-up policy, host preparation —
  is judgement, informed by the mechanisms this chapter documents. It is not
  machine-checkable and is not presented as if it were.
- Nothing here measures this repository's own performance. The BCIR performance
  evidence lives in [`../../docs/PERFORMANCE_AUDIT.md`](../../docs/PERFORMANCE_AUDIT.md)
  and the GEM+ baseline harness; this chapter supplies the method those use.

## What this chapter is not

It is not a benchmarking framework, and it does not replace one. It supplies
the analysis and the discipline that a framework's raw output still needs — the
half where most reported speedups actually go wrong.

## See also

- [`../15-binary-analysis/README.md`](../15-binary-analysis/README.md) — evidence schemas and provenance
- [`../07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) — profile-guided and post-link optimization
- [`../07-optimization/10-lto-bolt-experiment-matrix.md`](../07-optimization/10-lto-bolt-experiment-matrix.md) — the deterministic artifact matrix, which compares builds rather than runtimes
- [`../../docs/BCIR_TARGET_ACCESS.md`](../../docs/BCIR_TARGET_ACCESS.md) — what this repository's hosts can and cannot measure
