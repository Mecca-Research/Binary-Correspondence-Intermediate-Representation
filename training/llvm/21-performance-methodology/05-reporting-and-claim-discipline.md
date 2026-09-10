# Reporting and Claim Discipline

## The claim template

A performance claim that survives review contains six things. Missing any one
of them makes the claim unreproducible, and missing the last two makes it
unfalsifiable:

1. **What changed** — the commit or slice, not a description of the idea.
2. **What was measured** — workload, input identity, batch factor, metric class.
3. **Where** — host, CPU, kernel, governor, virtualization, compiler and flags.
4. **How** — sample count, warm-up policy, run order, trimming rule.
5. **The result with its interval** — median ratio and confidence interval, not
   a bare percentage.
6. **What the measurement could not have seen** — the rig's resolution.

```markdown
### matmul-256-tiled: 11.6% faster (ratio class)

Change:      slice@1234567 vs main@abcdef0
Workload:    matmul-256-tiled, input sha256:9f2c..., 1 op per timed region
Host:        <cpu>, <kernel>, governor=performance, turbo off, core-pinned,
             bare metal (not virtualized)
Build:       clang 18.1.3, -O2 -march=x86-64-v3, no PGO
Method:      40 samples per arm, interleaved A/B, 5 discarded warm-up
             iterations per arm, Tukey-fence outlier reporting (none removed)
Result:      median ratio 0.8839, 95% CI [0.8795, 0.8868], p < 0.0001
             (Mann-Whitney U), Cliff's delta +1.000 (large)
Resolution:  1.08% -- differences below this are not distinguishable on this rig
Reproduce:   python3 training/llvm/tools/analyze-benchmark-samples.py samples.json
```

The `Resolution` line is the one people leave out and the one a sceptical
reviewer looks for first. It converts "we found 11.6%" into "we found 11.6% with
a measurement capable of resolving 1.1%", which is a different and much stronger
statement.

## Measured, modelled, and estimated

Three words, three different meanings, and conflating them is the most common
integrity failure in performance reporting:

| Word | Means | Requires |
| --- | --- | --- |
| **measured** | it was executed and timed on real hardware | host provenance, sample count, interval |
| **modelled** | a cost model produced it | the model's name, version, and inputs |
| **estimated** | a human reasoned about it | the reasoning, stated |

A modelled number may inform a decision, and it may not be reported as a
measurement. When a table mixes them, label every row. When a number's origin is
unclear, it is not measured.

## What a skip is, and what it is not

A benchmark that did not run because a tool was missing is a **skip**. It is not
a pass, and it is not a zero.

```
[skip] hardware counters unavailable on this host (ENOENT from perf_event_open)
```

That line is a result. It says what was not measured and why, and a reader can
act on it. The failure mode it prevents is a green report whose green includes
everything that silently did not run — the shape this repository's own security
work found repeatedly, where an exclusion turns "the artifact is broken" into
"this does not run here", and in a green run those read identically.

Count the skips, report the count, and mark any aggregate score that includes
them as reduced-confidence.

## Retract cleanly

A performance claim that turns out to be wrong should be withdrawn in the same
place it was made, with the reason:

> **Retracted.** The 6% improvement reported in <ref> was measured with the
> baseline run entirely before the candidate; re-measuring interleaved gives
> `inconclusive` (baseline Spearman rho +0.98, thermal drift). No difference
> has been established.

This costs less than it feels like it does, and the alternative — leaving a
wrong number in the record — is what makes an entire body of performance work
untrustworthy. A retraction is evidence that the process works.

## Language that does not survive review

| Do not write | Write |
| --- | --- |
| "up to 2x faster" | "median ratio 0.51, CI [0.49, 0.53] on workload X" |
| "significantly faster" | "11.6% faster, CI [11.3%, 12.1%], p < 0.0001" |
| "no regression" | "no regression detected; this rig resolves 2.1%" |
| "should be faster" | "modelled as 8% faster; not measured" |
| "faster on our benchmarks" | name the benchmarks, or the claim is unfalsifiable |
| "optimal" | "best measured", unless a lower bound is proved |

"Up to" is the worst of these: it reports the maximum of a set of measurements
as if it characterized them, and its true content is "at least one measurement
was this good".

## Optimality claims need a bound

"Fastest we measured" and "optimal" are different claims. The second requires a
lower bound — a proof that nothing can do better under stated assumptions.
Without one, the honest ceiling is "best measured".

This repository formalizes that as a ladder: exact optimum, bounded gap, best
measured, heuristic-with-no-claim. Everything it currently emits sits at the
last rung until the lower-bound work lands, and it says so. A claim that names
its own rung is one a reader can check.

## The reviewer's checklist

Reading someone else's performance claim, ask in this order:

1. Could this measurement have detected the effect it reports? (resolution)
2. Were the two arms measured under the same conditions? (order, warm-up, host)
3. Is the interval given, and does it exclude no-change?
4. Is the magnitude material, against a threshold declared beforehand?
5. Is anything modelled being presented as measured?
6. What did not run, and is that counted?
7. Is there a neutral workload, and did it stay flat?

Question 1 eliminates more claims than the other six together.

## Pitfalls

- **A percentage with no interval.** Not yet a claim.
- **A ratio with no baseline identity.** Unreproducible.
- **A "no regression" with no resolution.** Compatible with a large regression.
- **Mixing measured and modelled rows unlabelled.** The reader cannot tell.
- **Counting a skip as a pass.** Green means nothing afterwards.
- **Quoting a number whose provenance you cannot state.** It is not measured.
- **"Optimal" without a lower bound.** Say "best measured".

## BCIR notes

- Claims in this repository carry machine-checked markers, so a summary that
  stops being true fails CI. When a claim's predicate changes, the claim changes
  with it, in the same commit.
- Counts belong in generated status output, never in hand-written prose — the
  mechanism that prevents a number from being right on the day it was written
  and wrong every day after.
- "A certificate is only worth what its refusal conditions cost." A performance
  report that has never produced an `inconclusive` has not demonstrated that it
  can.

## See also

- [`04-significance-and-effect-size.md`](04-significance-and-effect-size.md) — where the numbers come from
- [`06-hardware-counter-harness.md`](06-hardware-counter-harness.md) — refusing honestly, with a worked example
- [`../../docs/PERFORMANCE_AUDIT.md`](../../../docs/PERFORMANCE_AUDIT.md) — this repository's own evidence contract
