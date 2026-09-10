# Benchmark sample fixtures

Six sample files, one per *shape of wrong answer* that
[`../../tools/analyze-benchmark-samples.py`](../../tools/analyze-benchmark-samples.py)
exists to refuse. Each fixture's expected verdict is pinned by
[`../../tools/verify-benchmark-analysis.py`](../../tools/verify-benchmark-analysis.py).

```bash
python3 llvm-training/tools/analyze-benchmark-samples.py \
  llvm-training/21-performance-methodology/examples/clean-improvement.json

python3 llvm-training/tools/verify-benchmark-analysis.py   # grade them all
```

## The fixtures

| File | The mistake it represents | Expected verdict |
| --- | --- | --- |
| `clean-improvement.json` | none — a real, comfortably resolvable win | `improvement` |
| `noise-only.json` | reporting a difference between a build and itself | `no-detectable-difference` |
| `below-resolution.json` | a 1.5% effect claimed on a rig that resolves 6.5% | `no-detectable-difference` |
| `thermal-drift.json` | running all of the baseline first and heating the machine | `inconclusive` |
| `significant-but-trivial.json` | 400 samples per arm making a 0.4% effect significant | `detectable-but-immaterial` |
| `wall-clock-indicative.json` | using an absolute wall-clock row as a gate | refuses to gate (exit 3) |

`noise-only.json` doubles as the harness self-check described in
[`../03-repeated-trials-and-noise-control.md`](../03-repeated-trials-and-noise-control.md):
both series are drawn from the same distribution, so any verdict other than
"no detectable difference" means the pipeline is broken.

## Provenance — read before quoting any number here

**These values are synthetic.** They were produced by a seeded generator so the
fixtures are stable and the analysis is a regression test. They are not
measurements, they do not describe any machine, and no claim in this chapter is
derived from them. Each file says so in its own `environment` block
(`"host": "synthetic"`).

The distinction is the subject of
[`../05-reporting-and-claim-discipline.md`](../05-reporting-and-claim-discipline.md):
a number's provenance travels with it, or it is not evidence.

## Schema

`llvm-training/benchmark-samples/v1`:

```json
{
  "schema": "llvm-training/benchmark-samples/v1",
  "workload": "matmul-256-tiled",
  "unit": "ms",
  "metric_class": "ratio",
  "lower_is_better": true,
  "note": "free-form; what this file is for",
  "environment": {
    "host": "synthetic",
    "cpu": "...",
    "governor": "...",
    "generated": "..."
  },
  "series": {
    "baseline":  { "build": "main@abcdef0",  "samples": [100.1, 99.8, ...] },
    "candidate": { "build": "slice@1234567", "samples": [88.2, 88.0, ...] }
  }
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `schema` | yes | must match exactly; the loader refuses anything else |
| `workload` | no | free text; appears in the report header |
| `unit` | no | free text (`ms`, `ns`, `bytes`, `instructions`) |
| `metric_class` | no | `exact`, `ratio`, or `wall`. Defaults to `wall`, the most conservative choice — `wall` rows are indicative and `--gate` refuses them |
| `lower_is_better` | no | defaults to `true` |
| `environment` | no | free-form, and the place provenance belongs |
| `series.baseline.samples` | yes | numeric, non-negative, **in original run order** |
| `series.candidate.samples` | yes | same |

Two rules the loader enforces and two it cannot:

- Enforced: samples must be numeric and non-negative, and both series must be
  non-empty. A malformed file is an error (exit 2), never a silent pass.
- Not enforced, and load-bearing anyway: **run order must be preserved**
  (sorting destroys the drift signal that produces the `inconclusive` verdict),
  and **samples must not be pre-trimmed** (the outlier fraction is reported, and
  removing data is a decision that belongs to whoever can explain it).

## Metric classes

| Class | Band | May gate? |
| --- | --- | --- |
| `exact` | deterministic, ~2% | yes, anywhere |
| `ratio` | timed ratio, wide (~25%) | yes, with the band |
| `wall` | absolute time | **no** — indicative only |

`--gate` exits 3 on a `wall`-class row rather than letting a caller opt in.
Choosing the class is part of designing the measurement, not part of writing the
report.

## Regenerating

The fixtures are checked in and stable; the pinned verdicts and the seeded
bootstrap make them a regression test on the analysis itself. Editing a sample
value changes a verdict, which is the intended behaviour — if you add a fixture,
add its expected verdict to `EXPECTED` in
[`../../tools/verify-benchmark-analysis.py`](../../tools/verify-benchmark-analysis.py),
which fails on any fixture nobody grades.
