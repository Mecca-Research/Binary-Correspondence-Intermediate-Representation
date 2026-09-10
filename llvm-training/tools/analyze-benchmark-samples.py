#!/usr/bin/env python3
"""Turn repeated benchmark timings into a claim, or into an honest refusal.

Most reported speedups are not wrong about the arithmetic. They are wrong about
whether the measurement could have detected the difference they report. This
tool answers that question first and the speedup question second:

  1. What is this rig's resolution? -- the noise floor, measured by comparing
     the baseline against ITSELF. A rig that cannot distinguish a build from
     itself by less than 4% cannot report a 3% win, whatever the means say.
  2. Are the samples fit to compare at all? -- drift, dispersion, and outliers
     are reported before any test result, because a significant p-value over a
     drifting series is a significant result about the drift.
  3. Is the difference real? -- Mann-Whitney U (no normality assumption),
     Cliff's delta for magnitude, a Hodges-Lehmann shift estimate, and a
     deterministic bootstrap confidence interval on the ratio of medians.

Verdicts are one of:

  improvement / regression        a real, material difference
  detectable-but-immaterial       real, and smaller than the declared threshold
  no-detectable-difference        below the rig's resolution, or p >= alpha
  inconclusive                    the data cannot answer -- too few samples, a
                                  drifting series, or an interval that contains 1

The last three are real answers, and this tool reports them as often as the data
earns them.

Dependency-free (standard library only) and deterministic: the bootstrap is
seeded, so the same input file always produces the same interval.

    python3 llvm-training/tools/analyze-benchmark-samples.py samples.json
    python3 llvm-training/tools/analyze-benchmark-samples.py samples.json --format json
    python3 llvm-training/tools/analyze-benchmark-samples.py samples.json --gate

Input schema: see llvm-training/21-performance-methodology/examples/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

SCHEMA = "llvm-training/benchmark-samples/v1"

# Cliff's delta magnitude thresholds (Romano et al., the conventional set).
NEGLIGIBLE = 0.147
SMALL = 0.33
MEDIUM = 0.474

# A tool that reports a number it cannot support is the failure mode this whole
# file exists to prevent, so the refusal thresholds are named constants rather
# than buried literals.
DEFAULT_MIN_SAMPLES = 20
# The smallest difference worth calling a win. Declared up front, because a
# threshold chosen after seeing the result is not a threshold.
DEFAULT_MIN_EFFECT = 0.01
MANN_WHITNEY_MIN_PER_GROUP = 8
DRIFT_MIN_SAMPLES = 20
HIGH_DISPERSION_CV = 0.10
DRIFT_ABS_RHO = 0.5
MAX_PAIRS_FOR_EXACT_SHIFT = 400_000


# --------------------------------------------------------------------------
# Small statistics kit (no numpy, no scipy)
# --------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged -- required for a correct tie correction."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def tie_correction(values: Sequence[float]) -> float:
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sum(t**3 - t for t in counts.values() if t > 1)


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float | None]:
    """Two-sided Mann-Whitney U.

    Returns (U for `a`, p). p is None when the groups are too small for the
    normal approximation to mean anything -- reporting a p-value there would be
    a number with no support behind it.
    """
    n1, n2 = len(a), len(b)
    combined = list(a) + list(b)
    ranks = average_ranks(combined)
    rank_sum_a = sum(ranks[:n1])
    u_a = rank_sum_a - n1 * (n1 + 1) / 2.0

    if n1 < MANN_WHITNEY_MIN_PER_GROUP or n2 < MANN_WHITNEY_MIN_PER_GROUP:
        return u_a, None

    total = n1 + n2
    mean_u = n1 * n2 / 2.0
    correction = tie_correction(combined)
    variance = (n1 * n2 / 12.0) * ((total + 1) - correction / float(total * (total - 1)))
    if variance <= 0.0:
        # Every observation identical: no evidence of a difference, and the
        # normal approximation is undefined rather than significant.
        return u_a, 1.0

    # Continuity correction, applied toward the mean.
    diff = abs(u_a - mean_u)
    z = max(diff - 0.5, 0.0) / math.sqrt(variance)
    return u_a, 2.0 * (1.0 - normal_cdf(z))


def cliffs_delta(a: Sequence[float], b: Sequence[float], u_a: float) -> float:
    """Ordinal effect size in [-1, 1]; +1 means every `a` exceeds every `b`."""
    return (2.0 * u_a) / (len(a) * len(b)) - 1.0


def delta_magnitude(delta: float) -> str:
    size = abs(delta)
    if size < NEGLIGIBLE:
        return "negligible"
    if size < SMALL:
        return "small"
    if size < MEDIUM:
        return "medium"
    return "large"


def hodges_lehmann_shift(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Median of all pairwise (b - a): a robust estimate of the location shift."""
    if len(a) * len(b) > MAX_PAIRS_FOR_EXACT_SHIFT:
        return None
    return statistics.median(y - x for x in a for y in b)


def spearman_rho(values: Sequence[float]) -> float:
    """Rank correlation of value against sample order: the drift signal."""
    n = len(values)
    if n < 3:
        return 0.0
    value_ranks = average_ranks(values)
    index_ranks = [float(i + 1) for i in range(n)]
    mean_v = statistics.fmean(value_ranks)
    mean_i = statistics.fmean(index_ranks)
    num = sum((v - mean_v) * (i - mean_i) for v, i in zip(value_ranks, index_ranks))
    den_v = math.sqrt(sum((v - mean_v) ** 2 for v in value_ranks))
    den_i = math.sqrt(sum((i - mean_i) ** 2 for i in index_ranks))
    if den_v == 0.0 or den_i == 0.0:
        return 0.0
    return num / (den_v * den_i)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ratio_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    iterations: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for median(b) / median(a). Seeded, so stable."""
    rng = random.Random(seed)
    n1, n2 = len(a), len(b)
    ratios: list[float] = []
    for _ in range(iterations):
        resample_a = [a[rng.randrange(n1)] for _ in range(n1)]
        resample_b = [b[rng.randrange(n2)] for _ in range(n2)]
        median_a = statistics.median(resample_a)
        if median_a == 0:
            continue
        ratios.append(statistics.median(resample_b) / median_a)
    if not ratios:
        return (float("nan"), float("nan"))
    ratios.sort()
    return (
        percentile(ratios, alpha / 2.0),
        percentile(ratios, 1.0 - alpha / 2.0),
    )


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


@dataclass
class SeriesReport:
    name: str
    n: int
    minimum: float
    median: float
    mean: float
    stdev: float
    cv: float
    iqr: float
    mad: float
    outlier_fraction: float
    drift_rho: float
    warnings: list[str] = field(default_factory=list)


def describe(name: str, samples: Sequence[float]) -> SeriesReport:
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if n > 1 else 0.0
    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    iqr = q3 - q1
    mad = statistics.median([abs(x - median) for x in ordered])
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = sum(1 for x in ordered if x < low or x > high)
    # Drift is computed on the ORIGINAL order; sorting first would erase it.
    rho = spearman_rho(list(samples)) if n >= DRIFT_MIN_SAMPLES else 0.0

    report = SeriesReport(
        name=name,
        n=n,
        minimum=ordered[0],
        median=median,
        mean=mean,
        stdev=stdev,
        cv=(stdev / mean) if mean else 0.0,
        iqr=iqr,
        mad=mad,
        outlier_fraction=outliers / n,
        drift_rho=rho,
    )

    if report.cv > HIGH_DISPERSION_CV:
        report.warnings.append(
            f"high dispersion: CV {report.cv:.1%} exceeds {HIGH_DISPERSION_CV:.0%}; "
            "suspect a noisy host, frequency scaling, or a shared runner"
        )
    if n >= DRIFT_MIN_SAMPLES and abs(rho) >= DRIFT_ABS_RHO:
        direction = "upward" if rho > 0 else "downward"
        report.warnings.append(
            f"{direction} drift across the run order (Spearman rho {rho:+.2f}); "
            "suspect thermal throttling or a missing warm-up, and re-run "
            "interleaved rather than in blocks"
        )
    if report.outlier_fraction > 0.10:
        report.warnings.append(
            f"{report.outlier_fraction:.0%} of samples are Tukey outliers; "
            "investigate before trimming -- an outlier is evidence until "
            "something explains it"
        )
    return report


def noise_floor(samples: Sequence[float], *, iterations: int, alpha: float, seed: int) -> float:
    """This rig's resolution: the baseline compared against itself.

    Deterministically split the baseline into two interleaved halves -- so both
    halves span the whole run and share any drift -- and bootstrap the ratio of
    their medians. The half-width of that interval is the smallest difference
    this measurement can distinguish from nothing. Any claimed effect smaller
    than it is a claim about the noise.
    """
    if len(samples) < 4:
        return float("nan")
    first = list(samples[0::2])
    second = list(samples[1::2])
    low, high = bootstrap_ratio_ci(first, second, iterations=iterations, alpha=alpha, seed=seed)
    if math.isnan(low) or math.isnan(high):
        return float("nan")
    return max(abs(1.0 - low), abs(high - 1.0))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


@dataclass
class Comparison:
    workload: str
    unit: str
    metric_class: str
    lower_is_better: bool
    baseline: SeriesReport
    candidate: SeriesReport
    ratio_median: float
    ratio_ci: tuple[float, float]
    shift: float | None
    p_value: float | None
    cliffs_delta: float
    effect_magnitude: str
    resolution: float
    verdict: str
    reason: str
    gateable: bool
    warnings: list[str] = field(default_factory=list)


def analyze(
    document: dict,
    *,
    alpha: float,
    iterations: int,
    seed: int,
    min_samples: int,
    min_effect: float = DEFAULT_MIN_EFFECT,
) -> Comparison:
    series = document["series"]
    baseline_samples = [float(x) for x in series["baseline"]["samples"]]
    candidate_samples = [float(x) for x in series["candidate"]["samples"]]

    baseline = describe("baseline", baseline_samples)
    candidate = describe("candidate", candidate_samples)

    unit = document.get("unit", "unit")
    metric_class = document.get("metric_class", "wall")
    lower_is_better = bool(document.get("lower_is_better", True))

    ratio = candidate.median / baseline.median if baseline.median else float("nan")
    ci = bootstrap_ratio_ci(
        baseline_samples,
        candidate_samples,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )
    resolution = noise_floor(baseline_samples, iterations=iterations, alpha=alpha, seed=seed + 1)
    u_a, p_value = mann_whitney(baseline_samples, candidate_samples)
    delta = cliffs_delta(baseline_samples, candidate_samples, u_a)
    shift = hodges_lehmann_shift(baseline_samples, candidate_samples)

    warnings = list(baseline.warnings) + list(candidate.warnings)

    # A wall-clock row is indicative. It may inform, and it may not gate.
    gateable = metric_class in {"exact", "ratio"}
    if metric_class == "wall":
        warnings.append(
            "metric_class is 'wall': absolute timings are INDICATIVE and must "
            "not gate a change. Classify a gating row as 'exact' (deterministic) "
            "or 'ratio' (a timed ratio with a wide band)."
        )

    effect = abs(1.0 - ratio) if not math.isnan(ratio) else float("nan")
    verdict, reason = decide(
        baseline=baseline,
        candidate=candidate,
        min_samples=min_samples,
        p_value=p_value,
        alpha=alpha,
        delta=delta,
        ratio=ratio,
        ci=ci,
        resolution=resolution,
        effect=effect,
        lower_is_better=lower_is_better,
        min_effect=min_effect,
    )

    return Comparison(
        workload=document.get("workload", "(unnamed workload)"),
        unit=unit,
        metric_class=metric_class,
        lower_is_better=lower_is_better,
        baseline=baseline,
        candidate=candidate,
        ratio_median=ratio,
        ratio_ci=ci,
        shift=shift,
        p_value=p_value,
        cliffs_delta=delta,
        effect_magnitude=delta_magnitude(delta),
        resolution=resolution,
        verdict=verdict,
        reason=reason,
        gateable=gateable,
        warnings=warnings,
    )


def decide(
    *,
    baseline: SeriesReport,
    candidate: SeriesReport,
    min_samples: int,
    p_value: float | None,
    alpha: float,
    delta: float,
    ratio: float,
    ci: tuple[float, float],
    resolution: float,
    effect: float,
    lower_is_better: bool,
    min_effect: float,
) -> tuple[str, str]:
    """The decision ladder. Every rung can answer 'inconclusive'."""
    if baseline.n < min_samples or candidate.n < min_samples:
        return (
            "inconclusive",
            f"too few samples: {baseline.n} baseline / {candidate.n} candidate, "
            f"minimum {min_samples} each",
        )

    # A drifting series is not a measurement of the build; it is a measurement
    # of the machine changing underneath the build. Comparing against it
    # attributes the drift to the change.
    for series in (baseline, candidate):
        if series.n >= DRIFT_MIN_SAMPLES and abs(series.drift_rho) >= DRIFT_ABS_RHO:
            return (
                "inconclusive",
                f"the {series.name} series drifts monotonically across its run "
                f"order (Spearman rho {series.drift_rho:+.2f}); the comparison "
                "would attribute that drift to the change. Interleave the two "
                "builds, warm up first, and re-measure",
            )

    if p_value is None:
        return (
            "inconclusive",
            "groups too small for the rank test to be meaningful",
        )

    if math.isnan(ratio):
        return ("inconclusive", "baseline median is zero; a ratio is undefined")

    # The measurement's own resolution comes before its result.
    if not math.isnan(resolution) and effect <= resolution:
        return (
            "no-detectable-difference",
            f"the observed {effect:.2%} difference is within this rig's "
            f"{resolution:.2%} resolution; the measurement cannot distinguish it "
            "from noise",
        )

    if p_value >= alpha:
        return (
            "no-detectable-difference",
            f"p = {p_value:.4f} at alpha = {alpha}; no difference detected. "
            f"With {baseline.n}/{candidate.n} samples this run could only have "
            f"resolved differences above {resolution:.2%}",
        )

    if abs(delta) < NEGLIGIBLE:
        return (
            "inconclusive",
            f"p = {p_value:.4f} is significant but Cliff's delta {delta:+.3f} is "
            "negligible: with enough samples a trivial difference becomes "
            "significant, and significance is not magnitude",
        )

    low, high = ci
    if not math.isnan(low) and low <= 1.0 <= high:
        return (
            "inconclusive",
            f"p = {p_value:.4f} but the {int((1 - alpha) * 100)}% CI on the median "
            f"ratio [{low:.4f}, {high:.4f}] contains 1.0; the point estimate is "
            "not supported by the interval",
        )

    if effect < min_effect:
        return (
            "detectable-but-immaterial",
            f"the {effect:.2%} difference is statistically detectable "
            f"(p = {p_value:.4f}) and smaller than the {min_effect:.2%} threshold "
            "declared for this workload. Enough samples make any difference "
            "significant; significance is not importance",
        )

    improved = (ratio < 1.0) if lower_is_better else (ratio > 1.0)
    magnitude = delta_magnitude(delta)
    direction = "faster" if lower_is_better else "higher"
    change = abs(1.0 - ratio)
    if improved:
        return (
            "improvement",
            f"{change:.2%} {direction} (median ratio {ratio:.4f}, "
            f"CI [{low:.4f}, {high:.4f}], p = {p_value:.4f}, "
            f"Cliff's delta {delta:+.3f} = {magnitude})",
        )
    return (
        "regression",
        f"{change:.2%} worse (median ratio {ratio:.4f}, "
        f"CI [{low:.4f}, {high:.4f}], p = {p_value:.4f}, "
        f"Cliff's delta {delta:+.3f} = {magnitude})",
    )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"expected schema '{SCHEMA}', found '{schema}'")
    series = document.get("series")
    if not isinstance(series, dict):
        raise ValueError("'series' must be an object")
    for name in ("baseline", "candidate"):
        if name not in series:
            raise ValueError(f"missing series '{name}'")
        samples = series[name].get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"series '{name}' has no samples")
        for value in samples:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"series '{name}' contains a non-numeric sample")
            if value < 0:
                raise ValueError(f"series '{name}' contains a negative sample")
    return document


def render_text(result: Comparison) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"workload      : {result.workload}")
    add(
        f"metric class  : {result.metric_class} "
        f"({'gateable' if result.gateable else 'INDICATIVE -- must not gate'})"
    )
    add(
        f"unit          : {result.unit} "
        f"({'lower is better' if result.lower_is_better else 'higher is better'})"
    )
    add("")
    add(f"{'series':<10} {'n':>5} {'min':>14} {'median':>14} {'mean':>14} {'CV':>8} {'drift':>8}")
    for series in (result.baseline, result.candidate):
        add(
            f"{series.name:<10} {series.n:>5} {series.minimum:>14.4f} "
            f"{series.median:>14.4f} {series.mean:>14.4f} "
            f"{series.cv:>7.2%} {series.drift_rho:>+8.2f}"
        )
    add("")
    add(
        f"rig resolution: {result.resolution:.2%}  "
        "(smallest difference this measurement can distinguish from itself)"
    )
    add(
        f"median ratio  : {result.ratio_median:.4f}  "
        f"CI [{result.ratio_ci[0]:.4f}, {result.ratio_ci[1]:.4f}]"
    )
    if result.shift is not None:
        add(f"HL shift      : {result.shift:+.4f} {result.unit}")
    add(
        "Mann-Whitney p: "
        + (
            "(not reportable at this sample size)"
            if result.p_value is None
            else f"{result.p_value:.4f}"
        )
    )
    add(f"Cliff's delta : {result.cliffs_delta:+.3f} ({result.effect_magnitude})")
    add("")
    if result.warnings:
        add("warnings:")
        for warning in result.warnings:
            add(f"  ! {warning}")
        add("")
    add(f"VERDICT: {result.verdict}")
    add(f"  {result.reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="benchmark sample JSON file")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260101,
        help="bootstrap seed; fixed so reports are reproducible",
    )
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--min-effect",
        type=float,
        default=DEFAULT_MIN_EFFECT,
        help="smallest relative difference worth reporting as a win or a "
        "regression (default 1%%). Declare it before you look",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit nonzero on a regression, or on any verdict from a metric "
        "class that must not gate",
    )
    args = parser.parse_args(argv)

    try:
        document = load(args.samples)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark analysis: FAILED ({exc})", file=sys.stderr)
        return 2

    result = analyze(
        document,
        alpha=args.alpha,
        iterations=args.bootstrap,
        seed=args.seed,
        min_samples=args.min_samples,
        min_effect=args.min_effect,
    )

    if args.format == "json":
        payload = asdict(result)
        payload["ratio_ci"] = list(result.ratio_ci)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(result))

    if args.gate:
        if not result.gateable:
            print(
                f"\nbenchmark analysis: REFUSED to gate on a '{result.metric_class}' row",
                file=sys.stderr,
            )
            return 3
        if result.verdict == "regression":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
