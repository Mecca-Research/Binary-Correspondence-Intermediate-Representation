"""T3 — derived/aggregate telemetry metrics + a plan-cost SENSITIVITY ranking.

This is the SPICE ``.MEASURE`` / ``.SENS`` analogy from
``docs/kernel/TELEMETRY_PIPELINE_RESEARCH.md`` §3/§6, realized as two pure-Python,
deterministic, side-effect-free capabilities on top of the telemetry stream
(``bcir/telemetry.py``'s :class:`DataDNA` records):

  * **Derived metrics (``.MEASURE``)** — edge-computed scalar figures-of-merit
    over a *batch* of telemetry records (per-field max / min / avg / RMS /
    count) plus a SPICE ``.MEASURE TRIG…TARG`` span: the index from when a
    signal first crosses a TRIGGER threshold to when it later crosses a TARGET
    threshold. Instead of storing every raw waveform sample, the edge stores the
    figure-of-merit.
  * **Sensitivity ranking (``.SENS``)** — score WHICH telemetry signals most
    affect the optimized PLAN COST, by finite-differencing the *existing*
    optimizer (:func:`bcir.kbcir.realize.optimize`) over a perturbed
    :class:`~bcir.kbcir.cost.Theta`. The high-|Δscore| signals are where a
    fixed sampling budget should go (:func:`sampling_budget`).

**Two-truth quarantine (the governing invariant).** Telemetry is a graded
L2/L3 signal. Everything here lives on the COST/optimization side: it informs
plan cost via :class:`Theta` (the runtime state the optimizer's scalarization
weights read), and it NEVER becomes, alters, or even touches an R-law legality
verdict. There is deliberately no ``verify`` / Diagnostic surface in this
module. The sensitivity perturbs a Theta *pressure* (the channel through which
a telemetry signal reaches the cost) and re-runs the optimizer; the legality
path (``bcir/verify``) is untouched, deterministic, and never reads telemetry.
The same module still ``verify()``s clean before and after a sensitivity run.

**Sanitization.** Derived metrics are intended to run over RT3-SANITIZED
records (``bcir.telemetry.sanitize_events`` — the documented 0..100 / finite /
non-negative ingest gate). The functions here do NOT re-sanitize: they compute
honestly over whatever batch they are handed, so the caller is responsible for
passing sanitized records (mirroring how ``calibrate.py`` sanitizes first).
:func:`derive_field_metrics`/:func:`derive_all` validate the *field name* and
handle an empty batch without crashing, but they do not silently drop
out-of-range values — that is the ingest gate's job, kept separate so a metric
over a hostile batch is not silently different from a metric over a clean one.

Determinism: integer/exact where possible; ``avg``/``rms`` round-to-nearest
(documented on each function). No floats leak into the optimizer — the
sensitivity feeds integer Theta pressures only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..signal_registry import SignalRegistry, default_registry
from ..telemetry import _COUNTER_FIELDS, _NORM_FIELDS, DataDNA
from .cost import Theta
from .realize import optimize
from .weights import PERF, Policy

# The DataDNA fields a derived metric can be computed over (the 0..100 normalized
# pressures + the non-negative counters). Excludes the string fields (segment_id,
# provenance) and claim_id (an identifier, not a measured quantity).
METRIC_FIELDS: tuple[str, ...] = _NORM_FIELDS + _COUNTER_FIELDS


# === Derived metrics (.MEASURE) ======================================================


@dataclass(frozen=True)
class FieldMetrics:
    """The ``.MEASURE`` figures-of-merit for one DataDNA field over a batch.

    ``count`` is the number of records. ``maximum``/``minimum`` are exact integers
    (the field is integer). ``avg`` is the integer mean rounded HALF-UP
    (``(sum + count // 2) // count``) — a deterministic, float-free round-to-nearest.
    ``rms`` is ``round(sqrt(mean(x^2)))`` — the only place a float (the square root)
    is unavoidable; it is rounded to the nearest integer and never reaches the
    optimizer. An empty batch yields :func:`empty_metrics` (all ``None``)."""

    field: str
    count: int
    minimum: int | None
    maximum: int | None
    avg: int | None
    rms: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "avg": self.avg,
            "rms": self.rms,
        }


def empty_metrics(field: str) -> FieldMetrics:
    """The well-defined empty result for a batch with no records: count 0, every
    figure-of-merit ``None`` (there is no max/min/avg/rms of nothing). Never raises."""
    return FieldMetrics(field=field, count=0, minimum=None, maximum=None, avg=None, rms=None)


def _validate_field(field: str) -> None:
    if field not in METRIC_FIELDS:
        raise KeyError(f"unknown DataDNA metric field {field!r}; choose from {METRIC_FIELDS}")


def derive_field_metrics(records, field: str) -> FieldMetrics:
    """Compute the ``.MEASURE`` figures-of-merit (max / min / avg / rms / count) for
    one :class:`DataDNA` field over a batch.

    Rounding (documented + deterministic):

      * ``avg`` = integer mean, round HALF-UP: ``(sum_x + n // 2) // n``. Float-free.
      * ``rms`` = ``round(sqrt(sum_x2 / n))``. The square root is the one float; the
        result is rounded to the nearest integer (Python ``round``, banker's on a
        .5 boundary — irrelevant here since the input is a non-negative real). The
        rms is reported for inspection only; it does NOT feed the optimizer.

    An empty batch returns :func:`empty_metrics` (does not raise / divide by zero).
    Validates ``field`` (a typo raises ``KeyError``, not a silent wrong answer).
    Side-effect-free: it only reads the records. Caller passes RT3-sanitized records
    (see the module docstring) — this does not re-validate record values."""
    _validate_field(field)
    vals = [getattr(r, field) for r in records]
    n = len(vals)
    if n == 0:
        return empty_metrics(field)
    sum_x = sum(vals)
    sum_x2 = sum(v * v for v in vals)
    avg = (sum_x + n // 2) // n  # round-half-up integer mean
    rms = round(math.sqrt(sum_x2 / n))  # round-to-nearest of the true RMS
    return FieldMetrics(
        field=field, count=n, minimum=min(vals), maximum=max(vals), avg=avg, rms=rms
    )


def derive_all(records) -> dict[str, FieldMetrics]:
    """Compute :func:`derive_field_metrics` for every metric field (the 0..100
    normalized pressures + the non-negative counters) over the batch — the batch
    ``.MEASURE`` rollup. An empty batch yields the empty metrics for each field.
    Deterministic key order (``METRIC_FIELDS``)."""
    return {f: derive_field_metrics(records, f) for f in METRIC_FIELDS}


def measure_trig_targ(series, *, trig: int, targ: int, rising: bool = True) -> int | None:
    """The SPICE ``.MEASURE TRIG…TARG`` figure-of-merit: the span (number of samples)
    from when the signal FIRST crosses the ``trig`` threshold to when it LATER
    crosses the ``targ`` threshold.

    ``series`` is an ordered iterable of integers (one signal's samples in
    arrival/emit order — e.g. ``[r.thermal for r in batch]``). The returned span is
    the index difference ``i_targ - i_trig``, i.e. how many sample-steps elapsed
    between the two crossings (an *index* span; if the records carry a wall-clock
    order the caller maps indices→elapsed). Returns ``None`` if either crossing never
    happens (no TRIG, or no TARG at/after the TRIG index).

    Crossing semantics (documented):

      * ``rising=True``  (default): a "crossing" of threshold ``t`` is the first
        index whose value is ``>= t``. TRIG fires when the signal rises to ``trig``;
        TARG fires at the first index AT OR AFTER the TRIG index whose value is
        ``>= targ``. (The classic "time from when thermal first hits 60 to when it
        hits 90".)
      * ``rising=False`` (falling): a crossing is the first index whose value is
        ``<= t``. TRIG fires when the signal falls to ``trig``; TARG when it later
        falls to ``targ``.

    The TARG search starts AT the TRIG index (so a sample that satisfies both on the
    same step gives span 0). Deterministic, side-effect-free, integer-only."""
    vals = list(series)

    def _crosses(value: int, threshold: int) -> bool:
        return value >= threshold if rising else value <= threshold

    trig_i: int | None = None
    for i, v in enumerate(vals):
        if _crosses(v, trig):
            trig_i = i
            break
    if trig_i is None:
        return None
    for j in range(trig_i, len(vals)):
        if _crosses(vals[j], targ):
            return j - trig_i
    return None


# === Sensitivity ranking (.SENS) =====================================================

# How a telemetry signal's COST DIMENSION (a T1-registry ``cost_dim`` ∈ ``cost.DIMS``)
# reaches the optimizer: it perturbs a 0..100 pressure FIELD on `Theta` (the runtime
# state the scalarization weights / context coupling read). Only the dims that Theta
# actually exposes as a pressure are perturbable; a signal whose cost_dim is not in
# this map (e.g. `fabric`, `reliability`, `security`) has no Theta channel here and is
# reported with Δscore 0 (honest: the optimizer's Theta-coupling does not move for it).
#
# Mirrors the telemetry→Theta projection in `calibrate.py` (thermal→thermal,
# misses→mem_pressure, …) but keyed by the registry's cost_dim, since the sensitivity
# is about which COST DIM a signal calibrates.
_COST_DIM_TO_THETA_FIELD: dict[str, str] = {
    "thermal": "thermal",
    "power": "power",
    "memory": "mem_pressure",
    "contention": "contention",
}


@dataclass(frozen=True)
class SignalSensitivity:
    """One signal's plan-cost sensitivity (a ``.SENS`` row).

    ``signal`` is the T1 metric-definition name (e.g. ``"thermal.pressure"``).
    ``cost_dim`` is the dim it calibrates; ``theta_field`` is the perturbed Theta
    pressure (or ``None`` when the dim has no Theta channel). ``baseline_score`` and
    ``perturbed_score`` are the scalarized plan scores; ``delta`` =
    ``perturbed_score - baseline_score`` and ``abs_delta`` = ``|delta|`` is the
    ranking key."""

    signal: str
    cost_dim: str | None
    theta_field: str | None
    baseline_score: int
    perturbed_score: int
    delta: int

    @property
    def abs_delta(self) -> int:
        return abs(self.delta)


def _clamp01_100(v: int) -> int:
    return max(0, min(100, v))


def _perturb_theta(theta: Theta, theta_field: str, delta: int) -> Theta:
    """A copy of ``theta`` with ``theta_field`` bumped by ``delta``, clamped 0..100,
    via :func:`dataclasses.replace` (Theta is frozen). Never mutates the input."""
    new_val = _clamp01_100(getattr(theta, theta_field) + delta)
    return replace(theta, **{theta_field: new_val})


def signal_sensitivity(
    module,
    h,
    theta: Theta,
    *,
    signals=None,
    delta: int = 10,
    policy: Policy | None = None,
    registry: SignalRegistry | None = None,
) -> list[SignalSensitivity]:
    """Rank telemetry signals by how much perturbing them moves the optimized PLAN
    COST (the ``.SENS`` analysis) — a finite-difference over the EXISTING optimizer.

    For each telemetry signal that maps (via the T1 registry's ``cost_dim`, see
    ``signal_registry.MetricDefinition``) to a perturbable :class:`Theta` pressure,
    this:

      1. builds a perturbed Theta with that pressure bumped by ``delta`` and clamped
         to ``0..100`` (:func:`dataclasses.replace`),
      2. re-runs :func:`bcir.kbcir.realize.optimize` on the *same* module/H/policy
         with the perturbed Theta, and
      3. records ``Δscore = score(perturbed) - score(baseline)``.

    Signals are returned ranked by ``|Δscore|`` DESCENDING, with a deterministic
    tie-break by signal name (ascending). The high-|Δscore| signals are where the
    sampling budget should go (:func:`sampling_budget`). A signal whose ``cost_dim``
    has no Theta channel (e.g. ``fabric``/``reliability``) is included with Δscore 0
    (honest: the optimizer's Theta coupling does not move for it on this workload).

    Arguments:
      * ``module``/``h``/``theta``/``policy`` — exactly the optimizer's inputs;
        ``policy`` defaults to ``PERF`` (the same default as ``optimize``).
      * ``signals`` — an explicit list of T1 metric-definition names to score; default
        is every registry signal that maps to a perturbable Theta pressure. (Signals
        sharing a Theta field — e.g. ``thermal.pressure`` and ``thermal.die_millicelsius``
        — each get a row; they get the same Δscore by construction.)
      * ``delta`` — the pressure bump (default 10, on the 0..100 scale).
      * ``registry`` — the T1 :class:`SignalRegistry` providing the signal→cost_dim
        map; defaults to :func:`default_registry`.

    NEVER touches the legality path: it only reads the registry (Readings/None) and
    re-runs the cost optimizer on a perturbed Theta. No Diagnostic, no R-law. The
    cost model is NOT reimplemented — this is a pure finite-difference over
    ``optimize``. Deterministic + side-effect-free (Theta is frozen; perturbation is
    a fresh copy)."""
    if delta == 0:
        raise ValueError("signal_sensitivity delta must be non-zero (it is the perturbation step)")
    pol = PERF if policy is None else policy
    reg = default_registry() if registry is None else registry

    # Resolve which signals to score and each one's cost_dim, from the T1 registry.
    if signals is None:
        names = [
            p.name for p in reg.providers() if p.definition.cost_dim in _COST_DIM_TO_THETA_FIELD
        ]
    else:
        names = list(signals)

    baseline = optimize(module, h, theta, pol).score

    rows: list[SignalSensitivity] = []
    # Cache per-Theta-field re-optimizations: signals sharing a field perturb the
    # identical Theta, so we re-run `optimize` once per distinct perturbed field.
    score_for_field: dict[str, int] = {}
    for name in names:
        prov = reg.get(name)
        cost_dim = prov.definition.cost_dim if prov is not None else None
        theta_field = _COST_DIM_TO_THETA_FIELD.get(cost_dim) if cost_dim else None
        if theta_field is None:
            rows.append(
                SignalSensitivity(
                    signal=name,
                    cost_dim=cost_dim,
                    theta_field=None,
                    baseline_score=baseline,
                    perturbed_score=baseline,
                    delta=0,
                )
            )
            continue
        if theta_field not in score_for_field:
            perturbed = _perturb_theta(theta, theta_field, delta)
            score_for_field[theta_field] = optimize(module, h, perturbed, pol).score
        perturbed_score = score_for_field[theta_field]
        rows.append(
            SignalSensitivity(
                signal=name,
                cost_dim=cost_dim,
                theta_field=theta_field,
                baseline_score=baseline,
                perturbed_score=perturbed_score,
                delta=perturbed_score - baseline,
            )
        )

    # Rank by |Δscore| desc, deterministic tie-break by name asc.
    rows.sort(key=lambda r: (-r.abs_delta, r.signal))
    return rows


def sampling_budget(ranked, total: int, *, floor: int = 1) -> dict[str, int]:
    """Allocate a fixed sampling budget across signals proportional to ``|Δscore|`` —
    the ``.SENS`` payoff: steer the sampling budget toward the high-impact signals.

    ``ranked`` is the :func:`signal_sensitivity` output (or any iterable of objects
    with ``.signal`` and ``.abs_delta``). ``total`` is the integer budget to hand out.
    Each signal gets at least ``floor`` samples (the documented floor: a zero-
    sensitivity signal is still sampled occasionally — drift can change tomorrow's
    ranking, so it must not go fully dark). The remainder above the floors is split
    proportional to ``|Δscore|`` (integer largest-remainder apportionment, so the
    allocation is exact and deterministic — it sums to EXACTLY ``total``).

    Determinism: ties in the remainder break by the signal's position in ``ranked``
    (already ranked |Δscore| desc, name asc), so the result is reproducible. Raises if
    ``total`` cannot cover the floors (``total < floor * n``)."""
    rows = list(ranked)
    n = len(rows)
    if n == 0:
        return {}
    if floor < 0:
        raise ValueError("sampling_budget floor must be non-negative")
    if total < floor * n:
        raise ValueError(
            f"sampling_budget total {total} cannot cover the floor "
            f"({floor} x {n} signals = {floor * n})"
        )

    names = [r.signal for r in rows]
    weights = [max(0, r.abs_delta) for r in rows]
    remaining = total - floor * n  # the proportional pool above floors
    alloc = {name: floor for name in names}

    wsum = sum(weights)
    if remaining > 0 and wsum > 0:
        # Largest-remainder (Hamilton) apportionment of `remaining` by weight.
        exact = [remaining * w for w in weights]  # numerator (denominator wsum)
        base = [e // wsum for e in exact]
        rema = [e - b * wsum for e, b in zip(exact, base)]
        for name, b in zip(names, base):
            alloc[name] += b
        leftover = remaining - sum(base)
        # Hand out the leftover to the largest remainders; ties by ranked position.
        order = sorted(range(n), key=lambda i: (-rema[i], i))
        for i in order[:leftover]:
            alloc[names[i]] += 1
    elif remaining > 0:
        # All weights zero (no signal moved the cost): spread the pool evenly,
        # leftover to the earliest (highest-ranked) signals — still deterministic.
        per = remaining // n
        for name in names:
            alloc[name] += per
        for name in names[: remaining - per * n]:
            alloc[name] += 1

    return alloc
