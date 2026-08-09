"""J6 — the certified K_BCIR encoding choice.

`selection.py` measures. This module decides *when a measurement is allowed to decide*, and
the answer is: only from a frozen, generation-tagged table, on a target that was actually
measured. §6.2 is explicit about the failure mode it exists to prevent:

> The current Python harness is retained as an oracle experiment. **Production selection
> reads the frozen target table and refuses an unmeasured required target instead of
> substituting Python timings.**

That refusal is the whole point of this file. §2 of the roadmap already records why: on the
Python rail `json.loads` is native C while COER decode is pure Python, so an oracle timing
orders the candidates by *which implementation happened to be written in C*, not by what the
encoding costs on a target. Substituting that number for a missing measurement would produce
a confident, reproducible, wrong answer — which is worse than no answer, because it looks
like evidence.

**Three things are kept apart, and the type system keeps them apart.**

* **Legality** is a verdict. It comes from a round trip and never from a number, and
  `selection.select` already applies it first. Nothing here can promote a refused candidate.
* **Exact measurement** is `octets`: the same value under the same rule is the same length
  on every host, forever. It is arithmetic, not statistics.
* **Graded measurement** is timing. It has a distribution, so it is carried as an
  *interval*, never a scalar, and two candidates whose intervals overlap are **not
  distinguishable** — the selection says so rather than picking the lower median.

That last point is the one that changes behaviour. A median comparison always produces a
winner, including when the difference is noise; an interval comparison produces "these are
the same" and hands the decision to the exact measurement instead. Deterministic, honest,
and reproducible across hosts.

**The intervals are distribution-free.** They come from order statistics, not from a normal
approximation: for `n` samples the interval between the `k`th smallest and the `k`th largest
covers the true median with probability `1 - 2 * P(Binomial(n, 1/2) < k)`, whatever the
underlying distribution is. Timing distributions are heavy-tailed and asymmetric — a normal
CI on them would understate the spread precisely where scheduler noise lives — and this
needs no floating point, no library, and no distributional assumption.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .selection import ALL_CANDIDATES, Candidate, Measurement, Objective, measure_one
from .tags import Asn1Error

#: Bumped when a table's shape changes in a way an older reader would misread.
#: Bumped to 2 when `decode_kind` joined the table. A schema-free and a schema-directed table
#: describe *different questions* about the same candidates, so they must never share a digest
#: — the same argument that moves ECN's `SYNTAX_VERSION` when a spelling changes what a decoder
#: reads. A certificate carries the digest of the table it read, so the discriminator has to be
#: inside the digest rather than beside it.
COST_TABLE_VERSION = 2

#: A table with fewer samples than this is refused at construction. Not a statistical
#: threshold so much as a floor below which an order-statistic interval covers almost
#: nothing: at n = 7 the [2, 6] interval already covers the median with probability ~0.875,
#: while at n = 3 the widest useful interval is the full range.
MIN_SAMPLES = 7


def _binomial_tail(n: int, k: int) -> tuple[int, int]:
    """`P(Binomial(n, 1/2) < k)` as an exact rational `(numerator, 2**n)`.

    Exact integer arithmetic rather than floats: a coverage figure that varies with the
    host's rounding is not a coverage figure, and this feeds a certificate.
    """
    total = 0
    row = 1
    for i in range(k):
        total += row
        row = row * (n - i) // (i + 1)
    return total, 1 << n


@dataclass(frozen=True)
class Interval:
    """A distribution-free confidence interval for a median, from order statistics.

    `coverage_ppm` is parts per million rather than a float so the certificate carries an
    exact integer. `low` and `high` are observed samples — real measurements, not fitted
    parameters — which is what makes the interval reportable without a model.
    """

    low: int
    high: int
    median: int
    samples: int
    coverage_ppm: int

    def overlaps(self, other: "Interval") -> bool:
        """Whether the two are statistically indistinguishable at this coverage.

        Overlap is the honest reading of "we cannot tell these apart". A median comparison
        would always name a winner; this reports that there is not one, and lets the caller
        fall back to a measurement that has no distribution.
        """
        return self.low <= other.high and other.low <= self.high


def interval_of(samples: list[int], *, rank: int | None = None) -> Interval:
    """The order-statistic interval over `samples`.

    `rank` selects which order statistic bounds the interval; the default keeps coverage
    above ~95% for the sample counts this harness produces. A larger rank is a tighter
    interval and less coverage, and the certificate records which was used, because an
    interval whose construction is not stated is not evidence.
    """
    if len(samples) < MIN_SAMPLES:
        raise Asn1Error(
            f"an order-statistic interval needs at least {MIN_SAMPLES} samples, got "
            f"{len(samples)}; a narrower table is not a more precise one")
    ordered = sorted(samples)
    n = len(ordered)
    if rank is None:
        # The largest rank whose two-sided coverage still clears 95%.
        rank = 1
        for candidate_rank in range(1, n // 2 + 1):
            numerator, denominator = _binomial_tail(n, candidate_rank)
            if denominator - 2 * numerator >= (denominator * 95) // 100:
                rank = candidate_rank
            else:
                break
    numerator, denominator = _binomial_tail(n, rank)
    coverage_ppm = ((denominator - 2 * numerator) * 1_000_000) // denominator
    return Interval(low=ordered[rank - 1], high=ordered[n - rank], median=ordered[n // 2],
                    samples=n, coverage_ppm=coverage_ppm)


@dataclass(frozen=True)
class CostRow:
    """One (target, candidate) row of a frozen table.

    `octets` sits beside the two intervals deliberately. It is the exact measurement, and
    when the intervals overlap it is what decides — so a row that carried only timings
    could not resolve its own ties.
    """

    candidate: str
    octets: int
    encode: Interval
    decode: Interval


@dataclass(frozen=True)
class EncodingCostTable:
    """A frozen, generation-tagged encoding cost table for ONE target.

    Frozen in the sense that matters: it is data, it carries the provenance of how it was
    produced, and the planner never sees the measurement loop. `provenance` is not
    decoration — a table built from `modeled` numbers must never be mistaken for one
    measured on the part, and `certify` records it in the certificate.
    """

    target: str
    cal_gen: int
    provenance: str
    rows: tuple[CostRow, ...]
    version: int = COST_TABLE_VERSION
    #: WHICH decode question this table's `decode` interval answers.
    #:
    #: `schema-free` is a structural scan with no type in hand — the trust-boundary cost, and
    #: the only one X.690 and X.697 candidates can be measured under. `schema-directed` is the
    #: deployment cost, with the plan already compiled, and it is the only way X.696 6.2 lets
    #: OER be decoded at all.
    #:
    #: They are never merged. Averaging a structural scan with a plan-driven decode would give
    #: one number for two different pieces of work, which is the error 6.2 spends a paragraph
    #: on — so this is a discriminator, not a flag, and it is inside `digest`.
    decode_kind: str = "schema-free"

    def __post_init__(self) -> None:
        if self.decode_kind not in ("schema-free", "schema-directed"):
            raise Asn1Error(
                f"decode_kind {self.decode_kind!r} must be schema-free or schema-directed; "
                f"the two answer different questions and a table is exactly one of them")
        if self.provenance not in ("measured", "modeled", "oracle"):
            raise Asn1Error(
                f"cost table provenance {self.provenance!r} must be measured, modeled or "
                f"oracle; 'oracle' names a Python-harness table, which §6.2 forbids "
                f"production selection from reading")
        names = [row.candidate for row in self.rows]
        if len(set(names)) != len(names):
            raise Asn1Error("a cost table names a candidate twice")

    def row(self, candidate: str) -> CostRow | None:
        for row in self.rows:
            if row.candidate == candidate:
                return row
        return None

    def digest(self) -> str:
        """The content address of the table, over its canonical JSON."""
        return hashlib.sha256(self.serialize()).hexdigest()

    def serialize(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()


class UnmeasuredTarget(Asn1Error):
    """Raised when production selection is asked about a target with no frozen row.

    Its own class rather than a generic error, because the correct response is specific:
    measure the target, or declare the objective one that needs no timing. Falling back to
    an oracle number is the one thing that must not happen quietly, and a caller that
    catches `Asn1Error` broadly would otherwise be able to do exactly that by accident.
    """


@dataclass(frozen=True)
class Certificate:
    """§6.2's selection certificate.

    Every field is something a reader can check rather than trust: identities are content
    addresses, the verdict is separate from the costs, and the tie-break rule is named so a
    third party can re-run the decision and get the same answer.
    """

    version: int
    schema_digest: str
    value_digest: str
    objective: str
    target: str
    cal_gen: int
    provenance: str
    table_digest: str
    admitted: tuple[str, ...]
    refused: tuple[tuple[str, str], ...]
    selected: str | None
    tie_break: str
    interval_rank_coverage_ppm: int
    #: The candidates that were statistically indistinguishable from the winner. Recorded
    #: because "we chose A over B" and "A and B were the same and A sorted first" are
    #: different decisions, and a certificate that hid the difference would overstate what
    #: the measurement showed.
    indistinguishable: tuple[str, ...] = ()

    def serialize(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()


def measure_repeatedly(candidate: Candidate, kind, value, *, repeats: int = MIN_SAMPLES
                       ) -> tuple[Measurement, list[int], list[int]]:
    """Run one candidate `repeats` times, returning the verdict and the two sample sets.

    The verdict is taken from the FIRST run and is not re-derived per repeat: legality is a
    property of the value and the rule, so a candidate that flips legality between runs
    would be a defect in the codec rather than a measurement to average. If it ever did
    flip, the assertion below turns it into a loud failure instead of a quiet majority vote.
    """
    first = measure_one(candidate, kind, value)
    encode_ns: list[int] = []
    decode_ns: list[int] = []
    for _ in range(repeats):
        again = measure_one(candidate, kind, value)
        if again.legal != first.legal or again.octets != first.octets:
            raise Asn1Error(
                f"{candidate.name}: legality or exact size changed between runs "
                f"({first.legal}/{first.octets} then {again.legal}/{again.octets}); that "
                f"is a codec defect, not measurement noise")
        encode_ns.append(again.encode_ns)
        decode_ns.append(again.decode_ns)
    return first, encode_ns, decode_ns


def build_table(kind, value, *, target: str, cal_gen: int, provenance: str = "oracle",
                candidates=ALL_CANDIDATES, repeats: int = MIN_SAMPLES
                ) -> EncodingCostTable:
    """Measure every candidate `repeats` times and freeze the result.

    Defaults to `provenance="oracle"` on purpose. This function runs on whatever host
    imported it, under a Python codec, so `measured` would be a false claim — and
    `select_certified` refuses an oracle table for exactly that reason. Producing a
    `measured` table is a native-harness job, and this signature makes that a deliberate
    argument rather than an accident.
    """
    rows = []
    for candidate in candidates:
        verdict, encode_ns, decode_ns = measure_repeatedly(candidate, kind, value,
                                                           repeats=repeats)
        if not verdict.legal:
            continue
        rows.append(CostRow(candidate=candidate.name, octets=verdict.octets or 0,
                            encode=interval_of(encode_ns), decode=interval_of(decode_ns)))
    return EncodingCostTable(target=target, cal_gen=cal_gen, provenance=provenance,
                             rows=tuple(rows))


#: The tie-break, named so a certificate can state it and a third party can reproduce it.
TIE_BREAK = ("exact-octets-then-declared-order: candidates whose intervals overlap the "
             "winner's are statistically indistinguishable and are resolved by exact "
             "encoded size, then by the order the candidate table declares")


def select_certified(kind, value, table: EncodingCostTable, *,
                     objective: Objective = Objective.WIRE_SIZE,
                     candidates=ALL_CANDIDATES,
                     allow_oracle_table: bool = False) -> Certificate:
    """The production selection: legality first, then the frozen table, then a certificate.

    Refuses rather than substitutes. A candidate that is legal but absent from the table is
    an **unmeasured** candidate, and an objective that needs a timing cannot be decided for
    it — so `UnmeasuredTarget` is raised instead of an oracle number being quietly used.
    `Objective.WIRE_SIZE` needs no timing at all and is therefore decidable from exact
    arithmetic alone, which is why it is the default.
    """
    # The oracle-table refusal is checked where a timing is actually CONSULTED, not here.
    # A guard that fires when it is not needed is a guard people learn to pass
    # `allow_oracle_table=True` to by reflex -- and they would then be carrying that flag
    # into the timing decisions where it really matters. A wire-size objective reads no
    # interval, so an oracle table is harmless to it and refusing would be theatre.

    # --- law 1: legality, from a round trip, before any number is looked at ---------------
    admitted: list[tuple[Candidate, Measurement]] = []
    refused: list[tuple[str, str]] = []
    for candidate in candidates:
        verdict = measure_one(candidate, kind, value)
        if verdict.legal:
            admitted.append((candidate, verdict))
        else:
            refused.append((candidate.name, verdict.refusal))

    # --- law 2: canonical or excluded -----------------------------------------------------
    emittable = [(c, m) for c, m in admitted if c.canonical]
    for candidate, _ in admitted:
        if not candidate.canonical:
            refused.append((candidate.name, "not canonical; decodable but never selected"))

    schema_digest = hashlib.sha256(repr(kind).encode()).hexdigest()
    value_digest = hashlib.sha256(repr(value).encode()).hexdigest()
    common = dict(version=COST_TABLE_VERSION, schema_digest=schema_digest,
                  value_digest=value_digest, objective=objective.value,
                  target=table.target, cal_gen=table.cal_gen,
                  provenance=table.provenance, table_digest=table.digest(),
                  admitted=tuple(c.name for c, _ in emittable),
                  refused=tuple(sorted(refused)), tie_break=TIE_BREAK)

    if not emittable:
        return Certificate(**common, selected=None, interval_rank_coverage_ppm=0)

    # --- law 3: compare, on the objective, using the right kind of truth -------------------
    if objective in (Objective.NONE, Objective.WIRE_SIZE):
        # Exact arithmetic. No table row is needed and no interval is consulted, which is
        # why a wire-size decision is available on an unmeasured target.
        if objective is Objective.NONE:
            winner = emittable[0][1]
            tied: tuple[str, ...] = ()
        else:
            best = min(m.octets for _, m in emittable)
            same = [c.name for c, m in emittable if m.octets == best]
            winner = next(m for c, m in emittable if c.name == same[0])
            tied = tuple(same[1:])
        return Certificate(**common, selected=winner.candidate,
                           interval_rank_coverage_ppm=0, indistinguishable=tied)

    # A timing objective: from here on an interval decides, so the table's provenance is
    # load-bearing and §6.2's refusal applies.
    if table.provenance == "oracle" and not allow_oracle_table:
        raise UnmeasuredTarget(
            f"a {objective.value} objective would be decided from the {table.target!r} "
            f"table, whose provenance is 'oracle'; §6.2 requires production selection to "
            f"read a MEASURED table and refuse rather than substitute Python timings. On "
            f"this rail `json.loads` is native C while COER decode is Python, so an oracle "
            f"timing orders the candidates by which implementation happens to be compiled "
            f"(pass allow_oracle_table=True only in an experiment recorded as one)")

    # Every admitted candidate must also have a row, or the decision is not available at
    # this target -- the same refusal for a different missing thing.
    missing = [c.name for c, _ in emittable if table.row(c.name) is None]
    if missing:
        raise UnmeasuredTarget(
            f"target {table.target!r} (cal_gen {table.cal_gen}) has no measured row for "
            f"{', '.join(sorted(missing))}; a {objective.value} objective cannot be "
            f"decided here, and an oracle timing must not stand in for the missing "
            f"measurement")

    field_name = "encode" if objective is Objective.ENCODE_LATENCY else "decode"
    scored = [(table.row(c.name), m) for c, m in emittable]
    best_row = min(scored, key=lambda pair: getattr(pair[0], field_name).median)[0]
    best = getattr(best_row, field_name)
    # Everything whose interval overlaps the best is indistinguishable from it. Among
    # those, the exact measurement decides -- a number with no distribution, so the answer
    # is the same on every host.
    tie = [row for row, _ in scored if getattr(row, field_name).overlaps(best)]
    tie.sort(key=lambda row: (row.octets, [c.name for c in candidates].index(row.candidate)))
    return Certificate(**common, selected=tie[0].candidate,
                       interval_rank_coverage_ppm=best.coverage_ppm,
                       indistinguishable=tuple(sorted(r.candidate for r in tie[1:])))




# --- RCSP: a budgeted plan across several stages -------------------------------------------
#
# `select_certified` decides ONE encoding. A pipeline decides several — a manifest here, a
# telemetry frame there, a program artifact at the end — and the constraint that actually
# binds is usually global: total wire size under a link budget, total decode latency under a
# frame deadline. Choosing each stage's best in isolation solves a different problem and can
# miss the only feasible plan, because a stage whose local best is large may be the one that
# has to give way.
#
# That is a **resource-constrained shortest path**: minimize an additive cost along a chain
# subject to an additive resource staying inside a budget. With integer octets and a bounded
# budget it is exact by dynamic programming over (stage, octets spent) — no search, no
# heuristic, and no relaxation whose gap someone has to remember.
#
# P4's min-plus semiring is the algebra underneath: composition along the chain adds, and the
# choice at a stage is `min`. What P4 could not carry is the second dimension, which is the
# whole reason RCSP is not just a shortest path.


@dataclass(frozen=True)
class Stage:
    """One choice point: a name, and the candidates admissible there.

    The candidate names are checked against the table rather than assumed, so a stage that
    names something unmeasured refuses in the same way a single selection does.
    """

    name: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class BudgetedPlan:
    """The chosen candidate per stage, with the summed cost and what it cost in coverage.

    `coverage_ppm` is the load-bearing field. Summing intervals along a chain does **not**
    preserve their coverage: each is a statement that holds with some probability, and the
    statement about the sum holds only when they all do. `certified` says whether the answer
    is still evidence at that coverage or has decayed into a median comparison wearing an
    interval's clothes.
    """

    chosen: tuple[tuple[str, str], ...]
    total_octets: int
    latency_low: int
    latency_median: int
    latency_high: int
    coverage_ppm: int
    budget: int
    certified: bool
    indistinguishable: tuple[tuple[tuple[str, str], ...], ...] = ()


class Infeasible(Asn1Error):
    """No assignment of candidates to stages fits the budget.

    Its own class because the correct response differs from an unmeasured target: raise the
    budget, drop a stage, or admit a candidate that was excluded — none of which is
    "measure something".
    """


def _sum_coverage(parts: list[int]) -> int:
    """Union bound on the joint coverage of several independent interval statements.

    Each interval holds with probability `c_i`; the sum's bound holds when all of them do,
    and P(all) >= 1 - sum(1 - c_i). That is Bonferroni: conservative, distribution-free, and
    the same kind of truth the intervals themselves are.

    Multiplying the coverages would be tighter and would assume independence the harness has
    not established. Reporting the *component* coverage unchanged — the obvious shortcut —
    would claim a chain of ten 95% statements is itself 95%, which is false by a wide margin
    and in the optimistic direction.
    """
    deficit = sum(1_000_000 - c for c in parts)
    return max(0, 1_000_000 - deficit)


def select_budgeted(table: EncodingCostTable, stages, *, budget: int,
                    objective: Objective = Objective.DECODE_LATENCY,
                    min_coverage_ppm: int = 500_000,
                    allow_oracle_table: bool = False) -> BudgetedPlan:
    """Minimize the objective across `stages` subject to total octets <= `budget`.

    Exact, by dynamic programming over (stage, octets spent). The state space is bounded by
    the budget, which is why this terminates with an optimum rather than with a good guess.

    `min_coverage_ppm` is the floor below which the plan is reported as **not certified**:
    the answer is still the DP's optimum, but the intervals no longer separate it from its
    rivals and saying otherwise would be dressing a median up as evidence.
    """
    stages = tuple(stages)
    if not stages:
        raise Asn1Error("a budgeted plan needs at least one stage")
    if budget < 0:
        raise Asn1Error("a negative octet budget is not a constraint, it is a typo")
    if objective in (Objective.NONE, Objective.WIRE_SIZE):
        raise Asn1Error(
            f"{objective.value} is the RESOURCE here, not the objective; a budgeted plan "
            f"minimizes a timing subject to the octet budget, and minimizing octets subject "
            f"to an octet budget is a single selection with extra steps")
    if table.provenance == "oracle" and not allow_oracle_table:
        raise UnmeasuredTarget(
            f"a budgeted {objective.value} plan would be decided from the {table.target!r} "
            f"table, whose provenance is 'oracle'; §6.2 requires a MEASURED table (pass "
            f"allow_oracle_table=True only in an experiment recorded as one)")

    field_name = "encode" if objective is Objective.ENCODE_LATENCY else "decode"
    missing = sorted({name for stage in stages for name in stage.candidates
                      if table.row(name) is None})
    if missing:
        raise UnmeasuredTarget(
            f"target {table.target!r} (cal_gen {table.cal_gen}) has no measured row for "
            f"{', '.join(missing)}; a budgeted plan cannot be decided here")
    for stage in stages:
        if not stage.candidates:
            raise Asn1Error(f"stage {stage.name!r} admits no candidate at all")

    # state: octets spent -> (summed median, summed low, summed high, coverages, choices)
    State = tuple[int, int, int, tuple[int, ...], tuple[tuple[str, str], ...]]
    frontier: dict[int, State] = {0: (0, 0, 0, (), ())}
    for stage in stages:
        nxt: dict[int, State] = {}
        for spent, (median, low, high, coverages, chosen) in frontier.items():
            for name in stage.candidates:
                row = table.row(name)
                cost = getattr(row, field_name)
                total = spent + row.octets
                if total > budget:
                    continue
                state: State = (median + cost.median, low + cost.low, high + cost.high,
                                coverages + (cost.coverage_ppm,),
                                chosen + ((stage.name, name),))
                # Keep one optimum per resource level: the classic RCSP label rule, and it
                # is what makes the table's width the budget rather than the candidate count
                # raised to the number of stages.
                held = nxt.get(total)
                if held is None or state[0] < held[0]:
                    nxt[total] = state
        frontier = nxt
        if not frontier:
            break

    if not frontier:
        cheapest = sum(min(table.row(n).octets for n in stage.candidates) for stage in stages)
        raise Infeasible(
            f"no assignment fits {budget} octets across {len(stages)} stages; the cheapest "
            f"legal plan costs {cheapest}")

    best_spent = min(frontier, key=lambda spent: (frontier[spent][0], spent))
    median, low, high, coverages, chosen = frontier[best_spent]
    coverage = _sum_coverage(list(coverages))
    # Everything whose summed interval overlaps the winner's is indistinguishable from it,
    # and at a chain length where coverage has decayed that will be most of them — which is
    # the honest report, not a defect to hide behind a median.
    #
    # A LOWER BOUND, not the whole tie set: the label rule keeps one optimum per resource
    # level, so a rival that spends the same octets as a better plan was already discarded.
    # Reporting it as complete would overstate what the frontier retains, and widening the
    # rule to keep every plan would trade an exact pseudo-polynomial search for an
    # exponential one to enumerate answers the caller cannot act on differently.
    tied = tuple(sorted(
        state[4] for spent, state in frontier.items()
        if spent != best_spent and state[1] <= high and low <= state[2]))
    return BudgetedPlan(
        chosen=chosen, total_octets=best_spent, latency_low=low, latency_median=median,
        latency_high=high, coverage_ppm=coverage, budget=budget,
        certified=coverage >= min_coverage_ppm, indistinguishable=tied)


__all__ = [
    "COST_TABLE_VERSION", "MIN_SAMPLES", "TIE_BREAK", "BudgetedPlan", "Certificate",
    "CostRow", "EncodingCostTable", "Infeasible", "Interval", "Stage", "UnmeasuredTarget",
    "build_table", "interval_of", "measure_repeatedly", "select_budgeted",
    "select_certified",
]
