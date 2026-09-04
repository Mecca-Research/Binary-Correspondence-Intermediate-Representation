"""bcir.kbcir.twotruth -- the two-truth separation (MOPC): classical truth `v`
for the laws, graded confidence `w` for everything else, with a quarantine
between them.

MOPC carries two kinds of truth, and the single most important discipline is that
they never mix:

  * **Classical truth `v`.** Deterministic, binary, generation-independent: the
    legality verdicts of the R-laws (R1-R13). A claim is legal or it is not; a
    manifest reproduces or it does not; a memory module is a fixpoint or it is
    not. There is no "0.7 legal." This is the truth `verify.*` speaks.
  * **Graded truth `(v, w)`.** A *graded proposition*: a value `v` carried with a
    confidence `w in [0,1]`. This is the learned/measured machinery already in the
    repo -- the softdp plan posterior, the bayescal conformal interval, the regret
    ledger's evidence. It answers *which legal plan is best*, never *whether a plan
    is legal*.

**The quarantine (enforced, not stated): a graded proposition may inform but never
become a legality verdict.** Graded truth is kept out of the verifier. The only
sanctioned crossing is a `decide`: an explicit, *recorded* collapse of a graded
proposition to a classical value at a frozen threshold -- the anneal/freeze of
LangRef Sec. 2/13 made auditable. A crossing is never silent, and
`verify.verify_quarantine` (R13) is the guard that no confidence-weighted value
reaches the R-laws except as the classical value of a recorded decision.

This is how "dynamic truth tables that learn" are imported safely: the graded
algebra (`g_and`/`g_or`/`g_not`) lives entirely on the graded side; it proposes,
and the classical laws dispose. The graded side may use any learned weights; the
classical side stays deterministic.

Confidence is an integer in Q-milli (0..1000) so the graded *bookkeeping* is
itself deterministic and testable; the learned values that fill it are produced
offline (L2/L3), never on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

_MILLI = 1000


def _clamp_milli(w: int) -> int:
    return max(0, min(_MILLI, int(w)))


class GradedTruthLeak(Exception):
    """A graded proposition (value, confidence) was passed where a classical
    legality verdict is required. Collapse it with `decide` first -- graded truth
    may inform a verdict but never *be* one (the two-truth quarantine)."""


# --- graded truth: the (value, confidence) proposition ---------------------------


@dataclass(frozen=True)
class Graded:
    """A graded proposition `(v, w)`: a value with a confidence in Q-milli
    (0..1000 == [0,1]). The learned/measured truth. It may inform a decision but
    may never be a legality verdict (use `decide` to collapse it first)."""

    value: object
    confidence_milli: int
    source: str = "?"  # which graded organ produced it (softdp/bayescal/regret/...)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_milli <= _MILLI:
            raise ValueError(f"confidence {self.confidence_milli} out of range [0,{_MILLI}]")

    @property
    def confidence(self) -> float:
        return self.confidence_milli / _MILLI

    @property
    def is_certain(self) -> bool:
        return self.confidence_milli == _MILLI


@dataclass(frozen=True)
class Decision:
    """The sanctioned graded->classical collapse: a graded proposition reduced to
    a classical value at a frozen threshold. This is the ONLY place graded truth
    crosses into classical truth, and it is recorded -- the crossing is never
    silent. `value` is classical; the confidence/threshold are kept for the audit
    trail (so R13 can witness the collapse)."""

    value: object
    confidence_milli: int
    threshold_milli: int
    source: str = "?"

    @property
    def admitted(self) -> bool:
        """The collapse cleared its threshold (confidence >= threshold): the
        graded proposition is confident enough to inform the decision."""
        return self.confidence_milli >= self.threshold_milli


def decide(g: Graded, threshold_milli: int) -> Decision:
    """Collapse a graded proposition to a classical decision at a frozen
    threshold -- the sanctioned, recorded crossing of the quarantine boundary."""
    return Decision(
        value=g.value,
        confidence_milli=g.confidence_milli,
        threshold_milli=_clamp_milli(threshold_milli),
        source=g.source,
    )


# --- the quarantine predicate + boundary guard -----------------------------------


def is_classical(x) -> bool:
    """True iff `x` is classical truth (not a graded proposition). A `Decision`'s
    *value* is classical, but a raw `Graded` -- or a `Decision` still wrapping a
    `Graded` value -- is not."""
    if isinstance(x, Graded):
        return False
    if isinstance(x, Decision):
        return is_classical(x.value)
    return True


def assert_classical(*values) -> None:
    """Raise `GradedTruthLeak` if any value is a graded proposition. Call this at
    the boundary into the R-laws; it is the runtime form of the quarantine."""
    for v in values:
        if not is_classical(v):
            raise GradedTruthLeak(
                f"graded proposition {v!r} reached a classical verdict boundary; "
                f"collapse it with decide() first"
            )


# --- the graded algebra (dynamic truth tables, quarantined on the graded side) ---


def g_not(a: Graded) -> Graded:
    """Graded negation: value negated, confidence complemented (1 - w)."""
    return Graded(value=not a.value, confidence_milli=_MILLI - a.confidence_milli, source=a.source)


def g_and(a: Graded, b: Graded) -> Graded:
    """Graded conjunction (product t-norm): w = w_a * w_b. The learned AND of two
    propositions -- a row of a dynamic truth table, living on the graded side."""
    w = (a.confidence_milli * b.confidence_milli) // _MILLI
    return Graded(
        value=bool(a.value) and bool(b.value), confidence_milli=w, source=_join(a.source, b.source)
    )


def g_or(a: Graded, b: Graded) -> Graded:
    """Graded disjunction (probabilistic sum t-conorm): w = w_a + w_b - w_a*w_b."""
    w = (
        a.confidence_milli
        + b.confidence_milli
        - (a.confidence_milli * b.confidence_milli) // _MILLI
    )
    return Graded(
        value=bool(a.value) or bool(b.value),
        confidence_milli=_clamp_milli(w),
        source=_join(a.source, b.source),
    )


def _join(s: str, t: str) -> str:
    return s if s == t else f"{s}+{t}"


# --- duck-typed wrappers for the existing graded organs ---------------------------


def from_soft(soft, claim_id: int) -> Graded:
    """Wrap a softdp plan-posterior marginal as a graded proposition: the
    argmax-marginal candidate (value) and its probability (confidence)."""
    name = soft.map_by_claim[claim_id]
    p = soft.marginals.get(claim_id, {}).get(name, 0.0)
    return Graded(value=name, confidence_milli=_clamp_milli(round(p * _MILLI)), source="softdp")


def from_conformal(table) -> Graded:
    """Wrap a bayescal conformal-calibrated table as a graded proposition: its
    constants (value) at its stated coverage (confidence)."""
    value = (table.gather_penalty, table.base_overhead, table.mem_unit)
    cov = getattr(table, "coverage_milli", 0)
    return Graded(value=value, confidence_milli=_clamp_milli(cov), source="bayescal")


def from_regret(verdict) -> Graded:
    """Wrap a regret boundary verdict as a graded proposition: the recommendation
    (value) with a confidence from its MDL evidence margin (saturating: a larger
    data_fit-over-complexity surplus is higher confidence)."""
    fit = float(getattr(verdict, "data_fit_nats", 0.0))
    comp = float(getattr(verdict, "complexity_nats", 0.0))
    margin = max(0.0, fit - comp)
    denom = max(1e-9, fit + comp)
    return Graded(
        value=verdict.verdict,
        confidence_milli=_clamp_milli(round(margin / denom * _MILLI)),
        source="regret",
    )
