"""Cost-governed encoding selection — the measurement half of roadmap phase H.

Every other module in `bcir.asn1` implements *one* encoding rule. This one implements the
question the whole suite exists to answer: **given an abstract value, which rule should
carry it?** ECN (X.692) is a notation for defining encodings and never says how to choose
among them; BCIR has an optimizer, and this is the layer that hands the optimizer something
to optimize.

It is also the gate the roadmap puts in front of ECN's second half. §6 says that if
cost-governed selection over a *fixed* candidate set demonstrates the win, ECN's
user-defined encoding classes are not required for the thesis — and that "the decision is a
gate, not a preference … Record the decision with the measurement that justified it." So
this module produces the record, and until it ran the decision could not honestly be taken.

THE THREE LAWS, from the roadmap, implemented rather than asserted:

* **Legality first.** "An encoding is a candidate only if the abstract value is
  representable in it, which is a verifier question, never a cost question." So
  `Measurement.legal` is computed before any timing is looked at, and `select` filters on it
  before it compares anything. A candidate that cannot carry the value is not an expensive
  candidate — it is not a candidate.
* **Two-truth.** "A measured encode/decode cost is graded truth and must not become a
  legality verdict." The two kinds of truth are kept in different fields and are produced by
  different code paths: `legal` and `refusal` come from a round trip, `octets` and the two
  timings from measurement. Nothing here lets a slow encoding become an illegal one, or a
  fast one become legal.
* **Canonical or excluded.** "A rule with no canonical variant may be decoded but never
  selected for emission, since a selected encoding is a digested artifact." `Candidate
  .canonical` carries that, and it is why BER and the two BASIC-PER variants appear in
  `ALL_CANDIDATES` — they are real decode targets — while `select` will never return one.

WHAT IS EXACT AND WHAT IS INDICATIVE, which is the same distinction one level down.
`octets` is an exact, deterministic measurement: the same value under the same rule is the
same length on every host, forever. The two timings are **not** — they are Python-oracle
timings on a shared runner, and the roadmap is explicit that the calibrated cost table for
the real decision lives in `kbcir/microbench.py` and the frozen-table machinery. So
`Objective.WIRE_SIZE` is decided by arithmetic and the latency objectives are decided by
measurement that this module labels as such. Reporting a noisy number as though it were a
law is exactly what "two-truth" exists to prevent.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence as _Seq
from dataclasses import dataclass
from enum import Enum

from .codec import Strictness
from .schema import Asn1Type, Module
from .tags import Asn1Error


class Objective(Enum):
    """What the caller is short of. Mirrors the cost-vector axes phase H names.

    `NONE` is the degenerate case the roadmap pins: "with no budget it reproduces today's
    DER exactly … pinning that nothing regresses". It is not "no objective" in the sense of
    picking arbitrarily — it is a named objective whose answer is the status quo.
    """

    NONE = "none"
    WIRE_SIZE = "memory"
    ENCODE_LATENCY = "compute.encode"
    DECODE_LATENCY = "compute.decode"


@dataclass(frozen=True)
class Candidate:
    """One encoding rule, as something the optimizer can choose.

    `canonical` is a *property of the rule*, not a preference: it says whether one abstract
    value has exactly one encoding under it. A rule without that cannot carry a digested
    artifact, because two conforming senders could produce two different digests for the
    same value.
    """

    name: str
    #: The object identifier the defining Recommendation assigns, or None when it assigns
    #: none. JER is the interesting case: X.697 §42.2 registers exactly one OID and defines
    #: no canonical variant at all, so the canonical JER candidate below is BCIR's own
    #: profile and truthfully has nothing to point at.
    oid: tuple[int, ...] | None
    canonical: bool
    encode: Callable[[Asn1Type, object], bytes]
    decode: Callable[[bytes, Asn1Type], object]
    #: How the law rail spells this syntax — the `BCIR_Asn1Rules` case in
    #: `mlir/include/BCIR/BCIRAttrs.td`. It lives HERE, next to the encoder that defines
    #: what the name means, rather than in a table the parity test keeps privately: a
    #: transcription maintained by the checker is one the checker cannot catch drifting.
    #: `test_asn1_law_parity` reads this field, so adding a candidate without giving it a
    #: law-rail spelling fails the gate instead of silently going ungoverned by R24.
    rules: str = ""


def _der_encode(kind: Asn1Type, value) -> bytes:
    return Module("<selection>", (), {"T": kind}).encode("T", value)


def _der_decode(data: bytes, kind: Asn1Type):
    return Module("<selection>", (), {"T": kind}).decode("T", data, strictness=Strictness.DER)


def _ber_decode(data: bytes, kind: Asn1Type):
    return Module("<selection>", (), {"T": kind}).decode("T", data, strictness=Strictness.BER)


def _per(rules, variant):
    from .per import decode_per, encode_per

    def encode(kind, value):
        return encode_per(kind, value, variant=variant, rules=rules)

    def decode(data, kind):
        return decode_per(data, kind, variant=variant, rules=rules)

    return encode, decode


def _oer(rules):
    from .oer import decode_oer, encode_oer

    def encode(kind, value):
        return encode_oer(kind, value, rules=rules)

    def decode(data, kind):
        return decode_oer(kind, data, rules=rules)

    return encode, decode


def _jer(rules):
    from .jer import decode_jer, encode_jer

    def encode(kind, value):
        return encode_jer(kind, value, rules=rules)

    def decode(data, kind):
        return decode_jer(data, kind, rules=rules)

    return encode, decode


def _build_candidates() -> tuple[Candidate, ...]:
    from . import BER_OID, DER_OID
    from .jer import JER_OID
    from .jer import JerRules as _JerRules
    from .oer import BASIC_OER_OID, CANONICAL_OER_OID
    from .oer import OerRules as _OerRules
    from .per import (
        BASIC_PER_ALIGNED_OID,
        BASIC_PER_UNALIGNED_OID,
        CANONICAL_PER_ALIGNED_OID,
        CANONICAL_PER_UNALIGNED_OID,
    )
    from .per import PerRules as _PerRules
    from .per import PerVariant as _PerVariant

    aligned = _per(_PerRules.CANONICAL, _PerVariant.ALIGNED)
    unaligned = _per(_PerRules.CANONICAL, _PerVariant.UNALIGNED)
    basic_aligned = _per(_PerRules.BASIC, _PerVariant.ALIGNED)
    basic_unaligned = _per(_PerRules.BASIC, _PerVariant.UNALIGNED)
    coer = _oer(_OerRules.CANONICAL)
    basic_oer = _oer(_OerRules.BASIC)
    cjer = _jer(_JerRules.CANONICAL)
    bjer = _jer(_JerRules.BASIC)

    return (
        # --- selectable: one abstract value, one encoding ---------------------------------
        Candidate("DER", DER_OID, True, _der_encode, _der_decode, "der"),
        Candidate(
            "CANONICAL-PER-UNALIGNED",
            CANONICAL_PER_UNALIGNED_OID,
            True,
            *unaligned,
            rules="canonical_per_unaligned",
        ),
        Candidate(
            "CANONICAL-PER-ALIGNED",
            CANONICAL_PER_ALIGNED_OID,
            True,
            *aligned,
            rules="canonical_per_aligned",
        ),
        Candidate("COER", CANONICAL_OER_OID, True, *coer, rules="coer"),
        # X.697 registers no canonical variant, so this names BCIR's profile and carries no
        # object identifier of its own -- see the module docstring of `jer.py`.
        Candidate("JER-BCIR-CANONICAL", None, True, *cjer, rules="bcir_canonical_jer"),
        # --- decode targets only: a value has more than one legal encoding ----------------
        Candidate("BER", BER_OID, False, _der_encode, _ber_decode, "ber"),
        Candidate(
            "BASIC-PER-UNALIGNED",
            BASIC_PER_UNALIGNED_OID,
            False,
            *basic_unaligned,
            rules="basic_per_unaligned",
        ),
        Candidate(
            "BASIC-PER-ALIGNED",
            BASIC_PER_ALIGNED_OID,
            False,
            *basic_aligned,
            rules="basic_per_aligned",
        ),
        Candidate("BASIC-OER", BASIC_OER_OID, False, *basic_oer, rules="oer"),
        Candidate("JER", JER_OID, False, *bjer, rules="jer"),
    )


#: Every rule this repo can speak, selectable or not.
ALL_CANDIDATES: tuple[Candidate, ...] = _build_candidates()

#: The five the roadmap's §6 reduction gate names. Note the fifth: §6 calls it "CJER", but
#: X.697 defines no canonical variant, so the candidate is BCIR's own canonical JER profile.
SELECTABLE: tuple[Candidate, ...] = tuple(c for c in ALL_CANDIDATES if c.canonical)


@dataclass(frozen=True)
class Measurement:
    """What one candidate did with one value.

    The field split IS the two-truth law. `legal` and `refusal` are a verdict, produced by a
    round trip; `octets`, `encode_ns` and `decode_ns` are graded measurements. `octets` is
    exact and reproducible anywhere; the two timings are Python-oracle numbers on whatever
    host ran them, and are labelled `_ns` rather than `cost` so nobody mistakes them for the
    calibrated table phase H actually selects on.
    """

    candidate: str
    legal: bool
    octets: int | None = None
    encode_ns: int = 0
    decode_ns: int = 0
    refusal: str | None = None


def _time(action, repeats: int) -> tuple[object, int]:
    """Run `action` `repeats` times and keep the *minimum* elapsed time.

    The minimum rather than the mean, because on a shared runner every source of noise adds
    time and none subtracts it: the fastest observed run is the one least contaminated. This
    is the same reason `kbcir/microbench.py` reports a floor.
    """
    best = None
    result = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        result = action()
        elapsed = time.perf_counter_ns() - start
        best = elapsed if best is None else min(best, elapsed)
    return result, int(best or 0)


def measure_one(candidate: Candidate, kind: Asn1Type, value, *, repeats: int = 5) -> Measurement:
    """Encode, decode and compare — then, and only then, report cost.

    Legality is a **round trip**, not merely "the encoder did not raise". An encoder that
    silently loses a component produces octets and a wrong value, which is the failure mode
    a size comparison would otherwise reward: the shortest encoding of the wrong value is
    always going to win. So the comparison is part of the verdict.
    """
    try:
        data, encode_ns = _time(lambda: candidate.encode(kind, value), repeats)
    except (Asn1Error, ValueError, TypeError, OverflowError) as error:
        return Measurement(candidate.name, False, refusal=f"encode: {error}")
    if not isinstance(data, (bytes, bytearray)):
        return Measurement(candidate.name, False, refusal="encode: did not produce octets")
    try:
        back, decode_ns = _time(lambda: candidate.decode(bytes(data), kind), repeats)
    except (Asn1Error, ValueError, TypeError, OverflowError) as error:
        return Measurement(
            candidate.name, False, octets=len(data), encode_ns=encode_ns, refusal=f"decode: {error}"
        )
    if not _equivalent(back, value):
        return Measurement(
            candidate.name,
            False,
            octets=len(data),
            encode_ns=encode_ns,
            decode_ns=decode_ns,
            refusal="round trip returned a different value",
        )
    return Measurement(candidate.name, True, len(data), encode_ns, decode_ns)


def _equivalent(decoded, original) -> bool:
    """Round-trip equality, allowing for what an encoding rule is permitted to change.

    Two things are legitimately not preserved. A SET OF is unordered (X.680 §28.3 NOTE 2:
    "Encoding rules are not required to preserve the order of these values"), so a canonical
    rule that sorts has not lost anything. And a decode may add the `<name>.resolved`
    enrichment for an open type, which is information gained rather than a difference.
    """
    if isinstance(decoded, dict) and isinstance(original, dict):
        pruned = {k: v for k, v in decoded.items() if not k.endswith(".resolved")}
        if set(pruned) != set(original):
            return False
        return all(_equivalent(pruned[k], original[k]) for k in original)
    if isinstance(decoded, list) and isinstance(original, list):
        if len(decoded) != len(original):
            return False
        if all(_equivalent(a, b) for a, b in zip(decoded, original)):
            return True
        # An unordered collection: compare as multisets of their repr, which is total and
        # order-insensitive without requiring the elements to be hashable.
        return sorted(map(repr, decoded)) == sorted(map(repr, original))
    return decoded == original


def measure(
    kind: Asn1Type, value, *, candidates: _Seq[Candidate] | None = None, repeats: int = 5
) -> tuple[Measurement, ...]:
    """Measure one abstract value against every candidate."""
    return tuple(
        measure_one(c, kind, value, repeats=repeats)
        for c in (candidates if candidates is not None else ALL_CANDIDATES)
    )


def select(
    measurements: _Seq[Measurement],
    *,
    objective: Objective = Objective.NONE,
    candidates: _Seq[Candidate] | None = None,
) -> Measurement | None:
    """Choose an encoding for emission, applying the three laws in order.

    The order is the point. Legality is settled first and without reference to any number;
    then the canonical filter, which is also a property rather than a cost; and only what
    survives both is compared on the objective. Ties are broken by the declared candidate
    order rather than by whichever measurement happened to be fastest on the day, so the
    same corpus selects the same rule on every host.
    """
    pool = {c.name: c for c in (candidates if candidates is not None else ALL_CANDIDATES)}
    # Law 1: legality first, a verifier question.
    legal = [m for m in measurements if m.legal and m.candidate in pool]
    # Law 3: canonical or excluded.
    emittable = [m for m in legal if pool[m.candidate].canonical]
    if not emittable:
        return None
    order = [c.name for c in (candidates if candidates is not None else ALL_CANDIDATES)]
    emittable.sort(key=lambda m: order.index(m.candidate))
    if objective is Objective.NONE:
        # The degenerate case the roadmap pins: with no budget, today's answer, unchanged.
        for measurement in emittable:
            if measurement.candidate == "DER":
                return measurement
        return emittable[0]
    if objective is Objective.WIRE_SIZE:
        return min(emittable, key=lambda m: (m.octets, order.index(m.candidate)))
    if objective is Objective.ENCODE_LATENCY:
        return min(emittable, key=lambda m: (m.encode_ns, order.index(m.candidate)))
    return min(emittable, key=lambda m: (m.decode_ns, order.index(m.candidate)))


def report(kind: Asn1Type, value, *, label: str = "", repeats: int = 5) -> str:
    """A human-readable table — the artefact the roadmap's §6 gate asks to be recorded."""
    measurements = measure(kind, value, repeats=repeats)
    widest = max(len(m.candidate) for m in measurements)
    lines = [
        f"{label}" if label else "",
        f"{'candidate'.ljust(widest)}  octets  canonical  encode_ns  decode_ns  note",
    ]
    by_name = {c.name: c for c in ALL_CANDIDATES}
    baseline = next((m.octets for m in measurements if m.candidate == "DER"), None)
    for measurement in measurements:
        candidate = by_name[measurement.candidate]
        if not measurement.legal:
            lines.append(
                f"{measurement.candidate.ljust(widest)}       -          -"
                f"          -          -  {measurement.refusal}"
            )
            continue
        share = f"{100 * measurement.octets / baseline:5.1f}% of DER" if baseline else ""
        lines.append(
            f"{measurement.candidate.ljust(widest)}  {measurement.octets:6d}  "
            f"{'yes' if candidate.canonical else 'no ':>9}  "
            f"{measurement.encode_ns:9d}  {measurement.decode_ns:9d}  {share}"
        )
    return "\n".join(line for line in lines if line)


__all__ = [
    "ALL_CANDIDATES",
    "SELECTABLE",
    "Candidate",
    "Measurement",
    "Objective",
    "measure",
    "measure_one",
    "report",
    "select",
]
