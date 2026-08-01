"""J6's target calibration records: what a frozen cost table is allowed to be built from.

[`certified.py`](certified.py) answers "may the planner use a timing to choose an encoding?"
and its answer is: only from a frozen, generation-tagged table, on a target that was actually
measured. [`native_bench.py`](native_bench.py) produces such a table from a native C harness
— and `measured_table(target=...)` takes the target **on trust**. `target` is a string. A
table measured on a throttled shared runner and a table measured on a pinned dedicated core
are the same type, carry the same `provenance="measured"`, and select differently.

This module closes that. A calibration record is the *evidence* a table was built from, it
carries the host's own accounting of whether its declaration held, and `refusals()` says why
a record cannot support a cost table. Only an admissible record becomes an
`EncodingCostTable`.

**Why this is the same shape as `simd_hosts.py` and not a new idea.** That module already
decides admissibility for SIMD advantage records — dedicated tenancy corroborated by steal
and throttle counters, enough rounds for an order-statistic interval, every round on one CPU
— and it caught a real mis-measurement twice: once a bimodal unpinned run that looked like a
tenancy problem and was not, and once a phone whose big.LITTLE migration averaged a Cortex-X4
with an A520. A cost table has *more* to lose from those, not less: a SIMD record supports one
claim in a document, while a cost table steers `select_certified` toward one encoding over
another. Same failure modes, higher stakes, so the same gate rather than a weaker one.

**The big.LITTLE case is why this exists now.** The measured target for this phase is a
Snapdragon 8 Gen 3, whose cores are not interchangeable. A calibration averaged across a
prime core and an efficiency core describes no machine that exists, and would then be frozen,
generation-tagged, and trusted by the planner — which is worse than having no table, because
an absent table raises `UnmeasuredTarget` and a wrong one silently selects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .certified import MIN_SAMPLES, CostRow, EncodingCostTable, Interval, interval_of
from .tags import Asn1Error

#: A record's declaration that the CPU was not shared. The only tenancy a cost table may be
#: frozen from, for the reason §8 gives about shared runners: they gate validity and trend
#: evidence, never timing thresholds.
DEDICATED = "dedicated"

#: Where admitted records live, one per measured target.
STORE = "docs/measurements/asn1_calibration.json"


def calibration_corpus():
    """The ONE schema and value every target calibrates against.

    Fixed here rather than chosen per run, because a cost table's whole purpose is comparison
    — between candidates on one target, and between targets when deciding where a workload
    should run. Two targets that measured different corpora produce two tables that look
    identical, compare cleanly, and mean nothing. That failure is silent, which is why the
    corpus is a function in the repository instead of a flag on the harness.

    The shape is deliberately one of each thing the candidates price differently: a
    constrained INTEGER (PER takes its width from the constraint, DER does not), an
    unconstrained INTEGER (nothing can narrow it), a BOOLEAN (one bit in PER, a whole octet
    in DER), an ENUMERATED (PER encodes the index, everything else the value), and a
    bounded string (OER and PER read the size constraint, JER writes the characters).
    """
    from .constraints import Size, ValueRange
    from .schema import Component, Primitive, Sequence
    from .tags import Universal

    kind = Sequence((
        Component("small", Primitive(Universal.INTEGER, "INTEGER",
                                     constraint=ValueRange(0, 255))),
        Component("wide", Primitive(Universal.INTEGER, "INTEGER")),
        Component("flag", Primitive(Universal.BOOLEAN, "BOOLEAN")),
        Component("mode", Primitive(Universal.ENUMERATED, "ENUMERATED",
                                    enumeration=(("idle", 0), ("busy", 1), ("fault", 7)))),
        Component("label", Primitive(Universal.UTF8_STRING, "UTF8String",
                                     constraint=Size(ValueRange(4, 4)))),
    ), name="Calibration")
    # `mode` is the enumeration's NUMBER rather than its identifier: the format-neutral value
    # stream carries numbers, and each candidate's emitter decides what to write from them —
    # X.690 §8.4 the value, X.691 §14.1 the index, X.697 §22 the identifier. Handing it an
    # identifier would be pre-deciding one of those in the corpus.
    value = {"small": 200, "wide": 123456789, "flag": True, "mode": 1, "label": "bcir"}
    return kind, value


def corpus_digest() -> str:
    """A digest of the compiled plan and flattened stream the harness actually times.

    Of the *plan and stream*, not of the source text: two repository revisions that compile
    the same corpus to the same plan produce comparable tables and should say so, while a
    change to the plan format changes what is being measured even if the schema above is
    untouched. The digest tracks the thing that determines the octets.
    """
    import hashlib

    from .emit import flatten
    from .encode_plan import compile_encode_plan

    kind, value = calibration_corpus()
    plan = compile_encode_plan(kind, module="calibration", type_name="Calibration")
    stream = flatten(plan, value)
    serialized = plan.serialize()
    if isinstance(serialized, str):                     # the plan text, on rails that emit it
        serialized = serialized.encode("utf-8")
    payload = bytes(serialized) + b"\x00" + bytes(stream)
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateRow:
    """One candidate's measured cost on one target.

    Both axes are kept as raw samples rather than as intervals, because an interval is a
    conclusion and the store should hold the evidence. `interval_of` derives the conclusion
    the same way for this rail as for every other, so "significant" cannot come to mean two
    different things in two files.

    `decode_ns` may be empty. X.696 §6.2 denies OER a schema-free decode permanently, so
    CANONICAL-OER has a perfectly good encode number and can never have the other half — and
    a `CostRow` carries both axes. Such a row stays in the record as evidence and is excluded
    from the table, which is the same refusal-to-fabricate `native_bench` already applies.
    """

    candidate: str
    octets: int
    encode_ns: tuple[int, ...] = ()
    decode_ns: tuple[int, ...] = ()

    def complete(self) -> bool:
        return (len(self.encode_ns) >= MIN_SAMPLES and len(self.decode_ns) >= MIN_SAMPLES)

    def encode_interval(self) -> Interval:
        return interval_of(list(self.encode_ns))

    def decode_interval(self) -> Interval:
        return interval_of(list(self.decode_ns))


@dataclass(frozen=True)
class CalibrationRecord:
    """One measured target, and everything needed to decide whether to believe it."""

    target: str
    arch: str
    tenancy: str
    cal_gen: int
    rows: tuple[CandidateRow, ...] = ()
    #: Which CPU each round ran on. `(-1,)` means the host could not report it.
    cpus: tuple[int, ...] = (-1,)
    #: The host's own accounting, sampled ACROSS the measured rounds rather than before them.
    #: `None` means the host does not expose the counter, which cannot refuse a record: the
    #: check exists to catch a false declaration, not to invalidate an honest one.
    steal_ticks: int | None = None
    throttled_usec: int | None = None
    #: `corpus_digest()` as it stood when the rounds were run. Empty means a record made
    #: before the digest existed, which is reported rather than refused — the check exists to
    #: catch two targets that measured *different* corpora, and an absent digest proves
    #: nothing either way.
    corpus: str = ""
    notes: str = ""

    def refusals(self) -> tuple[str, ...]:
        """Why this record cannot support a frozen cost table, or empty when it can.

        Each entry is a reason the *numbers do not mean what they appear to mean*. A refused
        record is still evidence about a machine and worth keeping; it simply must not become
        a table the planner reads.
        """
        out: list[str] = []
        if self.tenancy == DEDICATED and self.steal_ticks:
            out.append(
                f"declared dedicated, but the CPU accumulated {self.steal_ticks} steal "
                f"tick(s) during the measured rounds: a hypervisor gave that time to another "
                f"tenant, which is what `dedicated` denies")
        if self.tenancy == DEDICATED and self.throttled_usec:
            out.append(
                f"declared dedicated, but the cgroup throttled the run for "
                f"{self.throttled_usec} us: a quota took CPU away mid-measurement, so these "
                f"samples describe the quota as much as the encoders")
        if self.tenancy != DEDICATED:
            out.append(
                f"tenancy is {self.tenancy!r}: a frozen table steers production selection, "
                f"and §8 admits a shared runner for validity and trend evidence but never "
                f"for timing thresholds")
        if len(self.cpus) > 1:
            out.append(
                f"the rounds ran on CPUs {sorted(self.cpus)}: on a big.LITTLE target that is "
                f"two machines averaged, and a table that describes no existing core would "
                f"then be frozen and trusted. Pin the run and measure again")
        if self.cpus == (-1,):
            out.append(
                "the host could not report which CPU each round ran on, so core migration is "
                "unobserved rather than absent — which is not the same thing")
        if self.cal_gen < 1:
            out.append(
                f"cal_gen is {self.cal_gen}: a frozen table is generation-tagged so a "
                f"certificate can name the calibration it read, and generation 0 names none")
        if not self.rows:
            out.append("the record carries no candidate rows, so there is nothing to freeze")
        for row in self.rows:
            if row.encode_ns and len(row.encode_ns) < MIN_SAMPLES:
                out.append(
                    f"{row.candidate}: {len(row.encode_ns)} encode rounds is below the "
                    f"{MIN_SAMPLES} an order-statistic interval needs to cover a median")
            if row.decode_ns and len(row.decode_ns) < MIN_SAMPLES:
                out.append(
                    f"{row.candidate}: {len(row.decode_ns)} decode rounds is below the "
                    f"{MIN_SAMPLES} an order-statistic interval needs to cover a median")
            if not row.encode_ns and not row.decode_ns:
                out.append(f"{row.candidate}: carries neither axis, so it measures nothing")
        if self.rows and not any(row.complete() for row in self.rows):
            out.append(
                "no candidate has BOTH axes, so no CostRow can be built and the table would "
                "be empty")
        if self.corpus and self.corpus != corpus_digest():
            out.append(
                f"measured against corpus {self.corpus} but this revision compiles "
                f"{corpus_digest()}: the numbers describe a different schema or a different "
                f"plan format, and comparing them to a current table would be comparing two "
                f"unlike measurements that happen to have the same shape")
        return tuple(out)

    def admissible(self) -> bool:
        return not self.refusals()

    def table(self) -> EncodingCostTable:
        """The frozen, generation-tagged cost table this record supports.

        Refuses outright when the record is inadmissible. Returning a table with a warning
        attached would put the decision in the caller's hands, and the caller is the planner,
        which by construction is the component that must not be deciding this.
        """
        problems = self.refusals()
        if problems:
            raise Asn1Error(
                f"calibration for {self.target!r} cannot be frozen into a cost table:\n  - "
                + "\n  - ".join(problems))
        # Never narrower than the clock. See `observed_quantum` for the aarch64 record that
        # forced this: forty-one identical samples are not a precise measurement when the
        # timer only ticks every 52 ns.
        quantum = self.observed_quantum()
        rows = tuple(
            CostRow(candidate=row.candidate, octets=row.octets,
                    encode=_at_least_resolution(row.encode_interval(), quantum),
                    decode=_at_least_resolution(row.decode_interval(), quantum))
            for row in self.rows if row.complete())
        return EncodingCostTable(target=self.target, cal_gen=self.cal_gen,
                                 provenance="measured", rows=rows)

    def observed_quantum(self) -> float:
        """The granularity this host's clock actually delivered, estimated from the samples.

        **Why this is not paranoia.** The first aarch64 calibration taken with this harness
        came back with every distinct value in the whole record — 104, 156, 208, 260, 417 —
        within 0.02 of an exact multiple of 52.083 ns, which is the period of the 19.2 MHz
        ARM architectural timer. Those figures are 2, 3, 4, 5 and 8 *timer ticks*, and the
        ±1 ns wobble on some of them is rounding. Forty-one rounds of a two-tick measurement
        all read 104 not because the cost is known to the nanosecond but because the clock
        cannot say anything else.

        An order-statistic interval over identical samples is `[104, 104]`, which is a claim
        of perfect precision. `select_certified` decides significance by `Interval.overlaps`,
        so on that record DER-encode (3 ticks) and JER-encode (4 ticks) are "significantly
        different" on the strength of one unit of clock resolution. `table()` widens by this
        figure so that cannot happen.

        Estimated by fitting rather than assumed, because the answer is a property of the
        host: the largest quantum that explains every observed value to within 5% of a tick.
        A host with a fine clock returns ~1 ns and nothing is widened, which is what the x86
        container does.
        """
        values = sorted({v for row in self.rows
                         for axis in (row.encode_ns, row.decode_ns) for v in axis})
        if len(values) < 2:
            return 1.0
        best = 1.0
        # Hundredths of a nanosecond, up to 200 ns. Wide enough for the 19.2 MHz timer and
        # anything slower a phone or an embedded target is likely to expose.
        for hundredths in range(100, 20001):
            quantum = hundredths / 100.0
            worst = max(min(v % quantum, quantum - (v % quantum)) / quantum for v in values)
            if worst <= 0.05:
                best = quantum
        return best

    def incomplete(self) -> tuple[str, ...]:
        """Candidates measured on one axis only, with why the other is missing being a
        property of the standard rather than of this run. Reported, never filled in."""
        return tuple(row.candidate for row in self.rows if not row.complete())


def _at_least_resolution(interval: Interval, quantum: float) -> Interval:
    """Widen an interval that is narrower than the clock that produced it.

    An interval is a claim about where the true median lies. When every sample reads the same
    tick, the order statistic reports `[v, v]` — and the honest statement is that the value
    lies somewhere within one tick of `v`. Widening symmetrically says that, and leaves an
    already-wider interval untouched: this corrects a false precision, it does not inflate a
    real measurement.

    `low` and `high` stop being observed samples here, which is a real cost — `Interval`'s
    docstring notes that being observed is what makes it reportable without a model. The
    trade is deliberate: a bound that is honest about the instrument beats a bound that is
    literally a sample and overstates what the instrument could see.
    """
    if quantum <= 1.0 or (interval.high - interval.low) >= quantum:
        return interval
    half = quantum / 2.0
    return Interval(low=max(0, int(interval.median - half)),
                    high=int(interval.median + half + 0.5),
                    median=interval.median, samples=interval.samples,
                    coverage_ppm=interval.coverage_ppm)


_REQUIRED = ("target", "arch", "tenancy", "cal_gen", "rows")


def load_records(path: str) -> list[CalibrationRecord]:
    """Read the store, refusing a record that is missing a field rather than defaulting it.

    A defaulted `tenancy` would read as a declaration nobody made, and the whole point of the
    field is that somebody declared it and the counters can contradict them.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("targets")
    if not isinstance(raw, list):
        raise Asn1Error(f"{path}: expected a top-level \"targets\" list")
    records: list[CalibrationRecord] = []
    for entry in raw:
        missing = [name for name in _REQUIRED if name not in entry]
        if missing:
            raise Asn1Error(
                f"{path}: a calibration record is missing {missing}; every one is a claim "
                f"about the measurement that cannot be defaulted")
        rows = tuple(
            CandidateRow(candidate=row["candidate"], octets=int(row["octets"]),
                         encode_ns=tuple(int(v) for v in row.get("encode_ns", ())),
                         decode_ns=tuple(int(v) for v in row.get("decode_ns", ())))
            for row in entry["rows"])
        records.append(CalibrationRecord(
            target=entry["target"], arch=entry["arch"], tenancy=entry["tenancy"],
            cal_gen=int(entry["cal_gen"]), rows=rows,
            cpus=tuple(int(c) for c in entry.get("cpus", (-1,))),
            steal_ticks=entry.get("steal_ticks"),
            throttled_usec=entry.get("throttled_usec"),
            corpus=entry.get("corpus", ""),
            notes=entry.get("notes", "")))
    return records


def render(records: list[CalibrationRecord]) -> str:
    """A report that states refusals as loudly as admissions.

    A store listing only what it admitted would hide the interesting half: a refused record
    is how a reader learns that a target was measured and why the numbers were not used.
    """
    lines: list[str] = []
    for record in records:
        verdict = "ADMITTED" if record.admissible() else "REFUSED"
        quantum = record.observed_quantum()
        clock = ""
        if quantum > 1.0:
            # Say it here, because the intervals below are the raw evidence and several of
            # them will be a single repeated value. That is the clock talking, not certainty.
            clock = (f"  clock ~{quantum:.1f}ns ({1e3 / quantum:.1f} MHz); intervals below "
                     f"are RAW and table() widens to this")
        lines.append(f"{record.target} [{record.arch}] cal_gen={record.cal_gen} {verdict}"
                     + clock)
        for problem in record.refusals():
            lines.append(f"    refused: {problem}")
        for row in record.rows:
            axes = []
            if row.encode_ns:
                axes.append(f"encode {row.encode_interval()}")
            if row.decode_ns:
                axes.append(f"decode {row.decode_interval()}")
            note = "" if row.complete() else "   (one axis only; excluded from the table)"
            lines.append(f"    {row.candidate:26} {row.octets:5}B  "
                         + "  ".join(axes) + note)
    return "\n".join(lines)


def measure(*, target: str, arch: str, tenancy: str, cal_gen: int, cpus, steal_ticks,
            throttled_usec, notes: str = "", **kwargs) -> CalibrationRecord:
    """Run both native benches over `calibration_corpus()` and return the record.

    The harness is `native_bench`'s, not a second one. A calibration measured by different
    code from the code the repository already trusts would be a second definition of "what a
    decode costs", which is the same mistake this project refuses one level down in the SIMD
    rail.
    """
    from .native_bench import run_native_bench, run_native_encode_bench

    kind, value = calibration_corpus()
    decode_samples, _skipped = run_native_bench(kind, value, **kwargs)
    try:
        encode_samples, _ = run_native_encode_bench(kind, value, **kwargs)
    except Asn1Error:
        encode_samples = {}

    # Wire size comes from the ORACLE's encoders, not from the decode bench.
    #
    # The bench only sizes what it can decode, which would leave PER and OER — encodable, and
    # permanently undecodable without a schema (X.691 §7.2, X.696 §6.2) — reporting zero
    # octets. Zero is not a small size, it is a wrong one, and it is on the axis a bandwidth
    # objective selects by. The oracle's figure is exact arithmetic over the same abstract
    # value, so it is host-independent and identical on every target, which is what a size
    # axis should be.
    from .selection import ALL_CANDIDATES

    octets: dict[str, int] = {}
    for candidate in ALL_CANDIDATES:
        try:
            octets[candidate.name] = len(candidate.encode(kind, value))
        except Asn1Error:
            continue                      # a rule that refuses this schema has no size here
    decode = {sample.candidate: tuple(sample.decode_ns) for sample in decode_samples}
    rows = tuple(
        CandidateRow(candidate=name, octets=octets.get(name, 0),
                     encode_ns=tuple(encode_samples.get(name, ())),
                     decode_ns=decode.get(name, ()))
        for name in sorted(set(decode) | set(encode_samples)))
    return CalibrationRecord(
        target=target, arch=arch, tenancy=tenancy, cal_gen=cal_gen, rows=rows,
        cpus=tuple(cpus), steal_ticks=steal_ticks, throttled_usec=throttled_usec,
        corpus=corpus_digest(), notes=notes)


def as_json(record: CalibrationRecord) -> str:
    """A record as the store spells it, ready to paste into `STORE`."""
    return json.dumps({
        "target": record.target, "arch": record.arch, "tenancy": record.tenancy,
        "cal_gen": record.cal_gen, "cpus": list(record.cpus),
        "steal_ticks": record.steal_ticks, "throttled_usec": record.throttled_usec,
        "corpus": record.corpus, "notes": record.notes,
        "rows": [{"candidate": r.candidate, "octets": r.octets,
                  "encode_ns": list(r.encode_ns), "decode_ns": list(r.decode_ns)}
                 for r in record.rows],
    }, indent=2, sort_keys=True)


def _main(argv: list[str] | None = None) -> int:
    """`python3 -m bcir.asn1.calibration` reports the store; `--measure` produces a record."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", action="store_true",
                        help="run the benches here and print a record for the store")
    parser.add_argument("--target", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--tenancy", default=DEDICATED)
    parser.add_argument("--cal-gen", type=int, default=1)
    parser.add_argument("--cpus", default="-1", help="comma-separated CPUs the rounds ran on")
    parser.add_argument("--steal-ticks", type=int, default=None)
    parser.add_argument("--throttled-usec", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=MIN_SAMPLES + 4)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    if not args.measure:
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(root, STORE)
        if not os.path.exists(path):
            print(f"no calibration store at {STORE}; run --measure on a declared target")
            return 0
        records = load_records(path)
        print(f"corpus {corpus_digest()}")
        print(render(records))
        return 0 if all(r.admissible() for r in records) else 1

    if not args.target:
        parser.error("--measure needs --target: \"linux\" is not a target, and the record "
                     "is a claim about a specific machine")
    import platform
    record = measure(target=args.target, arch=args.arch or platform.machine(),
                     tenancy=args.tenancy, cal_gen=args.cal_gen,
                     cpus=[int(c) for c in args.cpus.split(",") if c != ""],
                     steal_ticks=args.steal_ticks, throttled_usec=args.throttled_usec,
                     notes=args.notes, rounds=args.rounds, iterations=args.iterations)
    print(as_json(record))
    for problem in record.refusals():
        print(f"# refused: {problem}", flush=True)
    return 0


if __name__ == "__main__":                                    # pragma: no cover - CLI
    raise SystemExit(_main())


__all__ = [
    "DEDICATED", "STORE", "CalibrationRecord", "CandidateRow", "as_json",
    "calibration_corpus", "corpus_digest", "load_records", "measure", "render",
]
