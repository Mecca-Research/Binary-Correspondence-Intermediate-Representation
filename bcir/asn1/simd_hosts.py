"""J5's advantage clause: what makes a measured host count, and when two of them close it.

J5's gate asks for a *"statistically significant measured advantage on at least two hosts"*.
The correctness half has been met since the rail landed — every tier returns an identical
status and byte offset to the scalar rail, on x86-64 and on the aarch64 CI lane alike. This
module is about the other half, which is not a matter of running the benchmark twice.

**Tenancy is declared, and then checked against the machine's own accounting.** The first
version took `dedicated` on trust, which made the most important field the least verifiable
one — and led to this repository's own container being labelled `shared` on the strength of
*what kind of machine it is* rather than what it was doing. A record now carries the
hypervisor steal ticks and cgroup throttling accumulated **during the measured rounds**, and
either being nonzero refuses a `dedicated` claim. That can only catch a false declaration: a
host that does not report the counters is not penalised for it.

**§8 refuses a timing threshold on a shared runner** — *"shared CI gates validity and trend
evidence, not noisy timing thresholds"* — and admits SIMD *"on a declared target"*. Those two
sentences decide the whole design here: a measurement is evidence only if the machine it came
from is *named*, and only if that machine was not being shared with someone else's build. So
a host record carries its own admissibility, and `two_host_verdict` counts admissible hosts
rather than measurements. Adding a second CI lane can never close this clause, however many
numbers it produces.

**The advantage itself is decided by non-overlapping intervals, not by a ratio.** `certified`
already builds distribution-free order-statistic intervals for exactly this reason: a median
speedup of "2.4×" says nothing without a spread, and a normal-theory interval would assume a
distribution timing data does not have. Two intervals that do not overlap is a claim that
survives a reader checking it.

**The first dedicated aarch64 host available is a phone, and that changes what must be
recorded.** A Snapdragon 8 Gen 3 is big.LITTLE — one Cortex-X4, four A720s, three A520s — and
the same code on the largest and the smallest core differs by more than the advantage being
measured. Three consequences, all of them recorded rather than assumed away:

1. **Which CPU each round ran on is evidence.** The bench driver reports it per round, and a
   record whose rounds span more than one CPU is refused: it is two machines averaged.
2. **A run that migrates refuses itself even without that check**, which is the reassuring
   part. Migration makes the samples bimodal, the order-statistic interval widens to span
   both modes, and `overlaps` then reports no advantage. The protocol degrades to *unproven*
   rather than to *wrong* — and a test pins that, because a safety property nobody checks is
   a hope.
3. **Thermal throttling is monotone drift, and the interleaved round-robin already handles
   it**: alternating scalar and vector rounds spreads a downward frequency ramp evenly across
   both, so it becomes noise in each rather than a bias in one. That is the same argument
   `bcir_asn1_bench.c` makes for its own interleaving, and it is why the runbook interleaves
   rather than running all scalar rounds and then all vector rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .certified import MIN_SAMPLES, Interval, interval_of
from .tags import Asn1Error

#: A host that shares its CPU with other tenants cannot support a timing claim, whatever the
#: numbers say. Named rather than inferred: "is this machine shared" is a fact about the
#: procurement, not about the samples, and guessing it from variance would be exactly the
#: kind of inference §8 asks to be replaced with declared evidence.
SHARED = "shared"
#: A machine whose CPU is not contended by another tenant for the duration of the run.
DEDICATED = "dedicated"


@dataclass(frozen=True)
class HostRecord:
    """One machine's measurement of the SIMD rail against the scalar rail.

    Everything here is *declared* by whoever ran the measurement. That is deliberate: a
    record is a claim someone makes about a machine, and the reader's job is to say whether
    the claim is admissible — not to reconstruct the machine from its timings.
    """

    #: A name a reader can look up. "Samsung S24+ (SM-S926B), Snapdragon 8 Gen 3" is a host;
    #: "linux" is not.
    host: str
    #: uname -m. The clause wants two hosts; two ARCHITECTURES is what makes them different
    #: machines rather than the same machine twice.
    arch: str
    #: SHARED or DEDICATED.
    tenancy: str
    #: The tier `auto` resolved to, by name. A record whose tier is `scalar` measures the
    #: scalar rail against itself and proves nothing.
    tier: str
    #: Per-round medians, nanoseconds, for the scalar rail and for the resolved tier.
    scalar_ns: tuple[int, ...]
    vector_ns: tuple[int, ...]
    #: The distinct CPUs the rounds ran on, from `sched_getcpu`. `(-1,)` means the host
    #: could not say, which is UNKNOWN rather than clean.
    cpus: tuple[int, ...] = (-1,)
    #: Free text: thermal state, whether the run was pinned, anything a reader would want.
    notes: str = ""
    #: Hypervisor steal ticks accumulated DURING the measured rounds, from `/proc/stat`.
    #: `None` where the host does not report it.
    steal_ticks: int | None = None
    #: cgroup CPU throttling accumulated during the rounds, microseconds. `None` where the
    #: host does not report it.
    throttled_usec: int | None = None

    def scalar_interval(self) -> Interval:
        return interval_of(list(self.scalar_ns))

    def vector_interval(self) -> Interval:
        return interval_of(list(self.vector_ns))

    def refusals(self) -> tuple[str, ...]:
        """Why this record cannot support a timing claim, or empty when it can.

        Every one of these is a reason the *numbers do not mean what they appear to mean*,
        rather than a reason they are unwelcome. A record that fails here is still worth
        keeping — it is evidence about a machine — it simply does not count toward the gate.
        """
        out: list[str] = []
        # `dedicated` is a claim about the machine, and the machine keeps its own accounting
        # of whether that claim held. Steal counts time a hypervisor gave to another tenant;
        # cgroup throttling counts time a quota took away. Either being nonzero DURING the
        # measured rounds contradicts the declaration, so the record is refused on the host's
        # own evidence rather than on anybody's opinion of what kind of machine it is.
        #
        # `None` means the host does not report the counter, which does not refuse: the check
        # can only ever catch a false declaration, never invalidate an honest record made
        # before the counter was collected.
        if self.tenancy == DEDICATED and self.steal_ticks:
            out.append(
                f"declared dedicated, but the CPU accumulated {self.steal_ticks} steal "
                f"tick(s) during the measured rounds: a hypervisor gave that time to another "
                f"tenant, which is what `dedicated` denies")
        if self.tenancy == DEDICATED and self.throttled_usec:
            out.append(
                f"declared dedicated, but the cgroup throttled the run for "
                f"{self.throttled_usec} us: a quota took CPU away mid-measurement, so the "
                f"samples describe the quota as much as the code")
        if self.tenancy != DEDICATED:
            out.append(
                f"tenancy is {self.tenancy!r}: §8 admits SIMD on a declared target and "
                f"refuses timing thresholds on a shared runner, so a contended CPU cannot "
                f"support an advantage claim however clean its samples look")
        if self.tier == "scalar":
            out.append(
                "the resolved tier is `scalar`, so this measures the scalar rail against "
                "itself; the build had no vector tier for this CPU")
        for name, samples in (("scalar", self.scalar_ns), ("vector", self.vector_ns)):
            if len(samples) < MIN_SAMPLES:
                out.append(
                    f"{name} has {len(samples)} rounds, below the {MIN_SAMPLES} an "
                    f"order-statistic interval needs to cover a median at all")
        if len(self.cpus) > 1:
            out.append(
                f"the rounds ran on CPUs {sorted(self.cpus)}: on a big.LITTLE host that is "
                f"two machines averaged, and the difference between a Cortex-X4 and an A520 "
                f"exceeds the advantage being measured. Pin the run (`taskset -c N`) and "
                f"measure again")
        if self.cpus == (-1,):
            out.append(
                "the host could not report which CPU each round ran on, so core migration "
                "is unobserved rather than absent — which is not the same thing")
        return tuple(out)

    def admissible(self) -> bool:
        return not self.refusals()

    def shows_advantage(self) -> bool:
        """Whether the vector interval is strictly below the scalar interval, disjointly.

        Not a ratio. Two intervals that do not overlap is a claim a reader can check; a
        median speedup with no spread beside it is a number that happens to be true of one
        run. `Interval.overlaps` is `certified`'s, so this rail and the encoding-choice rail
        mean the same thing by "significant".
        """
        scalar, vector = self.scalar_interval(), self.vector_interval()
        return not scalar.overlaps(vector) and vector.high < scalar.low

    def speedup(self) -> float:
        """The headline ratio, for a report. Never the thing being decided."""
        return self.scalar_interval().median / max(1.0, self.vector_interval().median)


@dataclass(frozen=True)
class Verdict:
    """The clause's state, with the reason attached rather than implied."""

    met: bool
    admitted: tuple[str, ...] = ()
    rejected: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reason: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def two_host_verdict(records: list[HostRecord]) -> Verdict:
    """Whether J5's advantage clause is met by these records.

    Two admissible hosts, each showing a disjoint advantage, and **on different
    architectures**. The last requirement is the one worth spelling out: the clause exists
    because a vector rail can be fast on the ISA it was written for and a wash on another,
    so two x86 boxes are the same evidence twice. `bcir_jer_simd` compiles SSE2, AVX2 and
    NEON from one source, and NEON is the path no x86 host exercises at all.
    """
    admitted: list[HostRecord] = []
    rejected: list[tuple[str, tuple[str, ...]]] = []
    for record in records:
        why = record.refusals()
        if why:
            rejected.append((record.host, why))
            continue
        if not record.shows_advantage():
            rejected.append((record.host, (
                f"the intervals overlap: scalar {record.scalar_interval()!r} against vector "
                f"{record.vector_interval()!r}. That is 'no advantage demonstrated', not "
                f"'advantage too small' — a wider run may separate them",)))
            continue
        admitted.append(record)

    architectures = {record.arch for record in admitted}
    if len(admitted) < 2:
        return Verdict(
            met=False, admitted=tuple(r.host for r in admitted), rejected=tuple(rejected),
            reason=f"{len(admitted)} admissible host(s) with a demonstrated advantage; the "
                   f"clause asks for at least two")
    if len(architectures) < 2:
        return Verdict(
            met=False, admitted=tuple(r.host for r in admitted), rejected=tuple(rejected),
            reason=f"all {len(admitted)} admissible hosts are {sorted(architectures)}; a "
                   f"vector rail fast on one ISA says nothing about another, so two hosts of "
                   f"the same architecture are the same evidence twice")
    return Verdict(
        met=True, admitted=tuple(r.host for r in admitted), rejected=tuple(rejected),
        reason=f"{len(admitted)} dedicated hosts across {sorted(architectures)}, each with a "
               f"vector interval strictly below its scalar interval")


# --- the record store ---------------------------------------------------------------------


def load_records(path: str) -> list[HostRecord]:
    """Read the measurement store. Missing fields are a refusal, never a default.

    A record with no `tenancy` would otherwise silently become whatever the dataclass
    default is, and the whole point of the field is that somebody stated it.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("hosts"), list):
        raise Asn1Error(f"{path}: a measurement store is {{'hosts': [...]}}")
    out: list[HostRecord] = []
    for index, entry in enumerate(raw["hosts"]):
        missing = [name for name in ("host", "arch", "tenancy", "tier", "scalar_ns",
                                     "vector_ns") if name not in entry]
        if missing:
            raise Asn1Error(f"{path}: host {index} is missing {missing}")
        out.append(HostRecord(
            host=str(entry["host"]), arch=str(entry["arch"]),
            tenancy=str(entry["tenancy"]), tier=str(entry["tier"]),
            scalar_ns=tuple(int(value) for value in entry["scalar_ns"]),
            vector_ns=tuple(int(value) for value in entry["vector_ns"]),
            cpus=tuple(int(value) for value in entry.get("cpus", (-1,))),
            notes=str(entry.get("notes", "")),
            steal_ticks=(None if entry.get("steal_ticks") is None
                         else int(entry["steal_ticks"])),
            throttled_usec=(None if entry.get("throttled_usec") is None
                            else int(entry["throttled_usec"]))))
    return out


def render(records: list[HostRecord]) -> str:
    """A human-readable report, for pasting into §7.3."""
    verdict = two_host_verdict(records)
    lines = [f"J5 advantage clause: {'MET' if verdict.met else 'UNMET'} — {verdict.reason}",
             ""]
    for record in records:
        why = record.refusals()
        state = "admitted" if not why else "not admissible"
        lines.append(f"  {record.host} [{record.arch}, tier {record.tier}] — {state}")
        if not why:
            lines.append(f"    scalar {record.scalar_interval()!r}")
            lines.append(f"    vector {record.vector_interval()!r}")
            lines.append(f"    speedup {record.speedup():.2f}x, "
                         f"advantage {'shown' if record.shows_advantage() else 'NOT shown'}")
        for reason in why:
            lines.append(f"    - {reason}")
        if record.notes:
            lines.append(f"    note: {record.notes}")
    return "\n".join(lines)


__all__ = ["DEDICATED", "SHARED", "HostRecord", "Verdict", "load_records", "render",
           "two_host_verdict"]


#: Where the records live. A path rather than embedded data: a measurement is evidence
#: somebody produced on a machine, and it should be reviewable in a diff on its own.
STORE = "docs/measurements/jer_simd_hosts.json"


def _main() -> int:
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(render(load_records(os.path.join(root, STORE))))
    return 0


if __name__ == "__main__":  # pragma: no cover - a reporting entry point
    raise SystemExit(_main())
