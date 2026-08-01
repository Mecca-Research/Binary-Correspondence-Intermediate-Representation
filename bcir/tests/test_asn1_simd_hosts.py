"""J5's advantage clause: what counts as a measured host, and why.

The correctness half of J5's gate has been met since the rail landed. This is the other
half — *"a statistically significant measured advantage on at least two hosts"* — and the
tests here are about **admissibility**, not about speed. A number from the wrong kind of
machine is not a weak claim, it is a claim about something else.

The rules come from §8: SIMD is admitted *"on a declared target"*, and *"shared CI gates
validity and trend evidence, not noisy timing thresholds"*. So a shared runner cannot close
this clause however many rounds it runs, and adding a second CI lane never will.
"""

from __future__ import annotations

import json
import os
import tempfile

from bcir.asn1.certified import MIN_SAMPLES
from bcir.asn1.simd_hosts import (
    DEDICATED, SHARED, STORE, HostRecord, load_records, render, two_host_verdict,
)
from bcir.asn1.tags import Asn1Error

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _record(**overrides) -> HostRecord:
    """A clean, admissible record showing a large advantage. Overridden per test."""
    base = {
        "host": "a dedicated rig", "arch": "aarch64", "tenancy": DEDICATED, "tier": "neon",
        "scalar_ns": tuple([28000 + (index % 7) * 20 for index in range(41)]),
        "vector_ns": tuple([1100 + (index % 7) * 5 for index in range(41)]),
        "cpus": (7,),
    }
    base.update(overrides)
    return HostRecord(**base)


def test_a_shared_runner_cannot_close_the_clause_however_clean_its_numbers():
    """The rule that decides the whole design, from §8.

    A contended CPU produces samples that look like a machine's but are partly somebody
    else's build. §8 refuses timing thresholds there, so tenancy is checked *before* the
    samples are looked at — and a record with a perfect 25× separation is still refused.
    """
    shared = _record(tenancy=SHARED)
    assert shared.shows_advantage(), "the fixture should show a large, clean advantage"
    assert not shared.admissible()
    assert any("shared runner" in reason for reason in shared.refusals())
    # Two of them are still zero admissible hosts, which is the point: this clause cannot be
    # closed by running the benchmark more times on more shared machines.
    verdict = two_host_verdict([shared, _record(host="another shared rig", tenancy=SHARED)])
    assert not verdict.met
    assert verdict.admitted == ()


def test_two_hosts_of_the_same_architecture_are_the_same_evidence_twice():
    """Why the clause says *hosts* and this reads it as *architectures*.

    A vector rail can be fast on the ISA it was written for and a wash on another —
    `bcir_jer_simd` compiles SSE2, AVX2 and NEON from one source, and NEON is the path no
    x86 host exercises at all. Two x86 boxes agreeing says nothing about the third.
    """
    verdict = two_host_verdict([
        _record(host="rig A", arch="x86_64", tier="avx2"),
        _record(host="rig B", arch="x86_64", tier="avx2"),
    ])
    assert not verdict.met
    assert "same evidence twice" in verdict.reason
    assert set(verdict.admitted) == {"rig A", "rig B"}, "both are admissible individually"

    # One of each closes it.
    verdict = two_host_verdict([
        _record(host="rig A", arch="x86_64", tier="avx2"),
        _record(host="a phone", arch="aarch64", tier="neon"),
    ])
    assert verdict.met, verdict.reason
    assert set(verdict.admitted) == {"rig A", "a phone"}


def test_a_run_that_wandered_between_cores_is_two_machines_averaged():
    """The failure a phone introduces that a workstation does not.

    A Snapdragon 8 Gen 3 is big.LITTLE — one Cortex-X4, four A720s, three A520s — and the
    same code on the largest and the smallest core differs by more than the advantage being
    measured. A record whose rounds span more than one CPU is therefore refused rather than
    averaged, and the refusal names the fix.
    """
    wandered = _record(cpus=(0, 7))
    assert not wandered.admissible()
    assert any("taskset" in reason for reason in wandered.refusals())


def test_an_unknown_cpu_does_not_read_as_a_clean_one():
    """`-1` means the host could not say, which is not the same as "did not migrate".

    The bench driver reports `-1` off Linux, where `sched_getcpu` does not exist. Letting
    that pass would make the strongest-looking records the ones from the platforms that can
    say least about themselves.
    """
    unknown = _record(cpus=(-1,))
    assert not unknown.admissible()
    assert any("unobserved rather than absent" in reason for reason in unknown.refusals())


def test_core_migration_refuses_itself_even_when_the_cpu_field_is_missing():
    """The reassuring property: the statistics catch what the metadata would have caught.

    Migration makes the samples bimodal — half the rounds on a fast core, half on a slow
    one. An order-statistic interval then spans both modes, so the scalar and vector
    intervals overlap and `shows_advantage` reports none.

    That is the protocol degrading to **unproven** rather than to **wrong**, and it holds
    without the CPU field being present at all. It is asserted here because a safety
    property nobody checks is a hope.
    """
    fast, slow = 1100, 30000
    bimodal = tuple([fast if index % 2 else slow for index in range(41)])
    migrated = _record(vector_ns=bimodal, scalar_ns=bimodal)
    assert not migrated.shows_advantage()
    # And with the scalar rail steady, a migrating vector rail still cannot claim a win.
    half_migrated = _record(vector_ns=bimodal)
    assert not half_migrated.shows_advantage(), (
        f"scalar {half_migrated.scalar_interval()!r} vs vector "
        f"{half_migrated.vector_interval()!r}")


def test_the_advantage_is_disjoint_intervals_and_not_a_median_ratio():
    """A median comparison always names a winner; this reports when there is not one."""
    # Two candidates whose spreads overlap heavily: the medians differ, the intervals do not
    # separate, and the honest answer is "no advantage demonstrated".
    noisy = _record(
        scalar_ns=tuple([1000 + (index * 37) % 400 for index in range(41)]),
        vector_ns=tuple([950 + (index * 41) % 400 for index in range(41)]))
    assert noisy.scalar_interval().median > noisy.vector_interval().median
    assert not noisy.shows_advantage()
    verdict = two_host_verdict([noisy, _record(host="p", arch="aarch64")])
    assert not verdict.met
    assert any("intervals overlap" in reason
               for _host, reasons in verdict.rejected for reason in reasons)


def test_too_few_rounds_is_refused_rather_than_interpolated():
    """An order-statistic interval below `MIN_SAMPLES` covers almost nothing."""
    thin = _record(scalar_ns=(28000,) * (MIN_SAMPLES - 1),
                   vector_ns=(1100,) * (MIN_SAMPLES - 1))
    assert not thin.admissible()
    assert any(str(MIN_SAMPLES) in reason for reason in thin.refusals())


def test_a_scalar_only_build_cannot_measure_itself_against_itself():
    """On a CPU with no vector tier, `auto` resolves to scalar and there is nothing to time."""
    scalar_only = _record(tier="scalar")
    assert not scalar_only.admissible()
    assert any("against itself" in reason for reason in scalar_only.refusals())


def test_the_store_parses_and_a_missing_field_is_a_refusal():
    """A record with no `tenancy` must not silently become the dataclass default."""
    records = load_records(os.path.join(_ROOT, STORE))
    assert records, "the store should carry at least the host that produced it"
    for record in records:
        assert record.host and record.arch and record.tenancy
    assert render(records)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"hosts": [{"host": "x", "arch": "aarch64"}]}, handle)
        try:
            load_records(path)
        except Asn1Error as error:
            assert "tenancy" in str(error), error
        else:
            raise AssertionError("a record missing required fields loaded")


def test_the_recorded_clause_state_matches_what_the_store_actually_supports():
    """The roadmap's §7.3 and this store must agree, and the store is the authority.

    §7.3 says the clause is unmet. If a dedicated aarch64 record ever lands and closes it,
    this fails — which is the point: the prose is then stale and must be updated with the
    evidence rather than left claiming less than the repository can show.
    """
    verdict = two_host_verdict(load_records(os.path.join(_ROOT, STORE)))
    roadmap = os.path.join(_ROOT, "docs", "BCIR_ASN1_JSON_ROADMAP.md")
    with open(roadmap, encoding="utf-8") as handle:
        text = handle.read()
    if verdict.met:
        raise AssertionError(
            f"the store now closes J5's advantage clause ({verdict.reason}) — update §7.3 "
            f"and the J5 row, which still record it as unmet")
    assert "two-host clause is not met" in text.lower(), (
        "§7.3 no longer records the clause as unmet, but the store does not close it")
    # The J5 row spells the tally as prose, and prose drifts. Pinning the COUNT means adding
    # a host record cannot quietly leave the summary row claiming less — or more — than the
    # evidence supports.
    tally = f"UNMET at {len(verdict.admitted)} of 2"
    assert tally in text, (
        f"the J5 row does not say {tally!r}; the store admits "
        f"{len(verdict.admitted)} host(s) ({list(verdict.admitted)}) and the row must match")
