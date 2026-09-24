"""The data-plane hand-off (GEM+ roadmap G16, staged plan S3-C).

Before this slice the C++ seam took an artifact as a raw (pointer, length) pair: `admit()` gated
nothing by default and, handed numbers, compared only the header maxima; `dispatch()` never asked
it; a view that outlived its buffer read whatever the buffer held next (or freed memory); the
dynamic-graph backend was a stub and a pack too large for one node had no shipping format. These
tests pin the G16 gates and the laws under them:

    no copy where a borrow suffices   the artifact is written once into a slot and read in place;
                                      a view that outlives its owner is refused (BCIR_ERR_LIFETIME)
    generation gating                 admit() is the LIVE plane's predicate against the installed
                                      registry; dispatch runs only what was admitted at the resident
                                      generation of that registry, as a phase of the plane
    shard manifest                    shards named by digest reassemble to the whole pack's bytes,
                                      and each shard is a pack a node admits and runs by itself

The corpora are declared once, with the outcome the specification requires, in
`bcir/tests/handoff_fixtures.py`. The native tests build `runtime/c/test_handoff.c` and
`runtime/cpp/test_handoff.cpp` and skip without a compiler (the quick tier hides one on purpose) or
outside a source checkout (the wheel ships neither runtime tree).
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import tempfile
from dataclasses import replace
from unittest import mock

from bcir.abi import decode, encode
from bcir.abi.shard_manifest import (
    BLOB_MAX,
    MANIFEST_ENTRY_SIZE,
    MANIFEST_FIXED,
    MANIFEST_HEADER_SIZE,
    SHARDS_MAX,
    ShardError,
    decode_manifest,
    encode_manifest,
    frame_of,
    hydrated_layout,
    partition,
    reassemble,
    split,
    sub_pack,
)
from bcir.gem.control import ControlPlane
from bcir.gem.handoff import (
    EPOCH_MAX,
    HO_REFUSALS,
    SLOT_STATES,
    FreezeError,
    GraphClaim,
    GraphGeneration,
    Handle,
    PackTable,
    admit_manifest,
    freeze_claims,
)
from bcir.tests import control_fixtures as cf
from bcir.tests import handoff_fixtures as hf

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _plane(registry=None) -> ControlPlane:
    plane = ControlPlane(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT)
    for step in hf.boot(registry or cf.artifacts().registry):
        assert plane.submit(step.data).applied
    return plane


def _stored(table: PackTable, data: bytes) -> Handle:
    h = table.reserve().handle
    assert table.write(h, 0, data).applied and table.commit(h, len(data)).applied
    return h


def _status(callback) -> str:
    try:
        callback()
    except (ShardError, FreezeError) as exc:
        return exc.status
    return "BCIR_OK"


# --- explicit lifetime ---------------------------------------------------------------------------


def test_a_view_that_outlives_its_owner_is_refused_never_read():
    table = PackTable(1, 4096)
    art = cf.artifacts()
    h = _stored(table, art.pack)
    assert table.release(h).applied
    for outcome in (table.borrow(h)[0], table.release(h), table.admit(h, _plane()),
                    table.dispatch(h, _plane())):  # fmt: skip
        assert (outcome.verdict, outcome.refusal, outcome.status) == (
            "refused", "lifetime", "BCIR_ERR_LIFETIME")  # fmt: skip
    # the next incarnation reuses the slot one epoch on; the old handle still names nothing
    h2 = _stored(table, art.pack_old)
    assert (h2.index, h2.epoch) == (h.index, h.epoch + 1)
    assert table.borrow(h)[0].refusal == "lifetime"
    assert bytes(table.borrow(h2)[1].data) == art.pack_old


def test_a_borrow_in_progress_keeps_its_bytes_and_its_slot():
    table = PackTable(1, 4096)
    data = cf.artifacts().pack
    h = _stored(table, data)
    out, view = table.borrow(h)
    assert out.applied and table.release(h).applied
    assert table.slots[0].state == "retired" and bytes(view.data) == data
    assert table.reserve().refusal == "full"  # never reused under a reader
    assert table.give_back(view).applied and table.slots[0].state == "free"
    assert table.give_back(view).refusal == "lifetime"  # a return is not repeatable
    assert table.reserve().applied


def test_the_table_s_bounds_refuse_rather_than_wrap():
    table = PackTable(1, 64)
    h = table.reserve().handle
    assert table.write(h, 60, b"12345").refusal == "capacity"
    assert table.commit(h, 0).refusal == "capacity" and table.commit(h, 65).refusal == "capacity"
    assert table.commit(h, 64).applied
    table.slots[0].epoch = EPOCH_MAX  # 2^32 incarnations later
    h = Handle(0, EPOCH_MAX)
    out, view = table.borrow(h)
    assert table.release(h).applied and table.slots[0].state == "exhausted"
    assert table.give_back(view).applied and table.slots[0].state == "exhausted"
    assert table.reserve().refusal == "full"  # an epoch never wraps: out of service for good
    assert table.give_back(Handle(0, 0)).refusal == "lifetime"  # epoch 0 is never issued
    assert set(SLOT_STATES) == {"free", "building", "frozen", "retired", "exhausted"}


def test_a_refused_producer_step_publishes_nothing():
    table = PackTable(2, 1024)
    too_big = [hf.claim(i, hf.R_ADD, (1,), (2,)) for i in range(1, 60)]
    out = table.freeze(too_big, hf.BUILDER_GENS, 1)
    assert (out.refusal, out.status) == ("capacity", "BCIR_ERR_NOSPACE")
    assert all(s.state == "free" for s in table.slots) and table.slots[0].epoch == 2
    out = table.freeze([hf.claim(1, lane=6)], hf.BUILDER_GENS, 1)
    assert (out.refusal, out.status) == ("malformed", "BCIR_ERR_LANE")
    assert all(s.state == "free" for s in table.slots)


# --- generation gating ---------------------------------------------------------------------------


def test_admission_is_the_live_plane_s_predicate():
    """Every way a pack can be older than the live registry is refused at admit and never runs;
    the fresh ones are admitted and run."""
    for case in hf.admission_cases():
        if case.switch is not None:
            continue
        table = PackTable(2, 8192)
        plane = _plane(case.registry)
        h = _stored(table, case.pack)
        assert table.dispatch(h, plane).refusal == "unadmitted", case.label
        admitted = table.admit(h, plane)
        assert admitted.applied != case.stale, case.label
        ran = table.dispatch(h, plane)
        assert ran.applied != case.stale, case.label
        if not case.stale:
            assert ran.claims == tuple(s.claim_id for s in decode(case.pack).segments)


def test_an_admission_lives_within_one_generation():
    window = hf.admission_cases()[-1]
    table = PackTable(1, 8192)
    plane = _plane(window.registry)
    h = _stored(table, window.pack)
    assert table.admit(h, plane).applied and table.dispatch(h, plane).applied
    assert plane.submit(cf.generation_record(window.switch, seq=2, expect=1)).applied
    assert table.dispatch(h, plane).refusal == "stale"
    assert table.admit(h, plane).refusal == "stale"


def test_an_admission_is_bound_to_its_registry_not_only_its_generation_number():
    """A plane that restarts reaches generation 1 again. Under another registry the admission is
    stale -- only its registry binding tells the two incarnations apart -- and under the same
    registry it still holds: the binding is exactly (generation, registry)."""
    fresh = hf.admission_cases()[4]
    for registry, runs in ((cf.artifacts().registry, False), (fresh.registry, True)):
        table = PackTable(1, 8192)
        h = _stored(table, fresh.pack)
        assert table.admit(h, _plane(fresh.registry)).applied
        restarted = _plane(registry)
        assert restarted.generation == 1
        out = table.dispatch(h, restarted)
        assert (out.applied, out.refusal) == ((True, "none") if runs else (False, "stale"))


def test_a_switch_requested_during_a_dispatch_lands_at_the_boundary():
    after, window = hf.admission_cases()[4], hf.admission_cases()[-1]
    table = PackTable(1, 8192)
    plane = _plane(after.registry)
    h = _stored(table, after.pack)
    assert table.admit(h, plane).applied
    plane.enter()  # another phase in flight
    assert plane.submit(cf.generation_record(window.switch, seq=2, expect=1)).verdict == "deferred"
    assert table.dispatch(h, plane).applied and plane.generation == 1  # nested: no boundary yet
    assert plane.leave().applied and plane.generation == 2
    assert table.dispatch(h, plane).refusal == "stale"


# --- the per-step freeze -------------------------------------------------------------------------


def test_a_frozen_step_is_a_v4_pack_bound_to_the_live_registry():
    gens = hf.BUILDER_GENS
    data = freeze_claims(hf.step_graph(3, gens), gens, 1)
    pack = decode(data)
    assert struct.unpack_from("<H", data, 4)[0] == 4 and pack.source_plan == ""
    assert [g.rid for g in pack.generations] == [g.rid for g in gens]
    assert (pack.map_gen, pack.data_gen) == (max(g.map_gen for g in gens),
                                             max(g.data_gen for g in gens))  # fmt: skip
    assert [s.claim_id for s in pack.segments] == [1, 2, 3, 4, 5]
    assert all(t.claim_id == s.claim_id for t, s in zip(pack.trace_notes, pack.segments))
    assert hydrated_layout(pack)


def test_every_freeze_law_names_its_status():
    want = {
        "overflow": "BCIR_ERR_OVERFLOW", "no-vector": "BCIR_ERR_GENERATION",
        "unsorted-vector": "BCIR_ERR_GENERATION", "too-many-reads": "BCIR_ERR_PROVENANCE",
        "too-many-writes": "BCIR_ERR_PROVENANCE", "label-control-char": "BCIR_ERR_PROVENANCE",
        "label-no-terminator": "BCIR_ERR_PROVENANCE", "label-longest": "BCIR_OK",
        "ids-not-ascending": "BCIR_ERR_PROVENANCE", "nop-ids-count": "BCIR_ERR_PROVENANCE",
        "lane-out-of-range": "BCIR_ERR_LANE", "undeclared-rid": "BCIR_ERR_PROVENANCE",
        "capacity": "BCIR_ERR_NOSPACE", "empty-graph": "BCIR_OK", "nop-only": "BCIR_OK",
    }  # fmt: skip
    for name, claims, gens, _ in hf._law_graphs():
        assert _status(lambda c=claims, g=gens: freeze_claims(c, g, 1, 1024)) == want[name], name


# --- the manifest-of-shards ---------------------------------------------------------------------


def test_the_manifest_layout_is_the_c_twin_s():
    header = open(os.path.join(hf.C_DIR, "bcir_shard_manifest.h"), encoding="utf-8").read()

    def define(name):
        return int(re.search(rf"#define {name}\s+(\w+)", header).group(1).rstrip("u"), 0)

    assert define("BCIR_SHM_HEADER_SIZE") == MANIFEST_HEADER_SIZE
    assert define("BCIR_SHM_ENTRY_SIZE") == MANIFEST_ENTRY_SIZE
    assert define("BCIR_SHM_FIXED") == MANIFEST_FIXED
    assert define("BCIR_SHM_SHARDS_MAX") == SHARDS_MAX
    assert define("BCIR_SHM_BLOB_MAX") == BLOB_MAX


def test_shards_reassemble_to_the_whole_pack_s_bytes():
    for whole, ranges in hf.split_cases():
        sp = split(whole, ranges)
        assert reassemble(sp.manifest, sp.store().get) == whole
        assert sp.frame == frame_of(whole)
        for (b, e), shard in zip(ranges, sp.shards):
            assert shard == sub_pack(whole, b, e)
            pack = decode(shard)
            assert len(pack.segments) == e - b and hydrated_layout(pack)
            assert pack.generations == decode(whole).generations


def test_each_shard_is_admitted_and_run_by_itself_and_the_ranks_reproduce_the_whole():
    for whole, ranges in hf.split_cases():
        assert hf.reentry_python(whole, ranges), len(ranges)


def test_the_manifest_is_gated_at_the_plane_before_any_shard_is_fetched():
    fresh = hf.admission_cases()[4]
    sp = split(fresh.pack, partition(len(decode(fresh.pack).segments), 2))
    assert (
        admit_manifest(ControlPlane(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT), sp.manifest).refusal
        == "stale"
    )
    assert admit_manifest(_plane(fresh.registry), sp.manifest).applied
    assert admit_manifest(_plane(cf.artifacts().registry), sp.manifest).refusal == "stale"


def test_every_manifest_wire_law_names_its_status_and_the_encoder_refuses_it_too():
    for name, data, status in hf.malformed_manifests():
        assert _status(lambda d=data: decode_manifest(d)) == status, name
    good = split(hf.synthetic_pack(9), partition(9, 3))
    m = decode_manifest(good.manifest)
    assert encode_manifest(m) == good.manifest  # one spelling
    assert _status(lambda: encode_manifest(replace(m, pack_version=3))) == "BCIR_ERR_SHARD"
    assert _status(lambda: encode_manifest(replace(m, shards=m.shards[:1]))) == "BCIR_ERR_SHARD"
    assert _status(lambda: encode_manifest(replace(m, whole_sha256=b"\0"))) == "BCIR_ERR_SHARD"


def test_a_tampered_shard_set_never_reassembles():
    for name, manifest, blobs, status in hf.tampered_sets():
        assert hf.expected_reassemble(manifest, blobs) == "ERR " + status, name


def test_only_the_hydrated_layout_shards():
    for name, whole, ranges in hf.refused_splits():
        assert hf.expected_split(whole, ranges) == "ERR BCIR_ERR_SHARD", name
    for whole in hf.whole_packs():
        assert hydrated_layout(decode(whole))


def test_the_partition_is_the_seam_s():
    assert partition(10, 3) == [(0, 4), (4, 8), (8, 10)]
    assert partition(9, 4) == [(0, 3), (3, 6), (6, 9)]  # ceil: fewer shards than ranks
    assert partition(0, 5) == [(0, 0)] and partition(5, 0) == [(0, 5)]


# --- the scenarios and their grading -------------------------------------------------------------


def test_every_family_has_fixtures_and_every_refusal_a_witness():
    scenarios = hf.all_scenarios()
    assert {s.family for s in scenarios} == set(hf.FAMILIES)
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names))
    seen = {step.expect[1] for s in scenarios for step in s.steps if step.expect}
    assert seen >= set(HO_REFUSALS) - {"walk"}, set(HO_REFUSALS) - seen
    grades = {step.grade for s in scenarios for step in s.steps}
    assert grades >= {"stale-admit", "stale-dispatch", "fresh-admit", "fresh-dispatch",
                      "lifetime", "freeze"}  # fmt: skip


def test_every_scenario_decides_as_the_specification_requires_on_the_oracle():
    for index, scenario in enumerate(hf.all_scenarios()):
        outcomes, lines = hf.run_python(scenario, index)
        assert len(lines) == len(scenario.steps)
        miss = hf._grade(scenario, [(o.verdict, o.refusal) for o in outcomes])
        assert not any(miss.values()), (scenario.name, miss)


def test_the_grading_fails_a_rail_that_lies():
    """A witness must hit the law it tests (L11): an admitted stale pack, a lifetime refusal
    answered, a truncated trace -- each fails the row that owns it."""
    scenario = next(s for s in hf.all_scenarios() if s.name == "admission/stale/maxima-moved")
    outcomes = [(o.verdict, o.refusal) for o in hf.run_python(scenario)[0]]
    i = next(k for k, s in enumerate(scenario.steps) if s.grade == "stale-admit")
    lied = outcomes[:i] + [("applied", "none")] + outcomes[i + 1 :]
    assert hf._grade(scenario, lied)["stale-admit"] == 1
    j = next(k for k, s in enumerate(scenario.steps) if s.grade == "stale-dispatch")
    ran = outcomes[:j] + [("applied", "none")] + outcomes[j + 1 :]
    assert hf._grade(scenario, ran)["stale-dispatch"] == 1
    life = next(s for s in hf.all_scenarios() if s.name == "lifetime/view-after-release")
    outs = [(o.verdict, o.refusal) for o in hf.run_python(life)[0]]
    k = next(n for n, s in enumerate(life.steps) if s.grade == "lifetime")
    assert hf._grade(life, outs[:k] + [("applied", "none")] + outs[k + 1 :])["lifetime"] == 1
    reached = [s.expect is not None for s in scenario.steps]
    assert sum(hf._grade(scenario, outcomes[:1]).values()) == sum(reached[1:])  # the rest fail
    # a trace that differs only in its scenario ordinal is the same trace; one digest is not
    _, lines = hf.run_python(scenario, 3)
    renumbered = ["9" + line[1:] for line in lines]
    assert hf._body(renumbered) == hf._body(lines)
    bent = lines[:-1] + [lines[-1][:-1] + ("0" if lines[-1][-1] != "0" else "1")]
    assert hf._body(bent) != hf._body(lines)


# --- the Stage 3 exit flow -------------------------------------------------------------------------


def test_one_generation_flows_through_every_stage3_boundary_and_the_old_one_is_refused_at_each():
    """The staged plan's Stage 3 exit gate on the oracle: generation a flows plan -> control ->
    data -> telemetry -> evidence; after the switch its record, plan, pack (dispatch and
    admission), manifest and a late sample are each refused at their own boundary, with the
    boundary's own status; generation b then flows; and the evidence reconciles -- the two-slot
    ring's loss is exactly the intake's gap count."""
    case = hf.stage3_case()
    lines = hf.run_stage3_python(case)
    assert hf.grade_stage3(case, lines) == (0, 0)
    by_label = {line.split(" ", 1)[0]: line for line in lines}
    assert by_label["control:stale"].split()[1:3] == ["2", "8"]  # refused, stale (G14)
    assert by_label["plan:a-stale"].split()[1:4] == ["2", "8", "BCIR_ERR_STALE"]
    for label in ("data:dispatch-a-stale", "data:admit-a-stale", "data:manifest-a-stale"):
        assert by_label[label].split()[1:4] == ["1", "4", "BCIR_ERR_STALE"], label  # G16
    late = [line for line in lines if line.startswith("tel:") and " g=1 " in line][-1]
    assert late.split()[1:4] == ["2", "4", "BCIR_ERR_STALE"]  # the intake (G15)
    assert "tel:lost 2" in lines
    evidence = dict(kv.split("=") for kv in lines[-1].split()[1:])
    assert evidence["lost"] == evidence["missing"] == "2" and evidence["stale"] == "1"
    assert evidence["g"] == "2" and evidence["control"] == "4"
    # every generation handle on the way is the one the data plane admitted
    assert all(" g=2 " in line for line in lines if line.startswith("tel:") and "k=" in line
               and line.split()[0] not in ("tel:1", "tel:4", "tel:5", late.split()[0]))  # fmt: skip


def test_the_stage3_grading_fails_a_flow_that_accepts_the_old_generation():
    """L11: the grader must hit the law. A stale boundary that applies, a late sample accepted,
    a lost count that no longer matches the gaps, a missing evidence line -- each is a miss in the
    row that owns it."""
    case = hf.stage3_case()
    lines = hf.run_stage3_python(case)

    def bent(label, fields):
        return [
            " ".join([label, *fields]) if line.startswith(label + " ") else line for line in lines
        ]

    assert hf.grade_stage3(case, bent("control:stale", ["0", "0", "BCIR_OK", "g=3"]))[0] == 1
    assert hf.grade_stage3(case, bent("data:admit-a-stale", ["0", "0", "BCIR_OK", "g=2"]))[0] == 1
    late = next(line.split()[0] for line in lines if line.startswith("tel:") and " 2 4 " in line)
    assert hf.grade_stage3(case, bent(late, ["0", "0", "BCIR_OK", "g=1", "k=2"]))[0] == 1
    assert hf.grade_stage3(case, lines[:-1]) == (0, 1)
    assert hf.grade_stage3(case, [line.replace("lost=2", "lost=3") for line in lines]) == (0, 1)
    assert hf.grade_stage3(case, lines + [lines[3]]) == (0, 1)  # a repeated boundary
    stale_total = len(hf.STAGE3_STALE) + 1  # and the late sample
    assert hf.grade_stage3(case, [])[0] == stale_total


def test_the_stage3_flow_is_identical_on_the_c_twin_and_the_c_plus_plus_seam():
    case = hf.stage3_case()
    want = hf.run_stage3_python(case)
    with tempfile.TemporaryDirectory() as tmp:
        for build in (hf.build_harness, hf.build_cpp_harness):
            exe = build(tmp)
            if exe is None:
                continue
            got = hf.run_mode(exe, tmp, "stage3", hf.encode_stage3(case))
            assert got == want, build.__name__
            assert hf.grade_stage3(case, got) == (0, 0)


# --- the native rails -----------------------------------------------------------------------------


def test_the_c_twin_traces_every_scenario_identically():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_harness(tmp)
        if exe is None:
            return
        scenarios = hf.all_scenarios()
        traces, copies = hf.run_native(exe, tmp, scenarios)
        assert copies is None
        for index, (scenario, trace) in enumerate(zip(scenarios, traces)):
            assert trace == hf.run_python(scenario, index)[1], scenario.name


def test_the_c_twin_s_api_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_harness(tmp)
        if exe is None:
            return
        run = hf._run([exe, "--api"])
        assert run.returncode == 0 and run.stdout.startswith("OK "), run.stdout + run.stderr


def test_the_c_freeze_is_byte_identical_to_the_oracle():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_harness(tmp)
        if exe is None:
            return
        corpus = hf.freeze_corpus()
        got = hf.run_mode(exe, tmp, "freeze", hf.freeze_payload(corpus))
        want = [hf.expected_freeze(*case) for case in corpus]
        assert got == want
        assert sum(line.startswith("OK ") for line in want) >= len(want) // 3  # bytes, not refusals


def test_the_c_split_decode_and_reassembly_agree_with_the_oracle():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_harness(tmp)
        if exe is None:
            return
        cases = hf.split_cases()
        got = hf.run_mode(exe, tmp, "split", hf.split_payload(cases))
        assert got == [hf.expected_split(w, r) for w, r in cases]
        malformed = hf.malformed_manifests()
        got = hf.run_mode(exe, tmp, "decode", hf.decode_payload([d for _, d, _ in malformed]))
        assert got == ["ERR " + status for _, _, status in malformed]
        tampered = hf.tampered_sets()
        got = hf.run_mode(exe, tmp, "reassemble",
                          hf.reassemble_payload([(m, b, 1 << 16) for _, m, b, _ in tampered]))  # fmt: skip
        assert got == ["ERR " + status for *_, status in tampered]
        pairs = [(4, 3), (0, 9), (1, 1), (4097, 4097)]
        got = hf.run_mode(exe, tmp, "partition", struct.pack("<I", len(pairs)) +
                          b"".join(struct.pack("<II", n, w) for n, w in pairs))  # fmt: skip
        assert got[:3] == ["OK " + ",".join(f"{b}:{e}" for b, e in partition(n, w))
                           for n, w in pairs[:3]]  # fmt: skip
        assert got[3] == "ERR BCIR_ERR_SHARD"  # 4097 one-segment shards: past the bound


def test_the_c_plus_plus_seam_traces_every_scenario_it_can_spell_identically():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_cpp_harness(tmp)
        if exe is None:
            return
        chosen = [(i, s) for i, s in enumerate(hf.all_scenarios()) if not s.c_only]
        traces, copies = hf.run_native(exe, tmp, [s for _, s in chosen])
        assert copies == 0
        for (index, scenario), trace in zip(chosen, traces):
            assert hf._body(trace) == hf._body(hf.run_python(scenario, index)[1]), scenario.name


def test_the_c_plus_plus_lifetime_witnesses_hold():
    with tempfile.TemporaryDirectory() as tmp:
        exe = hf.build_cpp_harness(tmp)
        if exe is None:
            return
        lines = hf._run([exe, "--lifetime"]).stdout.splitlines()
        assert [line.split()[0] for line in lines[:-1]] == list(hf.CPP_LIFETIME_WITNESSES)
        assert all(line.endswith(" ok") for line in lines[:-1]) and lines[-1].startswith("OK "), (
            lines
        )


def test_every_g16_row_is_zero_on_every_rail():
    with tempfile.TemporaryDirectory() as tmp:
        exe, cpp = hf.build_harness(tmp), hf.build_cpp_harness(tmp)
        rows = hf.measure(exe, cpp, tmp)
        assert set(rows) == set(hf.ROWS), rows
        assert all(value == 0 for value in rows.values()), rows


def test_an_absent_rail_fails_every_fixture_it_was_handed():
    """L1: a harness that cannot run decides nothing -- it fails everything it was handed."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = hf.measure(os.path.join(tmp, "absent"), os.path.join(tmp, "absent_cpp"), tmp)
        assert rows["handoff.traces.divergent"] == len(hf.all_scenarios()) * 2 - sum(
            s.c_only for s in hf.all_scenarios()
        )
        assert rows["handoff.stale.admitted"] > 0 and rows["handoff.lifetime.unrefused"] > 0
        assert rows["handoff.copies"] > 0 and rows["handoff.shards.mismatches"] > 0


def test_an_oracle_that_cannot_build_its_corpus_fails_the_rows_it_feeds():
    """L1: the corpora are the oracle's own splits, so a defect in `split` can first surface as a
    corpus that cannot be built. Every row that corpus feeds fails -- never a traceback (the fault
    sweep found the traceback: a shard without its vector made the manifest encoder refuse its
    own output while the hostile corpus was being built)."""

    def broken(*_args, **_kwargs):
        raise ShardError("BCIR_ERR_SHARD", "a defective oracle")

    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hf, "split", broken):
        rows = hf.measure(None, None, tmp)
    for row in (
        "handoff.shards.malformed.accepted",
        "handoff.shards.mismatches",
        "handoff.reentry.divergent",
        "handoff.traces.divergent",
        "handoff.stale.admitted",
    ):
        assert rows[row] > 0, row
    assert set(rows) == set(hf.ROWS)


# --- the gates are wired to the sources they grade ------------------------------------------------


def test_the_gates_and_the_harnesses_link_the_same_sources():
    """check_runtime.sh's handoff link line, check_handoff.sh's C++ build and the fuzz target are
    lists each, and nothing but this test makes them agree with the fixtures' (the #719 trap)."""
    from bcir.tests.run_all import _is_source_checkout

    if not _is_source_checkout():
        return  # the wheel ships neither tools/ nor runtime/
    gate = open(os.path.join(_ROOT, "tools", "c", "check_runtime.sh"), encoding="utf-8").read()
    line = re.search(r"handoff_sources=\(([^)]*)\)", gate).group(1)
    assert set(re.findall(r"(\w+\.c)", line)) == set(hf.C_UNITS) | {"test_handoff.c"}
    cpp = open(os.path.join(_ROOT, "tools", "cpp", "check_handoff.sh"), encoding="utf-8").read()
    line = re.search(r"seam_cpp=\(([^)]*)\)", cpp).group(1)
    assert set(re.findall(r"(\w+\.cpp)", line)) == set(hf.CPP_UNITS)
    line = re.search(r"seam_c=\(([^)]*)\)", cpp).group(1)
    assert set(re.findall(r"(\w+\.c)", line)) == set(hf.C_UNITS)
    fuzz = open(os.path.join(_ROOT, "tools", "c", "fuzz_streampack.sh"), encoding="utf-8").read()
    target = re.search(r'add_target handoff "[^"]*" "[^"]*" \\\n\s*(.*)', fuzz).group(1)
    assert set(re.findall(r"(\w+\.c)", target)) == {"fuzz_handoff.c"} | set(hf.C_UNITS)


def test_every_frozen_blob_digest_is_the_store_key():
    sp = split(hf.synthetic_pack(7), partition(7, 2))
    store = sp.store()
    assert set(store) == {hashlib.sha256(b).digest() for b in (sp.frame, *sp.shards)}
