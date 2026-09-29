"""ControlRecordV1 -- the control plane as bytes (GEM+ roadmap G14, staged plan S3-A).

Before this slice lease, generation, quiescence, activation, rollback and cancellation were
prose, an `admit(map_gen, data_gen)` argument and two methods that raise: a stale generation
was refused only where a caller passed the right number, and a switch requested mid-phase was
refused rather than deferred. These tests pin the three G14 gates and the laws around them:

    control.record.bytes   every record bounded and versioned; an unknown version, a reserved
                           byte, a trailing byte and every other wire law refused on both
                           rails with the same status
    stale generation       a record, pack or plan minted against a generation the plane has
                           left is refused by its own bytes, at every boundary the tree has
    quiescent switch       a switch requested mid-phase (or before its boundary) is deferred
                           -- the resident generation does not move -- and applied exactly
                           once at the boundary, or refused there and reported: never applied
                           early, never applied twice, never lost

The corpora are declared once, with the outcome the specification requires, in
`bcir/tests/control_fixtures.py`. The C-rail tests build `runtime/c/test_control_plane.c`
and skip without a compiler (the quick tier hides one on purpose; the c-runtime tier exposes
it) or outside a source checkout (the wheel does not ship runtime/c).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import struct
import tempfile
from dataclasses import replace

from bcir.abi.control_abi import (
    CONTROL_BODY_BYTES,
    CONTROL_HEADER_SIZE,
    CONTROL_RECORD_MAX_BYTES,
    CONTROL_TRAILER_SIZE,
    ControlError,
    check_control_mac,
    control_version,
    decode_control,
    encode_control,
    issue_control,
    sign_control,
)
from bcir.examples import vector_add
from bcir.gem.control import (
    CAP_ALL,
    CAP_GRANTABLE,
    CAPABILITY,
    CONTROL_KINDS,
    CONTROL_SCOPES,
    LEASE_CAPACITY,
    REASONS,
    REFUSALS,
    SWITCH_KINDS,
    VERDICTS,
    Activate,
    Cancel,
    ControlPlane,
    LeaseGrant,
    Quiesce,
    is_stale,
    lease_key,
    registry_digest,
)
from bcir.gem.streampack import generation_vector
from bcir.tests import control_fixtures as cf
from bcir.verify import verify_control_record


# Every deferral the scenario corpus makes (each must reach exactly one reported decision).
DEFERRED_TOTAL = 11
# Every refusal the corpus makes that must leave the resident state untouched.
INERT_REFUSALS = 51


def _status(data: bytes) -> str:
    try:
        decode_control(data)
    except ControlError as exc:
        return exc.status
    return "BCIR_OK"


def _raises(status: str, callback) -> None:
    try:
        callback()
    except ControlError as exc:
        assert exc.status == status, (exc.status, status, exc)
        return
    raise AssertionError(f"expected {status}")


def _stepwise(scenario):
    """Run a scenario on the Python rail, yielding (step, outcome, digest before, digest after,
    plane) for every operation."""
    plane = ControlPlane(scenario.key, scenario.scope, scenario.subject)
    for step in scenario.steps:
        before = plane.state_digest()
        if step.op == cf.OP_SUBMIT:
            outcome = plane.submit(step.data)
        elif step.op == cf.OP_ENTER:
            outcome = plane.enter()
        elif step.op == cf.OP_LEAVE:
            outcome = plane.leave()
        elif step.op == cf.OP_ADVANCE:
            outcome = plane.advance()
        elif step.op == cf.OP_ADMIT_PACK:
            outcome = plane.admit_pack(step.data)
        elif step.op == cf.OP_ADMIT_PLAN:
            outcome = plane.admit_plan(step.data)
        else:
            (value,) = struct.unpack("<Q", step.data)
            setattr(plane, "boundary" if step.op == cf.OP_POKE_BOUNDARY else "in_flight", value)
            outcome = None
        yield step, outcome, before, plane.state_digest(), plane


# --- the bytes -------------------------------------------------------------------------------


def test_every_record_round_trips_byte_identically_at_its_fixed_length():
    corpus = cf.record_corpus()
    assert len(corpus) == 29
    kinds, scopes, reasons = set(), set(), set()
    for name, blob in corpus:
        record = decode_control(blob)
        assert encode_control(record) == blob, name
        assert control_version(record) == 1
        size = CONTROL_HEADER_SIZE + CONTROL_BODY_BYTES[1][record.kind] + CONTROL_TRAILER_SIZE
        assert len(blob) == size <= CONTROL_RECORD_MAX_BYTES, name
        kinds.add(record.kind)
        scopes.add(record.scope)
        reasons.add((record.kind, record.reason))
    # the corpus exercises every kind, every scope and every reason of every kind
    assert kinds == set(CONTROL_KINDS) and scopes == set(CONTROL_SCOPES)
    assert reasons == {(kind, reason) for kind in CONTROL_KINDS for reason in REASONS[kind]}
    sizes = {
        kind: CONTROL_HEADER_SIZE + n + CONTROL_TRAILER_SIZE
        for kind, n in CONTROL_BODY_BYTES[1].items()
    }
    assert sizes == {
        "lease": 140,
        "generation": 148,
        "quiesce": 108,
        "activate": 164,
        "rollback": 164,
        "cancel": 116,
    }
    assert max(sizes.values()) == 164 < CONTROL_RECORD_MAX_BYTES == 192


def test_every_wire_law_refuses_its_malformed_variant_with_its_status():
    variants = cf.malformed_variants()
    assert len(variants) == 41
    statuses = set()
    for name, blob, want in variants:
        assert _status(blob) == want, (name, _status(blob), want)
        statuses.add(want)
    # every status the codec can name has a witness (L22)
    assert statuses == {
        "BCIR_ERR_TRUNCATED",
        "BCIR_ERR_TRAILING",
        "BCIR_ERR_MAGIC",
        "BCIR_ERR_VERSION",
        "BCIR_ERR_RESERVED",
        "BCIR_ERR_CONTROL",
        "BCIR_ERR_CRC",
        "BCIR_ERR_MAC",
    }


def test_the_encoder_refuses_what_the_decoder_refuses():
    """One predicate on both sides of the wire: a record the decoder would refuse cannot be
    published, and the encoder's own output always decodes (both halves, class A)."""
    base = cf.record("activate", Activate(cf.A_DIGEST, cf.B_DIGEST), seq=4, expect=2, lease=1)
    key = lease_key(cf.ROOT_KEY, 1)
    refused = {
        "sequence 0": replace(base, sequence=0),
        "lease 0 on a switch": replace(base, lease=0),
        "switch not expect + 1": replace(base, generation=5),
        "switch wraps": replace(base, expect=(1 << 32) - 1, generation=0),
        "unknown scope": replace(base, scope="galaxy"),
        "reason outside the set": replace(base, reason="health"),
        "unchanged artifact": replace(base, body=Activate(cf.A_DIGEST, cf.A_DIGEST)),
        "zero artifact": replace(base, body=Activate(bytes(32), cf.B_DIGEST)),
        "short digest": replace(base, body=Activate(b"\x01" * 31, cf.B_DIGEST)),
        "wrong body type": replace(base, body=Quiesce(9)),
        "negative subject": replace(base, subject=-1),
        "subject past u64": replace(base, subject=1 << 64),
        "bool sequence": replace(base, sequence=True),
    }
    for name, record in refused.items():
        _raises("BCIR_ERR_CONTROL", lambda record=record: sign_control(record, key))
    lease = cf.record("lease", LeaseGrant(1, 0x3E, 0, 9, 42), seq=1, expect=0, lease=0)
    for body in (
        LeaseGrant(0, 0x3E, 0, 9, 42),  # lease id 0
        LeaseGrant(1, 0x3F, 0, 9, 42),  # delegation
        LeaseGrant(1, 0x40, 0, 9, 42),  # a bit outside v1
        LeaseGrant(1, 0, 0, 9, 42),  # nothing granted
        LeaseGrant(1, 0x3E, 9, 9, 42),  # an empty window
        LeaseGrant(1, 0x3E, 0, 9, 0),  # no holder
    ):
        _raises(
            "BCIR_ERR_CONTROL",
            lambda body=body: sign_control(replace(lease, body=body), cf.ROOT_KEY),
        )
    cancel = cf.record("cancel", Cancel(2, 3), seq=4, expect=0, lease=1)
    for body in (Cancel(0, 3), Cancel(3, 2), Cancel(2, 4)):
        _raises("BCIR_ERR_CONTROL", lambda body=body: sign_control(replace(cancel, body=body), key))
    _raises(
        "BCIR_ERR_CONTROL",
        lambda: sign_control(
            cf.record("quiesce", Quiesce(3), seq=1, expect=0, lease=1, boundary=4), key
        ),
    )
    # an unsigned record, a short MAC and an all-zero MAC are not publishable
    _raises("BCIR_ERR_MAC", lambda: encode_control(base))
    _raises("BCIR_ERR_MAC", lambda: encode_control(replace(base, mac=b"\x01" * 31)))
    _raises("BCIR_ERR_MAC", lambda: encode_control(replace(base, mac=bytes(32))))
    # ...while the honest record publishes and decodes to itself
    blob = issue_control(base, key)
    assert decode_control(blob) == sign_control(base, key)


def test_the_first_violated_law_is_the_one_named():
    """The laws run in the specification's order, so a record breaking two names the earlier
    one on both rails (the C twin mirrors this order; tools/c/check_runtime.sh grades it)."""
    act = cf.issue(
        cf.record("activate", Activate(cf.A_DIGEST, cf.B_DIGEST), seq=4, expect=2, lease=1)
    )
    gen = cf.generation_record((1, 1, 1, cf.digest_of("reg")), seq=2, expect=1)
    flip_crc = lambda data: data[:-4] + bytes(b ^ 0xFF for b in data[-4:])  # noqa: E731
    assert _status(flip_crc(b"BCTX" + act[4:])) == "BCIR_ERR_MAGIC"  # magic before CRC
    assert _status(cf._put(cf._put(act, 4, "H", 2), 6, "H", 1)) == "BCIR_ERR_VERSION"
    assert _status(flip_crc(cf._put(act, 8, "B", 7))) == "BCIR_ERR_CONTROL"  # kind before CRC
    assert _status(flip_crc(act + b"\x00")) == "BCIR_ERR_TRAILING"  # length before CRC
    # a header law before the generation body's reserved word, which comes before the body laws
    assert _status(cf._put(cf._put(gen, 76, "I", 1), 9, "B", 5)) == "BCIR_ERR_CONTROL"
    zero_digest = cf._recrc(gen[:80] + bytes(32) + gen[112:])
    assert _status(cf._put(zero_digest, 76, "I", 1)) == "BCIR_ERR_RESERVED"
    # a body law before the MAC law
    both = cf._recrc(act[:64] + bytes(32) + act[96:128] + bytes(32) + act[160:])
    assert _status(both) == "BCIR_ERR_CONTROL"


def test_the_crc_detects_and_the_mac_decides():
    """The CRC is the corruption gate a keyless reader applies; the MAC is authority. A body
    edit with the CRC recomputed is well-formed and still refused -- by the MAC, which covers
    the header and the body -- and a record under another key is refused though its CRC holds."""
    plane = ControlPlane(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT)
    grant = cf.grant(1, seq=1)
    assert check_control_mac(grant, cf.ROOT_KEY) and not check_control_mac(grant, cf.FOREIGN_KEY)
    edited = cf._put(grant, 72, "Q", 0x02)  # narrow the mask, CRC recomputed, MAC unchanged
    assert _status(edited) == "BCIR_OK" and not check_control_mac(edited, cf.ROOT_KEY)
    assert plane.submit(edited).refusal == "mac"
    corrupt = grant[:72] + bytes([grant[72] ^ 1]) + grant[73:]  # the CRC catches it first
    assert plane.submit(corrupt).status == "BCIR_ERR_CRC"
    foreign = cf.issue(decode_control(grant), cf.FOREIGN_KEY)
    outcome = plane.submit(foreign)
    assert (outcome.refusal, outcome.status) == ("mac", "BCIR_ERR_MAC")
    assert plane.submit(grant).applied and plane.leases[0].granted == 0x3E


def test_a_lease_key_is_scoped_to_its_lease():
    keys = {lease_key(cf.ROOT_KEY, lease_id) for lease_id in (1, 2, 3, (1 << 64) - 1)}
    assert len(keys) == 4 and cf.ROOT_KEY not in keys
    expected = hmac.new(
        cf.ROOT_KEY, b"BCTL/lease/v1\x00" + struct.pack("<Q", 7), hashlib.sha256
    ).digest()
    assert lease_key(cf.ROOT_KEY, 7) == expected
    assert registry_digest([]) == hashlib.sha256(b"BCTL/registry/v1\x00").digest()


# --- the plane --------------------------------------------------------------------------------


def test_every_scenario_decides_as_the_specification_requires():
    scenarios = cf.all_scenarios()
    families = {}
    for scenario in scenarios:
        plane, outcomes, _lines = cf.run_python(scenario)
        assert cf.conforms(scenario, outcomes, plane), scenario.name
        families[scenario.family] = families.get(scenario.family, 0) + 1
    assert families == {"transition": 6, "authority": 9, "stale": 13, "midphase": 8, "witness": 17}


def test_every_refusal_and_every_verdict_has_a_witness():
    refusals, verdicts = set(), set()
    for scenario in cf.all_scenarios():
        for step in scenario.steps:
            refusals.add(step.refusal)
            verdicts.add(step.verdict)
    assert refusals == set(REFUSALS) and verdicts == set(VERDICTS)


def test_a_refused_operation_changes_nothing():
    """Not a sequence counter, not a lease, not the pending slot: the resident state digest is
    identical before and after every refusal of every scenario."""
    refusals = 0
    for scenario in cf.all_scenarios():
        for step, outcome, before, after, _plane in _stepwise(scenario):
            if outcome is None or outcome.verdict != "refused":
                continue
            if step.op in (cf.OP_LEAVE, cf.OP_ADVANCE) and outcome.kind:
                continue  # a boundary was crossed; the refusal is the pending switch's, reported
            assert before == after, (scenario.name, step)
            refusals += 1
    assert refusals == INERT_REFUSALS


def test_a_switch_is_deferred_mid_phase_and_decided_exactly_once_at_a_boundary():
    """The quiescent-switch gate as a property over every scenario: a deferral leaves the
    resident generation where it was; no switch is applied while a phase is in flight; every
    deferred record reaches exactly one reported decision (applied or refused at a boundary,
    or withdrawn by a cancel)."""
    deferred_total = 0
    for scenario in cf.all_scenarios():
        open_deferral = None
        previous_generation = 0
        for step, outcome, _before, _after, plane in _stepwise(scenario):
            if outcome is None:
                previous_generation = plane.generation
                continue
            if outcome.verdict == "deferred":
                assert open_deferral is None, scenario.name  # one pending slot
                assert plane.pending is not None, scenario.name
                assert plane.generation == previous_generation, scenario.name  # did not move
                open_deferral = (outcome.kind, outcome.sequence, plane.generation)
                deferred_total += 1
                continue
            if open_deferral is not None:
                kind, sequence, generation = open_deferral
                decided = outcome.kind == kind and outcome.sequence == sequence
                withdrawn = step.op == cf.OP_SUBMIT and outcome.kind == "cancel" and outcome.applied
                if decided and step.op in (cf.OP_LEAVE, cf.OP_ADVANCE):
                    assert plane.in_flight == 0 and plane.pending is None, scenario.name
                    if outcome.applied:
                        assert plane.generation == generation + 1, scenario.name
                    open_deferral = None
                elif withdrawn:
                    assert plane.pending is None and plane.generation == generation
                    open_deferral = None
                else:  # anything else leaves the deferral pending and the generation unmoved
                    assert plane.generation == generation, (scenario.name, step)
            if outcome.applied and outcome.kind in SWITCH_KINDS:
                assert plane.in_flight == 0, scenario.name
            previous_generation = plane.generation
    assert deferred_total == DEFERRED_TOTAL


def test_a_stale_generation_is_refused_at_every_boundary():
    stale = cf.stale_scenarios()
    assert len(stale) == 13
    for scenario in stale:
        plane, outcomes, _lines = cf.run_python(scenario)
        assert cf.conforms(scenario, outcomes, plane), scenario.name
        assert outcomes[-1].refusal == "stale"
    boundaries = cf.python_stale_boundaries()
    assert [name for name, _ in boundaries] == [
        "staged.TrustedLoader.install",
        "context_shard.certify_context_activation",
        "verify.verify_control_record",
    ]
    assert all(holds for _, holds in boundaries), boundaries


def test_one_staleness_predicate_is_shared_by_every_python_boundary():
    import bcir.asn1.staged as staged
    import bcir.kbcir.context_shard as context_shard

    assert staged.is_stale is is_stale and context_shard.is_stale is is_stale
    assert is_stale(1, 2) and not is_stale(2, 2) and not is_stale(3, 2)


def test_the_plane_refuses_a_key_scope_or_subject_it_cannot_hold():
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(b"\x01" * 15))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(b"\x01" * 65))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane("text key, not bytes"))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(cf.ROOT_KEY, "galaxy"))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(cf.ROOT_KEY, "module", -1))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(cf.ROOT_KEY, "module", 1 << 64))
    _raises("BCIR_ERR_CONTROL", lambda: ControlPlane(cf.ROOT_KEY, "module", True))


def test_the_state_digest_moves_with_every_field():
    """Two planes differing only in one field differ in the digest: the parity traces compare
    the whole resident state, not a projection of it."""
    plane = ControlPlane(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT)
    for step in cf._bootstrap():
        assert plane.submit(step.data).applied
    base = plane.state_digest()
    mutations = {
        "generation": 9,
        "boundary": 9,
        "in_flight": 9,
        "draining": True,
        "drain_deadline": 9,
        "artifact": cf.B_DIGEST,
        "previous": cf.B_DIGEST,
        "token": cf.B_DIGEST,
        "registry": None,
        "root_sequence": 9,
        "last_lease_id": 9,
        "scope": "channel",
        "subject": 9,
        "pending": b"\x01",
    }
    for attribute, value in mutations.items():
        saved = getattr(plane, attribute)
        setattr(plane, attribute, value)
        assert plane.state_digest() != base, attribute
        setattr(plane, attribute, saved)
    for field_name in ("lease_id", "granted", "issued", "expiry", "holder", "last_sequence"):
        entry = plane.leases[0]
        saved = getattr(entry, field_name)
        setattr(entry, field_name, saved + 1)
        assert plane.state_digest() != base, field_name
        setattr(entry, field_name, saved)
    assert plane.state_digest() == base


# --- the verifier -------------------------------------------------------------------------------


def test_the_verifier_holds_a_generation_record_to_the_module_s_registry():
    module = vector_add(64)
    vector = generation_vector(module)
    maxima = (max(g.map_gen for g in vector), max(g.data_gen for g in vector))
    current = cf.generation_record((*maxima, 1, registry_digest(vector)), seq=1, expect=0)
    assert verify_control_record(module, current) == []
    assert verify_control_record(module, decode_control(current)) == []
    rid = min(module.resources)
    moved = [replace(g, map_gen=g.map_gen + 1) if g.rid == rid else g for g in vector]
    older = cf.generation_record((*maxima, 1, registry_digest(moved)), seq=1, expect=0)
    diags = verify_control_record(module, older)
    assert [d.law for d in diags] == ["R11"] and "stale control record" in diags[0].message
    wrong_maxima = cf.generation_record(
        (maxima[0] + 1, maxima[1], 1, registry_digest(vector)), seq=1, expect=0
    )
    diags = verify_control_record(module, wrong_maxima)
    assert [d.law for d in diags] == ["R11"] and "maxima" in diags[0].message
    malformed = verify_control_record(module, current[:-1])
    assert [d.law for d in malformed] == ["R11"] and "BCIR_ERR_TRUNCATED" in malformed[0].message
    # the other kinds carry nothing a module can judge
    assert verify_control_record(module, cf.grant(1, seq=1)) == []


# --- the C twin ----------------------------------------------------------------------------------


def test_the_c_twin_decodes_the_corpus_and_the_python_re_encode_is_byte_identical():
    """control.abi.mismatches on the C rail: Python encode -> C decode (a field-by-field dump)
    -> Python re-encode, byte for byte; and the C keyed check accepts every honest MAC."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = cf.build_harness(tmp)
        if exe is None:
            return
        corpus = cf.record_corpus()
        lines = cf.c_dump(exe, tmp, [blob for _, blob in corpus])
        assert len(lines) == len(corpus)
        for (name, blob), line in zip(corpus, lines):
            record, status, mac_ok = cf.parse_c_dump(line)
            assert status == "BCIR_OK" and mac_ok, (name, line)
            assert encode_control(record) == blob, name


def test_the_c_twin_refuses_every_malformed_variant_with_the_same_status():
    with tempfile.TemporaryDirectory() as tmp:
        exe = cf.build_harness(tmp)
        if exe is None:
            return
        variants = cf.malformed_variants()
        lines = cf.c_dump(exe, tmp, [blob for _, blob, _ in variants])
        assert len(lines) == len(variants)
        for (name, _blob, want), line in zip(variants, lines):
            assert line == f"refused status={want}", (name, line)
        # a foreign MAC decodes (the keyless laws hold) and fails the keyed check
        foreign = cf.issue(decode_control(cf.grant(1, seq=1)), cf.FOREIGN_KEY)
        (line,) = cf.c_dump(exe, tmp, [foreign])
        assert line.startswith("record ") and line.endswith("macok=0")


def test_the_two_rails_trace_every_scenario_identically():
    """Every verdict, refusal, status and resident state digest after every operation of every
    scenario, line for line -- and the C rail's decisions are the specification's."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = cf.build_harness(tmp)
        if exe is None:
            return
        scenarios = cf.all_scenarios()
        traces = cf.run_c(exe, tmp, scenarios)
        for scenario, trace in zip(scenarios, traces):
            _plane, _outcomes, lines = cf.run_python(scenario)
            assert trace == lines, scenario.name
            assert cf.conforms(scenario, cf.c_outcomes(trace)), scenario.name


def test_the_c_api_fails_closed_and_the_rails_share_their_codes():
    with tempfile.TemporaryDirectory() as tmp:
        exe = cf.build_harness(tmp)
        if exe is None:
            return
        code, out = cf.c_api(exe)
        assert code == 0 and out.strip() == "OK", out
        # the closed sets, read out of the C header itself rather than mirrored (L14)
        header = open(os.path.join(cf.C_DIR, "bcir_control_plane.h"), encoding="utf-8").read()
        refusals = re.findall(r"BCIR_CTL_REFUSAL_([A-Z]+)\s*=\s*(\d+)", header)
        assert [(name.lower(), int(code)) for name, code in refusals] == [
            (name, index) for index, name in enumerate(REFUSALS)
        ]
        kinds = re.findall(r"BCIR_CTL_([A-Z]+)\s*=\s*(\d+),?\s*/\*", header)
        kind_codes = {
            name.lower(): int(code) for name, code in kinds if name.lower() in CONTROL_KINDS
        }
        assert kind_codes == {kind: index + 1 for index, kind in enumerate(CONTROL_KINDS)}
        scopes = re.findall(r"BCIR_CTL_SCOPE_([A-Z]+)\s*=\s*(\d+)", header)
        assert [(n.lower(), int(c)) for n, c in scopes] == [
            (s, i) for i, s in enumerate(CONTROL_SCOPES)
        ]
        # the capability of a kind is its bit on both rails; v1 grants never delegate
        assert "#define BCIR_CTL_CAP(kind)     (UINT64_C(1) << ((kind) - 1))" in header
        assert all(CAPABILITY[kind] == 1 << index for index, kind in enumerate(CONTROL_KINDS))
        assert f"#define BCIR_CTL_CAP_ALL       UINT64_C(0x{CAP_ALL:X})" in header
        assert f"#define BCIR_CTL_CAP_GRANTABLE UINT64_C(0x{CAP_GRANTABLE:X})" in header
        assert f"#define BCIR_CTL_RECORD_MAX   {CONTROL_RECORD_MAX_BYTES}u" in header
        assert f"#define BCIR_CTL_LEASE_CAPACITY {LEASE_CAPACITY}u" in header
