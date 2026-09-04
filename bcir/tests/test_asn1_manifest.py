"""J4 part 3 — §5.4's two sinks, and the schema-bound manifests they read.

§5.4's law:

    JER -> typed value -> claims  ==  JER -> direct claim builder

The interesting content is *what makes that a real test rather than a tautology*. Both
sinks consume ONE event walk. Two independent readers would be two parsers, free to
disagree about what the document says, and then "the paths commute" would be testing
parser agreement rather than builder agreement. With one walk feeding both, any difference
in the result is a difference in the builders — which is what the law is about.

The corpus is the nine built-in `HardwareChannel`s, round-tripped through their own
`channel.json` form. They are the real objects the optimizer reasons over, not fixtures,
so a projection that lost a cost-model field would change what the optimizer sees.
"""

from __future__ import annotations

import json

from bcir.asn1.jer import JerRules, decode_jer, encode_jer
from bcir.asn1.jer_bounded import STRICT_LIMITS, JerBoundedError, decode_bounded
from bcir.asn1.manifest import (
    CHANNEL_MANIFEST,
    DEVICE_MANIFEST,
    MANIFEST_MODULE_OID,
    MEASUREMENT_TYPE,
    MODULE,
    SELECTION_ENVELOPE,
    SELECTION_ENVELOPE_VERSION,
    DirectChannelSink,
    TypedValueSink,
    channel_direct,
    channel_to_jer,
    channel_via_typed_value,
    device_manifest_to_value,
    selection_envelope,
    value_to_device_manifest,
    walk,
)
from bcir.asn1.tags import Asn1Error
from bcir.channel_plugin import ChannelManifest, manifest_from_channel
from bcir.channels import CHANNELS


def _manifests():
    for name in sorted(CHANNELS):
        yield name, manifest_from_channel(CHANNELS[name])


# --- §5.4's commutation -------------------------------------------------------------------


def test_the_two_sinks_build_the_same_channel():
    """§5.4, over every built-in channel.

    `channel_via_typed_value` goes through `ChannelManifest.from_dict` — the existing
    programmatic constructor, with all of `channel_plugin`'s validation. `channel_direct`
    never builds a `ChannelManifest` at all. They must agree, and agreeing is what says the
    direct path did not quietly drop a field the typed path carries.
    """
    count = 0
    for name, manifest in _manifests():
        octets = channel_to_jer(manifest)
        assert channel_via_typed_value(octets) == channel_direct(octets), name
        count += 1
    assert count >= 8, f"the channel registry collapsed to {count} channels"


def test_the_direct_builder_reproduces_the_original_channel():
    """Commutation alone would be satisfied by two identically-wrong builders.

    So the round trip is also anchored to the object the manifest was made FROM. Without
    this, both paths could agree on a channel that had lost its lane widths and the law
    above would still pass.
    """
    for name, manifest in _manifests():
        rebuilt = channel_direct(channel_to_jer(manifest))
        original = CHANNELS[name]
        assert rebuilt.name == original.name, name
        assert rebuilt.kind == original.kind, name
        assert rebuilt.llvm_triple == original.llvm_triple, name
        assert rebuilt.e_machine == original.e_machine, name
        assert rebuilt.capabilities == original.capabilities, name
        assert rebuilt.arch_match == original.arch_match, name
        assert rebuilt.runtime == original.runtime, name
        # The cost model is the part that must survive exactly: it is what the optimizer
        # prices, so a dropped field changes which plan wins.
        assert rebuilt.profile == original.profile, name


def test_the_direct_builder_constructs_no_intermediate_typed_record():
    """The structural claim, checked rather than asserted in prose.

    "Direct" is only meaningful if the direct path really does skip the typed record. This
    watches `ChannelManifest.__init__` while the direct builder runs; if it fires, the two
    paths are the same path wearing different names.
    """
    calls = []
    original_init = ChannelManifest.__init__

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original_init(self, *args, **kwargs)

    _name, manifest = next(iter(_manifests()))
    octets = channel_to_jer(manifest)
    ChannelManifest.__init__ = spy
    try:
        channel_direct(octets)
        assert not calls, "the direct builder constructed a ChannelManifest"
        channel_via_typed_value(octets)
        assert calls, "the typed path did NOT construct a ChannelManifest"
    finally:
        ChannelManifest.__init__ = original_init


def test_both_sinks_see_the_same_event_walk():
    """One walk, two consumers — the property that makes the commutation meaningful.

    Recorded explicitly so a future refactor that gives each sink its own reader fails
    here, loudly, rather than silently weakening every test above it.
    """
    seen: list[tuple] = []

    class Recorder(TypedValueSink):
        def member(self, path, value):
            seen.append(path)
            super().member(path, value)

    _name, manifest = next(iter(_manifests()))
    value = decode_bounded(channel_to_jer(manifest), CHANNEL_MANIFEST)
    recorder = Recorder()
    walk(CHANNEL_MANIFEST, value, recorder)

    direct_seen: list[tuple] = []

    class SpyDirect(DirectChannelSink):
        def member(self, path, value):
            direct_seen.append(path)
            super().member(path, value)

    walk(CHANNEL_MANIFEST, value, SpyDirect())
    assert seen == direct_seen and seen, "the two sinks saw different events"


def test_the_walk_visits_members_in_schema_order_not_document_order():
    """§5.2's contract: a generated consumer must see the table-driven event trace.

    A sink keyed on arrival order would be correct for canonical input and wrong for a
    BASIC document that wrote its members in another order. The walk visits what the TYPE
    declares, so both give the same trace — and the direct builder above is allowed to be
    order-dependent because of it.
    """
    _name, manifest = next(iter(_manifests()))
    document = manifest.to_dict()
    shuffled = dict(reversed(list(document.items())))
    assert list(shuffled) != list(document), "the fixture was already symmetric"

    def trace(source):
        seen = []

        class Recorder(TypedValueSink):
            def member(self, path, value):
                seen.append(path)
                super().member(path, value)

        walk(CHANNEL_MANIFEST, source, Recorder())
        return seen

    assert trace(shuffled) == trace(document)
    # And the two orders are the same channel, since BASIC admits both spellings.
    basic = encode_jer(CHANNEL_MANIFEST, shuffled, rules=JerRules.BASIC)
    assert channel_direct(basic, rules=JerRules.BASIC) == channel_direct(channel_to_jer(manifest))


def test_the_typed_sink_rebuilds_the_document_it_walked():
    """The event stream carries enough to reconstruct the value — §5.2's other half."""
    for name, manifest in _manifests():
        value = decode_bounded(channel_to_jer(manifest), CHANNEL_MANIFEST)
        sink = TypedValueSink()
        walk(CHANNEL_MANIFEST, value, sink)
        rebuilt = sink.finish()
        assert encode_jer(CHANNEL_MANIFEST, rebuilt) == encode_jer(CHANNEL_MANIFEST, value), name


# --- the schema beside the hand-written loader ------------------------------------------------


def test_the_schema_names_exactly_the_members_the_writer_emits():
    """The schema and `ChannelManifest.to_dict` must describe one document.

    This is the drift gate. A member the writer emits and the schema does not name would
    be silently dropped by the ASN.1 rail; one the schema names and the writer never emits
    would be a schema for a file this repository does not write.
    """
    _name, manifest = next(iter(_manifests()))
    document = manifest.to_dict()
    declared = {c.name for c in CHANNEL_MANIFEST.components}
    assert declared == set(document), (
        f"schema-only: {sorted(declared - set(document))}, "
        f"writer-only: {sorted(set(document) - declared)}"
    )
    for section, kind in (
        ("codegen", "codegen"),
        ("runtime", "runtime"),
        ("calibration", "calibration"),
        ("profile", "profile"),
    ):
        sub = next(c for c in CHANNEL_MANIFEST.components if c.name == section)
        assert {c.name for c in sub.type.components} == set(document[kind]), section


def test_the_schema_refuses_a_member_the_loader_would_also_refuse():
    """Both rails reject an unknown top-level key, for their own reasons.

    The loader refuses it by an explicit allowed-field set; the schema refuses it because
    a SEQUENCE has the components it has. Two mechanisms, one verdict — which is the point
    of putting a schema beside a hand-written reader rather than in place of it.
    """
    _name, manifest = next(iter(_manifests()))
    document = dict(manifest.to_dict(), surprise=1)
    try:
        encode_jer(CHANNEL_MANIFEST, document)
    except Asn1Error:
        pass
    else:
        raise AssertionError("the schema encoded an undeclared member")
    try:
        ChannelManifest.from_dict(document)
    except ValueError as error:
        assert "surprise" in str(error) or "unknown" in str(error).lower()
    else:
        raise AssertionError("the loader accepted an undeclared member")


def test_no_canonical_omission_produces_a_document_the_loader_refuses():
    """The finding this file's first run produced, pinned so it cannot come back.

    A DEFAULT in an ASN.1 schema is not free: X.690 §11.5, via X.697 §21.2, makes the
    canonical encoder **omit** a component equal to its default. `channel_plugin`'s
    validator requires every declared key to be present, so any DEFAULT whose value
    actually occurs produces canonical JER that the repository's own loader refuses.

    `MemoryTier.capacity` was the case: it defaults to 0 in the dataclass and is read with
    `.get("capacity", 0)`, which made `DEFAULT 0` look faithful — but `profile_to_schema`
    always writes the key, so the document format carries it and the DEFAULT was an ASN.1
    idiom imposed on a file that does not have that shape.

    The general rule this test enforces: for every schema member, the canonical encoding of
    a real manifest must still satisfy the loader. It fires on a zero-valued member, which
    is where a DEFAULT bites and an ordinary round trip does not.
    """
    _name, manifest = next(iter(_manifests()))
    document = manifest.to_dict()
    # Force every numeric leaf that CAN be zero to actually be zero, so any DEFAULT in the
    # schema would omit it and the loader would then see a missing key.
    for tier in document["profile"]["mem_tiers"]:
        tier["capacity"] = 0
    document["profile"]["warp"] = 0
    document["profile"]["cal_gen"] = 0
    document["calibration"]["cal_gen"] = 0
    document["codegen"]["e_machine"] = 0
    document["runtime"]["perf_syscall_nr"] = 0
    canonical = json.loads(encode_jer(CHANNEL_MANIFEST, document, rules=JerRules.CANONICAL))
    # Every key the writer produced is still present after canonical encoding.
    assert set(canonical["profile"]["mem_tiers"][0]) == set(document["profile"]["mem_tiers"][0])
    # And the loader — the one with the strict allowed-field set — accepts it.
    assert ChannelManifest.from_dict(canonical) is not None


def test_a_manifest_arrives_through_the_bounded_reader():
    """A plugin manifest is third-party input, so §4.3's limits apply before a value exists."""
    _name, manifest = next(iter(_manifests()))
    octets = channel_to_jer(manifest)
    assert channel_direct(octets) is not None
    tight = STRICT_LIMITS.tightened(input_bytes=len(octets) - 1)
    try:
        channel_direct(octets, limits=tight)
    except JerBoundedError as error:
        assert error.diagnostic.code.value == "input-too-large"
    else:
        raise AssertionError("the bounded reader ignored its input ceiling")


def test_the_json_the_schema_produces_is_the_json_the_loader_reads():
    """The strongest form of "the two rails agree": one file, both readers.

    The ASN.1 encoder's output is handed to `channel_plugin`'s own `from_dict`, with all of
    its validation. If the schema had reordered, renamed or retyped anything, this is where
    it would show.
    """
    for name, manifest in _manifests():
        document = json.loads(channel_to_jer(manifest))
        assert ChannelManifest.from_dict(document).to_channel() == CHANNELS[name], name


# --- the device manifest -----------------------------------------------------------------------


def test_the_device_manifest_round_trips_through_its_schema():
    from bcir.kbcir.device_manifest import DeviceManifest, MemoryBank
    from bcir.model import Domain

    man = DeviceManifest(
        device="acc0",
        target="x86_avx512",
        cal_gen=7,
        banks=(
            MemoryBank("hbm", Domain.HBM, 1 << 30, 128, 64),
            MemoryBank("ram", Domain.RAM, 1 << 20, 64, 64),
        ),
        distance=((0, 256), (256, 0)),
    )
    octets = encode_jer(DEVICE_MANIFEST, device_manifest_to_value(man))
    assert value_to_device_manifest(decode_jer(octets, DEVICE_MANIFEST)) == man


def test_the_distance_matrix_keeps_its_rows():
    """A matrix whose row length is implied by another field is one a truncated document
    can misread as a different shape, so it is SEQUENCE OF SEQUENCE OF INTEGER."""
    from bcir.kbcir.device_manifest import DeviceManifest, MemoryBank
    from bcir.model import Domain

    banks = tuple(MemoryBank(f"b{i}", Domain.RAM, 1 << 20, 64, 64) for i in range(3))
    distance = ((0, 1, 2), (1, 0, 3), (2, 3, 0))
    man = DeviceManifest(device="d", target="t", cal_gen=1, banks=banks, distance=distance)
    document = json.loads(encode_jer(DEVICE_MANIFEST, device_manifest_to_value(man)))
    assert document["distance"] == [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    assert (
        value_to_device_manifest(
            decode_jer(encode_jer(DEVICE_MANIFEST, device_manifest_to_value(man)), DEVICE_MANIFEST)
        ).distance
        == distance
    )


# --- the §6.2 selection envelope ------------------------------------------------------------------


def _measurements():
    from bcir.asn1.schema import Primitive
    from bcir.asn1.selection import ALL_CANDIDATES, measure_one
    from bcir.asn1.tags import Universal

    kind = Primitive(Universal.INTEGER)
    return [measure_one(c, kind, 42) for c in ALL_CANDIDATES]


def test_the_selection_envelope_records_legality_independently_of_cost():
    """§6.2's two-truth law, made structural rather than promised.

    `legal` and `refusal` are the verdict; `octets` and the two `_ns` members are graded
    measurement. A consumer that reads only the first pair has read the whole verdict, and
    no timing can promote a refusal into a candidate.
    """
    measurements = _measurements()
    envelope = selection_envelope(
        measurements, objective="memory", type_name="INTEGER", selected="DER"
    )
    octets = encode_jer(SELECTION_ENVELOPE, envelope)
    back = decode_jer(octets, SELECTION_ENVELOPE)
    assert back["version"] == SELECTION_ENVELOPE_VERSION
    assert back["selected"] == "DER"
    assert len(back["measurements"]) == len(measurements)
    verdict = {row["candidate"]: row["legal"] for row in back["measurements"]}
    assert verdict == {m.candidate: m.legal for m in measurements}
    # The two kinds of truth are different members, and the schema says which is which.
    names = [c.name for c in MEASUREMENT_TYPE.components]
    assert names.index("legal") < names.index("encode_ns")


def test_an_illegal_candidate_carries_no_octet_count():
    """ "No encoding exists" and "the encoding is zero octets" are different facts.

    `octets` is OPTIONAL rather than DEFAULT 0 so a certificate cannot spell them the same.
    """
    from bcir.asn1.selection import Candidate, measure_one
    from bcir.asn1.schema import Primitive, Sequence
    from bcir.asn1.tags import Universal

    def refuse(kind, value):
        raise Asn1Error("this candidate cannot carry the value")

    broken = Candidate("BROKEN", None, False, refuse, lambda d, k: None, "der")
    measurement = measure_one(broken, Primitive(Universal.INTEGER), 1)
    assert not measurement.legal and measurement.octets is None
    envelope = selection_envelope(
        [measurement], objective="memory", type_name="INTEGER", selected=None
    )
    document = json.loads(encode_jer(SELECTION_ENVELOPE, envelope))
    assert "octets" not in document["measurements"][0]
    assert "selected" not in document, "a certificate named a winner it did not have"
    assert document["measurements"][0]["refusal"]


def test_the_manifest_module_is_a_well_formed_private_enterprise_oid():
    """One module, one identity — and not colliding with the other BCIR modules."""
    from bcir.asn1.artifact_bundle import ARTIFACT_BUNDLE_MODULE_OID
    from bcir.asn1.dialect import DIALECT_MODULE_OID
    from bcir.asn1.streampack import BCIR_ARC, STREAMPACK_MODULE_OID

    assert MANIFEST_MODULE_OID[: len(BCIR_ARC)] == BCIR_ARC
    others = {STREAMPACK_MODULE_OID, ARTIFACT_BUNDLE_MODULE_OID, DIALECT_MODULE_OID}
    assert MANIFEST_MODULE_OID not in others
    assert MODULE.oid == MANIFEST_MODULE_OID
    assert set(MODULE.types) == {"ChannelManifest", "DeviceManifest", "SelectionEnvelope"}


def test_every_manifest_schema_is_canonical_and_idempotent():
    """One value, one canonical spelling — the property a digest over a manifest rests on."""
    _name, manifest = next(iter(_manifests()))
    for kind, value in (
        (CHANNEL_MANIFEST, manifest.to_dict()),
        (
            SELECTION_ENVELOPE,
            selection_envelope(
                _measurements(), objective="memory", type_name="INTEGER", selected="DER"
            ),
        ),
    ):
        raw = encode_jer(kind, value, rules=JerRules.CANONICAL)
        assert (
            encode_jer(
                kind, decode_jer(raw, kind, rules=JerRules.CANONICAL), rules=JerRules.CANONICAL
            )
            == raw
        ), kind.name
