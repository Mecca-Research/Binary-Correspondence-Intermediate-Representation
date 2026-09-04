"""The BCIR-StreamPack ASN.1 module and its DER projection.

The projection is *additive*: the native StreamPack wire format stays frozen and
byte-for-byte unchanged, and DER is a second transfer syntax for the same abstract
value. Three laws make that claim testable, and all three are gated here:

    faithful  : decode_pack(encode_pack(p)) == p
    canonical : encode_pack(p) is DER, and encoding it again returns identical octets
    additive  : abi.encode(decode_pack(encode_pack(abi.decode(b)))) == b

The third is the one that matters for compatibility — a pack can leave BCIR as DER,
be handled by an ASN.1-speaking peer, and come back as the *same native octets*, which
is what keeps StreamPack digests and provenance manifests valid across the round trip.
"""

from __future__ import annotations

from bcir.abi import decode as abi_decode, encode as abi_encode
from bcir.asn1 import Asn1Error, Strictness
from bcir.asn1.der import is_der
from bcir.asn1.streampack import (
    DISPATCH_VALUES,
    MODULE,
    PROJECTION_VERSION,
    STREAMPACK_MODULE_OID,
    decode_pack,
    encode_pack,
    pack_to_value,
)
from bcir.asn1.tlv import decode_one, encode_tlv
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.gem.streampack import LaneSegment, StreamPack
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta


def _packs():
    """One hydrated StreamPack per corpus program — the real artifacts, not fixtures."""
    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        yield name, hydrate(module, optimize(module, h, theta))


def test_projection_round_trips_every_corpus_program():
    """faithful: the abstract value survives DER exactly."""
    count = 0
    for name, pack in _packs():
        assert decode_pack(encode_pack(pack)) == pack, name
        count += 1
    assert count >= 10, f"corpus degenerated to {count} program(s)"


def test_projection_is_der_and_idempotent():
    """canonical: a StreamPack has exactly one DER spelling."""
    for name, pack in _packs():
        raw = encode_pack(pack)
        assert is_der(decode_one(raw)), name
        assert encode_pack(decode_pack(raw)) == raw, name


def test_native_streampack_bytes_survive_the_asn1_round_trip():
    """additive: the frozen wire format is unchanged by a trip through ASN.1."""
    for name, pack in _packs():
        native = abi_encode(pack)
        assert abi_encode(decode_pack(encode_pack(abi_decode(native)))) == native, name


def test_projection_omits_default_valued_components():
    """§11.5: a component equal to its DEFAULT is not encoded.

    This is why the DER projection is *smaller* than the native format on the corpus:
    the native encoder writes every field unconditionally, while DER drops the ones
    carrying their default. Pinned as a property, not a byte count, so it survives a
    corpus change.
    """
    _, pack = next(_packs())
    value = pack_to_value(pack)
    raw = encode_pack(pack)
    # topoGen defaults to 1; a pack that uses the default must not encode the field.
    default_pack = StreamPack(
        source_plan=pack.source_plan,
        topo_gen=1,
        map_gen=0,
        data_gen=0,
        pipeline_depth=1,
        segments=pack.segments,
    )
    smaller = encode_pack(default_pack)
    nondefault = StreamPack(
        source_plan=pack.source_plan,
        topo_gen=7,
        map_gen=0,
        data_gen=0,
        pipeline_depth=1,
        segments=pack.segments,
    )
    assert len(encode_pack(nondefault)) > len(smaller)
    assert decode_pack(smaller).topo_gen == 1  # absent means DEFAULT, not zero
    assert value["version"] == PROJECTION_VERSION
    del raw


def test_module_identity_is_a_well_formed_private_enterprise_oid():
    """The module OID must live in private-enterprise space and encode canonically."""
    from bcir.asn1.values import decode_oid, encode_oid

    assert STREAMPACK_MODULE_OID[:6] == (1, 3, 6, 1, 4, 1), STREAMPACK_MODULE_OID
    raw = encode_oid(STREAMPACK_MODULE_OID)
    assert decode_oid(raw) == STREAMPACK_MODULE_OID
    assert MODULE.oid == STREAMPACK_MODULE_OID


def test_enumerations_reject_values_outside_the_enumeration():
    """X.680 §20: a closed enumeration is the point of ENUMERATED over INTEGER."""
    _, pack = next(_packs())
    segment = pack.segments[0]
    bad = StreamPack(
        source_plan=pack.source_plan,
        segments=[
            LaneSegment(
                name=segment.name,
                claim_id=segment.claim_id,
                phase_id=segment.phase_id,
                lane=segment.lane,
                width=segment.width,
                opcode=segment.opcode,
                reads=segment.reads,
                writes=segment.writes,
                dispatch="not-a-dispatch",
            ),
        ],
    )
    try:
        encode_pack(bad)
        raise AssertionError("accepted a dispatch outside the enumeration")
    except Asn1Error as exc:
        assert "Dispatch" in str(exc), exc
    assert set(DISPATCH_VALUES) == {"core", "pim"}


def test_projection_accepts_ber_and_normalizes_to_der():
    """A peer may send BER; what BCIR stores is always the canonical form."""
    from bcir.asn1 import reencode_as_der

    _, pack = next(_packs())
    der = encode_pack(pack)
    # Re-spell the outer SEQUENCE with the indefinite form — legal BER (§8.1.3.2 b).
    tree = decode_one(der)
    tree.indefinite = True
    ber = _encode_indefinite(tree)
    assert ber != der
    assert decode_pack(ber, strictness=Strictness.BER) == pack
    assert reencode_as_der(ber) == der
    try:
        decode_pack(ber, strictness=Strictness.DER)
        raise AssertionError("DER strictness accepted an indefinite-length pack")
    except Asn1Error:
        pass


def _encode_indefinite(tlv) -> bytes:
    """Re-emit `tlv` with the outer constructed encoding in the indefinite form."""
    from bcir.asn1.tags import encode_tag

    body = b"".join(encode_tlv(child) for child in tlv.children)
    return encode_tag(tlv.tag) + b"\x80" + body + b"\x00\x00"
