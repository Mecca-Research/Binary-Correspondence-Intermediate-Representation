"""The BCIR-ControlPlane ASN.1 module and its DER/OER/JER projections (G14, S3-A).

The projection follows the StreamPack and ExecutionPlan precedents: it is *additive* -- the
native ControlRecordV1 wire format stays frozen and byte-for-byte unchanged, and DER (and
canonical OER, and JER) are further transfer syntaxes for the same abstract value. The laws:

    faithful  : decode_control_der(encode_control_der(r)) == r
    canonical : encode_control_der(r) is DER, and encoding it again returns identical octets
    additive  : encode_control(decode_control_der(encode_control_der(decode_control(b)))) == b
    lawful    : a well-formed ASN.1 document that violates a wire law is refused
"""

from __future__ import annotations

import os
from dataclasses import replace

from bcir.abi.control_abi import decode_control, encode_control, sign_control
from bcir.asn1 import Asn1Error
from bcir.asn1.artifact_bundle import ARTIFACT_BUNDLE_MODULE_OID
from bcir.asn1.control_plane import (
    CONTROL_PLANE_MODULE_OID,
    CONTROL_RECORD,
    MODULE,
    PROJECTION_VERSION,
    decode_control_der,
    decode_control_jer,
    decode_control_oer,
    encode_control_der,
    encode_control_jer,
    encode_control_oer,
    record_to_value,
    value_to_record,
)
from bcir.asn1.der import is_der
from bcir.asn1.execution_plan import EXECUTION_PLAN_MODULE_OID
from bcir.asn1.streampack import STREAMPACK_MODULE_OID
from bcir.asn1.tlv import decode_one
from bcir.gem.control import CONTROL_KINDS, Activate
from bcir.tests.control_fixtures import A_DIGEST, B_DIGEST, record, record_corpus

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ASN1 = os.path.join(_ROOT, "bcir", "asn1", "BCIR-ControlPlane.asn1")


def _records():
    for name, blob in record_corpus():
        yield name, decode_control(blob), blob


def _refused(callback) -> bool:
    try:
        callback()
    except Asn1Error:
        return True
    return False


def test_projection_round_trips_every_corpus_record():
    """faithful: the abstract value -- MAC included -- survives DER exactly."""
    kinds = set()
    for name, rec, _blob in _records():
        assert decode_control_der(encode_control_der(rec)) == rec, name
        kinds.add(rec.kind)
    assert kinds == set(CONTROL_KINDS)


def test_projection_is_der_and_idempotent():
    """canonical: a record has exactly one DER spelling."""
    for name, rec, _blob in _records():
        raw = encode_control_der(rec)
        assert is_der(decode_one(raw)), name
        assert encode_control_der(decode_control_der(raw)) == raw, name


def test_native_bytes_survive_the_projection():
    """additive: the native ControlRecordV1 octets -- MAC and CRC -- come back byte for byte."""
    for name, _rec, blob in _records():
        assert encode_control(decode_control_der(encode_control_der(decode_control(blob)))) == blob
        assert encode_control(decode_control_oer(encode_control_oer(decode_control(blob)))) == blob
        assert (
            encode_control(decode_control_jer(encode_control_jer(decode_control(blob)))) == blob
        ), name


def test_oer_and_jer_are_realizations_over_the_same_type_model():
    for name, rec, _blob in _records():
        oer = encode_control_oer(rec)
        assert decode_control_oer(oer, canonical=True) == rec, name
        assert encode_control_oer(decode_control_oer(oer)) == oer, name
        jer = encode_control_jer(rec)
        assert decode_control_jer(jer) == rec, name


def test_the_compiled_module_encodes_byte_identically_to_the_hand_built_one():
    """The X.680 source is the schema: the front-end's model reproduces the hand-built DER and,
    because the widths are constraints OER encodes by, the hand-built canonical OER too."""
    from bcir.asn1.oer import OerRules, encode_oer
    from bcir.frontends.asn1 import compile_module

    compiled = compile_module(open(_ASN1, encoding="utf-8").read(), "BCIR-ControlPlane.asn1").module
    assert compiled.oid == MODULE.oid
    for name, rec, _blob in _records():
        value = record_to_value(rec)
        hand, parsed = (
            MODULE.encode("ControlRecord", value),
            compiled.encode("ControlRecord", value),
        )
        assert hand == parsed, f"{name}: compiled module produced different DER"
        assert compiled.decode("ControlRecord", parsed) == MODULE.decode("ControlRecord", hand)
        compiled_oer = encode_oer(compiled.types["ControlRecord"], value, rules=OerRules.CANONICAL)
        assert compiled_oer == encode_oer(CONTROL_RECORD, value, rules=OerRules.CANONICAL), name


def test_the_module_has_its_own_arc_and_version():
    assert CONTROL_PLANE_MODULE_OID == (1, 3, 6, 1, 4, 1, 62596, 4)
    oids = {
        CONTROL_PLANE_MODULE_OID,
        EXECUTION_PLAN_MODULE_OID,
        STREAMPACK_MODULE_OID,
        ARTIFACT_BUNDLE_MODULE_OID,
    }
    assert len(oids) == 4
    _name, rec, _blob = next(_records())
    value = record_to_value(rec)
    assert value["version"] == PROJECTION_VERSION == 1
    assert value_to_record({k: v for k, v in value.items() if k != "version"}) == rec
    assert _refused(lambda: value_to_record({**value, "version": 2}))


def test_the_body_alternative_is_the_kind_and_its_tag_the_kind_code():
    for _name, rec, _blob in _records():
        raw = encode_control_der(rec)
        value = MODULE.decode("ControlRecord", raw)
        assert value["body"][0] == rec.kind
    # [9] EXPLICIT wraps the chosen alternative, whose context tag number is the kind code
    _name, rec, _blob = next(r for r in _records() if r[1].kind == "cancel")
    body = next(
        child for child in decode_one(encode_control_der(rec)).children if child.tag.number == 9
    )
    assert body.children[0].tag.number == CONTROL_KINDS.index("cancel") + 1


def test_a_well_formed_document_that_breaks_a_wire_law_is_refused():
    """lawful: the projection holds every decoded value to the codec's laws, so a document
    that is valid ASN.1 cannot smuggle in a record the native decoder would refuse."""
    rec = sign_control(
        record("activate", Activate(A_DIGEST, B_DIGEST), seq=4, expect=2, lease=1), b"k" * 32
    )
    value = record_to_value(rec)
    assert value_to_record(value) == rec
    breaks = {
        "sequence 0": {**value, "sequence": 0},
        "switch not expect + 1": {**value, "generation": 9},
        "lease 0 on a switch": {**value, "lease": 0},
        "all-zero MAC": {**value, "mac": bytes(32)},
        "reason outside the kind's set": {**value, "reason": 3},
        "unlisted scope": {**value, "scope": 5},
        "unchanged artifact": {
            **value,
            "body": ("activate", {"artifactSha256": A_DIGEST, "previousSha256": A_DIGEST}),
        },
        "unknown alternative": {**value, "body": ("galaxy", {})},
    }
    for label, broken in breaks.items():
        assert _refused(lambda broken=broken: value_to_record(broken)), label
    # the same documents are refused through every transfer syntax, not only by the mapper
    raw = MODULE.encode("ControlRecord", breaks["sequence 0"])
    assert _refused(lambda: decode_control_der(raw))
    # an unsigned record has no projection, and neither does one the laws refuse
    assert _refused(lambda: record_to_value(replace(rec, mac=b"")))
    assert _refused(lambda: record_to_value(replace(rec, sequence=0)))


def test_nothing_live_has_one_spelling():
    """An all-zero previousSha256 is the DEFAULT: DER omits it and refuses it present (the
    second spelling), while BER -- which admits it -- decodes the same record."""
    from bcir.asn1.codec import Strictness

    rec = sign_control(record("activate", Activate(A_DIGEST), seq=2, expect=1, lease=1), b"k" * 32)
    raw = encode_control_der(rec)
    assert MODULE.decode("ControlRecord", raw)["body"][1]["previousSha256"] == bytes(32)
    assert bytes(32) not in raw
    absent = {**record_to_value(rec), "body": ("activate", {"artifactSha256": A_DIGEST})}
    assert MODULE.encode("ControlRecord", absent) == raw
    # spell the default explicitly after artifactSha256, fixing every enclosing length
    artifact = bytes([0x80, 0x20]) + A_DIGEST
    insert_at = raw.index(artifact) + len(artifact)
    explicit = bytearray(raw[:insert_at] + bytes([0x81, 0x20]) + bytes(32) + raw[insert_at:])
    for offset in _enclosing_lengths(raw, insert_at):
        explicit[offset] += 34
    assert decode_control_der(bytes(explicit), strictness=Strictness.BER) == rec
    assert _refused(lambda: decode_control_der(bytes(explicit)))


def _enclosing_lengths(raw: bytes, position: int) -> list[int]:
    """The (short-form) length octets of every constructed TLV whose contents span
    `position`; the fixture is short-form throughout, which the walk asserts."""
    offsets: list[int] = []

    def walk(start: int, end: int) -> None:
        pos = start
        while pos < end:
            length = raw[pos + 1]
            assert length < 0x80
            body_start, body_end = pos + 2, pos + 2 + length
            if raw[pos] & 0x20 and body_start <= position <= body_end:
                offsets.append(pos + 1)
                walk(body_start, body_end)
            pos = body_end

    walk(0, len(raw))
    return offsets
