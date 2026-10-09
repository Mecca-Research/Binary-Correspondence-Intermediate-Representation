"""The ITS security-envelope study: ASN.1 against hand-made binary, on the same content.

Bittl, Gonzalez, Spähn and Heidrich, "Performance Comparison of Data Serialization Schemes for
ETSI ITS Car-to-X Communication Systems" (Int. J. Advances in Telecommunications 8(1&2), 2015)
measured the ETSI TS 103 097 security envelope in two encodings: the V1.1.1 binary format
(a TLS-style presentation language with its own byte rules) and an ASN.1 UPER encoding of a
draft V2.1.1 module. Their Table VI found binary shorter in three of four cases (222/233/230
against 240/249/247 octets) and ASN.1 shorter in one (88 against 96), their runtime figures
found binary "significantly faster" on the security envelope, and they concluded that its
standardization "should reconsider the recent shift from binary encoding towards usage of
ASN.1". That is a statement about one draft schema, one encoding rule and one table-driven
codec. This module is the first half of re-asking the question with all three as variables: the
content is fixed and the encoding is what varies.

**One abstract value, every encoding.** `ITS-SecuredMessage-V111.asn1` transcribes the V1.1.1
data model into ASN.1 one-to-one. `encode_binary` encodes a value of that module with the V1.1.1
binary rules; `bcir.asn1.per` / `oer` / DER encode the very same value. A byte of difference is
therefore a byte of ENCODING. Nothing is re-modelled between the columns.

**A reconstruction, held to the published numbers.** ETSI's server is not reachable from the
environment this was written in, so the binary rules below are TS 103 097 V1.1.1 as this study
reconstructs it, and the reconstruction is checked against the paper rather than trusted:
`PAPER_TABLE_VI` holds the paper's binary column and `envelopes()` must reproduce all four
lengths exactly (`test_its_security.py`). The pins, each fixed by a published number:

* `SecuredMessage` carries `security_profile` and a single `Payload` beside `protocol_version`
  and the two field lists -- five top-level fields (Table III's level-one count is 5), and the
  only arrangement that makes the CAM envelope without a certificate 96 octets;
* `message_type` is a uint16 header field: profiles 2 and 3 differ only in it (the paper, V-C)
  and their binary lengths differ by 3 (233 against 230);
* the DENM profile adds `generation_location`, a 10-octet ThreeDLocation: 233 against 222;
* a `<var>` vector is prefixed by its length in octets as an IntX, so the certificate cases'
  header list (over 127 octets) takes a two-octet length, and the certificate is then 133
  octets -- which a minimal V1.1.1 authorization ticket is exactly: a compressed-y verification
  key (the paper's Listing 1), an assurance level, one ITS-AID with a 3-octet SSP, a start and
  end time, and an x-only ECDSA signature.

Everything else is the V1.1.1 presentation language's plain reading: big-endian uintN, a uint8
before every `select`, fixed-size opaques raw.
"""

from __future__ import annotations

import hashlib
from functools import cache

from . import module_source
from .tags import Asn1Error

TRANSCRIPTION_MODULE = "ITS-SecuredMessage-V111.asn1"
TMSAO_MODULE = "ITS-SecuredMessage-TMSAO.asn1"

#: Table VI of the paper: encoded length in octets of the security envelope, per profile.
#: Profile 1 is the CAM profile (with an 8-octet certificate digest, or the certificate itself),
#: profile 2 the DENM profile and profile 3 the generic one. EXI-opt is the paper's flattened
#: schema variant.
PAPER_TABLE_VI = {
    "binary": {"p1": 96, "p1cert": 222, "p2": 233, "p3": 230},
    "asn1_uper_draft": {"p1": 88, "p1cert": 240, "p2": 249, "p3": 247},
    "protobuf": {"p1": 133, "p1cert": 306, "p2": 318, "p3": 312},
    "exi": {"p1": 90, "p1cert": 210, "p2": 215, "p3": 213},
    "exi_opt": {"p1": 87, "p1cert": 201, "p2": 206, "p3": 204},
}

#: The four envelopes, in the paper's order.
PROFILES = ("p1", "p1cert", "p2", "p3")


@cache
def transcription():
    """The compiled `ITS-SecuredMessage-V111` module (`LoweredModule`)."""
    from bcir.frontends.asn1 import compile_module

    return compile_module(module_source(TRANSCRIPTION_MODULE), TRANSCRIPTION_MODULE)


def transcription_type(name: str = "SecuredMessage"):
    return transcription().module.types[name]


@cache
def tmsao():
    """The compiled `ITS-SecuredMessage-TMSAO` module: the same value space, field lists as
    sets of OPTIONAL components (see the module's header and `to_tmsao`)."""
    from bcir.frontends.asn1 import compile_module

    return compile_module(module_source(TMSAO_MODULE), TMSAO_MODULE)


def tmsao_type(name: str = "SecuredMessage"):
    return tmsao().module.types[name]


# --- the V1.1.1 binary rules -------------------------------------------------------------------

#: The uint8 type code V1.1.1 writes before each `select`, per CHOICE of the transcription and
#: per alternative. The alternatives' ROOT order in the module is the order of these codes, so a
#: code is never inferred from a position.
TYPE_CODES = {
    "HeaderField": {
        "generationTime": 0,
        "generationTimeWithStandardDeviation": 1,
        "expiration": 2,
        "generationLocation": 3,
        "requestUnrecognizedCertificate": 4,
        "messageType": 5,
        "signerInfo": 128,
        "recipientInfo": 129,
        "encryptionParameters": 130,
    },
    "SignerInfo": {
        "self": 0,
        "certificateDigestWithEcdsap256": 1,
        "certificate": 2,
        "certificateChain": 3,
        "certificateDigestWithOtherAlgorithm": 4,
    },
    "SubjectAttribute": {
        "verificationKey": 0,
        "encryptionKey": 1,
        "assuranceLevel": 2,
        "reconstructionValue": 3,
        "itsAidList": 32,
        "itsAidSspList": 33,
    },
    "PublicKey": {"ecdsaNistp256WithSha256": 0, "eciesNistp256": 1},
    "EccPoint": {
        "xCoordinateOnly": 0,
        "compressedLsbY0": 2,
        "compressedLsbY1": 3,
        "uncompressed": 4,
    },
    "ValidityRestriction": {
        "timeEnd": 0,
        "timeStartAndEnd": 1,
        "timeStartAndDuration": 2,
        "region": 3,
    },
    "Signature": {"ecdsaNistp256WithSha256": 0},
    "Payload": {
        "unsecured": 0,
        "signed": 1,
        "encrypted": 2,
        "signedExternal": 3,
        "signedAndEncrypted": 4,
    },
    "TrailerField": {"signature": 1},
}

#: The widest IntX this codec writes: seven extra octets after a 0xFE prefix (56 value bits).
INTX_MAX = (1 << 56) - 1


def v111_rule():
    """The V1.1.1 binary rules as a `cgen.ByteRule`: the same type codes and IntX bound this
    module's Python codec uses, so the generated C and `encode_binary` cannot disagree on a
    table -- there is only one."""
    from .cgen import ByteRule

    return ByteRule(name="v111", type_codes=TYPE_CODES, intx_max=INTX_MAX)


def encode_intx(value: int) -> bytes:
    """IntX: `n` leading one bits, a zero bit, then the value in the remaining 7 + 7n bits;
    the shortest such spelling is the only one this encoder writes."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= INTX_MAX:
        raise Asn1Error(f"IntX: {value!r} is not an integer in 0..2^56-1")
    n = 0
    while value >= 1 << (7 + 7 * n):
        n += 1
    prefix = ((0xFF << (8 - n)) & 0xFF) if n else 0  # n leading ones, then the zero bit
    total = n + 1
    raw = value.to_bytes(total, "big")
    return bytes([raw[0] | prefix]) + raw[1:]


def decode_intx(data: bytes, at: int) -> tuple[int, int]:
    """(value, next offset). Refuses a truncated value and a non-minimal spelling: a decoder
    that accepted both would read two byte strings as one value (Class A, laws.md)."""
    if at >= len(data):
        raise Asn1Error("IntX: truncated before its first octet", at)
    first = data[at]
    n = 0
    while n < 8 and first & (0x80 >> n):
        n += 1
    if n > 7:
        raise Asn1Error("IntX: an 0xFF prefix is wider than this codec carries", at)
    if at + 1 + n > len(data):
        raise Asn1Error("IntX: truncated", at)
    value = first & (0x7F >> n)
    for b in data[at + 1 : at + 1 + n]:
        value = (value << 8) | b
    if n and value < 1 << (7 + 7 * (n - 1)):
        raise Asn1Error("IntX: not the shortest spelling of its value", at)
    return value, at + 1 + n


def _range(kind):
    """(lower, upper) of an INTEGER's root constraint; upper None for (0..MAX)."""
    from .per import _value_bounds

    (low, high), _ext = _value_bounds(kind)
    return low, high


def _int_width(kind) -> int | None:
    """Octets of the fixed-width integer V1.1.1 uses for this INTEGER, or None for IntX."""
    low, high = _range(kind)
    if low == 0 and high is None:
        return None
    for width, (lo, hi) in (
        (1, (0, 255)),
        (2, (0, 65535)),
        (4, (0, 4294967295)),
        (8, (0, 18446744073709551615)),
        (4, (-2147483648, 2147483647)),
    ):
        if (low, high) == (lo, hi):
            return width
    raise Asn1Error(f"binary: INTEGER ({low}..{high}) has no V1.1.1 width")


def _fixed_size(kind) -> int | None:
    from .per import _size_bounds

    low, high, _ext = _size_bounds(kind)
    return low if high == low else None


def _type_name(kind) -> str:
    return getattr(kind, "name", "?")


def _encode(kind, value, out: bytearray) -> None:
    from .schema import Choice, Primitive, Sequence, SequenceOf, resolve
    from .tags import Universal

    kind = resolve(kind)
    if isinstance(kind, Sequence):
        for comp in kind.components:
            if comp.name not in value:
                raise Asn1Error(f"binary: {kind.name}.{comp.name} is absent")
            _encode(comp.type, value[comp.name], out)
        return
    if isinstance(kind, SequenceOf):
        body = bytearray()
        for item in value:
            _encode(kind.element, item, body)
        out += encode_intx(len(body))
        out += body
        return
    if isinstance(kind, Choice):
        name, inner = value
        codes = TYPE_CODES.get(_type_name(kind))
        if codes is None or name not in codes:
            raise Asn1Error(f"binary: {kind.name} has no V1.1.1 type code for {name!r}")
        out.append(codes[name])
        alt = next((c for c in kind.alternatives if c.name == name), None)
        if alt is None:
            raise Asn1Error(f"binary: {kind.name} has no alternative {name!r}")
        _encode(alt.type, inner, out)
        return
    if isinstance(kind, Primitive):
        if kind.universal == Universal.NULL:
            return
        if kind.universal == Universal.ENUMERATED:  # an enum is one octet in V1.1.1
            if isinstance(value, bool) or value not in {n for _name, n in kind.enumeration}:
                raise Asn1Error(f"binary: {value!r} is not a root value of {kind.name}")
            out.append(value)
            return
        if kind.universal == Universal.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise Asn1Error(f"binary: {kind.name} needs an int, got {type(value).__name__}")
            width = _int_width(kind)
            if width is None:
                out += encode_intx(value)
                return
            low, high = _range(kind)
            if not low <= value <= high:
                raise Asn1Error(f"binary: {value} is outside {kind.name} ({low}..{high})")
            out += value.to_bytes(width, "big", signed=low < 0)
            return
        if kind.universal == Universal.OCTET_STRING:
            size = _fixed_size(kind)
            data = bytes(value)
            if size is None:
                out += encode_intx(len(data))
            elif len(data) != size:
                raise Asn1Error(f"binary: opaque[{size}] given {len(data)} octets")
            out += data
            return
    raise Asn1Error(f"binary: {type(kind).__name__} {getattr(kind, 'name', '')} is not in V1.1.1")


def _decode(kind, data: bytes, at: int):
    from .schema import Choice, Primitive, Sequence, SequenceOf, resolve
    from .tags import Universal

    kind = resolve(kind)
    if isinstance(kind, Sequence):
        value = {}
        for comp in kind.components:
            value[comp.name], at = _decode(comp.type, data, at)
        return value, at
    if isinstance(kind, SequenceOf):
        length, at = decode_intx(data, at)
        end = at + length
        if end > len(data):
            raise Asn1Error("binary: a vector runs past the end of the data", at)
        items = []
        while at < end:
            item, at = _decode(kind.element, data[:end], at)
            items.append(item)
        return items, at
    if isinstance(kind, Choice):
        if at >= len(data):
            raise Asn1Error(f"binary: {kind.name} truncated before its type code", at)
        code = data[at]
        names = {v: k for k, v in TYPE_CODES[_type_name(kind)].items()}
        if code not in names:
            raise Asn1Error(f"binary: {kind.name} has no alternative with type code {code}", at)
        alt = next(c for c in kind.alternatives if c.name == names[code])
        inner, at = _decode(alt.type, data, at + 1)
        return (names[code], inner), at
    if isinstance(kind, Primitive):
        if kind.universal == Universal.NULL:
            return None, at
        if kind.universal == Universal.ENUMERATED:
            if at >= len(data):
                raise Asn1Error("binary: enum truncated", at)
            if data[at] not in {n for _name, n in kind.enumeration}:
                raise Asn1Error(f"binary: enum value {data[at]} is not in {kind.name}", at)
            return data[at], at + 1
        if kind.universal == Universal.INTEGER:
            width = _int_width(kind)
            if width is None:
                return decode_intx(data, at)
            if at + width > len(data):
                raise Asn1Error("binary: integer truncated", at)
            low, _high = _range(kind)
            return int.from_bytes(data[at : at + width], "big", signed=low < 0), at + width
        if kind.universal == Universal.OCTET_STRING:
            size = _fixed_size(kind)
            if size is None:
                size, at = decode_intx(data, at)
            if at + size > len(data):
                raise Asn1Error("binary: opaque truncated", at)
            return bytes(data[at : at + size]), at + size
    raise Asn1Error(f"binary: {type(kind).__name__} is not in V1.1.1")


def encode_binary(value, type_name: str = "SecuredMessage") -> bytes:
    """`value` (a value of the transcription module) in the V1.1.1 binary encoding."""
    out = bytearray()
    _encode(transcription_type(type_name), value, out)
    return bytes(out)


def decode_binary(data: bytes, type_name: str = "SecuredMessage"):
    """The inverse of `encode_binary`; refuses trailing octets and every malformed field."""
    value, at = _decode(transcription_type(type_name), bytes(data), 0)
    if at != len(data):
        raise Asn1Error(f"binary: {len(data) - at} octets after the {type_name}", at)
    return value


# --- the TMSAO schema: the same values with the prose rules in the type ------------------------

#: Each V1.1.1 field list and the TMSAO SEQUENCE that carries it as a set: (transcription
#: CHOICE, TMSAO SEQUENCE). The SEQUENCE's component order is the canonical order the prose
#: fixes, and `test_its_security` holds it to `TYPE_CODES` (signer information first, the rest
#: ascending) so the two modules cannot drift apart.
FIELD_SETS = {
    "HeaderField": "HeaderFields",
    "TrailerField": "TrailerFields",
    "SubjectAttribute": "SubjectAttributes",
    "ValidityRestriction": "ValidityRestrictions",
}


def canonical_order(choice: str) -> tuple[str, ...]:
    """The order V1.1.1's prose puts a field list in: the TMSAO SEQUENCE's components."""
    from .schema import resolve

    return tuple(c.name for c in resolve(tmsao_type(FIELD_SETS[choice])).components)


def _as_set(choice: str, items) -> dict:
    """A V1.1.1 field list as the set it denotes. A list the prose forbids -- a type twice, or
    out of order -- is refused: reordering it would map two different octet strings to one
    value, and dropping a duplicate would lose content."""
    order = canonical_order(choice)
    out = {}
    last = -1
    for name, inner in items:
        if name not in order:
            raise Asn1Error(f"TMSAO: {choice} alternative {name!r} has no canonical position")
        at = order.index(name)
        if at <= last:
            raise Asn1Error(
                f"TMSAO: {choice} list is not in V1.1.1's canonical order with each type at "
                f"most once ({name!r} after {order[last]!r})"
            )
        last = at
        out[name] = inner
    return out


def _as_list(choice: str, fields: dict) -> list:
    order = canonical_order(choice)
    extra = set(fields) - set(order)
    if extra:
        raise Asn1Error(f"TMSAO: {sorted(extra)} are not {FIELD_SETS[choice]} components")
    return [(name, fields[name]) for name in order if name in fields]


def _signer_to(signer, convert):
    name, inner = signer
    if name == "certificate":
        return name, convert(inner)
    if name == "certificateChain":
        return name, [convert(c) for c in inner]
    return name, inner


def _certificate_to_tmsao(cert: dict) -> dict:
    out = dict(cert)
    out["signerInfo"] = _signer_to(cert["signerInfo"], _certificate_to_tmsao)
    out["subjectAttributes"] = _as_set("SubjectAttribute", cert["subjectAttributes"])
    out["validityRestrictions"] = _as_set("ValidityRestriction", cert["validityRestrictions"])
    return out


def _certificate_from_tmsao(cert: dict) -> dict:
    out = dict(cert)
    out["signerInfo"] = _signer_to(cert["signerInfo"], _certificate_from_tmsao)
    out["subjectAttributes"] = _as_list("SubjectAttribute", cert["subjectAttributes"])
    out["validityRestrictions"] = _as_list("ValidityRestriction", cert["validityRestrictions"])
    return out


def to_tmsao(message: dict) -> dict:
    """A `SecuredMessage` of the transcription as the TMSAO module's value of the same content."""
    headers = _as_set("HeaderField", message["headerFields"])
    if "signerInfo" in headers:
        headers["signerInfo"] = _signer_to(headers["signerInfo"], _certificate_to_tmsao)
    return {
        "protocolVersion": message["protocolVersion"],
        "securityProfile": message["securityProfile"],
        "headerFields": headers,
        "payloadField": message["payloadField"],
        "trailerFields": _as_set("TrailerField", message["trailerFields"]),
    }


def from_tmsao(message: dict) -> dict:
    """The inverse of `to_tmsao`."""
    headers = dict(message["headerFields"])
    if "signerInfo" in headers:
        headers["signerInfo"] = _signer_to(headers["signerInfo"], _certificate_from_tmsao)
    return {
        "protocolVersion": message["protocolVersion"],
        "securityProfile": message["securityProfile"],
        "headerFields": _as_list("HeaderField", headers),
        "payloadField": message["payloadField"],
        "trailerFields": _as_list("TrailerField", message["trailerFields"]),
    }


# --- the paper's four envelopes ----------------------------------------------------------------


def _octets(label: str, size: int) -> bytes:
    """Deterministic stand-ins for keys, digests and signature halves: what the encodings
    carry, never what they inspect (every one is a fixed-size opaque)."""
    out = b""
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{label}/{counter}".encode()).digest()
        counter += 1
    return out[:size]


#: 2015-01-15 12:00:00 UTC as TS 103 097 counts it (seconds since 2004-01-01, leap seconds
#: aside), and the same instant in microseconds for Time64.
GENERATION_TIME_S = 348_408_000
GENERATION_TIME_US = GENERATION_TIME_S * 1_000_000 + 123_456
#: Fraunhofer ESK, Munich, in TS 103 097's 1/10 micro-degree units.
LATITUDE = 481_374_000
LONGITUDE = 115_755_000
#: ETSI ITS message identifiers: DENM is 1, CAM is 2.
MESSAGE_TYPE = {"cam": 2, "denm": 1}
#: The CA basic service's ITS-AID, and its 3-octet SSP (version, two permission octets).
CAM_ITS_AID = 36
#: SubjectType's numbers (an ENUMERATED value is its number on every BCIR rail).
SUBJECT_TYPE = {"enrollmentCredential": 0, "authorizationTicket": 1, "authorizationAuthority": 2}


def _signature(label: str):
    return (
        "ecdsaNistp256WithSha256",
        {"r": ("xCoordinateOnly", _octets(label + "/r", 32)), "s": _octets(label + "/s", 32)},
    )


def authorization_ticket():
    """The minimal V1.1.1 authorization ticket the profile-1-with-certificate envelope carries
    (133 octets in the binary encoding)."""
    return {
        "version": 1,
        "signerInfo": ("certificateDigestWithEcdsap256", _octets("aa-digest", 8)),
        "subjectInfo": {"subjectType": SUBJECT_TYPE["authorizationTicket"], "subjectName": b""},
        "subjectAttributes": [
            (
                "verificationKey",
                ("ecdsaNistp256WithSha256", ("compressedLsbY0", _octets("at-key", 32))),
            ),
            ("assuranceLevel", 0x60),
            (
                "itsAidSspList",
                [{"itsAid": CAM_ITS_AID, "serviceSpecificPermissions": b"\x01\xff\xfc"}],
            ),
        ],
        "validityRestrictions": [
            (
                "timeStartAndEnd",
                {
                    "startValidity": GENERATION_TIME_S - 86_400,
                    "endValidity": GENERATION_TIME_S + 6 * 86_400,
                },
            ),
        ],
        "signature": _signature("aa-signature"),
    }


def envelope(profile: str):
    """The security envelope of one Table VI column entry, as a value of the transcription.
    Mandatory fields only, as in the paper (V-A), with the one-octet dummy payload V1.1.1
    requires."""
    with_cert = profile != "p1"
    signer = (
        ("certificate", authorization_ticket())
        if with_cert
        else ("certificateDigestWithEcdsap256", _octets("at-digest", 8))
    )
    headers = [("signerInfo", signer), ("generationTime", GENERATION_TIME_US)]
    if profile in ("p2", "p3"):
        headers.append(
            (
                "generationLocation",
                {"latitude": LATITUDE, "longitude": LONGITUDE, "elevation": b"\x02\x0a"},
            )
        )
    if profile in ("p1", "p1cert"):
        headers.append(("messageType", MESSAGE_TYPE["cam"]))
    elif profile == "p2":
        headers.append(("messageType", MESSAGE_TYPE["denm"]))
    return {
        "protocolVersion": 1,
        "securityProfile": {"p1": 1, "p1cert": 1, "p2": 2, "p3": 3}[profile],
        "headerFields": headers,
        "payloadField": ("signed", b"\x00"),
        "trailerFields": [("signature", _signature(f"{profile}/signature"))],
    }


def envelopes() -> dict:
    """Every Table VI envelope, by profile key."""
    return {p: envelope(p) for p in PROFILES}


__all__ = [
    "FIELD_SETS",
    "PAPER_TABLE_VI",
    "PROFILES",
    "TMSAO_MODULE",
    "TRANSCRIPTION_MODULE",
    "TYPE_CODES",
    "authorization_ticket",
    "canonical_order",
    "decode_binary",
    "decode_intx",
    "encode_binary",
    "encode_intx",
    "envelope",
    "envelopes",
    "from_tmsao",
    "tmsao",
    "tmsao_type",
    "to_tmsao",
    "transcription",
    "transcription_type",
    "v111_rule",
]
