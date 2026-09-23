"""BCIR-ControlPlane -- the ASN.1 module and DER/OER/JER projection of a ControlRecordV1.

The control plane as bytes (G14, staged plan S3-A) has a frozen native wire format
(`bcir/abi/control_abi.py`, `docs/kernel/BCIR_CONTROL_PLANE_ABI.md`); this module follows
the StreamPack and ExecutionPlan precedents (`bcir/asn1/streampack.py`,
`bcir/asn1/execution_plan.py`) and adds a **second transfer syntax for the same abstract
value** -- an X.680 module and its DER encoding, plus the canonical-OER and JER realizations
over the same type model -- so a record can cross a boundary that speaks ASN.1 and come back
byte-identical:

    decode_control_der(encode_control_der(r)) == r                               (faithful)
    encode_control(decode_control_der(encode_control_der(decode_control(b)))) == b  (native)

The non-obvious choices, stated:

* **The body is a CHOICE and its alternative IS the kind.** The native header carries the
  kind beside a body whose layout it selects; carrying both here would be a second spelling
  of one fact, and a decoder would have to refuse the pair that disagreed. The alternatives'
  tag numbers are the native kind codes (lease [1] ... cancel [6]).
* **Derived fields are not carried**: magic, flags, body_len and the capability (the kind's
  bit) follow from the kind, and the CRC from the octets. The MAC **is** carried -- it is
  authority, not framing -- which is what lets the native octets survive the round trip.
* **The widths are constraints**: `Uint32` / `Uint64` / `Digest ::= OCTET STRING (SIZE (32))`.
  Invisible to DER; canonical OER encodes them fixed-width, so the OER form is the native
  layout's close cousin rather than a length-prefixed one.
* **Defaults mirror the native zeros**, including an all-zero `previousSha256` (nothing
  live), so DER omits them (X.690 §11.5) and refuses a present default: one spelling each.
* **The laws are the codec's.** `value_to_record` holds every decoded value to
  `validate_control`, so a document that is well-formed ASN.1 and violates a wire law (a
  sequence of 0, a switch whose generation is not expect + 1, an all-zero MAC, ...) is refused
  here with the status the native decoder would name.
"""

from __future__ import annotations

from .codec import Strictness
from .constraints import SingleValue, Size, ValueRange
from .schema import Choice, Component, Module, Primitive, Sequence
from .tags import Asn1Error, Universal

BCIR_ARC: tuple[int, ...] = (1, 3, 6, 1, 4, 1, 62596)
CONTROL_PLANE_MODULE_OID: tuple[int, ...] = (*BCIR_ARC, 4)

#: Bumped only when the ASN.1 module changes shape; independent of the native version.
PROJECTION_VERSION = 1

_U32 = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, (1 << 32) - 1))
_U64 = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, (1 << 64) - 1))
_DIGEST = Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(SingleValue(32)))
_VERSION = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(1, 65535))
_REASON = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))

SCOPE = Primitive(
    Universal.ENUMERATED,
    "Scope",
    enumeration=(("module", 0), ("resource", 1), ("mapping", 2), ("session", 3), ("channel", 4)),
)

LEASE_GRANT = Sequence(
    (
        Component("leaseId", _U64, tag=0),
        Component("granted", _U64, tag=1),
        Component("issuedEpoch", _U64, tag=2, default=0),
        Component("expiryEpoch", _U64, tag=3),
        Component("holder", _U64, tag=4),
    ),
    name="LeaseGrant",
)
GENERATION_SWITCH = Sequence(
    (
        Component("mapGen", _U32, tag=0, default=0),
        Component("dataGen", _U32, tag=1, default=0),
        Component("topoGen", _U32, tag=2, default=0),
        Component("registryDigest", _DIGEST, tag=3),
    ),
    name="GenerationSwitch",
)
QUIESCE = Sequence((Component("drainDeadline", _U64, tag=0, default=0),), name="Quiesce")
ACTIVATE = Sequence(
    (
        Component("artifactSha256", _DIGEST, tag=0),
        Component("previousSha256", _DIGEST, tag=1, default=bytes(32)),
    ),
    name="Activate",
)
ROLLBACK = Sequence(
    (
        Component("restoreSha256", _DIGEST, tag=0),
        Component("rollbackToken", _DIGEST, tag=1),
    ),
    name="Rollback",
)
CANCEL = Sequence(
    (
        Component("firstSequence", _U64, tag=0),
        Component("lastSequence", _U64, tag=1),
    ),
    name="Cancel",
)

#: The alternatives in native kind-code order; each one's tag number is its kind code.
BODY = Choice(
    (
        Component("lease", LEASE_GRANT, tag=1),
        Component("generation", GENERATION_SWITCH, tag=2),
        Component("quiesce", QUIESCE, tag=3),
        Component("activate", ACTIVATE, tag=4),
        Component("rollback", ROLLBACK, tag=5),
        Component("cancel", CANCEL, tag=6),
    ),
    name="Body",
)

CONTROL_RECORD = Sequence(
    (
        Component("version", _VERSION, tag=0, default=1),
        Component("scope", SCOPE, tag=1, default=0),
        Component("reason", _REASON, tag=2, default=0),
        Component("subject", _U64, tag=3, default=0),
        Component("generation", _U32, tag=4, default=0),
        Component("expect", _U32, tag=5, default=0),
        Component("boundary", _U64, tag=6, default=0),
        Component("sequence", _U64, tag=7),
        Component("lease", _U64, tag=8, default=0),
        # X.680 31.2.7: a tagged CHOICE is tagged EXPLICITly, whatever the module default
        Component("body", BODY, tag=9, explicit=True),
        Component("mac", _DIGEST, tag=10),
    ),
    name="ControlRecord",
)

MODULE = Module(
    "BCIR-ControlPlane",
    CONTROL_PLANE_MODULE_OID,
    {
        "ControlRecord": CONTROL_RECORD,
        "Body": BODY,
        "LeaseGrant": LEASE_GRANT,
        "GenerationSwitch": GENERATION_SWITCH,
        "Quiesce": QUIESCE,
        "Activate": ACTIVATE,
        "Rollback": ROLLBACK,
        "Cancel": CANCEL,
    },
)


# --- projection: ControlRecord <-> the ASN.1 value -----------------------------------


def _body_value(record) -> dict:
    b = record.body
    if record.kind == "lease":
        return {
            "leaseId": b.lease_id,
            "granted": b.granted,
            "issuedEpoch": b.issued_epoch,
            "expiryEpoch": b.expiry_epoch,
            "holder": b.holder,
        }
    if record.kind == "generation":
        return {
            "mapGen": b.map_gen,
            "dataGen": b.data_gen,
            "topoGen": b.topo_gen,
            "registryDigest": bytes(b.registry_digest),
        }
    if record.kind == "quiesce":
        return {"drainDeadline": b.drain_deadline}
    if record.kind == "activate":
        return {
            "artifactSha256": bytes(b.artifact_sha256),
            "previousSha256": bytes(b.previous_sha256),
        }
    if record.kind == "rollback":
        return {"restoreSha256": bytes(b.restore_sha256), "rollbackToken": bytes(b.rollback_token)}
    return {"firstSequence": b.first_sequence, "lastSequence": b.last_sequence}


def record_to_value(record) -> dict:
    """The ASN.1 abstract value for a signed `gem.control.ControlRecord`. Refuses a record
    the wire laws refuse, and an unsigned one: the projection carries authority, not drafts."""
    from ..abi.control_abi import ControlError, validate_control
    from ..gem.control import CONTROL_SCOPES, REASONS

    try:
        validate_control(record)
    except ControlError as exc:
        raise Asn1Error(f"not a valid control record ({exc.status}): {exc}") from exc
    if not record.mac:
        raise Asn1Error("an unsigned control record has no projection (sign it first)")
    return {
        "version": PROJECTION_VERSION,
        "scope": CONTROL_SCOPES.index(record.scope),
        "reason": REASONS[record.kind].index(record.reason),
        "subject": record.subject,
        "generation": record.generation,
        "expect": record.expect,
        "boundary": record.boundary,
        "sequence": record.sequence,
        "lease": record.lease,
        "body": (record.kind, _body_value(record)),
        "mac": bytes(record.mac),
    }


def value_to_record(value: dict):
    """The inverse of `record_to_value`, holding the result to the codec's wire laws (imports
    the GEM types lazily -- cold organ)."""
    from ..abi.control_abi import ControlError, validate_control
    from ..gem.control import (
        CONTROL_SCOPES,
        REASONS,
        Activate,
        Cancel,
        ControlRecord,
        GenerationSwitch,
        LeaseGrant,
        Quiesce,
        Rollback,
    )

    version = value.get("version", 1)
    if version != PROJECTION_VERSION:
        raise Asn1Error(
            f"BCIR-ControlPlane projection version {version} is not {PROJECTION_VERSION}"
        )
    kind, b = value["body"]
    if kind == "lease":
        body = LeaseGrant(
            b["leaseId"], b["granted"], b.get("issuedEpoch", 0), b["expiryEpoch"], b["holder"]
        )
    elif kind == "generation":
        body = GenerationSwitch(
            b.get("mapGen", 0), b.get("dataGen", 0), b.get("topoGen", 0), b["registryDigest"]
        )
    elif kind == "quiesce":
        body = Quiesce(b.get("drainDeadline", 0))
    elif kind == "activate":
        body = Activate(b["artifactSha256"], b.get("previousSha256", bytes(32)))
    elif kind == "rollback":
        body = Rollback(b["restoreSha256"], b["rollbackToken"])
    elif kind == "cancel":
        body = Cancel(b["firstSequence"], b["lastSequence"])
    else:
        raise Asn1Error(f"Body alternative {kind!r} is not a control kind")
    scope = value.get("scope", 0)
    reason = value.get("reason", 0)
    if not 0 <= scope < len(CONTROL_SCOPES):
        raise Asn1Error(
            f"Scope value {scope} is not enumerated (X.680 20.4: a decoder shall reject an "
            f"unlisted enumeration value)"
        )
    if not 0 <= reason < len(REASONS[kind]):
        raise Asn1Error(f"reason {reason} is not one of {kind}'s {REASONS[kind]}")
    record = ControlRecord(
        kind=kind,
        scope=CONTROL_SCOPES[scope],
        subject=value.get("subject", 0),
        generation=value.get("generation", 0),
        expect=value.get("expect", 0),
        boundary=value.get("boundary", 0),
        sequence=value["sequence"],
        lease=value.get("lease", 0),
        body=body,
        reason=REASONS[kind][reason],
        mac=bytes(value["mac"]),
    )
    try:
        validate_control(record)
    except ControlError as exc:
        raise Asn1Error(f"not a valid control record ({exc.status}): {exc}") from exc
    return record


def encode_control_der(record) -> bytes:
    """DER octets for a ControlRecord under the BCIR-ControlPlane module."""
    return MODULE.encode("ControlRecord", record_to_value(record))


def decode_control_der(data: bytes, *, strictness: Strictness = Strictness.DER):
    """Recover a ControlRecord from its DER projection (BER admitted on request)."""
    return value_to_record(MODULE.decode("ControlRecord", data, strictness=strictness))


def encode_control_oer(record) -> bytes:
    """CANONICAL-OER octets over the same type model (the widths fixed by the constraints)."""
    from .oer import OerRules, encode_oer

    return encode_oer(CONTROL_RECORD, record_to_value(record), rules=OerRules.CANONICAL)


def decode_control_oer(data: bytes, *, canonical: bool = False):
    from .oer import OerRules, decode_oer

    rules = OerRules.CANONICAL if canonical else OerRules.BASIC
    return value_to_record(decode_oer(CONTROL_RECORD, data, rules=rules))


def encode_control_jer(record, *, canonical: bool = True) -> bytes:
    """JER text over the same type model (the readable, never the hot, rail)."""
    from .jer import JerRules, encode_jer

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    return encode_jer(CONTROL_RECORD, record_to_value(record), rules=rules)


def decode_control_jer(data: bytes, *, canonical: bool = True):
    """Recover a ControlRecord from its JER projection, through the bounded reader."""
    from .jer import JerRules
    from .jer_bounded import decode_bounded

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    return value_to_record(decode_bounded(data, CONTROL_RECORD, rules=rules))


__all__ = [
    "ACTIVATE",
    "BCIR_ARC",
    "BODY",
    "CANCEL",
    "CONTROL_PLANE_MODULE_OID",
    "CONTROL_RECORD",
    "GENERATION_SWITCH",
    "LEASE_GRANT",
    "MODULE",
    "PROJECTION_VERSION",
    "QUIESCE",
    "ROLLBACK",
    "SCOPE",
    "decode_control_der",
    "decode_control_jer",
    "decode_control_oer",
    "encode_control_der",
    "encode_control_jer",
    "encode_control_oer",
    "record_to_value",
    "value_to_record",
]
