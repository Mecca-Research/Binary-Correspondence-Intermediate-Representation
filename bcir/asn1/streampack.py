"""BCIR-StreamPack — the ASN.1 module and DER projection of a StreamPack.

This is the *ABI compatibility* half of the ASN.1 work. The native StreamPack wire
format (`bcir/abi/streampack_abi.py`, `docs/kernel/BCIR_STREAMPACK_ABI.md`) is frozen
at v1 and evolves append-only; nothing here touches a byte of it. What this module
adds is a **second transfer syntax for the same abstract value** — an ASN.1 type
definition plus a DER encoding of it — so a BCIR artifact can cross a boundary that
speaks ASN.1 (PKI tooling, telecom stacks, anything driven by an X.680 module) and
come back byte-identical.

The two syntaxes are related by a round-trip law, gated in the test suite:

    decode_pack(encode_pack(p)) == p                    (the projection is faithful)
    abi.encode(decode_pack(encode_pack(abi.decode(b)))) == b   (native bytes survive)

Design notes worth stating, because they are the non-obvious choices:

* **The projection is of the abstract StreamPack, not of its octets.** Wrapping the
  native bytes in an OCTET STRING would be trivial and useless — a peer could not read
  a field without implementing BCIR's format. The ASN.1 module names every field, so
  an ASN.1 peer can.
* **Enumerations are ENUMERATED, not INTEGER.** `lane` and `dispatch` are closed sets
  in BCIR; X.680 §20 says exactly that, and it lets a peer's schema reject an unknown
  lane rather than silently accept 250.
* **Defaults mirror the native format's implicit ones.** The native encoder emits v1
  bytes when no v2/v3 feature is used; the same values are DEFAULTs here, so §11.5
  omits them and the DER stays as small as the native form for the common case.
* **The `stride_k` field the native segment record reserves is not projected.** It is
  written as a constant zero and carried per-claim, so projecting it would put a field
  in the schema that never varies.
"""

from __future__ import annotations

from ..model import Lane
from .codec import Strictness
from .schema import Component, Module, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, Universal

#: {iso(1) identified-organization(3) dod(6) internet(1) private(4) enterprise(1)}
#: with a BCIR-local arc. Private-enterprise space is the correct home for a
#: project-defined module: it needs no registration and can never collide with an
#: allocation made by a registration authority.
BCIR_ARC: tuple[int, ...] = (1, 3, 6, 1, 4, 1, 62596)
STREAMPACK_MODULE_OID: tuple[int, ...] = (*BCIR_ARC, 1)

#: Bumped only when the ASN.1 module changes shape. Independent of the native
#: StreamPack version: the projection can gain a field without the native format
#: moving, and vice versa.
PROJECTION_VERSION = 1

_INTEGER = Primitive(Universal.INTEGER, "INTEGER")
_ENUMERATED = Primitive(Universal.ENUMERATED, "ENUMERATED")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")

#: `dispatch` on the wire (native `_DISPATCH_WIRE`), mirrored so the two rails agree
#: on the integer meaning of each name rather than each inventing one.
DISPATCH_VALUES = {"core": 0, "pim": 1}
DISPATCH_NAMES = {v: k for k, v in DISPATCH_VALUES.items()}

# --- the module ---------------------------------------------------------------------
#
# BCIR-StreamPack { iso(1) identified-organization(3) dod(6) internet(1) private(4)
#                   enterprise(1) 62596 1 }
# DEFINITIONS IMPLICIT TAGS ::= BEGIN
#
#   StreamPack ::= SEQUENCE {
#       version        [0] INTEGER DEFAULT 1,
#       sourcePlan     [1] UTF8String,
#       topoGen        [2] INTEGER DEFAULT 1,
#       mapGen         [3] INTEGER DEFAULT 0,
#       dataGen        [4] INTEGER DEFAULT 0,
#       pipelineDepth  [5] INTEGER DEFAULT 1,
#       segments       [6] SEQUENCE OF LaneSegment,
#       prefetches     [7] SEQUENCE OF Prefetch  DEFAULT {},
#       blocks         [8] SEQUENCE OF Block     DEFAULT {},
#       traceNotes     [9] SEQUENCE OF TraceNote DEFAULT {} }
#
#   LaneSegment ::= SEQUENCE {
#       name           [0] UTF8String,
#       claimId        [1] INTEGER,
#       phaseId        [2] INTEGER,
#       lane           [3] Lane,
#       width          [4] INTEGER,
#       opcode         [5] UTF8String,
#       reads          [6] SEQUENCE OF INTEGER,
#       writes         [7] SEQUENCE OF INTEGER,
#       prefetch       [8] UTF8String OPTIONAL,
#       fenceBefore    [9] SEQUENCE OF UTF8String DEFAULT {},
#       fenceAfter    [10] SEQUENCE OF UTF8String DEFAULT {},
#       dispatch      [11] Dispatch  DEFAULT core,
#       channel       [12] UTF8String DEFAULT "host" }
#
#   Lane     ::= ENUMERATED { u(0), ux(1), t(2), ggg(3), a(4), h(5) }
#   Dispatch ::= ENUMERATED { core(0), pim(1) }
#
#   Prefetch ::= SEQUENCE {
#       name           [0] UTF8String,
#       distance       [1] INTEGER,
#       targets        [2] SEQUENCE OF INTEGER,
#       hint           [3] UTF8String DEFAULT "T0",
#       pattern        [4] UTF8String DEFAULT "linear",
#       buffers        [5] INTEGER    DEFAULT 1 }
#
#   Block ::= SEQUENCE {
#       base           [0] INTEGER,
#       count          [1] INTEGER,
#       strides        [2] SEQUENCE OF INTEGER DEFAULT { 1 } }
#
#   TraceNote ::= SEQUENCE {
#       claimId        [0] INTEGER,
#       srcHash        [1] INTEGER DEFAULT 0,
#       traceHash      [2] INTEGER DEFAULT 0 }
#
# END

_INTEGERS = SequenceOf(_INTEGER, "SEQUENCE OF INTEGER")
_STRINGS = SequenceOf(_UTF8, "SEQUENCE OF UTF8String")

LANE_SEGMENT = Sequence((
    Component("name", _UTF8, tag=0),
    Component("claimId", _INTEGER, tag=1),
    Component("phaseId", _INTEGER, tag=2),
    Component("lane", _ENUMERATED, tag=3),
    Component("width", _INTEGER, tag=4),
    Component("opcode", _UTF8, tag=5),
    Component("reads", _INTEGERS, tag=6),
    Component("writes", _INTEGERS, tag=7),
    Component("prefetch", _UTF8, tag=8, optional=True),
    Component("fenceBefore", _STRINGS, tag=9, default=[]),
    Component("fenceAfter", _STRINGS, tag=10, default=[]),
    Component("dispatch", _ENUMERATED, tag=11, default=0),
    Component("channel", _UTF8, tag=12, default="host"),
), name="LaneSegment")

PREFETCH = Sequence((
    Component("name", _UTF8, tag=0),
    Component("distance", _INTEGER, tag=1),
    Component("targets", _INTEGERS, tag=2),
    Component("hint", _UTF8, tag=3, default="T0"),
    Component("pattern", _UTF8, tag=4, default="linear"),
    Component("buffers", _INTEGER, tag=5, default=1),
), name="Prefetch")

BLOCK = Sequence((
    Component("base", _INTEGER, tag=0),
    Component("count", _INTEGER, tag=1),
    Component("strides", _INTEGERS, tag=2, default=[1]),
), name="Block")

TRACE_NOTE = Sequence((
    Component("claimId", _INTEGER, tag=0),
    Component("srcHash", _INTEGER, tag=1, default=0),
    Component("traceHash", _INTEGER, tag=2, default=0),
), name="TraceNote")

STREAM_PACK = Sequence((
    Component("version", _INTEGER, tag=0, default=PROJECTION_VERSION),
    Component("sourcePlan", _UTF8, tag=1),
    Component("topoGen", _INTEGER, tag=2, default=1),
    Component("mapGen", _INTEGER, tag=3, default=0),
    Component("dataGen", _INTEGER, tag=4, default=0),
    Component("pipelineDepth", _INTEGER, tag=5, default=1),
    Component("segments", SequenceOf(LANE_SEGMENT, "SEQUENCE OF LaneSegment"), tag=6),
    Component("prefetches", SequenceOf(PREFETCH, "SEQUENCE OF Prefetch"), tag=7,
              default=[]),
    Component("blocks", SequenceOf(BLOCK, "SEQUENCE OF Block"), tag=8, default=[]),
    Component("traceNotes", SequenceOf(TRACE_NOTE, "SEQUENCE OF TraceNote"), tag=9,
              default=[]),
), name="StreamPack")

MODULE = Module("BCIR-StreamPack", STREAMPACK_MODULE_OID, {
    "StreamPack": STREAM_PACK,
    "LaneSegment": LANE_SEGMENT,
    "Prefetch": PREFETCH,
    "Block": BLOCK,
    "TraceNote": TRACE_NOTE,
})


# --- projection: StreamPack <-> the ASN.1 value ------------------------------------


def pack_to_value(pack) -> dict:
    """The ASN.1 abstract value for a `gem.streampack.StreamPack`."""
    return {
        "version": PROJECTION_VERSION,
        "sourcePlan": pack.source_plan,
        "topoGen": pack.topo_gen,
        "mapGen": pack.map_gen,
        "dataGen": pack.data_gen,
        "pipelineDepth": pack.pipeline_depth,
        "segments": [{
            "name": s.name,
            "claimId": s.claim_id,
            "phaseId": s.phase_id,
            "lane": int(s.lane),
            "width": s.width,
            "opcode": s.opcode,
            "reads": list(s.reads),
            "writes": list(s.writes),
            **({"prefetch": s.prefetch} if s.prefetch else {}),
            "fenceBefore": list(s.fence_before),
            "fenceAfter": list(s.fence_after),
            "dispatch": _dispatch_code(s.dispatch),
            "channel": s.channel,
        } for s in pack.segments],
        "prefetches": [{
            "name": p.name, "distance": p.distance, "targets": list(p.targets),
            "hint": p.hint, "pattern": p.pattern, "buffers": p.buffers,
        } for p in pack.prefetches],
        "blocks": [{
            "base": b.base, "count": b.count, "strides": list(b.strides),
        } for b in pack.blocks],
        "traceNotes": [{
            "claimId": t.claim_id, "srcHash": t.src_hash, "traceHash": t.trace_hash,
        } for t in pack.trace_notes],
    }


def value_to_pack(value: dict):
    """The inverse of `pack_to_value` (imports the GEM types lazily — cold organ)."""
    from ..gem.streampack import Block, LaneSegment, Prefetch, StreamPack, TraceNote

    return StreamPack(
        source_plan=value["sourcePlan"],
        topo_gen=value.get("topoGen", 1),
        map_gen=value.get("mapGen", 0),
        data_gen=value.get("dataGen", 0),
        pipeline_depth=value.get("pipelineDepth", 1),
        segments=[LaneSegment(
            name=s["name"], claim_id=s["claimId"], phase_id=s["phaseId"],
            lane=Lane(s["lane"]), width=s["width"], opcode=s["opcode"],
            reads=tuple(s["reads"]), writes=tuple(s["writes"]),
            prefetch=s.get("prefetch") or None,
            fence_before=tuple(s.get("fenceBefore", [])),
            fence_after=tuple(s.get("fenceAfter", [])),
            dispatch=_dispatch_name(s.get("dispatch", 0)),
            channel=s.get("channel", "host"),
        ) for s in value["segments"]],
        prefetches=[Prefetch(
            name=p["name"], distance=p["distance"], targets=tuple(p["targets"]),
            hint=p.get("hint", "T0"), pattern=p.get("pattern", "linear"),
            buffers=p.get("buffers", 1),
        ) for p in value.get("prefetches", [])],
        blocks=[Block(
            base=b["base"], count=b["count"], strides=tuple(b.get("strides", (1,))),
        ) for b in value.get("blocks", [])],
        trace_notes=[TraceNote(
            claim_id=t["claimId"], src_hash=t.get("srcHash", 0),
            trace_hash=t.get("traceHash", 0),
        ) for t in value.get("traceNotes", [])],
    )


def _dispatch_code(name: str) -> int:
    try:
        return DISPATCH_VALUES[name]
    except KeyError:
        raise Asn1Error(
            f"dispatch {name!r} is outside the Dispatch enumeration "
            f"{sorted(DISPATCH_VALUES)}") from None


def _dispatch_name(code: int) -> str:
    try:
        return DISPATCH_NAMES[code]
    except KeyError:
        raise Asn1Error(
            f"Dispatch value {code} is not enumerated (X.680 20.4: a decoder shall "
            f"reject an unlisted enumeration value)") from None


def encode_pack(pack) -> bytes:
    """DER octets for a StreamPack under the BCIR-StreamPack module."""
    return MODULE.encode("StreamPack", pack_to_value(pack))


def decode_pack(data: bytes, *, strictness: Strictness = Strictness.DER):
    """Recover a StreamPack from its DER projection (BER admitted on request)."""
    return value_to_pack(MODULE.decode("StreamPack", data, strictness=strictness))


__all__ = [
    "BCIR_ARC", "BLOCK", "DISPATCH_NAMES", "DISPATCH_VALUES", "LANE_SEGMENT", "MODULE",
    "PREFETCH", "PROJECTION_VERSION", "STREAM_PACK", "STREAMPACK_MODULE_OID",
    "TRACE_NOTE", "decode_pack", "encode_pack", "pack_to_value", "value_to_pack",
]
