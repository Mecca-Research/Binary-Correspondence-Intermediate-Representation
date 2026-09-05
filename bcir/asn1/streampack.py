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
#: moving, and vice versa. 2 since S0-E: the `generations` component (the per-resource
#: generation vector, native v4). The schema's `version [0] INTEGER DEFAULT 1` keeps its
#: original default, so a document without the field still means version 1 and a
#: version-2 document spells its version explicitly (X.680: a DEFAULT is not revised).
PROJECTION_VERSION = 2

_INTEGER = Primitive(Universal.INTEGER, "INTEGER")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")

#: X.680 §20: an ENUMERATED type IS its enumeration, and these two carry theirs.
#:
#: They did not, until J4 part 2. A bare `Primitive(Universal.ENUMERATED)` encodes fine
#: under DER and OER, which encode the enumeration *value* (X.690 §8.4, X.696 §11), so the
#: omission was invisible for as long as those were the only projections. It is not
#: invisible to the other two rails: X.691 §14 encodes the enumeration *index*, which needs
#: the root sorted, and X.697 §22.2 encodes the *identifier*, which cannot be derived from
#: the number at all. So the module was DER/OER-only by accident rather than by design, and
#: the JER projection is what surfaced it.
#:
#: The names and numbers are exactly the ones this file's own ASN.1 comment block above
#: already declares — the code simply had not carried them. DER octets are unchanged,
#: because DER never looked at the identifiers; `test_asn1_streampack.py` pins that.
LANE = Primitive(
    Universal.ENUMERATED,
    "Lane",
    enumeration=(("u", 0), ("ux", 1), ("t", 2), ("ggg", 3), ("a", 4), ("h", 5)),
)
DISPATCH = Primitive(Universal.ENUMERATED, "Dispatch", enumeration=(("core", 0), ("pim", 1)))

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
#       traceNotes     [9] SEQUENCE OF TraceNote DEFAULT {},
#       generations   [10] SEQUENCE OF Generation DEFAULT {} }
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
#   Generation ::= SEQUENCE {
#       rid            [0] INTEGER,
#       mapGen         [1] INTEGER DEFAULT 0,
#       dataGen        [2] INTEGER DEFAULT 0 }
#
# END

_INTEGERS = SequenceOf(_INTEGER, "SEQUENCE OF INTEGER")
_STRINGS = SequenceOf(_UTF8, "SEQUENCE OF UTF8String")

LANE_SEGMENT = Sequence(
    (
        Component("name", _UTF8, tag=0),
        Component("claimId", _INTEGER, tag=1),
        Component("phaseId", _INTEGER, tag=2),
        Component("lane", LANE, tag=3),
        Component("width", _INTEGER, tag=4),
        Component("opcode", _UTF8, tag=5),
        Component("reads", _INTEGERS, tag=6),
        Component("writes", _INTEGERS, tag=7),
        Component("prefetch", _UTF8, tag=8, optional=True),
        Component("fenceBefore", _STRINGS, tag=9, default=[]),
        Component("fenceAfter", _STRINGS, tag=10, default=[]),
        Component("dispatch", DISPATCH, tag=11, default=0),
        Component("channel", _UTF8, tag=12, default="host"),
    ),
    name="LaneSegment",
)

PREFETCH = Sequence(
    (
        Component("name", _UTF8, tag=0),
        Component("distance", _INTEGER, tag=1),
        Component("targets", _INTEGERS, tag=2),
        Component("hint", _UTF8, tag=3, default="T0"),
        Component("pattern", _UTF8, tag=4, default="linear"),
        Component("buffers", _INTEGER, tag=5, default=1),
    ),
    name="Prefetch",
)

BLOCK = Sequence(
    (
        Component("base", _INTEGER, tag=0),
        Component("count", _INTEGER, tag=1),
        Component("strides", _INTEGERS, tag=2, default=[1]),
    ),
    name="Block",
)

TRACE_NOTE = Sequence(
    (
        Component("claimId", _INTEGER, tag=0),
        Component("srcHash", _INTEGER, tag=1, default=0),
        Component("traceHash", _INTEGER, tag=2, default=0),
    ),
    name="TraceNote",
)

#: The per-resource generation vector (native StreamPack v4, law R11): one entry per
#: declared RID, in RID order, taken at hydration.
GENERATION = Sequence(
    (
        Component("rid", _INTEGER, tag=0),
        Component("mapGen", _INTEGER, tag=1, default=0),
        Component("dataGen", _INTEGER, tag=2, default=0),
    ),
    name="Generation",
)

STREAM_PACK = Sequence(
    (
        Component("version", _INTEGER, tag=0, default=1),
        Component("sourcePlan", _UTF8, tag=1),
        Component("topoGen", _INTEGER, tag=2, default=1),
        Component("mapGen", _INTEGER, tag=3, default=0),
        Component("dataGen", _INTEGER, tag=4, default=0),
        Component("pipelineDepth", _INTEGER, tag=5, default=1),
        Component("segments", SequenceOf(LANE_SEGMENT, "SEQUENCE OF LaneSegment"), tag=6),
        Component("prefetches", SequenceOf(PREFETCH, "SEQUENCE OF Prefetch"), tag=7, default=[]),
        Component("blocks", SequenceOf(BLOCK, "SEQUENCE OF Block"), tag=8, default=[]),
        Component("traceNotes", SequenceOf(TRACE_NOTE, "SEQUENCE OF TraceNote"), tag=9, default=[]),
        Component(
            "generations", SequenceOf(GENERATION, "SEQUENCE OF Generation"), tag=10, default=[]
        ),
    ),
    name="StreamPack",
)

MODULE = Module(
    "BCIR-StreamPack",
    STREAMPACK_MODULE_OID,
    {
        "StreamPack": STREAM_PACK,
        "LaneSegment": LANE_SEGMENT,
        "Prefetch": PREFETCH,
        "Block": BLOCK,
        "TraceNote": TRACE_NOTE,
        "Generation": GENERATION,
    },
)


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
        "segments": [
            {
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
            }
            for s in pack.segments
        ],
        "prefetches": [
            {
                "name": p.name,
                "distance": p.distance,
                "targets": list(p.targets),
                "hint": p.hint,
                "pattern": p.pattern,
                "buffers": p.buffers,
            }
            for p in pack.prefetches
        ],
        "blocks": [
            {
                "base": b.base,
                "count": b.count,
                "strides": list(b.strides),
            }
            for b in pack.blocks
        ],
        "traceNotes": [
            {
                "claimId": t.claim_id,
                "srcHash": t.src_hash,
                "traceHash": t.trace_hash,
            }
            for t in pack.trace_notes
        ],
        "generations": [
            {"rid": g.rid, "mapGen": g.map_gen, "dataGen": g.data_gen}
            for g in getattr(pack, "generations", ())
        ],
    }


def value_to_pack(value: dict):
    """The inverse of `pack_to_value` (imports the GEM types lazily — cold organ)."""
    from ..gem.streampack import Block, Generation, LaneSegment, Prefetch, StreamPack, TraceNote

    return StreamPack(
        source_plan=value["sourcePlan"],
        topo_gen=value.get("topoGen", 1),
        map_gen=value.get("mapGen", 0),
        data_gen=value.get("dataGen", 0),
        pipeline_depth=value.get("pipelineDepth", 1),
        segments=[
            LaneSegment(
                name=s["name"],
                claim_id=s["claimId"],
                phase_id=s["phaseId"],
                lane=Lane(s["lane"]),
                width=s["width"],
                opcode=s["opcode"],
                reads=tuple(s["reads"]),
                writes=tuple(s["writes"]),
                prefetch=s.get("prefetch") or None,
                fence_before=tuple(s.get("fenceBefore", [])),
                fence_after=tuple(s.get("fenceAfter", [])),
                dispatch=_dispatch_name(s.get("dispatch", 0)),
                channel=s.get("channel", "host"),
            )
            for s in value["segments"]
        ],
        prefetches=[
            Prefetch(
                name=p["name"],
                distance=p["distance"],
                targets=tuple(p["targets"]),
                hint=p.get("hint", "T0"),
                pattern=p.get("pattern", "linear"),
                buffers=p.get("buffers", 1),
            )
            for p in value.get("prefetches", [])
        ],
        blocks=[
            Block(
                base=b["base"],
                count=b["count"],
                strides=tuple(b.get("strides", (1,))),
            )
            for b in value.get("blocks", [])
        ],
        trace_notes=[
            TraceNote(
                claim_id=t["claimId"],
                src_hash=t.get("srcHash", 0),
                trace_hash=t.get("traceHash", 0),
            )
            for t in value.get("traceNotes", [])
        ],
        generations=[
            Generation(rid=g["rid"], map_gen=g.get("mapGen", 0), data_gen=g.get("dataGen", 0))
            for g in value.get("generations", [])
        ],
    )


def _dispatch_code(name: str) -> int:
    try:
        return DISPATCH_VALUES[name]
    except KeyError:
        raise Asn1Error(
            f"dispatch {name!r} is outside the Dispatch enumeration {sorted(DISPATCH_VALUES)}"
        ) from None


def _dispatch_name(code: int) -> str:
    try:
        return DISPATCH_NAMES[code]
    except KeyError:
        raise Asn1Error(
            f"Dispatch value {code} is not enumerated (X.680 20.4: a decoder shall "
            f"reject an unlisted enumeration value)"
        ) from None


def encode_pack(pack) -> bytes:
    """DER octets for a StreamPack under the BCIR-StreamPack module."""
    return MODULE.encode("StreamPack", pack_to_value(pack))


def decode_pack(data: bytes, *, strictness: Strictness = Strictness.DER):
    """Recover a StreamPack from its DER projection (BER admitted on request)."""
    return value_to_pack(MODULE.decode("StreamPack", data, strictness=strictness))


def encode_pack_oer(pack) -> bytes:
    """CANONICAL-OER octets for a StreamPack under the same BCIR-StreamPack module.

    The type model is the SAME object the DER projection uses -- nothing about the module
    changes to gain a second transfer syntax. That is the point roadmap section 0 turns on:
    an abstract value has several legal realizations, the encoding rules are a *realization
    choice*, and the schema is not where that choice lives. Phase H prices these
    candidates; this is one of them.
    """
    from .oer import OerRules, encode_oer

    return encode_oer(STREAM_PACK, pack_to_value(pack), rules=OerRules.CANONICAL)


def decode_pack_oer(data: bytes, *, canonical: bool = False):
    """Recover a StreamPack from its OER projection (BASIC-OER admitted by default)."""
    from .oer import OerRules, decode_oer

    rules = OerRules.CANONICAL if canonical else OerRules.BASIC
    return value_to_pack(decode_oer(STREAM_PACK, data, rules=rules))


def encode_pack_jer(pack, *, canonical: bool = True) -> bytes:
    """JER text for a StreamPack under the SAME BCIR-StreamPack module.

    The third transfer syntax over one type model, and it needed no change to the module —
    which is the property roadmap §0 turns on and the reason a schema is worth having.

    §6.3 draws the line this function sits on: **JER is never a replacement for native
    StreamPack.** A projection is additive and must reconstruct byte-identical native
    artifacts, which is a testable claim rather than a promise — `test_asn1_dialect.py`
    round-trips real packs through JER and compares the *native* octets, not the JSON.

    Nor is this a hot path. JER is roughly four times the size of the binary projections
    and is parsed as text; §1's boundary keeps it in the build, control, configuration and
    load planes. What it buys is a schema-bound rail a human can read and a foreign tool
    can produce without a BER toolkit.
    """
    from .jer import JerRules, encode_jer

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    return encode_jer(STREAM_PACK, pack_to_value(pack), rules=rules)


def decode_pack_jer(data: bytes, *, canonical: bool = True):
    """Recover a StreamPack from its JER projection.

    Defaults to the canonical profile because a pack that arrives over JER is a pack the
    caller is about to digest or execute, and BASIC would let a sender choose the spelling
    — and so the digest. Pass `canonical=False` only where the input is known-foreign and
    the result is not being bound to a content address.
    """
    from .jer import JerRules
    from .jer_bounded import decode_bounded

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    # Through the BOUNDED reader, never bare `decode_jer`: this is a trust boundary, and
    # §4.3's limits have to apply before a value graph exists. `encode_pack_jer`'s output
    # is the only input that is not attacker-chosen, and it costs nothing to bound that too.
    return value_to_pack(decode_bounded(data, STREAM_PACK, rules=rules))


__all__ = [
    "BCIR_ARC",
    "BLOCK",
    "DISPATCH_NAMES",
    "DISPATCH_VALUES",
    "GENERATION",
    "LANE_SEGMENT",
    "MODULE",
    "decode_pack_jer",
    "encode_pack_jer",
    "PREFETCH",
    "PROJECTION_VERSION",
    "STREAM_PACK",
    "STREAMPACK_MODULE_OID",
    "TRACE_NOTE",
    "decode_pack",
    "decode_pack_oer",
    "encode_pack",
    "encode_pack_oer",
    "pack_to_value",
    "value_to_pack",
]
