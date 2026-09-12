"""BCIR-ExecutionPlan — the ASN.1 module and DER/OER/JER projection of an ExecutionPlanV1.

The plan as bytes (G11, staged plan S1-C) has a frozen native wire format
(`bcir/abi/execution_plan_abi.py`, `docs/kernel/BCIR_EXECUTION_PLAN_ABI.md`); this module
follows the StreamPack precedent (`bcir/asn1/streampack.py`) and adds a **second transfer
syntax for the same abstract value** — an X.680 module and its DER encoding, plus the OER
and JER realizations over the same type model — so a plan can cross a boundary that speaks
ASN.1 and come back byte-identical:

    decode_plan_der(encode_plan_der(p)) == p                          (faithful)
    encode_plan(decode_plan_der(encode_plan_der(decode_plan(b)))) == b   (native bytes survive)

The non-obvious choices, stated:

* **Signed integers stay INTEGER.** A step's `cost` is signed and the tail stream is the
  scheduler's -1; X.680 INTEGER carries both, so the projection spells them as the model
  does (the native wire spells the tail as 0xFFFFFFFF and the cost as two's complement).
* **Closed sets are ENUMERATED**: `mode`, `lane`, a movement edge's `kind` and `coherence`.
* **Defaults mirror the native format's implicit ones**, so §11.5 omits them under DER and
  the common case stays as small as the native form.
* **`route` is OPTIONAL**, absent for the empty string, as a segment's `prefetch` is.
"""

from __future__ import annotations

from ..model import Lane
from .codec import Strictness
from .schema import Component, Module, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, Universal

BCIR_ARC: tuple[int, ...] = (1, 3, 6, 1, 4, 1, 62596)
EXECUTION_PLAN_MODULE_OID: tuple[int, ...] = (*BCIR_ARC, 3)

#: Bumped only when the ASN.1 module changes shape; independent of the native version.
PROJECTION_VERSION = 1

_INTEGER = Primitive(Universal.INTEGER, "INTEGER")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")

MODE = Primitive(Universal.ENUMERATED, "Mode", enumeration=(("eft", 0), ("tokens", 1)))
LANE = Primitive(
    Universal.ENUMERATED,
    "Lane",
    enumeration=(("u", 0), ("ux", 1), ("t", 2), ("ggg", 3), ("a", 4), ("h", 5)),
)
MOVE_KIND = Primitive(
    Universal.ENUMERATED,
    "MoveKind",
    enumeration=(
        ("direct", 0),
        ("peer", 1),
        ("staged", 2),
        ("rematerialized", 3),
        ("compressed", 4),
        ("evicted", 5),
    ),
)
COHERENCE = Primitive(
    Universal.ENUMERATED,
    "Coherence",
    enumeration=(("none", 0), ("flush", 1), ("invalidate", 2), ("writeback", 3)),
)

#: The wire codes, mirrored from the native format so the two rails agree on each name.
MODE_VALUES = {"eft": 0, "tokens": 1}
MODE_NAMES = {v: k for k, v in MODE_VALUES.items()}
MOVE_KIND_VALUES = {name: code for name, code in MOVE_KIND.enumeration}
MOVE_KIND_NAMES = {v: k for k, v in MOVE_KIND_VALUES.items()}
COHERENCE_VALUES = {name: code for name, code in COHERENCE.enumeration}
COHERENCE_NAMES = {v: k for k, v in COHERENCE_VALUES.items()}

PLAN_STEP = Sequence(
    (
        Component("claimId", _INTEGER, tag=0),
        Component("phaseId", _INTEGER, tag=1),
        Component("candidate", _UTF8, tag=2),
        Component("lane", LANE, tag=3),
        Component("width", _INTEGER, tag=4),
        Component("cost", _INTEGER, tag=5, default=0),
        Component("stream", _INTEGER, tag=6, default=0),
        Component("start", _INTEGER, tag=7, default=0),
        Component("duration", _INTEGER, tag=8, default=0),
    ),
    name="PlanStep",
)

LIFETIME = Sequence(
    (
        Component("rid", _INTEGER, tag=0),
        Component("bank", _UTF8, tag=1),
        Component("offset", _INTEGER, tag=2, default=0),
        Component("sizeBytes", _INTEGER, tag=3),
        Component("alignment", _INTEGER, tag=4, default=1),
        Component("firstPhase", _INTEGER, tag=5, default=0),
        Component("lastPhase", _INTEGER, tag=6, default=0),
    ),
    name="Lifetime",
)

MOVEMENT_EDGE = Sequence(
    (
        Component("rid", _INTEGER, tag=0),
        Component("srcBank", _UTF8, tag=1),
        Component("dstBank", _UTF8, tag=2),
        Component("offset", _INTEGER, tag=3, default=0),
        Component("sizeBytes", _INTEGER, tag=4),
        Component("route", _UTF8, tag=5, optional=True),
        Component("kind", MOVE_KIND, tag=6, default=0),
        Component("coherence", COHERENCE, tag=7, default=0),
        Component("mapGen", _INTEGER, tag=8, default=0),
        Component("dataGen", _INTEGER, tag=9, default=0),
        Component("afterClaim", _INTEGER, tag=10, default=0),
        Component("beforeClaim", _INTEGER, tag=11, default=0),
    ),
    name="MovementEdge",
)

GENERATION = Sequence(
    (
        Component("rid", _INTEGER, tag=0),
        Component("mapGen", _INTEGER, tag=1, default=0),
        Component("dataGen", _INTEGER, tag=2, default=0),
    ),
    name="Generation",
)

EXECUTION_PLAN = Sequence(
    (
        Component("version", _INTEGER, tag=0, default=1),
        Component("sourcePlan", _UTF8, tag=1),
        Component("mode", MODE, tag=2, default=0),
        Component("streams", _INTEGER, tag=3, default=1),
        Component("knee", _INTEGER, tag=4, default=1),
        Component("makespan", _INTEGER, tag=5, default=0),
        Component("moduleHash", _INTEGER, tag=6, default=0),
        Component("targetHash", _INTEGER, tag=7, default=0),
        Component("steps", SequenceOf(PLAN_STEP, "SEQUENCE OF PlanStep"), tag=8),
        Component("lifetimes", SequenceOf(LIFETIME, "SEQUENCE OF Lifetime"), tag=9, default=[]),
        Component(
            "moves", SequenceOf(MOVEMENT_EDGE, "SEQUENCE OF MovementEdge"), tag=10, default=[]
        ),
        Component(
            "generations", SequenceOf(GENERATION, "SEQUENCE OF Generation"), tag=11, default=[]
        ),
    ),
    name="ExecutionPlan",
)

MODULE = Module(
    "BCIR-ExecutionPlan",
    EXECUTION_PLAN_MODULE_OID,
    {
        "ExecutionPlan": EXECUTION_PLAN,
        "PlanStep": PLAN_STEP,
        "Lifetime": LIFETIME,
        "MovementEdge": MOVEMENT_EDGE,
        "Generation": GENERATION,
    },
)


# --- projection: ExecutionPlan <-> the ASN.1 value ----------------------------------


def _code(table: dict, name: str, what: str) -> int:
    try:
        return table[name]
    except KeyError:
        raise Asn1Error(f"{what} {name!r} is outside the enumeration {sorted(table)}") from None


def _name(table: dict, code: int, what: str) -> str:
    try:
        return table[code]
    except KeyError:
        raise Asn1Error(
            f"{what} value {code} is not enumerated (X.680 20.4: a decoder shall "
            f"reject an unlisted enumeration value)"
        ) from None


def plan_to_value(plan) -> dict:
    """The ASN.1 abstract value for a `gem.execution_plan.ExecutionPlan`."""
    return {
        "version": PROJECTION_VERSION,
        "sourcePlan": plan.source_plan,
        "mode": _code(MODE_VALUES, plan.mode, "mode"),
        "streams": plan.streams,
        "knee": plan.knee,
        "makespan": plan.makespan,
        "moduleHash": plan.module_hash,
        "targetHash": plan.target_hash,
        "steps": [
            {
                "claimId": s.claim_id,
                "phaseId": s.phase_id,
                "candidate": s.candidate,
                "lane": int(s.lane),
                "width": s.width,
                "cost": s.cost,
                "stream": s.stream,
                "start": s.start,
                "duration": s.duration,
            }
            for s in plan.steps
        ],
        "lifetimes": [
            {
                "rid": lt.rid,
                "bank": lt.bank,
                "offset": lt.offset,
                "sizeBytes": lt.size_bytes,
                "alignment": lt.alignment,
                "firstPhase": lt.first_phase,
                "lastPhase": lt.last_phase,
            }
            for lt in plan.lifetimes
        ],
        "moves": [
            {
                "rid": mv.rid,
                "srcBank": mv.src_bank,
                "dstBank": mv.dst_bank,
                "offset": mv.offset,
                "sizeBytes": mv.size_bytes,
                **({"route": mv.route} if mv.route else {}),
                "kind": _code(MOVE_KIND_VALUES, mv.kind, "kind"),
                "coherence": _code(COHERENCE_VALUES, mv.coherence, "coherence"),
                "mapGen": mv.map_gen,
                "dataGen": mv.data_gen,
                "afterClaim": mv.after_claim,
                "beforeClaim": mv.before_claim,
            }
            for mv in plan.moves
        ],
        "generations": [
            {"rid": g.rid, "mapGen": g.map_gen, "dataGen": g.data_gen} for g in plan.generations
        ],
    }


def value_to_plan(value: dict):
    """The inverse of `plan_to_value` (imports the GEM types lazily — cold organ)."""
    from ..gem.execution_plan import ExecutionPlan, Lifetime, MovementEdge, PlanStep
    from ..gem.streampack import Generation

    return ExecutionPlan(
        source_plan=value["sourcePlan"],
        mode=_name(MODE_NAMES, value.get("mode", 0), "Mode"),
        streams=value.get("streams", 1),
        knee=value.get("knee", 1),
        makespan=value.get("makespan", 0),
        module_hash=value.get("moduleHash", 0),
        target_hash=value.get("targetHash", 0),
        steps=[
            PlanStep(
                claim_id=s["claimId"],
                phase_id=s["phaseId"],
                candidate=s["candidate"],
                lane=Lane(s["lane"]),
                width=s["width"],
                cost=s.get("cost", 0),
                stream=s.get("stream", 0),
                start=s.get("start", 0),
                duration=s.get("duration", 0),
            )
            for s in value["steps"]
        ],
        lifetimes=[
            Lifetime(
                rid=lt["rid"],
                bank=lt["bank"],
                offset=lt.get("offset", 0),
                size_bytes=lt["sizeBytes"],
                alignment=lt.get("alignment", 1),
                first_phase=lt.get("firstPhase", 0),
                last_phase=lt.get("lastPhase", 0),
            )
            for lt in value.get("lifetimes", [])
        ],
        moves=[
            MovementEdge(
                rid=mv["rid"],
                src_bank=mv["srcBank"],
                dst_bank=mv["dstBank"],
                offset=mv.get("offset", 0),
                size_bytes=mv["sizeBytes"],
                route=mv.get("route") or "",
                kind=_name(MOVE_KIND_NAMES, mv.get("kind", 0), "MoveKind"),
                coherence=_name(COHERENCE_NAMES, mv.get("coherence", 0), "Coherence"),
                map_gen=mv.get("mapGen", 0),
                data_gen=mv.get("dataGen", 0),
                after_claim=mv.get("afterClaim", 0),
                before_claim=mv.get("beforeClaim", 0),
            )
            for mv in value.get("moves", [])
        ],
        generations=[
            Generation(rid=g["rid"], map_gen=g.get("mapGen", 0), data_gen=g.get("dataGen", 0))
            for g in value.get("generations", [])
        ],
    )


def encode_plan_der(plan) -> bytes:
    """DER octets for an ExecutionPlan under the BCIR-ExecutionPlan module."""
    return MODULE.encode("ExecutionPlan", plan_to_value(plan))


def decode_plan_der(data: bytes, *, strictness: Strictness = Strictness.DER):
    """Recover an ExecutionPlan from its DER projection (BER admitted on request)."""
    return value_to_plan(MODULE.decode("ExecutionPlan", data, strictness=strictness))


def encode_plan_oer(plan) -> bytes:
    """CANONICAL-OER octets over the same type model."""
    from .oer import OerRules, encode_oer

    return encode_oer(EXECUTION_PLAN, plan_to_value(plan), rules=OerRules.CANONICAL)


def decode_plan_oer(data: bytes, *, canonical: bool = False):
    from .oer import OerRules, decode_oer

    rules = OerRules.CANONICAL if canonical else OerRules.BASIC
    return value_to_plan(decode_oer(EXECUTION_PLAN, data, rules=rules))


def encode_plan_jer(plan, *, canonical: bool = True) -> bytes:
    """JER text over the same type model (the readable, never the hot, rail)."""
    from .jer import JerRules, encode_jer

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    return encode_jer(EXECUTION_PLAN, plan_to_value(plan), rules=rules)


def decode_plan_jer(data: bytes, *, canonical: bool = True):
    """Recover an ExecutionPlan from its JER projection, through the bounded reader."""
    from .jer import JerRules
    from .jer_bounded import decode_bounded

    rules = JerRules.CANONICAL if canonical else JerRules.BASIC
    return value_to_plan(decode_bounded(data, EXECUTION_PLAN, rules=rules))


__all__ = [
    "BCIR_ARC",
    "COHERENCE",
    "COHERENCE_NAMES",
    "COHERENCE_VALUES",
    "EXECUTION_PLAN",
    "EXECUTION_PLAN_MODULE_OID",
    "GENERATION",
    "LANE",
    "LIFETIME",
    "MODE",
    "MODE_NAMES",
    "MODE_VALUES",
    "MODULE",
    "MOVE_KIND",
    "MOVE_KIND_NAMES",
    "MOVE_KIND_VALUES",
    "MOVEMENT_EDGE",
    "PLAN_STEP",
    "PROJECTION_VERSION",
    "decode_plan_der",
    "decode_plan_jer",
    "decode_plan_oer",
    "encode_plan_der",
    "encode_plan_jer",
    "encode_plan_oer",
    "plan_to_value",
    "value_to_plan",
]
