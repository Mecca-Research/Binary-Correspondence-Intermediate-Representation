"""The G16 corpora (S3-C), shared by `test_handoff.py`, the C gate (`tools/c/check_handoff.py`) and
the baseline harness (`tools/perf/gemplus_baseline.py --group handoff`).

Every fixture declares the outcome the specification requires, so each rail is graded against
the specification (docs/languages/CPP_HANDOFF_BOUNDARY.md, docs/kernel/BCIR_SHARD_MANIFEST_ABI.md)
and not only against the other rails:

* scenarios -- sequences of pack-table and plane operations with the (verdict, refusal) every graded
  operation must produce, run on three rails: the Python oracle (`bcir.gem.handoff.PackTable`),
  the freestanding C twin (`runtime/c/test_handoff.c`) and the C++ RAII seam
  (`runtime/cpp/test_handoff.cpp`), whose traces -- outcome, handle, resident generation and the
  table's and the plane's state digests after every operation -- must be identical. Families:
  `lifetime` (a view that outlives its owner), `admission` (a pack older than the registry
  generation, at admit and at dispatch), `freeze` (the dynamic-graph builder's per-step freeze,
  transactional) and `manifest` (a shard manifest gated at the plane);
* the freeze corpus -- step graphs whose `bcir_hydrate_generations` bytes (or refusal status)
  must equal `freeze_claims`';
* the shard corpora -- whole packs cut by partitions (byte-identical manifest, frame and shards
  on every rail, reassembling to the whole), packs outside the hydrated layout (refused), one
  malformed manifest per wire law, and tampered shard sets (refused).

`measure()` turns them into the G16 rows.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass, field, replace
from functools import partial

from ..abi import decode, encode
from ..abi.shard_manifest import (
    MANIFEST_FIXED,
    ShardError,
    decode_manifest,
    frame_of,
    partition,
    reassemble,
    split,
    sub_pack,
)
from ..gem.control import (
    CONTROL_SCOPES,
    Activate,
    ControlOutcome,
    ControlPlane,
    Quiesce,
    registry_digest,
)
from ..gem.handoff import (
    EPOCH_MAX,
    HO_REFUSALS,
    HO_VERDICTS,
    PINS_MAX,
    FreezeError,
    GraphClaim,
    GraphGeneration,
    Handle,
    PackTable,
    View,
    admit_manifest,
    freeze_claims,
)
from ..gem.streampack import (
    Block,
    Generation,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    generation_vector,
    hydrate,
    hydrate_pipelined,
)
from ..model.lanes import Lane
from . import control_fixtures as cf

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
C_DIR = os.path.join(REPO, "runtime", "c")
CPP_DIR = os.path.join(REPO, "runtime", "cpp")
# The freestanding units the harnesses link (and the C gate compiles): one list, both builds.
C_UNITS = (
    "bcir_handoff.c",
    "bcir_shard_manifest.c",
    "bcir_hydrate.c",
    "bcir_plan.c",
    "bcir_control_plane.c",
    "bcir_sha256.c",
    "bcir_runtime.c",
    # the Stage 3 exit flow (test_stage3.h) carries its records over the live rings
    "bcir_ring.c",
    "bcir_telemetry_envelope.c",
)
CPP_UNITS = ("bcir_handoff.cpp", "bcir_orchestrator.cpp")

ROWS = (
    "handoff.stale.admitted",
    "handoff.stale.dispatched",
    "handoff.fresh.refused",
    "handoff.lifetime.unrefused",
    "handoff.copies",
    "handoff.builder.violations",
    "handoff.decisions.nonconforming",
    "handoff.shards.mismatches",
    "handoff.shards.malformed.accepted",
    "handoff.reentry.divergent",
    "handoff.traces.divergent",
    "handoff.stage3.stale.accepted",
    "handoff.stage3.flow.divergent",
)

# --- the scenario script ------------------------------------------------------------------------

OP_SUBMIT, OP_ENTER, OP_LEAVE, OP_ADVANCE = 1, 2, 3, 4
OP_RESERVE, OP_WRITE, OP_COMMIT, OP_ABORT = 5, 6, 7, 8
OP_BORROW, OP_GIVE_BACK, OP_RELEASE, OP_ADMIT, OP_DISPATCH = 9, 10, 11, 12, 13
OP_ADMIT_MANIFEST, OP_FREEZE = 14, 15
# C-only: a forged handle, a poked epoch or pin count (to reach the bounds 2^32 operations away),
# a copied borrow. The C++ seam cannot spell them -- its handles come from owners, its borrows
# are move-only -- so those scenarios run on the Python and C rails.
OP_FORGE, OP_POKE_EPOCH, OP_POKE_PINS, OP_COPY_VIEW = 16, 17, 18, 19
C_ONLY_OPS = {OP_FORGE, OP_POKE_EPOCH, OP_POKE_PINS, OP_COPY_VIEW}
# The plane restarts: re-initialized with the scenario's key, back at generation 0 with no
# registry -- so the SAME generation number can later name ANOTHER registry. Only the admission's
# registry binding tells those apart; every rail spells it.
OP_RESTART = 20
PLANE_OPS = {OP_SUBMIT, OP_ENTER, OP_LEAVE, OP_ADVANCE, OP_RESTART}


@dataclass(frozen=True)
class Step:
    """One operation. `expect` is the (verdict, refusal) the specification requires; `grade`
    names the row the step is graded in (None: traced only)."""

    op: int
    reg: int = 0
    vreg: int = 0
    data: bytes = b""
    graph: tuple | None = None  # (claims, gens, topo) for OP_FREEZE
    expect: tuple[str, str] | None = None
    grade: str | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    steps: tuple[Step, ...]
    n_slots: int = 4
    capacity: int = 4096
    key: bytes = cf.ROOT_KEY
    scope: str = cf.SCOPE
    subject: int = cf.SUBJECT

    @property
    def c_only(self) -> bool:
        return any(step.op in C_ONLY_OPS for step in self.steps)


APPLIED = ("applied", "none")


def _refused(refusal: str) -> tuple[str, str]:
    return ("refused", refusal)


def submit(data: bytes, grade: str | None = None) -> Step:
    return Step(OP_SUBMIT, data=data, grade=grade)


def store(pack: bytes, reg: int = 0) -> list[Step]:
    """A producer writes the artifact once into a reserved slot and commits it."""
    return [
        Step(OP_RESERVE, reg, expect=APPLIED),
        Step(OP_WRITE, reg, data=struct.pack("<I", 0) + pack, expect=APPLIED),
        Step(OP_COMMIT, reg, data=struct.pack("<Q", len(pack)), expect=APPLIED),
    ]


def admit(reg: int = 0, expect=APPLIED, grade: str | None = None) -> Step:
    return Step(OP_ADMIT, reg, expect=expect, grade=grade)


def dispatch(reg: int = 0, expect=APPLIED, grade: str | None = None) -> Step:
    return Step(OP_DISPATCH, reg, expect=expect, grade=grade)


def boot(registry: tuple) -> list[Step]:
    """Grant lease 1 and install `registry` at generation 1."""
    return [submit(cf.grant(1, seq=1)), submit(cf.generation_record(registry, seq=1, expect=0))]


# --- the admission corpus (a pack older than the registry generation) ------------------------

_TARGET = None


def _target():
    global _TARGET
    if _TARGET is None:
        from ..kbcir.cost import TargetProfile

        _TARGET = TargetProfile.x86_avx512()
    return _TARGET


def _plan(module):
    from ..kbcir import optimize
    from ..kbcir.cost import Theta

    return optimize(module, _target(), Theta.cool())


def reg_of(pack) -> tuple:
    return (pack.map_gen, pack.data_gen, pack.topo_gen, registry_digest(pack.generations))


@dataclass(frozen=True)
class AdmissionCase:
    label: str
    stale: bool
    registry: tuple  # installed at generation 1
    pack: bytes
    switch: tuple | None = None  # a registry installed after the admission (the window)


_ADMISSION: list[AdmissionCase] | None = None


def admission_cases() -> list[AdmissionCase]:
    """Every way a pack can be older than the live registry -- the maxima moved, a resource moved
    under unchanged maxima (map and data), no vector, a resource the registry no longer declares
    or one declared after hydration, another program whose maxima coincide, the topology moved --
    beside the fresh packs, and the admit -> dispatch window."""
    global _ADMISSION
    if _ADMISSION is not None:
        return _ADMISSION
    from ..examples import multi_histogram, vector_add

    art = cf.artifacts()
    cases = [
        AdmissionCase("fresh/current", False, art.registry, art.pack),
        AdmissionCase("stale/maxima-moved", True, art.registry, art.pack_old),
        AdmissionCase("stale/moved-under-maxima", True, art.old_registry, art.pack_undermax),
        AdmissionCase("stale/no-vector", True, art.registry, art.pack_novector),
    ]
    m = vector_add(64)
    res = _plan(m)
    rids = sorted(m.resources)
    m.resources[rids[0]] = replace(m.resources[rids[0]], data_gen=2)
    m.touch()
    before = hydrate(m, res)
    m.resources[rids[1]] = replace(m.resources[rids[1]], data_gen=1)
    m.touch()
    after = hydrate(m, res)
    cases += [
        AdmissionCase("fresh/data-current", False, reg_of(after), encode(after)),
        AdmissionCase("stale/data-moved-under-maxima", True, reg_of(after), encode(before)),
        AdmissionCase(
            "stale/undeclared-rid",
            True,
            reg_of(after),
            encode(replace(after, generations=[*after.generations, Generation(max(rids) + 1)])),
        ),
        AdmissionCase(
            "stale/missing-rid",
            True,
            reg_of(after),
            encode(replace(after, generations=after.generations[:-1])),
        ),
    ]
    other = hydrate(multi_histogram(), _plan(multi_histogram()))
    live = reg_of(after)
    gens = list(other.generations)
    gens[0] = Generation(gens[0].rid, live[0], live[1])
    forged = replace(other, map_gen=live[0], data_gen=live[1], generations=gens)
    cases.append(AdmissionCase("stale/other-program-same-maxima", True, live, encode(forged)))
    topo = (after.map_gen, after.data_gen, 2, registry_digest(after.generations))
    cases.append(AdmissionCase("stale/topology-moved", True, topo, encode(after)))
    moved = replace(
        after, generations=[replace(after.generations[1], map_gen=5), *after.generations[:1],
                            *after.generations[2:]]
    )  # fmt: skip
    moved = replace(
        moved,
        generations=sorted(moved.generations, key=lambda g: g.rid),
        map_gen=max(g.map_gen for g in moved.generations),
    )
    cases.append(
        AdmissionCase(
            "window/switch-after-admit", True, reg_of(after), encode(after), reg_of(moved)
        )
    )
    _ADMISSION = cases
    return cases


def admission_scenarios() -> list[Scenario]:
    """Each case through the seam: stale packs refused at admit AND at dispatch (never run);
    fresh packs admitted and run; the window -- admitted, the registry moves, then refused at
    dispatch and at re-admission; an activation between admit and dispatch (the admission lives
    within one resident generation); a plane with no registry; a malformed pack; a draining plane;
    and a switch requested while another phase is in flight, deferred past a dispatch and landing
    at the boundary."""
    out = []
    for case in admission_cases():
        steps = [*boot(case.registry), *store(case.pack)]
        if case.switch is not None:
            steps += [
                admit(expect=APPLIED, grade="fresh-admit"),
                dispatch(expect=APPLIED, grade="fresh-dispatch"),
                submit(cf.generation_record(case.switch, seq=2, expect=1)),
                dispatch(expect=_refused("stale"), grade="stale-dispatch"),
                admit(expect=_refused("stale"), grade="stale-admit"),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
            ]
        elif case.stale:
            steps += [
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
                admit(expect=_refused("stale"), grade="stale-admit"),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
            ]
        else:
            steps += [
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
                admit(expect=APPLIED, grade="fresh-admit"),
                dispatch(expect=APPLIED, grade="fresh-dispatch"),
                dispatch(expect=APPLIED, grade="fresh-dispatch"),
            ]
        out.append(Scenario(f"admission/{case.label}", "admission", tuple(steps)))
    art = cf.artifacts()
    act = cf.issue(cf.record("activate", Activate(cf.A_DIGEST), seq=2, expect=1, lease=1))
    out.append(
        Scenario(
            "admission/activation-between",
            "admission",
            (
                *boot(art.registry),
                *store(art.pack),
                admit(grade="fresh-admit"),
                submit(act),
                dispatch(expect=_refused("stale"), grade="stale-dispatch"),
                admit(grade="fresh-admit"),
                dispatch(grade="fresh-dispatch"),
            ),
        )
    )
    out.append(
        Scenario(
            "admission/no-registry",
            "admission",
            (
                submit(cf.grant(1, seq=1)),
                *store(art.pack),
                admit(expect=_refused("stale"), grade="stale-admit"),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
            ),
        )
    )
    corrupt = bytearray(art.pack)
    corrupt[70] ^= 0xFF  # a body byte: the CRC no longer holds
    out.append(
        Scenario(
            "admission/malformed",
            "admission",
            (
                *boot(art.registry),
                *store(bytes(corrupt)),
                admit(expect=_refused("malformed")),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
            ),
        )
    )
    drain = cf.issue(cf.record("quiesce", Quiesce(1000), seq=2, expect=1, lease=1))
    out.append(
        Scenario(
            "admission/draining",
            "admission",
            (
                *boot(art.registry),
                *store(art.pack),
                admit(grade="fresh-admit"),
                submit(drain),
                dispatch(expect=_refused("draining"), grade="stale-dispatch"),
            ),
        )
    )
    after = admission_cases()[4]  # fresh/data-current's registry and pack
    window = admission_cases()[-1]
    out.append(
        Scenario(
            "admission/switch-deferred-past-dispatch",
            "admission",
            (
                *boot(after.registry),
                *store(after.pack),
                admit(grade="fresh-admit"),
                Step(OP_ENTER),  # another worker's phase is in flight
                submit(cf.generation_record(window.switch, seq=2, expect=1)),  # deferred
                dispatch(grade="fresh-dispatch"),  # nested phase: no boundary, no switch
                Step(OP_LEAVE),  # the boundary: the switch lands here, once
                dispatch(expect=_refused("stale"), grade="stale-dispatch"),
            ),
        )
    )
    # The plane restarts and reaches generation 1 again. The generation number alone cannot tell
    # the two incarnations apart: under ANOTHER registry the admission is stale (the registry
    # binding refuses it), under the SAME registry it still holds (the binding is exactly
    # (generation, registry), not the plane's identity).
    out.append(
        Scenario(
            "admission/restart-other-registry",
            "admission",
            (
                *boot(after.registry),
                *store(after.pack),
                admit(grade="fresh-admit"),
                dispatch(grade="fresh-dispatch"),
                Step(OP_RESTART),
                dispatch(expect=_refused("stale"), grade="stale-dispatch"),  # generation 0
                *boot(art.registry),  # generation 1 again -- another registry
                dispatch(expect=_refused("stale"), grade="stale-dispatch"),
                admit(expect=_refused("stale"), grade="stale-admit"),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
            ),
        )
    )
    out.append(
        Scenario(
            "admission/restart-same-registry",
            "admission",
            (
                *boot(after.registry),
                *store(after.pack),
                admit(grade="fresh-admit"),
                Step(OP_RESTART),
                *boot(after.registry),
                dispatch(grade="fresh-dispatch"),
            ),
        )
    )
    return out


# --- lifetime -----------------------------------------------------------------------------------


def lifetime_scenarios() -> list[Scenario]:
    """A view that outlives its owner is refused, never read -- released, reused by the next
    step's same-length pack, aborted, never issued -- while a borrow in progress keeps its bytes
    and frees the slot when returned; the bounds (a full table, a capacity, an exhausted epoch or
    pin count) refuse rather than wrap."""
    art = cf.artifacts()
    life = _refused("lifetime")
    p1, p2 = admission_cases()[4].pack, admission_cases()[5].pack  # two same-length packs
    assert len(p1) == len(p2) and p1 != p2
    reg = admission_cases()[4].registry
    out = [
        Scenario(
            "lifetime/view-after-release",
            "lifetime",
            (
                *store(art.pack),
                Step(OP_RELEASE, expect=APPLIED),
                Step(OP_BORROW, expect=life, grade="lifetime"),
                Step(OP_RELEASE, expect=life, grade="lifetime"),
                Step(OP_ADMIT, expect=life, grade="lifetime"),
                Step(OP_DISPATCH, expect=life, grade="lifetime"),
            ),
        ),
        Scenario(
            "lifetime/reused-slot-same-length",
            "lifetime",
            (
                *boot(reg),
                *store(p1, reg=0),
                admit(0, grade="fresh-admit"),
                Step(OP_RELEASE, 0, expect=APPLIED),
                *store(p2, reg=1),  # the next step's pack in the same slot, one epoch on
                Step(OP_ADMIT, 0, expect=life, grade="lifetime"),
                Step(OP_DISPATCH, 0, expect=life, grade="lifetime"),
                Step(OP_BORROW, 0, 1, expect=life, grade="lifetime"),
                dispatch(1, expect=_refused("unadmitted"), grade="stale-dispatch"),
            ),
        ),
        Scenario(
            "lifetime/borrow-outlives-release",
            "lifetime",
            (
                *store(art.pack),
                Step(OP_BORROW, 0, 0, expect=APPLIED),
                Step(OP_RELEASE, expect=APPLIED),  # pinned: the slot retires, the bytes stay
                Step(OP_BORROW, 0, 1, expect=life, grade="lifetime"),
                Step(OP_RESERVE, 1, expect=APPLIED),  # a different, free slot
                Step(OP_GIVE_BACK, 0, 0, expect=APPLIED),  # the last return frees slot 0
                Step(OP_GIVE_BACK, 0, 0, expect=life, grade="lifetime"),
                Step(OP_RESERVE, 2, expect=APPLIED),  # slot 0 again, epoch 2
            ),
            n_slots=2,
        ),
        Scenario(
            "lifetime/producer-misuse",
            "lifetime",
            (
                Step(OP_BORROW, expect=life, grade="lifetime"),  # nothing issued yet
                Step(OP_RESERVE, expect=APPLIED),
                Step(OP_BORROW, expect=life, grade="lifetime"),  # building: not published
                Step(OP_ADMIT, expect=life, grade="lifetime"),
                Step(
                    OP_WRITE,
                    data=struct.pack("<I", 4090) + b"12345678",
                    expect=_refused("capacity"),
                ),
                Step(OP_COMMIT, data=struct.pack("<Q", 0), expect=_refused("capacity")),
                Step(OP_COMMIT, data=struct.pack("<Q", 4097), expect=_refused("capacity")),
                Step(OP_WRITE, data=struct.pack("<I", 0) + art.pack, expect=APPLIED),
                Step(OP_COMMIT, data=struct.pack("<Q", len(art.pack)), expect=APPLIED),
                Step(
                    OP_COMMIT, data=struct.pack("<Q", len(art.pack)), expect=life, grade="lifetime"
                ),
                Step(OP_ABORT, expect=life, grade="lifetime"),
                Step(OP_WRITE, data=struct.pack("<I", 0) + b"x", expect=life, grade="lifetime"),
            ),
        ),
        Scenario(
            "lifetime/abort-publishes-nothing",
            "lifetime",
            (
                Step(OP_RESERVE, expect=APPLIED),
                Step(OP_WRITE, data=struct.pack("<I", 0) + art.pack, expect=APPLIED),
                Step(OP_ABORT, expect=APPLIED),
                Step(OP_BORROW, expect=life, grade="lifetime"),
                Step(
                    OP_COMMIT, data=struct.pack("<Q", len(art.pack)), expect=life, grade="lifetime"
                ),
                Step(OP_RESERVE, 1, expect=APPLIED),  # slot 0, epoch 2
            ),
        ),
        Scenario(
            "lifetime/table-full",
            "lifetime",
            (
                *[Step(OP_RESERVE, r, expect=APPLIED) for r in range(2)],
                Step(OP_RESERVE, 2, expect=_refused("full")),
                Step(OP_ABORT, 0, expect=APPLIED),
                Step(OP_RESERVE, 2, expect=APPLIED),
            ),  # fmt: skip
            n_slots=2,
        ),
        # C-only witnesses: a forged handle; an epoch and a pin count at their bounds.
        Scenario(
            "lifetime/forged-handles",
            "lifetime",
            (
                *store(art.pack),
                Step(OP_FORGE, 1, data=struct.pack("<II", 99, 1)),
                Step(OP_BORROW, 1, expect=life, grade="lifetime"),  # out of range
                Step(OP_FORGE, 1, data=struct.pack("<II", 0, 0)),
                Step(OP_BORROW, 1, expect=life, grade="lifetime"),  # epoch 0: never issued
                Step(OP_FORGE, 1, data=struct.pack("<II", 0, 2)),
                Step(OP_BORROW, 1, expect=life, grade="lifetime"),  # a future epoch
                Step(OP_FORGE, 1, data=struct.pack("<II", 0, 1)),
                Step(OP_BORROW, 1, 1, expect=APPLIED),  # the live incarnation, spelled by hand
                Step(OP_GIVE_BACK, 0, 1, expect=APPLIED),
            ),
        ),
        Scenario(
            "lifetime/epoch-exhausted",
            "lifetime",
            (
                Step(OP_POKE_EPOCH, data=struct.pack("<II", 0, EPOCH_MAX)),
                *store(art.pack),
                Step(OP_BORROW, 0, 0, expect=APPLIED),
                Step(OP_RELEASE, expect=APPLIED),  # the last epoch: out of service for good
                Step(OP_BORROW, 0, 1, expect=life, grade="lifetime"),
                Step(OP_GIVE_BACK, 0, 0, expect=APPLIED),  # the pin drains; still out of service
                Step(OP_RESERVE, 1, expect=_refused("full")),
            ),
            n_slots=1,
        ),
        Scenario(
            "lifetime/pins-exhausted",
            "lifetime",
            (
                *store(art.pack),
                Step(OP_POKE_PINS, data=struct.pack("<II", 0, PINS_MAX)),
                Step(OP_BORROW, expect=_refused("exhausted")),
                Step(OP_POKE_PINS, data=struct.pack("<II", 0, 0)),
                Step(OP_BORROW, 0, 0, expect=APPLIED),
            ),
        ),
    ]
    return out


# --- the per-step freeze -------------------------------------------------------------------------

R_LOAD, R_STORE, R_ADD, R_MUL, R_NOP = 1, 2, 3, 5, 0


def claim(cid, op=R_ADD, reads=(), writes=(), label=None, lane=0, domain=0, count=1) -> GraphClaim:
    names = {R_LOAD: "c.load", R_STORE: "c.store", R_ADD: "c.add", R_MUL: "c.mul", R_NOP: "c.nop"}
    return GraphClaim(cid, op, lane, domain, count, tuple(reads), tuple(writes),
                      label if label is not None else names.get(op, "c.op"))  # fmt: skip


def step_graph(t: int, registry) -> tuple:
    """Step t of an agent that spawns a node per step: t+2 claims over the registry's RIDs --
    loads, a chain of adds and a store; the topology grows with t."""
    rids = [g.rid for g in registry]
    claims = [claim(1, R_LOAD, (rids[0],))]
    for k in range(t):
        claims.append(claim(2 + k, R_ADD, (rids[k % len(rids)], rids[-1]), (rids[-1],)))
    claims.append(claim(2 + t, R_STORE, (rids[-1],), (rids[0],)))
    return claims


def _graph_registry(gens) -> tuple:
    vector = [Generation(g.rid, g.map_gen, g.data_gen) for g in gens]
    return (max(g.map_gen for g in gens), max(g.data_gen for g in gens), 1,
            registry_digest(vector))  # fmt: skip


def freeze_step(claims, gens, topo=1, reg=0, expect=APPLIED, grade="freeze") -> Step:
    return Step(
        OP_FREEZE, reg, graph=(tuple(claims), tuple(gens), topo), expect=expect, grade=grade
    )


BUILDER_GENS = (GraphGeneration(1, 1, 0), GraphGeneration(2, 0, 3), GraphGeneration(4, 2, 1))
BUILDER_NEXT = (GraphGeneration(1, 1, 0), GraphGeneration(2, 1, 3), GraphGeneration(4, 2, 1))


def freeze_scenarios() -> list[Scenario]:
    """The dynamic-graph builder: a fresh graph frozen per step into a slot, admitted against the
    live registry, run (its claims in graph order) and released; a step frozen before a registry
    switch is refused after it, and the next step frozen under the new registry runs; every freeze
    law refuses transactionally -- the slot aborted, nothing published."""
    reg = _graph_registry(BUILDER_GENS)
    reg_next = _graph_registry(BUILDER_NEXT)
    steps = [*boot(reg)]
    for t in range(4):
        steps += [
            freeze_step(step_graph(t, BUILDER_GENS), BUILDER_GENS),
            admit(grade="fresh-admit"),
            dispatch(grade="fresh-dispatch"),
            Step(OP_RELEASE, expect=APPLIED),
        ]
    out = [Scenario("freeze/steps", "freeze", tuple(steps))]
    out.append(
        Scenario(
            "freeze/switch-between-steps",
            "freeze",
            (
                *boot(reg),
                freeze_step(step_graph(2, BUILDER_GENS), BUILDER_GENS),
                submit(cf.generation_record(reg_next, seq=2, expect=1)),
                admit(expect=_refused("stale"), grade="stale-admit"),
                dispatch(expect=_refused("unadmitted"), grade="stale-dispatch"),
                Step(OP_RELEASE, expect=APPLIED),
                freeze_step(step_graph(3, BUILDER_NEXT), BUILDER_NEXT, reg=1),
                admit(1, grade="fresh-admit"),
                dispatch(1, grade="fresh-dispatch"),
            ),
        )
    )
    bad = _refused("malformed")
    g = BUILDER_GENS
    laws = [
        ("overflow", [claim(1, R_LOAD, (1,), count=2**32 - 1)], g, bad),
        ("no-vector", step_graph(1, g), (), bad),
        ("unsorted-vector", step_graph(1, g), (g[1], g[0], g[2]), bad),
        ("too-many-reads", [claim(1, R_ADD, (1, 1, 1, 1, 1, 1, 1))], g, bad),
        ("label-control-char", [claim(1, label="c.a\tdd")], g, bad),
        ("label-no-terminator", [claim(1, label="x" * 32)], g, bad),
        ("ids-not-ascending", [claim(2), claim(2)], g, bad),
        ("lane-out-of-range", [claim(1, lane=6)], g, bad),
        ("undeclared-rid", [claim(1, R_LOAD, (3,))], g, bad),
        ("capacity", [claim(i, R_ADD, (1,), (2,)) for i in range(1, 60)], g, _refused("capacity")),
    ]
    for name, claims, gens, expect in laws:
        out.append(
            Scenario(
                f"freeze/law-{name}",
                "freeze",
                (
                    freeze_step(claims, gens, expect=expect),
                    Step(OP_RESERVE, 1, expect=APPLIED),  # the aborted slot is free again
                ),
                capacity=1024,
            )
        )
    return out


# --- the manifest at the plane -------------------------------------------------------------------


def manifest_scenarios() -> list[Scenario]:
    whole = admission_cases()[4]
    sp = split(whole.pack, partition(len(decode(whole.pack).segments), 2))
    old = admission_cases()[5]
    sp_old = split(old.pack, [(0, len(decode(old.pack).segments))])
    torn = bytearray(sp.manifest)
    torn[9] ^= 0x01  # n_shards, the CRC now wrong
    return [
        Scenario(
            "manifest/gated-at-the-plane",
            "manifest",
            (
                Step(OP_ADMIT_MANIFEST, data=sp.manifest, expect=_refused("stale"), grade="stale-admit"),
                *boot(whole.registry),
                Step(OP_ADMIT_MANIFEST, data=sp.manifest, expect=APPLIED, grade="fresh-admit"),
                Step(OP_ADMIT_MANIFEST, data=sp_old.manifest, expect=_refused("stale"),
                     grade="stale-admit"),
                Step(OP_ADMIT_MANIFEST, data=bytes(torn), expect=_refused("malformed")),
            ),
        )
    ]  # fmt: skip


def all_scenarios() -> list[Scenario]:
    return lifetime_scenarios() + admission_scenarios() + freeze_scenarios() + manifest_scenarios()


FAMILIES = ("lifetime", "admission", "freeze", "manifest")


# --- the Python rail -------------------------------------------------------------------------------


def _claims_digest(claims) -> str:
    return hashlib.sha256(b"".join(struct.pack("<Q", c) for c in claims)).hexdigest()[:32]


def trace_line(
    scn: int, index: int, op: int, outcome, table: PackTable, plane: ControlPlane
) -> str:
    """One operation's trace line -- the format both harnesses print."""
    hide = op in (OP_ADMIT, OP_DISPATCH) and outcome.refusal in ("malformed", "walk")
    handle = outcome.handle
    claims = _claims_digest(outcome.claims) if op == OP_DISPATCH and outcome.applied else "-"
    return (
        f"{scn} {index} {HO_VERDICTS.index(outcome.verdict)} {HO_REFUSALS.index(outcome.refusal)} "
        f"{'-' if hide else outcome.status} "
        f"h={f'{handle.index}:{handle.epoch}' if handle is not None else '-'} "
        f"g={outcome.generation} c={claims} t={table.state_digest().hex()} "
        f"p={plane.state_digest().hex()}"
    )


def _plane_outcome(c: ControlOutcome):
    from ..gem.handoff import HandoffOutcome

    return HandoffOutcome(
        "refused" if c.verdict == "refused" else "applied", "none", c.status, c.generation
    )


def run_python(scenario: Scenario, index: int = 0) -> tuple[list, list[str]]:
    """Run one scenario on the oracle: (outcomes, trace lines)."""
    from ..gem.handoff import HandoffOutcome

    table = PackTable(scenario.n_slots, scenario.capacity)
    plane = ControlPlane(scenario.key, scenario.scope, scenario.subject)
    regs: list[Handle | None] = [None] * 16
    views: list[View | None] = [None] * 16
    outcomes, lines = [], []
    for step_index, step in enumerate(scenario.steps):
        op, reg, vreg, data = step.op, step.reg, step.vreg, step.data
        if op == OP_SUBMIT:
            o = _plane_outcome(plane.submit(data))
        elif op == OP_ENTER:
            o = _plane_outcome(plane.enter())
        elif op == OP_LEAVE:
            o = _plane_outcome(plane.leave())
        elif op == OP_ADVANCE:
            o = _plane_outcome(plane.advance())
        elif op == OP_RESTART:
            plane = ControlPlane(scenario.key, scenario.scope, scenario.subject)
            o = HandoffOutcome("applied", generation=plane.generation)
        elif op == OP_RESERVE:
            o = table.reserve()
            if o.applied:
                regs[reg] = o.handle
        elif op == OP_WRITE:
            (offset,) = struct.unpack_from("<I", data)
            o = table.write(regs[reg], offset, data[4:])
        elif op == OP_COMMIT:
            (length,) = struct.unpack("<Q", data)
            o = table.commit(regs[reg], length)
        elif op == OP_ABORT:
            o = table.abort(regs[reg])
        elif op == OP_BORROW:
            o, views[vreg] = table.borrow(regs[reg])
        elif op == OP_GIVE_BACK:
            o = table.give_back(views[vreg])
            if o.applied:
                views[vreg] = None
        elif op == OP_RELEASE:
            o = table.release(regs[reg])
        elif op == OP_ADMIT:
            o = table.admit(regs[reg], plane)
        elif op == OP_DISPATCH:
            o = table.dispatch(regs[reg], plane)
        elif op == OP_ADMIT_MANIFEST:
            o = admit_manifest(plane, data)
        elif op == OP_FREEZE:
            claims, gens, topo = step.graph
            o = table.freeze(claims, gens, topo)
            if o.applied:
                regs[reg] = o.handle
        elif op == OP_FORGE:
            regs[reg] = Handle(*struct.unpack("<II", data))
            o = HandoffOutcome("applied")
        elif op in (OP_POKE_EPOCH, OP_POKE_PINS):
            slot, value = struct.unpack("<II", data)
            if slot < table.n_slots:
                setattr(table.slots[slot], "epoch" if op == OP_POKE_EPOCH else "pins", value)
            o = HandoffOutcome("applied")
        elif op == OP_COPY_VIEW:
            views[vreg] = views[reg]
            o = HandoffOutcome("applied")
        else:
            raise ValueError(f"unknown op {op}")
        outcomes.append(o)
        lines.append(trace_line(index, step_index, op, o, table, plane))
    return outcomes, lines


# --- encodings for the native harnesses -------------------------------------------------------


def encode_graph(claims, gens, topo) -> bytes:
    out = bytearray(struct.pack("<II", topo, len(gens)))
    for g in gens:
        out += struct.pack("<III", g.rid, g.map_gen, g.data_gen)
    out += struct.pack("<I", len(claims))
    for c in claims:
        label = c.label.encode("utf-8")
        if b"\x00" in label or len(label) > 255:
            raise ValueError("a label a C string cannot carry")  # no C representation
        out += struct.pack("<IBBBI", c.id, c.opcode, c.lane, c.domain, c.count)
        out += struct.pack("<B", len(c.reads)) + b"".join(struct.pack("<I", r) for r in c.reads)
        out += struct.pack("<B", len(c.writes)) + b"".join(struct.pack("<I", w) for w in c.writes)
        out += struct.pack("<B", len(label)) + label
    return bytes(out)


def encode_script(scenarios) -> bytes:
    out = bytearray(b"BHOS" + struct.pack("<I", len(scenarios)))
    for s in scenarios:
        out += struct.pack("<II", s.n_slots, s.capacity)
        out += struct.pack("<I", len(s.key)) + s.key
        out += struct.pack("<BQI", CONTROL_SCOPES.index(s.scope), s.subject, len(s.steps))
        for step in s.steps:
            data = encode_graph(*step.graph) if step.op == OP_FREEZE else step.data
            out += struct.pack("<BBBI", step.op, step.reg, step.vreg, len(data)) + data
    return bytes(out)


def _frames(blobs) -> bytes:
    return b"".join(struct.pack("<I", len(b)) + b for b in blobs)


# --- the native harnesses -------------------------------------------------------------------------


def compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def cxx_compiler() -> str | None:
    return shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")


def _run(argv, *, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a harness; one that cannot be launched or hangs is a RuntimeError the measurement counts
    as failed fixtures (L1: every exit is a verdict), never a traceback."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{argv[0]}: {exc}") from exc


def _missing(path: str) -> bool:
    """A harness source absent: a failure in a source checkout (L21), a skip in an install."""
    from .run_all import _is_source_checkout

    if os.path.isfile(path):
        return False
    if _is_source_checkout():
        raise RuntimeError(f"{os.path.relpath(path, REPO)} is missing from the checkout")
    return True


def build_harness(tmp: str, *, extra_flags=()) -> str | None:
    """Compile runtime/c/test_handoff.c against the freestanding units. None without a compiler
    (the quick tier hides one on purpose) or in an installed package."""
    cc = compiler()
    if cc is None or _missing(os.path.join(C_DIR, "test_handoff.c")):
        return None
    exe = os.path.join(tmp, "test_handoff" + (".exe" if os.name == "nt" else ""))
    build = _run(
        [cc, "-std=c11", "-O2", "-Wall", "-Wextra", *extra_flags, "-I", C_DIR,
         os.path.join(C_DIR, "test_handoff.c"), *[os.path.join(C_DIR, u) for u in C_UNITS],
         "-o", exe],
    )  # fmt: skip
    if build.returncode != 0:
        raise RuntimeError(f"handoff harness build failed: {build.stderr[-2000:]}")
    return exe


def build_cpp_program(tmp: str, source: str, name: str, *, extra_flags=()) -> str | None:
    """Compile one C++ program of runtime/cpp (`source`, a path) with the seam (CPP_UNITS) over
    the C units, compiled as C -- the one build every C++ harness and test uses, so no second
    source list can drift from the gates' (the #719 trap). None without both compilers or in an
    installed package."""
    cc, cxx = compiler(), cxx_compiler()
    if cc is None or cxx is None or _missing(source):
        return None
    objects = []
    for unit in C_UNITS:
        obj = os.path.join(tmp, unit.replace(".c", ".o"))
        build = _run([cc, "-std=c11", "-O2", *extra_flags, "-I", C_DIR, "-c",
                      os.path.join(C_DIR, unit), "-o", obj])  # fmt: skip
        if build.returncode != 0:
            raise RuntimeError(f"{unit} build failed: {build.stderr[-2000:]}")
        objects.append(obj)
    exe = os.path.join(tmp, name + (".exe" if os.name == "nt" else ""))
    build = _run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", *extra_flags, "-I", C_DIR, "-I", CPP_DIR,
         source, *[os.path.join(CPP_DIR, u) for u in CPP_UNITS], *objects, "-o", exe],
    )  # fmt: skip
    if build.returncode != 0:
        raise RuntimeError(f"C++ {name} build failed: {build.stderr[-2000:]}")
    return exe


def build_cpp_harness(tmp: str, *, extra_flags=()) -> str | None:
    """Compile runtime/cpp/test_handoff.cpp and the seam over the C units (compiled as C). None
    without both compilers or in an installed package."""
    source = os.path.join(CPP_DIR, "test_handoff.cpp")
    return build_cpp_program(tmp, source, "test_handoff_cpp", extra_flags=extra_flags)


# The fixture root key and the plane the seam's own harness (runtime/cpp/test_orchestrator.cpp)
# boots: scope "module", subject 7 -- the values its kRoot and bcir_ctl_init call hard-code.
SEAM_ROOT = bytes(range(0x10, 0x30))
SEAM_SCOPE, SEAM_SUBJECT = "module", 7


def seam_artifacts(example: str) -> tuple[bytes, bytes]:
    """(pack, plane) for test_orchestrator.cpp: a corpus program hydrated by the C/IR path, and
    three u32-framed ControlRecordV1 records under the seam's root key -- a lease, the switch
    installing the pack's own registry, and a second switch to a moved registry. The one minter
    tools/cpp/check_handoff.sh and bcir/tests/test_cpp_handoff.py share."""
    from ..abi.control_abi import issue_control
    from ..gem.control import (
        CAP_GRANTABLE,
        ControlRecord,
        GenerationSwitch,
        LeaseGrant,
        lease_key,
    )
    from .. import examples

    module = getattr(examples, example)
    m = module(1024) if example == "vector_add" else module()
    pack = hydrate(m, _plan(m))

    def rec(kind, body, seq, expect, lease):
        return ControlRecord(
            kind=kind, scope=SEAM_SCOPE, subject=SEAM_SUBJECT,
            generation=expect + 1 if kind == "generation" else expect, expect=expect,
            boundary=0, sequence=seq, lease=lease, body=body,
        )  # fmt: skip

    def switch(gens, seq, expect):
        body = GenerationSwitch(
            max(g.map_gen for g in gens), max(g.data_gen for g in gens), pack.topo_gen,
            registry_digest(gens),
        )  # fmt: skip
        return issue_control(rec("generation", body, seq, expect, 1), lease_key(SEAM_ROOT, 1))

    moved = list(pack.generations)
    moved[0] = replace(moved[0], map_gen=moved[0].map_gen + 5)
    records = (
        issue_control(rec("lease", LeaseGrant(1, CAP_GRANTABLE, 0, 1000, 42), 1, 0, 0), SEAM_ROOT),
        switch(pack.generations, 1, 0),
        switch(moved, 2, 1),
    )
    return encode(pack), _frames(records)


def run_native(exe: str, tmp: str, scenarios) -> tuple[list[list[str]], int | None]:
    """Run scenarios on a native harness: one list of trace lines per scenario, and the C++
    harness's count of segment views read outside their slot (None from the C harness)."""
    path = os.path.join(tmp, "handoff_script.bin")
    with open(path, "wb") as fh:
        fh.write(encode_script(scenarios))
    run = _run([exe, "--script", path])
    if run.returncode != 0:
        raise RuntimeError(f"handoff harness failed: {run.stderr[-2000:]}")
    traces: list[list[str]] = [[] for _ in scenarios]
    copies = None
    for line in run.stdout.splitlines():
        if line.startswith("COPIES "):
            copies = int(line.split()[1])
            continue
        head = line.split(" ", 1)[0]
        traces[int(head)].append(line)
    return traces, copies


def _body(lines: list[str]) -> list[str]:
    return [line.split(" ", 1)[1] if " " in line else line for line in lines]


def outcomes_of(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for line in lines:
        parts = line.split()
        out.append((HO_VERDICTS[int(parts[2])], HO_REFUSALS[int(parts[3])]))
    return out


def run_mode(exe: str, tmp: str, mode: str, payload: bytes) -> list[str]:
    path = os.path.join(tmp, f"handoff_{mode}.bin")
    with open(path, "wb") as fh:
        fh.write(payload)
    run = _run([exe, f"--{mode}", path])
    if run.returncode != 0:
        raise RuntimeError(f"handoff --{mode} failed: {run.stderr[-2000:]}")
    return run.stdout.splitlines()


# --- the freeze corpus ---------------------------------------------------------------------------


def freeze_corpus(seed: int = 16) -> list[tuple[tuple, tuple, int, int]]:
    """(claims, gens, topo, capacity): the builder's step graphs and every law's witness, then a
    seeded sweep that is mostly legal -- so the bytes, not only the refusals, are compared."""
    g = BUILDER_GENS
    cases = [(tuple(step_graph(t, g)), g, 1, 4096) for t in range(12)]
    cases += [(tuple(claims), tuple(gens), 1, 1024) for _, claims, gens, _ in _law_graphs()]
    rng = random.Random(seed)
    for _ in range(160):
        rids = sorted(rng.sample(range(1, 64), rng.randrange(1, 6)))
        gens = tuple(GraphGeneration(r, rng.randrange(0, 4), rng.randrange(0, 4)) for r in rids)
        claims, cid = [], rng.randrange(1, 9)
        for _ in range(rng.randrange(0, 24)):
            op = rng.choice((R_LOAD, R_STORE, R_ADD, R_MUL, R_NOP, 6, 9, 16))
            claims.append(
                GraphClaim(
                    cid,
                    op,
                    rng.randrange(0, 6),
                    rng.choice((0, 0, 1, 3)),
                    rng.randrange(0, 5000),
                    tuple(rng.choice(rids) for _ in range(rng.randrange(0, 7))),
                    tuple(rng.choice(rids) for _ in range(rng.randrange(0, 3))),
                    rng.choice(("c.add", "c.load", "c.store", "c.mul", "atomic", "")),
                )
            )
            cid += rng.randrange(1, 4)
        if rng.random() < 0.1 and claims:  # one law broken on purpose
            k = rng.randrange(len(claims))
            claims[k] = replace(claims[k], lane=rng.choice((6, 7, 255)))
        cases.append((tuple(claims), gens, rng.randrange(0, 3), rng.choice((4096, 4096, 256))))
    return cases


def _law_graphs():
    g = BUILDER_GENS
    return [
        ("overflow", [claim(1, R_LOAD, (1,), count=2**32 - 1)], g, None),
        ("no-vector", step_graph(1, g), (), None),
        ("unsorted-vector", step_graph(1, g), (g[1], g[0], g[2]), None),
        ("too-many-reads", [claim(1, R_ADD, (1,) * 7)], g, None),
        ("too-many-writes", [claim(1, R_ADD, (1,), (1, 2, 4))], g, None),
        ("label-control-char", [claim(1, label="c.a\tdd")], g, None),
        ("label-no-terminator", [claim(1, label="x" * 32)], g, None),
        ("label-longest", [claim(1, label="x" * 31)], g, None),
        ("ids-not-ascending", [claim(2), claim(2)], g, None),
        ("nop-ids-count", [claim(1), claim(1, R_NOP)], g, None),
        ("lane-out-of-range", [claim(1, lane=6)], g, None),
        ("undeclared-rid", [claim(1, R_LOAD, (3,))], g, None),
        ("capacity", [claim(i, R_ADD, (1,), (2,)) for i in range(1, 60)], g, None),
        ("empty-graph", [], g, None),
        ("nop-only", [claim(1, R_NOP)], g, None),
    ]


def expected_freeze(claims, gens, topo, cap) -> str:
    try:
        return "OK " + freeze_claims(claims, gens, topo, cap).hex()
    except FreezeError as exc:
        return "ERR " + exc.status


def freeze_payload(cases) -> bytes:
    out = bytearray(struct.pack("<I", len(cases)))
    for claims, gens, topo, cap in cases:
        g = encode_graph(claims, gens, topo)
        out += struct.pack("<I", len(g)) + g + struct.pack("<I", cap)
    return bytes(out)


# --- the shard corpora -------------------------------------------------------------------------


def synthetic_pack(n: int, *, shuffled_trace=False, twice_named=False, out_of_order=False) -> bytes:
    """A hydrated-layout v4 pack of `n` segments -- every third names its own prefetch, every
    seventh has an unnamed double-buffer contract beside it, one block per segment -- or, with a
    flag, one that breaks the layout (S1 or S2)."""
    segs, pfs, trace, blocks = [], [], [], []
    for i in range(n):
        pf = f"pf{i}" if i % 3 == 0 else None
        if pf:
            pfs.append(Prefetch(pf, 4, (i % 5 + 1,)))
        if i % 7 == 0:
            pfs.append(Prefetch(f"db{i}", 2, (1,), "T1", "double_buffer", 2))
        segs.append(
            LaneSegment(f"s{i}", 1000 + i, i % 3, Lane(i % 6), 1 << (i % 4), f"op{i}",
                        (i % 5 + 1,), (1,), pf)
        )  # fmt: skip
        trace.append(TraceNote(1000 + i, i, 3 * i))
        blocks.append(Block(i, i + 1, (1,)))
    if shuffled_trace and n > 1:
        trace[0], trace[1] = trace[1], trace[0]
    if twice_named and n > 3:
        segs[3] = replace(segs[3], prefetch="pf0", reads=(1,))
    if out_of_order and n > 3:
        pfs[0], pfs[2] = pfs[2], pfs[0]  # pf3's record before pf0's
    gens = [Generation(r, 3 if r == 2 else 1, 2 if r == 4 else 0) for r in range(1, 6)]
    return encode(StreamPack("plan-x", 1, 3, 2, 2, segs, pfs, blocks, trace, gens))


_WHOLES: list[bytes] | None = None


def whole_packs() -> list[bytes]:
    """Packs that shard: hydrated corpus programs (plain and pipelined) and synthetic ones."""
    global _WHOLES
    if _WHOLES is not None:
        return _WHOLES
    from ..examples import fused_chain, multi_histogram, tiled_matmul, vector_add

    packs = []
    for m in (vector_add(64), multi_histogram(), tiled_matmul(), fused_chain()):
        r = _plan(m)
        packs += [encode(hydrate(m, r)), encode(hydrate_pipelined(m, r))]
    packs += [synthetic_pack(n) for n in (0, 1, 2, 7, 33, 120)]
    _WHOLES = packs
    return packs


def split_cases() -> list[tuple[bytes, list[tuple[int, int]]]]:
    """Every whole under several world sizes, plus uneven hand partitions."""
    cases = []
    for w in whole_packs():
        n = len(decode(w).segments)
        for world in (1, 2, 3, 5, 16):
            cases.append((w, partition(n, world)))
        if n >= 7:
            cases.append((w, [(0, 1), (1, n - 1), (n - 1, n)]))
    return cases


def refused_splits() -> list[tuple[str, bytes, list[tuple[int, int]]]]:
    """Packs the manifest may not shard, each refused BCIR_ERR_SHARD by every rail."""
    art = cf.artifacts()
    n = 9
    return [
        ("no-vector", art.pack_novector, [(0, 1)]),
        ("trace-not-in-segment-order", synthetic_pack(n, shuffled_trace=True), [(0, n)]),
        ("prefetch-named-twice", synthetic_pack(n, twice_named=True), [(0, n)]),
        ("prefetch-out-of-order", synthetic_pack(n, out_of_order=True), [(0, n)]),
        ("range-outside", synthetic_pack(n), [(0, n + 1)]),
        ("range-gap", synthetic_pack(n), [(0, 2), (3, n)]),
        ("range-empty", synthetic_pack(n), [(0, 0), (0, n)]),
        ("not-a-pack", b"BSPK" + bytes(80), [(0, 1)]),
    ]


def split_payload(cases) -> bytes:
    out = bytearray(struct.pack("<I", len(cases)))
    for whole, ranges in cases:
        out += struct.pack("<I", len(whole)) + whole + struct.pack("<I", len(ranges))
        out += b"".join(struct.pack("<II", b, e) for b, e in ranges)
    return bytes(out)


def expected_split(whole, ranges) -> str:
    try:
        sp = split(whole, ranges)
    except ShardError as exc:
        return "ERR " + exc.status
    return "OK " + " ".join([sp.manifest.hex(), sp.frame.hex(), *(s.hex() for s in sp.shards)])


def _recrc(data: bytes) -> bytes:
    return data[:-4] + struct.pack("<I", zlib.crc32(data[:-4]) & 0xFFFFFFFF)


def _put(data: bytes, offset: int, fmt: str, value) -> bytes:
    b = bytearray(data)
    struct.pack_into(fmt, b, offset, value)
    return _recrc(bytes(b))


def malformed_manifests() -> list[tuple[str, bytes, str]]:
    """One variant per manifest wire law, each built from a valid manifest by one mutation (the
    CRC recomputed, so it reaches the law it names), with the status both decoders must return."""
    w = synthetic_pack(9)
    good = split(w, partition(9, 3)).manifest
    empty = split(synthetic_pack(0), [(0, 0)]).manifest
    entry = 160
    return [
        ("truncated", good[: MANIFEST_FIXED - 1], "BCIR_ERR_TRUNCATED"),
        ("magic", _put(good, 0, "<4s", b"BSHX"), "BCIR_ERR_MAGIC"),
        ("version", _put(good, 4, "<H", 1), "BCIR_ERR_VERSION"),
        ("crc", good[:-1] + bytes([good[-1] ^ 1]), "BCIR_ERR_CRC"),
        ("flags", _put(good, 6, "<H", 1), "BCIR_ERR_RESERVED"),
        ("reserved-46", _put(good, 46, "<H", 1), "BCIR_ERR_RESERVED"),
        ("reserved-52", _put(good, 60, "<B", 1), "BCIR_ERR_RESERVED"),
        ("no-shards", _put(good, 8, "<I", 0), "BCIR_ERR_SHARD"),
        ("too-many-shards", _put(good, 8, "<I", 4097), "BCIR_ERR_SHARD"),
        ("short-of-its-shards", _put(good, 8, "<I", 4), "BCIR_ERR_TRUNCATED"),
        ("trailing", _recrc(good[:-4] + bytes(48) + good[-4:]), "BCIR_ERR_TRAILING"),
        ("pack-version", _put(good, 44, "<H", 3), "BCIR_ERR_SHARD"),
        ("no-vector", _put(good, 48, "<I", 0), "BCIR_ERR_SHARD"),
        ("whole-over-bound", _put(good, 16, "<Q", 1 << 32), "BCIR_ERR_SHARD"),
        ("frame-under-minimum", _put(good, 24, "<Q", 10), "BCIR_ERR_SHARD"),
        ("whole-under-frame", _put(good, 16, "<Q", struct.unpack_from("<Q", good, 24)[0] - 1),
         "BCIR_ERR_SHARD"),
        ("segments-but-own-frame", _put(good, 16, "<Q", struct.unpack_from("<Q", good, 24)[0]),
         "BCIR_ERR_SHARD"),
        ("empty-but-bigger", _put(empty, 16, "<Q", struct.unpack_from("<Q", empty, 24)[0] + 1),
         "BCIR_ERR_SHARD"),
        ("shard-under-minimum", _put(good, entry + 8, "<Q", 12), "BCIR_ERR_SHARD"),
        ("shard-over-bound", _put(good, entry + 8, "<Q", 1 << 32), "BCIR_ERR_SHARD"),
        ("first-not-zero", _put(good, entry, "<I", 1), "BCIR_ERR_SHARD"),
        ("gap", _put(good, entry + 48, "<I", 4), "BCIR_ERR_SHARD"),
        ("reversed", _put(good, entry + 4, "<I", 0), "BCIR_ERR_SHARD"),
        ("empty-shard", _put(_put(good, entry + 4, "<I", 0), entry + 48, "<I", 0),
         "BCIR_ERR_SHARD"),
        ("short-of-the-whole", _put(good, 12, "<I", 10), "BCIR_ERR_SHARD"),
    ]  # fmt: skip


def decode_payload(blobs) -> bytes:
    return struct.pack("<I", len(blobs)) + _frames(blobs)


def expected_decode(data: bytes) -> str:
    try:
        m = decode_manifest(data)
    except ShardError as exc:
        return "ERR " + exc.status
    return (
        f"OK {len(m.shards)} {m.n_segments} {m.whole_length} {m.frame_length} {m.map_gen} "
        f"{m.data_gen} {m.topo_gen} {m.n_gens}"
    )


def tampered_sets() -> list[tuple[str, bytes, list[bytes], str]]:
    """(name, manifest, the store's blobs, status): shard sets that must not reassemble."""
    w = synthetic_pack(12)
    sp = split(w, partition(12, 3))
    other = split(synthetic_pack(15), partition(15, 3))
    frame, shards = sp.frame, list(sp.shards)
    twin = split(w, [(0, 4), (4, 12)])
    fake = bytearray(shards[1])
    fake[-5] ^= 1  # a gens byte: the pack's own CRC fails (and its digest)
    # A shard that IS a valid pack of the right segments but not the canonical sub-pack: an extra
    # (unnamed) prefetch record -- then a manifest that names it by its own digest.
    bent = decode(shards[1])
    bent = encode(replace(bent, prefetches=[*bent.prefetches, Prefetch("extra", 1, (1,))]))
    m = decode_manifest(sp.manifest)
    bent_entries = list(m.shards)
    bent_entries[1] = replace(
        bent_entries[1], length=len(bent), sha256=hashlib.sha256(bent).digest()
    )
    from ..abi.shard_manifest import encode_manifest

    bent_manifest = encode_manifest(replace(m, shards=tuple(bent_entries)))
    lie = encode_manifest(replace(m, map_gen=m.map_gen + 1))
    lie_digest = encode_manifest(replace(m, registry_digest=bytes(32)))
    return [
        ("missing-shard", sp.manifest, [frame, shards[0], shards[2]], "BCIR_ERR_SHARD"),
        ("missing-frame", sp.manifest, shards, "BCIR_ERR_SHARD"),
        ("wrong-digest", sp.manifest, [frame, shards[0], bytes(fake), shards[2]], "BCIR_ERR_SHARD"),
        ("another-packs-shard", sp.manifest, [frame, shards[0], other.shards[1], shards[2]],
         "BCIR_ERR_SHARD"),
        ("another-partition", twin.manifest, [frame, *shards], "BCIR_ERR_SHARD"),
        ("non-canonical-shard", bent_manifest, [frame, shards[0], bent, shards[2]],
         "BCIR_ERR_SHARD"),
        ("tags-lie", lie, [frame, *shards], "BCIR_ERR_SHARD"),
        ("registry-lie", lie_digest, [frame, *shards], "BCIR_ERR_SHARD"),
        ("corrupt-manifest", sp.manifest[:-1] + b"\x00", [frame, *shards], "BCIR_ERR_CRC"),
    ]  # fmt: skip


def reassemble_payload(cases) -> bytes:
    """cases: (manifest, blobs, capacity)."""
    out = bytearray(struct.pack("<I", len(cases)))
    for manifest, blobs, cap in cases:
        out += struct.pack("<I", len(manifest)) + manifest + struct.pack("<II", cap, len(blobs))
        out += _frames(blobs)
    return bytes(out)


def expected_reassemble(manifest, blobs) -> str:
    store = {hashlib.sha256(b).digest(): b for b in blobs}
    try:
        whole = reassemble(manifest, store.get)
    except ShardError as exc:
        return "ERR " + exc.status
    return "OK " + hashlib.sha256(whole).hexdigest()


def shard_run_cases() -> list[tuple[list[bytes], bytes, int]]:
    """(plane records, whole, world) for the C++ rail's cut + per-rank re-entry + reassembly: the
    live plane installs each whole's own registry."""
    cases = []
    for w in whole_packs():
        pack = decode(w)
        records = [cf.grant(1, seq=1), cf.generation_record(reg_of(pack), seq=1, expect=0)]
        for world in (1, 3, 8):
            cases.append((records, w, world))
    return cases


def shards_payload(cases) -> bytes:
    out = bytearray(struct.pack("<I", len(cases)))
    for records, whole, world in cases:
        out += struct.pack("<I", len(records)) + _frames(records)
        out += struct.pack("<I", len(whole)) + whole + struct.pack("<I", world)
    return bytes(out)


def reentry_python(whole: bytes, ranges) -> bool:
    """The oracle's per-rank re-entry: each shard admitted by a plane holding the whole's
    registry and dispatched by itself; the ranks' claims concatenate to the whole's."""
    pack = decode(whole)
    sp = split(whole, ranges)
    plane = ControlPlane(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT)
    for record in (cf.grant(1, seq=1), cf.generation_record(reg_of(pack), seq=1, expect=0)):
        plane.submit(record)
    table = PackTable(len(sp.shards) + 1, max(len(s) for s in sp.shards) + 64)
    claims = []
    for shard in sp.shards:
        h = table.reserve().handle
        table.write(h, 0, shard)
        table.commit(h, len(shard))
        if not table.admit(h, plane).applied:
            return False
        out = table.dispatch(h, plane)
        if not out.applied:
            return False
        claims += out.claims
        table.release(h)
    return claims == [s.claim_id for s in pack.segments]


# --- the Stage 3 exit flow ------------------------------------------------------------------------

# The staged plan's Stage 3 exit gate: one artifact generation flows plan -> control -> data ->
# telemetry -> evidence on the loopback with generation handles end to end, and the old generation
# is refused at every boundary. The flow is written once for the oracle (here) and once for both
# native rails (runtime/c/test_stage3.h, which the C harness drives through the pack table and the
# C++ harness through the seam); every rail prints the same lines.

S3_TELEMETRY_SLOT, S3_TELEMETRY_SLOTS, S3_SIGNAL = 128, 2, 1


@dataclass(frozen=True)
class Stage3Case:
    """Two generations of one program, as the flow consumes them: the control records (a lease,
    generation a, generation b, and a record the issuer witnessed against a after the plane
    moved), and per generation its ExecutionPlanV1, its StreamPack and a shard manifest of it."""

    key: bytes
    scope: str
    subject: int
    records: tuple[bytes, bytes, bytes, bytes]  # grant, gen-a, gen-b, stale
    a: tuple[bytes, bytes, bytes]  # plan, pack, manifest
    b: tuple[bytes, bytes, bytes]


_STAGE3: Stage3Case | None = None


def stage3_case() -> Stage3Case:
    """multi_histogram (four claims, so the two-slot telemetry ring really loses records) placed
    for AVX-512; generation b moves one resource's map generation past the maxima."""
    global _STAGE3
    if _STAGE3 is not None:
        return _STAGE3
    from ..abi import encode_plan
    from ..examples import multi_histogram
    from ..gem.execution_plan import plan_from_realization

    module = multi_histogram()
    result = _plan(module)

    def state(m):
        pack = hydrate(m, result)
        plan = encode_plan(plan_from_realization(m, result, _target(), "eft", plan="plan0"))
        whole = encode(pack)
        manifest = split(whole, partition(len(pack.segments), 2)).manifest
        return (plan, whole, manifest), reg_of(pack)

    a, reg_a = state(module)
    rid = sorted(module.resources)[0]
    module.resources[rid] = replace(
        module.resources[rid], map_gen=module.resources[rid].map_gen + 5
    )
    module.touch()
    b, reg_b = state(module)
    records = (
        cf.grant(1, seq=1),
        cf.generation_record(reg_a, seq=1, expect=0),
        cf.generation_record(reg_b, seq=2, expect=1),
        cf.generation_record(reg_a, seq=3, expect=1),  # witnessed generation 1; the plane is at 2
    )
    _STAGE3 = Stage3Case(cf.ROOT_KEY, cf.SCOPE, cf.SUBJECT, records, a, b)
    return _STAGE3


def encode_stage3(case: Stage3Case) -> bytes:
    out = bytearray(b"BST3" + struct.pack("<I", len(case.key)) + case.key)
    out += struct.pack("<BQ", CONTROL_SCOPES.index(case.scope), case.subject)
    for blob in (*case.records, *case.a, *case.b):
        out += struct.pack("<I", len(blob)) + blob
    return bytes(out)


def run_stage3_python(case: Stage3Case) -> list[str]:
    """The flow on the oracle: the lines runtime/c/test_stage3.h prints."""
    from ..abi.telemetry_envelope import TelemetryEnvelope, encode_envelope
    from ..gem.control import REFUSALS as CTL_REFUSALS
    from ..gem.control import VERDICTS as CTL_VERDICTS
    from ..gem.ring import (
        CONTROL_SLOT_MIN,
        RingConsumer,
        RingGeometry,
        RingProducer,
        format_ring,
    )
    from ..gem.ring import VERDICTS as RING_VERDICTS
    from ..telemetry_intake import (
        CLASSIFICATIONS,
        INTAKE_REASONS,
        INTAKE_VERDICTS,
        TelemetryIntake,
    )

    lines: list[str] = []
    plane = ControlPlane(case.key, case.scope, case.subject)
    intake = TelemetryIntake(plane.generation)
    table = PackTable(4, 4096)
    handles: list[Handle | None] = [None, None]
    ends = []
    for g in (
        RingGeometry("backpressure", "control", CONTROL_SLOT_MIN, 4, ring_id=1),
        RingGeometry("overwrite", "telemetry", S3_TELEMETRY_SLOT, S3_TELEMETRY_SLOTS, ring_id=2),
    ):
        region = bytearray(g.region_size)
        format_ring(region, g)
        producer, consumer = RingProducer(region), RingConsumer(region)
        if producer.attach().verdict != "ok" or consumer.attach().verdict != "ok":
            raise RuntimeError("the loopback rings would not attach")
        ends.append((producer, consumer))
    (cp, cc), (tp, tc) = ends
    state = {"seq": 0, "control": 0}

    def control(label: str, record: bytes) -> None:
        w = cp.publish(record)
        r = cc.consume() if w.verdict == "ok" else w
        if w.verdict != "ok" or r.verdict != "delivered" or r.position != w.position:
            lines.append(f"control:{label} ring {RING_VERDICTS.index(r.verdict)} {r.status}")
            return
        state["control"] += 1
        o = plane.submit(r.payload)
        intake.set_live_generation(plane.generation)
        lines.append(
            f"control:{label} {CTL_VERDICTS.index(o.verdict)} {CTL_REFUSALS.index(o.refusal)} "
            f"{o.status} g={o.generation}"
        )

    def plan(label: str, data: bytes) -> None:
        o = plane.admit_plan(data)
        lines.append(
            f"plan:{label} {CTL_VERDICTS.index(o.verdict)} {CTL_REFUSALS.index(o.refusal)} "
            f"{o.status} g={o.generation}"
        )

    def data_line(label: str, o, walked: bool = False) -> None:
        h = f"{o.handle.index}:{o.handle.epoch}" if o.handle is not None else "-"
        c = _claims_digest(o.claims) if walked and o.applied else "-"
        lines.append(
            f"data:{label} {HO_VERDICTS.index(o.verdict)} {HO_REFUSALS.index(o.refusal)} "
            f"{o.status} g={o.generation} h={h} c={c}"
        )

    def emit(generation: int, value: int) -> None:
        state["seq"] += 1
        env = TelemetryEnvelope(
            "sample", source=1, session=1, seq=state["seq"], generation=generation,
            signal=S3_SIGNAL, required=True, value=value,
        )  # fmt: skip
        if tp.publish(encode_envelope(env)).verdict != "ok":
            lines.append("tel:emit refused")

    def drain() -> None:
        while True:
            r = tc.consume()
            if r.verdict == "empty":
                return
            if r.verdict == "lost":
                lines.append(f"tel:lost {r.count}")
                continue
            if r.verdict != "delivered":
                lines.append(f"tel:ring {RING_VERDICTS.index(r.verdict)} {r.status}")
                return
            o = intake.admit(r.payload)
            env = o.envelope
            lines.append(
                f"tel:{env.seq if env else 0} {INTAKE_VERDICTS.index(o.verdict)} "
                f"{INTAKE_REASONS.index(o.reason)} {o.status} g={env.generation if env else 0} "
                f"k={CLASSIFICATIONS.index(o.classification)}"
            )

    def store(label: str, which: int, data: bytes) -> None:
        o = table.reserve()
        if o.applied:
            h = o.handle
            wrote = table.write(h, 0, data)
            o = table.commit(h, len(data)) if wrote.applied else wrote
            if o.applied:
                handles[which] = h
        data_line(label, o)

    def admit_step(label: str, which: int):
        o = table.admit(handles[which], plane)
        data_line(label, o)
        if o.applied:
            emit(o.generation, 0)
            drain()
        return o

    def dispatch(label: str, which: int, each: bool) -> None:
        o = table.dispatch(handles[which], plane)
        data_line(label, o, walked=True)
        if not o.applied:
            return
        for claim in o.claims:
            emit(o.generation, claim)
            if each:
                drain()
        drain()

    def manifest(label: str, data: bytes) -> None:
        data_line(label, admit_manifest(plane, data))

    grant, gen_a, gen_b, stale = case.records
    (plan_a, pack_a, man_a), (plan_b, pack_b, man_b) = case.a, case.b
    control("grant", grant)
    control("gen-a", gen_a)
    plan("a", plan_a)
    store("store-a", 0, pack_a)
    admitted_a = admit_step("admit-a", 0).generation
    manifest("manifest-a", man_a)
    dispatch("dispatch-a", 0, each=False)  # drained once: the two-slot ring loses the rest
    control("gen-b", gen_b)
    control("stale", stale)
    plan("a-stale", plan_a)
    dispatch("dispatch-a-stale", 0, each=True)
    admit_step("admit-a-stale", 0)
    manifest("manifest-a-stale", man_a)
    emit(admitted_a, -1)  # a late sample from a producer that missed the switch
    drain()
    plan("b", plan_b)
    store("store-b", 1, pack_b)
    admit_step("admit-b", 1)
    manifest("manifest-b", man_b)
    dispatch("dispatch-b", 1, each=True)
    data_line("release-a", table.release(handles[0]))
    dispatch("dispatch-a-released", 0, each=True)
    rep, acct = intake.report(), tc.accounting()
    lines.append(
        f"evidence g={plane.generation} accepted={rep.accepted} refused={rep.refused} "
        f"stale={rep.stale} missing={rep.missing} lost={acct.lost} delivered={acct.delivered} "
        f"control={state['control']} p={plane.state_digest().hex()} "
        f"t={table.state_digest().hex()}"
    )
    return lines


# The boundaries at which the old generation is offered after the switch -- each must refuse it.
STAGE3_STALE = (
    "control:stale",
    "plan:a-stale",
    "data:dispatch-a-stale",
    "data:admit-a-stale",
    "data:manifest-a-stale",
)


def stage3_expected(case: Stage3Case) -> list[tuple[str, tuple[str, str]]]:
    """The declared flow: (label, (verdict, reason)) per line, in order. The telemetry lines are
    named by sequence number; `tel:lost` carries its count as the verdict."""
    n_a = len(decode(case.a[1]).segments)
    n_b = len(decode(case.b[1]).segments)
    ok, accepted = ("applied", "none"), ("accepted", "none")
    out = [
        ("control:grant", ok), ("control:gen-a", ok), ("plan:a", ok), ("data:store-a", ok),
        ("data:admit-a", ok), ("tel:1", accepted), ("data:manifest-a", ok),
        ("data:dispatch-a", ok),
    ]  # fmt: skip
    lost = max(0, n_a - S3_TELEMETRY_SLOTS)
    if lost:
        out.append(("tel:lost", (str(lost), "none")))
    out += [(f"tel:{seq}", accepted) for seq in range(2 + lost, 2 + n_a)]
    stale_seq = 2 + n_a
    out += [
        ("control:gen-b", ok),
        ("control:stale", ("refused", "stale")),
        ("plan:a-stale", ("refused", "stale")),
        ("data:dispatch-a-stale", ("refused", "stale")),
        ("data:admit-a-stale", ("refused", "stale")),
        ("data:manifest-a-stale", ("refused", "stale")),
        (f"tel:{stale_seq}", ("refused", "stale")),
        ("plan:b", ok), ("data:store-b", ok), ("data:admit-b", ok), (f"tel:{stale_seq + 1}", accepted),
        ("data:manifest-b", ok), ("data:dispatch-b", ok),
    ]  # fmt: skip
    out += [(f"tel:{stale_seq + 2 + k}", accepted) for k in range(n_b)]
    out += [("data:release-a", ok), ("data:dispatch-a-released", ("refused", "lifetime"))]
    return out


def _stage3_decision(label: str, fields: list[str]) -> tuple[str, str]:
    """(verdict, reason) of one trace line, by its boundary's own names."""
    from ..gem.control import REFUSALS as CTL_REFUSALS
    from ..gem.control import VERDICTS as CTL_VERDICTS
    from ..telemetry_intake import INTAKE_REASONS, INTAKE_VERDICTS

    if label == "tel:lost":
        return (fields[0], "none")
    if not fields or fields[0] in ("ring", "refused") or len(fields) < 2:
        return ("broken", "broken")
    table = {
        "control": (CTL_VERDICTS, CTL_REFUSALS),
        "plan": (CTL_VERDICTS, CTL_REFUSALS),
        "data": (HO_VERDICTS, HO_REFUSALS),
        "tel": (INTAKE_VERDICTS, INTAKE_REASONS),
    }.get(label.split(":", 1)[0])
    try:
        verdicts, reasons = table
        return (verdicts[int(fields[0])], reasons[int(fields[1])])
    except (TypeError, ValueError, IndexError):
        return ("broken", "broken")


def grade_stage3(case: Stage3Case, lines: list[str]) -> tuple[int, int]:
    """(stale boundaries that did not refuse, other declared outcomes not as declared). A missing,
    repeated or unexpected line is a miss; the evidence line must reconcile: the ring's loss is
    the intake's gap count, exactly one stale record, every delivered record decided, every
    control record carried, the plane at generation 2, and the accepted count the declared one."""
    expected = stage3_expected(case)
    stale_labels = {*STAGE3_STALE, *(label for label, want in expected
                                     if label.startswith("tel:") and want == ("refused", "stale"))}  # fmt: skip
    seen: dict[str, list[str]] = {}
    evidence = None
    for line in lines:
        head, _, rest = line.partition(" ")
        if head == "evidence":
            evidence = evidence or rest
            continue
        seen.setdefault(head, []).append(rest)
    stale = flow = 0
    declared = {label for label, _ in expected}
    for label, want in expected:
        got = seen.get(label, [])
        wrong = len(got) != 1 or _stage3_decision(label, got[0].split()) != want
        if label in stale_labels:
            stale += wrong
        else:
            flow += wrong
    flow += sum(len(v) for k, v in seen.items() if k not in declared)
    fields = dict(part.split("=", 1) for part in (evidence or "").split() if "=" in part)
    try:
        n = {k: int(fields[k]) for k in ("g", "accepted", "refused", "stale", "missing", "lost",
                                         "delivered", "control")}  # fmt: skip
    except (KeyError, ValueError):
        return stale, flow + 1
    n_a = len(decode(case.a[1]).segments)
    n_b = len(decode(case.b[1]).segments)
    laws = (
        n["lost"] == n["missing"] == max(0, n_a - S3_TELEMETRY_SLOTS),
        n["stale"] == 1,
        n["accepted"] + n["refused"] == n["delivered"],
        n["control"] == len(case.records),
        n["g"] == 2,
        n["accepted"] == 1 + min(n_a, S3_TELEMETRY_SLOTS) + 1 + n_b,
    )
    return stale, flow + sum(not law for law in laws)


# --- the rows ------------------------------------------------------------------------------------


def _grade(scenario: Scenario, outcomes: list[tuple[str, str]]) -> dict[str, int]:
    """Per-row misses of one rail over one scenario. Every step that declares an outcome is graded
    -- a named row's steps in that row, the rest as decisions -- and a short trace fails every
    step it does not reach."""
    miss = dict.fromkeys(
        ("stale-admit", "stale-dispatch", "fresh", "lifetime", "freeze", "decision"), 0
    )
    for i, step in enumerate(scenario.steps):
        if step.expect is None:
            continue
        got = outcomes[i] if i < len(outcomes) else ("missing", "missing")
        wrong = got != step.expect
        if step.grade == "stale-admit":
            miss["stale-admit"] += wrong
        elif step.grade == "stale-dispatch":
            miss["stale-dispatch"] += wrong
        elif step.grade in ("fresh-admit", "fresh-dispatch"):
            miss["fresh"] += wrong
        elif step.grade == "lifetime":
            miss["lifetime"] += wrong
        elif step.grade == "freeze":
            miss["freeze"] += wrong
        else:
            miss["decision"] += wrong
    return miss


_MISS_ROWS = {
    "stale-admit": "handoff.stale.admitted",
    "stale-dispatch": "handoff.stale.dispatched",
    "fresh": "handoff.fresh.refused",
    "lifetime": "handoff.lifetime.unrefused",
    "freeze": "handoff.builder.violations",
    "decision": "handoff.decisions.nonconforming",
}


def _lines(run, count: int) -> list[str]:
    """A native mode's lines, padded so a rail that failed or stopped early fails what it was
    handed (L1) -- never a traceback, never a shorter comparison."""
    try:
        got = run()
    except RuntimeError:
        got = []
    return got + [""] * (count - len(got))


def _oracle(fn, *args) -> str:
    """An oracle verdict line, or a line no rail can match when the oracle itself raised."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 -- a rail that raises decided nothing (L1)
        return f"RAISED {type(exc).__name__}"


_SCENARIO_ROWS = (*_MISS_ROWS.values(), "handoff.traces.divergent")


def _built(build, rows, out: dict[str, float]) -> list:
    """A corpus, or none with every row it feeds failed. The corpora are the oracle's own splits,
    freezes and plans, so a defect in the oracle can first surface as a corpus that cannot be
    built -- a finding in a named row, never a traceback (L1)."""
    try:
        return build()
    except Exception:  # noqa: BLE001 -- an oracle that cannot build its corpus decided nothing
        for row in rows:
            out[row] += 1
        return []


def measure(exe: str | None, cpp_exe: str | None, tmp: str) -> dict[str, float]:
    """The G16 rows over the declared corpora, on every rail named (`exe`: the C harness, `cpp_exe`:
    the C++ harness; None leaves that rail out -- the caller grades an absent rail -- while a path
    that cannot run fails every fixture it was handed).

    handoff.stale.admitted        (stale admission step, rail) pairs not refused as stale -- at the
                                  table's admit and at the plane's manifest gate
    handoff.stale.dispatched      (dispatch the specification refuses, rail) pairs not refused as
                                  declared: no admission, a stale admission, a draining plane
    handoff.fresh.refused         (fresh admission or dispatch, rail) pairs not applied
    handoff.lifetime.unrefused    (access through a dead handle or view, rail) pairs not refused
                                  BCIR_ERR_LIFETIME, plus C++ lifetime witnesses that failed
    handoff.copies                segment views the C++ seam's dispatches read outside the slot the
                                  view names, plus the moved-once witness
    handoff.builder.violations    (freeze step, rail) pairs not deciding as declared, plus freeze
                                  cases whose C bytes (or refusal status) differ from the oracle's
    handoff.decisions.nonconforming  every other declared (step, rail) outcome not as declared --
                                  the table's capacity, fullness and bounds, a malformed admission
    handoff.shards.mismatches     split cases whose manifest, frame or shards differ between the
                                  oracle and the C twin or the C++ seam's cut; wholes a rail does not
                                  reassemble byte for byte; splits a rail does not refuse as declared
    handoff.shards.malformed.accepted  (malformed manifest or tampered set, rail) pairs not refused
                                  with the declared status
    handoff.reentry.divergent     (whole, world, rail) cases whose shards, each admitted and run by
                                  itself, do not reproduce the whole's dispatch
    handoff.traces.divergent      scenarios whose native trace differs from the oracle's in any line
    handoff.stage3.stale.accepted (boundary, rail) pairs at which the Stage 3 exit flow accepted the
                                  old generation after the switch: a control record witnessed
                                  against it, its plan, its pack's dispatch and admission, its
                                  manifest, a late telemetry sample bound to it
    handoff.stage3.flow.divergent (declared step, rail) pairs of the flow not as declared, evidence
                                  laws that do not reconcile, and native flows that differ from the
                                  oracle's in any line"""
    out = {row: 0.0 for row in ROWS}
    scenarios = _built(all_scenarios, _SCENARIO_ROWS, out)
    oracle_lines: list[list[str] | None] = []
    for index, scenario in enumerate(scenarios):
        try:
            outcomes, lines = run_python(scenario, index)
            got = [(o.verdict, o.refusal) for o in outcomes]
        except Exception:  # noqa: BLE001 -- an oracle that raises fails every step (L1)
            lines, got = None, []
        oracle_lines.append(lines)
        for key, value in _grade(scenario, got).items():
            out[_MISS_ROWS[key]] += value
    for name, binary in (("c", exe), ("cpp", cpp_exe)):
        if binary is None:
            continue
        chosen = [s for s in scenarios if name == "c" or not s.c_only]
        try:
            traces, copies = run_native(binary, tmp, chosen)
        except RuntimeError:
            traces, copies = [[] for _ in chosen], None
        if name == "cpp":
            out["handoff.copies"] += copies if copies is not None else len(chosen)
        by_name = dict(zip((s.name for s in chosen), traces))
        for index, scenario in enumerate(scenarios):
            if scenario.name not in by_name:
                continue
            lines = by_name[scenario.name]
            for key, value in _grade(scenario, outcomes_of(lines)).items():
                out[_MISS_ROWS[key]] += value
            oracle = oracle_lines[index]
            out["handoff.traces.divergent"] += oracle is None or _body(lines) != _body(oracle)

    # the freeze corpus: bytes (or refusal) identical to the oracle's
    corpus = _built(freeze_corpus, ("handoff.builder.violations",), out)
    expected = [_oracle(expected_freeze, *case) for case in corpus]
    out["handoff.builder.violations"] += sum(e.startswith("RAISED") for e in expected)
    if exe is not None:
        got = _lines(lambda: run_mode(exe, tmp, "freeze", freeze_payload(corpus)), len(expected))
        out["handoff.builder.violations"] += sum(a != b for a, b in zip(expected, got))

    # the shards: split and reassembly on every rail, refused splits, the hostile corpora
    cases = _built(split_cases, ("handoff.shards.mismatches", "handoff.reentry.divergent"), out)
    refused = _built(refused_splits, ("handoff.shards.mismatches",), out)
    malformed = _built(malformed_manifests, ("handoff.shards.malformed.accepted",), out)
    tampered = _built(tampered_sets, ("handoff.shards.malformed.accepted",), out)
    want = [_oracle(expected_split, w, r) for w, r in cases]
    out["handoff.shards.mismatches"] += sum(not line.startswith("OK ") for line in want)
    for w, ranges in cases:
        line = _oracle(lambda w=w, r=ranges: "OK" if reassemble(
            split(w, r).manifest, split(w, r).store().get) == w else "NO")  # fmt: skip
        out["handoff.shards.mismatches"] += line != "OK"
    refused_want = [_oracle(expected_split, w, r) for _, w, r in refused]
    out["handoff.shards.mismatches"] += sum(line != "ERR BCIR_ERR_SHARD" for line in refused_want)
    for _name, data, status in malformed:
        out["handoff.shards.malformed.accepted"] += (
            _oracle(expected_decode, data) != "ERR " + status
        )
    for _name, manifest, blobs, status in tampered:
        out["handoff.shards.malformed.accepted"] += (
            _oracle(expected_reassemble, manifest, blobs) != "ERR " + status
        )
    for w, ranges in cases:
        out["handoff.reentry.divergent"] += _oracle(reentry_python, w, ranges) is not True
    if exe is not None:
        payload = split_payload(cases + [(w, r) for _, w, r in refused])
        want_all = [
            *[_oracle(lambda w=w, r=r: "OK " + sp_hex(w, r)) for w, r in cases],
            *["ERR BCIR_ERR_SHARD"] * len(refused),
        ]
        got = _lines(lambda: run_mode(exe, tmp, "split", payload), len(want_all))
        out["handoff.shards.mismatches"] += sum(a != b for a, b in zip(want_all, got))
        honest = []
        for w, r in cases:
            try:
                honest.append((*_blobs(w, r), len(w) + 64))
            except Exception:  # noqa: BLE001 -- no oracle split: nothing to reassemble, the case fails
                honest.append((b"", [], 0))
        got = _lines(
            lambda: run_mode(exe, tmp, "reassemble", reassemble_payload(honest)), len(cases)
        )
        wants = ["OK " + hashlib.sha256(w).hexdigest() for w, _ in cases]
        out["handoff.shards.mismatches"] += sum(a != b for a, b in zip(wants, got))
        got = _lines(
            lambda: run_mode(exe, tmp, "decode", decode_payload([d for _, d, _ in malformed])),
            len(malformed),
        )
        for (_name, _data, status), line in zip(malformed, got):
            out["handoff.shards.malformed.accepted"] += line != "ERR " + status
        hostile = reassemble_payload([(m, b, 1 << 16) for _, m, b, _ in tampered])
        got = _lines(lambda: run_mode(exe, tmp, "reassemble", hostile), len(tampered))
        for (_name, _m, _b, status), line in zip(tampered, got):
            out["handoff.shards.malformed.accepted"] += line != "ERR " + status
    if cpp_exe is not None:
        runs = _built(
            shard_run_cases,
            ("handoff.shards.mismatches", "handoff.reentry.divergent", "handoff.copies"),
            out,
        )
        got = _lines(lambda: run_mode(cpp_exe, tmp, "shards", shards_payload(runs)), len(runs))
        for (_records, w, world), line in zip(runs, got):
            try:
                ranges = partition(len(decode(w).segments), world)
                claims = _claims_digest(s.claim_id for s in decode(w).segments)
            except Exception:  # noqa: BLE001 -- no oracle partition: the case fails (L1)
                out["handoff.shards.mismatches"] += 1
                out["handoff.reentry.divergent"] += 1
                continue
            head, _, tail = line.partition(" | ")
            out["handoff.shards.mismatches"] += head != _oracle(
                lambda w=w, r=ranges: "OK " + sp_hex(w, r)
            )
            fields = dict(part.split("=", 1) for part in tail.split() if "=" in part)
            out["handoff.shards.mismatches"] += fields.get("a") != hashlib.sha256(w).hexdigest()
            out["handoff.reentry.divergent"] += (
                fields.get("admit") != "BCIR_OK"
                or fields.get("dispatch") != "BCIR_OK"
                or fields.get("ranks") != "ok"
                or fields.get("w") != claims
                or fields.get("r") != claims
            )
            out["handoff.copies"] += int(fields["x"]) if fields.get("x", "").isdigit() else 1
        try:
            lines = _run([cpp_exe, "--lifetime"]).stdout.splitlines()
        except RuntimeError:
            lines = []
        witnessed = {line.split()[0]: line.split()[1] for line in lines if len(line.split()) == 2}
        for name in CPP_LIFETIME_WITNESSES:
            if witnessed.get(name) != "ok":
                row = (
                    "handoff.copies" if name == "bytes-moved-once" else "handoff.lifetime.unrefused"
                )
                out[row] += 1

    # the Stage 3 exit flow: one generation through every boundary on every rail, then the next
    stage3_rows = ("handoff.stage3.stale.accepted", "handoff.stage3.flow.divergent")
    for case in _built(lambda: [stage3_case()], stage3_rows, out):
        try:
            want = run_stage3_python(case)
        except Exception:  # noqa: BLE001 -- an oracle that raises decided nothing (L1)
            want = []
        flows = [("oracle", want)]
        payload = encode_stage3(case)
        for name, binary in (("c", exe), ("cpp", cpp_exe)):
            if binary is not None:
                run = partial(run_mode, binary, tmp, "stage3", payload)
                flows.append((name, _lines(run, 0)))
        for name, lines in flows:
            try:
                stale, flow = grade_stage3(case, lines)
            except Exception:  # noqa: BLE001 -- an ungradable flow fails both rows (L1)
                stale, flow = 1, 1
            out["handoff.stage3.stale.accepted"] += stale
            out["handoff.stage3.flow.divergent"] += flow + (name != "oracle" and lines != want)
    return out


def sp_hex(whole: bytes, ranges) -> str:
    """The oracle's split as the harnesses print it: the manifest `split` binds, then the frame and
    the shards through the public `frame_of` and `sub_pack` -- the twins of the C harness's
    `bcir_shm_frame` and `bcir_shm_sub_pack` -- so every entry point is on a measured path."""
    sp = split(whole, ranges)
    parts = [frame_of(whole), *(sub_pack(whole, b, e) for b, e in ranges)]
    return " ".join([sp.manifest.hex(), *(part.hex() for part in parts)])


def _blobs(whole: bytes, ranges) -> tuple[bytes, list[bytes]]:
    sp = split(whole, ranges)
    return sp.manifest, [sp.frame, *sp.shards]


def measure_overhead(cpp_exe: str, tmp: str) -> dict[str, float]:
    """`handoff.dispatch.overhead`: the C++ seam's dispatch of an admitted pack over the direct C
    walk of the same bytes, the same callback (`test_handoff_cpp --bench`, median of 7 rounds of
    2,000) on the 120-segment synthetic pack; {} when the bench cannot run (NOT-MEASURED)."""
    pack = synthetic_pack(120)
    records = [cf.grant(1, seq=1), cf.generation_record(reg_of(decode(pack)), seq=1, expect=0)]
    pack_path, plane_path = os.path.join(tmp, "bench.bin"), os.path.join(tmp, "bench.plane")
    with open(pack_path, "wb") as fh:
        fh.write(pack)
    with open(plane_path, "wb") as fh:
        fh.write(struct.pack("<I", len(records)) + _frames(records))
    try:
        run = _run([cpp_exe, "--bench", pack_path, plane_path])
    except RuntimeError:
        return {}
    for line in run.stdout.splitlines():
        if line.startswith("BENCH ratio="):
            return {"handoff.dispatch.overhead": float(line.split()[1].split("=", 1)[1])}
    return {}


CPP_LIFETIME_WITNESSES = (
    "view-after-release",
    "view-after-owner-scope",
    "view-after-arena",
    "reused-slot-old-view",
    "borrow-outlives-release",
    "reservation-aborts",
    "moved-owner",
    "double-give-back",
    "foreign-arena",
    "bytes-moved-once",
)


__all__ = [
    "C_UNITS",
    "CPP_UNITS",
    "FAMILIES",
    "ROWS",
    "Scenario",
    "Step",
    "all_scenarios",
    "build_cpp_harness",
    "build_harness",
    "measure",
    "run_python",
]
