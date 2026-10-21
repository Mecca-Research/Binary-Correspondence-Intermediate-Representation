"""PX-2b (PACK): the native training step as a BCIR program -- a StreamPack and its ExecutionPlanV1.

PX-2a runs the launch program `native_launch.step_program` states, as words. Here the same program
is stated as BCIR's own artifacts, and reaches the trainer through them alone:

* `training_pack` states it as a `Module`: ONE CLAIM PER KERNEL CALL, one phase per launch (a GEMM
  job's calls share theirs), each phase after the one before -- so GEM's dispatch order (topological
  phase, then ascending claim id) is the program's record order, and a phase's position is its
  launch. The resources are the program's VALUES. Each parameter tensor of the four parameter
  arenas, the bf16 workspace and the micro-batch's tokens and targets live through the whole
  program; a write of a span of the activations or the scratch starts a new value and ends every
  value of a span it meets, and a read takes the span's current value (a read of a span no live
  value holds is refused here: the program would read bytes another value left). The C step's own
  reuse of its buffers -- the scratch every layer and micro-batch, the RMS VJP's 1/rms in the head of
  dq, the loss's row terms in the head of cur, the slot every recomputed layer shares -- is then many
  values on the same bytes at disjoint times: what the plan's alias law judges.
* `pack_artifacts` plans the module with K_BCIR, hydrates it with GEM into a StreamPack, and mints
  its ExecutionPlanV1, whose lifetimes place every value: the bank it lives in (its arena; the
  workspace; the tokens or the targets), its byte offset and size there, and the phases it is live
  over -- the C step's own spans (`TrainingPack.lifetimes`) unless the caller hands another placement.
* PX-4 (PLACE): `place_by_lifetime` is that other placement -- every value of the two transient arenas
  laid out by its lifetime in one bank (GEM+'s static memory planner, `kbcir.static_memory`): the
  values split where none is live across a phase boundary, each part laid out by the bounded exact
  search from GEM+'s incumbent, the parts sharing addresses (no two are live at once). Its extent is
  the transient arena a trainer then lends (`NativeDecoder(transient=...)`), held against G26's
  alignment-aware concurrent-live bound over the same values: equal, the layout is a proved optimum
  (TMSAO-1), and `train.memory.transient` reads their ratio.

`bcir_decoder_lower_pack` (runtime/c/bcir_decoder_pack.c) takes the two artifacts and the state and
nothing else: the plan's and the pack's own verifiers (the plan's alias law among them: no two
lifetimes of one bank live at once on overlapping bytes), then each claim GEM's executor dispatches
lowered to the PX-2a record that runs it -- every dimension and flag derived from the state and the
operands' sizes, every operand's place from its value's lifetime, which must cover the claim's phase
-- so `bcir_decoder_run_program` checks and runs what GEM dispatched. `train.plan.pack.mismatch`
holds the lowering of every trace-corpus configuration's pack to `program_words(step_program(...))`.

A claim (the vocabulary both rails share, `OPS` and the GEMM grammar):

    train.gemm.<role>.<orient>[.acc][.pa|.pb]  rd (A, B[, C])  wr (C[, workspace])
        role: alone, pair (a linear's two VJPs, NDT-SCHED-0), chain or rest (a group, NDT-SCHED-1);
        orient: nt (y = x W^T), nn (dx = dy W), tn (dW = dy^T x), the token rows on the row side;
        acc: C accumulates (beta 1); pa / pb: A / B a bfloat16 panel (BF16-PANELS)
    train.scan | train.zero                    rd | wr: one value, or every tensor of a parameter
                                               arena in its order
    train.embed / train.scatter                rd (tokens, embed) wr (x) / rd (tokens, dx, dE) wr (dE)
    train.loss                                 rd (logits, targets) wr (dlogits, the row terms)
    train.rms, train.rms.vjp[.add][.first]     the norm, and its VJP (how: dx accumulates; dgamma
                                               written first)
    train.rope[.inverse], train.attention[.vjp], train.add, train.mul, train.provider.silu,
    train.silu_mul[.panels], train.swiglu.vjp[.panels], train.check.forward|ran|loss, train.update

Each claim's rd lists the RIDs of its read operands and wr of its written ones, each in operand
order (an in-place operand in both). Its block: `base` the micro-batch of an embed, a loss or a
scatter (else 0), `count` the call's work -- what K_BCIR prices (`work`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...frontends.models.decode import DecoderSpec
from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from . import native_launch as nl
from .native_program import ARENAS, TRANSIENT, training_step_program

#: The claims' names other than the GEMM's, in the C's table order (bcir_decoder_pack.c).
OPS = (
    "train.scan", "train.zero", "train.embed", "train.rms", "train.rope", "train.rope.inverse",
    "train.attention", "train.attention.vjp", "train.add", "train.mul", "train.provider.silu",
    "train.silu_mul", "train.silu_mul.panels", "train.swiglu.vjp", "train.swiglu.vjp.panels",
    "train.rms.vjp", "train.rms.vjp.add", "train.rms.vjp.first", "train.rms.vjp.add.first",
    "train.loss", "train.scatter", "train.check.forward", "train.check.ran", "train.check.loss",
    "train.update",
)  # fmt: skip
#: A GEMM claim's roles and orientations (its name's second and third parts).
GEMM_ROLES = ("alone", "pair", "chain", "rest")
GEMM_ORIENTS = {(0, 1): "nt", (0, 0): "nn", (1, 0): "tn"}
#: The banks a lifetime names, and the operand space each one is (`native_launch.SPACES`).
BANKS = {name: i + 1 for i, name in enumerate(ARENAS)} | {"workspace": 7, "tokens": 8, "targets": 8}
#: An operand's use: read, written, or both (its kernel's in-place operand).
R, W, RW = 1, 2, 3
#: Why `bcir_decoder_lower_pack` refused a pack: the StreamPack and plan verifiers' bcir_status
#: (bcir_runtime.h, 1-25), then the lowering's own reasons (bcir_decoder_pack.h's BCIR_DP_*).
REASONS = {
    1: "truncated", 2: "magic", 3: "version", 4: "crc", 5: "nospace", 6: "lane", 7: "width",
    8: "dispatch", 9: "provenance", 10: "stale", 11: "overflow", 12: "trailing", 13: "reserved",
    14: "utf8", 15: "generation", 16: "plan", 17: "control", 18: "mac", 19: "telemetry",
    20: "ring", 21: "full", 22: "busy", 23: "lifetime", 24: "shard", 25: "planner",
    101: "op", 102: "claims", 103: "rid", 104: "bank", 105: "live", 106: "slots", 107: "run",
    108: "shape", 109: "block", 110: "liveness",
}  # fmt: skip

_SIMPLE = {
    nl.SCAN: "train.scan", nl.ZERO: "train.zero", nl.EMBED: "train.embed", nl.RMS: "train.rms",
    nl.ATTN: "train.attention", nl.ATTN_VJP: "train.attention.vjp", nl.ADD: "train.add",
    nl.MUL: "train.mul", nl.PROVIDER_SILU: "train.provider.silu", nl.SILU_MUL: "train.silu_mul",
    nl.SILU_MUL_PANELS: "train.silu_mul.panels", nl.SWIGLU_VJP: "train.swiglu.vjp",
    nl.SWIGLU_VJP_PANELS: "train.swiglu.vjp.panels", nl.LOSS: "train.loss",
    nl.SCATTER: "train.scatter", nl.UPDATE: "train.update",
}  # fmt: skip
_CHECKS = {
    nl.CHECK_FORWARD: "train.check.forward",
    nl.CHECK_RAN: "train.check.ran",
    nl.CHECK_LOSS: "train.check.loss",
}
_RMSB_ADD, _RMSB_FIRST, _ATTN_BF16 = 1, 2, 1 << 8


def op_name(r: nl.Record) -> str:
    """The claim name of record `r` (module docstring)."""
    k, f, d = r.kind, r.flags, r.dims
    if k == nl.GEMM:
        entry, role = f & 15, f >> 4 & 15
        name = (
            "pair" if entry == nl.E_PAIR
            else ("chain" if role == 1 else "rest") if entry in (nl.E_GROUP, nl.E_BF16_GROUP)
            else "alone"
        )  # fmt: skip
        pa, pb = f >> 24 & 1, f >> 25 & 1
        if pa and pb:
            raise ValueError("a GEMM call reads at most one of its operands as a panel")
        return (f"train.gemm.{name}.{GEMM_ORIENTS[d[0], d[1]]}" + (".acc" if d[6] else "")
                + (".pa" if pa else ".pb" if pb else ""))  # fmt: skip
    if k == nl.ROPE:
        return "train.rope.inverse" if d[4] else "train.rope"
    if k == nl.RMS_VJP:
        return ("train.rms.vjp" + (".add" if f & _RMSB_ADD else "")
                + (".first" if f & _RMSB_FIRST else ""))  # fmt: skip
    if k == nl.CHECK:
        return _CHECKS[f]
    return _SIMPLE[k]


def uses(r: nl.Record) -> tuple[tuple[int, int], ...]:
    """(operand slot, use) for every operand of `r`, in slot order: what its kernel reads (R),
    writes (W) or updates in place (RW) -- the C check's own reading (bcir_decoder_train.c's
    px_kernel, px_gemm and px_update); the bf16 workspace a call packs into is written."""
    k, f = r.kind, r.flags
    if k == nl.GEMM:
        ws = ((3, W),) if f & 15 in (nl.E_BF16, nl.E_BF16_GROUP) else ()
        return ((0, R), (1, R), (2, RW if r.dims[6] else W)) + ws
    if k == nl.ATTN:
        return ((0, R), (1, R), (2, R), (3, W), (4, W)) + (((8, W),) if f & _ATTN_BF16 else ())
    if k == nl.ATTN_VJP:
        bf = bool(f & _ATTN_BF16)
        return ((0, R), (1, R), (2, R), (3, R if bf else RW), (4, R), (5, W), (6, W), (7, W)) + (
            ((8, W),) if bf else ()
        )
    if k == nl.RMS_VJP:
        return ((0, R), (1, R), (2, R), (3, RW if f & _RMSB_ADD else W),
                (4, W if f & _RMSB_FIRST else RW), (5, W))  # fmt: skip
    table = {
        nl.SCAN: ((0, R),), nl.ZERO: ((0, W),), nl.EMBED: ((0, R), (1, R), (2, W)),
        nl.RMS: ((0, R), (1, R), (2, W)), nl.ROPE: ((0, RW),), nl.ADD: ((0, R), (1, RW)),
        nl.MUL: ((0, R), (1, RW)), nl.PROVIDER_SILU: ((0, R), (1, W)),
        nl.SILU_MUL: ((0, R), (1, R), (2, W)), nl.SILU_MUL_PANELS: ((0, R), (1, R), (2, W), (3, W)),
        nl.SWIGLU_VJP: ((0, R), (1, R), (2, R), (3, W), (4, W)),
        nl.SWIGLU_VJP_PANELS: ((0, R), (1, R), (2, R), (3, W), (4, W), (5, W), (6, W)),
        nl.LOSS: ((0, R), (1, R), (2, W), (3, W)), nl.SCATTER: ((0, R), (1, R), (2, RW)),
        nl.CHECK: (), nl.UPDATE: ((0, R), (1, RW), (2, RW), (3, RW)),
    }  # fmt: skip
    if k not in table:
        raise ValueError(f"a program holds no record of kind {k}")
    return table[k]


def work(r: nl.Record) -> int:
    """A call's work, the claim's `count` (K_BCIR prices it; the C holds the claim to it): a
    GEMM's multiply-adds, the causal attention's (two contractions forward, four back), a
    rows x width record's elements, the passes' element counts, the update's parameters, and 1
    for a check."""
    k, d = r.kind, r.dims
    if k == nl.GEMM:
        return d[2] * d[3] * d[4]
    if k in (nl.ATTN, nl.ATTN_VJP):
        b, t, h, _kh, hd = d[:5]
        return (2 if k == nl.ATTN else 4) * b * h * hd * (t * (t + 1) // 2)
    if k == nl.ROPE:
        return d[0] * d[1] * d[2] * d[3]
    if k in (nl.EMBED, nl.SCATTER, nl.RMS, nl.RMS_VJP, nl.LOSS, nl.SILU_MUL_PANELS,
             nl.SWIGLU_VJP_PANELS):  # fmt: skip
        return d[0] * d[1]
    if k == nl.CHECK:
        return 1
    return d[0]  # the passes' counts and the update's parameters


def _form(k: int) -> tuple[Opcode, Lane, StrideClass, str, str]:
    """The claim's opcode, lane, stride class, cost class and hazard for the R-laws and K_BCIR."""
    if k in (nl.GEMM, nl.ATTN, nl.ATTN_VJP):
        return Opcode.T_MACC, Lane.T, StrideClass.TILE, "compute", "unique"
    if k == nl.EMBED:
        return Opcode.GGG_LOAD, Lane.GGG, StrideClass.RANDOM, "bandwidth", "unique"
    if k == nl.SCATTER:
        return Opcode.GGG_STORE, Lane.GGG, StrideClass.RANDOM, "bandwidth", "unique"
    if k == nl.CHECK:
        return Opcode.BARRIER, Lane.H, StrideClass.SCALAR, "latency", "barriered"
    if k == nl.SCAN:
        return Opcode.LOAD, Lane.U, StrideClass.UNIT, "bandwidth", "unique"
    if k == nl.ZERO:
        return Opcode.STORE, Lane.U, StrideClass.UNIT, "bandwidth", "unique"
    return Opcode.MUL, Lane.U, StrideClass.UNIT, "bandwidth", "unique"


@dataclass(frozen=True)
class Placement:
    """One value's place: the bank it lives in, its byte offset and size there, and the first
    and last phase (positions) it is live over."""

    rid: int
    bank: str
    offset: int
    size: int
    first: int
    last: int


@dataclass(frozen=True)
class TrainingPack:
    """A training program as a BCIR module: the launches it states (`step_program`'s), their
    words (what the C's lowering of the pack must equal), the module, every value's place in the
    state the C step itself uses, and each record's operand slots with the values bound to them
    (the record's claim reads and writes them, `placed_words`)."""

    spec: DecoderSpec
    batch: int
    time: int
    program: tuple[nl.Launch, ...]
    words: tuple[int, ...]
    module: Module
    placements: tuple[Placement, ...]
    tensors: int  # parameter tensors per arena: RIDs 1 .. 4 * tensors are theirs
    bindings: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...] = ()  # per record: (slot, rids)

    @property
    def claims(self) -> int:
        return sum(len(launch.records) for launch in self.program)

    @property
    def lifetimes(self):
        """The placements as ExecutionPlanV1 lifetimes (alignment 4: every operand is floats,
        bfloat16 pairs or row terms at a float boundary)."""
        from ...gem.execution_plan import Lifetime

        return [Lifetime(p.rid, p.bank, p.offset, p.size, 4, p.first, p.last)
                for p in self.placements]  # fmt: skip


def training_pack(spec: DecoderSpec, batch: int, time: int, **decisions) -> TrainingPack:
    """The program `step_program(spec, batch, time, **decisions)` states, as a module of its
    values and calls (module docstring)."""
    program = nl.step_program(spec, batch, time, **decisions)
    return program_pack(spec, batch, time, program, recompute=decisions.get("recompute", 0))


def program_pack(
    spec: DecoderSpec, batch: int, time: int, program, *, recompute: int = 0
) -> TrainingPack:
    """`program` -- launches over the step's buffers under the recompute set `recompute`, as
    `step_program` states them -- as a module of its values and calls (module docstring). A read of
    a span no live value holds is refused (ValueError)."""
    program = tuple(program)
    lay = nl._layout(training_step_program(spec, batch, time, recompute))
    # the parameter tensors, per arena in its order; their RIDs come first
    sizes: dict[int, list[int]] = {}
    tensor_at: dict[tuple[int, int], int] = {}
    containing: list[tuple[int, int, int, str]] = []  # transient buffers: (space, off, end, name)
    for name, (space, off, size) in lay.items():
        if space <= 4:
            tensor_at[space, off] = len(sizes.setdefault(space, []))
            sizes[space].append(size)
        else:
            containing.append((space, off, off + size, name))
    tensors = len(sizes[1])
    workspace_rid, tokens_rid, targets_rid = 4 * tensors + 1, 4 * tensors + 2, 4 * tensors + 3
    n = batch * time
    resources: dict[int, Resource] = {}
    places: dict[int, list] = {}  # rid -> [bank, offset, size, first, last]

    def declare(rid, bank, offset, size, elem, name, at):
        resources[rid] = Resource(rid=rid, elem_bytes=elem, shape=(size // elem,), align=4,
                                  name=name)  # fmt: skip
        places[rid] = [bank, offset, size, at, at]

    def touch(rid, at):
        place = places[rid]
        place[3], place[4] = min(place[3], at), max(place[4], at)

    def parameters(space, offset, nbytes, at) -> list[int]:
        i = tensor_at.get((space, offset))
        if i is None:
            raise ValueError(f"an operand starts inside a {ARENAS[space - 1]} tensor")
        rids, left = [], nbytes
        while left > 0 and i < tensors:
            rid = 1 + (space - 1) * tensors + i
            if rid not in resources:
                start = offset + nbytes - left
                declare(rid, ARENAS[space - 1], start, sizes[space][i], 4,
                        f"{ARENAS[space - 1]}:{i}", at)  # fmt: skip
            rids.append(rid)
            left -= sizes[space][i]
            i += 1
        if left:
            raise ValueError(f"an operand ends inside a {ARENAS[space - 1]} tensor")
        return rids

    current: dict[tuple[int, int, int], int] = {}  # a span -> its live value
    versions: dict[str, int] = {}
    next_rid = 4 * tensors + 4

    def end_meeting(space, offset, nbytes, keep=None) -> None:
        for key in [k for k in current if k[0] == space and k != keep]:
            if key[1] < offset + nbytes and offset < key[1] + key[2]:
                del current[key]

    def value_name(space, offset, nbytes) -> str:
        for sp, lo, hi, name in containing:
            if sp == space and lo <= offset < hi:
                part = "" if (offset, nbytes) == (lo, hi - lo) else f"[{offset - lo}:+{nbytes}]"
                base = name + part
                versions[base] = versions.get(base, 0) + 1
                return f"{base}#{versions[base]}"
        raise ValueError(f"an operand lies outside every {ARENAS[space - 1]} buffer")

    module = Module(name=f"train_{spec.n_layers}l_{spec.d_model}d_{batch}x{time}")
    phases: list[Phase] = []
    bindings: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
    claim_id = 0
    micro = 0
    for at, launch in enumerate(program):
        claims = []
        for r in launch.records:
            slots = uses(r)
            rd: list[int] = []
            wr: list[int] = []
            fresh: list[tuple[int, int, int]] = []  # written spans, made values after the reads
            bound: dict[int, list[int]] = {}
            for slot, use in slots:  # the reads first: a call reads what was there before it
                space, offset, nbytes = r.refs[slot]
                if space <= 4:
                    bound[slot] = parameters(space, offset, nbytes, at)
                elif space == 7:
                    if workspace_rid not in resources:
                        declare(workspace_rid, "workspace", 0, nbytes, 1, "workspace", at)
                    bound[slot] = [workspace_rid]
                elif space == 8:
                    tokens = (r.kind, slot) in ((nl.EMBED, 0), (nl.SCATTER, 0))
                    rid = tokens_rid if tokens else targets_rid
                    if rid not in resources:
                        declare(rid, "tokens" if tokens else "targets", 0, nbytes, 4,
                                "tokens" if tokens else "targets", at)  # fmt: skip
                    bound[slot] = [rid]
                elif use & R:
                    rid = current.get((space, offset, nbytes))
                    if rid is None:
                        raise ValueError(
                            f"launch {at} ({op_name(r)}) reads {ARENAS[space - 1]}+{offset}/"
                            f"{nbytes}, which no live value holds"
                        )
                    if use == RW:
                        end_meeting(space, offset, nbytes, keep=(space, offset, nbytes))
                    bound[slot] = [rid]
                else:
                    fresh.append((slot, space, offset, nbytes))
            for slot, space, offset, nbytes in fresh:
                end_meeting(space, offset, nbytes)
                elem = (
                    2
                    if r.kind in (nl.SILU_MUL_PANELS, nl.SWIGLU_VJP_PANELS)
                    else (8 if (r.kind, slot) == (nl.LOSS, 3) else 4)
                )
                declare(next_rid, ARENAS[space - 1], offset, nbytes, elem,
                        value_name(space, offset, nbytes), at)  # fmt: skip
                current[space, offset, nbytes] = next_rid
                bound[slot] = [next_rid]
                next_rid += 1
            for slot, use in slots:
                for rid in bound[slot]:
                    touch(rid, at)
                if use & R:
                    rd += bound[slot]
                if use & W:
                    wr += bound[slot]
            bindings.append(tuple((slot, tuple(bound[slot])) for slot, _use in slots))
            opcode, lane, stride, cost, hazard = _form(r.kind)
            base = micro if r.kind in (nl.EMBED, nl.LOSS, nl.SCATTER) else 0
            claims.append(Claim(id=claim_id, opcode=opcode, lane=lane, stride_class=stride,
                                count=work(r), rd=tuple(rd), wr=tuple(wr), hazard=hazard,
                                domain=Domain.RAM, verify="bounds", bounds="masked",
                                op=op_name(r), offset=base, cost_class=cost))  # fmt: skip
            claim_id += 1
            if r.kind == nl.CHECK and r.flags == nl.CHECK_LOSS:
                micro += 1
        phases.append(Phase(phase_id=at, deps=(at - 1,) if at else (), claims=claims))
    last = len(program) - 1
    for rid, place in places.items():  # what lives through the program lives through all of it
        if rid <= targets_rid:
            place[3], place[4] = 0, last
    for rid in sorted(resources):
        module.add_resource(resources[rid])
    for phase in phases:
        module.add_phase(phase)
    placements = tuple(Placement(rid, *places[rid]) for rid in sorted(places))
    return TrainingPack(spec, batch, time, program, tuple(nl.program_words(list(program))), module,
                        placements, tensors, tuple(bindings))  # fmt: skip


def placed_words(tp: TrainingPack, placements) -> tuple[int, ...]:
    """The words the lowering of `tp`'s pack placed by `placements` must write: the program's, every
    operand at its value's place -- a slot's space its value's bank, its offset the value's (a run
    of parameter tensors from its first) -- and every other word as it was. With `tp.placements`,
    `tp.words` itself."""
    where = {p.rid: p for p in placements}
    words = list(tp.words)
    for i, binding in enumerate(tp.bindings):
        base = i * nl.RECORD_WORDS + 11
        for slot, rids in binding:
            place = where[rids[0]]
            words[base + 3 * slot] = BANKS[place.bank]
            words[base + 3 * slot + 1] = place.offset
    return tuple(words)


#: PX-4: the alignment a placed transient value starts at -- a cache line, as the C step's own
#: buffers start (the kernels' arithmetic reads no address: any 4-aligned place trains the same bits).
PLACE_ALIGNMENT = 64


@dataclass(frozen=True)
class Placed:
    """PX-4: a pack's values with its transient ones placed by their lifetimes (`place_by_lifetime`):
    every value's place, the transient bank's extent in bytes, G26's bound over the same values, the
    parts laid out apart, and whether every part's layout is a proved optimum."""

    placements: tuple[Placement, ...]
    extent: int
    bound: int
    parts: int
    optimal: bool

    @property
    def floats(self) -> int:
        """The transient arena's floats: what a trainer lends for these placements."""
        return -(-self.extent // 4)


def place_by_lifetime(tp: TrainingPack, *, alignment: int = PLACE_ALIGNMENT,
                      budget: int | None = None) -> Placed:  # fmt: skip
    """`tp`'s values with every activation and scratch value placed by its lifetime in the one bank
    `activations` (module docstring): a value occupies its bytes from its first phase through its
    last (half-open ticks first .. last + 1, the plan's alias law); the values split into parts at
    every phase no value is live across, each part laid out by `exact_layout` (GEM+'s incumbent, then
    the bounded exact search; `budget` its work units, the solver's default when None), the parts
    overlapping in address. Every other value -- the parameters, the workspace, the tokens and the
    targets -- keeps its place."""
    from ...kbcir.static_memory import (
        DEFAULT_EXACT_BUDGET,
        LayoutItem,
        exact_layout,
        layout_lower_bound,
    )

    items = sorted(
        (LayoutItem(p.rid, p.size, alignment, p.first, p.last + 1)
         for p in tp.placements if p.bank in TRANSIENT),
        key=lambda item: (item.first_tick, item.rid),
    )  # fmt: skip
    parts: list[list] = []
    reach = 0  # the end of the part being gathered: a value starting there begins the next
    for item in items:
        if not parts or item.first_tick >= reach:
            parts.append([])
            reach = item.last_tick
        else:
            reach = max(reach, item.last_tick)
        parts[-1].append(item)
    offsets: dict[int, int] = {}
    extent, optimal = 0, True
    for part in parts:
        laid = exact_layout(part, DEFAULT_EXACT_BUDGET if budget is None else budget)
        offsets.update(laid.offsets)
        extent = max(extent, laid.extent)
        optimal = optimal and laid.stop_reason == "optimal"
    placements = tuple(
        Placement(p.rid, "activations", offsets[p.rid], p.size, p.first, p.last)
        if p.rid in offsets else p
        for p in tp.placements
    )  # fmt: skip
    return Placed(placements, extent, layout_lower_bound(items), len(parts), optimal)


def pack_artifacts(tp: TrainingPack, *, placements=None, target=None):
    """(pack bytes, plan bytes, pack, plan): `tp`'s module planned by K_BCIR on `target` (the
    AVX-512 profile by default), hydrated by GEM, and its ExecutionPlanV1 minted from the same
    realization, its lifetimes `placements` (the C step's own, `tp.placements`, by default)."""
    from ...abi.execution_plan_abi import encode_plan
    from ...abi.streampack_abi import encode
    from ...gem.execution_plan import Lifetime, plan_from_realization
    from ...gem.streampack import hydrate
    from ...kbcir import optimize
    from ...kbcir.cost import TargetProfile, Theta

    target = target if target is not None else TargetProfile.x86_avx512()
    result = optimize(tp.module, target, Theta.cool())
    pack = hydrate(tp.module, result)
    plan = plan_from_realization(tp.module, result, target)
    rows = tp.placements if placements is None else placements
    plan.lifetimes = [Lifetime(p.rid, p.bank, p.offset, p.size, 4, p.first, p.last)
                      for p in sorted(rows, key=lambda p: p.rid)]  # fmt: skip
    return encode(pack), encode_plan(plan), pack, plan


__all__ = [
    "BANKS",
    "REASONS",
    "GEMM_ORIENTS",
    "GEMM_ROLES",
    "OPS",
    "PLACE_ALIGNMENT",
    "Placed",
    "Placement",
    "TrainingPack",
    "op_name",
    "pack_artifacts",
    "place_by_lifetime",
    "placed_words",
    "program_pack",
    "training_pack",
    "uses",
    "work",
]
