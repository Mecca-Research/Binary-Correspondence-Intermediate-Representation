"""ExecutionPlanV1 binary ABI (frozen v1) + v2 (append-only) -- reference codec.

The plan as bytes (G11, staged plan S1-C; v2 from G5, S1-D). Layout (little-endian; the
normative spec is docs/kernel/BCIR_EXECUTION_PLAN_ABI.md and the C view
runtime/c/bcir_execution_plan.h):

    Header (64 bytes, cache-line; every u64 at an 8-aligned offset):
      magic[4]="BPLN"  version:u16  flags:u16
      mode:u8 @8 (0 = eft phase-barriered, 1 = tokens pipelined)
      [v2] liveness:u8 @9 (0 = phase positions, 1 = the placement's ticks; reserved on v1)
      reserved:u8[2] @10
      streams:u32 @12  knee:u32 @16
      n_steps:u32 @20  n_lifetimes:u32 @24  n_moves:u32 @28  n_gens:u32 @32
      reserved:u8[4] @36
      makespan:u64 @40  module_hash:u64 @48 (R13 hash_module)
      target_hash:u64 @56 (R13 hash_target; 0 = none)
    Body (sequential, length-prefixed records):
      source_plan:str
      steps[n_steps]           claim_id:u64 phase_id:u32 candidate:str lane:u8 width:u32
                               cost:i64 stream:u32 start:u64 duration:u64
      lifetimes[n_lifetimes]   rid:u32 bank:str offset:u64 size:u64 alignment:u32
                               first_phase:u32 last_phase:u32          (RIDs strictly ascending)
                               [v2: first_tick:u64 last_tick:u64]    (half-open, in the
                               liveness domain; a v1 record reads [first_phase, last_phase+1))
      moves[n_moves]           rid:u32 src_bank:str dst_bank:str offset:u64 size:u64 route:str
                               kind:u8 coherence:u8 map_gen:u32 data_gen:u32
                               after_claim:u64 before_claim:u64
                               [v3: claim:u64 version:u32 flags:u8 producer:u64 bits:u8
                                cert:u64]                     (G8: what executes and carries it)
      generations[n_gens]      rid:u32 map_gen:u32 data_gen:u32           (RIDs strictly ascending)
      [v3: source_hash:u64 spec_hash:u64]    (the goal graph and the movement spec it was planned
                                              for; kbcir.movement)
    Trailer:
      crc32:u32  (CRC-32 of every preceding byte)

Strings are u16 length + UTF-8, as in the StreamPack ABI whose conventions this format
shares (the same writer/reader primitives). `stream` spells the decoupled tail as
0xFFFFFFFF (the model's `TAIL_STREAM`, -1); `cost` is a two's-complement i64 because a
plan's step costs are signed. The format is frozen at v1 and evolves append-only: v2 carves
the liveness byte out of the header pad and appends the tick tail to the lifetime record; v3
(G8) appends a tail to the move record and a binding trailer after the generation vector.
The encoder emits the lowest carrying version (a plan under phase liveness whose lifetimes
carry the phase default is byte-identical v1, and a plan that moves nothing is never v3); a
reader rejects a newer version and refuses nonzero reserved bytes.

The wire laws (applied by the encoder AND the decoder, and by the C twin identically):
mode legal; streams >= 1 and 1 <= knee <= streams; every step's lane legal, width a nonzero
power of two, stream in range or the tail, duration == max(0, cost), start + duration <=
makespan; claim ids unique; lifetimes with strictly ascending RIDs, a power-of-two alignment
the offset honors, size >= 1, first_phase <= last_phase, last_tick > first_tick, and no two
lifetimes of one bank live at once at overlapping addresses (the alias law by bytes); moves
with legal kind/coherence codes, size >= 1, two different banks unless the edge is a remat
(which replays in place), and a writeback that is neither compressed nor a remat; the v3 move
laws (`_validate_moves_v3`); a generation vector with strictly ascending RIDs; the declared
records
consume the body exactly (undeclared trailing bytes are refused, never treated as an
extension point).
"""

from __future__ import annotations

import struct
import zlib

from ..gem.execution_plan import (
    COHERENCE_ACTIONS,
    LIVENESS_DOMAINS,
    MOVE_FLAGS,
    MOVE_HAS_AFTER,
    MOVE_HAS_BEFORE,
    MOVE_HAS_CLAIM,
    MOVE_HAS_PRODUCER,
    MOVE_KINDS,
    PLAN_MODES,
    TAIL_STREAM,
    ExecutionPlan,
    Lifetime,
    MovementEdge,
    PlanStep,
)
from ..gem.streampack import Generation
from ..model import Lane
from .streampack_abi import AbiError, _checked_uint, _Reader, _Writer

PLAN_MAGIC = b"BPLN"
PLAN_VERSION = 1
# v2 (G5, S1-D): the liveness byte in the header and the lifetime tick tail -- append-only.
# v3 (G8, S5-C): the move record tail and the source/spec binding trailer -- append-only.
PLAN_VERSION_MAX = 3
PLAN_HEADER_SIZE = 64
_LIVENESS_OFF = 9

#: The tail stream on the wire (the model's TAIL_STREAM, -1).
TAIL_STREAM_WIRE = 0xFFFFFFFF

_MODE_WIRE = {mode: code for code, mode in enumerate(PLAN_MODES)}
_MODE_FROM_WIRE = {v: k for k, v in _MODE_WIRE.items()}
_LIVENESS_WIRE = {domain: code for code, domain in enumerate(LIVENESS_DOMAINS)}
_LIVENESS_FROM_WIRE = {v: k for k, v in _LIVENESS_WIRE.items()}
_KIND_WIRE = {kind: code for code, kind in enumerate(MOVE_KINDS)}
_KIND_FROM_WIRE = {v: k for k, v in _KIND_WIRE.items()}
_COHERENCE_WIRE = {act: code for code, act in enumerate(COHERENCE_ACTIONS)}
_COHERENCE_FROM_WIRE = {v: k for k, v in _COHERENCE_WIRE.items()}

# magic, version, flags, mode, pad[3], streams, knee, n_steps, n_lifetimes, n_moves, n_gens,
# pad[4], makespan, module_hash, target_hash -> exactly the 64-byte line, the three u64
# fields 8-aligned so the C view (`bcir_ep_header`) is the wire layout without padding.
_HEADER = struct.Struct("<4sHHB3xIIIIII4xQQQ")
assert _HEADER.size == PLAN_HEADER_SIZE
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1
_U64_MAX = (1 << 64) - 1


def _checked_i64(name: str, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not _I64_MIN <= value <= _I64_MAX:
        raise AbiError(f"{name} value must be a signed 64-bit integer, got {value!r}")
    return value


def _checked_str(name: str, value) -> str:
    if not isinstance(value, str):
        raise AbiError(f"{name} must be str, got {type(value).__name__}")
    if len(value.encode("utf-8")) > 0xFFFF:
        raise AbiError(f"{name} exceeds the u16 wire length")
    return value


def validate_plan(plan: ExecutionPlan, version: int | None = None) -> None:
    """The wire laws, shared by `encode_plan` and `decode_plan` (rail symmetry with the C
    twin's `bcir_ep_verify`): a plan that violates them is refused before it is published
    and refused again when it is read. `version` is the wire version the plan travels in --
    the one it was read from, or by default the lowest that carries it: the v3 laws bind
    every v3 plan, so a v3 wire whose binding and move tails are all zero is refused rather
    than read as the v2 plan it would re-encode to."""
    if plan.mode not in _MODE_WIRE:
        raise AbiError(f"unknown plan mode {plan.mode!r}; expected one of {PLAN_MODES}")
    if plan.liveness not in _LIVENESS_WIRE:
        raise AbiError(
            f"unknown plan liveness {plan.liveness!r}; expected one of {LIVENESS_DOMAINS}"
        )
    streams = _checked_uint("streams", plan.streams, 32)
    knee = _checked_uint("knee", plan.knee, 32)
    if streams < 1 or not 1 <= knee <= streams:
        raise AbiError(f"plan needs streams >= 1 and 1 <= knee <= streams, got {streams}/{knee}")
    makespan = _checked_uint("makespan", plan.makespan, 64)
    _checked_uint("module_hash", plan.module_hash, 64)
    _checked_uint("target_hash", plan.target_hash, 64)
    _checked_str("source_plan", plan.source_plan)
    _checked_uint("n_steps", len(plan.steps), 32)
    _checked_uint("n_lifetimes", len(plan.lifetimes), 32)
    _checked_uint("n_moves", len(plan.moves), 32)
    _checked_uint("n_gens", len(plan.generations), 32)
    seen: set[int] = set()
    for index, s in enumerate(plan.steps):
        claim_id = _checked_uint(f"step[{index}].claim_id", s.claim_id, 64)
        if claim_id in seen:
            raise AbiError(f"step[{index}] duplicates claim {claim_id}")
        seen.add(claim_id)
        _checked_uint(f"step[{index}].phase_id", s.phase_id, 32)
        _checked_str(f"step[{index}].candidate", s.candidate)
        if isinstance(s.lane, bool):
            raise AbiError(f"step[{index}] lane must be a Lane value")
        try:
            Lane(s.lane)
        except (TypeError, ValueError) as exc:
            raise AbiError(f"step[{index}] has unknown lane {s.lane!r}") from exc
        width = _checked_uint(f"step[{index}].width", s.width, 32)
        if width == 0 or width & (width - 1):
            raise AbiError(f"step[{index}] width must be a nonzero power of two, got {width}")
        cost = _checked_i64(f"step[{index}].cost", s.cost)
        if s.stream != TAIL_STREAM:
            stream = _checked_uint(f"step[{index}].stream", s.stream, 32)
            if stream >= streams or stream == TAIL_STREAM_WIRE:
                raise AbiError(f"step[{index}] stream {stream} is outside {streams} streams")
        start = _checked_uint(f"step[{index}].start", s.start, 64)
        duration = _checked_uint(f"step[{index}].duration", s.duration, 64)
        if duration != max(0, cost):
            raise AbiError(
                f"step[{index}] duration {duration} is not the slot its cost {cost} places"
            )
        if start + duration > makespan:
            raise AbiError(f"step[{index}] finishes at {start + duration} past makespan {makespan}")
    previous = -1
    for index, lt in enumerate(plan.lifetimes):
        rid = _checked_uint(f"lifetime[{index}].rid", lt.rid, 32)
        if rid <= previous:
            raise AbiError(
                f"lifetime RIDs must be strictly ascending (lifetime[{index}] rid {rid} "
                f"after {previous})"
            )
        previous = rid
        if not _checked_str(f"lifetime[{index}].bank", lt.bank):
            raise AbiError(f"lifetime[{index}] names no bank")
        offset = _checked_uint(f"lifetime[{index}].offset", lt.offset, 64)
        size = _checked_uint(f"lifetime[{index}].size_bytes", lt.size_bytes, 64)
        alignment = _checked_uint(f"lifetime[{index}].alignment", lt.alignment, 32)
        if size == 0:
            raise AbiError(f"lifetime[{index}] has zero size")
        if alignment == 0 or alignment & (alignment - 1):
            raise AbiError(f"lifetime[{index}] alignment must be a power of two, got {alignment}")
        if offset % alignment:
            raise AbiError(f"lifetime[{index}] offset {offset} violates alignment {alignment}")
        if offset + size > _U64_MAX:
            raise AbiError(f"lifetime[{index}] end exceeds the 64-bit address space")
        first = _checked_uint(f"lifetime[{index}].first_phase", lt.first_phase, 32)
        last = _checked_uint(f"lifetime[{index}].last_phase", lt.last_phase, 32)
        if last < first:
            raise AbiError(f"lifetime[{index}] is reversed ({first} > {last})")
        first_tick = _checked_uint(f"lifetime[{index}].first_tick", lt.first_tick, 64)
        last_tick = _checked_uint(f"lifetime[{index}].last_tick", lt.last_tick, 64)
        if last_tick <= first_tick:
            raise AbiError(f"lifetime[{index}] liveness interval is empty or reversed")
    # The alias law by bytes: two lifetimes of one bank that are live at once (half-open
    # ticks) must not overlap in address. The C twin applies the same pairwise predicate.
    rows = list(plan.lifetimes)
    for i in range(len(rows)):
        for j in range(i):
            a, b = rows[i], rows[j]
            if (
                a.bank == b.bank
                and a.first_tick < b.last_tick
                and b.first_tick < a.last_tick
                and a.offset < b.offset + b.size_bytes
                and b.offset < a.offset + a.size_bytes
            ):
                raise AbiError(f"lifetimes of RIDs {b.rid} and {a.rid} alias in bank {a.bank!r}")
    for index, mv in enumerate(plan.moves):
        _checked_uint(f"move[{index}].rid", mv.rid, 32)
        for name in ("src_bank", "dst_bank"):
            if not _checked_str(f"move[{index}].{name}", getattr(mv, name)):
                raise AbiError(f"move[{index}] names no {name}")
        offset = _checked_uint(f"move[{index}].offset", mv.offset, 64)
        size = _checked_uint(f"move[{index}].size_bytes", mv.size_bytes, 64)
        if size == 0:
            raise AbiError(f"move[{index}] moves zero bytes")
        if offset + size > _U64_MAX:
            raise AbiError(f"move[{index}] range exceeds the 64-bit address space")
        _checked_str(f"move[{index}].route", mv.route)
        if mv.kind not in _KIND_WIRE:
            raise AbiError(f"move[{index}] has unknown kind {mv.kind!r}; expected {MOVE_KINDS}")
        if mv.coherence not in _COHERENCE_WIRE:
            raise AbiError(
                f"move[{index}] has unknown coherence {mv.coherence!r}; "
                f"expected {COHERENCE_ACTIONS}"
            )
        _checked_uint(f"move[{index}].map_gen", mv.map_gen, 32)
        _checked_uint(f"move[{index}].data_gen", mv.data_gen, 32)
        _checked_uint(f"move[{index}].after_claim", mv.after_claim, 64)
        _checked_uint(f"move[{index}].before_claim", mv.before_claim, 64)
        # a move moves between two banks; a remat replays its producer in place
        if (mv.src_bank == mv.dst_bank) != (mv.kind == "rematerialized"):
            raise AbiError(
                f"move[{index}] ({mv.kind}) from {mv.src_bank!r} to {mv.dst_bank!r}: only a remat "
                f"stays in its bank"
            )
        if mv.coherence == "writeback" and mv.kind in ("compressed", "rematerialized"):
            raise AbiError(f"move[{index}] writes back through a {mv.kind} edge")
    _checked_uint("source_hash", plan.source_hash, 64)
    _checked_uint("spec_hash", plan.spec_hash, 64)
    if (plan_version(plan) if version is None else version) >= 3:
        _validate_moves_v3(plan)
    previous = -1
    for index, g in enumerate(plan.generations):
        rid = _checked_uint(f"generation[{index}].rid", g.rid, 32)
        _checked_uint(f"generation[{index}].map_gen", g.map_gen, 32)
        _checked_uint(f"generation[{index}].data_gen", g.data_gen, 32)
        if rid <= previous:
            raise AbiError(
                f"generation vector RIDs must be strictly ascending (generation[{index}] "
                f"rid {rid} after {previous})"
            )
        previous = rid


def _validate_moves_v3(plan: ExecutionPlan) -> None:
    """The v3 move laws, from the plan's bytes alone (the C twin's `bcir_ep_verify` applies the
    same): flags within the defined set, and a reference the flags call absent is zero; a remat
    names its producer and nothing else does; a compressed edge names its codec's bits (1..31)
    and nothing else does; exactly the remat and compressed edges carry a certificate; every
    referenced claim is a step, the executing claim is not its own producer, and the window is
    ordered by the placement (the source's writer finishes before the move starts, and the move
    finishes before its first reader starts); a writeback lands in its resource's home -- the
    bank of its lifetime -- when the plan carries one; and the plan names the goal graph and the
    spec its movement was planned under."""
    if not plan.source_hash or not plan.spec_hash:
        raise AbiError("a v3 plan must name its source module and movement spec (nonzero hashes)")
    slots = {s.claim_id: (s.start, s.start + s.duration) for s in plan.steps}
    home = {lt.rid: lt.bank for lt in plan.lifetimes}
    for index, mv in enumerate(plan.moves):
        claim = _checked_uint(f"move[{index}].claim", mv.claim, 64)
        _checked_uint(f"move[{index}].version", mv.version, 32)
        flags = _checked_uint(f"move[{index}].flags", mv.flags, 8)
        producer = _checked_uint(f"move[{index}].producer", mv.producer, 64)
        bits = _checked_uint(f"move[{index}].bits", mv.bits, 8)
        cert = _checked_uint(f"move[{index}].cert", mv.cert, 64)
        if flags & ~MOVE_FLAGS:
            raise AbiError(f"move[{index}] has undefined flags 0x{flags:02x}")
        for flag, value, name in (
            (MOVE_HAS_AFTER, mv.after_claim, "after_claim"),
            (MOVE_HAS_BEFORE, mv.before_claim, "before_claim"),
            (MOVE_HAS_PRODUCER, producer, "producer"),
            (MOVE_HAS_CLAIM, claim, "claim"),
        ):
            if not flags & flag and value:
                raise AbiError(f"move[{index}] carries {name} {value}, which its flags call absent")
            if flags & flag and value not in slots:
                raise AbiError(f"move[{index}] {name} {value} is not a step of the plan")
        if bool(flags & MOVE_HAS_PRODUCER) != (mv.kind == "rematerialized"):
            raise AbiError(f"move[{index}] ({mv.kind}): exactly a remat names a producer")
        if flags & MOVE_HAS_PRODUCER and flags & MOVE_HAS_CLAIM and producer == claim:
            raise AbiError(f"move[{index}] remat replays itself")
        if (mv.kind == "compressed") != (1 <= bits <= 31) or (mv.kind != "compressed" and bits):
            raise AbiError(
                f"move[{index}] ({mv.kind}) codec bits {bits}: exactly a compressed edge names 1..31"
            )
        if bool(cert) != (mv.kind in ("rematerialized", "compressed")):
            raise AbiError(
                f"move[{index}] ({mv.kind}): exactly a remat or compressed edge carries a certificate"
            )
        if flags & MOVE_HAS_CLAIM:
            start, finish = slots[claim]
            if flags & MOVE_HAS_AFTER and slots[mv.after_claim][1] > start:
                raise AbiError(
                    f"move[{index}] starts before its source's writer {mv.after_claim} finishes"
                )
            if flags & MOVE_HAS_BEFORE and finish > slots[mv.before_claim][0]:
                raise AbiError(f"move[{index}] finishes after its reader {mv.before_claim} starts")
        if mv.coherence == "writeback" and mv.rid in home and mv.dst_bank != home[mv.rid]:
            raise AbiError(
                f"move[{index}] writes resource {mv.rid} back to {mv.dst_bank!r}, not its home "
                f"{home[mv.rid]!r}"
            )


def _write_step(w: _Writer, s: PlanStep) -> None:
    w.u64(s.claim_id)
    w.u32(s.phase_id)
    w.s(s.candidate)
    w.u8(int(s.lane))
    w.u32(s.width)
    w.u64(s.cost & _U64_MAX)  # two's complement i64
    w.u32(TAIL_STREAM_WIRE if s.stream == TAIL_STREAM else s.stream)
    w.u64(s.start)
    w.u64(s.duration)


def _write_lifetime(w: _Writer, lt: Lifetime, version: int = PLAN_VERSION) -> None:
    w.u32(lt.rid)
    w.s(lt.bank)
    w.u64(lt.offset)
    w.u64(lt.size_bytes)
    w.u32(lt.alignment)
    w.u32(lt.first_phase)
    w.u32(lt.last_phase)
    if version >= 2:
        w.u64(lt.first_tick)
        w.u64(lt.last_tick)


def plan_version(plan: ExecutionPlan) -> int:
    """The lowest wire version that carries `plan`: v3 when a move carries the v3 tail or the
    plan binds a movement transform (G8); v2 when the lifetimes live in the schedule domain or
    any lifetime's ticks are not the phase default; else the frozen v1."""
    if plan.source_hash or plan.spec_hash or any(mv.v3 for mv in plan.moves):
        return 3
    if plan.liveness != "phase" or any(not lt.phase_default for lt in plan.lifetimes):
        return 2
    return PLAN_VERSION


def _write_move(w: _Writer, mv: MovementEdge, version: int = PLAN_VERSION) -> None:
    w.u32(mv.rid)
    w.s(mv.src_bank)
    w.s(mv.dst_bank)
    w.u64(mv.offset)
    w.u64(mv.size_bytes)
    w.s(mv.route)
    w.u8(_KIND_WIRE[mv.kind])
    w.u8(_COHERENCE_WIRE[mv.coherence])
    w.u32(mv.map_gen)
    w.u32(mv.data_gen)
    w.u64(mv.after_claim)
    w.u64(mv.before_claim)
    if version >= 3:
        w.u64(mv.claim)
        w.u32(mv.version)
        w.u8(mv.flags)
        w.u64(mv.producer)
        w.u8(mv.bits)
        w.u64(mv.cert)


def _write_generation(w: _Writer, g: Generation) -> None:
    w.u32(g.rid)
    w.u32(g.map_gen)
    w.u32(g.data_gen)


def encode_plan(plan: ExecutionPlan) -> bytes:
    """Serialize an ExecutionPlan (the lowest carrying version, CRC trailer); refuses a plan
    the wire laws reject."""
    validate_plan(plan)
    version = plan_version(plan)
    header = bytearray(
        _HEADER.pack(
            PLAN_MAGIC,
            version,
            0,
            _MODE_WIRE[plan.mode],
            plan.streams,
            plan.knee,
            len(plan.steps),
            len(plan.lifetimes),
            len(plan.moves),
            len(plan.generations),
            plan.makespan,
            plan.module_hash,
            plan.target_hash,
        )
    )
    if version >= 2:
        header[_LIVENESS_OFF] = _LIVENESS_WIRE[plan.liveness]
    header = bytes(header)
    w = _Writer()
    w.s(plan.source_plan)
    for s in plan.steps:
        _write_step(w, s)
    for lt in plan.lifetimes:
        _write_lifetime(w, lt, version)
    for mv in plan.moves:
        _write_move(w, mv, version)
    for g in plan.generations:
        _write_generation(w, g)
    if version >= 3:
        w.u64(plan.source_hash)
        w.u64(plan.spec_hash)
    body = header + bytes(w.buf)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_plan(data: bytes) -> ExecutionPlan:
    """Parse the wire format (v1..v3) back into an ExecutionPlan (magic/version/reserved/CRC,
    bounds, exact consumption, then the wire laws)."""
    if len(data) < PLAN_HEADER_SIZE + 4:
        raise AbiError("buffer too small for an ExecutionPlan")
    (
        magic,
        version,
        flags,
        mode_code,
        streams,
        knee,
        n_steps,
        n_lifetimes,
        n_moves,
        n_gens,
        makespan,
        module_hash,
        target_hash,
    ) = _HEADER.unpack(data[:PLAN_HEADER_SIZE])
    if magic != PLAN_MAGIC:
        raise AbiError(f"bad magic {magic!r} (expected {PLAN_MAGIC!r})")
    if not (PLAN_VERSION <= version <= PLAN_VERSION_MAX):
        raise AbiError(
            f"unsupported ExecutionPlan version {version} "
            f"(this reader handles v{PLAN_VERSION}..v{PLAN_VERSION_MAX})"
        )
    if flags:
        raise AbiError(f"reserved ExecutionPlan flags must be zero, got 0x{flags:04x}")
    reserved_lo = data[10:12] if version >= 2 else data[9:12]
    if any(reserved_lo) or any(data[36:40]):
        raise AbiError("reserved ExecutionPlan header bytes must be zero")
    if mode_code not in _MODE_FROM_WIRE:
        raise AbiError(f"unknown plan mode code {mode_code} (0=eft, 1=tokens)")
    liveness_code = data[_LIVENESS_OFF] if version >= 2 else 0
    if liveness_code not in _LIVENESS_FROM_WIRE:
        raise AbiError(f"unknown plan liveness code {liveness_code} (0=phase, 1=schedule)")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise AbiError("CRC mismatch (corrupt ExecutionPlan)")

    r = _Reader(data, PLAN_HEADER_SIZE)
    plan = ExecutionPlan(
        source_plan=r.s(),
        mode=_MODE_FROM_WIRE[mode_code],
        liveness=_LIVENESS_FROM_WIRE[liveness_code],
        streams=streams,
        knee=knee,
        makespan=makespan,
        module_hash=module_hash,
        target_hash=target_hash,
    )
    for _ in range(n_steps):
        claim_id = r.u64()
        phase_id = r.u32()
        candidate = r.s()
        raw_lane = r.u8()
        try:
            lane = Lane(raw_lane)
        except ValueError as exc:
            raise AbiError(f"unknown step lane code {raw_lane}") from exc
        width = r.u32()
        raw_cost = r.u64()
        cost = raw_cost - (1 << 64) if raw_cost >> 63 else raw_cost
        raw_stream = r.u32()
        stream = TAIL_STREAM if raw_stream == TAIL_STREAM_WIRE else raw_stream
        start = r.u64()
        duration = r.u64()
        plan.steps.append(
            PlanStep(claim_id, phase_id, candidate, lane, width, cost, stream, start, duration)
        )
    for _ in range(n_lifetimes):
        rid, bank = r.u32(), r.s()
        offset, size = r.u64(), r.u64()
        alignment, first_phase, last_phase = r.u32(), r.u32(), r.u32()
        if version >= 2:
            first_tick, last_tick = r.u64(), r.u64()
        else:
            first_tick, last_tick = first_phase, last_phase + 1
        plan.lifetimes.append(
            Lifetime(
                rid=rid,
                bank=bank,
                offset=offset,
                size_bytes=size,
                alignment=alignment,
                first_phase=first_phase,
                last_phase=last_phase,
                first_tick=first_tick,
                last_tick=last_tick,
            )
        )
    for index in range(n_moves):
        rid = r.u32()
        src, dst = r.s(), r.s()
        offset, size = r.u64(), r.u64()
        route = r.s()
        kind_code, coherence_code = r.u8(), r.u8()
        if kind_code not in _KIND_FROM_WIRE:
            raise AbiError(f"move[{index}] has unknown kind code {kind_code}")
        if coherence_code not in _COHERENCE_FROM_WIRE:
            raise AbiError(f"move[{index}] has unknown coherence code {coherence_code}")
        map_gen, data_gen = r.u32(), r.u32()
        after_claim, before_claim = r.u64(), r.u64()
        tail = {}
        if version >= 3:
            tail = {
                "claim": r.u64(),
                "version": r.u32(),
                "flags": r.u8(),
                "producer": r.u64(),
                "bits": r.u8(),
                "cert": r.u64(),
            }
        plan.moves.append(
            MovementEdge(
                rid=rid,
                src_bank=src,
                dst_bank=dst,
                offset=offset,
                size_bytes=size,
                route=route,
                kind=_KIND_FROM_WIRE[kind_code],
                coherence=_COHERENCE_FROM_WIRE[coherence_code],
                map_gen=map_gen,
                data_gen=data_gen,
                after_claim=after_claim,
                before_claim=before_claim,
                **tail,
            )
        )
    for _ in range(n_gens):
        plan.generations.append(Generation(rid=r.u32(), map_gen=r.u32(), data_gen=r.u32()))
    if version >= 3:
        plan.source_hash, plan.spec_hash = r.u64(), r.u64()
    if r.pos != len(data) - 4:
        raise AbiError(
            f"unexpected trailing body bytes: decoded through offset {r.pos}, "
            f"CRC trailer starts at {len(data) - 4}"
        )
    validate_plan(plan, version)
    return plan


__all__ = [
    "PLAN_HEADER_SIZE",
    "PLAN_MAGIC",
    "PLAN_VERSION",
    "PLAN_VERSION_MAX",
    "TAIL_STREAM_WIRE",
    "decode_plan",
    "encode_plan",
    "plan_version",
    "validate_plan",
]
