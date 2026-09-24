"""The K_BCIR planner's records (version zero) -- reference codec (GEM+ G17, S4-A).

The native planner (`runtime/c/bcir_kplan.{h,c}`) may own a certificate only after it
reproduces the Python plan byte for byte over a generated corpus -- the way the C twins earn
their rails. That needs the planner's whole input, and its whole output, as bytes:

* **BKPI** -- the planner input: everything `realize.optimize(module, h, theta, policy)` reads
  and nothing else. The phases (ids and dependencies, in declaration order), the claims in each
  (the fields candidate enumeration, the CSE identity, the CSE exclusions and the fusion rules
  read), the operands as indices into a resource table (declared or not; domain and addressing
  model), the op strings as a sorted table, the target's cost constants with its memory
  hierarchy resolved per normative tier, Theta and the policy's base weights.
* **BKPR** -- the realization: per step the claim, the phase, the realization (lane, width, name)
  with its coupled base cost and its realized cost, and the score. `encode_realization(result)`
  is the plan's bytes; the C planner writes the same bytes from the same input.

Layouts (little-endian; every field fixed-width; the normative spec is
docs/kernel/BCIR_PLANNER_ABI.md and the C view runtime/c/bcir_kplan.h):

    BKPI header (64): magic "BKPI"  version:u16=0  flags:u16=0  n_phases:u32 @8  n_deps:u32 @12
      n_claims:u32 @16  n_refs:u32 @20  n_resources:u32 @24  n_ops:u32 @28  op_bytes:u32 @32
      n_widths:u32 @36  reserved[24] @40
    scope (168) @64: target 8 x u32 (cacheline, elem_bytes, gather_penalty, mem_unit,
      base_overhead, thermal_density, power_density, per_op_heat); tiers 7 x (bw:u32, lat:u32)
      indexed by MemTier; theta 8 x u32; policy 12 x u32
    then: widths n_widths x u32 | phases n_phases x (phase_id, n_deps, n_claims):u32 |
      deps n_deps x u32 | claims n_claims x 48 | refs n_refs x u32 |
      resources n_resources x 4 (declared, domain, access, reserved):u8 |
      op lengths n_ops x u16 | op bytes op_bytes | crc32:u32

    BKPR header (32): magic "BKPR"  version:u16=0  flags:u16=0  n_steps:u32 @8
      reserved:u32 @12  score:u64 @16  reserved:u64 @24
    steps n_steps x 120: claim_id:u32 phase_id:u32 width:u32 lane:u8 name:u8 reserved:u16
      cost:u64 base:12 x u64 | crc32:u32

One spelling per input: resource indices are assigned in first-reference order (phases in
declaration order, a claim's reads, then its writes, then its primary resource), the op table is
strictly ascending and fully referenced, and a claim's primary resource is written only when it
is the one the planner reads (no reads, and a declared resource). The encoder decodes its own
output before returning it, so it refuses exactly what the decoder refuses, and the C twin
(`bcir_kp_decode_input`, `bcir_kp_decode_realization`) applies the same laws in the same order.
Every law other than the framing laws refuses with `BCIR_ERR_PLANNER`; a realization value the
record cannot carry (above 2**63 - 1) is `BCIR_ERR_OVERFLOW` on both rails.

Version zero: no compatibility promise until a consumer outside this repository reads it; a
change is a version bump, never a reinterpretation. It is not the BCIR UAPI.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from ..kbcir.cost import MemTier
from ..model import Lane, Opcode, StrideClass
from ..model.lanes import Domain
from .streampack_abi import AbiError

INPUT_MAGIC = b"BKPI"
REALIZATION_MAGIC = b"BKPR"
VERSION = 0
INPUT_HEADER_SIZE = 64
SCOPE_SIZE = 168
CLAIM_SIZE = 48
RESOURCE_SIZE = 4
REALIZATION_HEADER_SIZE = 32
STEP_SIZE = 120
CRC_SIZE = 4

# The declared domain. Inside it every intermediate of the planner is exact in 128 bits (the
# bound is worked in docs/kernel/BCIR_PLANNER_ABI.md), so the C twin never wraps and never
# refuses a plan Python makes -- it refuses only a realization the record cannot carry.
CLAIMS_MAX = 1 << 24
REFS_MAX = 1 << 26
OPERANDS_MAX = 255  # reads + writes of one claim
WIDTHS_MAX = 16
PARAM_MAX = 1 << 16  # every target constant, lane width, tier factor and policy weight
THETA_MAX = 100
OP_BYTES_MAX = 1 << 24
OP_LEN_MAX = 0xFFFF
U32 = 0xFFFFFFFF
I64_MAX = (1 << 63) - 1
NO_RESOURCE = U32

#: Realization names, by their code on the wire.
NAMES = (
    "noop",
    "barrier",
    "atomic",
    "blocked",
    "gather",
    "scalar",
    "vec",
    "strided",
    "ux_bucket",
    "tile",
)
NAME_VEC = NAMES.index("vec")
_WIDTH_ONE = frozenset(
    NAMES.index(n) for n in ("noop", "barrier", "atomic", "blocked", "gather", "scalar", "strided")
)

HAZARDS = ("unique", "atomic", "barriered")
CONTRACTS = ("none", "bounds", "exact", "hash")
ACCESS = ("flat", "ham")

# Claim `flags` (byte 10) and `flags2` (byte 11) bits: the facts the CSE exclusions read.
F_VOLATILE, F_DYNAMIC, F_CALLEE_SIG, F_TIMING, F_LIFETIME, F_IMM, F_TOLERANCE, F_QUANTIZED = (
    1 << i for i in range(8)
)
F2_PRECISION = 1

_IN_HEADER = struct.Struct("<4sHHIIIIIIII24s")
assert _IN_HEADER.size == INPUT_HEADER_SIZE
_SCOPE = struct.Struct("<" + "I" * (8 + 14 + 8 + 12))
assert _SCOPE.size == SCOPE_SIZE
_CLAIM = struct.Struct("<IBBBBBBBBIIHHqqII")
assert _CLAIM.size == CLAIM_SIZE
_PHASE = struct.Struct("<III")
_R_HEADER = struct.Struct("<4sHHIIQQ")
assert _R_HEADER.size == REALIZATION_HEADER_SIZE
_STEP = struct.Struct("<IIIBBHQ" + "Q" * 12)
assert _STEP.size == STEP_SIZE


class PlannerAbiError(AbiError):
    """A planner record the laws refuse. `status` is the C twin's `bcir_status` name for the
    same bytes."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _refuse(message: str) -> PlannerAbiError:
    return PlannerAbiError("BCIR_ERR_PLANNER", message)


# --- values --------------------------------------------------------------------------------


@dataclass(frozen=True)
class InputClaim:
    id: int
    opcode: int
    lane: int
    stride_class: int
    domain: int
    hazard: int
    verify: int
    flags: int
    flags2: int
    count: int
    op: int
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    stride_k: int
    offset: int
    primary: int  # a resource index, or NO_RESOURCE


@dataclass(frozen=True)
class InputResource:
    declared: int
    domain: int
    access: int


@dataclass(frozen=True)
class PlannerInput:
    """A decoded BKPI record: the planner's whole input, as integers."""

    target: tuple[int, ...]  # cacheline, elem_bytes, gather_penalty, mem_unit, base_overhead,
    # thermal_density, power_density, per_op_heat
    tiers: tuple[tuple[int, int], ...]  # (bw_factor, lat_factor) per MemTier
    theta: tuple[int, ...]  # the eight Theta fields, in declaration order
    policy: tuple[int, ...]  # the 12 base weights
    widths: tuple[int, ...]  # lane_widths, declared order
    phases: tuple[tuple[int, tuple[int, ...], tuple[InputClaim, ...]], ...]
    resources: tuple[InputResource, ...]
    ops: tuple[bytes, ...]


@dataclass(frozen=True)
class RealizedStep:
    claim_id: int
    phase_id: int
    width: int
    lane: int
    name: int
    cost: int
    base: tuple[int, ...]


@dataclass(frozen=True)
class Realization:
    """A decoded BKPR record."""

    score: int
    steps: tuple[RealizedStep, ...]


# --- BKPI: the planner input ----------------------------------------------------------------


def _theta_fields(theta) -> tuple[int, ...]:
    return (
        theta.thermal,
        theta.power,
        theta.mem_pressure,
        theta.contention,
        theta.noise,
        theta.wear,
        theta.utilization,
        theta.voltage,
    )


def encode_input(module, h, theta, policy=None) -> bytes:
    """The planner's input as a BKPI record: exactly what `realize.optimize(module, h, theta,
    policy)` reads. Refused (`PlannerAbiError`) outside the declared domain -- a module the
    record cannot spell once (duplicate phase or claim ids, a hazard, contract or addressing
    model outside the model's vocabulary), or a value outside its bound."""
    from ..kbcir.weights import PERF

    pol = policy if policy is not None else PERF
    resources: dict = {}  # rid -> index, first reference order
    resource_rows: list[InputResource] = []

    def index(rid) -> int:
        if rid in resources:
            return resources[rid]
        res = module.resources.get(rid) if rid is not None else None
        if res is None:
            row = InputResource(0, 0, 0)
        else:
            if res.access not in ACCESS:
                raise _refuse(f"resource {rid}: addressing model {res.access!r} is not flat/ham")
            row = InputResource(1, int(res.domain), ACCESS.index(res.access))
        resources[rid] = len(resource_rows)
        resource_rows.append(row)
        return resources[rid]

    op_names = sorted({c.op.encode("utf-8") for ph in module.phases for c in ph.claims})
    op_index = {name: i for i, name in enumerate(op_names)}
    phases = []
    for ph in module.phases:
        claims = []
        for c in ph.claims:
            if c.hazard not in HAZARDS:
                raise _refuse(f"claim {c.id}: hazard {c.hazard!r} is not unique/atomic/barriered")
            if c.verify not in CONTRACTS:
                raise _refuse(
                    f"claim {c.id}: verify contract {c.verify!r} is not one of {CONTRACTS}"
                )
            reads = tuple(index(r) for r in c.rd)
            writes = tuple(index(r) for r in c.wr)
            primary = NO_RESOURCE
            if not c.rd and c.primary_rid is not None and c.primary_rid in module.resources:
                primary = index(c.primary_rid)
            flags = (
                (F_VOLATILE if c.volatile else 0)
                | (F_DYNAMIC if c.dynamic else 0)
                | (F_CALLEE_SIG if bool(c.callee_sig) else 0)
                | (F_TIMING if c.timing is not None else 0)
                | (F_LIFETIME if c.lifetime is not None else 0)
                | (F_IMM if bool(c.imm) else 0)
                | (F_TOLERANCE if bool(c.tolerance_ulp) else 0)
                | (F_QUANTIZED if bool(c.quantized_bits) else 0)
            )
            claims.append(
                InputClaim(
                    id=c.id,
                    opcode=int(c.opcode),
                    lane=int(c.lane),
                    stride_class=int(c.stride_class),
                    domain=int(c.domain),
                    hazard=HAZARDS.index(c.hazard),
                    verify=CONTRACTS.index(c.verify),
                    flags=flags,
                    flags2=F2_PRECISION if bool(c.precision) else 0,
                    count=c.count,
                    op=op_index[c.op.encode("utf-8")],
                    reads=reads,
                    writes=writes,
                    stride_k=c.stride_k,
                    offset=c.offset,
                    primary=primary,
                )
            )
        phases.append((ph.phase_id, tuple(ph.deps), tuple(claims)))
    mem = h.mem
    tiers = tuple((mem.by_name(t.name).bw_factor, mem.by_name(t.name).lat_factor) for t in MemTier)
    value = PlannerInput(
        target=(
            h.cacheline,
            h.elem_bytes,
            h.gather_penalty,
            h.mem_unit,
            h.base_overhead,
            h.thermal_density,
            h.power_density,
            h.per_op_heat,
        ),
        tiers=tiers,
        theta=_theta_fields(theta),
        policy=tuple(pol.base),
        widths=tuple(h.lane_widths),
        phases=tuple(phases),
        resources=tuple(resource_rows),
        ops=tuple(op_names),
    )
    data = _pack_input(value)
    if decode_input(data) != value:  # the encoder refuses what the decoder refuses
        raise _refuse("the input record does not round-trip")
    check_input(value)  # ... and what the planner refuses before it plans
    return data


def _u32(value, what: str) -> int:
    if not isinstance(value, int) or not 0 <= value <= U32:
        raise _refuse(f"{what} {value!r} is outside 0..2**32-1")
    return value


def _u8(value, what: str) -> int:
    if not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise _refuse(f"{what} {value!r} is outside 0..255")
    return value


def _i64(value, what: str) -> int:
    if not isinstance(value, int) or not -(1 << 63) <= value <= I64_MAX:
        raise _refuse(f"{what} {value!r} is outside the signed 64-bit range")
    return value


def _pack_input(v: PlannerInput) -> bytes:
    """Lay a `PlannerInput` out as bytes. Only the widths are checked here -- a value that
    does not fit its field cannot be written at all; every other law is the decoder's, which
    the encoder applies to its own output."""
    n_deps = sum(len(deps) for _, deps, _ in v.phases)
    claims = [c for _, _, cs in v.phases for c in cs]
    n_refs = sum(len(c.reads) + len(c.writes) for c in claims)
    for op in v.ops:
        if len(op) > OP_LEN_MAX:
            raise _refuse(f"an op string of {len(op)} bytes exceeds {OP_LEN_MAX}")
    for c in claims:
        if len(c.reads) > 0xFFFF or len(c.writes) > 0xFFFF:
            raise _refuse(f"claim {c.id}: {len(c.reads) + len(c.writes)} operands")
    out = bytearray()
    out += _IN_HEADER.pack(
        INPUT_MAGIC,
        VERSION,
        0,
        _u32(len(v.phases), "the phase count"),
        _u32(n_deps, "the dependency count"),
        _u32(len(claims), "the claim count"),
        _u32(n_refs, "the operand count"),
        _u32(len(v.resources), "the resource count"),
        _u32(len(v.ops), "the op count"),
        _u32(sum(len(o) for o in v.ops), "the op bytes"),
        _u32(len(v.widths), "the width count"),
        bytes(24),
    )
    scope = [*v.target, *(f for pair in v.tiers for f in pair), *v.theta, *v.policy]
    out += _SCOPE.pack(*(_u32(x, "a scope value") for x in scope))
    out += b"".join(struct.pack("<I", _u32(w, "a lane width")) for w in v.widths)
    for pid, deps, cs in v.phases:
        out += _PHASE.pack(_u32(pid, "a phase id"), len(deps), len(cs))
    for _, deps, _ in v.phases:
        out += b"".join(struct.pack("<I", _u32(d, "a phase dependency")) for d in deps)
    for c in claims:
        out += _CLAIM.pack(
            _u32(c.id, "a claim id"),
            _u8(c.opcode, "an opcode"),
            _u8(c.lane, "a lane"),
            _u8(c.stride_class, "a stride class"),
            _u8(c.domain, "a domain"),
            c.hazard,
            c.verify,
            c.flags,
            c.flags2,
            _u32(c.count, "a claim count"),
            c.op,
            len(c.reads),
            len(c.writes),
            _i64(c.stride_k, "a stride multiplier"),
            _i64(c.offset, "an offset"),
            c.primary,
            0,
        )
    for c in claims:
        out += b"".join(struct.pack("<I", r) for r in c.reads + c.writes)
    for r in v.resources:
        out += bytes((r.declared, r.domain, r.access, 0))
    out += b"".join(struct.pack("<H", len(o)) for o in v.ops)
    out += b"".join(v.ops)
    out += struct.pack("<I", zlib.crc32(bytes(out)) & U32)
    return bytes(out)


class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int) -> None:
        self.data, self.pos = data, pos

    def take(self, n: int) -> bytes:
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk


def _framing(data: bytes, magic: bytes, fixed: int) -> None:
    """The framing laws, in order: truncated, magic, version, CRC, flags."""
    if len(data) < fixed + CRC_SIZE:
        raise PlannerAbiError("BCIR_ERR_TRUNCATED", "shorter than the fixed part")
    if data[:4] != magic:
        raise PlannerAbiError("BCIR_ERR_MAGIC", f"magic {data[:4]!r} is not {magic!r}")
    version, flags = struct.unpack_from("<HH", data, 4)
    if version != VERSION:
        raise PlannerAbiError("BCIR_ERR_VERSION", f"version {version} is not {VERSION}")
    crc = struct.unpack_from("<I", data, len(data) - CRC_SIZE)[0]
    if crc != zlib.crc32(data[: len(data) - CRC_SIZE]) & U32:
        raise PlannerAbiError("BCIR_ERR_CRC", "the CRC does not match")
    if flags != 0:
        raise PlannerAbiError("BCIR_ERR_RESERVED", "flags are not zero")


def input_size(n_widths, n_phases, n_deps, n_claims, n_refs, n_resources, n_ops, op_bytes) -> int:
    """The exact size of a BKPI record with these section counts."""
    return (
        INPUT_HEADER_SIZE
        + SCOPE_SIZE
        + 4 * n_widths
        + 12 * n_phases
        + 4 * n_deps
        + CLAIM_SIZE * n_claims
        + 4 * n_refs
        + RESOURCE_SIZE * n_resources
        + 2 * n_ops
        + op_bytes
        + CRC_SIZE
    )


def decode_input(data: bytes) -> PlannerInput:
    """Decode a BKPI record, applying every law in the specification's order (the C twin's
    `bcir_kp_decode_input` names the same first violation):

    1. the framing laws (`_framing`), then the header's reserved bytes;
    2. the section counts against their bounds, then the size (short: TRUNCATED, long:
       TRAILING);
    3. the scope (cacheline, the target constants, the tier factors, Theta, the policy);
    4. the lane widths; the phases (their counts add up to the header's; unique ids);
    5. the resource table; the op table (lengths add up; UTF-8; strictly ascending);
    6. each claim in order (reserved, its vocabulary, its operand count, its operands in
       first-reference order, its primary resource);
    7. every operand and resource accounted for.

    The three laws that need memory proportional to the record -- a phase id names one phase,
    a claim id one claim, and every op string is referenced -- are `check_input`'s, applied by
    the C planner before it plans (the C decoder reads in place, with no scratch)."""
    data = bytes(data)
    _framing(data, INPUT_MAGIC, INPUT_HEADER_SIZE + SCOPE_SIZE)
    (_, _, _, n_phases, n_deps, n_claims, n_refs, n_resources, n_ops, op_bytes, n_widths, res) = (
        _IN_HEADER.unpack_from(data, 0)
    )
    if any(res):
        raise PlannerAbiError("BCIR_ERR_RESERVED", "the header's reserved bytes are not zero")
    if not 1 <= n_widths <= WIDTHS_MAX:
        raise _refuse(f"{n_widths} lane widths is outside 1..{WIDTHS_MAX}")
    if (
        n_phases > CLAIMS_MAX
        or n_claims > CLAIMS_MAX
        or n_ops > CLAIMS_MAX
        or n_deps > REFS_MAX
        or n_refs > REFS_MAX
        or n_resources > REFS_MAX
        or op_bytes > OP_BYTES_MAX
    ):
        raise _refuse("a section count exceeds its declared bound")
    size = input_size(n_widths, n_phases, n_deps, n_claims, n_refs, n_resources, n_ops, op_bytes)
    if len(data) < size:
        raise PlannerAbiError("BCIR_ERR_TRUNCATED", f"{len(data)} bytes; the sections need {size}")
    if len(data) > size:
        raise PlannerAbiError("BCIR_ERR_TRAILING", f"{len(data) - size} bytes after the sections")

    scope = _SCOPE.unpack_from(data, INPUT_HEADER_SIZE)
    target, tier_f, theta, policy = scope[:8], scope[8:22], scope[22:30], scope[30:42]
    cacheline, elem_bytes, _gp, mem_unit = target[:4]
    if cacheline == 0 or cacheline & (cacheline - 1):
        raise _refuse(f"cacheline {cacheline} is not a power of two")
    if elem_bytes == 0 or mem_unit == 0:
        raise _refuse("elem_bytes and mem_unit must be positive")
    if any(x > PARAM_MAX for x in target):
        raise _refuse(f"a target constant exceeds {PARAM_MAX}")
    if any(not 1 <= f <= PARAM_MAX for f in tier_f):
        raise _refuse(f"a tier factor is outside 1..{PARAM_MAX}")
    if any(t > THETA_MAX for t in theta):
        raise _refuse(f"a Theta field exceeds {THETA_MAX}")
    if any(w > PARAM_MAX for w in policy):
        raise _refuse(f"a policy weight exceeds {PARAM_MAX}")

    r = _Reader(data, INPUT_HEADER_SIZE + SCOPE_SIZE)
    widths = struct.unpack(f"<{n_widths}I", r.take(4 * n_widths))
    if any(not 1 <= w <= PARAM_MAX for w in widths):
        raise _refuse(f"a lane width is outside 1..{PARAM_MAX}")
    phase_rows = [_PHASE.unpack(r.take(12)) for _ in range(n_phases)]
    if sum(p[1] for p in phase_rows) != n_deps or sum(p[2] for p in phase_rows) != n_claims:
        raise _refuse("the phases' dependency or claim counts do not add up to the header's")
    deps = struct.unpack(f"<{n_deps}I", r.take(4 * n_deps))
    raw_claims = [_CLAIM.unpack(r.take(CLAIM_SIZE)) for _ in range(n_claims)]
    refs = struct.unpack(f"<{n_refs}I", r.take(4 * n_refs))
    res_rows = [tuple(r.take(RESOURCE_SIZE)) for _ in range(n_resources)]
    op_lens = struct.unpack(f"<{n_ops}H", r.take(2 * n_ops))
    blob = r.take(op_bytes)

    resources = []
    for declared, domain, access, reserved in res_rows:
        if reserved:
            raise PlannerAbiError("BCIR_ERR_RESERVED", "a resource's reserved byte is not zero")
        if declared > 1 or domain > max(Domain) or access >= len(ACCESS):
            raise _refuse("a resource's declared flag, domain or addressing model is invalid")
        if not declared and (domain or access):
            raise _refuse("an undeclared resource carries a domain or addressing model")
        resources.append(InputResource(declared, domain, access))

    if sum(op_lens) != op_bytes:
        raise _refuse("the op lengths do not add up to op_bytes")
    ops = []
    pos = 0
    for i, n in enumerate(op_lens):
        op = blob[pos : pos + n]
        pos += n
        try:
            op.decode("utf-8")
        except UnicodeDecodeError:
            raise PlannerAbiError("BCIR_ERR_UTF8", f"op {i} is not valid UTF-8") from None
        if ops and not ops[-1] < op:
            raise _refuse("the op table is not strictly ascending")
        ops.append(op)

    claims = []
    next_resource = 0
    ref_pos = 0
    for row in raw_claims:
        (cid, opcode, lane, sc, domain, hazard, verify, flags, flags2, count, op, n_rd, n_wr,
         stride_k, offset, primary, reserved) = row  # fmt: skip
        if reserved:
            raise PlannerAbiError("BCIR_ERR_RESERVED", f"claim {cid}: a reserved field is not zero")
        if (
            opcode > max(Opcode)
            or lane > max(Lane)
            or sc > max(StrideClass)
            or domain > max(Domain)
            or hazard >= len(HAZARDS)
            or verify >= len(CONTRACTS)
            or flags2 & ~F2_PRECISION
            or op >= n_ops
        ):
            raise _refuse(f"claim {cid}: a field is outside its vocabulary")
        if n_rd + n_wr > OPERANDS_MAX:
            raise _refuse(f"claim {cid}: {n_rd + n_wr} operands exceed {OPERANDS_MAX}")
        if n_rd + n_wr > n_refs - ref_pos:
            raise _refuse("the claims reference more operands than the record carries")
        operands = refs[ref_pos : ref_pos + n_rd + n_wr]
        ref_pos += n_rd + n_wr
        for x in operands:
            if x > next_resource or x >= n_resources:
                raise _refuse(f"claim {cid}: operand index {x} is not in first-reference order")
            if x == next_resource:
                next_resource += 1
        if primary != NO_RESOURCE:
            if n_rd or primary > next_resource or primary >= n_resources:
                raise _refuse(f"claim {cid}: a primary resource the planner does not read")
            if not resources[primary].declared:
                raise _refuse(f"claim {cid}: an undeclared primary resource")
            if primary == next_resource:
                next_resource += 1
        claims.append(
            InputClaim(
                cid,
                opcode,
                lane,
                sc,
                domain,
                hazard,
                verify,
                flags,
                flags2,
                count,
                op,
                tuple(operands[:n_rd]),
                tuple(operands[n_rd:]),
                stride_k,
                offset,
                primary,
            )  # fmt: skip
        )
    if ref_pos != n_refs:
        raise _refuse("the claims reference fewer operands than the record carries")
    if next_resource != n_resources:
        raise _refuse("a resource entry is never referenced")

    phases = []
    dpos = cpos = 0
    for pid, nd, nc in phase_rows:
        phases.append((pid, tuple(deps[dpos : dpos + nd]), tuple(claims[cpos : cpos + nc])))
        dpos += nd
        cpos += nc
    return PlannerInput(
        target=tuple(target),
        tiers=tuple((tier_f[2 * i], tier_f[2 * i + 1]) for i in range(7)),
        theta=tuple(theta),
        policy=tuple(policy),
        widths=tuple(widths),
        phases=tuple(phases),
        resources=tuple(resources),
        ops=tuple(ops),
    )


def check_input(value: PlannerInput) -> None:
    """The laws a record's decoder cannot apply without memory, in the C planner's order
    (`bcir_kp_plan` refuses them before it plans): a phase id names one phase, a claim id
    names one claim, and every op string in the table is referenced."""
    phase_ids = [pid for pid, _, _ in value.phases]
    if len(set(phase_ids)) != len(phase_ids):
        raise _refuse("a phase id names two phases")
    claim_ids = [c.id for _, _, cs in value.phases for c in cs]
    if len(set(claim_ids)) != len(claim_ids):
        raise _refuse("a claim id names two claims")
    if len({c.op for _, _, cs in value.phases for c in cs}) != len(value.ops):
        raise _refuse("an op string is never referenced")


# --- BKPR: the realization -------------------------------------------------------------------


def _name_code(name: str, width: int) -> int:
    if name.startswith("vec") and name != "vec":
        if name != f"vec{width}":
            raise _refuse(f"realization {name!r} does not name its width {width}")
        return NAME_VEC
    if name not in NAMES or name == "vec":
        raise _refuse(f"realization name {name!r} is not one the planner offers")
    return NAMES.index(name)


def encode_realization(result) -> bytes:
    """A `RealizationResult` as a BKPR record: the plan's bytes. `BCIR_ERR_OVERFLOW` when a
    cost the record carries exceeds 2**63 - 1 -- the C planner refuses the same plan."""
    steps = []
    for s in result.steps:
        c = s.candidate
        base = tuple(c.base.v)
        values = (s.cost, *base)
        if any(not isinstance(x, int) or x < 0 for x in values):
            raise _refuse(f"claim {s.claim_id}: a cost is negative or not an integer")
        if any(x > I64_MAX for x in values):
            raise PlannerAbiError(
                "BCIR_ERR_OVERFLOW", f"claim {s.claim_id}: a cost exceeds 2**63-1"
            )
        steps.append(
            RealizedStep(
                _u32(s.claim_id, "a claim id"),
                _u32(s.phase_id, "a phase id"),
                _u32(c.width, "a width"),
                int(c.lane),
                _name_code(c.name, c.width),
                s.cost,
                base,
            )
        )
    if not isinstance(result.score, int) or result.score < 0:
        raise _refuse("the score is negative or not an integer")
    if result.score > I64_MAX:
        raise PlannerAbiError("BCIR_ERR_OVERFLOW", "the score exceeds 2**63-1")
    value = Realization(result.score, tuple(steps))
    data = _pack_realization(value)
    if decode_realization(data) != value:
        raise _refuse("the realization record does not round-trip")
    return data


def _pack_realization(v: Realization) -> bytes:
    out = bytearray(_R_HEADER.pack(REALIZATION_MAGIC, VERSION, 0, len(v.steps), 0, v.score, 0))
    for s in v.steps:
        out += _STEP.pack(s.claim_id, s.phase_id, s.width, s.lane, s.name, 0, s.cost, *s.base)
    out += struct.pack("<I", zlib.crc32(bytes(out)) & U32)
    return bytes(out)


def decode_realization(data: bytes) -> Realization:
    """Decode a BKPR record (the C twin's `bcir_kp_decode_realization` applies the same laws,
    in the same order)."""
    data = bytes(data)
    _framing(data, REALIZATION_MAGIC, REALIZATION_HEADER_SIZE)
    _, _, _, n_steps, reserved, score, reserved2 = _R_HEADER.unpack_from(data, 0)
    if reserved or reserved2:
        raise PlannerAbiError("BCIR_ERR_RESERVED", "the header's reserved fields are not zero")
    size = REALIZATION_HEADER_SIZE + STEP_SIZE * n_steps + CRC_SIZE
    if len(data) < size:
        raise PlannerAbiError(
            "BCIR_ERR_TRUNCATED", f"{len(data)} bytes; {n_steps} steps need {size}"
        )
    if len(data) > size:
        raise PlannerAbiError("BCIR_ERR_TRAILING", f"{len(data) - size} bytes after the steps")
    if score > I64_MAX:
        raise PlannerAbiError("BCIR_ERR_OVERFLOW", "the score exceeds 2**63-1")
    steps = []
    total = 0
    for i in range(n_steps):
        row = _STEP.unpack_from(data, REALIZATION_HEADER_SIZE + STEP_SIZE * i)
        claim_id, phase_id, width, lane, name, res, cost = row[:7]
        base = tuple(row[7:])
        if res:
            raise PlannerAbiError("BCIR_ERR_RESERVED", f"step {i}: a reserved field is not zero")
        if cost > I64_MAX or any(b > I64_MAX for b in base):
            raise PlannerAbiError("BCIR_ERR_OVERFLOW", f"step {i}: a cost exceeds 2**63-1")
        if lane > max(Lane) or name >= len(NAMES) or width == 0:
            raise _refuse(f"step {i}: lane, name or width is outside its vocabulary")
        if (name == NAME_VEC and width < 2) or (name in _WIDTH_ONE and width != 1):
            raise _refuse(f"step {i}: realization {NAMES[name]!r} cannot have width {width}")
        total += cost
        steps.append(RealizedStep(claim_id, phase_id, width, lane, name, cost, base))
    if total != score:
        raise _refuse(f"the score {score} is not the sum of the step costs {total}")
    return Realization(score, tuple(steps))
