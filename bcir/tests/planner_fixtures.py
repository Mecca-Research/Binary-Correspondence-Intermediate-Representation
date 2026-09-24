"""The G17 fixtures and grader (S4-A): the compact planner, held to the pre-G17 planner and to its
native twin, byte for byte.

One function, `measure`, grades both rows over the declared corpora. The tests, the harness
(`tools/perf/gemplus_baseline.py --group kplan`), `tools/c/check_planner.py` and
`tools/c/check_runtime.sh` all call it:

    planner.parity              (case, comparison) pairs that differ: the compact planner against
                                the reference (`realize_reference`) -- the RealizationResult, its
                                BKPR bytes, and for the fixed corpus the ExecutionPlanV1 bytes
                                built from it -- and the native planner (`bcir_kplan.c`) against
                                the compact one: the BKPR bytes, or the same refusal
    planner.malformed.accepted  (malformed record, rail) pairs not refused with the declared
                                status: one BKPI variant per wire law and per planning law, one
                                BKPR variant per wire law
    planner.r9.misjudged        (plan, case) pairs R9 misjudges through the compact offer: the
                                planner's own plan with an R8/R9 diagnostic, a forgery of any
                                field a step carries without one, or a verdict that raised

The corpora are the whole fixed corpus (`examples.PROGRAMS`, the audit's matmul fixture and
`coverage_modules`, one module per construct the random generator may never draw) under every
target, Theta and policy, and a seeded sample of `differential.gen_module` under every target.
"""

from __future__ import annotations

import os
import random
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass, replace

from bcir.abi import planner_abi as pa

ROWS = ("planner.parity", "planner.malformed.accepted", "planner.r9.misjudged")

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
C_DIR = os.path.join(_ROOT, "runtime", "c")
C_UNITS = ("bcir_kplan.c", "bcir_runtime.c")
HARNESS = "test_kplan.c"

TARGET_NAMES = ("x86_avx2", "x86_avx512", "arm64_neon", "arm64_sve", "riscv_rvv", "nvidia_ptx")
GENERATED = 240  # modules drawn from differential.gen_module
GENERATED_SEED = 17


def compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def build_harness(tmp: str, extra_flags=(), name: str = "test_kplan") -> str | None:
    """Compile runtime/c/test_kplan.c against the freestanding planner. None without a compiler
    (the quick tier hides one on purpose) and in an installed package (the wheel does not ship
    runtime/c); in a source checkout a visible compiler must build it -- a missing source there is
    a failure, never a skip (L21)."""
    from .run_all import _is_source_checkout

    cc = compiler()
    if cc is None:
        return None
    if not os.path.isfile(os.path.join(C_DIR, HARNESS)):
        if _is_source_checkout():
            raise RuntimeError(f"runtime/c/{HARNESS} is missing from the checkout")
        return None
    exe = os.path.join(tmp, name)
    build = subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            *extra_flags,
            "-I",
            C_DIR,
            os.path.join(C_DIR, HARNESS),
            *[os.path.join(C_DIR, unit) for unit in C_UNITS],
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if build.returncode != 0:
        raise RuntimeError(f"planner harness build failed: {build.stderr[-2000:]}")
    return exe


# --- the corpora ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    label: str
    module: object
    h: object
    theta: object
    policy: object
    fixed: bool  # the fixed corpus also compares the ExecutionPlanV1 bytes


def targets() -> dict:
    from bcir.kbcir.cost import TargetProfile

    return {name: getattr(TargetProfile, name)() for name in TARGET_NAMES}


def _claim(cid, opcode, **kw):
    from bcir.model import Claim

    return Claim(id=cid, opcode=opcode, **kw)


def coverage_modules() -> list[tuple[str, object]]:
    """One module per construct the planner branches on that the random generator may never draw:
    every opcode, every stride class and lane, HAM addressing, the reducible gather, CSE with and
    without an intervening write and across a phase boundary, ineligible duplicates, deforestation
    and the barrier fence on either side, MMIO operands, undeclared operands, primary-only claims
    of every domain, the numeric contracts, the phase DAG's missing, self and cyclic dependencies
    and an empty phase, and counts whose costs no record can carry."""
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass
    from bcir.model.graph import Lifetime, Timing
    from bcir.model.lanes import Domain

    out: list[tuple[str, object]] = []

    def module(name, resources, phases):
        m = Module(name=name)
        for r in resources:
            m.add_resource(r)
        for ph in phases:
            m.add_phase(ph)
        return m

    ram = [Resource(rid=i, domain=Domain.RAM, shape=(64,)) for i in range(1, 9)]

    # every opcode, UNIT geometry
    claims = [
        _claim(
            100 + int(op),
            op,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(1, 2),
            wr=(3,),
            op=f"op.{op.name.lower()}",
        )
        for op in Opcode
    ]
    out.append(("coverage.opcodes", module("cov_opcodes", ram, [Phase(0, (), claims)])))

    # every stride class x the lanes a claim may declare, and the strided multiplier's clamp
    claims = []
    cid = 200
    for sc in StrideClass:
        for lane in (Lane.U, Lane.UX, Lane.T, Lane.GGG, Lane.H):
            for k in (0, 3, 1000):
                claims.append(
                    _claim(
                        cid,
                        Opcode.ADD,
                        lane=lane,
                        stride_class=sc,
                        stride_k=k,
                        count=100,
                        rd=(4,),
                        wr=(5,),
                        op="vector.add",
                    )
                )
                cid += 1
    out.append(("coverage.geometry", module("cov_geometry", ram, [Phase(0, (), claims)])))

    # HAM addressing: gp = max(1, bit_length(n - 1)) for n = max(1, count)
    ham = [
        Resource(rid=11, domain=Domain.RAM, shape=(64,), access="ham"),
        Resource(rid=12, domain=Domain.HBM, shape=(64,)),
    ]
    claims = [
        _claim(
            300 + i,
            Opcode.GGG_LOAD,
            lane=Lane.GGG,
            stride_class=sc,
            count=count,
            rd=(11,),
            wr=(12,),
            op="histogram.scatter",
        )
        for i, (sc, count) in enumerate(
            (s, c)
            for s in (StrideClass.RANDOM, StrideClass.STRIDED, StrideClass.CACHELINE)
            for c in (0, 1, 2, 3, 1000, 65536)
        )
    ]
    out.append(("coverage.ham", module("cov_ham", ham, [Phase(0, (), claims)])))

    # the reducible-permutation gather, and an atomic that also names it
    claims = [
        _claim(
            400,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.RANDOM,
            count=512,
            rd=(1,),
            wr=(2,),
            op="reduce.gather",
        ),  # fmt: skip
        _claim(
            401,
            Opcode.ATOMIC_ADD,
            lane=Lane.A,
            stride_class=StrideClass.UNIT,
            count=8,
            rd=(2,),
            wr=(2,),
            op="reduce.gather",
            hazard="atomic",
        ),  # fmt: skip
    ]
    out.append(("coverage.reduce_gather", module("cov_rg", ram, [Phase(0, (), claims)])))

    # CSE: an identical pair, a pair split by a write, an ineligible (volatile) pair, and the
    # same pair across a phase boundary (no credit)
    def dup(cid, **kw):
        base = dict(lane=Lane.U, stride_class=StrideClass.UNIT, count=256, rd=(1, 2), wr=(3,),
                    op="vector.mul")  # fmt: skip
        base.update(kw)
        return _claim(cid, Opcode.MUL, **base)

    p0 = [
        dup(500),
        dup(501, wr=(4,)),  # the same value: CSE
        _claim(
            502,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=256,
            rd=(5,),
            wr=(1,),
            op="vector.add",
        ),  # fmt: skip  -- rewrites rid 1
        dup(503, wr=(6,)),  # rid 1 at a new version: no credit
        dup(504, wr=(7,), volatile=True),
        dup(505, wr=(8,), volatile=True),
        dup(506, wr=(7,), imm=(3,)),
        dup(507, wr=(8,), dynamic=True),
    ]
    # a seed read at version 1, then a SECOND rewrite of rid 1: the duplicate reads version 2
    # and must not take the seed's value (a counter that stops at 1 would hand it the credit)
    rewrite = dict(lane=Lane.U, stride_class=StrideClass.UNIT, count=256, rd=(5,), wr=(1,),
                   op="vector.add")  # fmt: skip
    p0 += [
        _claim(520, Opcode.ADD, **rewrite),
        dup(521, wr=(6,)),
        _claim(522, Opcode.ADD, **rewrite),
        dup(523, wr=(7,)),
    ]
    p1 = [dup(510, wr=(4,)), dup(511, wr=(5,))]
    out.append(
        (
            "coverage.cse",
            module("cov_cse", ram, [Phase(0, (), p0), Phase(1, (0,), p1)]),
        )
    )

    # deforestation, and the barrier fence on either side of it
    claims = [
        _claim(
            600,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(1,),
            wr=(2,),
            op="a",
        ),  # fmt: skip
        _claim(
            601,
            Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(2,),
            wr=(3,),
            op="b",
        ),  # fmt: skip  -- consumes 2: deforested
        _claim(
            602,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(3,),
            wr=(4,),
            op="c",
            hazard="barriered",
        ),  # fmt: skip  -- a barriered consumer
        _claim(
            603,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(4,),
            wr=(5,),
            op="d",
        ),  # fmt: skip  -- its producer was barriered: fenced
        _claim(
            604,
            Opcode.BARRIER,
            lane=Lane.H,
            stride_class=StrideClass.SCALAR,
            count=1,
            rd=(5,),
            wr=(6,),
            op="fence",
            hazard="barriered",
        ),  # fmt: skip
        _claim(
            605,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(6, 5),
            wr=(7,),
            op="e",
        ),  # fmt: skip
    ]
    out.append(("coverage.deforest", module("cov_deforest", ram, [Phase(0, (), claims)])))

    # MMIO operands (CSE-ineligible), undeclared operands, primary-only claims of every domain
    mixed = [
        Resource(rid=21, domain=Domain.MMIO, shape=(4,)),
        Resource(rid=22, domain=Domain.RAM, shape=(64,)),
        *[
            Resource(rid=30 + int(d), domain=d, shape=(64,), access="ham" if d == 2 else "flat")
            for d in Domain
        ],
    ]
    claims = [
        _claim(
            700,
            Opcode.LOAD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=4,
            rd=(21,),
            wr=(22,),
            op="mmio.read",
            domain=Domain.MMIO,
            hazard="barriered",
        ),  # fmt: skip
        _claim(
            701,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(22, 21),
            wr=(99,),
            op="x",
        ),  # fmt: skip  -- 99 is undeclared
        _claim(
            702,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(22, 21),
            wr=(98,),
            op="x",
        ),  # fmt: skip
        _claim(
            703,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=64,
            rd=(97,),
            wr=(22,),
            op="y",
        ),  # fmt: skip  -- an undeclared first read
        *[
            _claim(
                710 + int(d),
                Opcode.STORE,
                lane=Lane.U,
                stride_class=StrideClass.RANDOM,
                count=128,
                rd=(),
                wr=(22,),
                op="z",
                primary_rid=30 + int(d),
            )  # fmt: skip
            for d in Domain
        ],
        _claim(
            720,
            Opcode.STORE,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=128,
            rd=(),
            wr=(22,),
            op="z",
            primary_rid=12345,
        ),  # fmt: skip  -- undeclared primary
        _claim(
            721,
            Opcode.STORE,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=128,
            rd=(),
            wr=(),
            op="z",
        ),  # fmt: skip  -- no operand at all
        _claim(
            722,
            Opcode.STORE,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=128,
            rd=(22,),
            wr=(31,),
            op="z",
            primary_rid=35,
        ),  # fmt: skip  -- reads win
    ]
    out.append(("coverage.resources", module("cov_resources", mixed, [Phase(0, (), claims)])))

    # the numeric and metadata contracts the CSE exclusions read, and the priced contracts
    def meta(cid, **kw):
        base = dict(lane=Lane.U, stride_class=StrideClass.UNIT, count=128, rd=(1,), wr=(2,),
                    op="vector.add")  # fmt: skip
        base.update(kw)
        return _claim(cid, Opcode.ADD, **base)

    claims = [
        meta(800),
        meta(801, callee_sig="i32(i32)"),
        meta(802, timing=Timing(latency_cycles=3)),
        meta(803, lifetime=Lifetime()),
        meta(804, tolerance_ulp=2),
        meta(805, quantized_bits=4),
        meta(806, precision="compensated"),
        meta(807, verify="exact"),
        meta(808, verify="hash"),
        meta(809, verify="none"),
        meta(810, dynamic=True),
        meta(811, offset=16),
        meta(812, count=0),
        meta(813),
    ]
    out.append(("coverage.contracts", module("cov_contracts", ram, [Phase(0, (), claims)])))

    # the phase DAG: declared out of dependency order, a missing dependency, a self dependency,
    # a cycle, and an empty phase
    def one(cid):
        return [_claim(cid, Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=32,
                       rd=(1,), wr=(2,), op="p")]  # fmt: skip

    phases = [
        Phase(5, (3, 42), one(900)),  # 42 names no phase
        Phase(3, (3,), one(901)),  # itself
        Phase(7, (8,), one(902)),
        Phase(8, (7,), one(903)),  # a cycle
        Phase(9, (), []),  # empty
        Phase(1, (9, 5), one(904)),
    ]
    out.append(("coverage.phases", module("cov_phases", ram, phases)))

    # costs no record can carry: the base, a step and the score each over 2**63 - 1
    big = [
        Resource(rid=40, domain=Domain.NVM, shape=(1,), access="flat"),
        Resource(rid=41, domain=Domain.NVM, shape=(1,)),
    ]
    claims = [
        _claim(
            1000,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.STRIDED,
            stride_k=1 << 20,
            count=(1 << 32) - 1,
            rd=(40,) * 60,
            wr=(41,) * 60,
            op="huge",
        )  # fmt: skip
    ]
    out.append(("coverage.overflow", module("cov_overflow", big, [Phase(0, (), claims)])))
    claims = [
        _claim(
            1100 + i,
            Opcode.T_MACC,
            lane=Lane.T,
            stride_class=StrideClass.TILE,
            count=(1 << 32) - 1,
            rd=(40,),
            wr=(41,),
            op="large",
        )  # fmt: skip
        for i in range(3)
    ]
    out.append(("coverage.large", module("cov_large", big, [Phase(0, (), claims)])))
    return out


def custom_scopes():
    """Targets, Theta and policies at the edges the factories never reach: one lane width, a
    declaration order that is not ascending (with a duplicate), a hierarchy without most tiers,
    the widest constants the domain admits, the hottest Theta and the heaviest policy."""
    from bcir.kbcir.cost import MemoryHierarchy, TargetProfile, Theta, Tier
    from bcir.kbcir.weights import PERF, Policy

    scalar_only = TargetProfile(name="scalar", triple="x86_64-scalar", lane_widths=(1,))
    unsorted = TargetProfile(name="unsorted", triple="aarch64-x", lane_widths=(8, 1, 8, 4))
    sparse = TargetProfile(
        name="sparse",
        triple="riscv64-x",
        lane_widths=(1, 32),
        mem=MemoryHierarchy((Tier("DRAM", 200, 256, 256), Tier("SSD", 9, 4096, 65536))),
    )
    extreme = TargetProfile(
        name="extreme",
        triple="x86_64-extreme",
        cacheline=1 << 16,
        elem_bytes=1,
        lane_widths=(1, 16, 1 << 16),
        gather_penalty=1 << 16,
        mem_unit=1 << 16,
        base_overhead=1 << 16,
        thermal_density=1 << 16,
        power_density=1 << 16,
        per_op_heat=1 << 16,
    )
    heavy = Policy("heavy", (1 << 16,) * 12)
    zero = Policy("zero", (0,) * 12)
    fire = Theta(thermal=100, power=100, mem_pressure=100, contention=100, noise=100, wear=100,
                 utilization=100, voltage=100)  # fmt: skip
    return [
        ("scalar_only", scalar_only, Theta.hot(), PERF),
        ("unsorted", unsorted, Theta.hot(), PERF),
        ("sparse", sparse, Theta.mem_bound(), heavy),
        ("extreme", extreme, fire, heavy),
        ("extreme-zero", extreme, Theta.cool(), zero),
    ]


def wide_path_case() -> Case:
    """A plan whose losing paths weigh more than 2**64 while the winning one fits a u64: the
    scalar realization of each claim prices its access and heat per element, the 65536-wide one
    per issue, under weights that magnify both. A native planner that wrapped a path weight would
    pick a scalar realization here; one that saturated would tie it with the winner."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.weights import Policy
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass

    m = Module(name="wide_path")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_resource(Resource(rid=2, shape=(64,)))
    claims = [
        _claim(
            1200 + i,
            Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=(1 << 32) - 1,
            rd=(1,),
            wr=(2,),
            op=f"w{i}",
        )  # fmt: skip
        for i in range(4)
    ]
    m.add_phase(Phase(0, (), claims))
    h = TargetProfile(
        name="wide",
        triple="x86_64-wide",
        lane_widths=(1, 16, 1 << 16),
        mem_unit=1,
        base_overhead=1 << 16,
        thermal_density=1 << 16,
        power_density=1 << 16,
        per_op_heat=1 << 16,
    )
    return Case("wide_path", m, h, Theta.hot(), Policy("magnify", (1 << 16,) * 12), True)


def carry_case() -> Case:
    """A plan whose one edge crosses 2**64 only by ADDING terms that each fit a u64: the heat of
    a one-wide tile, per element, under the heaviest weights (the count is the one where the sum
    lands just past 2**64). The plan must be refused -- its score cannot be carried -- and a
    planner whose 128-bit addition dropped its carry would report the wrapped sum, which fits:
    the case exists so that defect is a divergence (tools/c/check_runtime.sh injects it)."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.weights import Policy
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass

    m = Module(name="carry")
    m.add_resource(Resource(rid=1, shape=(64,)))
    m.add_resource(Resource(rid=2, shape=(64,)))
    m.add_phase(
        Phase(
            0,
            (),
            [
                _claim(
                    1300,
                    Opcode.T_MACC,
                    lane=Lane.T,
                    stride_class=StrideClass.TILE,
                    count=2281596933,
                    rd=(1,),
                    wr=(2,),
                    op="carry",
                )
            ],
        )  # fmt: skip
    )
    h = TargetProfile(
        name="carry",
        triple="x86_64-carry",
        lane_widths=(1,),
        mem_unit=1,
        base_overhead=1,
        thermal_density=0,
        power_density=0,
        per_op_heat=1 << 16,
    )
    return Case("carry", m, h, Theta.cool(), Policy("carry", (1 << 16,) * 12), True)


def corpus_cases() -> list[Case]:
    from bcir.examples import PROGRAMS, matmul_tiled
    from bcir.kbcir.differential import THETAS, gen_module
    from bcir.kbcir.weights import POLICIES

    tg = targets()
    fixed = [(name, build()) for name, build in sorted(PROGRAMS.items())]
    fixed.append(("audit.kbcir-streampack.1", matmul_tiled(n=32, tile=8)))
    fixed += coverage_modules()
    cases = [
        Case(f"{name}/{tn}/{thn}/{pn}", m, h, th, pol, True)
        for name, m in fixed
        for tn, h in tg.items()
        for thn, th in THETAS.items()
        for pn, pol in POLICIES.items()
    ]
    for sname, h, th, pol in custom_scopes():
        cases += [Case(f"{name}/{sname}", m, h, th, pol, True) for name, m in fixed]
    cases.append(wide_path_case())
    cases.append(carry_case())
    rng = random.Random(GENERATED_SEED)
    thetas, policies = list(THETAS.items()), list(POLICIES.items())
    for k in range(GENERATED):
        m = gen_module(rng)
        thn, th = thetas[k % len(thetas)]
        pn, pol = policies[(k // len(thetas)) % len(policies)]
        cases += [Case(f"gen{k}/{tn}/{thn}/{pn}", m, h, th, pol, False) for tn, h in tg.items()]
    return cases


# --- the Python rails ---------------------------------------------------------------------------


def realization_bytes(result) -> tuple[str, bytes]:
    """(status, BKPR bytes): "BCIR_OK" and the record, or the encoder's refusal and nothing."""
    try:
        return "BCIR_OK", pa.encode_realization(result)
    except pa.PlannerAbiError as exc:
        return exc.status, b""


def plan_bytes(module, result, h) -> bytes:
    """The ExecutionPlanV1 bytes of a realization (G11): the plan as the rest of the tree reads it."""
    from bcir.abi.execution_plan_abi import encode_plan
    from bcir.gem.execution_plan import plan_from_realization

    return encode_plan(plan_from_realization(module, result, h, "eft", plan="plan0"))


def _steps(result) -> list:
    return [(s.claim_id, s.phase_id, s.candidate, s.cost) for s in result.steps]


# --- the malformed corpora ---------------------------------------------------------------------


def _reseal(data: bytes) -> bytes:
    body = data[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def _put(data: bytes, offset: int, fmt: str, value) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    return _reseal(bytes(out))


def _seed_input() -> tuple[bytes, pa.PlannerInput]:
    """A valid record with two phases, two ops, operands and a primary resource: every law has a
    field to break."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass
    from bcir.model.lanes import Domain

    m = Module(name="seed")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(64,)))
    m.add_resource(Resource(rid=2, domain=Domain.HBM, shape=(64,), access="ham"))
    m.add_resource(Resource(rid=3, domain=Domain.VRAM, shape=(64,)))  # an operand, nothing more
    m.add_resource(Resource(rid=4, domain=Domain.CXL, shape=(64,)))
    m.add_phase(
        Phase(
            0,
            (),
            [
                _claim(
                    1,
                    Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(1,),
                    wr=(2,),
                    op="a",
                ),  # fmt: skip
                _claim(
                    2,
                    Opcode.STORE,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(),
                    wr=(1,),
                    op="b",
                    primary_rid=2,
                ),  # fmt: skip
            ],
        )
    )
    m.add_phase(
        Phase(
            1,
            (0,),
            [
                _claim(
                    3,
                    Opcode.MUL,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=8,
                    rd=(2, 1),
                    wr=(3, 4),
                    op="a",
                )
            ],
        )  # fmt: skip
    )
    data = pa.encode_input(m, TargetProfile.x86_avx2(), Theta.mem_bound())
    return data, pa.decode_input(data)


def malformed_inputs() -> list[tuple[str, bytes, str]]:
    """(label, bytes, the status both rails must refuse with) -- one per BKPI law, in the
    specification's order, each resealed so the record reaches the law it breaks."""
    good, v = _seed_input()
    hdr = pa.INPUT_HEADER_SIZE
    sc = hdr  # the scope
    widths = hdr + pa.SCOPE_SIZE
    phases = widths + 4 * len(v.widths)
    claims = phases + 12 * len(v.phases) + 4 * sum(len(d) for _, d, _ in v.phases)
    n_claims = sum(len(cs) for _, _, cs in v.phases)
    refs = claims + pa.CLAIM_SIZE * n_claims
    n_refs = sum(len(c.reads) + len(c.writes) for _, _, cs in v.phases for c in cs)
    resources = refs + 4 * n_refs
    op_lens = resources + pa.RESOURCE_SIZE * len(v.resources)
    ops = op_lens + 2 * len(v.ops)
    c2 = claims + pa.CLAIM_SIZE  # the second claim (the primary-only one)
    P = "BCIR_ERR_PLANNER"
    out = [
        ("short", good[:100], "BCIR_ERR_TRUNCATED"),
        ("magic", good[:3] + b"X" + good[4:], "BCIR_ERR_MAGIC"),
        ("version", _put(good, 4, "<H", 1), "BCIR_ERR_VERSION"),
        ("crc", good[:-1] + bytes([good[-1] ^ 1]), "BCIR_ERR_CRC"),
        ("flags", _put(good, 6, "<H", 1), "BCIR_ERR_RESERVED"),
        ("header-pad", _put(good, 50, "<B", 1), "BCIR_ERR_RESERVED"),
        ("no-widths", _put(good, 36, "<I", 0), P),
        ("too-many-widths", _put(good, 36, "<I", 17), P),
        ("claims-bound", _put(good, 16, "<I", (1 << 24) + 1), P),
        ("refs-bound", _put(good, 20, "<I", (1 << 26) + 1), P),
        ("op-bytes-bound", _put(good, 32, "<I", (1 << 24) + 1), P),
        ("size-short", _reseal(good[:ops] + good[ops + 1 :]), "BCIR_ERR_TRUNCATED"),
        ("size-long", _reseal(good[:-4] + b"\0" + good[-4:]), "BCIR_ERR_TRAILING"),
        ("cacheline-zero", _put(good, sc, "<I", 0), P),
        ("cacheline-not-pow2", _put(good, sc, "<I", 48), P),
        ("elem-zero", _put(good, sc + 4, "<I", 0), P),
        ("mem-unit-zero", _put(good, sc + 12, "<I", 0), P),
        ("cacheline-over", _put(good, sc, "<I", 1 << 17), P),
        ("target-over", _put(good, sc + 20, "<I", (1 << 16) + 1), P),
        ("tier-zero", _put(good, sc + 32, "<I", 0), P),
        ("tier-over", _put(good, sc + 36, "<I", (1 << 16) + 1), P),
        ("theta-over", _put(good, sc + 88, "<I", 101), P),
        ("policy-over", _put(good, sc + 120, "<I", (1 << 16) + 1), P),
        ("width-zero", _put(good, widths, "<I", 0), P),
        ("width-over", _put(good, widths, "<I", (1 << 16) + 1), P),
        ("phase-claims-sum", _put(good, phases + 8, "<I", 3), P),
        ("resource-pad", _put(good, resources + 3, "<B", 1), "BCIR_ERR_RESERVED"),
        ("resource-declared", _put(good, resources, "<B", 2), P),
        ("resource-domain", _put(good, resources + 1, "<B", 6), P),
        ("resource-access", _put(good, resources + 2, "<B", 2), P),
        # rid 3 is a write operand only: its declaration is read by nothing but this law
        ("resource-undeclared-domain", _put(good, resources + 8, "<B", 0), P),
        ("op-lengths", _put(good, op_lens, "<H", 2), P),
        ("op-utf8", _put(good, ops, "<B", 0xFF), "BCIR_ERR_UTF8"),
        ("ops-descending", _put(good, ops, "<B", ord("c")), P),
        ("claim-pad", _put(good, claims + 44, "<I", 1), "BCIR_ERR_RESERVED"),
        ("claim-opcode", _put(good, claims + 4, "<B", 18), P),
        ("claim-lane", _put(good, claims + 5, "<B", 6), P),
        ("claim-stride", _put(good, claims + 6, "<B", 6), P),
        ("claim-domain", _put(good, claims + 7, "<B", 6), P),
        ("claim-hazard", _put(good, claims + 8, "<B", 3), P),
        ("claim-verify", _put(good, claims + 9, "<B", 4), P),
        ("claim-flags2", _put(good, claims + 11, "<B", 2), P),
        ("claim-op", _put(good, claims + 16, "<I", 2), P),
        ("claim-operands", _put(good, claims + 20, "<H", 300), P),
        ("claim-operand-order", _put(good, refs, "<I", 1), P),
        # one operand fewer than the claims name, and one more: the header and the size agree
        (
            "operands-short",
            _put(good[: resources - 4] + good[resources:], 20, "<I", n_refs - 1),
            P,
        ),
        (
            "operands-long",
            _put(good[:resources] + bytes(4) + good[resources:], 20, "<I", n_refs + 1),
            P,
        ),
        ("primary-with-reads", _put(good, claims + 40, "<I", 0), P),
        # claim 2 names rid 4 (index 3) while index 2 is the next unreferenced one
        ("primary-out-of-order", _put(good, c2 + 40, "<I", 3), P),
    ]
    # the laws a value-level edit reaches (the packer writes what the decoder must refuse)
    extra_resource = replace(v, resources=v.resources + (pa.InputResource(1, 0, 0),))
    out.append(("resource-unreferenced", _pack(extra_resource), P))
    undeclared = replace(v, resources=(v.resources[0], pa.InputResource(0, 0, 0), *v.resources[2:]))
    out.append(("primary-undeclared", _pack(undeclared), P))
    pid = v.phases
    out.append(
        ("phase-duplicate", _pack(replace(v, phases=(pid[0], (pid[0][0],) + pid[1][1:]))), P)
    )
    c0 = pid[0][2][0]
    dup_claim = (pid[0][0], pid[0][1], (c0, replace(pid[0][2][1], id=c0.id)))
    out.append(("claim-duplicate", _pack(replace(v, phases=(dup_claim, pid[1]))), P))
    out.append(("op-unreferenced", _pack(replace(v, ops=v.ops + (b"zz",))), P))
    return out


def _pack(value: pa.PlannerInput) -> bytes:
    return pa._pack_input(value)


def malformed_realizations() -> list[tuple[str, bytes, str]]:
    """(label, bytes, status) -- one per BKPR law, in order, each resealed."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize

    module = coverage_modules()[0][1]
    good = pa.encode_realization(optimize(module, TargetProfile.x86_avx512(), Theta.cool()))
    h = pa.REALIZATION_HEADER_SIZE
    step = h  # the first step
    P = "BCIR_ERR_PLANNER"
    big = (1 << 63) + 5
    score = struct.unpack_from("<Q", good, 16)[0]
    cost0 = struct.unpack_from("<Q", good, step + 16)[0]
    return [
        ("short", good[:20], "BCIR_ERR_TRUNCATED"),
        ("magic", b"BKPX" + good[4:], "BCIR_ERR_MAGIC"),
        ("version", _put(good, 4, "<H", 1), "BCIR_ERR_VERSION"),
        ("crc", good[:-1] + bytes([good[-1] ^ 1]), "BCIR_ERR_CRC"),
        ("flags", _put(good, 6, "<H", 1), "BCIR_ERR_RESERVED"),
        ("header-pad", _put(good, 12, "<I", 1), "BCIR_ERR_RESERVED"),
        ("header-pad2", _put(good, 24, "<Q", 1), "BCIR_ERR_RESERVED"),
        ("size-short", _reseal(good[:-5] + good[-4:]), "BCIR_ERR_TRUNCATED"),
        ("size-long", _reseal(good[:-4] + b"\0" + good[-4:]), "BCIR_ERR_TRAILING"),
        ("score-overflow", _put(good, 16, "<Q", big), "BCIR_ERR_OVERFLOW"),
        ("step-pad", _put(good, step + 14, "<H", 1), "BCIR_ERR_RESERVED"),
        ("cost-overflow", _put(good, step + 16, "<Q", big), "BCIR_ERR_OVERFLOW"),
        ("base-overflow", _put(good, step + 24 + 8 * 11, "<Q", big), "BCIR_ERR_OVERFLOW"),
        ("lane", _put(good, step + 12, "<B", 6), P),
        ("name", _put(good, step + 13, "<B", 10), P),
        ("width-zero", _put(good, step + 8, "<I", 0), P),
        ("scalar-wide", _put(_put(good, step + 13, "<B", 5), step + 8, "<I", 2), P),
        ("vec-narrow", _put(_put(good, step + 13, "<B", 6), step + 8, "<I", 1), P),
        ("score-sum", _put(good, 16, "<Q", score + 1), P),
        ("cost-sum", _put(good, step + 16, "<Q", cost0 + 1), P),
    ]


def python_input_status(data: bytes) -> str:
    try:
        pa.check_input(pa.decode_input(data))
    except pa.PlannerAbiError as exc:
        return exc.status
    return "BCIR_OK"


def python_realization_status(data: bytes) -> str:
    try:
        pa.decode_realization(data)
    except pa.PlannerAbiError as exc:
        return exc.status
    return "BCIR_OK"


# --- the C rail ---------------------------------------------------------------------------------

_STATUS_NAMES: dict[int, str] = {}


def status_name(code: int) -> str:
    """A `bcir_status` value's name, read out of runtime/c/bcir_runtime.h (never a mirrored list)."""
    if not _STATUS_NAMES:
        import re

        with open(os.path.join(C_DIR, "bcir_runtime.h"), encoding="utf-8") as f:
            text = f.read()
        for name, value in re.findall(r"\b(BCIR_(?:OK|ERR_[A-Z0-9_]+))\s*=\s*(\d+)", text):
            _STATUS_NAMES[int(value)] = name
    return _STATUS_NAMES.get(code, f"BCIR_STATUS_{code}")


def c_batch(exe: str, tmp: str, records: list[bytes]) -> list[tuple[str, bytes]]:
    """Plan every record in one process: (status name, BKPR bytes) each."""
    src, dst = os.path.join(tmp, "kplan.in"), os.path.join(tmp, "kplan.out")
    with open(src, "wb") as f:
        for r in records:
            f.write(struct.pack("<I", len(r)) + r)
    run = subprocess.run([exe, "--batch", src, dst], capture_output=True, text=True, timeout=600)
    if run.returncode != 0:
        raise RuntimeError(f"the batch run failed ({run.returncode}): {run.stderr[-500:]}")
    with open(dst, "rb") as f:
        blob = f.read()
    out, pos = [], 0
    for _ in records:
        status, n = struct.unpack_from("<II", blob, pos)
        pos += 8
        out.append((status_name(status), blob[pos : pos + n]))
        pos += n
    if pos != len(blob):
        raise RuntimeError("the batch output does not match the records sent")
    return out


def c_realization_status(exe: str, tmp: str, data: bytes) -> str:
    path = os.path.join(tmp, "kplan.bkpr")
    with open(path, "wb") as f:
        f.write(data)
    run = subprocess.run([exe, "--decode-realization", path], capture_output=True, text=True,
                         timeout=60)  # fmt: skip
    return run.stdout.strip() if run.returncode == 0 else f"EXIT {run.returncode}"


# --- R9 through the compact offer -------------------------------------------------------------

R9_SCOPES = (("x86_avx512", "cool", "latency"), ("nvidia_ptx", "hot", "energy"))


def r9_cases() -> list[Case]:
    """The LEGAL modules of the fixed corpus (those `verify` passes clean) under two scopes: the
    plans R9 is held to, honest and forged. The coverage modules that drive the planner through
    illegal geometry stay out -- R9 refusing the plan of an illegal module is R9 working."""
    from bcir.kbcir.differential import THETAS
    from bcir.kbcir.weights import POLICIES
    from bcir.verify import verify

    tg = targets()
    fixed = [c for c in corpus_cases() if c.fixed and "/" in c.label]
    seen, out = set(), []
    for case in fixed:
        name = case.label.split("/")[0]
        if name in seen:
            continue
        seen.add(name)
        if verify(case.module):
            continue
        for tn, thn, pn in R9_SCOPES:
            out.append(Case(f"{name}/{tn}/{thn}/{pn}", case.module, tg[tn], THETAS[thn],
                            POLICIES[pn], True))  # fmt: skip
    return out


def r9_forgeries(result) -> list[tuple[str, object]]:
    """(label, forged plan) -- one per field a step carries, at the first and the last step. R9
    must refuse each, with a diagnostic and never a traceback."""
    from bcir.kbcir.cost import CostVector
    from bcir.model import Lane

    out: list[tuple[str, object]] = []
    for index in sorted({0, len(result.steps) - 1} if result.steps else ()):
        s = result.steps[index]
        c = s.candidate

        def with_step(i=index, **fields):
            steps = list(result.steps)
            steps[i] = replace(steps[i], **fields)
            return replace(result, steps=steps)

        def with_cand(i=index, **fields):
            return with_step(i, candidate=replace(result.steps[i].candidate, **fields))

        other = Lane.A if c.lane is not Lane.A else Lane.U
        out += [
            (f"name@{index}", with_cand(name="forged")),
            (f"width@{index}", with_cand(width=c.width * 2 + 1)),
            (f"base@{index}", with_cand(base=CostVector(tuple(v + 1 for v in c.base.v)))),
            (f"int-lane@{index}", with_cand(lane=int(c.lane))),
            (f"other-lane@{index}", with_cand(lane=other)),
            (f"unhashable-name@{index}", with_cand(name=["forged"])),
            (f"cost@{index}", replace(with_step(cost=s.cost + 1), score=result.score + 1)),
            (f"phase@{index}", with_step(phase_id=s.phase_id + 1000003)),
            (f"unhashable-phase@{index}", with_step(phase_id=[s.phase_id])),
        ]
    return out


def r9_misjudged(verify_plan, optimize) -> int:
    """planner.r9.misjudged, with the verifier and the planner as parameters (the RED run hands
    the parent's own)."""
    bad = 0
    for case in r9_cases():
        scope = dict(theta=case.theta, policy=case.policy)
        try:
            plan = optimize(case.module, case.h, case.theta, case.policy)
            honest = verify_plan(case.module, plan, case.h, **scope)
            forgeries = r9_forgeries(plan)
        except Exception:  # noqa: BLE001 -- an oracle that raises misjudged the case (L1)
            bad += 1
            continue
        bad += any(d.law in ("R8", "R9") for d in honest)
        try:
            bare = verify_plan(case.module, plan, case.h)
        except Exception:  # noqa: BLE001
            bare = [None]
        bad += any(d is None or d.law in ("R8", "R9") for d in bare)
        for label, forged in forgeries:
            # with the scope, and with the target alone: without Theta the offer is the only
            # guard of a realization's base (the scope's cost re-derivation would mask it)
            for scoped in (True, False):
                if not scoped and label.startswith("cost@"):
                    continue  # a forged cost is the scope's to see
                kwargs = scope if scoped else {}
                try:
                    verdict = verify_plan(case.module, forged, case.h, **kwargs)
                    bad += not any(d.law == "R9" for d in verdict)
                except Exception:  # noqa: BLE001 -- a traceback is a lost verdict (L1)
                    bad += 1
    return bad


# --- the grader ---------------------------------------------------------------------------------


def _built(build, rows, out: dict[str, float]) -> list:
    """A corpus, or none with every row it feeds failed: the corpora are the oracle's own plans
    and records, so a defect in the oracle can first surface as a corpus that cannot be built --
    a finding in a named row, never a traceback (L1)."""
    try:
        return build()
    except Exception:  # noqa: BLE001 -- an oracle that cannot build its corpus decided nothing
        for row in rows:
            out[row] += 1
        return []


def measure(exe: str | None, tmp: str) -> dict[str, float]:
    """The G17 rows (module docstring). `exe` is the C harness; None leaves the native rail out
    (the caller grades an absent rail), while a harness that cannot run fails every case it was
    handed."""
    from bcir.kbcir import realize, realize_reference

    out = {row: 0.0 for row in ROWS}
    cases = _built(corpus_cases, ["planner.parity"], out)
    records: list[bytes] = []
    expected: list[tuple[str, bytes]] = []
    for case in cases:
        args = (case.module, case.h, case.theta, case.policy)
        # before/after, the plan bytes (fixed corpus), Python/native
        comparisons = 1 + case.fixed + (exe is not None)
        try:
            new = realize.optimize(*args)
            old = realize_reference.optimize(*args)
            new_bytes, old_bytes = realization_bytes(new), realization_bytes(old)
            if new != old or _steps(new) != _steps(old) or new_bytes != old_bytes:
                out["planner.parity"] += 1
            if case.fixed and new_bytes[0] == "BCIR_OK":
                if plan_bytes(case.module, new, case.h) != plan_bytes(case.module, old, case.h):
                    out["planner.parity"] += 1
            record = pa.encode_input(*args)
        except Exception:  # noqa: BLE001 -- a rail that raises fails every comparison it owns (L1)
            out["planner.parity"] += comparisons
            continue
        records.append(record)
        expected.append(new_bytes)
    if exe is not None:
        try:
            native = c_batch(exe, tmp, records)
        except Exception:  # noqa: BLE001 -- a native rail that cannot run fails every case
            native = [("EXIT", b"")] * len(records)
        for got, want in zip(native, expected):
            if got != want:
                out["planner.parity"] += 1

    from bcir.verify import verify_plan

    try:
        out["planner.r9.misjudged"] += r9_misjudged(verify_plan, realize.optimize)
    except Exception:  # noqa: BLE001 -- a corpus that cannot be built fails the row (L1)
        out["planner.r9.misjudged"] += 1

    inputs = _built(malformed_inputs, ["planner.malformed.accepted"], out)
    reals = _built(malformed_realizations, ["planner.malformed.accepted"], out)
    for _label, data, status in inputs:
        try:
            ok = python_input_status(data) == status
        except Exception:  # noqa: BLE001
            ok = False
        out["planner.malformed.accepted"] += not ok
    for _label, data, status in reals:
        try:
            ok = python_realization_status(data) == status
        except Exception:  # noqa: BLE001
            ok = False
        out["planner.malformed.accepted"] += not ok
    if exe is not None:
        try:
            native = c_batch(exe, tmp, [data for _, data, _ in inputs])
        except Exception:  # noqa: BLE001
            native = [("EXIT", b"")] * len(inputs)
        for (_label, _data, status), (got, _bytes) in zip(inputs, native):
            out["planner.malformed.accepted"] += got != status
        for _label, data, status in reals:
            try:
                got = c_realization_status(exe, tmp, data)
            except Exception:  # noqa: BLE001
                got = "EXIT"
            out["planner.malformed.accepted"] += got != status
    return out


# --- the harness rows beyond the gate ----------------------------------------------------------

CALLS_SCALE = 8  # the audit's K_BCIR->StreamPack fixture at scale 8: 32,768 claims


def audit_fixture(scale: int):
    """(module, target, theta): the audit's K_BCIR->StreamPack fixture (bcir/performance_audit.py)
    at `scale` -- matmul (32 * scale)^2 tiled by 8, (4 * scale)^3 claims, x86 AVX2, memory-bound."""
    from bcir.examples import matmul_tiled
    from bcir.kbcir.cost import TargetProfile, Theta

    grid = 4 * scale
    return matmul_tiled(n=grid * 8, tile=8), TargetProfile.x86_avx2(), Theta.mem_bound()


def call_count(planner, scale: int = CALLS_SCALE) -> int:
    """The calls `planner(module, h, theta)` makes planning the audit fixture at `scale` (cProfile's
    total, builtins included -- the count the 2026-09-04 profile quoted). Deterministic for one
    interpreter; it differs between CPython versions, so a gate compares two planners in one
    process and the recorded baseline names its interpreter."""
    import cProfile
    import pstats

    module, h, theta = audit_fixture(scale)
    planner(module, h, theta)  # warm imports and caches outside the count
    profile = cProfile.Profile()
    profile.enable()
    planner(module, h, theta)
    profile.disable()
    return pstats.Stats(profile).total_calls


def native_ms(exe: str, tmp: str, scale: int = 4) -> float:
    """The native planner's median time per plan of the audit fixture at `scale`, in ms."""
    module, h, theta = audit_fixture(scale)
    path = os.path.join(tmp, f"audit_{scale}.bkpi")
    with open(path, "wb") as f:
        f.write(pa.encode_input(module, h, theta))
    run = subprocess.run([exe, "--bench", path, "5", "9"], capture_output=True, text=True,
                         timeout=600)  # fmt: skip
    if run.returncode != 0 or not run.stdout.startswith("BENCH "):
        raise RuntimeError(f"the native bench failed: {run.stdout} {run.stderr[-300:]}")
    return int(run.stdout.split()[1]) / 1e6
