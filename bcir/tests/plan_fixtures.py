"""Shared fixtures for the G11 gates (S1-C): the C plan harness, its record-dump parser, the
stale-vector and malformed-plan variants and the static-memory lifetime agreement.

Used by `test_execution_plan.py` and by the GEM+ baseline harness
(`tools/perf/gemplus_baseline.py::measure_plan`), so the tests and the graded rows measure
ONE definition of each gate. Not a test module (run_all collects `test_*.py`).
"""

from __future__ import annotations

from dataclasses import replace
import os
import shutil
import struct
import subprocess
import zlib

from bcir.abi.execution_plan_abi import (
    _HEADER,
    _MODE_WIRE,
    PLAN_HEADER_SIZE,
    PLAN_MAGIC,
    PLAN_VERSION,
    PLAN_VERSION_MAX,
    TAIL_STREAM_WIRE,
    _write_generation,
    _write_lifetime,
    _write_move,
    _write_step,
    decode_plan,
    encode_plan,
)
from bcir.abi.streampack_abi import AbiError, _Writer, encode as encode_pack
from bcir.gem.execution_plan import (
    TAIL_STREAM,
    ExecutionPlan,
    Lifetime,
    MovementEdge,
    PlanStep,
    plan_from_realization,
)
from bcir.gem.streampack import Generation, generation_vector, hydrate
from bcir.model import Lane, Resource

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
C_DIR = os.path.join(ROOT, "runtime", "c")


# --- the C harness -------------------------------------------------------------------------


def compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def build_harness(tmp: str) -> str | None:
    """Compile runtime/c/test_execution_plan.c against the freestanding runtime; None without
    a compiler (the quick tier hides one on purpose)."""
    cc = compiler()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_execution_plan")
    build = subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            os.path.join(C_DIR, "bcir_runtime.c"),
            os.path.join(C_DIR, "test_execution_plan.c"),
            "-I",
            C_DIR,
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise AssertionError(build.stderr)
    return exe


def run_harness(
    exe: str,
    tmp: str,
    plan_bytes: bytes,
    *,
    dump: bool = False,
    pack_bytes: bytes | None = None,
    live=None,
) -> tuple[int, str]:
    """Run the harness over `plan_bytes`; returns (exit status, stdout)."""
    plan_path = os.path.join(tmp, "plan.bin")
    with open(plan_path, "wb") as f:
        f.write(plan_bytes)
    argv = [exe, plan_path]
    if dump:
        argv.append("--dump")
    if pack_bytes is not None:
        pack_path = os.path.join(tmp, "pack.bin")
        with open(pack_path, "wb") as f:
            f.write(pack_bytes)
        argv += ["--pack", pack_path]
    if live is not None:
        argv.append("--live")
        argv += [f"{g.rid}:{g.map_gen}:{g.data_gen}" for g in live]
    run = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    return run.returncode, run.stdout


def c_roundtrip(exe: str, tmp: str, plan_bytes: bytes) -> str:
    """The C decode of `plan_bytes` as the harness's record dump (asserts the C rail accepted)."""
    code, out = run_harness(exe, tmp, plan_bytes, dump=True)
    assert code == 0 and "\nOK\n" in out + "\n", out
    return out


def _fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split(" ")[1:])


def _text(hexed: str) -> str:
    return bytes.fromhex(hexed).decode("utf-8")


def parse_c_dump(text: str) -> ExecutionPlan:
    """Rebuild the ExecutionPlan from the C harness's dump -- the C decode, as a value."""
    from bcir.gem.execution_plan import COHERENCE_ACTIONS, LIVENESS_DOMAINS, MOVE_KINDS, PLAN_MODES

    plan = ExecutionPlan()
    for line in text.splitlines():
        kind = line.split(" ", 1)[0]
        if kind == "header":
            f = _fields(line)
            plan.source_plan = _text(f["source_plan"])
            plan.mode = PLAN_MODES[int(f["mode"])]
            plan.liveness = LIVENESS_DOMAINS[int(f["liveness"])]
            plan.streams = int(f["streams"])
            plan.knee = int(f["knee"])
            plan.makespan = int(f["makespan"])
            plan.module_hash = int(f["module_hash"])
            plan.target_hash = int(f["target_hash"])
        elif kind == "step":
            f = _fields(line)
            stream = int(f["stream"])
            plan.steps.append(
                PlanStep(
                    claim_id=int(f["claim"]),
                    phase_id=int(f["phase"]),
                    candidate=_text(f["candidate"]),
                    lane=Lane(int(f["lane"])),
                    width=int(f["width"]),
                    cost=int(f["cost"]),
                    stream=TAIL_STREAM if stream == TAIL_STREAM_WIRE else stream,
                    start=int(f["start"]),
                    duration=int(f["duration"]),
                )
            )
        elif kind == "lifetime":
            f = _fields(line)
            plan.lifetimes.append(
                Lifetime(
                    rid=int(f["rid"]),
                    bank=_text(f["bank"]),
                    offset=int(f["offset"]),
                    size_bytes=int(f["size"]),
                    alignment=int(f["alignment"]),
                    first_phase=int(f["first"]),
                    last_phase=int(f["last"]),
                    first_tick=int(f["first_tick"]),
                    last_tick=int(f["last_tick"]),
                )
            )
        elif kind == "move":
            f = _fields(line)
            plan.moves.append(
                MovementEdge(
                    rid=int(f["rid"]),
                    src_bank=_text(f["src"]),
                    dst_bank=_text(f["dst"]),
                    offset=int(f["offset"]),
                    size_bytes=int(f["size"]),
                    route=_text(f["route"]),
                    kind=MOVE_KINDS[int(f["kind"])],
                    coherence=COHERENCE_ACTIONS[int(f["coherence"])],
                    map_gen=int(f["map_gen"]),
                    data_gen=int(f["data_gen"]),
                    after_claim=int(f["after"]),
                    before_claim=int(f["before"]),
                    claim=int(f.get("claim", "0")),
                    version=int(f.get("version", "0")),
                    flags=int(f.get("flags", "0")),
                    producer=int(f.get("producer", "0")),
                    bits=int(f.get("bits", "0")),
                    cert=int(f.get("cert", "0")),
                )
            )
        elif kind == "binding":
            f = _fields(line)
            plan.source_hash = int(f["source_hash"])
            plan.spec_hash = int(f["spec_hash"])
        elif kind == "gen":
            f = _fields(line)
            plan.generations.append(
                Generation(int(f["rid"]), int(f["map_gen"]), int(f["data_gen"]))
            )
    return plan


# --- byte surgery ----------------------------------------------------------------------------


def reseal(blob: bytes) -> bytes:
    body = bytes(blob[:-4])
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def raw_encode(plan: ExecutionPlan, version: int | None = None) -> bytes:
    """The wire bytes of `plan` WITHOUT the wire laws -- the only way to mint the malformed
    variants both rails must refuse (the public encoder refuses to emit them). `version`
    forces the wire version; by default it is the lowest that carries the plan."""
    from bcir.abi.execution_plan_abi import _LIVENESS_OFF, _LIVENESS_WIRE, plan_version

    version = plan_version(plan) if version is None else version
    header = _HEADER.pack(
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
    if version >= 2:
        header = bytearray(header)
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


def audit_fixture():
    """The audit's K_BCIR->StreamPack fixture at scale 1 (matmul_tiled n=32: 64 claims in
    three phases, 48 resources) under the profile the audit measures it with."""
    from bcir.examples import matmul_tiled
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize

    module = matmul_tiled(n=32, tile=8)
    target, theta = TargetProfile.x86_avx2(), Theta.mem_bound()
    return module, target, theta, optimize(module, target, theta)


def malformed_plan_bytes(module, plan: ExecutionPlan) -> list[tuple[str, bytes, frozenset]]:
    """(name, bytes, rails that must refuse) for every malformed variant of `plan`.

    The wire-level variants are refused on BOTH rails by the codec/decoder; the three that
    need the module (an unknown claim, steps out of topological phase order, a missing claim)
    are well-formed bytes the C twin cannot see through and the Python verifier refuses."""
    both, python = frozenset({"python", "c"}), frozenset({"python"})
    good = encode_plan(plan)
    variants: list[tuple[str, bytes, frozenset]] = []

    def add(name, blob, rails=both):
        variants.append((name, bytes(blob), rails))

    # header surgery
    b = bytearray(good)
    b[0:4] = b"BPLM"
    add("magic", reseal(b))
    b = bytearray(good)
    struct.pack_into("<H", b, 4, PLAN_VERSION_MAX + 1)  # a newer version than the reader's
    add("version", reseal(b))
    b = bytearray(good)
    struct.pack_into("<H", b, 6, 1)
    add("flags", reseal(b))
    b = bytearray(good)
    b[9] = 1
    add("reserved", reseal(b))
    b = bytearray(good)
    b[8] = 7
    add("mode", reseal(b))
    b = bytearray(good)
    struct.pack_into("<I", b, 16, plan.streams + 1)
    add("knee", reseal(b))
    b = bytearray(good)
    b[PLAN_HEADER_SIZE + 3] ^= 0x55
    add("crc", bytes(b))
    # body surgery
    add(
        "truncated", reseal(good[:-16])
    )  # the last generation record is gone; n_gens still promises it
    add("trailing", reseal(good[:-4] + b"\x00\x00\x00\x00" + good[-4:]))
    # model-level variants through the raw writer
    steps = list(plan.steps)
    assert len(steps) >= 2
    dup = replace(plan, steps=[steps[0], replace(steps[1], claim_id=steps[0].claim_id), *steps[2:]])
    add("duplicated", raw_encode(dup))
    add(
        "stream",
        raw_encode(replace(plan, steps=[replace(steps[0], stream=plan.streams), *steps[1:]])),
    )
    add(
        "duration",
        raw_encode(
            replace(plan, steps=[replace(steps[0], duration=steps[0].duration + 1), *steps[1:]])
        ),
    )
    add("makespan", raw_encode(replace(plan, makespan=0)))
    b = bytearray(raw_encode(plan))
    # the lane byte of step 0: after source_plan(str) + claim(8) + phase(4) + candidate(str)
    off = (
        PLAN_HEADER_SIZE
        + 2
        + len(plan.source_plan.encode())
        + 8
        + 4
        + 2
        + len(steps[0].candidate.encode())
    )
    b[off] = 9
    add("lane", reseal(b))
    add("width", raw_encode(replace(plan, steps=[replace(steps[0], width=3), *steps[1:]])))
    gens = list(plan.generations)
    assert len(gens) >= 2
    add("gens.unsorted", raw_encode(replace(plan, generations=[gens[1], gens[0], *gens[2:]])))
    add(
        "lifetime.alignment",
        raw_encode(replace(plan, lifetimes=[Lifetime(1, "ram", 0, 64, 3, 0, 0)])),
    )
    add(
        "lifetime.ticks",
        raw_encode(replace(plan, lifetimes=[Lifetime(1, "ram", 0, 64, 64, 0, 0, 5, 5)])),
    )
    add(
        "lifetime.alias",
        raw_encode(
            replace(
                plan,
                lifetimes=[
                    Lifetime(1, "ram", 0, 128, 64, 0, 1),
                    Lifetime(2, "ram", 64, 64, 64, 1, 1),
                ],
            )
        ),
    )
    add("move.kind", _bad_move_kind(plan))
    # the two the module alone can refuse (the third, phase order, needs two phases:
    # `out_of_order_variant`)
    add(
        "unknown-claim",
        encode_plan(replace(plan, steps=[replace(steps[0], claim_id=999_999_999), *steps[1:]])),
        python,
    )
    add("missing-claim", encode_plan(replace(plan, steps=steps[:-1])), python)
    return variants


def two_phase_fixture():
    """The first corpus program with two or more phases, planned under x86_avx512/cool."""
    from bcir.examples import PROGRAMS
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize

    target, theta = TargetProfile.x86_avx512(), Theta.cool()
    for _name, build in sorted(PROGRAMS.items()):
        module = build()
        if len(module.phases) >= 2:
            return module, target, theta, optimize(module, target, theta)
    raise AssertionError("the corpus has no two-phase program")


def out_of_order_variant() -> tuple[str, object, bytes, frozenset]:
    """(name, module, bytes, rails): a well-formed plan whose steps put a later phase's claim
    before an earlier phase's -- the C twin cannot see the module's phase order, the Python
    verifier refuses it (R9)."""
    module, target, _theta, result = two_phase_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    steps = list(plan.steps)
    phases = sorted({s.phase_id for s in steps})
    first = next(s for s in steps if s.phase_id == phases[0])
    later = next(s for s in steps if s.phase_id == phases[-1])
    swapped = [later if s is first else first if s is later else s for s in steps]
    return "out-of-order", module, encode_plan(replace(plan, steps=swapped)), frozenset({"python"})


def _bad_move_kind(plan: ExecutionPlan) -> bytes:
    """A movement edge whose kind code (9) is outside the closed set."""
    b = raw_encode(replace(plan, moves=[MovementEdge(1, "ram", "hbm", 0, 64)]))
    # the move record sits after the steps and lifetimes: locate it from the end (generations
    # are 12 bytes each, the move is the only one): kind byte = end - 4(crc) - gens - (8+8+4+4+1) - 1
    kind_off = len(b) - 4 - 12 * len(plan.generations) - (8 + 8 + 4 + 4) - 2
    b = bytearray(b)
    b[kind_off] = 9
    return reseal(b)


def python_refuses(module, blob: bytes) -> bool:
    """The Python rail's verdict on `blob`: the codec refuses it, or the verifier does."""
    from bcir.verify import verify_execution_plan

    try:
        plan = decode_plan(blob)
    except AbiError:
        return True
    return any(d.law == "R9" for d in verify_execution_plan(module, plan))


def c_refuses(exe: str, tmp: str, blob: bytes) -> bool:
    code, out = run_harness(exe, tmp, blob)
    return code != 0 and "\nOK\n" not in out + "\n"


def malformed_variants(exe: str | None, tmp: str | None) -> tuple[int, int]:
    """(refused, count) over (variant, rail) pairs; the C rail is counted only with a harness."""
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    rows = [(name, module, blob, rails) for name, blob, rails in malformed_plan_bytes(module, plan)]
    rows.append(out_of_order_variant())
    refused = count = 0
    for _name, owner, blob, rails in rows:
        if "python" in rails:
            count += 1
            refused += python_refuses(owner, blob)
        if "c" in rails and exe is not None:
            count += 1
            refused += c_refuses(exe, tmp, blob)
    return refused, count


# --- the v3 wire (G8, S5-C) -------------------------------------------------------------------


def v3_plan() -> ExecutionPlan:
    """The smallest v3 plan: three steps, resource 9's lifetime in `ram`, and one edge that
    names every reference -- the claim that executes it (2), the writer it follows (1) and the
    reader it precedes (3) -- plus the source/spec binding."""
    from bcir.gem.execution_plan import MOVE_HAS_AFTER, MOVE_HAS_BEFORE, MOVE_HAS_CLAIM

    steps = [
        PlanStep(1, 0, "scalar", Lane.U, 1, 5, 0, 0, 5),
        PlanStep(2, 0, "scalar", Lane.U, 1, 5, 0, 5, 5),
        PlanStep(3, 1, "scalar", Lane.U, 1, 5, 0, 10, 5),
    ]
    edge = MovementEdge(
        9,
        "ram",
        "hbm",
        0,
        64,
        "ram>hbm",
        "direct",
        "none",
        0,
        0,
        1,
        3,
        claim=2,
        version=1,
        flags=MOVE_HAS_AFTER | MOVE_HAS_BEFORE | MOVE_HAS_CLAIM,
    )
    return ExecutionPlan(
        makespan=15,
        steps=steps,
        lifetimes=[Lifetime(9, "ram", 0, 64, 64, 0, 1)],
        moves=[edge],
        source_hash=11,
        spec_hash=12,
    )


def v3_variants() -> list[tuple[str, bytes, ExecutionPlan | None]]:
    """(name, bytes, plan) for every malformed move and binding of the wire -- each refused by
    BOTH rails' decoders (`decode_plan` and `bcir_ep_verify`), and each `plan` refused by the
    encoder before it is published. Minted through `raw_encode`, since the encoder will not
    emit them. Two carry no plan: a well-formed plan spelled as v3 with no binding
    (`unbound.zero`, which re-encodes as v1) and a binding cut short (`binding.truncated`) --
    only their bytes are illegal."""
    from bcir.gem.execution_plan import (
        MOVE_HAS_AFTER,
        MOVE_HAS_BEFORE,
        MOVE_HAS_CLAIM,
        MOVE_HAS_PRODUCER,
    )

    plan = v3_plan()
    mv = plan.moves[0]

    def edge(**kw) -> ExecutionPlan:
        return replace(plan, moves=[replace(mv, **kw)])

    remat = dict(kind="rematerialized", dst_bank="ram", route="ram", cert=5)
    bare = replace(mv, claim=0, version=0, flags=0, producer=0, bits=0, cert=0)
    v1 = replace(plan, moves=[bare], source_hash=0, spec_hash=0)
    plans = {
        # every version: a move changes banks unless it is a remat; a writeback is exact
        "same.bank": edge(dst_bank="ram", route="ram"),
        "remat.moves": edge(
            kind="rematerialized", producer=1, cert=5, flags=mv.flags | MOVE_HAS_PRODUCER
        ),
        "writeback.lossy": edge(  # home-bound, so only the lossy-writeback law is broken
            src_bank="hbm",
            dst_bank="ram",
            route="hbm>ram",
            kind="compressed",
            coherence="writeback",
            bits=8,
            cert=5,
        ),
        "writeback.remat": edge(
            coherence="writeback", producer=1, flags=mv.flags | MOVE_HAS_PRODUCER, **remat
        ),
        "v1.same.bank": replace(v1, moves=[replace(bare, dst_bank="ram", route="ram")]),
        "v1.writeback.lossy": replace(
            v1, moves=[replace(bare, kind="compressed", coherence="writeback")]
        ),
        # the flags: defined, and what they call absent is zero
        "flags.undefined": edge(flags=mv.flags | 0x10),
        "after.absent": edge(flags=mv.flags & ~MOVE_HAS_AFTER),
        "before.absent": edge(flags=mv.flags & ~MOVE_HAS_BEFORE),
        "claim.absent": edge(flags=mv.flags & ~MOVE_HAS_CLAIM),
        "producer.absent": edge(producer=1),
        # what they call present is a step of the plan
        "claim.unknown": edge(claim=99),
        "after.unknown": edge(after_claim=42),
        "before.unknown": edge(before_claim=42),
        "producer.unknown": edge(producer=99, flags=mv.flags | MOVE_HAS_PRODUCER, **remat),
        # exactly a remat names a producer, and not itself
        "remat.no_producer": edge(**remat),
        "producer.direct": edge(producer=1, flags=mv.flags | MOVE_HAS_PRODUCER),
        "producer.self": edge(producer=2, flags=mv.flags | MOVE_HAS_PRODUCER, **remat),
        # exactly a compressed edge names 1..31 codec bits; exactly remat/compressed certify
        "compressed.no_bits": edge(kind="compressed", cert=5),
        "compressed.wide": edge(kind="compressed", bits=32, cert=5),
        "compressed.no_cert": edge(kind="compressed", bits=8),
        "direct.bits": edge(bits=8),
        "direct.cert": edge(cert=5),
        "remat.no_cert": edge(
            producer=1, flags=mv.flags | MOVE_HAS_PRODUCER, **{**remat, "cert": 0}
        ),
        # the window is ordered by the placement
        "window.after": edge(after_claim=3),
        "window.before": edge(before_claim=1),
        # a writeback lands home
        "writeback.home": edge(coherence="writeback"),
        # the binding names both hashes
        "unbound.source": replace(plan, source_hash=0),
        "unbound.spec": replace(plan, spec_hash=0),
    }
    variants: list[tuple[str, bytes, ExecutionPlan | None]] = [
        (name, raw_encode(broken), broken) for name, broken in plans.items()
    ]
    variants.append(("unbound.zero", raw_encode(v1, 3), None))
    variants.append(("binding.truncated", reseal(raw_encode(plan)[:-12]), None))
    return variants


def wire_refuses(blob: bytes) -> bool:
    """The Python codec's verdict on `blob` alone."""
    try:
        decode_plan(blob)
    except AbiError:
        return True
    return False


def asn1_accepted() -> tuple[int, int]:
    """(accepted, count) over (variant, transfer syntax): each malformed plan of `v3_variants`
    spelled as a DER, a CANONICAL-OER and a JER document of the BCIR-ExecutionPlan module --
    minted through the raw projection, since `plan_to_value` refuses them -- and read back by
    that syntax's decoder, which must refuse every one (the value space is the native one)."""
    from bcir.asn1 import Asn1Error
    from bcir.asn1.execution_plan import (
        EXECUTION_PLAN,
        MODULE,
        _plan_value,
        decode_plan_der,
        decode_plan_jer,
        decode_plan_oer,
    )
    from bcir.asn1.jer import JerRules, encode_jer
    from bcir.asn1.oer import OerRules, encode_oer

    syntaxes = (
        (lambda v: MODULE.encode("ExecutionPlan", v), decode_plan_der),
        (lambda v: encode_oer(EXECUTION_PLAN, v, rules=OerRules.CANONICAL), decode_plan_oer),
        (lambda v: encode_jer(EXECUTION_PLAN, v, rules=JerRules.CANONICAL), decode_plan_jer),
    )
    accepted = count = 0
    for _name, _blob, broken in v3_variants():
        if broken is None:
            continue
        value = _plan_value(broken)
        for encode, decode in syntaxes:
            count += 1
            try:
                decode(encode(value))
            except Asn1Error:
                continue
            accepted += 1
    return accepted, count


# --- stale generation vectors ------------------------------------------------------------------


def stale_cases():
    """The three stale-vector fixtures, each as (name, module, plan, pack-or-None, live).

    1. the registry moved after the plan was minted (a resource's map_gen bumped);
    2. a resource was declared after the plan was minted;
    3. the pack was hydrated before the registry moved and the plan minted after -- the pack's
       vector is older than its plan's.
    `live` is the registry the C rail is handed (`generation_vector(module)` after the move)."""
    from bcir.examples import matmul_tiled
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize
    from bcir.model import Domain

    target, theta = TargetProfile.x86_avx2(), Theta.mem_bound()

    module = matmul_tiled(n=32, tile=8)
    result = optimize(module, target, theta)
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    rid = min(module.resources)
    module.resources[rid] = replace(
        module.resources[rid], map_gen=module.resources[rid].map_gen + 1
    )
    module.touch()
    yield "registry-moved", module, plan, None, generation_vector(module)

    module = matmul_tiled(n=32, tile=8)
    result = optimize(module, target, theta)
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    module.add_resource(Resource(max(module.resources) + 1, Domain.RAM, 4, (16,), name="late"))
    yield "declared-after", module, plan, None, generation_vector(module)

    module = matmul_tiled(n=32, tile=8)
    result = optimize(module, target, theta)
    pack = hydrate(module, result, "plan0")
    rid = min(module.resources)
    module.resources[rid] = replace(
        module.resources[rid], data_gen=module.resources[rid].data_gen + 1
    )
    module.touch()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    yield "pack-older-than-plan", module, plan, pack, generation_vector(module)


def stale_fixtures(exe: str | None, tmp: str | None) -> tuple[int, int]:
    """(refused, count) over (fixture, rail) pairs: the Python verifier must emit R11 and the C
    twin must return BCIR_ERR_STALE (from the registry check, or from the plan/pack binding)."""
    from bcir.verify import verify_execution_plan

    refused = count = 0
    for _name, module, plan, pack, live in stale_cases():
        count += 1
        diags = verify_execution_plan(module, plan, pack=pack)
        refused += any(d.law == "R11" for d in diags)
        if exe is not None:
            count += 1
            blob = encode_plan(plan)
            if pack is None:
                code, out = run_harness(exe, tmp, blob, live=live)
                refused += code != 0 and "vector=BCIR_ERR_STALE" in out
            else:
                code, out = run_harness(exe, tmp, blob, pack_bytes=encode_pack(pack))
                refused += code != 0 and "pack=BCIR_ERR_STALE" in out
    return refused, count


# --- static memory -------------------------------------------------------------------------------


def static_memory_lifetimes_agree() -> bool:
    """The static-memory reader: a plan minted with the static memory plan carries its
    allocations exactly (rid, bank, offset, size, alignment, first/last phase), and they
    survive the bytes."""
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.static_memory import plan_static_memory
    from bcir.performance_audit import _AuditHardware, static_memory_module

    module = static_memory_module(1)
    hardware = _AuditHardware()
    bindings = {rid: "ram" for rid in module.resources}
    static_plan = plan_static_memory(module, bindings, hardware)
    target, theta = TargetProfile.x86_avx2(), Theta.cool()
    result = optimize(module, target, theta)
    plan = plan_from_realization(module, result, target, "eft", static_plan=static_plan)
    expected = [
        (a.rid, a.bank, a.offset, a.size_bytes, a.alignment, a.first_phase, a.last_phase)
        for a in sorted(static_plan.allocations, key=lambda a: a.rid)
    ]
    carried = [
        (lt.rid, lt.bank, lt.offset, lt.size_bytes, lt.alignment, lt.first_phase, lt.last_phase)
        for lt in decode_plan(encode_plan(plan)).lifetimes
    ]
    return carried == expected and len(carried) == len(module.resources)
