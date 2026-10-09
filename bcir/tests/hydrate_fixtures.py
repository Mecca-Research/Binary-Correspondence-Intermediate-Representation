"""The CXX4 fixtures and grader: the native hydrate held to the oracle's StreamPack, byte for byte.

`bcir_kp_hydrate` (runtime/c/bcir_kplan.c) writes, from the planner's own input (BKPI), its
realization (BKPR) and the binding (BKPB: the RIDs, the generation vector, the plan's name), the
StreamPack `streampack_abi.encode(gem.streampack.hydrate(module, result, plan))` writes. One
function, `measure`, grades both rows; the tests, the harness (`tools/perf/gemplus_baseline.py
--group hydrate`) and the `kplan` section of tools/c/check_runtime.sh call it:

    hydrate.native.parity              cases whose native pack differs from the oracle's -- the
                                       bytes, or the same refusal -- over the planner's whole
                                       corpus (`planner_fixtures.corpus_cases`)
    hydrate.native.malformed.accepted  (malformed record, rail) pairs not refused with the
                                       declared status: one BKPB variant per wire law and per law
                                       against the input, one BKPR forgery per hydrate law, and
                                       the two values a StreamPack cannot carry (a width that is
                                       not a power of two, a negative block field), on the Python
                                       oracle and the C twin
"""

from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import replace
from types import SimpleNamespace

from bcir.abi import planner_abi as pa
from bcir.tests.planner_fixtures import _built, _claim, _reseal, corpus_cases, status_name

ROWS = ("hydrate.native.parity", "hydrate.native.malformed.accepted")
PLAN = "plan0"


# --- the oracle ---------------------------------------------------------------------------------


def _result_of(real: pa.Realization):
    """The realization a BKPR record carries, in the shape `gem.streampack.hydrate` reads."""
    from bcir.model import Lane

    steps = []
    for s in real.steps:
        name = f"vec{s.width}" if s.name == pa.NAME_VEC else pa.NAMES[s.name]
        cand = SimpleNamespace(name=name, lane=Lane(s.lane), width=s.width)
        steps.append(SimpleNamespace(claim_id=s.claim_id, phase_id=s.phase_id, candidate=cand))
    return SimpleNamespace(steps=steps)


def oracle_pack(module, bkpi: bytes, bkpr: bytes, bkpb: bytes) -> tuple[str, bytes]:
    """The oracle's verdict on the three records: their decoders and the binding's laws, then
    `hydrate` over the module and the realization, then `encode` -- each refusal named by the
    status the C twin returns for it."""
    from bcir.abi.streampack_abi import AbiError, encode
    from bcir.gem.streampack import hydrate

    try:
        value = pa.decode_input(bkpi)
        pa.check_input(value)
        real = pa.decode_realization(bkpr)
        binding = pa.decode_binding(bkpb)
        pa.check_binding(binding, value)
    except pa.PlannerAbiError as exc:
        return exc.status, b""
    try:
        pack = hydrate(module, _result_of(real), binding.plan)
    except ValueError:
        return "BCIR_ERR_PROVENANCE", b""
    try:
        return "BCIR_OK", encode(pack)
    except AbiError as exc:
        if "width must be a nonzero power of two" in str(exc):
            return "BCIR_ERR_WIDTH", b""
        return "BCIR_ERR_OVERFLOW", b""


# --- the native rail -----------------------------------------------------------------------------


def c_hydrate_batch(exe: str, tmp: str, triples: list[tuple[bytes, bytes, bytes]]):
    """Hydrate every (BKPI, BKPR, BKPB) in one process: (status name, StreamPack bytes) each."""
    src, dst = os.path.join(tmp, "khydrate.in"), os.path.join(tmp, "khydrate.out")
    with open(src, "wb") as f:
        for triple in triples:
            for record in triple:
                f.write(struct.pack("<I", len(record)) + record)
    run = subprocess.run(
        [exe, "--hydrate-batch", src, dst], capture_output=True, text=True, timeout=600
    )
    if run.returncode != 0:
        raise RuntimeError(f"the hydrate batch failed ({run.returncode}): {run.stderr[-500:]}")
    with open(dst, "rb") as f:
        blob = f.read()
    out, pos = [], 0
    for _ in triples:
        status, n = struct.unpack_from("<II", blob, pos)
        pos += 8
        out.append((status_name(status), blob[pos : pos + n]))
        pos += n
    if pos != len(blob):
        raise RuntimeError("the hydrate batch output does not match the records sent")
    return out


# --- the corpora ----------------------------------------------------------------------------------


def parity_cases() -> list[tuple[str, object, tuple[bytes, bytes, bytes], tuple[str, bytes]]]:
    """(label, module, (BKPI, BKPR, BKPB), the oracle's verdict) for every planner corpus case
    whose plan the realization record carries (a plan the BKPR encoder refuses has no record to
    hydrate: `planner.parity` grades that refusal), and the hydrate's own cases
    (`hydrate_cases`): the constructs the planner corpus never draws."""
    from bcir.kbcir import realize

    out = []
    scoped = [(c.label, c.module, (c.module, c.h, c.theta, c.policy), PLAN) for c in corpus_cases()]
    for label, module, args, plan in scoped + hydrate_cases():
        result = realize.optimize(*args)
        try:
            bkpr = pa.encode_realization(result)
        except pa.PlannerAbiError:
            continue
        triple = (pa.encode_input(*args), bkpr, pa.encode_binding(module, plan))
        try:
            from bcir.abi.streampack_abi import encode
            from bcir.gem.streampack import hydrate

            want = ("BCIR_OK", encode(hydrate(module, result, plan)))
        except Exception:  # noqa: BLE001 -- graded below: the oracle's verdict, by status
            want = oracle_pack(module, *triple)
        out.append((label, module, triple, want))
    return out


def _blanked(module):
    """`module` with every claim's op empty: each segment then carries its realization's name
    (`noop`, `scalar`, `vec<w>`, `gather`, ...) -- the hydrate's other spelling of an opcode."""
    import copy

    m = copy.deepcopy(module)
    for ph in m.phases:
        ph.claims = [replace(c, op="") for c in ph.claims]
    return m


def hydrate_cases() -> list[tuple[str, object, tuple, str]]:
    """(label, module, planner args, plan name) for the constructs the planner corpus never
    draws: every realization name as an opcode (the fixed and coverage modules with their ops
    blanked), generations above zero, a module that declares no resource (a v1 pack), a module
    with no claims, RIDs and generations at the top of their range, and plan names other than
    `plan0` (empty, multi-byte UTF-8, 65535 bytes)."""
    from bcir.examples import PROGRAMS
    from bcir.kbcir.cost import Theta
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass
    from bcir.tests.planner_fixtures import coverage_modules, targets

    tg = targets()
    hosts = [(name, tg[name]) for name in ("x86_avx512", "arm64_neon", "nvidia_ptx")]
    fixed = [(name, build()) for name, build in sorted(PROGRAMS.items())] + coverage_modules()
    out = []
    for name, m in fixed:
        blank = _blanked(m)
        for k, rid in enumerate(sorted(blank.resources)):
            blank.resources[rid] = replace(
                blank.resources[rid], map_gen=(k * 7 + 1) % 5, data_gen=(k * 3) % 4
            )
        for tn, h in hosts:
            out.append((f"blank:{name}/{tn}", blank, (blank, h, Theta.cool()), PLAN))
    seed = seed_module()
    for tn, h in hosts:
        for plan in ("", "plan-\u00fc\u00df-\u65e5", "p" * pa.PLAN_NAME_MAX):
            out.append((f"seed/{tn}/plan{len(plan)}", seed, (seed, h, Theta.hot()), plan))
    unit = {"lane": Lane.U, "stride_class": StrideClass.UNIT}
    bare = Module(name="undeclared")  # every operand undeclared: no generation vector, a v1 pack
    bare.add_phase(
        Phase(0, (), [_claim(1, Opcode.ADD, count=64, rd=(5, 6), wr=(5,), op="vector.add", **unit)])
    )
    empty = Module(name="empty")
    empty.add_resource(Resource(rid=2, shape=(8,), map_gen=1, data_gen=1))
    nothing = Module(name="nothing")
    top = Module(name="top")
    big = 0xFFFFFFFF
    top.add_resource(Resource(rid=big, shape=(64,), map_gen=big, data_gen=big - 1))
    top.add_resource(Resource(rid=big - 1, shape=(64,), map_gen=0, data_gen=big))
    top.add_phase(
        Phase(
            9,
            (),
            [_claim(big, Opcode.ADD, count=64, rd=(big,), wr=(big - 1,), op="vector.add", **unit)],
        )
    )
    for label, m in (("v1", bare), ("no-claims", empty), ("nothing", nothing), ("top", top)):
        for tn, h in hosts:
            out.append((f"{label}/{tn}", m, (m, h, Theta.cool()), PLAN))
    return out


def seed_module():
    """Two phases (the second after the first), a vector claim with reads (a prefetch), a claim
    with an empty op (its realization names its segment), a store with a primary resource, an
    operand the module does not declare, and a declared resource no claim names."""
    from bcir.model import Lane, Module, Opcode, Phase, Resource, StrideClass
    from bcir.model.lanes import Domain

    m = Module(name="hydrate-seed")
    m.add_resource(Resource(rid=7, domain=Domain.RAM, shape=(64,), map_gen=3, data_gen=1))
    m.add_resource(Resource(rid=3, domain=Domain.HBM, shape=(64,), map_gen=1, data_gen=4))
    m.add_resource(Resource(rid=11, domain=Domain.RAM, shape=(64,)))
    m.add_resource(Resource(rid=40, domain=Domain.CXL, shape=(64,), map_gen=2))  # never named
    unit = {"lane": Lane.U, "stride_class": StrideClass.UNIT}
    m.add_phase(
        Phase(
            0,
            (),
            [
                _claim(1, Opcode.ADD, count=64, rd=(7, 3), wr=(11,), op="vector.add", **unit),
                _claim(2, Opcode.STORE, count=64, rd=(), wr=(7,), op="", primary_rid=3, **unit),
            ],
        )
    )
    m.add_phase(
        Phase(
            1,
            (0,),
            [_claim(3, Opcode.MUL, count=8, rd=(11, 99), wr=(3,), op="", offset=4, **unit)],
        )
    )
    return m


def _seed():
    from bcir.kbcir import realize
    from bcir.kbcir.cost import TargetProfile, Theta

    m = seed_module()
    args = (m, TargetProfile.x86_avx512(), Theta.cool())
    result = realize.optimize(*args)
    return m, pa.encode_input(*args), pa.encode_realization(result), pa.encode_binding(m, PLAN)


def _put_binding(data: bytes, offset: int, fmt: str, value) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    return _reseal(bytes(out))


def _forge_steps(bkpr: bytes, edit) -> bytes:
    """A BKPR record whose steps `edit` changed, the score kept the sum of their costs."""
    real = pa.decode_realization(bkpr)
    steps = list(edit(list(real.steps)))
    return pa._pack_realization(pa.Realization(sum(s.cost for s in steps), tuple(steps)))


def malformed() -> list[tuple[str, object, tuple[bytes, bytes, bytes], str]]:
    """(label, module, (BKPI, BKPR, BKPB), declared status): each record breaks one law."""
    from bcir.kbcir import realize
    from bcir.kbcir.cost import TargetProfile, Theta

    m, bkpi, bkpr, bkpb = _seed()
    rids_at = pa.BINDING_HEADER_SIZE
    n_res = struct.unpack_from("<I", bkpb, 8)[0]
    gens_at = rids_at + 4 * n_res
    n_gens = struct.unpack_from("<I", bkpb, 12)[0]
    plan_at = gens_at + 12 * n_gens
    body = bkpb[:-4]
    out: list = []

    def binding(label, data, status):
        out.append((label, m, (bkpi, bkpr, data), status))

    # the binding's wire laws, in order
    binding("binding: shorter than its fixed part", bkpb[:20], "BCIR_ERR_TRUNCATED")
    binding("binding: magic", _reseal(b"BKPX" + bkpb[4:]), "BCIR_ERR_MAGIC")
    binding("binding: version", _put_binding(bkpb, 4, "<H", 1), "BCIR_ERR_VERSION")
    binding("binding: CRC", body + struct.pack("<I", 0), "BCIR_ERR_CRC")
    binding("binding: flags", _put_binding(bkpb, 6, "<H", 1), "BCIR_ERR_RESERVED")
    binding("binding: reserved", _put_binding(bkpb, 24, "<Q", 1), "BCIR_ERR_RESERVED")
    binding(
        "binding: resource count over its bound",
        _put_binding(bkpb, 8, "<I", pa.REFS_MAX + 1),
        "BCIR_ERR_PLANNER",
    )
    binding(
        "binding: plan name over 65535 bytes",
        _put_binding(bkpb, 20, "<I", pa.PLAN_NAME_MAX + 1),
        "BCIR_ERR_PLANNER",
    )
    binding(
        "binding: a declared count past the end",
        _put_binding(bkpb, 12, "<I", n_gens + 1),
        "BCIR_ERR_TRUNCATED",
    )
    binding("binding: trailing bytes", _reseal(body + b"\0" + b"\0\0\0\0"), "BCIR_ERR_TRAILING")
    bad_plan = bytearray(bkpb)
    bad_plan[plan_at] = 0xFF
    binding("binding: plan name not UTF-8", _reseal(bytes(bad_plan)), "BCIR_ERR_UTF8")
    swapped = bytearray(bkpb)
    first, second = (
        bytes(swapped[gens_at : gens_at + 12]),
        bytes(swapped[gens_at + 12 : gens_at + 24]),
    )
    swapped[gens_at : gens_at + 24] = second + first
    binding("binding: RIDs not ascending", _reseal(bytes(swapped)), "BCIR_ERR_GENERATION")
    # the binding against the input
    fewer = pa.Binding(1, pa.decode_binding(bkpb).rids[:-1], pa.decode_binding(bkpb).gens, PLAN)
    binding("binding: another resource count", _encode_binding(fewer), "BCIR_ERR_PLANNER")
    b = pa.decode_binding(bkpb)
    # index 1 (declared) names index 0's RID, which has a generation: every other law holds (a
    # generation for a RID the table no longer names is legal -- a resource no claim names)
    assert b.rids[0] in {g[0] for g in b.gens}
    binding(
        "binding: two indices name one RID",
        _encode_binding(replace(b, rids=(b.rids[0], b.rids[0]) + b.rids[2:])),
        "BCIR_ERR_PLANNER",
    )
    binding(
        "binding: a declared resource without a generation",
        _encode_binding(replace(b, gens=tuple(g for g in b.gens if g[0] != b.rids[0]))),
        "BCIR_ERR_PLANNER",
    )
    undeclared = next(rid for rid in b.rids if rid not in m.resources)
    binding(
        "binding: an undeclared resource with a generation",
        _encode_binding(replace(b, gens=tuple(sorted((*b.gens, (undeclared, 0, 0)))))),
        "BCIR_ERR_PLANNER",
    )

    def forged(label, edit, status):
        out.append((label, m, (bkpi, _forge_steps(bkpr, edit), bkpb), status))

    # the hydrate's laws over the steps
    forged(
        "step: an unknown claim",
        lambda s: [replace(s[0], claim_id=999)] + s[1:],
        "BCIR_ERR_PROVENANCE",
    )
    forged("step: a claim planned twice", lambda s: s[:2] + [s[0]], "BCIR_ERR_PROVENANCE")
    forged(
        "step: a phase that is not its claim's",
        lambda s: [replace(s[0], phase_id=1)] + s[1:],
        "BCIR_ERR_PROVENANCE",
    )
    forged("step: out of phase order", lambda s: [s[2], s[0], s[1]], "BCIR_ERR_PROVENANCE")
    forged("step: a claim no step plans", lambda s: s[:2], "BCIR_ERR_PROVENANCE")
    forged(
        "step: a width the StreamPack cannot carry",
        lambda s: [replace(s[0], name=pa.NAMES.index("ux_bucket"), width=12)] + s[1:],
        "BCIR_ERR_WIDTH",
    )
    # a block field the StreamPack cannot carry: a negative offset, a negative stride
    for label, field in (("offset", {"offset": -4}), ("stride", {"stride_k": -1})):
        neg = seed_module()
        neg.phases[1].claims[0] = replace(neg.phases[1].claims[0], **field)
        args = (neg, TargetProfile.x86_avx512(), Theta.cool())
        triple = (
            pa.encode_input(*args),
            pa.encode_realization(realize.optimize(*args)),
            pa.encode_binding(neg, PLAN),
        )  # fmt: skip
        out.append((f"claim: a negative {label}", neg, triple, "BCIR_ERR_OVERFLOW"))
    return out


def _encode_binding(b: pa.Binding) -> bytes:
    """A binding record spelled from its fields (the encoder builds one from a module)."""
    import zlib

    raw = b.plan.encode("utf-8")
    body = struct.pack(
        "<4sHHIIIIQ", pa.BINDING_MAGIC, 0, 0, len(b.rids), len(b.gens), b.topo_gen, len(raw), 0
    )
    body += b"".join(struct.pack("<I", r) for r in b.rids)
    body += b"".join(struct.pack("<III", *g) for g in b.gens)
    body += raw
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


# --- the grader ------------------------------------------------------------------------------------


def measure(exe: str | None, tmp: str) -> dict[str, float]:
    """The CXX4 rows (module docstring). `exe` is the C harness (test_kplan); None leaves the
    native rail out -- every parity case then fails, since the row compares the two rails -- while
    a harness that cannot run fails every case it was handed."""
    out = {row: 0.0 for row in ROWS}
    cases = _built(parity_cases, ["hydrate.native.parity"], out)
    if exe is None:
        out["hydrate.native.parity"] += len(cases)
    else:
        try:
            native = c_hydrate_batch(exe, tmp, [triple for _, _, triple, _ in cases])
        except Exception:  # noqa: BLE001 -- a native rail that cannot run fails every case (L1)
            native = [("EXIT", b"")] * len(cases)
        for (_label, _m, _triple, want), got in zip(cases, native):
            out["hydrate.native.parity"] += got != want
    bad = _built(malformed, ["hydrate.native.malformed.accepted"], out)
    for _label, module, triple, status in bad:
        try:
            ok = oracle_pack(module, *triple)[0] == status
        except Exception:  # noqa: BLE001
            ok = False
        out["hydrate.native.malformed.accepted"] += not ok
    if exe is None:
        out["hydrate.native.malformed.accepted"] += len(bad)
    else:
        try:
            native = c_hydrate_batch(exe, tmp, [triple for _, _, triple, _ in bad])
        except Exception:  # noqa: BLE001
            native = [("EXIT", b"")] * len(bad)
        for (_label, _m, _triple, status), (got, _bytes) in zip(bad, native):
            out["hydrate.native.malformed.accepted"] += got != status
    return out


# --- the timed rows ----------------------------------------------------------------------------


def _scale_records(scale: int):
    from bcir.kbcir import realize
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(scale)
    result = realize.optimize(module, h, theta)
    records = (
        pa.encode_input(module, h, theta),
        pa.encode_realization(result),
        pa.encode_binding(module, PLAN),
    )
    return module, result, records


def native_ms(exe: str, tmp: str, scale: int = 4) -> float:
    """The native hydrate's median time per pack of the audit fixture at `scale` (4,096 claims
    at 4), in ms: the three records decoded once, the hydrate (laws, counting pass, write)
    timed, as the planner's `--bench` times its plan."""
    _module, _result, records = _scale_records(scale)
    paths = []
    for name, data in zip(("bkpi", "bkpr", "bkpb"), records):
        paths.append(os.path.join(tmp, f"audit_{scale}.{name}"))
        with open(paths[-1], "wb") as f:
            f.write(data)
    run = subprocess.run([exe, "--bench-hydrate", *paths, "5", "9"], capture_output=True,
                         text=True, timeout=600)  # fmt: skip
    if run.returncode != 0 or not run.stdout.startswith("BENCH "):
        raise RuntimeError(f"the native hydrate bench failed: {run.stdout} {run.stderr[-300:]}")
    return int(run.stdout.split()[1]) / 1e6


def native_floor(exe: str, tmp: str, scale: int = 4) -> tuple[float, int]:
    """(ms, bytes): the harness's `--bench-hydrate-floor` on the same records -- writing the
    pack's bytes once, the floor of `native_ms` -- and the size of the pack it wrote."""
    _module, _result, records = _scale_records(scale)
    paths = []
    for name, data in zip(("bkpi", "bkpr", "bkpb"), records):
        paths.append(os.path.join(tmp, f"audit_{scale}.{name}"))
        with open(paths[-1], "wb") as f:
            f.write(data)
    run = subprocess.run([exe, "--bench-hydrate-floor", *paths, "5", "9"], capture_output=True,
                         text=True, timeout=600)  # fmt: skip
    if run.returncode != 0 or not run.stdout.startswith("FLOOR "):
        raise RuntimeError(f"the native hydrate floor failed: {run.stdout} {run.stderr[-300:]}")
    fields = run.stdout.split()
    return int(fields[1]) / 1e6, int(fields[2])


def oracle_ms(scale: int = 4, rounds: int = 9) -> float:
    """The oracle's median time for the same pack: `encode(hydrate(module, result, PLAN))`."""
    import time

    from bcir.abi.streampack_abi import encode
    from bcir.gem.streampack import hydrate

    module, result, _records = _scale_records(scale)
    encode(hydrate(module, result, PLAN))  # warm
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        encode(hydrate(module, result, PLAN))
        times.append((time.perf_counter() - start) * 1e3)
    return sorted(times)[rounds // 2]
