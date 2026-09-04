"""The C StreamPack executor (runtime/c/bcir_exec.c): Python<->C parity + robustness.

Roadmap (BCIR_MASTER_ROADMAP.md §5.2 / §6): port the deterministic executor
(`bcir/gem/execute.py`) to C so the StreamPack is a no-Python hot artifact a driver runs
end to end. These tests pin the C executor's **dispatch order + per-phase telemetry**
against the oracle `gem.execute` over a Python-encoded pack (the executor's parity gate,
the analog of test_c_runtime for the decoder), and exercise the bounds/`NOSPACE` paths.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.abi import encode
from bcir.examples import matmul_tiled, multi_histogram, scan, vector_add
from bcir.gem import execute, hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _out_of_order_module():
    """One phase whose claims are *declared* 30, 10, 20 -- the executor must dispatch
    them ascending by id (10, 20, 30), matching gem.execute's intra-phase id sort."""
    m = Module(name="ooo")
    for rid in range(1, 10):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    claims = [
        Claim(
            id=i,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(1, 2),
            wr=(3,),
            op="vector.add",
            domain=Domain.RAM,
        )
        for i in (30, 10, 20)
    ]
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _build_harness(tmp):
    cc = _cc()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_exec")
    r = subprocess.run(
        [
            cc,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            _RUNTIME_C,
            os.path.join(_RUNTIME_C, "bcir_exec.c"),
            os.path.join(_RUNTIME_C, "bcir_runtime.c"),
            os.path.join(_RUNTIME_C, "test_exec.c"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return exe


def _run(exe, tmp, pack: bytes):
    path = os.path.join(tmp, "pack.bin")
    with open(path, "wb") as f:
        f.write(pack)
    r = subprocess.run([exe, path], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    order, phases = [], []
    for line in r.stdout.strip().splitlines():
        if line.startswith("order:"):
            order = [int(x) for x in line.split()[1:]]
        elif line.startswith("phase "):
            _, pid, sched, ex = line.split()
            phases.append((int(pid), int(sched), int(ex)))
    return order, phases


def test_c_executor_builds_freestanding():
    cc = _cc()
    if cc is None:
        return
    for std in ("c11", "c23"):
        r = subprocess.run(
            [
                cc,
                "-ffreestanding",
                "-nostdlib",
                f"-std={std}",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                "-c",
                os.path.join(_RUNTIME_C, "bcir_exec.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{std}: {r.stderr}"


def test_dispatch_order_and_telemetry_match_gem_execute():
    """The headline parity: over a real pack, the C executor's dispatch order + per-phase
    (scheduled, executed) + phase order equal bcir.gem.execute on the same module."""
    if _cc() is None:
        return
    mods = {
        "vector_add": vector_add(),
        "multi_histogram": multi_histogram(),
        "matmul_tiled": matmul_tiled(),
        "scan": scan(),
        "out_of_order": _out_of_order_module(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        assert exe is not None
        for name, m in mods.items():
            pack = encode(hydrate(m, optimize(m, AVX, COOL)))
            c_order, c_phases = _run(exe, tmp, pack)
            er = execute(m)
            assert c_order == er.order, f"{name}: order {c_order} != {er.order}"
            assert [p[0] for p in c_phases] == er.phase_order, f"{name}: phase order"
            py_phases = [(ps.phase_id, ps.scheduled, ps.executed) for ps in er.phases]
            assert c_phases == py_phases, f"{name}: telemetry {c_phases} != {py_phases}"


def test_intra_phase_sort_is_by_claim_id():
    """The declared order is 30,10,20; the executor must dispatch 10,20,30 (id order)."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        pack = encode(hydrate(_out_of_order_module(), optimize(_out_of_order_module(), AVX, COOL)))
        order, phases = _run(exe, tmp, pack)
        assert order == [10, 20, 30]
        assert phases == [(0, 3, 3)]


def test_malformed_pack_is_rejected_not_crashed():
    """A truncated / garbage pack returns a status (the harness prints ERR), never an
    out-of-bounds read (ASan/UBSan-fuzzed separately in tools/c/fuzz_streampack.sh)."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        good = encode(hydrate(vector_add(), optimize(vector_add(), AVX, COOL)))
        for bad in (b"", b"BSPK", good[:40], good[:-1], bytes(len(good))):
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as f:
                f.write(bad)
            r = subprocess.run([exe, path], capture_output=True, text=True)
            # Either a clean ERR <status> line or (for the rare valid prefix) a clean run;
            # the point is no crash / nonzero-from-signal.
            assert r.returncode in (0, 1), (bad[:8], r.returncode, r.stdout, r.stderr)
