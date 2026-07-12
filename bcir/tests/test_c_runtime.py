"""Freestanding C StreamPack runtime parity (Phase 8): Python encodes, C decodes."""

import os
import subprocess
import tempfile
from shutil import which

from bcir.abi import encode
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_C_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime", "c")


def _pack():
    m = vector_add(1024)
    return hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))


def test_runtime_is_freestanding():
    # The runtime itself must compile with no libc / freestanding.
    clang = which("clang")
    if clang is None:
        return
    r = subprocess.run(
        [clang, "-ffreestanding", "-nostdlib", "-std=c11", "-Wall", "-Wextra",
         "-c", os.path.join(_C_DIR, "bcir_runtime.c"), "-o", os.devnull],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_python_encodes_c_decodes():
    clang = which("clang")
    if clang is None:
        return  # skip cleanly without a C compiler
    pack = _pack()
    seg = pack.segments[0]
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "test_runtime")
        build = subprocess.run(
            [clang, "-std=c11", "-O2",
             os.path.join(_C_DIR, "bcir_runtime.c"),
             os.path.join(_C_DIR, "test_runtime.c"),
             "-I", _C_DIR, "-o", exe],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        blob = os.path.join(d, "pack.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)

        # Cross-language ABI fidelity: the C decode matches the Python encode.
        assert "OK" in run.stdout
        assert int(out["version"]) == 1
        assert int(out["data_gen"]) == pack.data_gen
        assert int(out["n_segments"]) == len(pack.segments)
        assert int(out["walked"]) == len(pack.segments)
        assert int(out["seg0.claim_id"]) == seg.claim_id
        assert int(out["seg0.lane"]) == int(seg.lane)
        assert int(out["seg0.width"]) == seg.width
        assert out["seg0.opcode"] == seg.opcode
        assert int(out["seg0.read0"]) == seg.reads[0]
        assert int(out["seg0.write0"]) == seg.writes[0]
        assert int(out["pipeline_depth"]) == 1  # v1 pack: single phase in flight


def test_python_encodes_v2_c_decodes():
    # StreamPack v2 (append-only): the C runtime accepts the new version and
    # reads the pipeline contract; the segment stream stays v1-shaped.
    clang = which("clang")
    if clang is None:
        return
    from bcir.gem import hydrate_pipelined
    from bcir.kbcir import optimize as _opt
    m = vector_add(1024)
    pack = hydrate_pipelined(m, _opt(m, TargetProfile.x86_avx512(), Theta.cool()),
                             depth=2)
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "test_runtime")
        build = subprocess.run(
            [clang, "-std=c11", "-O2",
             os.path.join(_C_DIR, "bcir_runtime.c"),
             os.path.join(_C_DIR, "test_runtime.c"),
             "-I", _C_DIR, "-o", exe],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        blob = os.path.join(d, "pack.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
        assert int(out["version"]) == 2
        assert int(out["pipeline_depth"]) == 2
        assert int(out["walked"]) == len(pack.segments)


def _build_runtime_harness(clang, d, name="test_runtime", srcs=("bcir_runtime.c", "test_runtime.c")):
    exe = os.path.join(d, name)
    build = subprocess.run(
        [clang, "-std=c23", "-O2", "-Wall", "-Wextra", *[os.path.join(_C_DIR, s) for s in srcs],
         "-I", _C_DIR, "-o", exe],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe


def _v3_pack():
    from bcir.gem.streampack import LaneSegment, Prefetch, StreamPack, TraceNote
    from bcir.model import Lane
    p = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    p.prefetches.append(Prefetch("pf0", 4, (10, 11)))
    p.segments.append(LaneSegment(
        name="seg0", claim_id=1000, phase_id=0, lane=Lane.GGG, width=16,
        opcode="reduce.add", reads=(10, 11), writes=(12,), prefetch="pf0",
        dispatch="pim", channel="nvidia_ptx"))
    p.trace_notes.append(TraceNote(claim_id=1000))
    return p


def test_python_encodes_v3_c_decodes_dispatch_channel():
    # StreamPack v3 (append-only): the C runtime accepts v3 and reads the on-wire
    # segment dispatch (pim) + channel (nvidia_ptx) -- the routing fields are now
    # inside the CRC, so a mutation is no longer invisible to byte-identity.
    clang = which("clang")
    if clang is None:
        return
    pack = _v3_pack()
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(clang, d)
        blob = os.path.join(d, "v3.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
        assert int(out["version"]) == 3
        assert int(out["seg0.dispatch"]) == 1            # pim
        assert out["seg0.channel"] == "nvidia_ptx"
        assert out["seg0.prefetch"] == "pf0"


def test_c_decode_equals_python_decode_differential():
    # C-decode == Python-decode on every field, over v1 / v2 / v3 packs (the
    # lane-asymmetry class: a value the Python decode normalizes/raises on must
    # decode identically on the C rail).
    clang = which("clang")
    if clang is None:
        return
    from bcir.abi import decode as py_decode
    from bcir.gem import hydrate_pipelined
    m = vector_add(1024)
    packs = {
        1: _pack(),
        2: hydrate_pipelined(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()), depth=2),
        3: _v3_pack(),
    }
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(clang, d)
        for want_ver, pack in packs.items():
            blob = os.path.join(d, f"p{want_ver}.bin")
            with open(blob, "wb") as f:
                f.write(encode(pack))
            run = subprocess.run([exe, blob], capture_output=True, text=True)
            assert run.returncode == 0, run.stdout + run.stderr
            c = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
            py = py_decode(encode(pack))
            s = py.segments[0]
            assert int(c["version"]) == want_ver
            assert int(c["pipeline_depth"]) == py.pipeline_depth
            assert int(c["map_gen"]) == py.map_gen
            assert int(c["data_gen"]) == py.data_gen
            assert int(c["n_segments"]) == len(py.segments)
            assert int(c["seg0.claim_id"]) == s.claim_id
            assert int(c["seg0.lane"]) == int(s.lane)
            assert int(c["seg0.width"]) == s.width
            assert c["seg0.opcode"] == s.opcode
            assert int(c["seg0.read0"]) == s.reads[0]
            assert int(c["seg0.write0"]) == s.writes[0]
            assert int(c["seg0.dispatch"]) == {"core": 0, "pim": 1}[s.dispatch]
            assert c["seg0.channel"] == s.channel        # "host" on v1/v2, set on v3
            assert c["seg0.prefetch"] == (s.prefetch or "")


def test_c_rejects_crc_valid_semantically_corrupt_packs():
    # The freestanding trust boundary (R10/R11 + lane/width/dispatch range): each CRC-VALID
    # but semantically-corrupt pack is REJECTED by bcir_sp_verify_semantic + the checked
    # executor, where the bare decoder used to accept it silently. Mirrors verify_pack.
    clang = which("clang")
    if clang is None:
        return
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mutator = os.path.join(repo, "tools", "c", "streampack_corrupt.py")
    want = {
        "dangling_claim": "BCIR_ERR_PROVENANCE",
        "swapped_rid": "BCIR_ERR_PROVENANCE",
        "out_of_range_lane": "BCIR_ERR_LANE",
        "out_of_range_width": "BCIR_ERR_WIDTH",
        "unresolved_prefetch": "BCIR_ERR_PROVENANCE",
        "stale_generation": "BCIR_ERR_STALE",
        "tampered_dispatch": "BCIR_ERR_DISPATCH",
    }
    env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(
            clang, d, name="test_sem",
            srcs=("bcir_exec.c", "bcir_runtime.c", "test_streampack_semantic.c"))
        for kind, code in want.items():
            bin_path = os.path.join(d, f"{kind}.bin")
            meta = subprocess.run([sys.executable, mutator, kind, bin_path],
                                  capture_output=True, text=True, env=env)
            assert meta.returncode == 0, meta.stderr
            _k, _n, mapg, datag = meta.stdout.split()
            argv = [exe, bin_path]
            if kind == "stale_generation":
                argv += [mapg, datag]                    # tell the C rail the live generation
            run = subprocess.run(argv, capture_output=True, text=True)
            # harness prints:  crc=<name>\nstatus=<n> name=<NAME>\nexec=<NAME>
            fields = {}
            for line in run.stdout.splitlines():
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        fields[k] = v
            # the pack still passes CRC (it is a CRC-fixed corruption) ...
            assert fields.get("crc") == "BCIR_OK", (kind, run.stdout)
            # ... but the semantic verifier AND the checked executor both reject it.
            assert fields.get("name") == code, (kind, run.stdout)
            assert fields.get("exec") == code, (kind, run.stdout)
            assert run.returncode == 1, (kind, run.stdout)


def test_c_planner_rejects_unrepresentable_cost_instead_of_wrapping():
    clang = which("clang")
    if clang is None:
        return
    source = (
        '#include <stdint.h>\n#include <string.h>\n#include "bcir_plan.h"\n'
        'int main(void){bcir_claim c;bcir_func f;bcir_plan_step s;bcir_plan p;'
        'memset(&c,0,sizeof c);memset(&f,0,sizeof f);'
        'c.opcode=BCIR_OP_ATOMIC_ADD;c.domain=BCIR_DOM_MMIO;c.count=UINT32_MAX;'
        'f.claims=&c;f.n_claims=1;'
        'return bcir_plan_func(&f,&s,1,&p)==BCIR_ERR_OVERFLOW?0:1;}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "plan_overflow.c")
        with open(driver, "w", encoding="utf-8") as f:
            f.write(source)
        exe = os.path.join(d, "plan_overflow")
        build = subprocess.run([clang, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", _C_DIR,
                                os.path.join(_C_DIR, "bcir_plan.c"), driver, "-o", exe],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        assert subprocess.run([exe], capture_output=True).returncode == 0
