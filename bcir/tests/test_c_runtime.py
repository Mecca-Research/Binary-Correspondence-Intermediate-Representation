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
