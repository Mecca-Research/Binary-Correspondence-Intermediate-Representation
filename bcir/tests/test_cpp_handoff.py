"""The C<->C++ hand-off seam scaffold (`runtime/cpp/`): compile + round-trip identity.

The seam is the boundary between BCIR's deterministic single-node C/IR rail and the C++
layer ABOVE it (dynamic graph topology + distributed MPI/NCCL orchestration); the contract
is `docs/languages/CPP_HANDOFF_BOUNDARY.md`. This test exercises the REAL part of the scaffold: the
`SingleNodeOrchestrator` consumes a StreamPack the existing C/IR path produces, re-enters
the existing freestanding C decoder, and its dispatch order must equal the DIRECT C/IR
decode of the same artifact (round-trip identity). The dynamic-graph + distributed backends
are documented stubs (no real MPI/NCCL) and are not exercised here beyond their sharding.

Toolchain-gated like the other C-runtime parity tests: the quick tier still checks the
artifact the seam consumes is producible + decodable on the Python rail; the C++ compile +
round-trip runs under c-runtime/thorough (where a C++ compiler is visible).
"""

import os
import shutil
import subprocess
import tempfile

from bcir.abi import decode, encode
from bcir.examples import multi_histogram, vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CPP = os.path.join(_ROOT, "runtime", "cpp")
_CXX = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def _pack(module_fn) -> bytes:
    m = module_fn()
    r = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    return encode(hydrate(m, r))


def _build(d: str) -> str:
    exe = os.path.join(d, "test_orch")
    for std in ("c++17", "c++14"):
        b = subprocess.run(
            [_CXX, f"-std={std}", "-O2", "-Wall", "-Wextra", "-I", _C,
             os.path.join(_CPP, "bcir_orchestrator.cpp"),
             os.path.join(_CPP, "test_orchestrator.cpp"),
             os.path.join(_C, "bcir_runtime.c"), "-o", exe],
            capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"C++ hand-off scaffold build failed:\n{b.stderr}")


def test_artifact_the_seam_consumes_is_producible_and_decodable():
    """Quick-tier coverage (no compiler needed): the StreamPack the C++ seam consumes is
    produced by the existing C/IR path and decodes on the Python rail -- the seam input
    contract is well-formed independent of the C++ build."""
    for fn in (multi_histogram, vector_add):
        data = _pack(fn)
        assert data[:4] == b"BSPK", "the hand-off artifact must be a StreamPack"
        pack = decode(data)            # the Python rail decodes it (round-trippable artifact)
        assert pack.segments, "a non-trivial artifact has segments to dispatch"


def test_single_node_orchestrator_round_trip_identity():
    """The REAL seam: the C++ single-node Orchestrator's dispatch order == the direct C/IR
    decode of the same artifact (round-trip identity), on two fixtures."""
    if not _CXX:                       # quick tier: deferred to where a C++ compiler is visible.
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build(d)
        for fn in (multi_histogram, vector_add):
            pack_path = os.path.join(d, "pack.bin")
            open(pack_path, "wb").write(_pack(fn))
            r = subprocess.run([exe, pack_path], capture_output=True, text=True)
            assert r.returncode == 0, f"seam round-trip failed: {r.stdout}{r.stderr}"
            assert r.stdout.startswith("OK "), f"unexpected seam output: {r.stdout!r}"


def test_corrupted_artifact_rejected_at_the_boundary():
    """The two-truth quarantine across the seam: admit() carries the C/IR verifier's verdict
    (it does not re-derive legality), so a corrupted artifact is rejected at the boundary."""
    if not _CXX:
        return
    with tempfile.TemporaryDirectory() as d:
        drv = os.path.join(d, "reject.cpp")
        open(drv, "w").write(
            '#include <cstdio>\n#include <cstdint>\n#include <vector>\n'
            '#include "bcir_orchestrator.hpp"\n'
            'int main(int c, char** v){ if(c<2) return 2;\n'
            '  std::FILE* f=std::fopen(v[1],"rb"); if(!f) return 2;\n'
            '  std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);\n'
            '  std::vector<std::uint8_t> b(n>0?(size_t)n:0);\n'
            '  if(n>0 && std::fread(b.data(),1,b.size(),f)!=b.size()){std::fclose(f);return 2;}\n'
            '  std::fclose(f);\n'
            '  bcir::SingleNodeOrchestrator s;\n'
            '  bcir_status st=s.admit(b.data(), b.size());\n'
            '  return st==BCIR_OK ? 1 : 0; }\n')
        exe = os.path.join(d, "reject")
        built = False
        for std in ("c++17", "c++14"):
            b = subprocess.run(
                [_CXX, f"-std={std}", "-O2", "-I", _C, "-I", _CPP, drv,
                 os.path.join(_CPP, "bcir_orchestrator.cpp"),
                 os.path.join(_C, "bcir_runtime.c"), "-o", exe],
                capture_output=True, text=True)
            if b.returncode == 0:
                built = True
                break
        assert built, f"reject harness build failed:\n{b.stderr}"
        # a clean pack admits (rc 1 from the probe == ADMITTED); a corrupted one is rejected (rc 0).
        clean = os.path.join(d, "clean.bin")
        open(clean, "wb").write(_pack(multi_histogram))
        assert subprocess.run([exe, clean]).returncode == 1, "clean pack should admit"
        bad = bytearray(open(clean, "rb").read())
        bad[20] ^= 0xFF                # corrupt n_segments -> breaks the CRC
        badp = os.path.join(d, "bad.bin")
        open(badp, "wb").write(bad)
        assert subprocess.run([exe, badp]).returncode == 0, "corrupted pack must be rejected"
