"""Python<->C dual-rail parity for the plug-in C compiler (runtime/c/bcir_cfront.c).

The C frontend (`runtime/c/bcir_cfront.c`) is the production port of the Python prototype
(`bcir/frontends/cfront/`). This gate lowers the SAME register-map source through both rails and
asserts they produce the same RID-independent structural summary (claim count, MMIO loads, bitfield
extracts, constants, binops, verifier status) — the C twin of the oracle, exactly as `bcir_exec.c`
twins `gem.execute`. Toolchain-gated: it compiles `bcir_cfront.c` with the host C compiler, so it
self-skips in the quick tier and runs for real under c-runtime / thorough.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.model import Domain

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_FIXTURE = os.path.join(_C, "cfront_regmap.c")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _oracle_summary(src: str) -> str:
    r = compile_unit(src, check_clang=False)
    fn = next(reversed(r.lowered.functions))
    lf = r.lowered.functions[fn]
    mmio = sum(1 for c in lf.claims if c.op == "c.load" and c.domain == Domain.MMIO)
    bf = sum(1 for c in lf.claims if c.op == "c.bf.get")
    kn = sum(1 for c in lf.claims if c.op == "c.const")
    binop = sum(1 for c in lf.claims if c.op.startswith("c.bin."))
    return (f"claims={len(lf.claims)} mmio={mmio} bf={bf} const={kn} binop={binop} "
            f"ok={1 if r.is_clean else 0}")


def _c_summary(fixture: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "t")
        build = subprocess.run(
            [_CC, "-std=c23", "-O2", "-Wall", "-Wextra", "-I", _C,
             os.path.join(_C, "bcir_cfront.c"), os.path.join(_C, "test_cfront.c"), "-o", exe],
            capture_output=True, text=True)
        if build.returncode != 0:                          # fall back to C11 if c23 unsupported
            build = subprocess.run(
                [_CC, "-std=c11", "-O2", "-I", _C, os.path.join(_C, "bcir_cfront.c"),
                 os.path.join(_C, "test_cfront.c"), "-o", exe], capture_output=True, text=True)
            assert build.returncode == 0, f"C frontend build failed:\n{build.stderr}"
        run = subprocess.run([exe, fixture], capture_output=True, text=True)
        assert run.returncode == 0, f"C frontend run failed:\n{run.stdout}\n{run.stderr}"
        return run.stdout.strip().splitlines()[0]


def test_c_frontend_builds_warning_clean():
    if not _CC:
        return                                             # quick tier: toolchain hidden
    b = subprocess.run([_CC, "-std=c23", "-Wall", "-Wextra", "-Werror", "-I", _C, "-c",
                        os.path.join(_C, "bcir_cfront.c"), "-o", os.devnull],
                       capture_output=True, text=True)
    if b.returncode != 0 and "c23" in (b.stderr or ""):    # older clang without c23
        b = subprocess.run([_CC, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", _C, "-c",
                            os.path.join(_C, "bcir_cfront.c"), "-o", os.devnull],
                           capture_output=True, text=True)
    assert b.returncode == 0, f"C frontend has warnings:\n{b.stderr}"


def test_python_c_register_map_parity():
    src = open(_FIXTURE, encoding="utf-8").read()
    oracle = _oracle_summary(src)
    # the C frontend lowers a clean register map: an MMIO load + 3 bitfield extracts.
    assert "mmio=1" in oracle and "bf=3" in oracle and "ok=1" in oracle
    if not _CC:
        return                                             # quick tier: the C build is deferred
    assert _c_summary(_FIXTURE) == oracle, "Python and C frontends diverged on the register map"


def test_c_frontend_rejects_an_unknown_field():
    if not _CC:
        return
    bad = ("struct r { volatile uint32_t s; };\n"
           "uint32_t f(volatile struct r *p) { return p->missing; }\n")
    with tempfile.TemporaryDirectory() as d:
        exe, fx = os.path.join(d, "t"), os.path.join(d, "bad.c")
        open(fx, "w").write(bad)
        subprocess.run([_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_cfront.c"),
                        os.path.join(_C, "test_cfront.c"), "-o", exe], check=True,
                       capture_output=True)
        out = subprocess.run([exe, fx], capture_output=True, text=True).stdout
        assert "PARSE-ERR" in out and "field" in out      # the C frontend diagnoses it
