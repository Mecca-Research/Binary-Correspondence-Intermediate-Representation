"""BCIR Make's C twin (BUILD-8): `runtime/c/bcir_make.c` judges and plans as the oracle does, byte
for byte, over a generated corpus (tools/build/make_parity.py, the gate; CTest build-make-parity
runs it over 400 cases and the rails' own BCIRfile). Here a smaller corpus runs where a C compiler
is visible, and the gate is held to failing: a twin that words one finding differently, and one
that schedules a wave one target wider, must each be caught (docs/security/laws.md L2, L11).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import tempfile
from pathlib import Path

from bcir.toolchain import host_c_compiler

_ROOT = Path(__file__).resolve().parents[2]
_C = _ROOT / "runtime" / "c"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}_under_test", _ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(cc: str, source: str, out: Path) -> bool:
    unit = out.with_suffix(".c")
    unit.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [cc, "-std=c11", "-O1", f"-I{_C}", str(unit), str(_C / "bcir_sha256.c"), "-o", str(out)],
        capture_output=True,
    )
    return done.returncode == 0


def _parity(twin: Path, cases: int) -> tuple[int, str]:
    parity = _load("make_parity")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = parity.main(["--twin", str(twin), "--cases", str(cases), "--no-rails", "--seed", "8"])
    return rc, out.getvalue()


def test_the_twin_plans_as_the_oracle_and_the_gate_catches_a_twin_that_does_not():
    cc = host_c_compiler()
    if cc is None or os.name != "posix":
        return  # no compiler visible (a tier that hides the toolchain), or no POSIX host
    source = (_C / "bcir_make.c").read_text(encoding="utf-8")
    faults = {
        "a finding worded otherwise": (
            "which it neither reads nor writes",
            "which it does not claim",
        ),
        "a wave one wider": (
            "while (w < n && waves[w].n >= workers) w++;",
            "while (w < n && waves[w].n > workers) w++;",
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        twin = Path(tmp) / "bcir-make"
        assert _build(cc, source, twin), "the twin does not compile"
        rc, text = _parity(twin, 84)
        assert rc == 0 and "make-parity: PASS" in text, text
        for what, (old, new) in faults.items():
            assert source.count(old) == 1, what
            broken = Path(tmp) / f"broken{len(what)}"
            assert _build(cc, source.replace(old, new), broken), what
            rc, text = _parity(broken, 84)
            assert rc == 1 and "differ" in text, (what, text[-600:])
