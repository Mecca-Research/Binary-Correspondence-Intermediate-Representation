"""Phase 4 — segment 5 (toolchain integration): the bcir-cfront driver exposes the phase's
capabilities -- the cross-platform ABI target, Clang-style diagnostics + machine-readable JSON, and
the LLVM-backend fallback contract -- through real command-line flags."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cli(args):
    p = subprocess.run([sys.executable, "-m", "bcir.frontends.cfront", *args],
                       capture_output=True, text=True, cwd=_ROOT)
    return p.returncode, p.stdout, p.stderr


def _write(d, name, text):
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def test_cli_default_compile_reports_target_and_clean():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "ok.c", "unsigned f(unsigned x){ return x * 2u + 1u; }\n")
        rc, out, err = _cli([src])
        assert rc == 0, err
        assert "R1-R18: CLEAN" in out and "target: x86_64-linux" in out


def test_cli_target_selects_the_abi():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "s.c", "struct S { long a; char b; }; unsigned f(void){ return sizeof(struct S); }\n")
        rc, out, err = _cli(["--target", "x86_64-windows", src])
        assert rc == 0, err
        assert "target: x86_64-windows" in out and "cross-target" in out   # LLP64, equivalence skipped


def test_cli_syntax_only_prints_a_clang_style_caret():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "bad.c", "int f(void){ return zzz + 1; }\n")
        rc, out, err = _cli(["-fsyntax-only", src])
        assert rc == 1
        assert "use of undeclared identifier 'zzz'" in err and "^" in err   # diagnostics on stderr


def test_cli_emit_json_is_machine_readable():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "bad.c", "int f(void){ return 1 }\n")            # missing ';'
        rc, out, _ = _cli(["--emit-json", src])
        obj = json.loads(out)
        assert rc == 1 and obj[0]["severity"] == "error" and obj[0]["phase"] == "parse"
        assert "line" in obj[0] and "column" in obj[0]


def test_cli_fallback_reports_unsupported_then_exits_2():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "fb.c", "int g; int f(void){ static int x = g; return x; }\n")
        rc, out, err = _cli(["--fallback", src])
        assert rc == 2 and "fallback to LLVM" in err


def test_cli_unknown_target_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "ok.c", "int f(void){ return 0; }\n")
        rc, out, err = _cli(["--target", "sparc-solaris", src])
        assert rc == 2 and "unknown --target" in err
