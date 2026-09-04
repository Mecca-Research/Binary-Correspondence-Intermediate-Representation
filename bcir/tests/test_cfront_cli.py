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
    p = subprocess.run(
        [sys.executable, "-m", "bcir.frontends.cfront", *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
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
        src = _write(
            d, "s.c", "struct S { long a; char b; }; unsigned f(void){ return sizeof(struct S); }\n"
        )
        rc, out, err = _cli(["--target", "x86_64-windows", src])
        assert rc == 0, err
        assert (
            "target: x86_64-windows" in out and "cross-target" in out
        )  # LLP64, equivalence skipped


def test_cli_syntax_only_prints_a_clang_style_caret():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "bad.c", "int f(void){ return zzz + 1; }\n")
        rc, out, err = _cli(["-fsyntax-only", src])
        assert rc == 1
        assert "use of undeclared identifier 'zzz'" in err and "^" in err  # diagnostics on stderr


def test_cli_emit_json_is_machine_readable():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "bad.c", "int f(void){ return 1 }\n")  # missing ';'
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


# --- Phase 3 project orchestration: multi-file verdict, compile database, dependency output ---


def test_cli_project_verdict_clean_over_multiple_files():
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.c", "unsigned f(unsigned x){ return x + 1u; }\n")
        b = _write(d, "b.c", "unsigned g(unsigned x){ return x * 2u; }\n")
        rc, out, err = _cli([a, b])
        assert rc == 0, err
        assert "project: CLEAN (2 files)" in out, out


def test_cli_project_verdict_partial_fallback():
    # one clean unit + one outside the subset under --fallback: the project verdict names the split
    # and the exit code keeps the per-unit fallback contract (2).
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.c", "unsigned f(unsigned x){ return x + 1u; }\n")
        b = _write(
            d, "b.c", "unsigned k(void);\nunsigned g = k();\nunsigned h(void){ return g; }\n"
        )
        rc, out, err = _cli(["--fallback", a, b])
        assert rc == 2, (rc, err)
        assert "PARTIAL-FALLBACK (1/2" in out, out


def test_cli_dep_output_lists_header_dependencies():
    # -M prints a make rule naming the unit and every on-disk header the preprocessor resolved;
    # -MF writes the same rules to a file (build-system integration, Phase 3).
    with tempfile.TemporaryDirectory() as d:
        _write(d, "regs.h", "typedef unsigned int reg32;\n")
        u = _write(d, "u.c", '#include "regs.h"\nreg32 f(reg32 x){ return x; }\n')
        rc, out, err = _cli(["-M", u])
        assert rc == 0, err
        line = [ln for ln in out.splitlines() if ln.startswith("u.o:")][0]
        assert u in line and os.path.join(d, "regs.h") in line, line
        df = os.path.join(d, "u.d")
        rc, out, err = _cli(["-MF", df, u])
        assert rc == 0 and os.path.exists(df)
        with open(df, encoding="utf-8") as f:
            assert "regs.h" in f.read()


def test_cli_compile_database_drives_per_entry_flags():
    # -p compile_commands.json: every entry compiles with ITS OWN flags -- one entry uses the
    # "arguments" form (with a -D the source requires), the other the "command" string form.
    with tempfile.TemporaryDirectory() as d:
        a = _write(
            d,
            "a.c",
            "#if NEED != 2\n#error need NEED=2\n#endif\n"
            "unsigned f(unsigned x){ return x + NEED; }\n",
        )
        b = _write(d, "b.c", "unsigned g(unsigned x){ return x * 3u; }\n")
        db = [
            {"directory": d, "file": "a.c", "arguments": ["cc", "-c", "-DNEED=2", "a.c"]},
            {"directory": d, "file": "b.c", "command": "cc -c b.c"},
        ]
        _write(d, "compile_commands.json", json.dumps(db))
        rc, out, err = _cli(["-p", d])
        assert rc == 0, (err, out)
        assert "project: CLEAN (2 files)" in out, out


def test_cli_rejects_missing_option_arguments_and_ambiguous_compile_databases():
    for option in ("-I", "-D", "-U", "-MF", "-MT", "-p", "--target", "--r21", "-o"):
        rc, _out, err = _cli([option])
        assert rc == 2 and "requires an argument" in err, (option, rc, err)

    with tempfile.TemporaryDirectory() as d:
        _write(d, "ok.c", "int f(void){ return 0; }\n")
        valid = json.dumps([{"directory": d, "file": "ok.c", "arguments": ["cc", "-c", "ok.c"]}])
        duplicate = valid.replace('"directory":', '"directory": "forged", "directory":', 1)
        db = _write(d, "compile_commands.json", duplicate)
        rc, _out, err = _cli(["-p", db])
        assert rc == 2 and "duplicate JSON key" in err

        _write(
            d,
            "compile_commands.json",
            json.dumps(
                [
                    {"directory": d, "file": "ok.c", "arguments": ["cc", "-I"]},
                ]
            ),
        )
        rc, _out, err = _cli(["-p", d])
        assert rc == 2 and "requires an argument" in err

        _write(
            d,
            "compile_commands.json",
            json.dumps(
                [
                    {
                        "directory": d,
                        "file": "ok.c",
                        "arguments": ["cc", "ok.c"],
                        "command": "cc ok.c",
                    },
                ]
            ),
        )
        rc, _out, err = _cli(["-p", d])
        assert rc == 2 and "both command and arguments" in err
