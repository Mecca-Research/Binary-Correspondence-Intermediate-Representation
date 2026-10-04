"""The UBSan witness gate (`tools/testing/ubsan_witnesses.py`, CF-UBGATE), driven into each verdict it gives.

The gate runs the whole cfront suite, and that run belongs to the CI step that owns it (docs/security/laws.md L19).
These tests drive its parts: the predicate that decides what a witness is, the sanitized build's arguments, a shim
that must fail a witness running undefined behaviour and must leave a defined one's output alone, and the driver's
verdicts -- a report and a failing test are findings, no UBSan runtime is an honest skip unless `--require-ubsan`
asked for one, and a run that sanitized nothing is no pass (L1, L2). The tests that build need a UBSan runtime that
reports here (`ubsan_witnesses._engines`) and return without one; the rest need nothing.

Every test is a module-level function: `run_all` discovers `test_*` names (see test_red_sweep.py).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.testing import ubsan_witnesses as uw  # noqa: E402

_SCRIPT = str(_REPO_ROOT / "tools" / "testing" / "ubsan_witnesses.py")
# a witness whose ORIGINAL runs undefined behaviour (signed overflow) and one whose arithmetic wraps as C defines
_UNDEFINED = (
    "#include <stdio.h>\n"
    "int main(int argc, char **argv) { volatile int x = 2147483647; (void)argv; x = x + argc;"
    ' printf("%d\\n", x < 0); return 0; }\n'
)
_DEFINED = (
    "#include <stdio.h>\n"
    'int main(void) { volatile unsigned x = 4294967295u; x = x + 1u; printf("%u\\n", x); return 0; }\n'
)
# CI-ARM's own defect: a negative `double` converted to `uint32_t` (C11 6.3.1.4p1), wrapped on x86-64, saturated on
# AArch64 -- `float-cast-overflow`, which `-fsanitize=undefined` alone does not name under GCC
_FLOAT_CAST = (
    "#include <stdint.h>\n#include <stdio.h>\n"
    'int main(void) { volatile double d = -130070.0; uint32_t u = (uint32_t)d; printf("%u\\n", u); return 0; }\n'
)
# a witness's own shape: an original compared with an emit that overflows alike -- GCC above -O0 folds the comparison
# and deletes both checks, so under GCC this reports only unoptimized (`ubsan_witnesses.LEVEL`)
_FOLDED = (
    "#include <stdio.h>\n"
    "static unsigned f(unsigned a) { int x = (int)a; return (unsigned)(x * x - x); }\n"
    "static unsigned bcir_f(unsigned a) { int x = (int)a, t = x * x; t = t - x; return (unsigned)t; }\n"
    "int main(int argc, char **argv) { unsigned a = 65536u + (unsigned)argc; (void)argv;\n"
    '  if (f(a) != bcir_f(a)) { printf("MISMATCH\\n"); return 1; } printf("MATCH\\n"); return 0; }\n'
)
# an original that discards an overflowing value (`x * x;`): GCC's front end drops the statement unchecked at every
# level, -O0 included, so only Clang's UBSan reports it -- the gate's instrument (`ubsan_witnesses.LEVEL`)
_DISCARDED = (
    "#include <stdio.h>\n"
    "static unsigned f(unsigned a) { int x = (int)a; x * x; return a; }\n"
    "static unsigned bcir_f(unsigned a) { return a; }\n"
    "int main(int argc, char **argv) { unsigned a = 65536u + (unsigned)argc; (void)argv;\n"
    '  if (f(a) != bcir_f(a)) { printf("MISMATCH\\n"); return 1; } printf("MATCH\\n"); return 0; }\n'
)
#: `check_tests.py --require` reads this in place of the cfront suite's `_CC` when the driver runs this module's helpers
_ALWAYS = True


class _Engines:
    """Truthy where a UBSan runtime reports here, probed when read: what `check_tests.py --require` asks of this
    module when `tools/testing/faults/ubsan-witnesses.json` runs it, for the tests that build return without one."""

    def __bool__(self) -> bool:
        return bool(uw._engines())


_UBSAN = _Engines()


def _build_and_run(text: str) -> subprocess.CompletedProcess:
    """Build `text` at -O2 with the first C compiler on PATH -- the gate's shim, under the gate -- and run it."""
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    assert cc, "no C compiler"
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "witness.c"), os.path.join(d, "witness")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(text)
        b = subprocess.run(
            [cc, "-std=c11", "-O2", "-Wall", src, "-o", exe], capture_output=True, text=True
        )
        assert b.returncode == 0, b.stderr
        return subprocess.run([exe], capture_output=True, text=True)


def _witness_undefined() -> None:
    """A run-as-the-original witness over an original that overflows `int` (the driver's subject, not a test)."""
    assert _build_and_run(_UNDEFINED).returncode == 0


def _witness_defined() -> None:
    run = _build_and_run(_DEFINED)
    assert run.returncode == 0 and run.stdout == "0\n", (run.returncode, run.stdout, run.stderr)


def _witness_none() -> None:
    """A test that builds no witness: under the gate it sanitizes nothing."""


def _witness_fails() -> None:
    raise AssertionError("the emit is not the original")


def test_a_witness_link_is_an_executable_built_from_sources_outside_the_repository():
    """The declared scope: a link of an executable (`-o`, no `-c`/`-S`/`-E`/`-shared`/`-fsyntax-only`/`--analyze`/
    `-M`) from C sources outside the repository or among its `cfront_*.c` fixtures -- never the twin or a harness
    driver built from the runtime's own sources, never a build with no C source (an `.ll` alone)."""
    repo = str(_REPO_ROOT)
    tmp = os.path.join(tempfile.gettempdir(), "w")
    fixture = os.path.join(repo, "runtime", "c", "cfront_signed.c")
    twin = os.path.join(repo, "runtime", "c", "bcir_cfront.c")
    assert uw.is_witness_link(["-std=c23", "-O2", tmp + ".c", "-o", tmp, "-lm"], repo)
    assert uw.is_witness_link(["-O2", fixture, tmp + "_emit.c", "-o", tmp], repo)
    assert uw.is_witness_link([tmp + ".c", tmp + "_emit.ll", "-o", tmp], repo)
    for flag in ("-c", "-S", "-E", "-shared", "-fsyntax-only", "--analyze", "-M", "-MM"):
        assert not uw.is_witness_link([flag, tmp + ".c", "-o", tmp], repo), flag
    # no output named: no witness the gate can place
    assert not uw.is_witness_link([tmp + ".c"], repo)
    assert not uw.is_witness_link([tmp + ".ll", "-o", tmp], repo)
    assert not uw.is_witness_link(["-O2", twin, "-o", tmp], repo)
    assert not uw.is_witness_link(["-O2", tmp + ".c", twin, "-o", tmp], repo)
    # an include directory is no source
    assert not uw.is_witness_link(["-I", tmp + ".c", "-o", tmp], repo)
    assert not uw.is_witness_link(["-o", tmp + ".c", tmp + ".ll"], repo)  # nor is the output


def test_the_sanitized_build_keeps_the_program_and_drops_only_optimisation_and_warnings():
    """The sanitized twin builds the same program: the same standard, sources, includes, defines and libraries,
    its own output, `-O*`/`-W*`/`-pedantic`/`-w` dropped (the `-O2` miscompile witness keeps them) and the engine's
    level and the UBSan flags last -- Clang's optimized, GCC's unoptimized, for GCC's optimizer deletes a check
    (`_FOLDED`), and so any engine the gate does not know."""
    args = ["-std=c23", "-O2", "-Werror=incompatible-pointer-types", "-pedantic", "-w", "-I", "inc", "-DX=1",
            "-fsigned-char", "a.c", "-o", "a", "-lm"]  # fmt: skip
    program = ["-std=c23", "-I", "inc", "-DX=1", "-fsigned-char", "a.c", "-o", "a.ubsan", "-lm"]
    ubsan = ["-g", "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all"]
    assert uw.SANITIZE == ubsan
    assert uw.sanitized_args(args, "a.ubsan", "/usr/bin/clang") == [*program, "-O2", *ubsan]
    assert uw.sanitized_args(args, "a.ubsan", "/usr/bin/gcc") == [*program, "-O0", *ubsan]
    assert uw.sanitized_args(args, "a.ubsan", "/opt/cc-other") == [*program, "-O0", *ubsan]
    assert uw.ENGINES == ("clang", "gcc")  # Clang's UBSan first: the instrument no optimizer folds
    assert args[1] == "-O2"  # the witness's own arguments are not touched


def _shim_env(log: str, engine: str) -> dict:
    env = dict(
        os.environ, BCIR_UBSAN_LOG=log, BCIR_UBSAN_ENGINES=engine, BCIR_CHECK_TEST="the-test"
    )
    env[f"BCIR_UBSAN_REAL_{os.path.basename(engine).upper()}"] = engine
    return env


def test_a_shimmed_witness_fails_on_undefined_behaviour_and_runs_a_defined_one_unchanged():
    """Under each UBSan engine at hand: a witness whose original overflows `int` is built twice (its own `-O2`
    build kept beside the sanitized one), and running it fails with the report on stderr and in the log, exit 125;
    a defined witness runs, prints what its unsanitized build prints, and its sanitized run is counted."""
    engines = uw._engines()
    if not engines:
        return
    for engine in engines:
        name = os.path.basename(engine)
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "log")
            os.makedirs(log)
            env = _shim_env(log, engine)
            for label, text in (
                ("undefined", _UNDEFINED),
                ("defined", _DEFINED),
                ("cast", _FLOAT_CAST),
            ):
                src, exe = os.path.join(d, f"{label}.c"), os.path.join(d, label)
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write(text)
                b = subprocess.run([sys.executable, _SCRIPT, "--shim", name, "-O2", src, "-o", exe], env=env,
                                   capture_output=True, text=True)  # fmt: skip
                assert b.returncode == 0, (name, b.stderr)
                assert os.path.exists(exe + ".real") and os.path.exists(exe + ".ubsan"), name
            bad = subprocess.run(
                [os.path.join(d, "undefined")], env=env, capture_output=True, text=True
            )
            assert bad.returncode == 125 and "runtime error:" in bad.stderr, (
                name,
                bad.returncode,
                bad.stderr,
            )
            assert "signed integer overflow" in bad.stderr, (name, bad.stderr)
            good = subprocess.run(
                [os.path.join(d, "defined")], env=env, capture_output=True, text=True
            )
            assert (good.returncode, good.stdout, good.stderr) == (0, "0\n", ""), name
            cast = subprocess.run(
                [os.path.join(d, "cast")], env=env, capture_output=True, text=True
            )
            assert cast.returncode == 125, (name, cast.returncode, cast.stderr)
            assert "outside the range of representable values" in cast.stderr, (name, cast.stderr)
            reports = sorted(
                f for f in os.listdir(log) if f.startswith("report.") and ".src" not in f
            )
            assert len(reports) == 2, (name, reports)
            for r in reports:  # each names its test and witness, and keeps the witness's source
                with open(os.path.join(log, r), encoding="utf-8") as fh:
                    head = fh.read().splitlines()[:2]
                assert head[0] == "test: the-test" and head[1] in (
                    "witness: undefined.c",
                    "witness: cast.c",
                ), head
                with open(os.path.join(log, r + ".src0.c"), encoding="utf-8") as fh:
                    assert fh.read() in (_UNDEFINED, _FLOAT_CAST), r
            with open(os.path.join(log, "runs.txt"), encoding="utf-8") as fh:
                assert fh.read().split() == ["report", "clean", "report"], name
            with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh]
            assert [(e["event"], e["engine"]) for e in events] == [("sanitized", name)] * 3, events
            assert [e["sources"] for e in events] == [["undefined.c"], ["defined.c"], ["cast.c"]], (
                events
            )


def test_a_witness_whose_original_and_emit_overflow_alike_is_reported_by_every_engine():
    """A witness compares an original with an emit that computes the same overflow. GCC's checks of signed arithmetic
    are pure functions of their operands, so above -O0 GCC folds the comparison and deletes both: GCC's sanitized
    program is built unoptimized, Clang's -- whose checks no optimizer deletes -- at -O2, and under every engine at
    hand this witness fails with the report -- not runs on as `MATCH`."""
    for engine in uw._engines():
        name = os.path.basename(engine)
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "log")
            os.makedirs(log)
            env = _shim_env(log, engine)
            src, exe = os.path.join(d, "folded.c"), os.path.join(d, "folded")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(_FOLDED)
            b = subprocess.run([sys.executable, _SCRIPT, "--shim", name, "-O2", src, "-o", exe], env=env,
                               capture_output=True, text=True)  # fmt: skip
            assert b.returncode == 0, (name, b.stderr)
            run = subprocess.run([exe], env=env, capture_output=True, text=True)
            assert run.returncode == 125 and "signed integer overflow" in run.stderr, (
                name,
                run.returncode,
                run.stdout,
                run.stderr,
            )


def test_a_witness_whose_original_discards_an_overflow_is_reported_by_the_gates_engines():
    """An original that discards an overflowing value (`x * x;`) runs undefined behaviour, and GCC's front end drops
    the statement unchecked at every level: under the gate's engines in the gate's order -- Clang's first -- the
    witness, built by GCC as the cfront tests build theirs, fails with Clang's report."""
    engines, gcc = uw._engines(), shutil.which("gcc")
    if not gcc or not any(os.path.basename(e) == "clang" for e in engines):
        return  # Clang's UBSan, the one engine that reports this, is not here
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "log")
        os.makedirs(log)
        env = dict(_shim_env(log, gcc), BCIR_UBSAN_ENGINES=os.pathsep.join(engines))
        src, exe = os.path.join(d, "discarded.c"), os.path.join(d, "discarded")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_DISCARDED)
        b = subprocess.run([sys.executable, _SCRIPT, "--shim", "gcc", "-O2", src, "-o", exe], env=env,
                           capture_output=True, text=True)  # fmt: skip
        assert b.returncode == 0, b.stderr
        run = subprocess.run([exe], env=env, capture_output=True, text=True)
        assert run.returncode == 125 and "signed integer overflow" in run.stderr, (
            run.returncode,
            run.stdout,
            run.stderr,
        )
        with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh]
        assert [(e["event"], e["compiler"], e["engine"]) for e in events] == [
            ("sanitized", "gcc", "clang")
        ], events


def test_a_c23_witness_is_sanitized_by_an_engine_that_spells_c23_by_its_draft_name():
    """A witness the real compiler built with `-std=c23` (Clang 18) is sanitized by an engine that knows C23 only as
    `c2x` (GCC 13) -- the same standard under its draft name -- not left unsanitized; the arguments keep
    `-std=c23` unless asked for the draft spelling."""
    assert uw.sanitized_args(["-std=c23", "a.c", "-o", "a"], "a.u", "gcc") == [
        "-std=c23",
        "a.c",
        "-o",
        "a.u",
        *uw.engine_flags("gcc"),
    ]
    assert (
        uw.sanitized_args(["-std=gnu23", "a.c", "-o", "a"], "a.u", "gcc", True)[0] == "-std=gnu2x"
    )
    engines, clang = uw._engines(), shutil.which("clang")
    if not engines or not clang:
        return
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "w.c")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_DEFINED)
        if subprocess.run(
            [clang, "-std=c23", "-fsyntax-only", src], capture_output=True
        ).returncode:
            return  # a Clang older than C23's name: no witness is built that way here
        for engine in engines:
            log = os.path.join(d, "log." + os.path.basename(engine))
            os.makedirs(log)
            env = dict(_shim_env(log, engine), BCIR_UBSAN_REAL_CLANG=clang)
            exe = os.path.join(d, "w." + os.path.basename(engine))
            b = subprocess.run([sys.executable, _SCRIPT, "--shim", "clang", "-std=c23", "-O2", src, "-o", exe],
                               env=env, capture_output=True, text=True)  # fmt: skip
            assert b.returncode == 0, (engine, b.stderr)
            with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh]
            assert [e["event"] for e in events] == ["sanitized"], (engine, events)
            run = subprocess.run([exe], capture_output=True, text=True)
            assert (run.returncode, run.stdout) == (0, "0\n"), (engine, run.stderr)


def test_a_build_that_is_no_witness_passes_through_the_shim_untouched():
    """An object build (`-c`) is the real compiler's alone: no sanitized twin, no launcher, nothing logged."""
    engines = uw._engines()
    if not engines:
        return
    engine = engines[0]
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "log")
        os.makedirs(log)
        src, obj = os.path.join(d, "u.c"), os.path.join(d, "u.o")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_UNDEFINED)
        b = subprocess.run([sys.executable, _SCRIPT, "--shim", os.path.basename(engine), "-c", src, "-o", obj],
                           env=_shim_env(log, engine), capture_output=True, text=True)  # fmt: skip
        assert b.returncode == 0, b.stderr
        assert sorted(os.listdir(d)) == ["log", "u.c", "u.o"] and os.listdir(log) == []


def test_an_engine_that_builds_but_never_reports_is_no_engine():
    """The probe runs signed overflow and must see the report: a compiler that takes the UBSan flags and builds a
    program that never fires (here a real one told `-fno-sanitize=all` last) is no engine, and neither is one that
    cannot be started (L2, L1)."""
    engines = uw._engines()
    if not engines:
        return
    with tempfile.TemporaryDirectory() as d:
        mute = os.path.join(d, "gcc")
        with open(mute, "w", encoding="utf-8") as fh:
            fh.write(f'#!/bin/sh\nexec {engines[0]} "$@" -fno-sanitize=all\n')
        os.chmod(mute, 0o755)
        assert subprocess.run([mute, "--version"], capture_output=True).returncode == 0
        assert uw._engines(("gcc",), which=lambda name: mute) == []
        assert uw._engines(("gcc",), which=lambda name: os.path.join(d, "absent")) == []
        assert uw._engines(("gcc",), which=lambda name: engines[0]) == [engines[0]]


def test_the_ci_job_that_installs_the_runtime_requires_it():
    """The CI job that runs the gate installs Clang's compiler-rt beside the runner's GCC, so there the gate runs
    with `--require-ubsan`: a runtime gone missing fails the job instead of skipping it (L2)."""
    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    start = re.search(r"^  cfront-ubsan:\s*$", ci, re.M)
    assert start, "ci.yml has no cfront-ubsan job"
    end = re.compile(r"^  [A-Za-z0-9_-]+:\s*$", re.M).search(ci, start.end())
    job = ci[start.start() : end.start() if end else len(ci)]
    assert "libclang-rt-18-dev" in job, "the job no longer installs compiler-rt"
    runs = [line.strip() for line in job.splitlines() if "tools/testing/ubsan_witnesses.py" in line]
    assert runs, "no step of the job runs the UBSan witness gate"
    assert all("--require-ubsan" in line for line in runs), runs


def test_a_witness_no_engine_can_build_is_recorded_unsanitized():
    """A witness the engines cannot build sanitized (here an engine that always fails) keeps its own build, runs as
    it always ran, and is recorded with the engine's reason -- for the driver to fail under `--require-ubsan`. The
    reason is the engine's `error:` line, not the caret its diagnostic ends with."""
    engines = uw._engines()
    if not engines:
        return
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "log")
        os.makedirs(log)
        src, exe = os.path.join(d, "w.c"), os.path.join(d, "w")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_DEFINED)
        env = _shim_env(log, engines[0])
        env["BCIR_UBSAN_ENGINES"] = shutil.which("false") or "/bin/false"
        b = subprocess.run([sys.executable, _SCRIPT, "--shim", os.path.basename(engines[0]), src, "-o", exe],
                           env=env, capture_output=True, text=True)  # fmt: skip
        assert b.returncode == 0, b.stderr
        assert sorted(os.listdir(d)) == ["log", "w", "w.c"]
        run = subprocess.run([exe], capture_output=True, text=True)
        assert (run.returncode, run.stdout) == (0, "0\n")
        with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh]
        assert [(e["event"], e["sources"]) for e in events] == [("unsanitized", ["w.c"])], events
        assert events[0]["why"].startswith("false: "), events
        noisy = os.path.join(d, "cc-fails")
        with open(noisy, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\nprintf 'w.c: In function f:\\nw.c:1:10: error: unknown type name nope\\n"
                     "    1 | nope x;\\n      | ^~~~\\n' >&2\nexit 1\n")  # fmt: skip
        os.chmod(noisy, 0o755)
        os.remove(os.path.join(log, "events.jsonl"))
        env["BCIR_UBSAN_ENGINES"] = noisy
        b = subprocess.run([sys.executable, _SCRIPT, "--shim", os.path.basename(engines[0]), src, "-o", exe],
                           env=env, capture_output=True, text=True)  # fmt: skip
        assert b.returncode == 0, b.stderr
        with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh]
        assert [e["why"] for e in events] == [
            "cc-fails: w.c:1:10: error: unknown type name nope"
        ], events


@contextlib.contextmanager
def _driver(engines):
    """`ubsan_witnesses.main` in this process with its engine probe answering `engines` and the capability the
    tests need read from this module; its stdout captured."""
    probe, capability = uw._engines, uw.CAPABILITY
    uw._engines = lambda names=uw.ENGINES: [e for e in engines if os.path.basename(e) in names]
    uw.CAPABILITY = "bcir.tests.test_ubsan_witnesses:_ALWAYS"
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            yield out
    finally:
        uw._engines, uw.CAPABILITY = probe, capability


def _verdict(argv, engines):
    with tempfile.TemporaryDirectory() as d, _driver(engines) as out:
        path = os.path.join(d, "report.json")
        code = uw.main([*argv, "--json-out", path])
        with open(path, encoding="utf-8") as fh:
            return code, json.load(fh), out.getvalue()


def test_no_ubsan_runtime_is_a_skip_unless_one_was_required():
    """Where no engine reports, the gate says so and passes as a skip; under `--require-ubsan` (the CI job that
    installs the runtime) the same absence is no pass (L2) -- and so is an `--engine` the host lacks, and, with no
    engine named, GCC's alone: Clang's is the gate's instrument, and the job installs it. Each run names one helper,
    so that a gate which no longer refuses here runs it -- not the whole cfront suite under GCC's unoptimized UBSan."""
    one = "bcir.tests.test_ubsan_witnesses:_witness_none"
    code, report, out = _verdict([one], [])
    assert (code, report["verdict"]) == (uw.EXIT_OK, "UNAVAILABLE") and "no UBSan runtime" in out
    code, report, _ = _verdict(["--require-ubsan", one], [])
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "UNAVAILABLE")
    code, report, out = _verdict(["--require-ubsan", "--engine", "clang", one], ["/usr/bin/gcc"])
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "UNAVAILABLE") and "clang's" in out
    code, report, out = _verdict(["--require-ubsan", one], ["/usr/bin/gcc"])
    assert (code, report["verdict"], report["engines"]) == (
        uw.EXIT_UNAVAILABLE,
        "UNAVAILABLE",
        ["gcc"],
    )
    assert "Clang's UBSan runtime" in out, out


def test_a_run_that_sanitizes_nothing_or_fails_a_test_is_no_pass():
    """The driver's own verdicts, with no compiler run: tests that build no witness examined nothing (INVALID, L2),
    a failing test is a finding, and a name that resolves to no test cannot be run (INVALID)."""
    here = "bcir.tests.test_ubsan_witnesses"
    fake = ["/nonexistent/gcc"]
    code, report, _ = _verdict([f"{here}:_witness_none"], fake)
    assert (code, report["verdict"], report["sanitized"]) == (uw.EXIT_UNAVAILABLE, "INVALID", 0)
    code, report, out = _verdict([f"{here}:_witness_none", f"{here}:_witness_fails"], fake)
    assert (code, report["verdict"]) == (uw.EXIT_FINDINGS, "FAIL")
    assert report["test_findings"] == [
        "- _witness_fails: AssertionError: the emit is not the original"
    ], report
    code, report, _ = _verdict([f"{here}:_no_such_test"], fake)
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "INVALID")
    code, report, _ = _verdict(["-j", "0", f"{here}:_witness_none"], fake)
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "INVALID")


def test_shards_partition_the_witnesses_and_an_empty_or_malformed_shard_is_refused():
    """`--shard I/N` runs the Ith stride of the witness list (CI cells): the slices are disjoint and together they
    are the whole list, so the failing test sits in exactly one of them; a slice that holds no test, an index past
    its count and a spelling that is no I/N are each refused as INVALID, never run as an empty pass (L2)."""
    here = "bcir.tests.test_ubsan_witnesses"
    fake = ["/nonexistent/gcc"]
    three = [f"{here}:_witness_none", f"{here}:_witness_fails", f"{here}:_witness_none"]
    assert uw.parse_shard(None) == (1, 1) and uw.parse_shard("2/3") == (2, 3)
    for bad in ("0/2", "3/2", "1/0", "x", "1", "1/x"):
        try:
            uw.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_shard accepted {bad!r}")
    # 1/2 is [none, none]: nothing sanitized
    code, report, _ = _verdict(["--shard", "1/2", *three], fake)
    assert (code, report["verdict"], report["shard"]) == (uw.EXIT_UNAVAILABLE, "INVALID", "1/2")
    assert report["test_findings"] == [], report
    # 2/2 is [fails]: the finding, and only it
    code, report, out = _verdict(["--shard", "2/2", *three], fake)
    assert (code, report["verdict"], report["shard"]) == (uw.EXIT_FINDINGS, "FAIL", "2/2"), report
    assert report["test_findings"] == [
        "- _witness_fails: AssertionError: the emit is not the original"
    ], report
    assert "shard=2/2" in out, out
    # three tests, a fourth slice: empty
    code, report, out = _verdict(["--shard", "4/4", *three], fake)
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "INVALID") and "holds no test" in out
    code, report, out = _verdict(["--shard", "3/2", *three], fake)
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "INVALID") and "out of range" in out
    code, report, out = _verdict(["--shard", "two", *three], fake)
    assert (code, report["verdict"]) == (uw.EXIT_UNAVAILABLE, "INVALID") and "needs I/N" in out
    code, report, _ = _verdict(three, fake)  # no shard: the whole list, and the report says so
    assert (code, report["verdict"], report["shard"]) == (uw.EXIT_FINDINGS, "FAIL", "1/1"), report


def test_the_gate_fires_on_a_witness_whose_original_runs_undefined_behaviour():
    """End to end, under each engine at hand: the driver runs a witness whose original overflows `int` through its
    shims and fails it with the sanitizer's report (the RED half); a defined witness passes with its sanitized run
    counted (the GREEN half)."""
    engines = uw._engines()
    if not engines:
        return
    here = "bcir.tests.test_ubsan_witnesses"
    for engine in engines:
        name = os.path.basename(engine)
        code, report, _ = _verdict(["--engine", name, f"{here}:_witness_undefined"], [engine])
        assert (code, report["verdict"]) == (uw.EXIT_FINDINGS, "FAIL"), (name, report)
        assert len(report["reports"]) == 1 and "signed integer overflow" in report["reports"][0], (
            report
        )
        # each report names the test it ran under
        assert report["reports"][0].startswith(f"{here}:_witness_undefined: "), report
        assert report["test_findings"] and report["sanitized"] == 1 and report["runs"] == 1, report
        code, report, _ = _verdict(
            ["--require-ubsan", "--engine", name, f"{here}:_witness_defined"], [engine]
        )
        assert (code, report["verdict"], report["engines"]) == (uw.EXIT_OK, "PASS", [name]), report
        assert (report["sanitized"], report["runs"], report["reports"]) == (1, 1, []), report
        assert report["sanitized_by"] == {name: 1}, report  # the engine that built it, counted
