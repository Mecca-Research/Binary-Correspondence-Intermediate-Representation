#!/usr/bin/env python3
"""Every cfront run-as-the-original witness, built once more under UBSan: the gate that a witness's verdict is the
emit's and not the platform's (CF-UBGATE; docs/security/laws.md L12).

A witness compiles the original C beside an emit into one program and compares them. If the original runs undefined
behaviour, the two compute one undefined value in ways of their own and the platform decides the verdict: on PR #797
`cfront_unaryops.c` converted a negative `double` to `uint32_t`, which x86-64 wraps and AArch64 saturates, and only
the AArch64 job failed. This runs the cfront test modules with compiler shims first on `PATH` (`clang`, `gcc`,
`cc`). A shim runs the real compiler with the arguments it was given -- the witness's own `-O2` build, unchanged --
and, when the build links a witness, builds the same sources once more with `-fsanitize=undefined,float-cast-overflow
-fno-sanitize-recover=all` (Clang's UBSan at `-O2`, GCC's unoptimized where Clang cannot build the program: `LEVEL`)
and puts a launcher at the output path: the sanitized program runs first, and a report fails the run (exit 125, the
report on stderr and in the log); else the launcher runs the unsanitized program, so the test sees what it always
saw.

Scope, declared: a build is a witness when it links an executable (`-o`, and none of `-c`, `-S`, `-E`, `-shared`,
`-fsyntax-only`, `--analyze`, `-M`) from C sources none of which is a repository file but a `runtime/c/cfront_*.c`
fixture -- so the twin compiler and the harness drivers built from the runtime's own sources are not witnesses (the
cfront sanitizer sweep, `tools/c/sanitize_cfront.sh`, owns them). POSIX only (the launcher is a shell script).

Every exit is a verdict (L1):

    0  every witness ran clean under UBSan and every test passed (or, without --require-ubsan, no UBSan runtime
       here: an honest skip, printed)
    1  a finding: a UBSan report, a failing test, or (with --require-ubsan) a witness no engine could sanitize
    2  the gate examined nothing (no sanitized witness ran), could not run its tests, or found under
       --require-ubsan no UBSan runtime -- or, with no --engine named, not Clang's, the gate's instrument (L2)

    python3 tools/testing/ubsan_witnesses.py [--require-ubsan] [--engine gcc|clang] [-j N] [--json-out FILE]
        [MODULE:FUNCTION ...]

`--engine` names the one UBSan runtime to build every witness with; by default Clang's, and GCC's where Clang cannot
build the program. GCC's builds run unoptimized (`LEVEL`), so `--engine gcc` is for named tests: over the whole suite
it runs every witness's long loops -- 2**31 trips, folded away at `-O2` -- unoptimized, for hours.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXIT_OK, EXIT_FINDINGS, EXIT_UNAVAILABLE = 0, 1, 2
SANITIZE = ["-g", "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all"]
#: each engine's level, before `SANITIZE`. Clang's check is a branch to a call of its handler, which no optimizer
#: deletes while the branch can be taken, so Clang's sanitized build is optimized as the witness's own is -- at `-O0` a
#: witness looping 2**31 times, which `-O2` folds to its closed form, ran for hours. GCC's check of signed arithmetic
#: is a pure function of its operands, deleted with any value nothing reads at every level above `-O0`: a witness
#: comparing an original with an emit that overflows alike folds the comparison and both checks with it
#: (`cfront_signed.c` with its overflow restored ran clean under GCC at -O1 and -O2, and reported under Clang at every
#: level). So Clang's UBSan is the gate's instrument, and GCC's, the engine of a program Clang cannot build, runs
#: unoptimized -- where it still drops an expression statement's overflow, which its front end discards unchecked.
#: An engine not named here runs unoptimized.
LEVEL = {"clang": "-O2", "gcc": "-O0"}
COMPILERS = ("gcc", "clang", "cc")
ENGINES = ("clang", "gcc")
#: the capability the cfront tests return early without (`check_tests.py --require`): no C compiler, nothing examined
CAPABILITY = "bcir.tests.test_c_cfront:_CC"
# the cfront test modules: every one that builds a witness, and the rest of the cfront suite with them
MODULES = (
    "test_c_cfront", "test_cfront_roundtrip", "test_cfront_constexpr", "test_cfront_effects", "test_cfront_volatile",
    "test_cfront_link", "test_cfront_cli", "test_cfront_diagnostics", "test_cfront_fuzz", "test_cfront_abi",
    "test_cfront", "test_c_cfront_asm", "test_c_cfront_barrier", "test_c_cfront_portio", "test_cfront_fallback",
    "test_cfront_ipo", "test_cfront_asm_portio_redteam",
)  # fmt: skip
_NOT_A_LINK = {"-c", "-S", "-E", "-shared", "-fsyntax-only", "--analyze", "-M", "-MM"}
_REPORT = ("runtime error:", "UndefinedBehaviorSanitizer")


def is_witness_link(args: list[str], repo: str = REPO) -> bool:
    """Whether a compiler invocation links a witness (the declared scope above)."""
    if "-o" not in args or any(a in _NOT_A_LINK for a in args):
        return False
    sources = [
        a
        for i, a in enumerate(args)
        if a.endswith(".c") and (i == 0 or args[i - 1] not in ("-o", "-I"))
    ]
    if not sources:
        return False
    for s in sources:
        path = os.path.realpath(s)
        if path.startswith(repo + os.sep) and not os.path.basename(path).startswith("cfront_"):
            return False
    return True


#: C23's spellings an engine older than C23 (GCC 13) takes by their draft names, for a witness a newer real compiler
#: (Clang 18) built with `-std=c23` -- the standard is the same; only the flag's name differs
_STD_DRAFT = {"-std=c23": "-std=c2x", "-std=gnu23": "-std=gnu2x"}


def engine_flags(engine: str) -> list[str]:
    """The UBSan flags a build under `engine` (a path or a name) takes: its level (`LEVEL`), then `SANITIZE`."""
    return [LEVEL.get(os.path.basename(engine), "-O0"), *SANITIZE]


def sanitized_args(args: list[str], out: str, engine: str, draft_std: bool = False) -> list[str]:
    """The witness build's arguments for its sanitized twin under `engine`: the same sources, standard, includes,
    defines and libraries, its optimization and warning flags dropped (the `-O2` build keeps them) for the engine's
    own (`engine_flags`), the output beside it -- with `draft_std`, C23 spelled by its draft name (`_STD_DRAFT`)."""
    kept, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == "-o":
            kept += ["-o", out]
            skip = True
        elif (
            a.startswith("-O") or a.startswith("-W") or a in ("-pedantic", "-pedantic-errors", "-w")
        ):
            continue
        else:
            kept.append(_STD_DRAFT.get(a, a) if draft_std else a)
    return kept + engine_flags(engine)


def _diagnostic(stderr: str) -> str:
    """A failed build's reason: its first `error:` line -- not its last line, which is a caret under the source --
    else its last line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if "error:" in line]
    return (errors[0] if errors else lines[-1] if lines else "?")[:200]


def _log(event: dict) -> None:
    path = os.path.join(os.environ["BCIR_UBSAN_LOG"], "events.jsonl")
    # one short line per event: an atomic append
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def shim(name: str, args: list[str]) -> int:
    """A compiler shim: the real compiler as asked, then, for a witness link, its sanitized twin and launcher."""
    real = os.environ[f"BCIR_UBSAN_REAL_{name.upper()}"]
    try:
        rc = subprocess.call([real, *args])
    except OSError as exc:
        print(f"ubsan_witnesses shim: cannot run {real}: {exc}", file=sys.stderr)
        return 127
    if rc != 0 or not is_witness_link(args):
        return rc
    out = os.path.abspath(args[args.index("-o") + 1])
    paths = [
        os.path.abspath(a) for a in args if a.endswith(".c") and a != args[args.index("-o") + 1]
    ]
    sources = [os.path.basename(a) for a in paths]
    engines = [e for e in os.environ.get("BCIR_UBSAN_ENGINES", "").split(os.pathsep) if e]
    why, built = "no engine", None
    for engine in engines:
        for draft in (False, True) if any(a in _STD_DRAFT for a in args) else (False,):
            try:
                b = subprocess.run([engine, *sanitized_args(args, out + ".ubsan", engine, draft)],
                                   capture_output=True, text=True, errors="replace")  # fmt: skip
            except OSError as exc:
                why = f"{engine}: {exc}"
                break
            if b.returncode == 0:
                built = engine
                break
            why = f"{os.path.basename(engine)}: {_diagnostic(b.stderr)}"
        if built:
            break
    if not built:
        _log({"event": "unsanitized", "out": out, "compiler": name, "sources": sources, "why": why})
        return 0
    engine = built
    os.replace(out, out + ".real")
    log = os.environ["BCIR_UBSAN_LOG"]
    q = shlex.quote
    copies = "".join(f'cp {q(p)} "$r.src{k}.c" 2>/dev/null\n' for k, p in enumerate(paths))
    launcher = f"""#!/bin/sh
{q(out + ".ubsan")} "$@" </dev/null >/dev/null 2>{q(out + ".ubsan.err")}
if grep -q -e 'runtime error:' -e 'UndefinedBehaviorSanitizer' {q(out + ".ubsan.err")}; then
  cat {q(out + ".ubsan.err")} >&2
  r={q(log)}/report.$$.$(date +%s%N)
  {{ echo "test: ${{BCIR_CHECK_TEST:-?}}"; echo "witness:" {q(" ".join(sources))}; cat {q(out + ".ubsan.err")}; }} > "$r"
{copies}  echo report >> {q(log)}/runs.txt
  exit 125
fi
echo clean >> {q(log)}/runs.txt
exec {q(out + ".real")} "$@"
"""
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(launcher)
    os.chmod(out, 0o755)
    _log(
        {
            "event": "sanitized",
            "out": out,
            "compiler": name,
            "engine": os.path.basename(engine),
            "sources": sources,
        }
    )
    return 0


def _engines(names=ENGINES, which=shutil.which) -> list[str]:
    """The real compilers among `names` whose UBSan runtime links AND reports here -- a probe that runs signed
    overflow and must see the report, so an engine that builds but never fires is no engine (L2). In `names`' order:
    Clang first. A compiler that cannot be started, or hangs, is no engine either (L1)."""
    found = []
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.c")
        with open(src, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("int main(void) { volatile int x = 2147483647; x = x + 1; return 0; }\n")
        for name in names:
            cc = which(name)
            if not cc:
                continue
            exe = os.path.join(d, name)
            try:
                if subprocess.run(
                    [cc, src, "-o", exe, *engine_flags(name)], capture_output=True, timeout=120
                ).returncode:
                    continue
                run = subprocess.run(
                    [exe], capture_output=True, text=True, errors="replace", timeout=60
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if run.returncode != 0 and "runtime error:" in run.stderr:
                found.append(cc)
    return found


def parse_shard(raw: str | None) -> tuple[int, int]:
    """``I/N`` as (I, N), 1 <= I <= N; ``None`` is (1, 1). Anything else is a ValueError naming the spelling."""
    if raw is None:
        return (1, 1)
    try:
        index, count = (int(part) for part in raw.split("/", 1))
    except ValueError:
        raise ValueError(f"--shard needs I/N, got {raw!r}") from None
    if not (count >= 1 and 1 <= index <= count):
        raise ValueError(f"--shard {raw} is out of range")
    return (index, count)


def _tests(selected: list[str]) -> list[str]:
    if selected:
        return selected
    sys.path.insert(0, REPO)
    names = []
    for m in MODULES:
        mod = importlib.import_module("bcir.tests." + m)
        names += [
            f"bcir.tests.{m}:{n}"
            for n in sorted(dir(mod))
            if n.startswith("test_") and callable(getattr(mod, n))
        ]
    return names


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--shim"]:
        return shim(argv[1], argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--require-ubsan",
        action="store_true",
        help="no UBSan runtime, or an unsanitized witness, fails",
    )
    parser.add_argument(
        "--engine", choices=ENGINES, help="build every witness with this UBSan runtime only"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=2, help="test processes at once (default 2)"
    )
    parser.add_argument("--json-out")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds for each test process")
    parser.add_argument(
        "--shard",
        metavar="I/N",
        help="run the Ith of N disjoint stride slices of the witnesses (CI cells); default all",
    )
    parser.add_argument(
        "tests", nargs="*", metavar="MODULE:FUNCTION", help="default: every cfront test"
    )
    args = parser.parse_args(argv)
    report = {"verdict": None, "witnesses": 0, "sanitized": 0, "sanitized_by": {}, "unsanitized": [], "runs": 0,
              "reports": [], "test_findings": [], "engines": [], "shard": "1/1"}  # fmt: skip

    def finish(verdict: str, code: int) -> int:
        report["verdict"] = verdict
        print(
            f"ubsan_witnesses: {verdict} sanitized={report['sanitized']} runs={report['runs']} "
            f"reports={len(report['reports'])} unsanitized={len(report['unsanitized'])} "
            f"test_findings={len(report['test_findings'])} engines={','.join(report['engines']) or '-'} "
            f"by={','.join(f'{k}:{v}' for k, v in sorted(report['sanitized_by'].items())) or '-'}"
            + (f" shard={report['shard']}" if args.shard else "")
        )
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
        return code

    engines = _engines((args.engine,) if args.engine else ENGINES)
    report["engines"] = [os.path.basename(e) for e in engines]
    if not engines:
        wanted = f"{args.engine}'s" if args.engine else "Clang's compiler-rt or GCC's libubsan"
        print(f"ubsan_witnesses: no UBSan runtime that reports here ({wanted})")
        return finish("UNAVAILABLE", EXIT_UNAVAILABLE if args.require_ubsan else EXIT_OK)
    if args.require_ubsan and not args.engine and "clang" not in report["engines"]:
        # GCC's alone deletes the checks of values nothing reads (`LEVEL`): the run asked for is not this one (L2)
        print(
            "ubsan_witnesses: Clang's UBSan runtime (compiler-rt), the gate's instrument, does not report here"
        )
        return finish("UNAVAILABLE", EXIT_UNAVAILABLE)
    if args.jobs < 1:
        return finish("INVALID", EXIT_UNAVAILABLE)
    try:
        shard = parse_shard(args.shard)
    except ValueError as exc:
        print(f"  - unresolved: {exc}")
        return finish("INVALID", EXIT_UNAVAILABLE)
    try:
        tests = _tests(args.tests)
    except Exception as exc:  # noqa: BLE001 -- an unimportable module: the gate cannot run what it was asked (L1)
        print(f"  - unresolved: {type(exc).__name__}: {exc}")
        return finish("INVALID", EXIT_UNAVAILABLE)
    if args.shard:
        # The Ith stride of N (run_all's `--shard`): the N slices are disjoint and their union is the whole
        # list, so a CI matrix of N cells runs every witness exactly once; a slice holding no test would
        # examine nothing and is refused (L2).
        index, count = shard
        report["shard"] = f"{index}/{count}"
        tests = tests[index - 1 :: count]
        if not tests:
            print(f"  - unresolved: shard {index}/{count} holds no test")
            return finish("INVALID", EXIT_UNAVAILABLE)
    with tempfile.TemporaryDirectory() as work:
        shims, log = os.path.join(work, "bin"), os.path.join(work, "log")
        os.makedirs(shims)
        os.makedirs(log)
        env = dict(os.environ, BCIR_UBSAN_LOG=log, BCIR_UBSAN_ENGINES=os.pathsep.join(engines))
        for name in COMPILERS:
            real = shutil.which(name)
            if not real:
                continue
            env[f"BCIR_UBSAN_REAL_{name.upper()}"] = real
            path = os.path.join(shims, name)
            me = f"{shlex.quote(sys.executable)} {shlex.quote(os.path.abspath(__file__))}"
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(f'#!/bin/sh\nexec {me} --shim {name} "$@"\n')
            os.chmod(path, 0o755)
        env["PATH"] = shims + os.pathsep + env.get("PATH", "")
        shards = [tests[i :: args.jobs] for i in range(args.jobs) if tests[i :: args.jobs]]
        procs = []
        for shard in shards:
            cmd = [sys.executable, os.path.join(REPO, "tools", "testing", "check_tests.py"),
                   "--require", CAPABILITY, *shard]  # fmt: skip
            try:
                procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True, errors="replace"))  # fmt: skip
            except OSError as exc:
                print(f"  - unresolved: cannot start the tests: {exc}")
                for p in procs:
                    p.kill()
                    p.communicate()
                return finish("INVALID", EXIT_UNAVAILABLE)
        unavailable = False
        for p in procs:
            try:
                out, _ = p.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                report["test_findings"].append("timeout")
                continue
            report["test_findings"] += [
                line.strip() for line in out.splitlines() if line.startswith("  - ")
            ]
            unavailable |= p.returncode == EXIT_UNAVAILABLE
        events = []
        if os.path.exists(os.path.join(log, "events.jsonl")):
            with open(os.path.join(log, "events.jsonl"), encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh if line.strip()]
        report["sanitized"] = sum(1 for e in events if e["event"] == "sanitized")
        # which engine built each: GCC's is the fallback, and a run says how often it fell back
        for e in events:
            if e["event"] == "sanitized":
                report["sanitized_by"][e["engine"]] = report["sanitized_by"].get(e["engine"], 0) + 1
        report["unsanitized"] = [
            f"{','.join(e['sources'])} ({e['why']})" for e in events if e["event"] == "unsanitized"
        ]
        if os.path.exists(os.path.join(log, "runs.txt")):
            with open(os.path.join(log, "runs.txt"), encoding="utf-8") as fh:
                report["runs"] = sum(1 for _ in fh)
        for name in sorted(os.listdir(log)):
            if name.startswith("report.") and ".src" not in name:
                with open(os.path.join(log, name), encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
                test = lines[0][len("test: ") :] if lines and lines[0].startswith("test: ") else "?"
                first = [line for line in lines if any(k in line for k in _REPORT)]
                report["reports"].append(f"{test}: {first[0][:300] if first else name}")
    for r in report["reports"]:
        print(f"  - ubsan: {r}")
    for f in report["test_findings"]:
        print(f"  {f}")
    for u in report["unsanitized"]:
        print(f"  {'-' if args.require_ubsan else '~'} unsanitized: {u}")
    if unavailable:
        return finish("INVALID", EXIT_UNAVAILABLE)
    if (
        report["reports"]
        or report["test_findings"]
        or (args.require_ubsan and report["unsanitized"])
    ):
        return finish("FAIL", EXIT_FINDINGS)
    if report["sanitized"] == 0 or report["runs"] == 0:
        # it examined nothing: no sanitized witness ran (L2)
        return finish("INVALID", EXIT_UNAVAILABLE)
    return finish("PASS", EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
