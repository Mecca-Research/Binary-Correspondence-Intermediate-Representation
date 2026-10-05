#!/usr/bin/env python3
"""The section-parity gate: a gate section judges BCIR Make's build of its binaries exactly as it
judges the CMake build's.

BCIR Make builds the C rails from runtime/manifest.json (tools/build/bcirfile.py), and the runtime
gate, tools/c/check_runtime.sh, shows each section's verdict as BCIR Make recorded it over that
build; the CMake project builds the same binaries and runs the same scripts as CTest entries
(docs/BCIR_BUILD_ROADMAP.md, BUILD-2 and BUILD-8). This gate builds the rails with BCIR Make into
--make-dir, a directory of the source tree, runs each section's script once over BCIR Make's
binaries and once over the CMake-built ones, and requires stdout, stderr and the exit status to
agree byte for byte -- and the section to pass, printing at least one PASS line, so an empty or
failing section cannot read as parity. Both runs get the compiler as CC, so a section that compiles
what a tool emits (the bcir-cc sections) compiles it alike. Until BUILD-8 the comparison was with
the gate's own one-command recipes; those compile lines are retired, and BCIR Make's build is the
one the gate runs. A value a section prints that is not a function of its binaries (the
ring's count of concurrent runs that caught an injected race) is declared in the manifest as a
`varies` pattern: a regex whose one capture group is the varying value, masked in both outputs
before they are compared. A declaration that matches nothing is stale and fails; an output that
differs between two runs of the same binaries without a declaration is reported as such, not
as a build difference. A `compiler_only` section takes no binary -- it judges what the Python
oracle emits under CC -- so there is no build to compare: it is run twice all the same, and its
two outputs must agree and pass, which holds its verdict to its inputs.

    section_parity.py --harness-dir build/cmake/harnesses --tool-dir build/cmake/runtime/c \
        --cc gcc [--cxx g++] [--make-dir build/bcir-make-parity] [--section runtime ...]

Exit 0 when every section agrees and passes, 1 when a section's two runs differ, the section fails,
or one build made a binary the other did not, 2 when the gate cannot run (no sections, a missing
binary, no compiler, no POSIX shell, a BCIR Make build that fails): a skipped gate never reads as a
passing one. The shell is probed before it is trusted: on a Windows runner `bash` on PATH is the
WSL launcher, which prints a UTF-16 notice and exits 1 -- an engine that is no engine
(docs/security/laws.md L2).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# BCIR Make's build of the rails, under the source tree (a BCIRfile names repo-relative paths).
MAKE_DIR = "build/bcir-make-parity"


STREAMS = ("status", "stdout", "stderr")


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool():
    return _load_tool("manifest")


def posix_shell() -> str | None:
    """A bash that is a shell: BCIR_SHELL or `bash` on PATH, and only if `bash -c 'echo ok'` prints
    ok and exits 0 (the Windows WSL launcher is named bash and does neither)."""
    candidate = os.environ.get("BCIR_SHELL") or shutil.which("bash")
    if not candidate:
        return None
    try:
        proc = subprocess.run([candidate, "-c", "echo ok"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return candidate if proc.returncode == 0 and proc.stdout.strip() == b"ok" else None


def run_section(
    shell: str,
    script: Path,
    binaries: list[Path],
    python: str,
    timeout: float,
    cc: str | None = None,
) -> tuple[object, bytes, bytes]:
    env = dict(os.environ, PYTHON=python)
    if cc is not None:
        env["CC"] = cc
    try:
        proc = subprocess.run(
            [shell, str(script), *map(str, binaries)],
            capture_output=True,
            timeout=timeout,
            cwd=ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ("timeout", b"", b"")
    except OSError as exc:
        return (f"oserror: {exc}", b"", b"")
    return (proc.returncode, proc.stdout, proc.stderr)


VARIES_MARK = b"<varies>"


def mask(output: bytes, varies: list[str]) -> tuple[bytes, list[str]]:
    """The output with every declared varying value replaced by a marker, line by line, and the
    patterns that matched nothing (a stale declaration masks nothing and is a finding)."""
    patterns = [re.compile(v.encode("utf-8")) for v in varies]
    hits = [0] * len(patterns)
    lines = output.split(b"\n")
    for i, line in enumerate(lines):
        for k, pattern in enumerate(patterns):
            match = pattern.search(line)
            if match is not None:
                hits[k] += 1
                line = line[: match.start(1)] + VARIES_MARK + line[match.end(1) :]
        lines[i] = line
    return b"\n".join(lines), [v for v, n in zip(varies, hits) if n == 0]


def compare_section(
    script: Path,
    make_binaries: list[Path],
    cmake_binaries: list[Path],
    python: str = sys.executable,
    timeout: float = 600.0,
    shell: str | None = None,
    varies: tuple[str, ...] | list[str] = (),
    cc: str | None = None,
) -> dict:
    """One section over two builds: the streams that differ (stdout after the declared `varies`
    masks), whether the CMake run is a pass (exit 0 with at least one PASS line; an empty or failing
    section is not parity), the declarations that matched nothing, and -- when the builds differ --
    whether a second run over BCIR Make's binaries differs from the first (`unstable`: the output
    is not a function of the binaries, so the difference is undeclared nondeterminism, not the
    build). `shell` is a probed POSIX shell (posix_shell()); without one there is nothing to run."""
    if shell is None:
        raise RuntimeError("compare_section needs a probed POSIX shell (posix_shell() found none)")
    varies = list(varies)
    make = run_section(shell, script, make_binaries, python, timeout, cc)
    cmake = run_section(shell, script, cmake_binaries, python, timeout, cc)
    make_out, stale_make = mask(make[1], varies)
    cmake_out, stale_cmake = mask(cmake[1], varies)
    masked_make = (make[0], make_out, make[2])
    masked_cmake = (cmake[0], cmake_out, cmake[2])
    differ = [name for name, a, b in zip(STREAMS, masked_make, masked_cmake) if a != b]
    unstable = False
    if differ:
        again = run_section(shell, script, make_binaries, python, timeout, cc)
        unstable = (again[0], mask(again[1], varies)[0], again[2]) != masked_make
    passed = cmake[0] == 0 and b"PASS" in cmake[1]
    stale = sorted(set(stale_make) | set(stale_cmake))
    return {
        "make": make,
        "cmake": cmake,
        "differ": differ,
        "passed": passed,
        "stale": stale,
        "unstable": unstable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--harness-dir", required=True, type=Path, help="the CMake build's harness directory"
    )
    parser.add_argument(
        "--tool-dir",
        type=Path,
        default=None,
        help="the CMake build's tool directory (needed when a section takes a manifest tool)",
    )
    parser.add_argument(
        "--cc", default=os.environ.get("CC") or "cc", help="the C compiler: BCIR Make's, and CC"
    )
    parser.add_argument("--cxx", help="the C++ compiler BCIR Make builds the seam with (optional)")
    parser.add_argument(
        "--make-dir",
        default=MAKE_DIR,
        help=f"where BCIR Make builds, relative to the source tree (default {MAKE_DIR})",
    )
    parser.add_argument(
        "--section", action="append", default=[], help="only this section (repeatable)"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="the interpreter the sections' fixtures run under"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per section run")
    parser.add_argument(
        "--unbuilt",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help="a variant or kernel the CMake build did not build here, and why (repeatable; the build passes them)",
    )
    args = parser.parse_args(argv)

    manifest_tool = _load_manifest_tool()
    generator = _load_tool("bcirfile")
    try:
        manifest = manifest_tool.load()
    except manifest_tool.ManifestError as exc:
        print(f"section-parity: UNUSABLE: {exc}")
        return 2
    sections = manifest.get("sections") if isinstance(manifest.get("sections"), dict) else {}
    wanted = {name: sections[name] for name in (args.section or sections) if name in sections}
    missing = [name for name in args.section if name not in sections]
    if missing:
        print(f"section-parity: UNUSABLE: no manifest section named {', '.join(missing)}")
        return 2
    if not wanted:
        print("section-parity: INVALID: the manifest lists no sections (nothing would be compared)")
        return 2
    variants = manifest.get("variants") if isinstance(manifest.get("variants"), dict) else {}
    kernels = manifest.get("kernels") if isinstance(manifest.get("kernels"), dict) else {}
    unbuilt: dict[str, str] = {}
    for entry in args.unbuilt:
        variant, _, reason = entry.partition("=")
        if (variant not in variants and variant not in kernels) or not reason.strip():
            print(
                f"section-parity: UNUSABLE: --unbuilt {entry!r} is not NAME=REASON for a manifest variant or kernel"
            )
            return 2
        unbuilt[variant] = reason.strip()
    # A section whose binaries the CMake build did not build is reported, by name and reason, and
    # not compared: nothing of it exists to compare. Every other section must be compared, and at
    # least one must be (L2).
    skipped = {
        name: [f"{h}: {unbuilt[h]}" for h in section.get("harnesses", []) if h in unbuilt]
        for name, section in wanted.items()
    }
    skipped = {name: why for name, why in skipped.items() if why}
    for name in skipped:
        del wanted[name]
    if not wanted:
        print(
            "section-parity: INVALID: every requested section's binaries are unbuilt here (nothing would be compared)"
        )
        return 2
    cc = shutil.which(args.cc) or (args.cc if os.access(args.cc, os.X_OK) else None)
    if cc is None:
        print(f"section-parity: UNUSABLE: no compiler {args.cc!r} on PATH")
        return 2
    shell = posix_shell()
    if shell is None:
        print(
            "section-parity: UNUSABLE: no POSIX shell (a `bash` that runs `echo ok`; set BCIR_SHELL)"
        )
        return 2
    family = generator.compiler_family(cc)
    tools = manifest.get("tools") if isinstance(manifest.get("tools"), dict) else {}

    def cmake_binary(name: str) -> Path | None:
        """Where the CMake build put a section's binary: a tool in the tool directory (None when
        none was given), a harness or variant in the harness directory."""
        if name in tools:
            return None if args.tool_dir is None else args.tool_dir / name
        return args.harness_dir / name

    for name, section in wanted.items():
        for harness in section.get("harnesses", []):
            binary = cmake_binary(harness)
            if binary is None:
                print(
                    f"section-parity: UNUSABLE: section {name} takes the tool {harness}; pass --tool-dir"
                )
                return 2
            if not binary.is_file() or not os.access(binary, os.X_OK):
                print(
                    f"section-parity: UNUSABLE: {binary} is not an executable (build the targets first)"
                )
                return 2
        if not (ROOT / section.get("script", "")).is_file():
            print(
                f"section-parity: UNUSABLE: section {name} script {section.get('script')!r} is missing"
            )
            return 2

    # BCIR Make builds every binary once; a rerun over an unchanged tree runs nothing.
    try:
        built = generator.build(cc, args.cxx, args.make_dir)
    except generator.BuildError as exc:
        print(f"section-parity: UNUSABLE: BCIR Make did not build the rails: {exc}")
        return 2
    failures: list[str] = []
    compiler_only = 0
    for name, section in wanted.items():
        unmade = [h for h in section["harnesses"] if h not in built]
        if unmade:
            failures.append(
                f"{name}: the CMake build made {', '.join(unmade)} and BCIR Make did not: "
                "the two builds disagree about this host"
            )
            continue
        make_binaries = [built[h] for h in section["harnesses"]]
        cmake_binaries = [cmake_binary(h) for h in section["harnesses"]]
        result = compare_section(
            ROOT / section["script"],
            make_binaries,
            cmake_binaries,
            args.python,
            args.timeout,
            shell,
            section.get("varies", []),
            cc,
        )
        if result["stale"]:
            failures.append(
                f"{name}: declared `varies` pattern(s) match nothing: {result['stale']}"
            )
        if result["differ"] and result["unstable"]:
            failures.append(
                f"{name}: the output differs between two runs of the same binaries -- declare the varying "
                f"value as a `varies` pattern in the manifest (value-level, line-scoped); not a build difference"
            )
            for label, run in (("BCIR Make", result["make"]), ("CMake", result["cmake"])):
                print(f"--- {name} under the {label} build (status {run[0]}) ---")
                print(run[1].decode("utf-8", "replace")[-2000:])
        elif result["differ"]:
            failures.append(
                f"{name}: {', '.join(result['differ'])} differ (BCIR Make status {result['make'][0]}, CMake status {result['cmake'][0]})"
            )
            for label, run in (("BCIR Make", result["make"]), ("CMake", result["cmake"])):
                print(f"--- {name} under the {label} build (status {run[0]}) ---")
                print(run[1].decode("utf-8", "replace")[-2000:])
                print(run[2].decode("utf-8", "replace")[-1000:])
        elif not result["passed"]:
            failures.append(
                f"{name}: the section does not pass under either build (status {result['cmake'][0]}; identical output)"
            )
            print(f"--- {name} (status {result['cmake'][0]}) ---")
            print(result["cmake"][1].decode("utf-8", "replace")[-2000:])
            print(result["cmake"][2].decode("utf-8", "replace")[-1000:])
        elif not result["stale"] and section.get("compiler_only") is True:
            passes = sum(1 for line in result["cmake"][1].splitlines() if b"PASS" in line)
            compiler_only += 1
            print(
                f"section {name}: compiler-only, the same over two runs ({passes} PASS line(s), no binary)"
            )
        elif not result["stale"]:
            passes = sum(1 for line in result["cmake"][1].splitlines() if b"PASS" in line)
            masked = (
                f", {len(section.get('varies', []))} declared varying value(s) masked"
                if section.get("varies")
                else ""
            )
            print(
                f"section {name}: identical under both builds ({passes} PASS line(s), {len(section['harnesses'])} binary(ies){masked})"
            )
    for name, why in skipped.items():
        print(
            f"section {name}: NOT COMPARED here -- its binaries were not built ({'; '.join(why)})"
        )
    not_compared = f", {len(skipped)} not built here" if skipped else ""
    of_which = f" ({compiler_only} compiler-only)" if compiler_only else ""
    print(
        f"section-parity: {len(wanted)} section(s) compared{of_which}, {len(failures)} failing{not_compared}, compiler {cc} ({family}), shell {shell}"
    )
    if failures:
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print("section-parity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
