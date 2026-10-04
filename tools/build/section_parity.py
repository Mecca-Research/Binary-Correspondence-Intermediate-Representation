#!/usr/bin/env python3
"""The section-parity gate: a migrated gate section judges the CMake-built harnesses exactly as it
judges the gate's own.

tools/c/check_runtime.sh compiles each harness in one compiler command and runs the section's
script over it; the CMake project builds the same harness from runtime/manifest.json and runs the
same script as a CTest entry (docs/BCIR_BUILD_ROADMAP.md, BUILD-2). This gate builds the gate's
recipe for every harness a section takes, runs the section script once over those binaries and
once over the CMake-built ones, and requires stdout, stderr and the exit status to agree byte for
byte -- and the section to pass, printing at least one PASS line, so an empty or failing section
cannot read as parity.

    section_parity.py --harness-dir build/cmake/harnesses --cc gcc [--section runtime ...]

Exit 0 when every section agrees and passes, 1 when a section's two runs differ or the section
fails, 2 when the gate cannot run (no sections, a missing binary, no compiler, a recipe that does
not build): a skipped gate never reads as a passing one.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "runtime" / "c"
# The gate compiles every migrated harness with -std=c23 -O2 (its -Wall/-Wextra/-Werror choices do
# not change the code); a compiler that predates the c23 spelling takes the gate's own fallbacks.
RECIPE_FLAGS = (("-std=c23", "-O2"), ("-std=c2x", "-O2"), ("-std=c11", "-O2"))
STREAMS = ("status", "stdout", "stderr")


def _load_manifest_tool():
    spec = importlib.util.spec_from_file_location(
        "bcir_build_manifest", ROOT / "tools" / "build" / "manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def recipe_sources(
    manifest_tool, manifest: dict, harness: str
) -> tuple[list[str], list[str], list[str]]:
    """The gate's one-command build for a harness, derived from the manifest: its sources, then its
    libraries' (dependents first), the system libraries and the compiler options its closure names."""
    data = manifest_tool.closure(manifest, "harnesses", harness)
    return data["sources"], data["link"], data["options"]


def compiler_options(options: list[str], cc_id: str) -> list[str]:
    """The manifest's options for this compiler (`gcc:`/`clang:` prefixes scope one to a compiler)."""
    out: list[str] = []
    for option in options:
        if option.startswith("gcc:"):
            if cc_id == "gcc":
                out.append(option[4:])
        elif option.startswith("clang:"):
            if cc_id == "clang":
                out.append(option[6:])
        else:
            out.append(option)
    return out


def compiler_id(cc: str) -> str:
    try:
        text = subprocess.run([cc, "--version"], capture_output=True, timeout=30).stdout.decode(
            "utf-8", "replace"
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lowered = text.lower()
    if "clang" in lowered:
        return "clang"
    if "gcc" in lowered or "free software foundation" in lowered:
        return "gcc"
    return "unknown"


def build_recipe(
    cc: str,
    cc_id: str,
    sources: list[str],
    link: list[str],
    options: list[str],
    out: Path,
    log: list[str],
) -> bool:
    paths = [str(C_DIR / s) for s in sources]
    link_flags = ["-lm" if lib == "m" else "-pthread" for lib in link]
    for flags in RECIPE_FLAGS:
        cmd = [
            cc,
            *flags,
            *compiler_options(options, cc_id),
            "-I",
            str(C_DIR),
            *paths,
            "-o",
            str(out),
            *link_flags,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.append(f"recipe {out.name}: {cc} {' '.join(flags)}: {exc}")
            continue
        if proc.returncode == 0:
            log.append(f"recipe {out.name}: {cc} {' '.join(flags)} over {len(paths)} sources")
            return True
        log.append(
            f"recipe {out.name}: {cc} {' '.join(flags)} failed:\n{proc.stderr.decode('utf-8', 'replace')[-1500:]}"
        )
    return False


def run_section(
    script: Path, binaries: list[Path], python: str, timeout: float
) -> tuple[object, bytes, bytes]:
    env = dict(os.environ, PYTHON=python)
    try:
        proc = subprocess.run(
            ["bash", str(script), *map(str, binaries)],
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


def compare_section(
    script: Path,
    gate_binaries: list[Path],
    cmake_binaries: list[Path],
    python: str = sys.executable,
    timeout: float = 600.0,
) -> dict:
    """One section over two builds: the streams that differ, and whether the CMake run is a pass
    (exit 0 with at least one PASS line; an empty or failing section is not parity)."""
    gate = run_section(script, gate_binaries, python, timeout)
    cmake = run_section(script, cmake_binaries, python, timeout)
    differ = [name for name, a, b in zip(STREAMS, gate, cmake) if a != b]
    passed = cmake[0] == 0 and b"PASS" in cmake[1]
    return {"gate": gate, "cmake": cmake, "differ": differ, "passed": passed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--harness-dir", required=True, type=Path, help="the CMake build's harness directory"
    )
    parser.add_argument(
        "--cc", default=os.environ.get("CC") or "cc", help="the compiler for the gate's recipes"
    )
    parser.add_argument(
        "--section", action="append", default=[], help="only this section (repeatable)"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="the interpreter the sections' fixtures run under"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per section run")
    args = parser.parse_args(argv)

    manifest_tool = _load_manifest_tool()
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
    cc = shutil.which(args.cc) or (args.cc if os.access(args.cc, os.X_OK) else None)
    if cc is None:
        print(f"section-parity: UNUSABLE: no compiler {args.cc!r} on PATH")
        return 2
    cc_id = compiler_id(cc)
    for name, section in wanted.items():
        for harness in section.get("harnesses", []):
            binary = args.harness_dir / harness
            if not binary.is_file() or not os.access(binary, os.X_OK):
                print(
                    f"section-parity: UNUSABLE: {binary} is not an executable (build the harness targets first)"
                )
                return 2
        if not (ROOT / section.get("script", "")).is_file():
            print(
                f"section-parity: UNUSABLE: section {name} script {section.get('script')!r} is missing"
            )
            return 2

    log: list[str] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="bcir-section-parity-") as tmp:
        work = Path(tmp)
        for name, section in wanted.items():
            gate_binaries: list[Path] = []
            for harness in section["harnesses"]:
                sources, link, options = recipe_sources(manifest_tool, manifest, harness)
                out = work / f"{harness}-recipe"
                if not build_recipe(cc, cc_id, sources, link, options, out, log):
                    print("\n".join(log))
                    print(
                        f"section-parity: UNUSABLE: the gate's recipe for {harness} did not build"
                    )
                    return 2
                gate_binaries.append(out)
            cmake_binaries = [args.harness_dir / h for h in section["harnesses"]]
            result = compare_section(
                ROOT / section["script"], gate_binaries, cmake_binaries, args.python, args.timeout
            )
            if result["differ"]:
                failures.append(
                    f"{name}: {', '.join(result['differ'])} differ (gate status {result['gate'][0]}, cmake status {result['cmake'][0]})"
                )
                for label, run in (("gate", result["gate"]), ("cmake", result["cmake"])):
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
            else:
                passes = sum(1 for line in result["cmake"][1].splitlines() if b"PASS" in line)
                print(
                    f"section {name}: identical under both builds ({passes} PASS line(s), {len(section['harnesses'])} harness(es))"
                )
    print("\n".join(log))
    print(
        f"section-parity: {len(wanted)} section(s), {len(failures)} failing, compiler {cc} ({cc_id})"
    )
    if failures:
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print("section-parity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
