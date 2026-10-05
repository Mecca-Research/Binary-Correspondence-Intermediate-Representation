#!/usr/bin/env python3
"""The install gate (BUILD-3, docs/BCIR_BUILD_ROADMAP.md): an installed BCIR serves an out-of-tree
consumer as the source tree serves its own harnesses.

    install_consumer.py --build-dir build/cmake-gcc [--harness test_runtime] [--keep DIR]

1. `cmake --install <build-dir> --prefix <scratch>/prefix`: the configured build's own install rules
   (cmake/BCIRInstall.cmake).
2. The prefix is held to the manifest: an archive for every library, every tool, every public header
   (runtime/c/bcir_*.h; the C++ seam's bcir_*.h/.hpp when the build has the seam; bcir_version.h) and
   the package files. A missing entry is a finding, and so is an extra header: the package publishes
   what the tree publishes, nothing else.
3. tools/build/consumer is staged in a scratch directory with a copy of the harness's source -- so a
   quoted #include cannot reach the source tree -- and configured against the prefix with the build's
   own generator and compilers (read from its CMakeCache.txt), then built: find_package(BCIR <version>),
   every manifest library and tool an imported target, every public header compiled alone, the version
   header agreeing with the package, the harness linked against BCIR::* alone.
4. A consumer asking for the next major version must fail its configure: the package's version file is
   consulted, not decorative.
5. The section that judges the harness runs over the consumer's binary and over the tree's own
   (<build-dir>/harnesses/<harness>): byte-identical output and a PASS line, as the section-parity gate
   holds the gate's builds (tools/build/section_parity.py's compare_section).

Exit 0 when every step holds, 1 with a finding, 2 when the gate cannot run (no cmake, no configured
build, an install that does not run, no POSIX shell): every exit is a verdict (docs/security/laws.md
L1), and a skipped gate never reads as a passing one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONSUMER = ROOT / "tools" / "build" / "consumer" / "CMakeLists.txt"
PYPROJECT = ROOT / "pyproject.toml"
VERSION_LINE = re.compile(r'^version = "([0-9]+)\.([0-9]+)\.([0-9]+)"$', re.M)
PACKAGE_FILES = ("BCIRConfig.cmake", "BCIRConfigVersion.cmake", "BCIRTargets.cmake")
CACHE_ENTRY = re.compile(r"^([A-Za-z0-9_.-]+):[A-Z_]+=(.*)$")


class Unusable(Exception):
    """The gate cannot run here; the message says why."""


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def package_version(text: str) -> str:
    """pyproject.toml's one `version = "X.Y.Z"` line, as the CMake project reads it."""
    found = VERSION_LINE.findall(text)
    if len(found) != 1:
        raise Unusable(f'pyproject.toml carries {len(found)} `version = "X.Y.Z"` lines, not one')
    return ".".join(found[0])


def read_cache(build_dir: Path) -> dict[str, str]:
    """The entries of a configured build's CMakeCache.txt."""
    try:
        text = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise Unusable(f"{build_dir} is no configured build ({exc})") from exc
    cache: dict[str, str] = {}
    for line in text.splitlines():
        match = CACHE_ENTRY.match(line)
        if match:
            cache[match.group(1)] = match.group(2)
    return cache


def public_headers(root: Path, with_seam: bool) -> list[str]:
    """The headers the package publishes, by the rule cmake/BCIRInstall.cmake installs them by."""
    names = {p.name for p in (root / "runtime" / "c").glob("bcir_*.h")}
    if with_seam:
        cpp = root / "runtime" / "cpp"
        names |= {p.name for p in cpp.glob("bcir_*.h")} | {p.name for p in cpp.glob("bcir_*.hpp")}
    return sorted(names | {"bcir_version.h"})


def layout_problems(
    prefix: Path, manifest: dict, headers: list[str], dirs: dict[str, str], with_seam: bool
) -> list[str]:
    """What the installed prefix lacks, or carries beyond what the tree publishes."""
    problems: list[str] = []
    libdir, bindir = prefix / dirs["lib"], prefix / dirs["bin"]
    includedir = prefix / dirs["include"] / "bcir"
    libraries = list(manifest.get("libraries", {}))
    if with_seam:
        libraries += list(manifest.get("seam_libraries", {}))
    for name in libraries:
        if not (libdir / f"lib{name}.a").is_file():
            problems.append(f"library {name}: no {dirs['lib']}/lib{name}.a")
    for name in manifest.get("tools", {}):
        tool = bindir / name
        if not tool.is_file() or not os.access(tool, os.X_OK):
            problems.append(f"tool {name}: no executable {dirs['bin']}/{name}")
    installed = sorted(p.name for p in includedir.glob("*")) if includedir.is_dir() else []
    for name in sorted(set(headers) - set(installed)):
        problems.append(f"header {name}: not installed in {dirs['include']}/bcir")
    for name in sorted(set(installed) - set(headers)):
        problems.append(f"header {name}: installed, but the tree publishes no such header")
    for name in PACKAGE_FILES:
        if not (libdir / "cmake" / "BCIR" / name).is_file():
            problems.append(f"package file {name}: not in {dirs['lib']}/cmake/BCIR")
    return problems


def stage_consumer(work: Path, harness_source: Path) -> Path:
    """The consumer's source directory: its CMakeLists.txt and a copy of the harness, nothing else."""
    src = work / "consumer-src"
    src.mkdir(parents=True)
    shutil.copyfile(CONSUMER, src / "CMakeLists.txt")
    shutil.copyfile(harness_source, src / harness_source.name)
    return src


def refused_by_version(output: str, wanted: str, version: str) -> bool:
    """Whether a configure's output is find_package's own refusal of the package for its version:
    CMake names the requested version and the package version it considered and did not accept
    (its message wraps lines, so whitespace is compared as one space)."""
    text = " ".join(output.split())
    return f'compatible with requested version "{wanted}"' in text and f"version: {version}" in text


def _run(cmd: list[str], log: list[str], timeout: float, cwd: Path | None = None) -> int:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        log.append(f"$ {' '.join(cmd)}\n(timed out after {timeout:.0f} s)")
        return -1
    except OSError as exc:
        log.append(f"$ {' '.join(cmd)}\n({exc})")
        return -1
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    log.append(f"$ {' '.join(cmd)}\n{out[-4000:]}")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", required=True, type=Path, help="a configured, built tree")
    parser.add_argument("--harness", default="test_runtime", help="the manifest harness to rebuild")
    parser.add_argument("--keep", type=Path, help="work here instead of a scratch directory")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per command")
    args = parser.parse_args(argv)
    try:
        return _gate(args)
    except Unusable as exc:
        print(f"install-consumer: UNUSABLE: {exc}")
        return 2


def _gate(args: argparse.Namespace) -> int:
    manifest_tool = _load("manifest")
    section_tool = _load("section_parity")
    try:
        manifest = manifest_tool.load()
    except manifest_tool.ManifestError as exc:
        raise Unusable(str(exc)) from exc
    version = package_version(PYPROJECT.read_text(encoding="utf-8"))
    build_dir = args.build_dir.resolve()
    cache = read_cache(build_dir)
    cmake = shutil.which("cmake") or cache.get("CMAKE_COMMAND")
    if not cmake:
        raise Unusable("no cmake")
    harness = manifest.get("harnesses", {}).get(args.harness)
    if not isinstance(harness, dict) or len(harness.get("sources", [])) != 1:
        raise Unusable(f"{args.harness} is no single-source manifest harness")
    harness_source = ROOT / "runtime" / "c" / harness["sources"][0]
    tree_binary = build_dir / "harnesses" / args.harness
    if not tree_binary.is_file():
        raise Unusable(f"{tree_binary} is not built (build the tree first)")
    sections = [
        (name, s)
        for name, s in (manifest.get("sections") or {}).items()
        if isinstance(s, dict) and s.get("harnesses") == [args.harness]
    ]
    if not sections:
        raise Unusable(f"no manifest section judges {args.harness} alone")
    shell = section_tool.posix_shell()
    if shell is None:
        raise Unusable("no POSIX shell (a `bash` that runs `echo ok`; set BCIR_SHELL)")
    with_seam = cache.get("BCIR_BUILD_CPP", "ON").upper() in ("ON", "1", "TRUE", "YES")
    dirs = {
        "lib": cache.get("CMAKE_INSTALL_LIBDIR", "lib"),
        "bin": cache.get("CMAKE_INSTALL_BINDIR", "bin"),
        "include": cache.get("CMAKE_INSTALL_INCLUDEDIR", "include"),
    }
    headers = public_headers(ROOT, with_seam)
    libraries = list(manifest.get("libraries", {}))
    if with_seam:
        libraries += list(manifest.get("seam_libraries", {}))
    log: list[str] = []
    findings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="bcir-install-") as scratch:
        work = Path(args.keep).resolve() if args.keep else Path(scratch)
        if args.keep:
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
        prefix = work / "prefix"
        if _run([cmake, "--install", str(build_dir), "--prefix", str(prefix)], log, args.timeout):
            print("\n".join(log))
            raise Unusable(f"`cmake --install {build_dir}` did not run")
        findings += layout_problems(prefix, manifest, headers, dirs, with_seam)
        src = stage_consumer(work, harness_source)
        configure = [
            cmake,
            "-S",
            str(src),
            "-G",
            cache.get("CMAKE_GENERATOR", "Ninja"),
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_C_COMPILER={cache.get('CMAKE_C_COMPILER', 'cc')}",
            f"-DCMAKE_CXX_COMPILER={cache.get('CMAKE_CXX_COMPILER', 'c++')}",
            f"-DBCIR_CONSUMER_LIBRARIES={';'.join(libraries)}",
            f"-DBCIR_CONSUMER_TOOLS={';'.join(manifest.get('tools', {}))}",
            f"-DBCIR_CONSUMER_HEADERS={len(headers)}",
            f"-DBCIR_CONSUMER_HARNESS={args.harness}",
            f"-DBCIR_CONSUMER_HARNESS_LIBRARIES={';'.join(harness.get('libraries', []))}",
        ]
        if cache.get("CMAKE_MAKE_PROGRAM"):
            configure.append(f"-DCMAKE_MAKE_PROGRAM={cache['CMAKE_MAKE_PROGRAM']}")
        build = work / "consumer-build"
        if _run(
            [*configure, "-B", str(build), f"-DBCIR_CONSUMER_VERSION={version}"], log, args.timeout
        ):
            findings.append(f"the consumer did not configure against BCIR {version}")
        elif _run([cmake, "--build", str(build), "-j", "2"], log, args.timeout):
            findings.append("the consumer did not build against the installed package")
        else:
            major = int(version.split(".")[0])
            refused = work / "consumer-refused"
            wanted = f"{major + 1}.0.0"
            if not _run(
                [*configure, "-B", str(refused), f"-DBCIR_CONSUMER_VERSION={wanted}"],
                log,
                args.timeout,
            ):
                findings.append(
                    f"a consumer asking for BCIR {wanted} configured against {version}: "
                    "the package's version file decided nothing"
                )
            elif not refused_by_version(log[-1], wanted, version):
                # The consumer also refuses a version it did not ask for: a refusal is the version
                # file's only when find_package itself says so (L11: the witness must hit its law).
                findings.append(
                    f"a consumer asking for BCIR {wanted} was refused, but not by find_package's "
                    "version check: the package's version file was never what refused it"
                )
            consumer_binary = build / args.harness
            for name, section in sections:
                result = section_tool.compare_section(
                    ROOT / section["script"],
                    [tree_binary],
                    [consumer_binary],
                    sys.executable,
                    args.timeout,
                    shell,
                    section.get("varies", []),
                    cache.get("CMAKE_C_COMPILER"),
                )
                if result["differ"]:
                    findings.append(
                        f"section {name}: {', '.join(result['differ'])} differ between the tree's "
                        f"{args.harness} and the consumer's"
                    )
                elif not result["passed"]:
                    findings.append(
                        f"section {name}: does not pass over the consumer's {args.harness}"
                    )
                else:
                    passes = sum(1 for line in result["cmake"][1].splitlines() if b"PASS" in line)
                    print(
                        f"install-consumer: section {name} identical over the tree's and the "
                        f"consumer's {args.harness} ({passes} PASS line(s))"
                    )
    if findings:
        print("\n".join(log))
        for finding in findings:
            print(f"  FAIL {finding}")
        print(f"install-consumer: FAIL ({len(findings)} finding(s))")
        return 1
    summary = {
        "version": version,
        "libraries": len(libraries),
        "tools": len(manifest.get("tools", {})),
        "headers": len(headers),
        "harness": args.harness,
    }
    print(f"install-consumer: PASS {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
