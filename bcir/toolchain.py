"""Host compiler/linker adaptation, coherent LLVM tool discovery, and the host's dependencies.

BCIR keeps target link metadata logical (for example ``-lm``).  This module adapts
that metadata only at the point where BCIR invokes the resident host compiler, and
finds version-suffixed LLVM installations without mixing incompatible major versions.

It is also the one place a Python harness asks what the host has (BUILD-4,
docs/BCIR_BUILD_ROADMAP.md): which C compiler to build with (`host_c_compiler`) and whether
an optional numeric library links, with which flags (`optional_library`). The CMake configure
asks the same predicate (``python -m bcir.toolchain probe-libraries``) and records the answers
in ``bcir-deps.json``; a harness run with ``BCIR_DEPS_INDEX`` pointing at that index reads them
from it instead of probing, so a harness and the configure cannot disagree about a dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


def _is_windows(platform: str) -> bool:
    value = platform.lower()
    return value.startswith("win") or value in ("nt", "windows")


def host_link_args(logical_flags: Iterable[str], platform: str | None = None) -> list[str]:
    """Return resident-host linker arguments for logical BCIR flags.

    ``-lm`` remains part of BCIR's target/link attestation.  Windows' CRT already
    supplies those symbols, while forwarding ``-lm`` to an MSVC-style linker asks
    for a nonexistent ``m.lib``.  No other argument is rewritten or reordered.
    """

    host = sys.platform if platform is None else platform
    return [flag for flag in logical_flags if not (_is_windows(host) and flag == "-lm")]


def host_bash(
    platform: str | None = None,
    *,
    which: Callable[[str], str | None] | None = None,
    is_file: Callable[[str], bool] | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return a real Bash executable for the resident host, if one exists.

    Native Windows still ships ``System32\\bash.exe`` as the legacy WSL launcher.
    Merely finding that executable does not mean a WSL distribution is installed,
    and invoking it from a portability test produces a UTF-16 WSL setup diagnostic.
    Prefer the Bash installed beside Git for Windows and reject a lone WSL launcher.

    The injected lookup hooks keep this policy deterministic in unit tests; normal
    callers use the resident filesystem and ``PATH``.
    """

    host = sys.platform if platform is None else platform
    find = shutil.which if which is None else which
    exists = os.path.isfile if is_file is None else is_file
    environ = os.environ if env is None else env
    if not _is_windows(host):
        return find("bash")

    candidates: list[str] = []
    git = find("git")
    if git:
        # The common layout is Git/cmd/git.exe plus Git/bin/bash.exe.  Also try
        # one ancestor higher for installations that expose mingw64/bin/git.exe.
        parent = os.path.dirname(os.path.abspath(git))
        for root in (os.path.dirname(parent), os.path.dirname(os.path.dirname(parent))):
            candidates.extend(
                (
                    os.path.join(root, "bin", "bash.exe"),
                    os.path.join(root, "usr", "bin", "bash.exe"),
                )
            )
    for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root = environ.get(key)
        if root:
            candidates.extend(
                (
                    os.path.join(root, "Git", "bin", "bash.exe"),
                    os.path.join(root, "Git", "usr", "bin", "bash.exe"),
                )
            )

    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized not in seen and exists(candidate):
            return candidate
        seen.add(normalized)

    bash = find("bash")
    if bash is None:
        return None
    normalized = os.path.normcase(os.path.abspath(bash)).replace("\\", "/").lower()
    if normalized.endswith("/windows/system32/bash.exe"):
        return None
    return bash


@dataclass(frozen=True)
class LLVMToolchain:
    """One coherent LLVM tool resolution attempt."""

    paths: Mapping[str, str]
    major: int | None
    pipeline: str
    attempted: tuple[str, ...]
    missing: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing

    @property
    def message(self) -> str:
        if self.ok:
            label = "unversioned" if self.major is None else f"LLVM {self.major}"
            return f"{self.pipeline}: resolved coherent {label} toolchain"
        tried = ", ".join(self.attempted) or "(none)"
        return f"missing tool(s) for {self.pipeline}: {', '.join(self.missing)}; attempted {tried}"


def _suffix_value(value: str | None) -> str | None:
    if not value:
        return None
    return value if value.startswith("-") else f"-{value}"


def _candidate_dirs(env: Mapping[str, str], path: str | None) -> list[str]:
    dirs: list[str] = []
    llvm_bin = env.get("LLVM_BIN")
    if llvm_bin:
        dirs.append(os.path.abspath(os.path.expanduser(llvm_bin)))
    for entry in (env.get("PATH", "") if path is None else path).split(os.pathsep):
        if entry:
            full = os.path.abspath(os.path.expanduser(entry))
            if full not in dirs:
                dirs.append(full)
    return dirs


def _find_named(name: str, dirs: Sequence[str]) -> str | None:
    """Find an exact basename, including ``.exe`` when present."""

    suffixes = (".exe", "") if _is_windows(sys.platform) else ("", ".exe")
    for directory in dirs:
        for suffix in suffixes:
            candidate = os.path.join(directory, name + suffix)
            # Route exact-path probes through shutil.which too: quick-tier capability
            # gating wraps it, so PATH scanning cannot accidentally bypass the tier.
            found = shutil.which(candidate)
            if found:
                return os.path.abspath(found)
    # Preserve platform-specific PATHEXT behavior for ordinary PATH resolution.
    found = shutil.which(name, path=os.pathsep.join(dirs))
    return os.path.abspath(found) if found else None


def _resolve_exact(
    names: Sequence[str], spelling, dirs: Sequence[str], attempted: list[str]
) -> dict[str, str] | None:
    spellings = {name: spelling(name) for name in names}
    for candidate in spellings.values():
        if candidate not in attempted:
            attempted.append(candidate)
    # A coherent installation is a single bin directory, not a per-tool mixture
    # assembled from unrelated PATH entries that merely share a version suffix.
    for directory in dirs:
        out: dict[str, str] = {}
        for name, candidate in spellings.items():
            found = _find_named(candidate, (directory,))
            if found is None:
                break
            out[name] = found
        if len(out) == len(names):
            return out
    return None


def _available_majors(name: str, dirs: Sequence[str], minimum_major: int) -> set[int]:
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)(?:\.exe)?$", re.IGNORECASE)
    majors: set[int] = set()
    for directory in dirs:
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            match = pattern.fullmatch(entry)
            if match and int(match.group(1)) >= minimum_major:
                majors.add(int(match.group(1)))
    return majors


def _reported_llvm_major(path: str) -> int | None:
    """Best-effort major for an unversioned LLVM-family executable.

    Distribution alternatives commonly expose ``/usr/bin/clang`` as a symlink into
    ``/usr/lib/llvm-N/bin``.  Fall back to the bounded ``--version`` banner when the
    real path carries no version. Unknown wrapper scripts remain usable; a positively
    identified old or mixed set does not bypass ``minimum_major`` merely because its
    filenames are unversioned.
    """
    real = os.path.realpath(path)
    match = re.search(r"(?:^|[\\/])llvm-(\d+)(?:[\\/]|$)", real, re.IGNORECASE)
    if match:
        return int(match.group(1))
    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    banner = (result.stdout + "\n" + result.stderr)[:16384]
    for pattern in (r"\b(?:clang|LLVM)\s+version\s+(\d+)", r"\bLLD\s+(\d+)(?:\.|\b)"):
        match = re.search(pattern, banner, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _coherent_reported_major(
    paths: Mapping[str, str], minimum_major: int
) -> tuple[bool, int | None]:
    majors = {_reported_llvm_major(path) for path in paths.values()}
    majors.discard(None)
    if not majors:
        return True, None
    if len(majors) != 1:
        return False, None
    major = next(iter(majors))
    return major >= minimum_major, major


def resolve_llvm_tools(
    *tool_names: str,
    pipeline: str = "LLVM pipeline",
    minimum_major: int = 15,
    env: Mapping[str, str] | None = None,
    path: str | None = None,
) -> LLVMToolchain:
    """Resolve all requested tools from one coherent LLVM installation.

    Resolution order is intentional: an explicit ``LLVM_SUFFIX``; a complete
    unversioned toolchain; then the highest common ``name-N`` major. ``LLVM_BIN``
    is searched before PATH.  A failed resolution reports every spelling attempted.
    """

    if not tool_names:
        raise ValueError("at least one LLVM tool name is required")
    names = tuple(dict.fromkeys(tool_names))
    environ = os.environ if env is None else env
    dirs = _candidate_dirs(environ, path)
    attempted: list[str] = []

    suffix = _suffix_value(environ.get("LLVM_SUFFIX"))
    if suffix is not None:
        resolved = _resolve_exact(names, lambda name: name + suffix, dirs, attempted)
        if resolved is not None:
            major_text = suffix[1:]
            major = int(major_text) if major_text.isdigit() else None
            return LLVMToolchain(resolved, major, pipeline, tuple(attempted))

    resolved = _resolve_exact(names, lambda name: name, dirs, attempted)
    if resolved is not None:
        acceptable, reported_major = _coherent_reported_major(resolved, minimum_major)
        if acceptable:
            return LLVMToolchain(resolved, reported_major, pipeline, tuple(attempted))

    major_sets = [_available_majors(name, dirs, minimum_major) for name in names]
    common = set.intersection(*major_sets) if major_sets else set()
    for major in sorted(common, reverse=True):
        resolved = _resolve_exact(names, lambda name, m=major: f"{name}-{m}", dirs, attempted)
        if resolved is not None:
            return LLVMToolchain(resolved, major, pipeline, tuple(attempted))

    # Include the useful versioned spellings even when no common major exists.
    for name, majors in zip(names, major_sets):
        for major in sorted(majors, reverse=True):
            spelling = f"{name}-{major}"
            if spelling not in attempted:
                attempted.append(spelling)
    missing = tuple(
        name
        for name in names
        if not any(
            _find_named(candidate, dirs)
            for candidate in ([name + suffix] if suffix else [])
            + [name]
            + [
                f"{name}-{major}"
                for major in sorted(_available_majors(name, dirs, minimum_major), reverse=True)
            ]
        )
    )
    if not missing:
        # Every tool exists, but not at one common major: report all as incompatible.
        missing = names
    return LLVMToolchain({}, None, pipeline, tuple(attempted), missing)


# --- the host's dependencies: one predicate for every harness and the configure (BUILD-4) ---------

DEPS_INDEX_ENV = "BCIR_DEPS_INDEX"
DEPS_SCHEMA = "bcir-deps.v1"
# The rows every index carries: the configure's own needs, then the optional numeric libraries.
DEPS_REQUIRED = ("PYTHON3", "THREADS", "MLIR")


class DepsIndexError(Exception):
    """``BCIR_DEPS_INDEX`` names an index this module cannot answer from: unreadable, of another
    schema, malformed, or without the row asked for. It is never answered by probing instead -- a
    harness that fell back to a probe of its own could disagree with the configure that wrote the
    index, which is the disagreement the index exists to rule out (docs/security/laws.md L12, L14)."""


@dataclass(frozen=True)
class OptionalLibrary:
    """A numeric library the twin's link-flag rules name: a probe program that includes its header
    and CALLS one of its functions -- so the link must resolve the symbol, not only find a file --
    and the flag sets to link it with, tried in order."""

    name: str
    probe: str
    candidates: tuple[tuple[str, ...], ...]
    needs_libm: bool = False


OPTIONAL_LIBRARIES: dict[str, OptionalLibrary] = {
    lib.name: lib
    for lib in (
        OptionalLibrary(
            "FFTW3F",
            "#include <fftw3.h>\n"
            "int main(void){ fftwf_plan p = 0; if (p) fftwf_execute(p); return 0; }\n",
            (("-lfftw3f",), ("-lfftw3",)),
        ),
        OptionalLibrary(
            "LAPACKE",
            "#include <lapacke.h>\n"
            "int main(void){ float a[1] = {1.0f}, b[1] = {1.0f}; lapack_int ipiv[1];\n"
            "  return (int)LAPACKE_sgesv(LAPACK_ROW_MAJOR, 1, 1, a, 1, ipiv, b, 1) * 0; }\n",
            (("-llapacke", "-llapack"), ("-llapacke",), ("-llapack",)),
        ),
        OptionalLibrary(
            "GSL",
            "#include <gsl/gsl_statistics.h>\n"
            "int main(void){ double x[2] = {1, 2}; return (int)gsl_stats_mean(x, 1, 2) * 0; }\n",
            (("-lgsl", "-lgslcblas"), ("-lgsl",)),
            needs_libm=True,
        ),
        OptionalLibrary(
            "SLEEF",
            "#include <sleef.h>\nint main(void){ return (int)Sleef_expf1_u10(0.0f) * 0; }\n",
            (("-lsleef",),),
            needs_libm=True,
        ),
        OptionalLibrary(
            "CERF",
            "#include <cerf.h>\nint main(void){ return (int)erfcxf(0.0f) * 0; }\n",
            (("-lcerf",),),
            needs_libm=True,
        ),
    )
}


def host_c_compiler(env: Mapping[str, str] | None = None) -> str | None:
    """The C compiler a Python harness builds with: ``$CC`` when it is set -- and then only it, None
    when it does not resolve, never a substitute the caller did not name -- else the first of
    clang, gcc and cc on PATH. One pick for every harness (native_bench, the link-flag rule tests,
    the model gates), where each had chosen in its own order. Looked up through ``shutil.which``
    at call time, so a test tier that hides the toolchain hides it here too."""

    environ = os.environ if env is None else env
    named = environ.get("CC", "").strip()
    if named:
        return shutil.which(named)
    for name in ("clang", "gcc", "cc"):
        found = shutil.which(name)
        if found:
            return found
    return None


def probe_library(
    lib: OptionalLibrary,
    cc: str,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    timeout: float = 120.0,
) -> tuple[tuple[str, ...] | None, str]:
    """Compile and link the library's probe under ``cc`` with each flag set in turn: the first set
    that links and a detail, or None and why not."""

    runner = subprocess.run if run is None else run
    with tempfile.TemporaryDirectory(prefix="bcir-deps-") as tmp:
        src = os.path.join(tmp, "probe.c")
        with open(src, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(lib.probe)
        for flags in lib.candidates:
            cmd = host_link_args(
                [cc, src, *flags, *(["-lm"] if lib.needs_libm else []), "-o", src[:-2]]
            )
            try:
                proc = runner(cmd, capture_output=True, timeout=timeout)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return None, f"the probe did not run under {cc} ({exc})"
            if proc.returncode == 0:
                return flags, f"links with {' '.join(flags)} under {cc}"
    tried = ", ".join(" ".join(flags) for flags in lib.candidates)
    return None, f"no flag set links under {cc} (tried {tried})"


def deps_index_problems(data: object) -> list[str]:
    """What is wrong with a dependency index (``bcir-deps.json``), empty when it is well-formed:
    schema bcir-deps.v1, the compilers and system named, one {name, found, detail} row per
    dependency with every required and every optional-library row present, and each optional
    library's ``link`` flags -- non-empty exactly when it was found. tools/build/manifest.py's D1
    asks this same predicate."""

    if not isinstance(data, dict):
        return ["the index is not a JSON object"]
    problems: list[str] = []
    if data.get("schema") != DEPS_SCHEMA:
        problems.append(f"schema is {data.get('schema')!r}, not {DEPS_SCHEMA!r}")
    for key in ("c_compiler", "cxx_compiler", "system"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"{key} is missing or empty")
    rows = data.get("dependencies")
    if not isinstance(rows, list) or not rows:
        return problems + [
            "dependencies is missing or empty (an index that records nothing indexes nothing)"
        ]
    names: list[str] = []
    for i, row in enumerate(rows):
        library = isinstance(row, dict) and row.get("name") in OPTIONAL_LIBRARIES
        keys = {"name", "found", "detail", "link"} if library else {"name", "found", "detail"}
        if not isinstance(row, dict) or set(row) != keys:
            shape = "{name, found, detail, link}" if library else "{name, found, detail}"
            problems.append(f"dependency row {i} is not {shape}")
            continue
        if not isinstance(row["name"], str) or not row["name"]:
            problems.append(f"dependency row {i} has no name")
        if not isinstance(row["found"], bool):
            problems.append(
                f"dependency {row.get('name')!r} has found={row['found']!r}, not a JSON boolean"
            )
        if not isinstance(row["detail"], str):
            problems.append(f"dependency {row.get('name')!r} detail is not a string")
        if library:
            link = row["link"]
            if not isinstance(link, list) or any(
                not isinstance(flag, str) or not flag.startswith("-") for flag in link
            ):
                problems.append(f"dependency {row['name']} link is not a list of flags")
            elif bool(link) != (row["found"] is True):
                problems.append(
                    f"dependency {row['name']} is found={row['found']!r} with link {link!r}: a "
                    "found library names its flags and an absent one none"
                )
        names.append(str(row["name"]))
    for name in sorted(set(names)):
        if names.count(name) > 1:
            problems.append(f"dependency {name} is recorded twice")
    for name in (*DEPS_REQUIRED, *OPTIONAL_LIBRARIES):
        if name not in names:
            problems.append(f"dependency {name} is not recorded")
    return problems


def load_deps_index(path: str | os.PathLike[str]) -> dict:
    """The configure's dependency index, held to its schema (``deps_index_problems``)."""

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DepsIndexError(f"{path}: {exc}") from exc
    problems = deps_index_problems(data)
    if problems:
        raise DepsIndexError(f"{path}: {'; '.join(problems)}")
    return data


def optional_library(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[str, ...] | None:
    """The link flags of the optional library ``name`` (a key of OPTIONAL_LIBRARIES), or None when
    the host lacks it. With ``BCIR_DEPS_INDEX`` set, the configure's answer, read from its index --
    an unreadable index or a missing row raises DepsIndexError, and nothing is probed; without it,
    the probe the configure runs, under ``host_c_compiler()``."""

    if name not in OPTIONAL_LIBRARIES:
        raise ValueError(f"{name!r} is not an optional library ({', '.join(OPTIONAL_LIBRARIES)})")
    environ = os.environ if env is None else env
    index = environ.get(DEPS_INDEX_ENV, "").strip()
    if index:
        rows = {row["name"]: row for row in load_deps_index(index)["dependencies"]}
        row = rows[name]  # deps_index_problems has required every optional library's row
        return tuple(row["link"]) if row["found"] else None
    cc = host_c_compiler(environ)
    if cc is None:
        return None
    flags, _detail = probe_library(OPTIONAL_LIBRARIES[name], cc, run)
    return flags


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m bcir.toolchain probe-libraries --cc CC``: the configure's question about the
    optional libraries, asked of the predicate the harnesses use. Prints one JSON object, by
    library name, of {found, link, detail}; exit 0 with an answer for every library, 2 when it
    cannot ask (no such compiler)."""

    parser = argparse.ArgumentParser(prog="python -m bcir.toolchain")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser(
        "probe-libraries", help="probe every optional library under one compiler"
    )
    probe.add_argument("--cc", required=True, help="the C compiler the configure uses")
    args = parser.parse_args(argv)
    cc = shutil.which(args.cc) or (args.cc if os.access(args.cc, os.X_OK) else None)
    if cc is None:
        print(f"bcir.toolchain: UNUSABLE: no compiler {args.cc!r}", file=sys.stderr)
        return 2
    answers = {}
    for name, lib in OPTIONAL_LIBRARIES.items():
        flags, detail = probe_library(lib, cc)
        answers[name] = {"found": flags is not None, "link": list(flags or ()), "detail": detail}
    print(json.dumps(answers, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
