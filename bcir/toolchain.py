"""Host compiler/linker adaptation and coherent LLVM tool discovery.

BCIR keeps target link metadata logical (for example ``-lm``).  This module adapts
that metadata only at the point where BCIR invokes the resident host compiler, and
finds version-suffixed LLVM installations without mixing incompatible major versions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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
            candidates.extend((os.path.join(root, "bin", "bash.exe"),
                               os.path.join(root, "usr", "bin", "bash.exe")))
    for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root = environ.get(key)
        if root:
            candidates.extend((os.path.join(root, "Git", "bin", "bash.exe"),
                               os.path.join(root, "Git", "usr", "bin", "bash.exe")))

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
        return (f"missing tool(s) for {self.pipeline}: {', '.join(self.missing)}; "
                f"attempted {tried}")


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


def _resolve_exact(names: Sequence[str], spelling, dirs: Sequence[str],
                   attempted: list[str]) -> dict[str, str] | None:
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
        result = subprocess.run([path, "--version"], capture_output=True, text=True,
                                timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    banner = (result.stdout + "\n" + result.stderr)[:16384]
    for pattern in (r"\b(?:clang|LLVM)\s+version\s+(\d+)",
                    r"\bLLD\s+(\d+)(?:\.|\b)"):
        match = re.search(pattern, banner, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _coherent_reported_major(paths: Mapping[str, str], minimum_major: int) -> tuple[bool, int | None]:
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
    missing = tuple(name for name in names if not any(
        _find_named(candidate, dirs) for candidate in
        ([name + suffix] if suffix else []) + [name] +
        [f"{name}-{major}" for major in sorted(_available_majors(name, dirs, minimum_major), reverse=True)]
    ))
    if not missing:
        # Every tool exists, but not at one common major: report all as incompatible.
        missing = names
    return LLVMToolchain({}, None, pipeline, tuple(attempted), missing)
