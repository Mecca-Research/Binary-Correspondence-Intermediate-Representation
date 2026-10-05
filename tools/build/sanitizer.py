#!/usr/bin/env python3
"""Whether a compiler's sanitizer runtime works on this host: one predicate for every build.

A sanitizer build needs the sanitizer's runtime (compiler-rt for Clang, libtsan for GCC), and a
runtime that is installed may still refuse to start: older ThreadSanitizer runtimes reject the
high-entropy mappings newer kernels hand out, which is why the runs go through `setarch -R`.
Available therefore means that a trivial program builds with -fsanitize=<name> AND runs, under
the no-randomisation wrapper the section's own runs use.

Three builds ask this question about the manifest's sanitizer `variants`: tools/c/check_runtime.sh
before it builds the ring's ThreadSanitizer binaries, the CMake configure before it adds the
variants as targets, and (through the CMake build's record of what it did not build)
tools/build/section_parity.py. With one predicate they cannot disagree about this host
(docs/security/laws.md L12, L14). Where a job installed the runtime, its absence is a failure
that job owns (BCIR_REQUIRE_TSAN, L2); this tool only answers.

    sanitizer.py --cc clang thread

Exit 0 when available; 1 when unavailable, with the reason; 2 when the request is unusable. The
verdict is one line on stdout either way.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The sanitizers a manifest variant may name (tools/build/manifest.py's M14 holds the manifest to
# this list): the ones a section builds a variant with today.
NAMES = ("thread",)
PROBE = "int main(void) { return 0; }\n"


def norandom() -> list[str]:
    """`setarch <machine> -R` when this host can run a program without address randomisation --
    the wrapper the gate's section runs use -- else nothing."""
    setarch = shutil.which("setarch")
    machine = os.uname().machine if hasattr(os, "uname") else ""
    if not setarch or not machine:
        return []
    try:
        ok = subprocess.run(
            [setarch, machine, "-R", "true"], capture_output=True, timeout=30
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [setarch, machine, "-R"] if ok == 0 else []


def _tail(data: bytes) -> str:
    lines = data.decode("utf-8", "replace").strip().splitlines()
    return lines[-1].strip() if lines else "no output"


def probe(cc: str, name: str, timeout: float = 120.0) -> tuple[bool, str]:
    """(available, reason): a trivial program built with `cc -fsanitize=name` and run."""
    if name not in NAMES:
        raise ValueError(f"unknown sanitizer {name!r} (known: {', '.join(NAMES)})")
    flag = f"-fsanitize={name}"
    with tempfile.TemporaryDirectory(prefix="bcir-sanitizer-") as tmp:
        source = Path(tmp) / "probe.c"
        binary = Path(tmp) / "probe"
        source.write_text(PROBE, encoding="utf-8", newline="\n")
        try:
            built = subprocess.run(
                [cc, flag, str(source), "-o", str(binary)], capture_output=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{cc} {flag} could not be run: {exc}"
        if built.returncode != 0:
            return False, (
                f"{cc} {flag} does not build a trivial program "
                f"(no {name} sanitizer runtime): {_tail(built.stderr)}"
            )
        try:
            ran = subprocess.run([*norandom(), str(binary)], capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"a trivial {flag} program could not be run: {exc}"
        if ran.returncode != 0:
            return False, (
                f"a trivial {flag} program built but did not run "
                f"(status {ran.returncode}): {_tail(ran.stderr)}"
            )
    return True, f"{cc} {flag} builds and runs a trivial program"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default=os.environ.get("CC") or "cc", help="the compiler to probe")
    parser.add_argument("name", help=f"the sanitizer ({', '.join(NAMES)})")
    args = parser.parse_args(argv)
    if args.name not in NAMES:
        print(f"sanitizer: UNUSABLE: unknown sanitizer {args.name!r} (known: {', '.join(NAMES)})")
        return 2
    available, reason = probe(args.cc, args.name)
    print(f"sanitizer {args.name}: {'AVAILABLE' if available else 'UNAVAILABLE'}: {reason}")
    return 0 if available else 1


if __name__ == "__main__":
    sys.exit(main())
