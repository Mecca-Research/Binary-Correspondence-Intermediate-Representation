#!/usr/bin/env python3
"""The build-parity gate: a CMake-built bcir-cc behaves exactly as the shell gate's recipe builds it.

The C twin is judged by its bytes (docs/PARITY.md): tools/c/check_runtime.sh builds bcir-cc in
one compiler command, and until every gate section runs against the CMake targets (BUILD-2),
this proves the two builds are the same compiler over the same sources with the same outcome.
The gate's recipe is compiled here, under the same compiler the CMake build used, and both
binaries run every mode over every fixture of the corpus; stdout, stderr, the exit status and
the --emit-pack bytes must agree on every row.

    build_parity.py --bcir-cc build/cmake/runtime/c/bcir-cc --cc gcc

Exit 0 when every row agrees, 1 when a row differs (the rows are printed), 2 when the gate
cannot run -- no compiler, the recipe does not build, no fixtures (an empty corpus is not a
pass; docs/security/laws.md L2) -- so a skipped gate never reads as a passing one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "runtime" / "c"
# tools/c/check_runtime.sh's bcir-cc recipe: the sources in its order, -std=c23 -O2 -Wall -Wextra,
# with its fallback to -std=c11 -O2 for a compiler that predates the c23 spelling.
RECIPE_SOURCES = (
    "bcir_cc.c",
    "bcir_cpp.c",
    "bcir_cfront.c",
    "bcir_verify.c",
    "bcir_runtime.c",
    "bcir_plan.c",
    "bcir_hydrate.c",
)
RECIPE_FLAGS = (("-std=c23", "-O2", "-Wall", "-Wextra"), ("-std=c11", "-O2"))
# Every bcir-cc mode whose output the oracle's harnesses compare; --emit-pack writes bytes to -o.
MODES: tuple[tuple[str, ...], ...] = (
    (),
    ("--emit-c",),
    ("--emit-claimgraph",),
    ("--emit-effects",),
    ("--emit-escape",),
    ("--emit-link-flags",),
    ("--fallback",),
    ("--emit-pack",),
)


def build_recipe(cc: str, out: Path, log: list[str]) -> bool:
    sources = [str(C_DIR / s) for s in RECIPE_SOURCES]
    for flags in RECIPE_FLAGS:
        cmd = [cc, *flags, "-I", str(C_DIR), *sources, "-o", str(out)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.append(f"recipe: {cc} {' '.join(flags)}: {exc}")
            continue
        if proc.returncode == 0:
            log.append(f"recipe: {cc} {' '.join(flags)} over {len(sources)} sources")
            return True
        log.append(
            f"recipe: {cc} {' '.join(flags)} failed:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}"
        )
    return False


def run_mode(
    exe: Path, mode: tuple[str, ...], fixture: Path, workdir: Path, timeout: float
) -> tuple:
    """(exit status, stdout, stderr, pack bytes) for one binary over one fixture in one mode."""
    pack = workdir / f"{exe.name}.pack"
    if pack.exists():
        pack.unlink()
    argv = [str(exe), *mode]
    if mode == ("--emit-pack",):
        argv += ["-o", str(pack)]
    argv.append(str(fixture))
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return ("timeout", b"", b"", b"")
    except OSError as exc:
        return (f"oserror: {exc}", b"", b"", b"")
    pack_bytes = pack.read_bytes() if pack.exists() else b""
    return (proc.returncode, proc.stdout, proc.stderr, pack_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bcir-cc", required=True, type=Path, help="the CMake-built bcir-cc")
    parser.add_argument(
        "--cc", default=os.environ.get("CC") or "cc", help="the compiler for the gate's recipe"
    )
    parser.add_argument(
        "--fixtures", type=Path, default=C_DIR, help="directory of cfront_*.c fixtures"
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N fixtures (0 = all)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per bcir-cc run")
    args = parser.parse_args(argv)

    if not args.bcir_cc.is_file() or not os.access(args.bcir_cc, os.X_OK):
        print(
            f"build-parity: UNUSABLE: {args.bcir_cc} is not an executable (build the bcir-cc target first)"
        )
        return 2
    cc = shutil.which(args.cc) or (args.cc if os.access(args.cc, os.X_OK) else None)
    if cc is None:
        print(f"build-parity: UNUSABLE: no compiler {args.cc!r} on PATH")
        return 2
    fixtures = sorted(args.fixtures.glob("cfront_*.c"))
    if args.limit > 0:
        fixtures = fixtures[: args.limit]
    if not fixtures:
        print(
            f"build-parity: INVALID: no cfront_*.c fixtures under {args.fixtures} (an empty corpus proves nothing)"
        )
        return 2

    log: list[str] = []
    differ: list[str] = []
    rows = 0
    with tempfile.TemporaryDirectory(prefix="bcir-build-parity-") as tmp:
        workdir = Path(tmp)
        recipe = workdir / "bcir-cc-recipe"
        if not build_recipe(cc, recipe, log):
            print("build-parity: UNUSABLE: the gate's recipe did not build")
            print("\n".join(log))
            return 2
        for fixture in fixtures:
            for mode in MODES:
                rows += 1
                ours = run_mode(args.bcir_cc.resolve(), mode, fixture, workdir, args.timeout)
                theirs = run_mode(recipe, mode, fixture, workdir, args.timeout)
                if ours != theirs:
                    what = [
                        name
                        for name, a, b in zip(("status", "stdout", "stderr", "pack"), ours, theirs)
                        if a != b
                    ]
                    differ.append(
                        f"{fixture.name} {' '.join(mode) or '(summary)'}: {', '.join(what)} differ"
                        f" (cmake status {ours[0]}, recipe status {theirs[0]})"
                    )
    print("\n".join(log))
    print(
        f"build-parity: {len(fixtures)} fixtures x {len(MODES)} modes = {rows} rows, {len(differ)} differ"
        f" (cmake {args.bcir_cc} vs the gate's recipe under {cc})"
    )
    if differ:
        for line in differ[:40]:
            print(f"  DIFFER {line}")
        if len(differ) > 40:
            print(f"  ... {len(differ) - 40} more")
        return 1
    print("build-parity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
