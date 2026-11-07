#!/usr/bin/env python3
"""The build-parity gate: the CMake-built bcir-cc behaves exactly as BCIR Make builds it.

The C twin is judged by its bytes (docs/PARITY.md). BCIR Make builds bcir-cc from
runtime/manifest.json (tools/build/bcirfile.py) -- the build tools/c/check_runtime.sh's sections
run over -- and the CMake project builds it from the same manifest; this proves the two builds are
the same compiler over the same sources with the same outcome. Both binaries run every mode over
every fixture of the corpus; stdout, stderr, the exit status and the --emit-pack bytes must agree
on every row. Until BUILD-8 the comparison was with the gate's own one-command recipe; those
compile lines are retired, and BCIR Make's build is the one the gate runs. Each fixture's rows
run in their own working directory, so `--workers` fixtures run at once; the rows are reported in
the fixtures' order.

    build_parity.py --bcir-cc build/cmake/runtime/c/bcir-cc --cc gcc [--make-dir DIR]

Exit 0 when every row agrees, 1 when a row differs (the rows are printed), 2 when the gate
cannot run -- no compiler, a BCIR Make build that fails, no fixtures (an empty corpus is not a
pass; docs/security/laws.md L2) -- so a skipped gate never reads as a passing one.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "runtime" / "c"
# BCIR Make's build of the rails, under the source tree (shared with the section-parity gate).
MAKE_DIR = "build/bcir-make-parity"
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


def _generator():
    spec = importlib.util.spec_from_file_location(
        "bcir_build_bcirfile", ROOT / "tools" / "build" / "bcirfile.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def compare_fixture(
    ours: Path, theirs: Path, fixture: Path, workdir: Path, timeout: float
) -> list[str]:
    """Every mode over one fixture under both binaries, in a working directory of its own (a run's
    --emit-pack writes there): the rows that differ, each named with the streams that differ."""
    differ: list[str] = []
    for mode in MODES:
        a = run_mode(ours, mode, fixture, workdir, timeout)
        b = run_mode(theirs, mode, fixture, workdir, timeout)
        if a != b:
            what = [
                name for name, x, y in zip(("status", "stdout", "stderr", "pack"), a, b) if x != y
            ]
            differ.append(
                f"{fixture.name} {' '.join(mode) or '(summary)'}: {', '.join(what)} differ"
                f" (CMake status {a[0]}, BCIR Make status {b[0]})"
            )
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bcir-cc", required=True, type=Path, help="the CMake-built bcir-cc")
    parser.add_argument(
        "--cc", default=os.environ.get("CC") or "cc", help="the C compiler BCIR Make builds with"
    )
    parser.add_argument("--cxx", help="the C++ compiler, as the section-parity gate passes it")
    parser.add_argument(
        "--make-dir",
        default=MAKE_DIR,
        help=f"where BCIR Make builds, relative to the source tree (default {MAKE_DIR})",
    )
    parser.add_argument(
        "--fixtures", type=Path, default=C_DIR, help="directory of cfront_*.c fixtures"
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N fixtures (0 = all)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per bcir-cc run")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="tasks at once: BCIR Make's, then the fixtures' rows (default 2, the local cap; CI's"
        " 4-vCPU runners pass 4)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        print(f"build-parity: UNUSABLE: --workers {args.workers} runs nothing")
        return 2

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

    generator = _generator()
    try:
        made = generator.build(cc, args.cxx, args.make_dir, workers=args.workers).get("bcir-cc")
    except generator.BuildError as exc:
        print(f"build-parity: UNUSABLE: BCIR Make did not build the rails: {exc}")
        return 2
    if made is None or not os.access(made, os.X_OK):
        print("build-parity: UNUSABLE: BCIR Make's build has no bcir-cc")
        return 2
    differ: list[str] = []
    rows = len(fixtures) * len(MODES)
    ours = args.bcir_cc.resolve()
    with tempfile.TemporaryDirectory(prefix="bcir-build-parity-") as tmp:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = []
            for i, fixture in enumerate(fixtures):
                workdir = Path(tmp) / str(i)
                workdir.mkdir()
                jobs.append(
                    pool.submit(compare_fixture, ours, made, fixture, workdir, args.timeout)
                )
            for job in jobs:
                differ.extend(job.result())
    print(
        f"build-parity: {len(fixtures)} fixtures x {len(MODES)} modes = {rows} rows, {len(differ)} differ"
        f" (CMake's {args.bcir_cc} vs BCIR Make's {made} under {cc})"
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
