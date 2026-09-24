#!/usr/bin/env python3
"""The G16 rows as a gate: the data-plane hand-off on the oracle, the C twin and the C++ seam.

Builds `runtime/c/test_handoff.c` and `runtime/cpp/test_handoff.cpp` (or grades the harnesses named
with `--exe` / `--cpp-exe`; `--no-c` / `--no-cpp` leave a rail out), measures every G16 row with
`bcir.tests.handoff_fixtures.measure` -- the one function the tests, the harness
(`tools/perf/gemplus_baseline.py --group handoff`), `tools/c/check_runtime.sh` and
`tools/cpp/check_handoff.sh` share -- and prints one finding per row that is not zero, in the form
`tools/testing/red_sweep.py` reads:

    - handoff.stale.admitted: 1

The last line is the summary (`rows: stale.admitted=0 ...`). Every exit is a verdict (L1):

    0  every row is zero on every rail graded
    1  a row fired, a harness source failed to build, or the rows measured are not the declared set
    2  UNAVAILABLE: a rail asked for has no compiler, or no runtime tree in this installation --
       never a pass

`tools/testing/faults/handoff.json` runs this once per injected defect: the standing evidence that
each of these rows can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import handoff_fixtures as hf  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2


def grade(exe: str | None, cpp_exe: str | None, tmp: str) -> int:
    rows = hf.measure(exe, cpp_exe, tmp)
    status = EXIT_PASS
    if set(rows) != set(hf.ROWS):
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(hf.ROWS)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    rails = " + ".join(["oracle"] + (["C"] if exe else []) + (["C++"] if cpp_exe else []))
    summary = " ".join(
        f"{key.removeprefix('handoff.')}={int(value)}" for key, value in rows.items()
    )
    print(f"rows ({rails}): {summary}")
    return status


def _harness(given: str | None, skip: bool, build, tmp: str, what: str):
    """(path or None, exit code or None): a named harness must exist; a built one must build."""
    if skip:
        return None, None
    if given is not None:
        if not os.path.isfile(given):
            print(f"UNAVAILABLE: no {what} harness at {given}")
            return None, EXIT_UNAVAILABLE
        return given, None
    try:
        exe = build(tmp)
    except RuntimeError as exc:  # a source present but broken: the build is a finding
        print(f"  - build: {exc}")
        return None, EXIT_FIRED
    if exe is None:
        print(f"UNAVAILABLE: no compiler for the {what} harness, or no runtime tree here")
        return None, EXIT_UNAVAILABLE
    return exe, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", help="grade this built runtime/c/test_handoff.c harness")
    parser.add_argument("--cpp-exe", help="grade this built runtime/cpp/test_handoff.cpp harness")
    parser.add_argument("--no-c", action="store_true", help="leave the C twin out")
    parser.add_argument("--no-cpp", action="store_true", help="leave the C++ seam out")
    parser.add_argument("--tmp", help="the scratch directory (default: a fresh temporary one)")
    args = parser.parse_args(argv)
    if args.no_c and args.no_cpp:
        print("  - rails: --no-c and --no-cpp leave only the oracle -- nothing native graded")
        return EXIT_FIRED
    with tempfile.TemporaryDirectory(prefix="bcir-handoff-") as fresh:
        tmp = args.tmp or fresh
        exe, code = _harness(args.exe, args.no_c, hf.build_harness, tmp, "C")
        if code is not None:
            return code
        cpp, code = _harness(args.cpp_exe, args.no_cpp, hf.build_cpp_harness, tmp, "C++")
        if code is not None:
            return code
        return grade(exe, cpp, tmp)


if __name__ == "__main__":
    sys.exit(main())
