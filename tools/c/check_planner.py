#!/usr/bin/env python3
"""The G17 rows as a gate: the compact planner against the pre-G17 planner and its native twin.

Builds `runtime/c/test_kplan.c` (or grades the harness named with `--exe`), measures every G17 row
with `bcir.tests.planner_fixtures.measure` -- the one function the tests, the harness
(`tools/perf/gemplus_baseline.py --group kplan`) and `tools/c/check_runtime.sh` share -- and prints
one finding per row that is not zero, in the form `tools/testing/red_sweep.py` reads:

    - planner.parity: 1

The last line is the summary (`rows: parity=0 malformed.accepted=0`). Every exit is a verdict (L1):

    0  every row is zero
    1  a row fired, the harness source failed to build, or the rows measured are not the declared
       set
    2  UNAVAILABLE: no C compiler, or no runtime tree in this installation -- never a pass

`tools/testing/faults/planner.json` runs this once per injected defect: the standing evidence that
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

from bcir.tests import planner_fixtures as pf  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2


def grade(exe: str, tmp: str) -> int:
    rows = pf.measure(exe, tmp)
    status = EXIT_PASS
    if set(rows) != set(pf.ROWS):
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(pf.ROWS)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    summary = " ".join(
        f"{key.removeprefix('planner.')}={int(value)}" for key, value in rows.items()
    )
    print(f"rows (oracle + reference + C): {summary}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", help="grade this built runtime/c/test_kplan.c harness")
    parser.add_argument("--tmp", help="the scratch directory (default: a fresh temporary one)")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="bcir-kplan-") as fresh:
        tmp = args.tmp or fresh
        if args.exe is not None:
            if not os.path.isfile(args.exe):
                print(f"UNAVAILABLE: no planner harness at {args.exe}")
                return EXIT_UNAVAILABLE
            return grade(args.exe, tmp)
        try:
            exe = pf.build_harness(tmp)
        except RuntimeError as exc:  # a source present but broken: the build is a finding
            print(f"  - build: {exc}")
            return EXIT_FIRED
        if exe is None:
            print("UNAVAILABLE: no compiler for the planner harness, or no runtime tree here")
            return EXIT_UNAVAILABLE
        return grade(exe, tmp)


if __name__ == "__main__":
    sys.exit(main())
