#!/usr/bin/env python3
"""The G15 rows as a gate: the live SPSC ring and the version-zero triple on both rails.

Builds `runtime/c/test_ring.c` (or grades a harness named with `--exe`), measures every G15 row
with `bcir.tests.ring_fixtures.measure` -- the one function the tests, the harness
(`tools/perf/gemplus_baseline.py --group ring`) and `tools/c/check_runtime.sh` share -- and prints
one finding per row that is not zero, in the form `tools/testing/red_sweep.py` reads:

    - ring.torn.delivered: 1

The last line is the summary (`rows: abi.mismatches=0 ...`). Every exit is a verdict (L1):

    0  every row is zero
    1  a row fired, or the rows measured are not the declared set
    2  UNAVAILABLE: no C compiler, or no runtime/c in this installation -- never a pass

`tools/testing/faults/ring.json` runs this once per injected defect: the standing evidence that
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

from bcir.tests import ring_fixtures as rf  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2


def grade(exe: str, tmp: str) -> int:
    rows = rf.measure(exe, tmp)
    expected = set(rf.ROWS) - (set() if os.name == "posix" else {"ring.concurrent.violations"})
    status = EXIT_PASS
    if set(rows) != expected:
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(expected)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    summary = " ".join(f"{key.removeprefix('ring.')}={int(value)}" for key, value in rows.items())
    print(f"rows: {summary}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", help="grade this built runtime/c/test_ring.c harness")
    parser.add_argument("--tmp", help="the scratch directory (default: a fresh temporary one)")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="bcir-ring-") as fresh:
        tmp = args.tmp or fresh
        exe = args.exe
        if exe is None:
            try:
                exe = rf.build_harness(tmp)
            except RuntimeError as exc:  # a source present but broken: the build is a finding
                print(f"  - build: {exc}")
                return EXIT_FIRED
            if exe is None:
                print("UNAVAILABLE: no C compiler, or no runtime/c in this installation")
                return EXIT_UNAVAILABLE
        elif not os.path.isfile(exe):
            print(f"UNAVAILABLE: no harness at {exe}")
            return EXIT_UNAVAILABLE
        return grade(exe, tmp)


if __name__ == "__main__":
    sys.exit(main())
