#!/usr/bin/env python3
"""The G18 rows as a gate: the K_BCIR -> StreamPack chain advanced by declared deltas, held to the
chain run from scratch.

Measures every G18 exact row with `bcir.tests.delta_fixtures.measure` -- the one function the tests
and the harness (`tools/perf/gemplus_baseline.py --group delta`) share -- and prints one finding
per row that is not zero, in the form `tools/testing/red_sweep.py` reads:

    - planner.delta.parity: 1

The last line is the summary (`rows: planner.delta.parity=0 ...`). Every exit is a verdict (L1):

    0  every row is zero
    1  a row fired, the grader raised, or the rows measured are not the declared set

The rows need nothing but the interpreter, so there is no UNAVAILABLE exit.
`tools/testing/faults/delta.json` runs this once per injected defect: the standing evidence that
each of these rows can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import delta_fixtures as df  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1


def grade(stride: int) -> int:
    try:
        rows = df.measure(stride)
    except Exception as exc:  # noqa: BLE001 -- a grader that raised decided nothing (L1)
        print(f"  - grader: {type(exc).__name__}: {exc}")
        return EXIT_FIRED
    status = EXIT_PASS
    if set(rows) != set(df.ROWS):
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(df.ROWS)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    print("rows: " + " ".join(f"{key}={int(value)}" for key, value in rows.items()))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stride", type=int, default=1, help="grade every N-th case (default: all of them)"
    )
    args = parser.parse_args(argv)
    if args.stride < 1:
        print(f"  - stride: {args.stride} grades nothing")  # a vacuous run is not a pass (L2)
        return EXIT_FIRED
    return grade(args.stride)


if __name__ == "__main__":
    raise SystemExit(main())
