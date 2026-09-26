#!/usr/bin/env python3
"""The volatile rows as a gate: every place a C program puts `volatile`, against every access form,
at every element width, lowered by both cfront rails and judged by Clang.

Measures every row with `bcir.tests.volatile_fixtures.measure` -- the one function the tests and the
harness (`tools/perf/gemplus_baseline.py --group volatile`) share -- and prints one finding per row
off its bound, in the form `tools/testing/red_sweep.py` reads:

    - volatile.emit.mismatch: 3

Every row must be 0: a form a rail refuses, whose emitted C does not perform the original's volatile
accesses in order at their types, that returns or leaves different bytes, whose two claim graphs
differ, or whose volatile access a rail's effect report does not treat as a device access. The last
line is the summary (`rows: volatile.refused=0 ...`). Every exit is a verdict (L1):

    0  every row is at its bound
    1  a row fired, the grader raised, or the rows measured are not the declared set
    2  --require-cc and no Clang (or no runtime/c to build the twin from): nothing was measured,
       which in the job that installed Clang is a failure, not a pass (L2)

Without `--require-cc` a host that cannot grade is reported and exits 0 with no rows.
`tools/testing/faults/volatile.json` runs this once per injected defect: the standing evidence that
each row can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import volatile_fixtures as vf  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2

#: The rows gated and the bound each must sit at: every one is a count of failing forms.
BOUNDS = {row: 0 for row in vf.ROWS}


def grade(require_cc: bool, bcir_cc: str | None) -> int:
    if not vf.available(bcir_cc):
        print("  - cc: no Clang, or no runtime/c to build the twin; the volatile rows NOT-MEASURED")
        print("rows: -")
        return EXIT_UNAVAILABLE if require_cc else EXIT_PASS
    try:
        rows = vf.measure(bcir_cc)
    except Exception as exc:  # noqa: BLE001 -- a grader that raised decided nothing (L1)
        print(f"  - grader: {type(exc).__name__}: {exc}")
        return EXIT_FIRED
    status = EXIT_PASS
    if set(rows) != set(vf.ROWS):
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(vf.ROWS)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value > BOUNDS.get(key, 0):
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    print("rows: " + " ".join(f"{key}={int(value)}" for key, value in rows.items()))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-cc",
        action="store_true",
        help="exit 2 when no Clang (or no twin) can grade the forms (every row would go unmeasured)",
    )
    parser.add_argument(
        "--bcir-cc",
        help="a prebuilt bcir-cc to judge (default: build the twin from runtime/c)",
    )
    args = parser.parse_args(argv)
    return grade(args.require_cc, args.bcir_cc)


if __name__ == "__main__":
    sys.exit(main())
