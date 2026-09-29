#!/usr/bin/env python3
"""The G8 rows as a gate: data movement as a first-class transformation, held to the exact optimum
of the joint objective, to the movement laws, and to one wire read verdict for verdict by every
rail that reads it.

Measures every G8 row with `bcir.tests.movement_fixtures.measure` -- the one function the tests
and the harness (`tools/perf/gemplus_baseline.py --group movement`) share -- and prints one finding
per row off its bound, in the form `tools/testing/red_sweep.py` reads:

    - movement.excess: 13312

Every gated row must be 0: no implicit cross-tier access left in the example programs, no excess
over the exact optimum of any fixture (TMSAO-1), no law variant accepted or refused by a law other
than its own, no drift in a plan that moves nothing, no plan that does not round-trip, no planned
plan a law refuses, no malformed plan an ASN.1 decoder accepts, and no verdict on which the C twin
and the Python rail differ. The last line is the summary (`rows: movement.excess=0 ...`). Every exit
is a verdict (L1):

    0  every gated row is at its bound
    1  a row fired, the grader raised, or the rows measured are not the declared set
    2  --require-cc and no C compiler builds the plan harness: the parity row was not measured,
       which in the job that installed a compiler is a failure, not a pass (L2)

Without `--require-cc` a missing compiler is reported and the interpreter rows still gate.
`tools/testing/faults/movement.json` runs this once per injected defect: the standing evidence that
each of these rows can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import movement_fixtures as mf  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2

#: The rows gated, and the bound each must sit at.
BOUNDS = {key: 0 for key in mf.ROWS}


def grade(require_cc: bool) -> int:
    try:
        rows = mf.measure()
    except Exception as exc:  # noqa: BLE001 -- a grader that raised decided nothing (L1)
        print(f"  - grader: {type(exc).__name__}: {exc}")
        return EXIT_FIRED
    status = EXIT_PASS
    declared = set(mf.ROWS)
    missing = declared - set(rows)
    if missing - set(mf.CC_ROWS) or set(rows) - declared:
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(declared)}")
        status = EXIT_FIRED
    if missing & set(mf.CC_ROWS):
        print(f"  - cc: no C compiler builds the plan harness; {sorted(missing)} NOT-MEASURED")
        if require_cc and status == EXIT_PASS:
            status = EXIT_UNAVAILABLE
    for key, value in rows.items():
        bound = BOUNDS.get(key)
        if bound is not None and value != bound:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    print("rows: " + " ".join(f"{key}={int(value)}" for key, value in rows.items()))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-cc",
        action="store_true",
        help="exit 2 when no C compiler builds the plan harness (its row would go unmeasured)",
    )
    args = parser.parse_args(argv)
    return grade(args.require_cc)


if __name__ == "__main__":
    sys.exit(main())
