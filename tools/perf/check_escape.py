#!/usr/bin/env python3
"""The G10 rows as a gate: escape analysis, indirect-call narrowing and the effect footprint of
the cfront frontend, held to the verdicts and targets the generated units have by construction,
to a dynamic witness of what does not commute, and to the C twin's reports byte for byte.

Measures every G10 row with `bcir.tests.escape_fixtures.measure` -- the one function the tests and
the harness (`tools/perf/gemplus_baseline.py --group escape`) share -- and prints one finding per
row off its bound, in the form `tools/testing/red_sweep.py` reads:

    - effects.commute.unsound: 3

Each gated row has a range. The mismatch, unsound and parity rows must be 0. The corpus counts
(`escape.unproved`, `icall.unknown`, `icall.unresolved`) count what is not yet proved or narrowed:
above the range is a lost proof or a lost narrowing, and below it is a claim no sound analysis can
make -- the floor of `icall.unknown` is the open world's (15 sites whose function pointer another
unit may pass), so a value under it proves a site was narrowed unsoundly. The last line is the
summary (`rows: escape.unproved=0 ...`). Every exit is a verdict (L1):

    0  every gated row is at its bound
    1  a row fired, the grader raised, or the rows measured are not the declared set
    2  --require-cc and no C compiler that builds the witness: the witness and parity rows were not
       measured, which in the job that installed a compiler is a failure, not a pass (L2)

Without `--require-cc` a missing compiler is reported and the interpreter rows still gate.
`tools/testing/faults/escape.json` runs this once per injected defect: the standing evidence that
each of these rows can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import escape_fixtures as ef  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2

#: The rows gated, and the range each must sit in: (the proved floor, the value this slice landed).
BOUNDS = {
    "escape.unproved": (0, 0),
    "icall.unknown": (15, 15),
    "icall.unresolved": (16, 18),
    "escape.verdict.mismatch": (0, 0),
    "icall.target.mismatch": (0, 0),
    "effects.commute.unsound": (0, 0),
    "effects.parity.mismatch": (0, 0),
    "escape.parity.mismatch": (0, 0),
}


def grade(require_cc: bool, bcir_cc: str | None) -> int:
    try:
        rows = ef.measure(cc=True, bcir_cc=bcir_cc)
    except Exception as exc:  # noqa: BLE001 -- a grader that raised decided nothing (L1)
        print(f"  - grader: {type(exc).__name__}: {exc}")
        return EXIT_FIRED
    status = EXIT_PASS
    declared = set(ef.ROWS)
    missing = declared - set(rows)
    if missing - set(ef.CC_ROWS) or set(rows) - declared:
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(declared)}")
        status = EXIT_FIRED
    if missing & set(ef.CC_ROWS):
        print(f"  - cc: no C compiler builds the witness; {sorted(missing)} NOT-MEASURED")
        if require_cc and status == EXIT_PASS:
            status = EXIT_UNAVAILABLE
    for key, value in rows.items():
        rule = BOUNDS.get(key)
        if rule is None:
            continue
        floor, landed = rule
        if value < floor:
            print(f"  - {key}: {int(value)} (under its proved floor of {floor}: unsound)")
            status = EXIT_FIRED
        elif value > landed:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    print("rows: " + " ".join(f"{key}={int(value)}" for key, value in rows.items()))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-cc",
        action="store_true",
        help="exit 2 when no C compiler builds the witness (its rows would go unmeasured)",
    )
    parser.add_argument(
        "--bcir-cc",
        help="a prebuilt bcir-cc to judge (default: build the twin from runtime/c)",
    )
    args = parser.parse_args(argv)
    return grade(args.require_cc, args.bcir_cc)


if __name__ == "__main__":
    sys.exit(main())
