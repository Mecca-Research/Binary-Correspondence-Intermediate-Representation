#!/usr/bin/env python3
"""The SP-ENC rows as a gate: the StreamPack encoder's compiled record layouts, held byte for byte
and refusal for refusal to the encoder they replace, and to the calls they save.

Measures `streampack.encode.parity` with `bcir.tests.encode_fixtures.measure` -- the one function
the tests and the harness (`tools/perf/gemplus_baseline.py --group encode`) share -- and
`streampack.encode.calls.over`: 1 when one encode of the audit pack at scale 2 makes a sixth or
more of the reference encoder's calls (`encode_fixtures.CALLS_FRACTION`), else 0. Prints one
finding per row that is not zero, in the form `tools/testing/red_sweep.py` reads:

    - streampack.encode.parity: 3

The last line is the summary (`rows: streampack.encode.parity=0 ...`). Every exit is a verdict
(L1):

    0  every row is zero
    1  a row fired, the grader raised, or the rows measured are not the declared set

`tools/testing/faults/encode.json` runs this once per injected defect: the standing evidence that
each row can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import encode_fixtures as ef  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
ROWS = (*ef.ROWS, "streampack.encode.calls.over")


def _calls_row() -> dict[str, float]:
    return {"streampack.encode.calls.over": float(ef.calls_over())}


def grade() -> int:
    # Each row is graded on its own: a grader that raises decided nothing (L1), and it must not
    # hide the finding another row made -- `calls_over` refuses to time two encoders that disagree,
    # and the parity row is the one that says so.
    status = EXIT_PASS
    rows: dict[str, float] = {}
    for measure in (ef.measure, _calls_row):
        try:
            rows.update(measure())
        except Exception as exc:  # noqa: BLE001
            print(f"  - grader: {type(exc).__name__}: {exc}")
            status = EXIT_FIRED
    if set(rows) != set(ROWS):
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(ROWS)}")
        status = EXIT_FIRED
    for key, value in rows.items():
        if value:
            print(f"  - {key}: {int(value)}")
            status = EXIT_FIRED
    print("rows: " + " ".join(f"{key}={int(value)}" for key, value in rows.items()))
    return status


def main(argv: list[str] | None = None) -> int:
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    return grade()


if __name__ == "__main__":
    raise SystemExit(main())
