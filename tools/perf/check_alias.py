#!/usr/bin/env python3
"""The G9 rows as a gate: the declared alias facts carried the rest of the way to LLVM, held fact by
fact to the claim's declaration on every emitter of the elementwise kernel.

Measures every G9 row with `bcir.tests.alias_fixtures.measure` -- the one function the tests and the
harness (`tools/perf/gemplus_baseline.py --group alias`) share -- and prints one finding per row
that is not zero, in the form `tools/testing/red_sweep.py` reads:

    - alias.scope.mismatch: 3

The last line is the summary (`rows: alias.noalias.mismatch=0 ...`). With `--llvm` it adds the rows
LLVM itself judges (its alias analysis, clang's IR for the C kernel, every self-check runner the
host has). Every exit is a verdict (L1):

    0  every row is zero
    1  a row fired, the grader raised, or the rows measured are not the declared set
    2  --require-llvm and no coherent LLVM toolset (clang, llvm-link, opt): the rows LLVM judges
       were not measured, which in the job that installed LLVM is a failure, not a pass (L2)

Without `--require-llvm` a missing toolset is reported and the text rows still gate.
`tools/testing/faults/alias.json` runs this once per injected defect: the standing evidence that
each of these rows can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from bcir.tests import alias_fixtures as af  # noqa: E402

EXIT_PASS = 0
EXIT_FIRED = 1
EXIT_UNAVAILABLE = 2


def grade(stride: int, llvm: bool, require_llvm: bool) -> int:
    try:
        rows = af.measure(stride)
        judged = af.measure_llvm() if llvm else None
    except Exception as exc:  # noqa: BLE001 -- a grader that raised decided nothing (L1)
        print(f"  - grader: {type(exc).__name__}: {exc}")
        return EXIT_FIRED
    status = EXIT_PASS
    declared = set(af.ROWS)
    if llvm and judged is None:
        print("  - llvm: no coherent LLVM toolset (clang, llvm-link, opt); its rows NOT-MEASURED")
        if require_llvm:
            status = EXIT_UNAVAILABLE
    elif judged is not None:
        rows.update(judged)
        declared |= set(af.LLVM_ROWS)
    if set(rows) != declared:
        print(f"  - rows: measured {sorted(rows)}, declared {sorted(declared)}")
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
        "--stride", type=int, default=1, help="grade every N-th lawful case (default: all of them)"
    )
    parser.add_argument("--llvm", action="store_true", help="add the rows LLVM itself judges")
    parser.add_argument(
        "--require-llvm",
        action="store_true",
        help="exit 2 when the LLVM toolset is missing (implies --llvm)",
    )
    args = parser.parse_args(argv)
    if args.stride < 1:
        print(f"  - stride: {args.stride} grades nothing")  # a vacuous run is not a pass (L2)
        return EXIT_FIRED
    return grade(args.stride, args.llvm or args.require_llvm, args.require_llvm)


if __name__ == "__main__":
    raise SystemExit(main())
