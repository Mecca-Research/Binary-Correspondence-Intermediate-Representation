#!/usr/bin/env python3
"""Named test functions as a gate: the check a fault table points at when its law lives in tests.

`tools/testing/red_sweep.py` reads a gate's findings as lines of the form

    - <check>: <detail>

and most gates here measure rows (`tools/c/check_ring.py`, `tools/perf/check_volatile.py`). The C
frontends' laws are held by test functions instead -- a fixture run against the original on both
rails' emits, a refusal asserted per rail -- so this runs the `module:function` tests it is given,
in order and in one process, and prints one finding per failing test, named by the test:

    python3 tools/testing/check_tests.py --require bcir.tests.test_c_cfront:_CC \\
        bcir.tests.test_c_cfront:test_brace_elision_runs_as_the_original_on_both_rails

Every exit is a verdict (L1):

    0  every named test passed
    1  at least one finding: a test raised, or tried to end the process
    2  the gate could not run what it was asked to: a name that resolves to no test function, or a
       `--require`d capability that is absent -- the tests return early without it, so their pass
       would examine nothing (L2) -- never a pass

`tools/testing/faults/cfront-*.json` run this once per injected defect: the standing evidence
that each of those tests can fire (docs/security/laws.md L2, L25).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(
    0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_UNAVAILABLE = 2


def _resolve(spec: str):
    """`module:name` -> the named attribute. A malformed or unknown name is a `LookupError`."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise LookupError(f"{spec!r} is not MODULE:NAME")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise LookupError(f"{spec!r}: {exc}") from exc
    if not hasattr(module, attr):
        raise LookupError(f"{spec!r}: {module_name} has no {attr!r}")
    return getattr(module, attr)


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else "").strip()[:300]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="MODULE:NAME",
        help="a capability the tests return early without; the gate is UNAVAILABLE unless it is truthy",
    )
    parser.add_argument("tests", nargs="+", metavar="MODULE:FUNCTION")
    args = parser.parse_args(argv)

    try:
        required = [(spec, _resolve(spec)) for spec in args.require]
        tests = [(spec, _resolve(spec)) for spec in args.tests]
    except LookupError as exc:
        print(f"  - unresolved: {exc}")
        return EXIT_UNAVAILABLE
    not_tests = [spec for spec, fn in tests if not callable(fn)]
    if not_tests:
        print(f"  - unresolved: {', '.join(not_tests)} is not a test function")
        return EXIT_UNAVAILABLE
    absent = [spec for spec, value in required if not value]
    if absent:
        print(
            f"  - unavailable: {', '.join(absent)} is not set, so the tests would return "
            "without examining anything"
        )
        return EXIT_UNAVAILABLE

    findings = 0
    for spec, fn in tests:
        name = spec.partition(":")[2]
        try:
            fn()
        except (Exception, SystemExit) as exc:  # a test that ends the process has not passed
            findings += 1
            print(f"  - {name}: {type(exc).__name__}: {_first_line(exc)}", flush=True)
        else:
            print(f"  ok {name}", flush=True)
    print(f"tests: {len(tests)} run, {findings} finding(s)")
    return EXIT_FINDINGS if findings else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
