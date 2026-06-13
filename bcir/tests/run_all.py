"""Dependency-free test runner (works without pytest).

Discovers ``test_*`` callables in the sibling ``test_*`` modules, runs them, and
reports PASS/FAIL. Usable two ways:

    python -m bcir.tests.run_all          # from the repo root
    python -m pytest bcir/tests           # if pytest is installed (same tests)
"""

from __future__ import annotations

import importlib
import sys
import traceback

_MODULES = [
    "bcir.tests.test_kbcir",
    "bcir.tests.test_rcsp",
    "bcir.tests.test_overlap",
    "bcir.tests.test_schedule",
    "bcir.tests.test_microbench",
    "bcir.tests.test_bayescal",
    "bcir.tests.test_moegate",
    "bcir.tests.test_accel",
    "bcir.tests.test_portfolio",
    "bcir.tests.test_regret",
    "bcir.tests.test_softdp",
    "bcir.tests.test_targets",
    "bcir.tests.test_verify",
    "bcir.tests.test_etl",
    "bcir.tests.test_gem",
    "bcir.tests.test_concurrency",
    "bcir.tests.test_frontends",
    "bcir.tests.test_telemetry",
    "bcir.tests.test_abi",
    "bcir.tests.test_stackify",
    "bcir.tests.test_lowering",
    "bcir.tests.test_jit",
    "bcir.tests.test_wasm",
    "bcir.tests.test_c_runtime",
    "bcir.tests.test_memory_model",
    "bcir.tests.test_codegen",
    "bcir.tests.test_async",
]


def main() -> int:
    passed = 0
    failed = 0
    for modname in _MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:  # noqa: BLE001 - report every failure
                failed += 1
                print(f"FAIL {modname}.{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {modname}.{name}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
