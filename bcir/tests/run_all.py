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
    "bcir.tests.test_fusion",
    "bcir.tests.test_calibrator",
    "bcir.tests.test_schedule",
    "bcir.tests.test_microbench",
    "bcir.tests.test_calibloop",
    "bcir.tests.test_bayescal",
    "bcir.tests.test_moegate",
    "bcir.tests.test_accel",
    "bcir.tests.test_provenance",
    "bcir.tests.test_egraph",
    "bcir.tests.test_memory",
    "bcir.tests.test_twotruth",
    "bcir.tests.test_mapping",
    "bcir.tests.test_operad",
    "bcir.tests.test_hot_cold",
    "bcir.tests.test_throttle",
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
    "bcir.tests.test_c_kernel",
    "bcir.tests.test_api",
    "bcir.tests.test_bench",
    "bcir.tests.test_jit",
    "bcir.tests.test_wasm",
    "bcir.tests.test_c_runtime",
    "bcir.tests.test_memory_model",
    "bcir.tests.test_codegen",
    "bcir.tests.test_async",
    "bcir.tests.test_perf",
    "bcir.tests.test_allocator",
    "bcir.tests.test_cim",
    "bcir.tests.test_resident_egraph",
    "bcir.tests.test_specialist",
    "bcir.tests.test_sensing",
    "bcir.tests.test_ring",
    "bcir.tests.test_fuzzy_routing",
    "bcir.tests.test_dvfs",
    "bcir.tests.test_silicon",
    "bcir.tests.test_persistent_oracles",
    "bcir.tests.test_precision",
    "bcir.tests.test_differential",
    "bcir.tests.test_target_matrix",
    "bcir.tests.test_c23_kernels",
    "bcir.tests.test_q8_embed",
    "bcir.tests.test_etl_binrec",
    "bcir.tests.test_native_object_gate",
    "bcir.tests.test_c_executor",
    "bcir.tests.test_c_encoder",
    "bcir.tests.test_verify_cost",
    "bcir.tests.test_precision_lowering",
    "bcir.tests.test_bundle",
    "bcir.tests.test_proof",
    "bcir.tests.test_verify_differential",
    "bcir.tests.test_smart_laws",
    "bcir.tests.test_fuzz",
    "bcir.tests.test_measured_silicon",
    "bcir.tests.test_clang_compare",
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
