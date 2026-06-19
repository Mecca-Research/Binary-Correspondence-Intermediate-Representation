"""Dependency-free test runner (works without pytest).

Discovers ``test_*`` callables in the sibling ``test_*`` modules, runs them, and
reports PASS/FAIL. Usable two ways:

    python -m bcir.tests.run_all                       # quick chain (default)
    python -m bcir.tests.run_all --tier c-runtime      # + the C byte-identity tier
    python -m bcir.tests.run_all --tier silicon-degrade # + measured benchmarks (degrade-ok)
    python -m bcir.tests.run_all --tier thorough        # everything (full toolchain + campaigns)
    python -m pytest bcir/tests                         # if pytest is installed (same tests)

Named tiers are an *escalating capability ladder* — each adds what the previous
exposes (``quick`` ⊂ ``c-runtime`` ⊂ ``silicon-degrade`` ⊂ ``thorough``). They work
with the suite's existing self-gating: a tier decides which host tools are *visible*
(via a `shutil.which` gate) and whether the full ``BCIR_THOROUGH`` campaigns run; the
individual C/native/silicon tests already early-return through their own `which(...)`
guards or degrade honestly when a capability is absent, so no per-tier module list has
to be hand-maintained (it would drift). All tests always run; the tier only changes
which of them do real work vs. self-skip.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import traceback

# --- named test tiers (the capability ladder) ------------------------------------
# Measured: the ~130 compile / native-execute tests (the C-ABI byte-identity + measured-
# silicon tier) dominate wall-time (~77%); the pure-Python oracle / law / parity coverage
# is ~6%. So by default run_all is the FAST quick chain (~5-6s): hide the C/LLVM toolchain
# so those tests early-return through their existing which(...) guards. Coverage preserved
# in the quick chain: every law R1-R18, all six targets, the StreamPack/ABI *logic*, and the
# budget/RCSP correctness property are pure-Python and still run; only the C *byte-identity*
# and native-*execute* checks defer to a heavier tier (CI runs `thorough`).
_C_COMPILER = {"clang", "clang++", "cc", "gcc", "g++", "ld.lld", "lld", "ar",
               "objdump", "llvm-objdump"}                      # build + inspect the C runtime
_LLVM_JIT_WASM = {"lli", "opt", "llc", "llvm-as", "llvm-link", "wasm-ld", "node"}
_ALL_TOOLCHAIN = _C_COMPILER | _LLVM_JIT_WASM

# tier -> (host tools made visible to which(...), run the full BCIR_THOROUGH campaigns?)
TIERS: dict[str, dict[str, object]] = {
    # pure-Python oracle/law/parity + honest-degrade silicon; instant, runs anywhere.
    "quick":           {"visible": frozenset(),                    "thorough": False},
    # + a C compiler: the freestanding-runtime / StreamPack-ABI byte-identity / C-kernel tier.
    "c-runtime":       {"visible": frozenset(_C_COMPILER),         "thorough": False},
    # + the measured benchmarks compiled & run in *degrade* mode (assert correctness + valid
    #   measurement, never a faked speedup); the tier you run on real silicon to exercise the
    #   perf syscall / RAPL / cpufreq path, tolerant of a degraded rig (shared CI runner).
    "silicon-degrade": {"visible": frozenset(_C_COMPILER),         "thorough": False},
    # everything: full toolchain (LLVM IR / JIT / WASM / native-execute) + the large campaigns.
    "thorough":        {"visible": frozenset(_ALL_TOOLCHAIN),      "thorough": True},
}
_DEFAULT_TIER = "quick"


def resolve_tier(argv: list[str] | None = None) -> str:
    """Pick the tier: ``--tier X`` > ``$BCIR_TIER`` > (``$BCIR_THOROUGH`` -> thorough) > quick.

    The ``BCIR_THOROUGH`` fallback keeps existing CI (which exports it and calls run_all)
    mapping to the full tier with no workflow change.
    """
    argv = sys.argv[1:] if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--tier" and i + 1 < len(argv):
            tier = argv[i + 1]
            break
        if a.startswith("--tier="):
            tier = a.split("=", 1)[1]
            break
    else:
        tier = os.environ.get("BCIR_TIER") or ("thorough" if os.environ.get("BCIR_THOROUGH")
                                               else _DEFAULT_TIER)
    if tier not in TIERS:
        sys.stderr.write(f"[run_all] unknown tier {tier!r}; choose from {', '.join(TIERS)}\n")
        raise SystemExit(2)
    return tier


def _apply_tier(tier: str) -> None:
    """Install the tier's capability gate *before* any test/bcir module is imported, so both
    `shutil.which(...)` and `from shutil import which` bindings resolve to the gated version and
    module-level `BCIR_THOROUGH` reads (campaign sizes) see the right value."""
    spec = TIERS[tier]
    os.environ["BCIR_TIER"] = tier
    if spec["thorough"]:
        os.environ["BCIR_THOROUGH"] = "1"
    visible: frozenset[str] = spec["visible"]  # type: ignore[assignment]
    hidden = _ALL_TOOLCHAIN - visible
    if not hidden:
        return                                                  # thorough: nothing gated
    _orig_which = shutil.which

    def _gated_which(cmd, *args, **kwargs):
        return None if os.path.basename(str(cmd)) in hidden \
            else _orig_which(cmd, *args, **kwargs)

    shutil.which = _gated_which


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
    "bcir.tests.test_channels",
    "bcir.tests.test_channel_plugin",
    "bcir.tests.test_verify",
    "bcir.tests.test_etl",
    "bcir.tests.test_gem",
    "bcir.tests.test_concurrency",
    "bcir.tests.test_frontends",
    "bcir.tests.test_cfront",
    "bcir.tests.test_cfront_diagnostics",
    "bcir.tests.test_cfront_abi",
    "bcir.tests.test_cfront_effects",
    "bcir.tests.test_cfront_fallback",
    "bcir.tests.test_cfront_ipo",
    "bcir.tests.test_cfront_fuzz",
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
    "bcir.tests.test_c_cfront",
    "bcir.tests.test_c_channel",
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
    "bcir.tests.test_compose",
    "bcir.tests.test_compose_differential",
    "bcir.tests.test_proof",
    "bcir.tests.test_verify_differential",
    "bcir.tests.test_smart_laws",
    "bcir.tests.test_fuzz",
    "bcir.tests.test_measured_silicon",
    "bcir.tests.test_silicon_runbook",
    "bcir.tests.test_clang_compare",
    "bcir.tests.test_tiers",
    "bcir.tests.test_perf_budget",
    "bcir.tests.test_import_quarantine",
]


_TIER_BLURB = {
    "quick": "pure-Python oracle/law/parity + honest-degrade silicon (toolchain hidden)",
    "c-runtime": "+ C compiler: freestanding-runtime / StreamPack-ABI byte-identity / C kernels",
    "silicon-degrade": "+ measured benchmarks in degrade mode (correctness + valid measurement, "
                       "no faked speedup)",
    "thorough": "everything: full toolchain (IR/JIT/WASM/native) + the large campaigns",
}


def main() -> int:
    if "--list-tiers" in sys.argv:
        print("tiers (escalating capability ladder):")
        for name in TIERS:
            print(f"  {name:<16} {_TIER_BLURB[name]}")
        return 0
    tier = resolve_tier()
    _apply_tier(tier)                       # gate the toolchain BEFORE importing test modules
    print(f"[run_all] tier={tier} — {_TIER_BLURB[tier]}\n")
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
