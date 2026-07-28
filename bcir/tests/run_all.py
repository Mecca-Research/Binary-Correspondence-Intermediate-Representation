"""Dependency-free test runner (works without pytest).

Discovers ``test_*`` callables in the sibling ``test_*`` modules, runs them, and
reports PASS/FAIL. Usable two ways:

    python -m bcir.tests.run_all                       # quick chain (default)
    python -m bcir.tests.run_all --tier c-runtime      # + the C byte-identity tier
    python -m bcir.tests.run_all --tier silicon-degrade # + measured benchmarks (degrade-ok)
    python -m bcir.tests.run_all --tier thorough        # everything (full toolchain + campaigns)
    python -m bcir.tests.run_all -j1                    # force the serial path (safe default: <=2)
    python -m pytest bcir/tests                         # if pytest is installed (same tests)

The suite is dominated by independent compile-and-run tests, but compiler-heavy tests
can each create child processes and substantial scratch state. The safe default is at
most two workers (``fork`` where available, otherwise ``spawn``); ``-j N`` /
``$BCIR_JOBS`` overrides it and explicit ``-j0``/``auto`` opts into all cores. Every worker applies the selected
tier before importing a test module, so capability gating is identical on every host.

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

import concurrent.futures
import importlib
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import traceback

from bcir.toolchain import resolve_llvm_tools

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
_REAL_WHICH = shutil.which
_REAL_SUBPROCESS_RUN = subprocess.run
_BASE_PATH = os.environ.get("PATH", "")
_DEFAULT_CHILD_TIMEOUT_SECONDS = 300.0


def _child_timeout_seconds() -> float:
    raw = os.environ.get("BCIR_TEST_CHILD_TIMEOUT", str(_DEFAULT_CHILD_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("BCIR_TEST_CHILD_TIMEOUT must be a number") from exc
    if not 1.0 <= value <= 3600.0:
        raise ValueError("BCIR_TEST_CHILD_TIMEOUT must be in [1, 3600] seconds")
    return value


def _bounded_subprocess_run(*args, **kwargs):
    """Bound every compiler/generated-program child launched by the test inventory.

    Individual tests may select a tighter timeout.  The default catches stuck generated
    loops and wedged toolchains without requiring hundreds of call sites to remember a
    resource-control keyword.
    """
    kwargs.setdefault("timeout", _child_timeout_seconds())
    return _REAL_SUBPROCESS_RUN(*args, **kwargs)


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
    os.environ["PATH"] = _BASE_PATH                    # idempotent across repeated tier changes
    shutil.which = _REAL_WHICH                         # idempotent across repeated tier changes
    subprocess.run = _bounded_subprocess_run           # every tier bounds child processes
    if spec["thorough"]:
        os.environ["BCIR_THOROUGH"] = "1"
    else:
        os.environ.pop("BCIR_THOROUGH", None)
    visible: frozenset[str] = spec["visible"]  # type: ignore[assignment]
    hidden = _ALL_TOOLCHAIN - visible
    if not hidden:
        # Many historical differential tests ask ``which("clang")`` directly. Promote
        # one supported coherent LLVM installation before those modules import, so an
        # old distro alternative cannot shadow an installed versioned toolchain.
        llvm = resolve_llvm_tools("clang", "llvm-as", "llc",
                                  pipeline="thorough test tier", minimum_major=15)
        if llvm.ok:
            bins = {os.path.dirname(os.path.realpath(path)) for path in llvm.paths.values()}
            if len(bins) == 1:
                selected = next(iter(bins))
                entries = [entry for entry in os.environ["PATH"].split(os.pathsep)
                           if entry and os.path.abspath(entry) != os.path.abspath(selected)]
                os.environ["PATH"] = os.pathsep.join([selected, *entries])
        return                                                  # thorough: nothing gated
    def _gated_which(cmd, *args, **kwargs):
        base = os.path.basename(str(cmd)).lower()
        if base.endswith(".exe"):
            base = base[:-4]
        family = re.sub(r"-\d+$", "", base)
        return None if family in hidden else _REAL_WHICH(cmd, *args, **kwargs)

    shutil.which = _gated_which


_MODULES = [
    "bcir.tests.test_kbcir",
    "bcir.tests.test_rcsp",
    "bcir.tests.test_overlap",
    "bcir.tests.test_fusion",
    "bcir.tests.test_calibrator",
    "bcir.tests.test_schedule",
    "bcir.tests.test_schedule_artifact",
    "bcir.tests.test_microbench",
    "bcir.tests.test_calibloop",
    "bcir.tests.test_tile_prior",
    "bcir.tests.test_channel_prior",
    "bcir.tests.test_device_manifest",
    "bcir.tests.test_event_phases",
    "bcir.tests.test_dma",
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
    "bcir.tests.test_signal_registry",
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
    "bcir.tests.test_cfront_perf",
    "bcir.tests.test_cfront_cli",
    "bcir.tests.test_telemetry",
    "bcir.tests.test_telemetry_security",
    "bcir.tests.test_telemetry_frame",
    "bcir.tests.test_telemetry_metrics",
    "bcir.tests.test_telemetry_export",
    "bcir.tests.test_abi",
    "bcir.tests.test_streampack_tool",
    "bcir.tests.test_artifact_bundle",
    "bcir.tests.test_registry_tool",
    "bcir.tests.test_stackify",
    "bcir.tests.test_stackify_exec",
    "bcir.tests.test_lowering",
    "bcir.tests.test_c_kernel",
    "bcir.tests.test_api",
    "bcir.tests.test_bench",
    "bcir.tests.test_jit",
    "bcir.tests.test_wasm",
    "bcir.tests.test_c_runtime",
    "bcir.tests.test_c_cfront",
    "bcir.tests.test_c_cfront_asm",
    "bcir.tests.test_c_cfront_barrier",
    "bcir.tests.test_c_cfront_portio",
    "bcir.tests.test_cfront_asm_portio_redteam",
    "bcir.tests.test_abi_contract",
    "bcir.tests.test_bounds_provenance",
    "bcir.tests.test_indirect_effect",
    "bcir.tests.test_volatile_atomic_law",
    "bcir.tests.test_provenance_twin",
    "bcir.tests.test_ir_structural_parity",
    "bcir.tests.test_cfront_roundtrip",
    "bcir.tests.test_cfront_link",
    "bcir.tests.test_cfront_constexpr",
    "bcir.tests.test_driver_gpio",
    "bcir.tests.test_c_channel",
    "bcir.tests.test_cpp_handoff",
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
    "bcir.tests.test_quantize",
    "bcir.tests.test_lowbit",
    "bcir.tests.test_native_ai",
    "bcir.tests.test_matmul",
    "bcir.tests.test_matmul_law_parity",
    "bcir.tests.test_activation",
    "bcir.tests.test_activation_law_parity",
    "bcir.tests.test_conv",
    "bcir.tests.test_conv_law_parity",
    "bcir.tests.test_attention",
    "bcir.tests.test_attention_law_parity",
    "bcir.tests.test_transformer",
    "bcir.tests.test_transformer_grads",
    "bcir.tests.test_adaptive_transformer",
    "bcir.tests.test_byte_latent",
    "bcir.tests.test_sequence_interfaces",
    "bcir.tests.test_shape_dtype_laws",
    "bcir.tests.test_train_graph",
    "bcir.tests.test_train_pack_exec",
    "bcir.tests.test_train_c_kernels",
    "bcir.tests.test_model_manifest",
    "bcir.tests.test_model_tokenizer",
    "bcir.tests.test_model_decode",
    "bcir.tests.test_model_quantized",
    "bcir.tests.test_model_ingest",
    "bcir.tests.test_model_weights_io",
    "bcir.tests.test_model_assets",
    "bcir.tests.test_model_assessment",
    "bcir.tests.test_hardware_rl",
    "bcir.tests.test_ham_memory_fabric",
    "bcir.tests.test_tmsao",
    "bcir.tests.test_performance_audit",
    "bcir.tests.test_model_spm",
    "bcir.tests.test_model_serve",
    "bcir.tests.test_hosted_model_spec",
    "bcir.tests.test_hosted_training_pipeline",
    "bcir.tests.test_paged_kv",
    "bcir.tests.test_decode_c_kernels",
    "bcir.tests.test_recurrent",
    "bcir.tests.test_classical",
    "bcir.tests.test_unsupervised",
    "bcir.tests.test_autodiff",
    "bcir.tests.test_autodiff_closure",
    "bcir.tests.test_autodiff_advanced",
    "bcir.tests.test_autodiff_kernel",
    "bcir.tests.test_autodiff_law",
    "bcir.tests.test_losses",
    "bcir.tests.test_optimizers",
    "bcir.tests.test_training",
    "bcir.tests.test_blas_gemm",
    "bcir.tests.test_fftw",
    "bcir.tests.test_fftw2",
    "bcir.tests.test_cerf",
    "bcir.tests.test_lapack",
    "bcir.tests.test_numerical_providers",
    "bcir.tests.test_ols",
    "bcir.tests.test_pca",
    "bcir.tests.test_gsl",
    "bcir.tests.test_sleef",
    "bcir.tests.test_area_b_redteam",
    "bcir.tests.test_sycl_channel",
    "bcir.tests.test_sycl_dispatch",
    "bcir.tests.test_sycl_reduce_matmul",
    "bcir.tests.test_differential",
    "bcir.tests.test_target_matrix",
    "bcir.tests.test_c23_kernels",
    "bcir.tests.test_q8_embed",
    "bcir.tests.test_etl_binrec",
    "bcir.tests.test_asn1_x690",
    "bcir.tests.test_asn1_streampack",
    "bcir.tests.test_asn1_artifact_bundle",
    "bcir.tests.test_c_asn1",
    "bcir.tests.test_asn1_law_parity",
    "bcir.tests.test_asn1_frontend",
    "bcir.tests.test_c_asn1_streampack",
    "bcir.tests.test_asn1_oer",
    "bcir.tests.test_asn1_constraints",
    "bcir.tests.test_asn1_per",
    "bcir.tests.test_asn1_objects",
    "bcir.tests.test_asn1_xer",
    "bcir.tests.test_c_xer",
    "bcir.tests.test_asn1_ecn",
    "bcir.tests.test_asn1_jer",
    "bcir.tests.test_asn1_selection",
    "bcir.tests.test_asn1_jer_bounded",
    "bcir.tests.test_asn1_jer_plan",
    "bcir.tests.test_asn1_dialect",
    "bcir.tests.test_asn1_manifest",
    "bcir.tests.test_asn1_certified",
    "bcir.tests.test_asn1_native_bench",
    "bcir.tests.test_c_jer",
    "bcir.tests.test_c_oer",
    "bcir.tests.test_c_per",
    "bcir.tests.test_native_object_gate",
    "bcir.tests.test_c_executor",
    "bcir.tests.test_c_encoder",
    "bcir.tests.test_verify_cost",
    "bcir.tests.test_precision_lowering",
    "bcir.tests.test_bundle",
    "bcir.tests.test_barrier_ordering",
    "bcir.tests.test_fusion_matmul_activation",
    "bcir.tests.test_fusion_matmul_activation_law_parity",
    "bcir.tests.test_inference_kernel",
    "bcir.tests.test_calling_side_tuning",
    "bcir.tests.test_layout_pivot",
    "bcir.tests.test_layout_pivot_law_parity",
    "bcir.tests.test_cache_predict",
    "bcir.tests.test_cache_contention_law_parity",
    "bcir.tests.test_compose",
    "bcir.tests.test_compose_differential",
    "bcir.tests.test_proof",
    "bcir.tests.test_verify_differential",
    "bcir.tests.test_smart_laws",
    "bcir.tests.test_timing_laws",
    "bcir.tests.test_lifetime_laws",
    "bcir.tests.test_fuzz",
    "bcir.tests.test_measured_silicon",
    "bcir.tests.test_silicon_runbook",
    "bcir.tests.test_clang_compare",
    "bcir.tests.test_tiers",
    "bcir.tests.test_toolchain",
    "bcir.tests.test_security_hardening",
    "bcir.tests.test_perf_budget",
    "bcir.tests.test_import_quarantine",
    "bcir.tests.test_registry_complete",
]


_TIER_BLURB = {
    "quick": "pure-Python oracle/law/parity + honest-degrade silicon (toolchain hidden)",
    "c-runtime": "+ C compiler: freestanding-runtime / StreamPack-ABI byte-identity / C kernels",
    "silicon-degrade": "+ measured benchmarks in degrade mode (correctness + valid measurement, "
                       "no faked speedup)",
    "thorough": "everything: full toolchain (IR/JIT/WASM/native) + the large campaigns",
}


def _collect_tests() -> list[tuple[str, str]]:
    """Import every module and return the flat (module, test) work list, in declared module order
    then alphabetical test order (so a serial run is byte-identical to the historic behaviour)."""
    pairs: list[tuple[str, str]] = []
    for modname in _MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                pairs.append((modname, name))
    return pairs


def _run_one(pair: tuple[str, str]) -> tuple[str, str, str | None]:
    """Run one test function and return (module, test, traceback-or-None)."""
    modname, name = pair
    try:
        getattr(importlib.import_module(modname), name)()
    except Exception:  # noqa: BLE001 - capture every failure for the parent to report
        return modname, name, traceback.format_exc()
    return modname, name, None


def _worker_init(tier: str) -> None:
    """Apply capability gating before a forked or spawned worker imports tests."""
    _apply_tier(tier)


def _pool_context():
    """Use cheap fork inheritance where supported and portable spawn elsewhere."""
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    return multiprocessing.get_context(method)


def _resolve_shard(argv: list[str] | None = None) -> tuple[int, int]:
    """Parse ``--shard I/N``: run the Ith of N disjoint slices. Default ``1/1`` (all).

    The slice is a STRIDE (`pairs[i-1::n]`, I being 1-based), not a contiguous block. That matters because the
    per-test cost is wildly uneven -- `test_c_cfront` alone is ~60% of the suite's CPU and
    its tests are adjacent in discovery order, so contiguous blocks would put nearly all of
    it in one shard and the split would buy almost nothing. Striding interleaves the
    expensive module across every shard.

    The union of all N shards is exactly the full inventory and the shards are disjoint, so
    sharding cannot silently drop a test -- which is the property that makes it safe to
    split a conformance suite across CI jobs at all.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    raw: str | None = None
    for i, a in enumerate(argv):
        if a == "--shard" and i + 1 < len(argv):
            raw = argv[i + 1]; break
        if a.startswith("--shard="):
            raw = a.split("=", 1)[1]; break
    if raw is None:
        return (1, 1)
    try:
        index, count = (int(part) for part in raw.split("/", 1))
    except ValueError:
        sys.stderr.write(f"[run_all] --shard needs I/N, got {raw!r}\n")
        raise SystemExit(2) from None
    if not (count >= 1 and 1 <= index <= count):
        sys.stderr.write(f"[run_all] --shard {raw} is out of range\n")
        raise SystemExit(2)
    return (index, count)


def _resolve_jobs(argv: list[str] | None = None) -> int:
    """Resolve a bounded worker count.

    ``-j N`` / ``--jobs N`` argv takes precedence over ``$BCIR_JOBS``. No
    setting, a malformed setting, or a missing option value uses the safe
    ``min(cpu, 2)`` default. ``-j0``/``auto`` is the explicit all-core opt-in;
    numeric values clamp to ``[1, cpu]``.
    """
    val: str | None = None
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv):
        if a in ("-j", "--jobs") and i + 1 < len(argv):
            val = argv[i + 1]; break
        if a.startswith("-j") and a != "-j":
            val = a[2:]; break
        if a.startswith("--jobs="):
            val = a.split("=", 1)[1]; break
    if val is None:
        val = os.environ.get("BCIR_JOBS")
    cpu = os.cpu_count() or 1
    safe_default = min(cpu, 2)
    if val is None:
        return safe_default
    if val in ("auto", "0"):
        return cpu
    try:
        return max(1, min(cpu, int(val)))
    except ValueError:
        return safe_default


def main() -> int:
    if "-h" in sys.argv or "--help" in sys.argv:
        print("usage: python -m bcir.tests.run_all [--tier TIER] [-j N] [--shard I/N]")
        print("       python -m bcir.tests.run_all --list-tiers")
        print("safe default: at most 2 workers; -j0 or --jobs=auto opts into all cores")
        return 0
    if "--list-tiers" in sys.argv:
        print("tiers (escalating capability ladder):")
        for name in TIERS:
            print(f"  {name:<16} {_TIER_BLURB[name]}")
        return 0
    tier = resolve_tier()
    _apply_tier(tier)                       # gate the toolchain BEFORE importing test modules
    jobs = _resolve_jobs()
    shard, shards = _resolve_shard()
    print(f"[run_all] tier={tier} — {_TIER_BLURB[tier]}  (jobs={jobs}"
          + (f", shard {shard}/{shards}" if shards > 1 else "") + ")\n")
    pairs = _collect_tests()                # stable discovery; fork also benefits from the warm imports
    total = len(pairs)
    if shards > 1:
        pairs = pairs[shard - 1::shards]
        print(f"[run_all] shard {shard}/{shards}: {len(pairs)} of {total} tests\n")
    passed = failed = 0

    def _report(modname: str, name: str, tb: str | None) -> None:
        nonlocal passed, failed
        if tb is None:
            passed += 1
            print(f"PASS {modname}.{name}")
        else:
            failed += 1
            print(f"FAIL {modname}.{name}\n{tb}", end="")

    if jobs <= 1:
        for pair in pairs:                  # the historic serial path, unchanged
            _report(*_run_one(pair))
    else:
        # Fork where available; otherwise spawn. The initializer installs the tier gate before any
        # worker test import. ProcessPoolExecutor workers are NON-daemonic, so a test that spawns its
        # own pool still works. `.map` preserves submission order for a stable log; chunksize=1
        # dispatches one test at a time so a worker that frees up pulls the next -- natural load
        # balancing for the very uneven per-test compile cost (the C twin's build cache is a
        # worker-process global, so it is still built once per worker).
        ctx = _pool_context()
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=ctx,
                initializer=_worker_init, initargs=(tier,)) as pool:
            for res in pool.map(_run_one, pairs, chunksize=1):
                _report(*res)

    suffix = f" (shard {shard}/{shards} of {total} total)" if shards > 1 else ""
    print(f"\n{passed} passed, {failed} failed{suffix}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
