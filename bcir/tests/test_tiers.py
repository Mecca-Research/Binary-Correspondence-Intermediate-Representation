"""Named test tiers (`python -m bcir.tests.run_all --tier ...`) as a pinned contract.

The four tiers form an *escalating capability ladder* — each tier exposes a superset of
the host tools the previous one does (quick ⊂ c-runtime ⊂ silicon-degrade ⊂ thorough).
These guards keep that property (and the resolution precedence CI depends on) from drifting
as tools are added to the gate.
"""

import os

from bcir.tests import run_all as R

_LADDER = ["quick", "c-runtime", "silicon-degrade", "thorough"]


def test_the_four_named_tiers_exist():
    assert list(R.TIERS) == _LADDER


def test_capability_ladder_is_monotonic():
    # visible(quick) ⊆ visible(c-runtime) ⊆ visible(silicon-degrade) ⊆ visible(thorough)
    for lo, hi in zip(_LADDER, _LADDER[1:]):
        assert R.TIERS[lo]["visible"] <= R.TIERS[hi]["visible"], f"{lo} not ⊆ {hi}"


def test_quick_hides_everything_thorough_hides_nothing():
    assert R.TIERS["quick"]["visible"] == frozenset()
    assert R.TIERS["thorough"]["visible"] == R._ALL_TOOLCHAIN


def test_only_thorough_runs_the_full_campaigns():
    assert R.TIERS["thorough"]["thorough"] is True
    assert all(not R.TIERS[t]["thorough"] for t in _LADDER if t != "thorough")


def test_c_runtime_exposes_a_compiler_but_not_the_llvm_jit_wasm_tools():
    vis = R.TIERS["c-runtime"]["visible"]
    assert {"clang", "cc", "gcc"} <= vis                       # can build the C runtime
    assert not (R._LLVM_JIT_WASM & vis)                        # IR/JIT/WASM still deferred


def test_resolve_tier_precedence():
    # explicit --tier wins over everything
    assert R.resolve_tier(["--tier", "c-runtime"]) == "c-runtime"
    assert R.resolve_tier(["--tier=thorough"]) == "thorough"
    # $BCIR_TIER is next
    os.environ.pop("BCIR_THOROUGH", None)
    os.environ["BCIR_TIER"] = "silicon-degrade"
    try:
        assert R.resolve_tier([]) == "silicon-degrade"
        # then the BCIR_THOROUGH back-compat fallback (existing CI exports only this)
        del os.environ["BCIR_TIER"]
        os.environ["BCIR_THOROUGH"] = "1"
        assert R.resolve_tier([]) == "thorough"
        # default
        del os.environ["BCIR_THOROUGH"]
        assert R.resolve_tier([]) == "quick"
    finally:
        os.environ.pop("BCIR_TIER", None)
        os.environ.pop("BCIR_THOROUGH", None)


def test_unknown_tier_is_rejected():
    try:
        R.resolve_tier(["--tier", "nonexistent"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("unknown tier should raise SystemExit(2)")


def test_versioned_and_windows_tool_names_are_gated_by_family():
    calls = []

    def fake_which(cmd, *args, **kwargs):
        calls.append(cmd)
        return f"/tools/{cmd}"

    saved_real, saved_which = R._REAL_WHICH, R.shutil.which
    try:
        R._REAL_WHICH = fake_which
        R._apply_tier("quick")
        for name in ("clang-20", "llvm-link-22.exe", "/opt/llvm/bin/llc-19", "gcc.exe"):
            assert R.shutil.which(name) is None
        assert R.shutil.which("python.exe") == "/tools/python.exe"
        R._apply_tier("thorough")
        assert R.shutil.which("clang-20") == "/tools/clang-20"
    finally:
        R._REAL_WHICH = saved_real
        R.shutil.which = saved_which
        os.environ.pop("BCIR_TIER", None)
        os.environ.pop("BCIR_THOROUGH", None)


def test_pool_context_uses_spawn_when_fork_is_unavailable():
    saved_methods, saved_context = R.multiprocessing.get_all_start_methods, R.multiprocessing.get_context
    seen = []
    try:
        R.multiprocessing.get_all_start_methods = lambda: ["spawn"]
        R.multiprocessing.get_context = lambda method: seen.append(method) or method
        assert R._pool_context() == "spawn"
        assert seen == ["spawn"]
    finally:
        R.multiprocessing.get_all_start_methods = saved_methods
        R.multiprocessing.get_context = saved_context


def test_explicit_clang_check_reports_no_compiler_honestly():
    from bcir.frontends.cfront import pipeline

    saved = pipeline.shutil.which
    try:
        pipeline.shutil.which = lambda *_args, **_kwargs: None
        result = pipeline.compile_unit("int f(int x){return x+1;}", check_clang=True)
        assert result.equivalence == "skip:no-cc"
    finally:
        pipeline.shutil.which = saved
