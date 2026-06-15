"""Performance-audit regression guards: keep the *machinery* out of the simple path.

The lowest-level performance finding (docs/BCIR_STRATEGY_AND_ROADMAP.md, the
performance-audit section): the dominant cost of the simplest BCIR use -- plan a
program and emit a kernel -- was never the planning math (~0.05 ms) or the emitted
loop (memory-bandwidth-bound; loop form is within measurement noise), but the
*fixed import overhead* of eagerly loading the Phase-13..26 learned/categorical
research stack and the GEM executor. That is exactly the "our own machinery gets in
the way at the simplest levels" tax. The packages now import lazily (PEP 562).

These guards are STRUCTURAL -- they assert *which modules load*, not wall-clock
time -- so they are deterministic on any CI host (timing thresholds would be
flaky). Each check runs in a fresh interpreter so the in-process module cache (the
full suite imports everything) cannot mask a regression.
"""

import importlib
import subprocess
import sys
import types

# Modules that must NOT load merely to plan a program and emit its kernel. Includes
# the opt-in "smart" organs (allocator / sensing / cim / dvfs / specialist): they
# are capabilities the caller invokes explicitly, never on the simple path.
_HEAVY_KBCIR = ("calibloop", "microbench", "accel", "portfolio", "moegate", "regret",
                "softdp", "egraph", "operad", "twotruth", "memory", "throttle",
                "mapping", "bayescal", "allocator", "sensing")
_HEAVY_LOWER = ("jit", "wasm", "specialist")                    # JIT + WASM/stack + specialist
_HEAVY_GEM = ("concurrency", "overlap", "schedule", "async_tokens", "cim", "dvfs")


def _bcir_modules_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter; return the set of imported ``bcir.*`` modules."""
    src = code + "\nimport sys\nprint('\\n'.join(m for m in sys.modules if m.startswith('bcir')))"
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert out.returncode == 0, f"probe failed:\n{out.stderr}"
    return set(out.stdout.split())


def _heavy() -> set[str]:
    return ({f"bcir.kbcir.{m}" for m in _HEAVY_KBCIR}
            | {f"bcir.lower.{m}" for m in _HEAVY_LOWER}
            | {f"bcir.gem.{m}" for m in _HEAVY_GEM})


# --- the simple path stays lean ---------------------------------------------------

def test_simple_emit_path_does_not_load_the_research_stack():
    # the whole public "plan + emit a deployable kernel" path.
    loaded = _bcir_modules_after("from bcir.api import build_artifact; build_artifact('vector_add')")
    leaked = sorted(_heavy() & loaded)
    assert not leaked, f"simple emit path eagerly imported heavy modules: {leaked}"


def test_importing_the_planner_package_is_lean():
    loaded = _bcir_modules_after("import bcir.kbcir")
    leaked = sorted({f"bcir.kbcir.{m}" for m in _HEAVY_KBCIR} & loaded)
    assert not leaked, f"import bcir.kbcir eagerly pulled the learned stack: {leaked}"


def test_cli_plan_does_not_load_executor_or_research_stack():
    # `python -m bcir.run vector_add` (the default plan-and-print) is a "simple
    # process"; it must not pull the executor/scheduler or the learned organs.
    loaded = _bcir_modules_after(
        "import sys; sys.argv = ['bcir.run', 'vector_add']; from bcir.run import main; main()")
    leaked = sorted(_heavy() & loaded)
    assert not leaked, f"CLI plan eagerly imported heavy modules: {leaked}"


# --- the full public API is still reachable + collisions resolve correctly --------

def test_full_public_api_is_reachable_lazily():
    for pkg in ("bcir.kbcir", "bcir.lower", "bcir.gem"):
        mod = importlib.import_module(pkg)
        for name in mod.__all__:
            assert getattr(mod, name) is not None, f"{pkg}.{name} did not resolve"


def test_submodule_name_collisions_resolve_to_the_function_not_the_module():
    # `weights`/`stackify`/`execute` are each both an exported function AND a
    # submodule; lazy loading must keep the function reachable (the bug this guards).
    import bcir.gem as g
    import bcir.kbcir as k
    import bcir.lower as low
    for value in (k.weights, low.stackify, g.execute):
        assert callable(value) and not isinstance(value, types.ModuleType)


def test_submodules_remain_directly_importable():
    for m in ("bcir.kbcir.egraph", "bcir.kbcir.microbench", "bcir.lower.wasm",
              "bcir.gem.execute"):
        assert importlib.import_module(m) is not None
