"""Hot/cold separation: the L0 law as a regression test.

The hot path carries *decisions, not models* (LangRef Sec. 13, L0). Execution
replays a StreamPack whose decisions were compiled out at plan time; it must not
import -- let alone run -- the learned organs or the planner. This test pins that
separation structurally so it survives the move of intelligence into C/C++: the
C++ ports must preserve the same boundary, and these are the modules that must
stay lean.
"""

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The learned / plan-time "intelligence" organs (a model on the hot path == an L0
# violation): calibration, the soft/Bayesian/MoE/regret/accelerator learners, and
# the e-graph / memory / operad / two-truth / mapping plan-time machinery.
_LEARNING = {
    "softdp",
    "bayescal",
    "moegate",
    "regret",
    "accel",
    "microbench",
    "calibrate",
    "egraph",
    "memory",
    "operad",
    "twotruth",
    "mapping",
}
# The plan-time search itself (execution replays decisions, it does not re-plan).
_PLANNING = {"realize", "rcsp", "semiring", "portfolio"}

# module path -> the import families it may NOT pull in.
_HOT = {
    "bcir/gem/execute.py": _LEARNING | _PLANNING,  # the executor: neither
    "bcir/abi/streampack_abi.py": _LEARNING | _PLANNING,  # the wire codec: neither
    "bcir/gem/streampack.py": _LEARNING,  # hydration carries the plan, not the models
}


def _imported_modules(rel_path: str) -> set:
    """The set of module *leaf names* a file imports (direct imports only)."""
    tree = ast.parse((_ROOT / rel_path).read_text())
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.rsplit(".", 1)[-1])
    return names


def test_hot_path_carries_no_models_or_planner():
    for rel_path, forbidden in _HOT.items():
        imported = _imported_modules(rel_path)
        leaked = imported & forbidden
        assert not leaked, f"{rel_path} (hot) imports forbidden modules: {sorted(leaked)}"


def test_executor_runtime_imports_no_learning():
    # Belt-and-suspenders: importing + using the executor must not drag a learned
    # organ into sys.modules under the bcir.kbcir namespace.
    import sys

    learning_mods = {f"bcir.kbcir.{m}" for m in _LEARNING}
    before = set(sys.modules)
    # a fresh import of the executor path
    import importlib

    importlib.import_module("bcir.gem.execute")
    pulled = (set(sys.modules) - before) & learning_mods
    assert not pulled, f"importing the executor pulled in learning modules: {pulled}"
