"""The run_all registry completeness guard (the no-silent-caps discipline, on the suite itself).

`run_all._MODULES` is an explicit, thematically-ordered registry -- and CI runs exactly it.
An unregistered `test_*.py` file therefore passes locally (direct execution) while NEVER
running in CI: fifteen files, three merged waves deep, drifted that way before this guard
existed. The law: every `bcir/tests/test_*.py` on disk is registered, and every registered
module exists on disk. A new test file that forgets the registry line now fails the suite
it was left out of.

Registration is a claim about what *runs*, though, and the first law here only
checked that the name appears in two places. `run_all` discovers tests with
`dir(module)` and a `test_` prefix, so a module whose tests live as methods on a
`unittest.TestCase` subclass contributes **nothing**: the class name does not start
with `test_`, and nothing reaches its methods. `test_red_sweep.py` landed that way
-- registered, counted among the files, thirteen tests, zero of them run, and every
gate green. The second law below closes it: a registered module must yield at least
one discoverable test, and must not hide a test where the discovery cannot see it
(`docs/security/laws.md` L2, L21)."""

import ast
import os


def test_every_test_file_on_disk_is_registered_and_vice_versa():
    from bcir.tests.run_all import _MODULES

    d = os.path.dirname(os.path.abspath(__file__))
    on_disk = {
        f"bcir.tests.{f[:-3]}" for f in os.listdir(d) if f.startswith("test_") and f.endswith(".py")
    }
    registered = set(_MODULES)
    missing = sorted(on_disk - registered)  # written but never run by run_all/CI
    ghosts = sorted(registered - on_disk)  # registered but deleted/renamed on disk
    assert not missing, f"test files not in run_all._MODULES (CI never runs them): {missing}"
    assert not ghosts, f"run_all._MODULES entries with no file on disk: {ghosts}"
    assert len(_MODULES) == len(registered), "duplicate entries in run_all._MODULES"


#: A registered module contributing fewer than this many discoverable tests is
#: reporting a registration that runs almost nothing. One is the honest floor: a
#: file exists to run at least one test, and anything above one would be a quota.
_MIN_DISCOVERABLE = 1


def _hidden_tests(path: str) -> list[str]:
    """`test_*` functions nested inside a class, which `run_all` cannot discover.

    Read statically rather than by importing, so the finding does not depend on
    whether this host can import the module's dependencies -- a module that is
    skipped here for a missing optional rail would otherwise hide the same defect.

    Declared scope: a `def`/`async def` named `test_*` whose nearest enclosing
    block is a `ClassDef`. A test attached to a class at run time, or produced by a
    decorator or metaclass, is out of scope and belongs to a linter; no module in
    this tree does either, and the answer to the next soundness question is to
    point here rather than to grow an interpreter.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    hidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith(
                "test_"
            ):
                hidden.append(f"{node.name}.{child.name}")
    return hidden


def test_every_registered_module_actually_yields_tests_to_the_runner():
    """Registration is a claim about what runs, so check what would run."""
    import importlib

    from bcir.tests.run_all import _MODULES, _REPO_ONLY_MODULES, _is_source_checkout

    # The same classifier `run_all` uses, not a second one: a module it declines to
    # import from an installed wheel must be declined here by the identical rule, or
    # this check reports a hole the runner does not have (L14).
    source = _is_source_checkout()
    directory = os.path.dirname(os.path.abspath(__file__))
    # What this environment is supposed to examine, from the registry and the runner's
    # own skip rule. Everything below is accounted against it, so no module can be
    # dropped by a `continue` without saying so.
    runnable = [name for name in _MODULES if source or name not in _REPO_ONLY_MODULES]
    empty, hidden, absent, unimportable, examined = [], [], [], [], 0
    for modname in _MODULES:
        path = os.path.join(directory, modname.rsplit(".", 1)[-1] + ".py")
        if not os.path.exists(path):
            # The sibling law above owns this case and reports it better; counted
            # here only so the accounting below stays total.
            if modname in runnable:
                absent.append(modname)
            continue
        found = _hidden_tests(path)
        if found:
            hidden.append(f"{modname}: {', '.join(found)}")
        if not source and modname in _REPO_ONLY_MODULES:
            continue  # installed-package run: the runner skips these, so this does too
        try:
            module = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001 - an import failure is the suite's own report
            unimportable.append(f"{modname}: {type(exc).__name__}")
            continue
        examined += 1
        discoverable = [
            name
            for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        if len(discoverable) < _MIN_DISCOVERABLE:
            empty.append(f"{modname}: {len(discoverable)} discoverable test(s)")

    assert not hidden, (
        "run_all discovers module-level `test_*` callables only, so a test defined "
        "inside a class is registered and never runs: " + "; ".join(hidden)
    )
    assert not empty, "registered module(s) that contribute no test to the runner: " + "; ".join(
        empty
    )
    # Anti-vacuity, in two statements neither of which is a number typed in here.
    #
    # This used to be `examined >= 200`, which is a second, weaker description of the
    # environment the test runs in -- and it described the wrong one. A checkout
    # imports all 260 registered modules and clears it; the installed-package run the
    # suite also has to pass skips the 65 in `_REPO_ONLY_MODULES`, leaves 195, and
    # fails an anti-vacuity guard on a run where nothing is vacuous. A mirror of a
    # count will drift from the count (`docs/security/laws.md` L15), and this one had
    # already drifted before it was written.
    #
    # First: the accounting is total. Every module the runner would have taken here is
    # examined, absent (the sibling law's finding), or unimportable -- a `continue`
    # that quietly loses modules cannot hide inside the loop.
    assert examined + len(absent) + len(unimportable) == len(runnable), (
        f"accounting: {len(runnable)} module(s) to examine, but {examined} examined + "
        f"{len(absent)} absent + {len(unimportable)} unimportable does not add up"
    )
    assert not unimportable, (
        "registered module(s) that do not import, so their tests were never examined: "
        + "; ".join(unimportable)
    )
    # Second: most of the registry really runs here, whichever environment this is.
    # 260 of 260 on a checkout, 195 of 260 from a wheel. If exclusions ever grew past
    # half the registry this would fire, which is the finding rather than a nuisance
    # -- a skip is where a shipping defect hides (L21).
    assert 2 * examined >= len(_MODULES), (
        f"anti-vacuity: only {examined} of {len(_MODULES)} registered module(s) were "
        f"imported and examined here, so the assertions above are about a minority of "
        "the suite"
    )


if __name__ == "__main__":
    import sys

    failed = 0
    for check in (
        test_every_test_file_on_disk_is_registered_and_vice_versa,
        test_every_registered_module_actually_yields_tests_to_the_runner,
    ):
        try:
            check()
            print(f"PASS {check.__name__}")
        except AssertionError as e:
            print(f"FAIL {check.__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
