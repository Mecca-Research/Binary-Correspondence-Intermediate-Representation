"""The run_all registry completeness guard (the no-silent-caps discipline, on the suite itself).

`run_all._MODULES` is an explicit, thematically-ordered registry -- and CI runs exactly it.
An unregistered `test_*.py` file therefore passes locally (direct execution) while NEVER
running in CI: fifteen files, three merged waves deep, drifted that way before this guard
existed. The law: every `bcir/tests/test_*.py` on disk is registered, and every registered
module exists on disk. A new test file that forgets the registry line now fails the suite
it was left out of."""

import os


def test_every_test_file_on_disk_is_registered_and_vice_versa():
    from bcir.tests.run_all import _MODULES
    d = os.path.dirname(os.path.abspath(__file__))
    on_disk = {f"bcir.tests.{f[:-3]}" for f in os.listdir(d)
               if f.startswith("test_") and f.endswith(".py")}
    registered = set(_MODULES)
    missing = sorted(on_disk - registered)           # written but never run by run_all/CI
    ghosts = sorted(registered - on_disk)            # registered but deleted/renamed on disk
    assert not missing, f"test files not in run_all._MODULES (CI never runs them): {missing}"
    assert not ghosts, f"run_all._MODULES entries with no file on disk: {ghosts}"
    assert len(_MODULES) == len(registered), "duplicate entries in run_all._MODULES"


if __name__ == "__main__":
    import sys

    try:
        test_every_test_file_on_disk_is_registered_and_vice_versa()
        print("PASS test_every_test_file_on_disk_is_registered_and_vice_versa")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
