"""The RED-sweep harness, driven into each way it could produce a false verdict.

`docs/security/laws.md` L25 says a fault injection must reach the code under test,
and lists four properties that make a sweep's verdict worth reading. A law with no
witness is a paragraph (L22), so each property is exercised here against a
synthetic gate rather than a real one -- the expensive rails stay out of the unit
tier (L19), and a synthetic subject lets the harness be driven into refusals that
a real gate would never produce on demand.

The load-bearing test is `test_a_same_length_fault_is_not_compiled_away`: it primes
the bytecode cache exactly the way a control run does, then injects a replacement
of identical length, and asserts the gate subprocess observes the *new* value. That
is the defect L25 was written for, and without the harness's bytecode discard it
fails.

**Every test here is a module-level function, and that is load-bearing rather than
stylistic.** `run_all` discovers tests with `dir(module)` and a `test_` prefix, so a
`unittest.TestCase` subclass contributes *nothing* to the suite -- the class name
does not start with `test_`, and its methods are never reached. This file was first
written with four TestCase classes: it was registered in `run_all._MODULES`,
`test_registry_complete` passed because the registration existed, the runner
reported its usual total, and all thirteen tests below ran zero times. Registration
is a claim about what *runs*, and `test_registry_complete` now checks that claim
directly.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.testing.red_sweep import Fault, Sweep, SweepError, load_table  # noqa: E402

#: A subject with one constant, and a gate that reports the constant it actually
#: imported. The gate is red exactly when the constant is not the expected value,
#: and it names its check the way every gate in this tree does.
_SUBJECT = "THRESHOLD = 0.70\n"
_GATE = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subject

if subject.THRESHOLD != 0.70:
    print("gate: FAILED")
    print(f"  - threshold: subject says {subject.THRESHOLD}, expected 0.70")
    raise SystemExit(1)
print("gate: PASSED")
"""


class _Bench:
    """A throwaway subject-and-gate pair, removed however the test leaves.

    A context manager rather than `addCleanup` because these are plain functions;
    the directory has to go even when an assertion raises, and a `finally` around
    every body would be the same thing written thirteen times.
    """

    def __enter__(self) -> tuple[Path, Path, list[str]]:
        self.directory = Path(tempfile.mkdtemp(prefix="bcir-red-sweep-"))
        subject = self.directory / "subject.py"
        subject.write_text(_SUBJECT, encoding="utf-8", newline="\n")
        (self.directory / "gate.py").write_text(_GATE, encoding="utf-8", newline="\n")
        return self.directory, subject, [sys.executable, str(self.directory / "gate.py")]

    def __exit__(self, *_exc) -> bool:
        shutil.rmtree(self.directory, ignore_errors=True)
        return False


def _sweep(directory: Path, command: list[str], faults: list[Fault]) -> Sweep:
    return Sweep(command=command, faults=faults, cwd=directory, timeout=120)


def _refused(call, *args) -> str:
    """Run `call` expecting a `SweepError`, and hand back its message.

    A refusal asserted by `try/except` alone passes when nothing is raised at all,
    which is the one outcome these tests exist to catch.
    """
    try:
        call(*args)
    except SweepError as exc:
        return str(exc)
    raise AssertionError(f"{call!r} returned instead of refusing")


# -- each way the harness could report something it cannot stand behind --------


def test_an_empty_fault_table_is_refused():
    with _Bench() as (directory, _, command):
        message = _refused(_sweep(directory, command, []).run)
        assert "injects nothing" in message, message


def test_an_anchor_that_matches_nothing_is_a_harness_error():
    """Otherwise a gate that passes because nothing changed reads as 'not caught'."""
    with _Bench() as (directory, subject, command):
        fault = Fault("absent anchor", "threshold", subject, "NOT_IN_THE_FILE = 1", "x = 2")
        message = _refused(_sweep(directory, command, [fault]).run)
        assert "occurs 0 time(s)" in message, message
        assert subject.read_text(encoding="utf-8") == _SUBJECT


def test_an_anchor_that_matches_twice_is_refused():
    """Two defects attributed to one check is not one experiment."""
    with _Bench() as (directory, subject, command):
        subject.write_text(_SUBJECT + _SUBJECT, encoding="utf-8", newline="\n")
        fault = Fault(
            "doubled anchor", "threshold", subject, "THRESHOLD = 0.70", "THRESHOLD = 0.55"
        )
        message = _refused(_sweep(directory, command, [fault]).run)
        assert "occurs 2 time(s)" in message, message


def test_a_replacement_identical_to_its_anchor_is_refused():
    """A fault that changes nothing would be reported as a defect nobody caught."""
    with _Bench() as (directory, subject, command):
        fault = Fault("no-op fault", "threshold", subject, "THRESHOLD = 0.70", "THRESHOLD = 0.70")
        message = _refused(_sweep(directory, command, [fault]).run)
        assert "did not change" in message, message


def test_a_control_run_that_is_already_red_invalidates_the_sweep():
    """Every row below a red control is meaningless, so there are no rows."""
    with _Bench() as (directory, subject, command):
        subject.write_text("THRESHOLD = 0.11\n", encoding="utf-8", newline="\n")
        fault = Fault("anything", "threshold", subject, "0.11", "0.22")
        message = _refused(_sweep(directory, command, [fault]).run)
        assert "already FAILING" in message, message


# -- the outcomes that look alike in a bad sweep -------------------------------


def test_a_caught_fault_is_red_and_the_tree_is_restored():
    with _Bench() as (directory, subject, command):
        before = subject.read_bytes()
        fault = Fault("threshold moves", "threshold", subject, "0.70", "0.55")
        sweep = _sweep(directory, command, [fault])
        assert sweep.run() == 0
        assert sweep.results[0].verdict == "RED"
        assert "threshold" in sweep.results[0].fired
        assert subject.read_bytes() == before, "the sweep left the tree modified"


def test_a_gate_that_passes_with_the_defect_in_is_not_caught():
    """The honest negative: the check exists, ran, and did not notice."""
    with _Bench() as (directory, subject, command):
        subject.write_text(_SUBJECT + "UNUSED = 1\n", encoding="utf-8", newline="\n")
        fault = Fault("an ignored constant", "threshold", subject, "UNUSED = 1", "UNUSED = 2")
        sweep = _sweep(directory, command, [fault])
        assert sweep.run() == 1
        assert sweep.results[0].verdict == "NOT CAUGHT"


def test_the_wrong_check_firing_is_not_a_catch():
    """A gate going red for an unrelated reason has not demonstrated this check."""
    with _Bench() as (directory, subject, command):
        fault = Fault("threshold moves", "some-other-check", subject, "0.70", "0.55")
        sweep = _sweep(directory, command, [fault])
        assert sweep.run() == 1
        assert sweep.results[0].verdict == "WRONG CHECK"


def test_a_gate_that_cannot_be_launched_is_a_verdict_not_a_catch():
    """L1: a harness error is reported as one, never as evidence."""
    with _Bench() as (directory, subject, _):
        fault = Fault("threshold moves", "threshold", subject, "0.70", "0.55")
        sweep = _sweep(directory, [str(directory / "no-such-binary")], [fault])
        _refused(sweep.run)


# -- L25's own defect: the fault that never reached the interpreter ------------


def test_a_same_length_fault_is_not_compiled_away():
    """The regression test for the observed defect.

    The control run imports `subject` and leaves a `.pyc` behind. The fault below
    replaces four characters with four characters, so the cached module can still
    satisfy CPython's `(mtime, size)` check within the same second. Without the
    harness's bytecode discard the gate re-imports the *old* constant, passes, and
    the sweep reports a defect nobody caught.
    """
    with _Bench() as (directory, subject, command):
        fault = Fault("same-length threshold", "threshold", subject, "0.70", "0.55")
        assert len("0.70") == len("0.55"), "the premise of this test"

        sweep = _sweep(directory, command, [fault])
        assert sweep.run() == 0, "a same-length fault was compiled away"
        assert sweep.results[0].verdict == "RED"


def test_discarding_bytecode_removes_what_a_run_leaves_behind():
    with _Bench() as (directory, _, command):
        sweep = _sweep(directory, command, [])
        sweep.run_gate()
        cached = list(directory.rglob("*.pyc"))
        assert cached, "the gate did not leave bytecode; the premise is gone"
        assert sweep.discard_bytecode() >= len(cached)
        assert not list(directory.rglob("*.pyc"))


# -- the committed tables are loadable, anchored, and about real gates ---------


def test_every_committed_table_loads_and_anchors_exactly_once():
    tables = sorted((_REPO_ROOT / "tools" / "testing" / "faults").glob("*.json"))
    assert len(tables) >= 3, "the standing evidence has gone missing"
    for table in tables:
        command, faults = load_table(table)
        assert faults, f"{table.name} declares no faults"
        assert Path(command[-1]).name.endswith(".py")
        for fault in faults:
            text = fault.path.read_text(encoding="utf-8")
            assert fault.anchor_count(text) == 1, (
                f"{table.name}: {fault.label!r} anchors {fault.anchor_count(text)} time(s) "
                f"in {fault.path.name} -- the table has drifted from the code it injects into"
            )
            assert fault.old != fault.new, f"{table.name}: {fault.label!r} changes nothing"


def test_a_malformed_table_is_refused_rather_than_partly_run():
    directory = Path(tempfile.mkdtemp(prefix="bcir-red-table-"))
    try:
        for name, payload in (
            ("not-an-object.json", "[]"),
            ("no-command.json", json.dumps({"faults": []})),
            ("empty-faults.json", json.dumps({"command": ["@python", "x.py"], "faults": []})),
            (
                "missing-field.json",
                json.dumps({"command": ["@python", "x.py"], "faults": [{"label": "a"}]}),
            ),
            ("unreadable.json", "{not json"),
        ):
            path = directory / name
            path.write_text(payload, encoding="utf-8", newline="\n")
            _refused(load_table, path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
