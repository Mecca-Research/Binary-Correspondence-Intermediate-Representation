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
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
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


def _bench(case: unittest.TestCase) -> tuple[Path, Path, list[str]]:
    directory = Path(tempfile.mkdtemp(prefix="bcir-red-sweep-"))
    case.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
    subject = directory / "subject.py"
    subject.write_text(_SUBJECT, encoding="utf-8", newline="\n")
    (directory / "gate.py").write_text(_GATE, encoding="utf-8", newline="\n")
    return directory, subject, [sys.executable, str(directory / "gate.py")]


def _sweep(directory: Path, command: list[str], faults: list[Fault]) -> Sweep:
    return Sweep(command=command, faults=faults, cwd=directory, timeout=120)


class RedSweepRefusals(unittest.TestCase):
    """Each way the harness could report something it cannot stand behind."""

    def test_an_empty_fault_table_is_refused(self):
        directory, _, command = _bench(self)
        with self.assertRaises(SweepError) as caught:
            _sweep(directory, command, []).run()
        self.assertIn("injects nothing", str(caught.exception))

    def test_an_anchor_that_matches_nothing_is_a_harness_error(self):
        """Otherwise a gate that passes because nothing changed reads as 'not caught'."""
        directory, subject, command = _bench(self)
        fault = Fault("absent anchor", "threshold", subject, "NOT_IN_THE_FILE = 1", "x = 2")
        with self.assertRaises(SweepError) as caught:
            _sweep(directory, command, [fault]).run()
        self.assertIn("occurs 0 time(s)", str(caught.exception))
        self.assertEqual(subject.read_text(encoding="utf-8"), _SUBJECT)

    def test_an_anchor_that_matches_twice_is_refused(self):
        """Two defects attributed to one check is not one experiment."""
        directory, subject, command = _bench(self)
        subject.write_text(_SUBJECT + _SUBJECT, encoding="utf-8", newline="\n")
        fault = Fault(
            "doubled anchor", "threshold", subject, "THRESHOLD = 0.70", "THRESHOLD = 0.55"
        )
        with self.assertRaises(SweepError) as caught:
            _sweep(directory, command, [fault]).run()
        self.assertIn("occurs 2 time(s)", str(caught.exception))

    def test_a_replacement_identical_to_its_anchor_is_refused(self):
        """A fault that changes nothing would be reported as a defect nobody caught."""
        directory, subject, command = _bench(self)
        fault = Fault("no-op fault", "threshold", subject, "THRESHOLD = 0.70", "THRESHOLD = 0.70")
        with self.assertRaises(SweepError) as caught:
            _sweep(directory, command, [fault]).run()
        self.assertIn("did not change", str(caught.exception))

    def test_a_control_run_that_is_already_red_invalidates_the_sweep(self):
        """Every row below a red control is meaningless, so there are no rows."""
        directory, subject, command = _bench(self)
        subject.write_text("THRESHOLD = 0.11\n", encoding="utf-8", newline="\n")
        fault = Fault("anything", "threshold", subject, "0.11", "0.22")
        with self.assertRaises(SweepError) as caught:
            _sweep(directory, command, [fault]).run()
        self.assertIn("already FAILING", str(caught.exception))


class RedSweepVerdicts(unittest.TestCase):
    """The harness distinguishes the outcomes that look alike in a bad sweep."""

    def test_a_caught_fault_is_red_and_the_tree_is_restored(self):
        directory, subject, command = _bench(self)
        before = subject.read_bytes()
        fault = Fault("threshold moves", "threshold", subject, "0.70", "0.55")
        sweep = _sweep(directory, command, [fault])
        self.assertEqual(sweep.run(), 0)
        self.assertEqual(sweep.results[0].verdict, "RED")
        self.assertIn("threshold", sweep.results[0].fired)
        self.assertEqual(subject.read_bytes(), before, "the sweep left the tree modified")

    def test_a_gate_that_passes_with_the_defect_in_is_not_caught(self):
        """The honest negative: the check exists, ran, and did not notice."""
        directory, subject, command = _bench(self)
        subject.write_text(_SUBJECT + "UNUSED = 1\n", encoding="utf-8", newline="\n")
        fault = Fault("an ignored constant", "threshold", subject, "UNUSED = 1", "UNUSED = 2")
        sweep = _sweep(directory, command, [fault])
        self.assertEqual(sweep.run(), 1)
        self.assertEqual(sweep.results[0].verdict, "NOT CAUGHT")

    def test_the_wrong_check_firing_is_not_a_catch(self):
        """A gate going red for an unrelated reason has not demonstrated this check."""
        directory, subject, command = _bench(self)
        fault = Fault("threshold moves", "some-other-check", subject, "0.70", "0.55")
        sweep = _sweep(directory, command, [fault])
        self.assertEqual(sweep.run(), 1)
        self.assertEqual(sweep.results[0].verdict, "WRONG CHECK")

    def test_a_gate_that_cannot_be_launched_is_a_verdict_not_a_catch(self):
        """L1: a harness error is reported as one, never as evidence."""
        directory, subject, _ = _bench(self)
        fault = Fault("threshold moves", "threshold", subject, "0.70", "0.55")
        sweep = _sweep(directory, [str(directory / "no-such-binary")], [fault])
        self.assertRaises(SweepError, sweep.run)


class RedSweepBytecode(unittest.TestCase):
    """L25's own defect: the fault that never reached the interpreter."""

    def test_a_same_length_fault_is_not_compiled_away(self):
        """The regression test for the observed defect.

        The control run imports `subject` and leaves a `.pyc` behind. The fault
        below replaces four characters with four characters, so the cached module
        can still satisfy CPython's `(mtime, size)` check within the same second.
        Without the harness's bytecode discard the gate re-imports the *old*
        constant, passes, and the sweep reports a defect nobody caught.
        """
        directory, subject, command = _bench(self)
        fault = Fault("same-length threshold", "threshold", subject, "0.70", "0.55")
        self.assertEqual(len("0.70"), len("0.55"), "the premise of this test")

        sweep = _sweep(directory, command, [fault])
        self.assertEqual(sweep.run(), 0, "a same-length fault was compiled away")
        self.assertEqual(sweep.results[0].verdict, "RED")

    def test_discarding_bytecode_removes_what_a_run_leaves_behind(self):
        directory, _, command = _bench(self)
        sweep = _sweep(directory, command, [])
        sweep.run_gate()
        cached = list(directory.rglob("*.pyc"))
        self.assertTrue(cached, "the gate did not leave bytecode; the premise is gone")
        self.assertGreaterEqual(sweep.discard_bytecode(), len(cached))
        self.assertFalse(list(directory.rglob("*.pyc")))


class RedSweepTables(unittest.TestCase):
    """The committed fault tables are loadable, anchored, and about real gates."""

    def _tables(self):
        return sorted((_REPO_ROOT / "tools" / "testing" / "faults").glob("*.json"))

    def test_every_committed_table_loads_and_anchors_exactly_once(self):
        tables = self._tables()
        self.assertGreaterEqual(len(tables), 3, "the standing evidence has gone missing")
        for table in tables:
            with self.subTest(table=table.name):
                command, faults = load_table(table)
                self.assertTrue(faults, f"{table.name} declares no faults")
                self.assertTrue(Path(command[-1]).name.endswith(".py"))
                for fault in faults:
                    text = fault.path.read_text(encoding="utf-8")
                    self.assertEqual(
                        fault.anchor_count(text),
                        1,
                        f"{table.name}: {fault.label!r} anchors "
                        f"{fault.anchor_count(text)} time(s) in {fault.path.name} -- the "
                        "table has drifted from the code it injects into",
                    )
                    self.assertNotEqual(
                        fault.old, fault.new, f"{table.name}: {fault.label!r} changes nothing"
                    )

    def test_a_malformed_table_is_refused_rather_than_partly_run(self):
        directory = Path(tempfile.mkdtemp(prefix="bcir-red-table-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
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
            with self.subTest(table=name):
                self.assertRaises(SweepError, load_table, path)


if __name__ == "__main__":
    unittest.main()
