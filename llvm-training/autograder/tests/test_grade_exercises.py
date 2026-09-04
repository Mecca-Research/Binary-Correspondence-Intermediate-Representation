import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent
GRADER = TRAINING_ROOT / "tools" / "grade-exercises.py"
FIXTURES = TRAINING_ROOT / "autograder" / "fixtures" / "incomplete" / "attempts"


class GradeExercisesTests(unittest.TestCase):
    def run_grader(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(GRADER), *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

    def test_checked_in_references_meet_expected_executed_score(self) -> None:
        result = self.run_grader("--self-test", "--format", "json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertTrue(report["results"])
        for exercise in report["results"]:
            self.assertEqual(exercise["executed_score_percent"], 100.0, exercise["exercise_id"])

    def test_incomplete_fixtures_receive_partial_credit_and_failures(self) -> None:
        result = self.run_grader("--attempts", str(FIXTURES), "--format", "json")
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        for exercise in report["results"]:
            self.assertGreater(exercise["points_earned"], 0, exercise["exercise_id"])
            self.assertLess(
                exercise["points_earned"], exercise["points_available"], exercise["exercise_id"]
            )
            self.assertTrue(any(check["status"] == "fail" for check in exercise["checks"]))

    def test_missing_optional_tools_reduce_confidence_without_inflating_score(self) -> None:
        answer = REPO_ROOT / "llvm-training" / "exercises" / "001-add.solution.ll"
        result = self.run_grader(
            "--exercise",
            "001",
            "--answer",
            str(answer),
            "--format",
            "json",
            env={"PATH": ""},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        exercise = report["results"][0]
        self.assertGreater(exercise["skipped_checks"], 0)
        self.assertEqual(exercise["score_confidence"], "reduced")
        self.assertLess(exercise["score_percent"], exercise["executed_score_percent"])
        self.assertEqual(report["aggregate"]["score_confidence"], "reduced")

    def test_single_answer_and_output_file(self) -> None:
        answer = REPO_ROOT / "llvm-training" / "exercises" / "013-mixed-stride-indexing.solution.ll"
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "score.json"
            result = self.run_grader(
                "--exercise",
                "013",
                "--answer",
                str(answer),
                "--format",
                "json",
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["exercise_id"] for item in report["results"]], ["013"])


if __name__ == "__main__":
    unittest.main()
