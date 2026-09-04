import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent
GRADER = TRAINING_ROOT / "tools" / "grade-exercises.py"
RUN_EVAL = TRAINING_ROOT / "tools" / "run-eval.py"
SAFE_PROCESS = TRAINING_ROOT / "tools" / "safe_process.py"

spec = importlib.util.spec_from_file_location("safe_process", SAFE_PROCESS)
safe_process = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = safe_process
assert spec.loader is not None
spec.loader.exec_module(safe_process)


class SecurityTests(unittest.TestCase):
    def run_python(self, script: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def test_answer_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            attempts = root / "attempts"
            attempts.mkdir()
            outside = root / "outside.md"
            outside.write_text("atomic ordering", encoding="utf-8")
            result = self.run_python(
                GRADER,
                "--exercise",
                "025",
                "--answer",
                str(attempts / ".." / "outside.md"),
                "--attempt-root",
                str(attempts),
                "--format",
                "json",
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["error"]["outcome"], "invalid_answer")
            self.assertIn("escapes configured attempt root", result.stdout)

    def test_bounded_capture_and_timeout(self) -> None:
        flooded = safe_process.run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"],
            timeout=5,
            max_output_bytes=1024,
        )
        self.assertEqual(flooded.returncode, 0)
        self.assertTrue(flooded.output_truncated)
        self.assertLess(len(flooded.stdout), 1200)

        timed_out = safe_process.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05
        )
        self.assertEqual(timed_out.returncode, 124)
        self.assertTrue(timed_out.timed_out)
        self.assertIn("timed out", timed_out.stderr)

    def test_malformed_manifest_is_a_clean_grader_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            manifests = root / "manifests"
            manifests.mkdir()
            (manifests / "001.json").write_text("{not-json", encoding="utf-8")
            result = self.run_python(
                RUN_EVAL,
                "prepare",
                "--output-dir",
                str(root / "output"),
                "--exercise",
                "001",
                "--manifests-dir",
                str(manifests),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("run-eval: error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_shell_metacharacters_in_answer_filename_are_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            marker = root / "PWNED"
            answer = root / "answer;touch PWNED.md"
            answer.write_text("calling convention attributes ABI contract", encoding="utf-8")
            result = self.run_python(
                GRADER,
                "--exercise",
                "025",
                "--answer",
                str(answer),
                "--attempt-root",
                str(root),
                "--format",
                "json",
            )
            self.assertIn(result.returncode, {0, 1})
            self.assertFalse(marker.exists())
            report = json.loads(result.stdout)
            self.assertEqual(report["results"][0]["answer"], str(answer.resolve()))


if __name__ == "__main__":
    unittest.main()
