import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent
RUNNER = TRAINING_ROOT / "tools" / "run-eval.py"


class RunEvalTests(unittest.TestCase):
    def run_eval(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_reference_run_writes_contract_reports_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "run"
            result = self.run_eval(
                "run", "--output-dir", str(output), "--exercise", "001",
                "--fixture-adapter", "reference",
                "--generation-parameters", '{"seed": 7}',
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            generator_input = json.loads((output / "prepared/001/input.json").read_text())
            self.assertEqual(generator_input["exercise_id"], "001")
            self.assertIn("prompt_text", generator_input)
            self.assertIsInstance(generator_input["context_paths"], list)
            self.assertEqual(generator_input["generation_parameters"], {"seed": 7})
            self.assertEqual(generator_input["output_contract"]["answer_kind"], "llvm-ir")
            serialized_input = json.dumps(generator_input)
            self.assertNotIn(".solution.", serialized_input)
            self.assertNotIn("reference_solution", serialized_input)

            lines = (output / "results.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["outcome"], "passed")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["pass_rate"], 1.0)
            self.assertEqual(summary["mean_score"], 100.0)
            self.assertEqual(summary["outcomes"]["infrastructure_failures"], 0)
            self.assertIn("beginner", summary["score_by_difficulty"])
            self.assertIn("standalone-llvm-ir", summary["score_by_topic"])

            manifest = json.loads((output / "reproducibility-manifest.json").read_text())
            self.assertEqual(manifest["exercise_manifest"]["schema_version"], "1.0")
            self.assertEqual(manifest["generator"]["identity"], "reference")
            self.assertTrue(manifest["attempt_hashes"]["001"]["sha256"])

            attempt = output / "attempts/001/answer.ll"
            before = attempt.stat().st_mtime_ns
            time.sleep(0.01)
            resumed = self.run_eval(
                "generate", "--output-dir", str(output), "--fixture-adapter", "reference"
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            self.assertIn("[resume] 001", resumed.stdout)
            self.assertEqual(attempt.stat().st_mtime_ns, before)

    def test_empty_and_partial_are_quality_results_not_infrastructure_failures(self) -> None:
        for adapter in ("empty", "partial"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as temp_name:
                output = Path(temp_name) / "run"
                result = self.run_eval(
                    "run", "--output-dir", str(output), "--exercise", "001",
                    "--fixture-adapter", adapter,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["outcomes"]["infrastructure_failures"], 0)
                self.assertEqual(summary["outcomes"]["quality_failures"], 1)
                self.assertLess(summary["mean_score"], 100.0)

    def test_local_command_adapter_uses_json_contract_without_a_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            adapter = root / "adapter.py"
            adapter.write_text(
                """import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input")
p.add_argument("--output")
a = p.parse_args()
request = json.loads(Path(a.input).read_text())
answer = Path(request["output_contract"]["primary_artifact_path"])
answer.parent.mkdir(parents=True, exist_ok=True)
answer.write_text("define i32 @add(i32 %a, i32 %b) {\\nentry:\\n  %sum = add i32 %a, %b\\n  ret i32 %sum\\n}\\n")
response = {
    "generated_artifact_paths": [str(answer)],
    "model": {"identity": "test-adapter"},
    "token_counts": {"input": 1, "output": 1, "total": 2},
    "status": "completed",
}
Path(a.output).write_text(json.dumps(response))
""",
                encoding="utf-8",
            )
            output = root / "run with spaces"
            command = f"{shlex_quote(sys.executable)} {shlex_quote(str(adapter))} --input {{input}} --output {{output}}"
            result = self.run_eval(
                "run", "--output-dir", str(output), "--exercise", "001",
                "--generator-command", command, "--generator-id", "test-local",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            generated = json.loads((output / "attempts/001/generator-output.json").read_text())
            self.assertEqual(generated["model"]["identity"], "test-adapter")
            self.assertEqual(generated["token_counts"]["total"], 2)
            self.assertEqual(json.loads((output / "summary.json").read_text())["pass_rate"], 1.0)


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


if __name__ == "__main__":
    unittest.main()
