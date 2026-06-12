import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent
GENERATOR = TRAINING_ROOT / "tools" / "generate-exercise-variants.py"
FIXED_SEEDS = (294, 20260612)


class GenerateExerciseVariantsTests(unittest.TestCase):
    def generate(self, seed: int, directory: Path, suffix: str) -> tuple[bytes, dict]:
        output = directory / f"variants-{seed}-{suffix}.jsonl"
        report = directory / f"report-{seed}-{suffix}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--seed",
                str(seed),
                "--budget",
                "5",
                "--output",
                str(output),
                "--report",
                str(report),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return output.read_bytes(), json.loads(report.read_text(encoding="utf-8"))

    def test_fixed_seed_set_is_byte_for_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            for seed in FIXED_SEEDS:
                first, first_report = self.generate(seed, temp, "first")
                second, second_report = self.generate(seed, temp, "second")
                self.assertEqual(first, second, f"seed {seed} changed records")
                self.assertEqual(first_report, second_report, f"seed {seed} changed report")

                records = [json.loads(line) for line in first.decode("utf-8").splitlines()]
                self.assertEqual(len(records), 5)
                self.assertEqual(first_report["accepted"], 5)
                self.assertEqual(first_report["attempted"], first_report["accepted"] + first_report["rejected"])
                self.assertEqual(
                    {record["generator"]["seed"] for record in records},
                    {seed},
                )
                self.assertTrue(all(record["oracle_result"]["score_percent"] == 100.0 for record in records))
                self.assertTrue(all(record["grading_manifest"]["safe_deterministic_lli"] is True for record in records))
                self.assertTrue(all(record["review"]["markdown_solution_generated"] is False for record in records))
                self.assertEqual(len({record["artifacts"]["reference_solution_sha256"] for record in records}), 5)
                self.assertEqual(len({record["artifacts"]["normalized_ir_structure_sha256"] for record in records}), 5)
                if seed == 294:
                    self.assertGreater(first_report["rejected"], 0)
                    self.assertIn("constants cancel and make the task semantically trivial", first_report["rejection_counts"])

    def test_existing_records_seed_all_deduplication_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            base, _ = self.generate(294, temp, "base")
            existing = temp / "existing.jsonl"
            existing.write_bytes(base)
            output = temp / "next.jsonl"
            report = temp / "next-report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--seed",
                    "294",
                    "--budget",
                    "1",
                    "--family",
                    "scalar-add",
                    "--existing",
                    str(existing),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["existing_records"], 5)
            self.assertEqual(summary["accepted"], 1)
            self.assertGreaterEqual(summary["rejection_counts"]["semantic-equivalence group already exists"], 1)

    def test_seed_is_required_and_budget_is_capped(self) -> None:
        missing_seed = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(missing_seed.returncode, 0)
        self.assertIn("--seed", missing_seed.stderr)

        over_budget = subprocess.run(
            [sys.executable, str(GENERATOR), "--seed", "1", "--budget", "6"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(over_budget.returncode, 0)
        self.assertIn("reviewed maximum", over_budget.stderr)


if __name__ == "__main__":
    unittest.main()
