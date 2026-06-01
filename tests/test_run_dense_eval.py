import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_dense_eval.sh"
PARSER = REPO_ROOT / "scripts" / "parse_eval_ppl_log.py"
FIXTURE_EMITTER = REPO_ROOT / "tests" / "fixtures" / "emit_eval_ppl_fixture.py"
FIXTURE_LOG = REPO_ROOT / "tests" / "fixtures" / "eval_ppl_sample.txt"


class RunDenseEvalTest(unittest.TestCase):
    def run_command(self, command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_parser_extracts_wikitext2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dense_fixture" / "evaluation_metrics.csv"
            result = self.run_command(
                [
                    "python",
                    "-B",
                    str(PARSER),
                    "--log",
                    str(FIXTURE_LOG),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"run_id": "dense_fixture", "dataset": "wikitext2", "ppl": "5.42"}])

    def test_launcher_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "dense_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER), "--dry-run"],
                {
                    "OUTPUT_DIR": str(output_dir),
                    "EXPERIMENT_LOG": str(experiment_log),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--eval_datasets wikitext2", result.stdout)
            self.assertIn("--sequence_length 2048", result.stdout)
            self.assertFalse(output_dir.exists())
            self.assertFalse(experiment_log.exists())

    def test_launcher_fixture_writes_artifacts_and_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "dense_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "EVAL_PPL_SCRIPT": str(FIXTURE_EMITTER),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "dense_fixture",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in ["command.sh", "run.log", "runtime.txt", "evaluation_metrics.csv"]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["method"], "dense")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["wikitext2_ppl"], "5.42")
            self.assertEqual(row["train_ppl"], "")

    def test_launcher_failure_preserves_artifacts_and_failed_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "dense_failure"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "EVAL_PPL_SCRIPT": str(temp_path / "missing_eval_ppl.py"),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "dense_failure",
                },
            )

            self.assertNotEqual(result.returncode, 0)
            for filename in ["command.sh", "run.log", "runtime.txt"]:
                self.assertTrue((output_dir / filename).is_file(), filename)
            self.assertFalse((output_dir / "evaluation_metrics.csv").exists())

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("last_successful_step=dense_evaluation_process_started", rows[0]["notes"])


if __name__ == "__main__":
    unittest.main()
