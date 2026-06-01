import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_drop_search.sh"
GRID_SCRIPT = REPO_ROOT / "scripts" / "run_drop_search_grid.sh"
FIXTURE_EMITTER = REPO_ROOT / "tests" / "fixtures" / "emit_depth_search_fixture.py"


class RunDropSearchScriptsTest(unittest.TestCase):
    def run_script(self, script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), *args],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def test_single_run_fixture_writes_artifacts_and_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "depth_fixture_s0.375_seed1"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_script(
                SINGLE_RUN_SCRIPT,
                env={
                    "EVO_DROP_SEARCH_SCRIPT": str(FIXTURE_EMITTER),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "depth_fixture_s0.375_seed1",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("status=completed", result.stdout)
            for filename in [
                "command.sh",
                "run.log",
                "runtime.txt",
                "layer_drop_config.txt",
                "generation_metrics.csv",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            with (output_dir / "generation_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 6)
            self.assertEqual(metric_rows[-1]["phase"], "final")
            self.assertEqual(metric_rows[-1]["wikitext2_ppl"], "85.00")

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                experiment_rows = list(csv.DictReader(handle))
            self.assertEqual(len(experiment_rows), 1)
            row = experiment_rows[0]
            self.assertEqual(row["run_id"], "depth_fixture_s0.375_seed1")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["wikitext2_ppl"], "85.00")
            self.assertEqual(row["train_ppl"], "8.40e+01")
            self.assertIn("dropped_attn_modules=2", row["notes"])
            self.assertIn("dropped_mlp_modules=2", row["notes"])

    def test_grid_dry_run_prepares_four_commands_without_log_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_script(
                GRID_SCRIPT,
                "--dry-run",
                env={
                    "OUTPUTS_ROOT": str(temp_path / "outputs"),
                    "EXPERIMENT_LOG": str(experiment_log),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No experiments were launched", result.stdout)
            self.assertFalse(experiment_log.exists())
            self.assertFalse((temp_path / "outputs").exists())

            for sparsity in ["0.125", "0.25", "0.375", "0.50"]:
                self.assertIn(f"--sparsity {sparsity}", result.stdout)
                self.assertIn(f"depth_mistral7b_s{sparsity}_seed1", result.stdout)
            self.assertIn("--generations 10", result.stdout)
            self.assertIn("--offspring 8", result.stdout)
            self.assertIn("--tokens_per_selection 512 2048", result.stdout)
            self.assertNotIn("--drop_entire_block", result.stdout)

    def test_single_run_failure_preserves_artifacts_and_failed_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "depth_fixture_failure"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_script(
                SINGLE_RUN_SCRIPT,
                env={
                    "EVO_DROP_SEARCH_SCRIPT": str(temp_path / "missing_evo_drop_search.py"),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "depth_fixture_failure",
                },
            )

            self.assertNotEqual(result.returncode, 0)
            for filename in ["command.sh", "run.log", "runtime.txt"]:
                self.assertTrue((output_dir / filename).is_file(), filename)
            self.assertFalse((output_dir / "generation_metrics.csv").exists())

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                experiment_rows = list(csv.DictReader(handle))
            self.assertEqual(len(experiment_rows), 1)
            row = experiment_rows[0]
            self.assertEqual(row["run_id"], "depth_fixture_failure")
            self.assertEqual(row["status"], "failed")
            self.assertIn("last_successful_step=depth_search_process_started", row["notes"])
            self.assertIn("command_exit_code=", row["notes"])


if __name__ == "__main__":
    unittest.main()
