import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_depth_baseline.sh"
GRID_SCRIPT = REPO_ROOT / "scripts" / "run_depth_baseline_grid.sh"
FIXTURE_EMITTER = REPO_ROOT / "tests" / "fixtures" / "emit_depth_baseline_fixture.py"
REFERENCE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "depth_reference_config.txt"


class RunDepthBaselineScriptsTest(unittest.TestCase):
    def run_script(self, script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), *args],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def test_single_run_fixture_writes_artifacts_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "baseline_random_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            baseline_log = temp_path / "results" / "depth_baseline_runs.csv"
            result = self.run_script(
                SINGLE_RUN_SCRIPT,
                env={
                    "BASELINE_EVAL_SCRIPT": str(FIXTURE_EMITTER),
                    "REFERENCE_CONFIG": str(REFERENCE_CONFIG),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "BASELINE_RESULTS_LOG": str(baseline_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "baseline_random_fixture",
                    "METHOD": "random",
                    "SPARSITY": "0.125",
                    "SEED": "2",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("status=completed", result.stdout)
            for filename in ["command.sh", "run.log", "runtime.txt", "layer_drop_config.txt", "baseline_metrics.csv"]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                experiment_rows = list(csv.DictReader(handle))
            self.assertEqual(len(experiment_rows), 1)
            experiment_row = experiment_rows[0]
            self.assertEqual(experiment_row["method"], "depth_baseline_random")
            self.assertEqual(experiment_row["status"], "completed")
            self.assertEqual(experiment_row["wikitext2_ppl"], "12.34")
            self.assertEqual(experiment_row["train_ppl"], "1.11e+01")
            self.assertIn("dropped_attn_modules=2", experiment_row["notes"])
            self.assertIn("dropped_mlp_modules=2", experiment_row["notes"])
            self.assertIn("calibration_seed=1", experiment_row["notes"])

            with baseline_log.open(newline="", encoding="utf-8") as handle:
                baseline_rows = list(csv.DictReader(handle))
            self.assertEqual(len(baseline_rows), 1)
            self.assertEqual(baseline_rows[0]["method"], "random")
            self.assertIn("status=completed", baseline_rows[0]["notes"])

    def test_single_run_failure_preserves_artifacts_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "baseline_failure"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            baseline_log = temp_path / "results" / "depth_baseline_runs.csv"
            result = self.run_script(
                SINGLE_RUN_SCRIPT,
                env={
                    "BASELINE_EVAL_SCRIPT": str(temp_path / "missing_evaluator.py"),
                    "REFERENCE_CONFIG": str(REFERENCE_CONFIG),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "BASELINE_RESULTS_LOG": str(baseline_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "baseline_failure",
                },
            )

            self.assertNotEqual(result.returncode, 0)
            for filename in ["command.sh", "run.log", "runtime.txt"]:
                self.assertTrue((output_dir / filename).is_file(), filename)
            with experiment_log.open(newline="", encoding="utf-8") as handle:
                experiment_rows = list(csv.DictReader(handle))
            self.assertEqual(experiment_rows[0]["status"], "failed")
            with baseline_log.open(newline="", encoding="utf-8") as handle:
                baseline_rows = list(csv.DictReader(handle))
            self.assertIn("status=failed", baseline_rows[0]["notes"])

    def test_grid_dry_run_prepares_required_baselines_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = self.run_script(
                GRID_SCRIPT,
                "--dry-run",
                env={
                    "OUTPUTS_ROOT": str(temp_path / "outputs"),
                    "EXPERIMENT_LOG": str(temp_path / "experiment_log.csv"),
                    "BASELINE_RESULTS_LOG": str(temp_path / "depth_baseline_runs.csv"),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No experiments were launched", result.stdout)
            self.assertFalse((temp_path / "outputs").exists())
            self.assertEqual(result.stdout.count("=== Preparing depth baseline:"), 16)
            for sparsity in ["0.125", "0.25", "0.375", "0.50"]:
                for seed in ["1", "2", "3"]:
                    self.assertIn(f"baseline_random_mistral7b_s{sparsity}_seed{seed}", result.stdout)
                self.assertIn(f"baseline_late_layer_mistral7b_s{sparsity}_seed1", result.stdout)
            self.assertNotIn("baseline_early_layer", result.stdout)

    def test_grid_can_skip_existing_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dir = temp_path / "outputs" / "baseline_random_mistral7b_s0.125_seed1"
            existing_dir.mkdir(parents=True)
            (existing_dir / "run.log").write_text("preserved\n", encoding="utf-8")
            result = self.run_script(
                GRID_SCRIPT,
                "--dry-run",
                "--skip-existing",
                env={
                    "SPARSITIES": "0.125",
                    "RANDOM_SEEDS": "1",
                    "OUTPUTS_ROOT": str(temp_path / "outputs"),
                    "EXPERIMENT_LOG": str(temp_path / "experiment_log.csv"),
                    "BASELINE_RESULTS_LOG": str(temp_path / "depth_baseline_runs.csv"),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Skipping existing output directory", result.stdout)
            self.assertEqual((existing_dir / "run.log").read_text(encoding="utf-8"), "preserved\n")


if __name__ == "__main__":
    unittest.main()
