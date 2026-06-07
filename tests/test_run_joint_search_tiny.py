import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_joint_search_tiny.sh"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "emit_joint_search_fixture.py"


class RunJointSearchTinyTest(unittest.TestCase):
    def run_command(self, command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_dry_run_matches_joint_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs" / "joint"
            result = self.run_command([str(LAUNCHER), "--dry-run"], {"OUTPUT_DIR": str(output_dir)})

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--drop_sparsity 0.125", result.stdout)
            self.assertIn("--target_bitwidth 3.0", result.stdout)
            self.assertIn("--generations 10", result.stdout)
            self.assertIn("--offspring 8", result.stdout)
            self.assertIn("--calibration_sequence_length 1024", result.stdout)
            self.assertFalse(output_dir.exists())

    def test_fixture_run_writes_joint_artifacts_and_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "joint"
            quant_db = temp_path / "quant_db"
            module_dir = quant_db / "model.layers.0.self_attn.q_proj"
            module_dir.mkdir(parents=True)
            (module_dir / "3.pth").write_bytes(b"fixture")
            experiment_log = temp_path / "results" / "experiment_log.csv"

            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "CHECK_RUNTIME_DEPENDENCIES": "0",
                    "PYTHON_BIN": sys.executable,
                    "EVO_JOINT_SEARCH_SCRIPT": str(FIXTURE),
                    "QUANT_WEIGHTS_PATH": str(quant_db),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "joint_fixture",
                    "GENERATIONS": "2",
                    "MEMORY_POLL_INTERVAL_SECONDS": "0.1",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in [
                "command.sh",
                "run.log",
                "runtime.txt",
                "generation_metrics.csv",
                "joint_drop_config.txt",
                "joint_quant_config.txt",
                "joint_config.json",
                "memory_samples.csv",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["method"], "combined_depth_quant_search")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["wikitext2_ppl"], "11.70")
            self.assertEqual(row["train_ppl"], "12.00")
            self.assertIn("actual_average_bitwidth=3.0000e+00", row["notes"])
            self.assertIn("dropped_attn_modules=1", row["notes"])
            self.assertIn("dropped_mlp_modules=1", row["notes"])

    def test_dry_run_supports_active_quant_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs" / "joint"
            result = self.run_command(
                [str(LAUNCHER), "--dry-run"],
                {
                    "OUTPUT_DIR": str(output_dir),
                    "ACTIVE_QUANT_BUDGET": "1",
                    "GROUP_RULE": "size",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--group_rule size", result.stdout)
            self.assertIn("--active_quant_budget", result.stdout)


if __name__ == "__main__":
    unittest.main()
