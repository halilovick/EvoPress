import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_gptq_tiny_debug.sh"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "emit_quant_db_fixture.py"


class RunGptqTinyDebugTest(unittest.TestCase):
    def run_command(self, command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_dry_run_uses_planned_quantization_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "quant_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER), "--dry-run"],
                {
                    "OUTPUT_DIR": str(output_dir),
                    "EXPERIMENT_LOG": str(experiment_log),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--calibration_tokens 4096", result.stdout)
            self.assertIn("--calibration_sequence_length 1024", result.stdout)
            self.assertIn("--bitwidth_options 2 3 4", result.stdout)
            self.assertIn("--calibration_bitwidth 3", result.stdout)
            self.assertIn("--group_size 128", result.stdout)
            self.assertFalse(output_dir.exists())
            self.assertFalse(experiment_log.exists())

    def test_fixture_run_validates_database_and_logs_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "quant_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "CHECK_RUNTIME_DEPENDENCIES": "0",
                    "PYTHON_BIN": sys.executable,
                    "TORCHRUN_BIN": str(FIXTURE),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "quant_fixture",
                    "MEMORY_POLL_INTERVAL_SECONDS": "0.1",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in [
                "command.sh",
                "run.log",
                "runtime.txt",
                "memory_samples.csv",
                "quant_db_summary.txt",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            summary = (output_dir / "quant_db_summary.txt").read_text(encoding="utf-8")
            self.assertIn("generated_module_dirs=2", summary)
            self.assertIn("generated_weight_files=6", summary)
            self.assertIn("missing_expected_weight_files=0", summary)

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["method"], "quant_db")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["sparsity_or_bits"], "2 3 4")
            self.assertIn("generated_module_dirs=2", row["notes"])
            self.assertIn("generated_weight_files=6", row["notes"])


if __name__ == "__main__":
    unittest.main()
