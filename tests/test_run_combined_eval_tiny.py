import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_combined_eval_tiny.sh"
FIXTURE_EMITTER = REPO_ROOT / "tests" / "fixtures" / "emit_eval_ppl_fixture.py"


class RunCombinedEvalTinyTest(unittest.TestCase):
    def run_command(self, command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_launcher_dry_run_is_side_effect_free_and_includes_compression_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "combined_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER), "--dry-run"],
                {
                    "OUTPUT_DIR": str(output_dir),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "DROP_LAYER_CONFIG": "drop-config.txt",
                    "SPARSE_WEIGHTS_PATH": "sparse-db",
                    "SPARSE_CONFIG_PATH": "sparse-config.txt",
                    "SPARSE_DEFAULT_LEVEL": "1",
                    "RUN_ID": "combined_fixture",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--eval_datasets wikitext2", result.stdout)
            self.assertIn("--sequence_length 1024", result.stdout)
            self.assertIn("--eval_tokens 4096", result.stdout)
            self.assertIn("--sparse_weights_path sparse-db", result.stdout)
            self.assertIn("--sparse_config_path sparse-config.txt", result.stdout)
            self.assertIn("--sparse_default_level 1", result.stdout)
            self.assertIn("--drop_layer_config drop-config.txt", result.stdout)
            self.assertFalse(output_dir.exists())
            self.assertFalse(experiment_log.exists())

    def test_launcher_fixture_writes_artifacts_and_log_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs" / "combined_fixture"
            experiment_log = temp_path / "results" / "experiment_log.csv"
            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "CHECK_RUNTIME_DEPENDENCIES": "0",
                    "EVAL_PPL_SCRIPT": str(FIXTURE_EMITTER),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "OUTPUT_DIR": str(output_dir),
                    "RUN_ID": "combined_fixture",
                    "METHOD": "combined_depth_sparse_eval",
                    "DROP_LAYER_CONFIG": "drop-config.txt",
                    "SPARSE_WEIGHTS_PATH": "sparse-db",
                    "SPARSE_DEFAULT_LEVEL": "0",
                    "SPARSITY_OR_BITS": "depth0.125+sparse0.50",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in [
                "command.sh",
                "run.log",
                "runtime.txt",
                "evaluation_metrics.csv",
                "combined_config_summary.md",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            summary = (output_dir / "combined_config_summary.md").read_text(encoding="utf-8")
            self.assertIn("- method: `combined_depth_sparse_eval`", summary)
            self.assertIn("- sparse_weights_path: `sparse-db`", summary)

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["method"], "combined_depth_sparse_eval")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["wikitext2_ppl"], "5.42")
            self.assertEqual(row["sparsity_or_bits"], "depth0.125+sparse0.50")
            self.assertIn("drop_layer_config=drop-config.txt", row["notes"])
            self.assertIn("sparse_weights_path=sparse-db", row["notes"])


if __name__ == "__main__":
    unittest.main()
