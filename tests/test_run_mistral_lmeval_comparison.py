import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_mistral_lmeval_comparison.sh"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "emit_lmeval_fixture.py"


class RunMistralLmEvalComparisonTest(unittest.TestCase):
    def run_command(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_default_dry_run_builds_downstream_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_command(
                [str(LAUNCHER), "--dry-run"],
                {
                    "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                    "EXPERIMENT_LOG": str(
                        Path(temp_dir) / "experiment_log.csv"
                    ),
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lmeval_dense_mistral_tasks_seed0", result.stdout)
        for seed in range(3):
            self.assertIn(
                f"lmeval_depth_mistral_s0.25_tasks_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "lmeval_independent_depth_quant_mistral_"
                f"s0.25_qproj3.0_tasks_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "lmeval_joint_g50_mistral_"
                f"s0.25_qproj3.0_tasks_seed{seed}",
                result.stdout,
            )
        self.assertIn("--tasks", result.stdout)
        self.assertIn("arc_easy", result.stdout)
        self.assertIn("piqa", result.stdout)
        self.assertIn("winogrande", result.stdout)
        self.assertIn("--quant_weights_path", result.stdout)
        self.assertIn("--drop_layer_config", result.stdout)
        self.assertIn("Mistral LM-eval dry run complete", result.stdout)

    def test_fixture_run_writes_artifacts_and_experiment_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            outputs_root = temp_path / "outputs"
            results_root = temp_path / "runs"
            experiment_log = temp_path / "experiment_log.csv"
            source_root = temp_path / "sources"
            quant_db = temp_path / "quant_db"
            quant_db.mkdir()
            for seed in (0,):
                depth_dir = (
                    source_root
                    / f"thesis_medium_depth_mistral_s0.25_g20_o16_seed{seed}"
                )
                quant_dir = (
                    source_root
                    / f"thesis_medium_quant_mistral_qproj3.0_g20_o16_seed{seed}"
                )
                joint_dir = (
                    source_root
                    / (
                        "thesis_compute_matched_joint_mistral_"
                        f"s0.25_qproj3.0_g50_o16_seed{seed}"
                    )
                )
                depth_dir.mkdir(parents=True)
                quant_dir.mkdir(parents=True)
                joint_dir.mkdir(parents=True)
                (depth_dir / "layer_drop_config.txt").write_text(
                    "none\n", encoding="utf-8"
                )
                (quant_dir / "quant_configuration.txt").write_text(
                    "model.layers.0.self_attn.q_proj: 3\n",
                    encoding="utf-8",
                )
                (joint_dir / "joint_drop_config.txt").write_text(
                    "none\n", encoding="utf-8"
                )
                (joint_dir / "joint_quant_config.txt").write_text(
                    "model.layers.0.self_attn.q_proj: 3\n",
                    encoding="utf-8",
                )

            result = self.run_command(
                [str(LAUNCHER)],
                {
                    "CHECK_RUNTIME_DEPENDENCIES": "0",
                    "METHODS": "joint_g50",
                    "SEEDS": "0",
                    "OUTPUTS_ROOT": str(outputs_root),
                    "RESULTS_RUNS_ROOT": str(results_root),
                    "SOURCE_RUNS_ROOT": str(source_root),
                    "EXPERIMENT_LOG": str(experiment_log),
                    "QUANT_WEIGHTS_PATH": str(quant_db),
                    "LMEVAL_SCRIPT": str(FIXTURE),
                    "PYTHON_BIN": sys.executable,
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = (
                outputs_root
                / "lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed0"
            )
            for filename in [
                "command.sh",
                "run.log",
                "runtime.txt",
                "lmeval_results.json",
                "lmeval_config_summary.md",
            ]:
                self.assertTrue((run_dir / filename).is_file(), filename)
            for filename in [
                "command.sh",
                "runtime.txt",
                "lmeval_results.json",
                "lmeval_config_summary.md",
            ]:
                self.assertTrue(
                    (results_root / run_dir.name / filename).is_file(),
                    filename,
                )

            with experiment_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "lmeval_joint_g50")
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(
                rows[0]["sparsity_or_bits"],
                "depth0.25+qproj3.0_joint_g50",
            )
            self.assertIn("tasks=arc_easy,piqa,winogrande", rows[0]["notes"])

    def test_invalid_method_is_rejected(self) -> None:
        result = self.run_command(
            [str(LAUNCHER), "--dry-run"],
            {"METHODS": "unknown"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported method", result.stderr)


if __name__ == "__main__":
    unittest.main()
