import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_mistral_medium_grid.sh"


class RunMistralMediumGridTest(unittest.TestCase):
    def test_default_dry_run_prepares_dense_and_three_seed_search_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                    "EXPERIMENT_LOG": str(Path(temp_dir) / "experiment_log.csv"),
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("thesis_medium_dense_mistral_seq1024_seed0", result.stdout)
            for seed in range(3):
                self.assertIn(
                    f"thesis_medium_depth_mistral_s0.25_g20_o16_seed{seed}",
                    result.stdout,
                )
                self.assertIn(
                    f"thesis_medium_quant_mistral_qproj3.0_g20_o16_seed{seed}",
                    result.stdout,
                )
                self.assertIn(
                    f"thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed{seed}",
                    result.stdout,
                )

            self.assertIn("--calibration_tokens 8192", result.stdout)
            self.assertIn("--calibration_sequence_length 1024", result.stdout)
            self.assertIn("--generations 20", result.stdout)
            self.assertIn("--offspring 16", result.stdout)
            self.assertIn("--initially_generated 32", result.stdout)
            self.assertIn("--survivors_per_selection 8 2 1", result.stdout)
            self.assertIn("--tokens_per_selection 512 2048 8192", result.stdout)
            self.assertIn("--eval_every 5", result.stdout)
            self.assertIn("--active_quant_budget", result.stdout)
            self.assertIn("--group_rule size", result.stdout)
            self.assertIn("Dry run complete", result.stdout)
            self.assertFalse((Path(temp_dir) / "outputs").exists())

    def test_invalid_selection_schedule_is_rejected(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "TOKENS_PER_SELECTION": "512 2048",
                "SURVIVORS_PER_SELECTION": "8 2 1",
            },
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must have equal lengths", result.stderr)

    def test_dry_run_preserves_incomplete_run_and_uses_retry_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs_root = Path(temp_dir) / "outputs"
            incomplete = (
                outputs_root
                / "thesis_medium_depth_mistral_s0.25_g20_o16_seed0"
            )
            incomplete.mkdir(parents=True)
            (incomplete / "run.log").write_text("interrupted\n", encoding="utf-8")

            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "OUTPUTS_ROOT": str(outputs_root),
                    "RUN_DENSE": "0",
                    "METHODS": "depth",
                    "SEEDS": "0",
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("selected retry id", result.stdout)
            self.assertIn(
                "thesis_medium_depth_mistral_s0.25_g20_o16_seed0_retry1",
                result.stdout,
            )

    def test_dry_run_accepts_custom_quant_scope_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                    "EXPERIMENT_LOG": str(Path(temp_dir) / "experiment_log.csv"),
                    "RUN_DENSE": "0",
                    "METHODS": "quant joint",
                    "SEEDS": "0",
                    "QUANT_SCOPE_LABEL": "attention",
                    "QUANT_WEIGHTS_PATH": "outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit",
                    "EXPECTED_QUANT_MODULES": "128",
                    "EXPECTED_QUANT_WEIGHT_FILES": "384",
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "thesis_medium_quant_mistral_attention3.0_g20_o16_seed0",
                result.stdout,
            )
            self.assertIn(
                "thesis_medium_joint_mistral_s0.25_attention3.0_g20_o16_seed0",
                result.stdout,
            )
            self.assertIn(
                "--quant_weights_path outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
