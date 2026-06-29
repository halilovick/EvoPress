import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_mistral_generalization_eval.sh"


class RunMistralGeneralizationEvalTest(unittest.TestCase):
    def test_default_dry_run_builds_eval_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                    "EXPERIMENT_LOG": str(
                        Path(temp_dir) / "experiment_log.csv"
                    ),
                },
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "generalization_dense_mistral_multidataset_seq1024_seed0",
            result.stdout,
        )
        for seed in range(3):
            self.assertIn(
                f"generalization_depth_mistral_s0.25_multidataset_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "generalization_independent_depth_quant_mistral_"
                f"s0.25_qproj3.0_multidataset_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "generalization_joint_g50_mistral_"
                f"s0.25_qproj3.0_multidataset_seed{seed}",
                result.stdout,
            )
        self.assertIn(
            "--eval_datasets wikitext2 c4 fineweb_edu",
            result.stdout,
        )
        self.assertIn("--eval_tokens 131072", result.stdout)
        self.assertIn("Mistral generalization dry run complete", result.stdout)

    def test_invalid_method_is_rejected(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={**os.environ, "METHODS": "unknown"},
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported method", result.stderr)

    def test_attention_scope_dry_run_uses_attention_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                    "EXPERIMENT_LOG": str(Path(temp_dir) / "experiment_log.csv"),
                    "RUN_PREFIX": "generalization_attention",
                    "METHODS": "independent joint_g50",
                    "QUANT_WEIGHTS_PATH": "outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit",
                    "DEPTH_SOURCE_PREFIX": "thesis_medium",
                    "QUANT_SOURCE_PREFIX": "thesis_attention",
                    "JOINT_SOURCE_PREFIX": "thesis_attention_g50",
                    "QUANT_SCOPE_LABEL": "attention",
                },
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        for seed in range(3):
            self.assertIn(
                "generalization_attention_independent_depth_quant_mistral_"
                f"s0.25_attention3.0_multidataset_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "generalization_attention_joint_g50_mistral_"
                f"s0.25_attention3.0_multidataset_seed{seed}",
                result.stdout,
            )
            self.assertIn(
                "results/runs/thesis_attention_quant_mistral_"
                f"attention3.0_g20_o16_seed{seed}/quant_configuration.txt",
                result.stdout,
            )
            self.assertIn(
                "results/runs/thesis_attention_g50_joint_mistral_"
                f"s0.25_attention3.0_g50_o16_seed{seed}/joint_quant_config.txt",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
