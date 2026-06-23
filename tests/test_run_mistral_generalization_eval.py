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


if __name__ == "__main__":
    unittest.main()
