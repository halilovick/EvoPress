import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "run_mistral_fixed_mutation_strength.sh"
)


class RunMistralFixedMutationStrengthTest(unittest.TestCase):
    def test_default_dry_run_builds_matched_three_seed_ablation(self) -> None:
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
        for seed in range(3):
            self.assertIn(
                "thesis_fixedstrength_joint_mistral_"
                f"s0.25_qproj3.0_g50_o16_seed{seed}",
                result.stdout,
            )
        self.assertEqual(
            result.stdout.count("--max_drop_mutations 1"),
            3,
        )
        self.assertNotIn("--adaptive_mutation ", result.stdout)
        self.assertNotIn("--joint_aware_mutation ", result.stdout)
        self.assertNotIn("_dense_mistral_", result.stdout)

    def test_non_unit_strength_is_rejected(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={**os.environ, "MAX_DROP_MUTATIONS": "2"},
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires MAX_DROP_MUTATIONS=1", result.stderr)


if __name__ == "__main__":
    unittest.main()
