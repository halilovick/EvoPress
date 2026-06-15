import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "run_coarse_to_fine_mutation_screen.sh"
)


class RunCoarseToFineMutationScreenTest(unittest.TestCase):
    def test_default_dry_run_builds_three_seed_screen(self) -> None:
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
                f"screen_coarsetofine_tiny_s3_e1_g20_o8_seed{seed}",
                result.stdout,
            )
        self.assertEqual(
            result.stdout.count("--coarse_to_fine_mutation"),
            3,
        )
        self.assertEqual(
            result.stdout.count("--coarse_to_fine_start_strength 3"),
            3,
        )
        self.assertEqual(
            result.stdout.count("--coarse_to_fine_end_strength 1"),
            3,
        )
        self.assertNotIn("--adaptive_mutation ", result.stdout)

    def test_invalid_strength_range_is_rejected(self) -> None:
        result = subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "START_STRENGTH": "1",
                "END_STRENGTH": "2",
            },
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("START_STRENGTH >= END_STRENGTH", result.stderr)


if __name__ == "__main__":
    unittest.main()
