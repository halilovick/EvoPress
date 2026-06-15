import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_adaptive_mutation_screen.sh"


class AdaptiveMutationScreenTest(unittest.TestCase):
    def run_launcher(
        self,
        temp_dir: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "OUTPUTS_ROOT": str(Path(temp_dir) / "outputs"),
                "RESULTS_RUNS_ROOT": str(Path(temp_dir) / "results" / "runs"),
                "EXPERIMENT_LOG": str(
                    Path(temp_dir) / "results" / "experiment_log.csv"
                ),
                **(extra_env or {}),
            },
            capture_output=True,
            text=True,
        )

    def test_default_dry_run_builds_three_seed_adaptive_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_launcher(temp_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            for seed in range(3):
                self.assertIn(
                    f"screen_adaptive_tiny_pat3_max3_g20_o8_seed{seed}",
                    result.stdout,
                )
            self.assertEqual(result.stdout.count("--adaptive_mutation "), 3)
            self.assertEqual(
                result.stdout.count("--adaptive_mutation_patience 3"),
                3,
            )
            self.assertEqual(
                result.stdout.count("--adaptive_mutation_max_strength 3"),
                3,
            )
            self.assertEqual(result.stdout.count("--active_quant_budget"), 3)
            self.assertNotIn("--joint_aware_mutation", result.stdout)
            self.assertIn("baseline artifact validation deferred", result.stdout)
            self.assertIn("Dry run complete", result.stdout)
            self.assertFalse((Path(temp_dir) / "outputs").exists())

    def test_incomplete_run_uses_retry_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs_root = Path(temp_dir) / "outputs"
            incomplete = (
                outputs_root
                / "screen_adaptive_tiny_pat3_max3_g20_o8_seed0"
            )
            incomplete.mkdir(parents=True)
            (incomplete / "run.log").write_text(
                "interrupted\n", encoding="utf-8"
            )

            result = self.run_launcher(
                temp_dir,
                {"SEEDS": "0"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("selected retry id", result.stdout)
            self.assertIn(
                "screen_adaptive_tiny_pat3_max3_g20_o8_seed0_retry1",
                result.stdout,
            )

    def test_invalid_adaptive_patience_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_launcher(
                temp_dir,
                {"ADAPTIVE_MUTATION_PATIENCE": "0"},
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("must be at least 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
