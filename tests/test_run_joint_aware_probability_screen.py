import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_joint_aware_probability_screen.sh"


class JointAwareProbabilityScreenTest(unittest.TestCase):
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

    def test_default_dry_run_builds_matched_six_run_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_launcher(temp_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            for seed in range(3):
                self.assertIn(
                    f"screen_jointaware_tiny_p0_g20_o8_seed{seed}",
                    result.stdout,
                )
                self.assertIn(
                    f"screen_jointaware_tiny_p025_g20_o8_seed{seed}",
                    result.stdout,
                )
            self.assertEqual(result.stdout.count("--joint_aware_mutation"), 3)
            self.assertEqual(
                result.stdout.count("--joint_aware_probability 0.25"),
                3,
            )
            self.assertEqual(result.stdout.count("--active_quant_budget"), 6)
            self.assertIn("--eval_every 2", result.stdout)
            self.assertIn("Dry run complete", result.stdout)
            self.assertFalse((Path(temp_dir) / "outputs").exists())

    def test_incomplete_run_uses_retry_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs_root = Path(temp_dir) / "outputs"
            incomplete = (
                outputs_root
                / "screen_jointaware_tiny_p025_g20_o8_seed0"
            )
            incomplete.mkdir(parents=True)
            (incomplete / "run.log").write_text(
                "interrupted\n", encoding="utf-8"
            )

            result = self.run_launcher(
                temp_dir,
                {
                    "SEEDS": "0",
                    "VARIANTS": "p025",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("selected retry id", result.stdout)
            self.assertIn(
                "screen_jointaware_tiny_p025_g20_o8_seed0_retry1",
                result.stdout,
            )

    def test_invalid_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_launcher(
                temp_dir,
                {"VARIANTS": "unsupported"},
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("Unsupported screening variant", result.stderr)


if __name__ == "__main__":
    unittest.main()
