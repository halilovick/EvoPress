import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_depth_warmstart_g50_grid.sh"


class RunDepthWarmstartG50GridTest(unittest.TestCase):
    def run_launcher(
        self,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LAUNCHER), "--dry-run"],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def test_standard_conditions_use_matched_g50_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = self.run_launcher(
                {
                    "OUTPUTS_ROOT": str(root / "outputs"),
                    "RESULTS_RUNS_ROOT": str(root / "runs"),
                    "RESULTS_DIR": str(root / "results"),
                    "EXPERIMENT_LOG": str(root / "experiment_log.csv"),
                    "CONDITIONS": "standard_standard standard_interaction",
                    "SEEDS": "0",
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0",
            result.stdout,
        )
        self.assertIn(
            "thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed0",
            result.stdout,
        )
        self.assertIn("--generations 50", result.stdout)
        self.assertIn("--offspring 16", result.stdout)
        self.assertIn("--initially_generated 32", result.stdout)
        self.assertIn("--survivors_per_selection 8 2 1", result.stdout)
        self.assertIn("--tokens_per_selection 512 2048 8192", result.stdout)
        self.assertIn("--eval_every 5", result.stdout)
        self.assertIn("--active_quant_budget", result.stdout)
        self.assertIn("--group_rule size", result.stdout)
        self.assertIn("--joint_mutation_mode standard", result.stdout)
        self.assertIn("--joint_mutation_mode interaction_aware", result.stdout)
        self.assertNotIn("--joint_aware_mutation", result.stdout)

    def test_warm_conditions_reuse_seed_matched_depth_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = self.run_launcher(
                {
                    "OUTPUTS_ROOT": str(root / "outputs"),
                    "RESULTS_RUNS_ROOT": str(REPO_ROOT / "results" / "runs"),
                    "RESULTS_DIR": str(root / "results"),
                    "EXPERIMENT_LOG": str(root / "experiment_log.csv"),
                    "CONDITIONS": "depthwarm_standard depthwarm_interaction",
                    "SEEDS": "0",
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        stage1 = (
            REPO_ROOT
            / "results"
            / "runs"
            / "thesis_medium_depth_mistral_s0.25_g20_o16_seed0"
        )
        self.assertIn("--sequential_mode depth_to_joint_warm", result.stdout)
        self.assertIn(f"--stage1_run_dir {stage1}", result.stdout)
        self.assertIn("--sequential_quant_initialization_policy strict", result.stdout)
        self.assertIn("--joint_mutation_mode standard", result.stdout)
        self.assertIn("--joint_mutation_mode interaction_aware", result.stdout)
        self.assertNotIn("depth_to_quant_frozen", result.stdout)
        self.assertNotIn("quant_to_depth_frozen", result.stdout)
        self.assertNotIn("quant_to_joint_warm", result.stdout)
        self.assertNotIn("--joint_aware_mutation", result.stdout)

    def test_valid_completed_baselines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = self.run_launcher(
                {
                    "OUTPUTS_ROOT": str(root / "outputs"),
                    "RESULTS_RUNS_ROOT": str(REPO_ROOT / "results" / "runs"),
                    "RESULTS_DIR": str(root / "results"),
                    "CONDITIONS": "standard_standard standard_interaction",
                    "SEEDS": "0",
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Skipping valid completed run"), 2)

    def test_incomplete_warm_output_gets_retry_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            incomplete = (
                root
                / "outputs"
                / (
                    "thesis_depthwarm_standard_joint_"
                    "mistral_s0.25_qproj3.0_g50_o16_seed0"
                )
            )
            incomplete.mkdir(parents=True)
            (incomplete / "run.log").write_text("interrupted\n", encoding="utf-8")
            result = self.run_launcher(
                {
                    "OUTPUTS_ROOT": str(root / "outputs"),
                    "RESULTS_RUNS_ROOT": str(REPO_ROOT / "results" / "runs"),
                    "RESULTS_DIR": str(root / "results"),
                    "CONDITIONS": "depthwarm_standard",
                    "SEEDS": "0",
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("selected retry id", result.stdout)
        self.assertIn(
            "thesis_depthwarm_standard_joint_"
            "mistral_s0.25_qproj3.0_g50_o16_seed0_retry1",
            result.stdout,
        )

    def test_non_g50_override_is_rejected(self) -> None:
        result = self.run_launcher(
            {
                "GENERATIONS": "20",
                "CONDITIONS": "standard_standard",
                "SEEDS": "0",
            }
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires GENERATIONS=50", result.stderr)


if __name__ == "__main__":
    unittest.main()
