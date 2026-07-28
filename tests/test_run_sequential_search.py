import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_sequential_search.sh"


class RunSequentialSearchTest(unittest.TestCase):
    def run_launcher(
        self,
        arguments: list[str],
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LAUNCHER), *arguments],
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_all_modes_are_forwarded_to_joint_launcher(self) -> None:
        modes = (
            "depth_to_quant_frozen",
            "depth_to_joint_warm",
            "quant_to_depth_frozen",
            "quant_to_joint_warm",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage1_dir = root / "stage1"
            stage1_dir.mkdir()
            for mode in modes:
                result = self.run_launcher(
                    [
                        "--mode",
                        mode,
                        "--stage1-run-dir",
                        str(stage1_dir),
                        "--output-dir",
                        str(root / mode),
                        "--dry-run",
                    ],
                    {
                        "ACTIVE_QUANT_BUDGET": "1",
                        "GROUP_RULE": "size",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--sequential_mode {mode}", result.stdout)
                self.assertIn(
                    f"--stage1_run_dir {stage1_dir}",
                    result.stdout,
                )
                self.assertIn("--max_initialization_attempts 100000", result.stdout)
                self.assertIn("--max_offspring_attempts 10000", result.stdout)
                self.assertFalse((root / mode).exists())

    def test_direct_candidate_and_repair_policy_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "final_candidate.json"
            candidate.write_text("{}\n", encoding="utf-8")
            output_dir = root / "output"
            result = self.run_launcher(
                [
                    "--mode",
                    "quant_to_joint_warm",
                    "--stage1-candidate",
                    str(candidate),
                    "--output-dir",
                    str(output_dir),
                    "--policy",
                    "repair",
                    "--dry-run",
                ],
                {
                    "ACTIVE_QUANT_BUDGET": "1",
                    "GROUP_RULE": "size",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--stage1_candidate {candidate}", result.stdout)
            self.assertIn(
                "--sequential_quant_initialization_policy repair",
                result.stdout,
            )
            self.assertFalse(output_dir.exists())

    def test_missing_source_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            result = self.run_launcher(
                [
                    "--mode",
                    "depth_to_joint_warm",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("exactly one", result.stderr)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
