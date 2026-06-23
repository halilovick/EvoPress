import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_mistral_lmeval_comparison.py"


def write_lmeval_result(run_dir: Path, scores: dict[str, float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "results": {
            task: {
                "acc,none": score,
                "acc_stderr,none": 0.01,
            }
            for task, score in scores.items()
        },
        "config": {"batch_sizes": [4]},
    }
    (run_dir / "lmeval_results.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


class SummarizeMistralLmEvalComparisonTest(unittest.TestCase):
    def test_summarizes_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root = root / "runs"
            output_dir = root / "results"
            write_lmeval_result(
                runs_root / "lmeval_dense_mistral_tasks_seed0",
                {
                    "arc_easy": 0.80,
                    "piqa": 0.75,
                    "winogrande": 0.70,
                },
            )
            for seed in (0, 1, 2):
                write_lmeval_result(
                    runs_root / f"lmeval_depth_mistral_s0.25_tasks_seed{seed}",
                    {
                        "arc_easy": 0.60 + seed * 0.01,
                        "piqa": 0.61 + seed * 0.01,
                        "winogrande": 0.62 + seed * 0.01,
                    },
                )
                write_lmeval_result(
                    runs_root
                    / (
                        "lmeval_independent_depth_quant_mistral_"
                        f"s0.25_qproj3.0_tasks_seed{seed}"
                    ),
                    {
                        "arc_easy": 0.58 + seed * 0.01,
                        "piqa": 0.59 + seed * 0.01,
                        "winogrande": 0.60 + seed * 0.01,
                    },
                )
                write_lmeval_result(
                    runs_root
                    / (
                        "lmeval_joint_g50_mistral_"
                        f"s0.25_qproj3.0_tasks_seed{seed}"
                    ),
                    {
                        "arc_easy": 0.62 + seed * 0.01,
                        "piqa": 0.63 + seed * 0.01,
                        "winogrande": 0.64 + seed * 0.01,
                    },
                )

            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--runs-root",
                    str(runs_root),
                    "--output-dir",
                    str(output_dir),
                    "--skip-plots",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Wrote 30 task score rows.", result.stdout)
            with (output_dir / "mistral_lmeval_aggregate.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                aggregate_rows = list(csv.DictReader(handle))
            joint_arc = [
                row
                for row in aggregate_rows
                if row["method"] == "joint_g50"
                and row["task"] == "arc_easy"
            ][0]
            self.assertEqual(joint_arc["runs"], "3")
            self.assertAlmostEqual(float(joint_arc["score_mean"]), 0.63)

            with (
                output_dir / "mistral_lmeval_paired_deltas.csv"
            ).open(newline="", encoding="utf-8") as handle:
                paired_rows = list(csv.DictReader(handle))
            self.assertEqual(len(paired_rows), 9)
            seed0_arc = [
                row
                for row in paired_rows
                if row["task"] == "arc_easy" and row["seed"] == "0"
            ][0]
            self.assertAlmostEqual(
                float(seed0_arc["joint_minus_independent"]),
                0.04,
            )
            self.assertTrue(
                (output_dir / "mistral_lmeval_comparison.md").is_file()
            )

    def test_missing_results_fail_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--runs-root",
                    str(Path(temp_dir) / "runs"),
                    "--output-dir",
                    str(Path(temp_dir) / "results"),
                    "--skip-plots",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing LM-eval results", result.stderr)


if __name__ == "__main__":
    unittest.main()
