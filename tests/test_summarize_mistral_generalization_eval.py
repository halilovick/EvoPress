import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_mistral_generalization_eval.py"


def write_metrics(run_dir: Path, values: dict[str, float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "evaluation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_id", "dataset", "ppl"),
            lineterminator="\n",
        )
        writer.writeheader()
        for dataset, ppl in values.items():
            writer.writerow(
                {
                    "run_id": run_dir.name,
                    "dataset": dataset,
                    "ppl": ppl,
                }
            )


class SummarizeMistralGeneralizationEvalTest(unittest.TestCase):
    def test_summarizes_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root = root / "runs"
            output_dir = root / "results"
            datasets = {
                "wikitext2": 10.0,
                "c4": 11.0,
                "fineweb_edu": 12.0,
            }

            write_metrics(
                runs_root
                / "generalization_dense_mistral_multidataset_seq1024_seed0",
                datasets,
            )
            for seed in (0, 1, 2):
                write_metrics(
                    runs_root
                    / f"generalization_depth_mistral_s0.25_multidataset_seed{seed}",
                    {
                        "wikitext2": 20.0 + seed,
                        "c4": 30.0 + seed,
                        "fineweb_edu": 40.0 + seed,
                    },
                )
                write_metrics(
                    runs_root
                    / (
                        "generalization_independent_depth_quant_mistral_"
                        f"s0.25_qproj3.0_multidataset_seed{seed}"
                    ),
                    {
                        "wikitext2": 22.0 + seed,
                        "c4": 32.0 + seed,
                        "fineweb_edu": 42.0 + seed,
                    },
                )
                write_metrics(
                    runs_root
                    / (
                        "generalization_joint_g50_mistral_"
                        f"s0.25_qproj3.0_multidataset_seed{seed}"
                    ),
                    {
                        "wikitext2": 21.0 + seed,
                        "c4": 31.0 + seed,
                        "fineweb_edu": 41.0 + seed,
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
            self.assertIn("Wrote 30 run rows.", result.stdout)
            with (
                output_dir / "mistral_generalization_aggregate.csv"
            ).open(newline="", encoding="utf-8") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            joint_wikitext = [
                row
                for row in aggregate_rows
                if row["method"] == "joint_g50"
                and row["dataset"] == "wikitext2"
            ][0]
            self.assertEqual(joint_wikitext["runs"], "3")
            self.assertEqual(joint_wikitext["ppl_mean"], "22.0")

            with (
                output_dir / "mistral_generalization_paired_deltas.csv"
            ).open(newline="", encoding="utf-8") as handle:
                paired_rows = list(csv.DictReader(handle))
            self.assertEqual(len(paired_rows), 9)
            first_wikitext = [
                row
                for row in paired_rows
                if row["dataset"] == "wikitext2" and row["seed"] == "0"
            ][0]
            self.assertEqual(first_wikitext["joint_minus_independent"], "-1.0")
            self.assertTrue(
                (output_dir / "mistral_generalization_eval.md").is_file()
            )

    def test_missing_metrics_fail_by_default(self) -> None:
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
        self.assertIn("Missing evaluation metrics", result.stderr)

    def test_attention_scope_summary_uses_separate_output_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root = root / "runs"
            output_dir = root / "results"
            datasets = {
                "wikitext2": 10.0,
                "c4": 11.0,
                "fineweb_edu": 12.0,
            }

            write_metrics(
                runs_root
                / "generalization_attention_dense_mistral_multidataset_seq1024_seed0",
                datasets,
            )
            for seed in (0, 1, 2):
                write_metrics(
                    runs_root
                    / (
                        "generalization_attention_depth_mistral_"
                        f"s0.25_multidataset_seed{seed}"
                    ),
                    {
                        "wikitext2": 20.0 + seed,
                        "c4": 30.0 + seed,
                        "fineweb_edu": 40.0 + seed,
                    },
                )
                write_metrics(
                    runs_root
                    / (
                        "generalization_attention_independent_depth_quant_mistral_"
                        f"s0.25_attention3.0_multidataset_seed{seed}"
                    ),
                    {
                        "wikitext2": 22.0 + seed,
                        "c4": 32.0 + seed,
                        "fineweb_edu": 42.0 + seed,
                    },
                )
                write_metrics(
                    runs_root
                    / (
                        "generalization_attention_joint_g50_mistral_"
                        f"s0.25_attention3.0_multidataset_seed{seed}"
                    ),
                    {
                        "wikitext2": 21.0 + seed,
                        "c4": 31.0 + seed,
                        "fineweb_edu": 41.0 + seed,
                    },
                )

            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--runs-root",
                    str(runs_root),
                    "--output-dir",
                    str(output_dir),
                    "--run-prefix",
                    "generalization_attention",
                    "--quant-scope-label",
                    "attention",
                    "--output-stem",
                    "mistral_attention_generalization",
                    "--skip-plots",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Wrote 30 run rows.", result.stdout)
            with (
                output_dir / "mistral_attention_generalization_aggregate.csv"
            ).open(newline="", encoding="utf-8") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            independent = [
                row
                for row in aggregate_rows
                if row["method"] == "independent"
                and row["dataset"] == "wikitext2"
            ][0]
            self.assertEqual(
                independent["method_label"],
                "Independent depth + attention quant",
            )
            self.assertTrue(
                (
                    output_dir / "mistral_attention_generalization_eval.md"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
