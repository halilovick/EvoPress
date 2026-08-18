import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_apples_to_apples import (
    REPO_ROOT,
    build_command,
    default_quant_db,
    expected_search_compute,
    read_json,
    validate_config,
)


PAPER_CONFIG = (
    REPO_ROOT
    / "configs"
    / "apples_to_apples"
    / "mistral7b_v03_paper_matched.json"
)
COMPUTE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "apples_to_apples"
    / "mistral7b_v03_compute_matched.json"
)
LAUNCHER = REPO_ROOT / "scripts" / "run_apples_to_apples.py"
AGGREGATOR = REPO_ROOT / "scripts" / "aggregate_apples_to_apples.py"


class ApplesToApplesWorkflowTest(unittest.TestCase):
    def test_configs_round_trip_and_validate(self) -> None:
        for path in (PAPER_CONFIG, COMPUTE_CONFIG):
            config = read_json(path)
            validate_config(config)
            self.assertEqual(config, json.loads(json.dumps(config)))
            self.assertEqual(config["quant_database"]["bitwidths"], [2, 3, 4, 5, 6])
            self.assertEqual(config["quant_database"]["expected_modules"], 224)
            self.assertEqual(
                config["budget"]["reference_metadata_inclusive_target_bits"],
                26982023168,
            )
        paper = read_json(PAPER_CONFIG)
        self.assertEqual(
            expected_search_compute(paper, "quant_only"),
            {"candidate_evaluations": 22350, "candidate_tokens": 176947200},
        )
        self.assertEqual(
            expected_search_compute(paper, "quant_only"),
            expected_search_compute(paper, "joint"),
        )
        compute = read_json(COMPUTE_CONFIG)
        self.assertEqual(
            expected_search_compute(compute, "quant_only"),
            {"candidate_evaluations": 541, "candidate_tokens": 983552},
        )
        self.assertEqual(
            expected_search_compute(compute, "quant_only"),
            expected_search_compute(compute, "joint"),
        )

    def test_paper_commands_separate_original_and_standard_joint_search(self) -> None:
        config = read_json(PAPER_CONFIG)
        quant_db = default_quant_db(config, None)
        output_dir = REPO_ROOT / "results" / "fixture"
        quant = build_command(
            "quant_only", config, 0, quant_db, output_dir, sys.executable, "torchrun", None
        )
        joint = build_command(
            "joint", config, 0, quant_db, output_dir, sys.executable, "torchrun", None
        )
        quant_text = " ".join(quant)
        joint_text = " ".join(joint)
        for text in (quant_text, joint_text):
            self.assertIn("--compression_budget_mode match_uniform_quantization_total", text)
            self.assertIn("--expected_bitwidths 2 3 4 5 6", text)
            self.assertIn("--survivors_per_selection 16 4 1", text)
            self.assertIn("--tokens_per_selection 2048 16384 131072", text)
            self.assertIn("--expected_quantized_modules 224", text)
            self.assertIn("--expected_target_cost_bits 26982023168", text)
            self.assertNotIn("--use_fast_tokenizer", text)
        self.assertIn("--skip_initial_uniform_evaluation", quant)
        self.assertIn("--joint_mutation_mode standard", joint_text)
        self.assertIn("--skip_initial_single_candidate_evaluation", joint)
        self.assertNotIn("--active_quant_budget", joint)
        self.assertNotIn("--joint_aware_mutation", joint)

    def test_launcher_dry_run_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "not-created"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "joint",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("evo_joint_search.py", result.stdout)
            self.assertFalse(output_dir.exists())

    def test_aggregator_computes_paired_joint_minus_quant_differences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "runs"
            output = Path(temp_dir) / "aggregate"
            for search_type, seed, wiki, c4, kl, runtime in (
                ("dense", 0, 4.82, 7.72, None, 10.0),
                ("uniform_quantization", 0, 5.54, 8.57, None, 20.0),
                ("quant_only", 0, 5.21, 8.42, 0.20, 100.0),
                ("joint_depth_quant", 0, 5.10, 8.30, 0.18, 110.0),
            ):
                run_dir = root / search_type / f"seed{seed}"
                run_dir.mkdir(parents=True)
                target = 115968376832 if search_type == "dense" else 26982023168
                summary = {
                    "search_type": search_type,
                    "search_config": {
                        "seed": seed,
                        "candidate_evaluations_search_total": (
                            0 if search_type in {"dense", "uniform_quantization"} else 10
                        ),
                    },
                    "final_metrics": {
                        "wikitext2_ppl": wiki,
                        "c4_ppl": c4,
                        "final_calibration_kl": kl,
                        "runtime_seconds": runtime,
                        "compression_target_bits": target,
                        "compression_realized_bits": target,
                        "compression_difference_bits": 0,
                        "estimated_compression_ratio": (
                            1.0 if search_type == "dense" else 4.297986704330444
                        ),
                    },
                }
                (run_dir / "run_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(AGGREGATOR),
                    "--input-root",
                    str(root),
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "paired_seed_differences.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(
                float(rows[0]["joint_minus_quant_only_wikitext2_ppl"]), -0.11
            )
            self.assertAlmostEqual(
                float(rows[0]["joint_minus_quant_only_runtime_seconds"]), 10.0
            )
            self.assertIn("Joint depth+quant", (output / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
