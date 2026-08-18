import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.calibration_utils import configured_calibration_partition
from scripts.run_apples_to_apples import (
    REPO_ROOT,
    build_command,
    default_quant_db,
    expected_search_compute,
    read_json,
    validate_config,
    validate_quant_database,
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
    def test_one_process_uses_same_configured_calibration_prefix(self) -> None:
        total_sequences = 1027
        configured_processes = 8
        used_one, one_start, one_end = configured_calibration_partition(
            total_sequences, configured_processes, 1, 0
        )
        self.assertEqual((used_one, one_start, one_end), (1024, 0, 1024))

        eight_process_slices = [
            configured_calibration_partition(
                total_sequences, configured_processes, 8, rank
            )
            for rank in range(8)
        ]
        self.assertTrue(all(used == used_one for used, _, _ in eight_process_slices))
        combined_indices = [
            index
            for _, start, end in eight_process_slices
            for index in range(start, end)
        ]
        self.assertEqual(combined_indices, list(range(one_start, one_end)))

    def test_prepare_db_process_override_is_explicit(self) -> None:
        config = read_json(PAPER_CONFIG)
        quant_db = default_quant_db(config, None)
        output_dir = REPO_ROOT / "results" / "fixture"
        default_command = build_command(
            "prepare_db",
            config,
            0,
            quant_db,
            output_dir,
            sys.executable,
            "torchrun",
            None,
        )
        override_command = build_command(
            "prepare_db",
            config,
            0,
            quant_db,
            output_dir,
            sys.executable,
            "torchrun",
            None,
            1,
        )
        self.assertIn("--nproc-per-node=8", default_command)
        self.assertIn("--nproc-per-node=1", override_command)
        configured_index = override_command.index("--configured_torchrun_processes")
        self.assertEqual(override_command[configured_index + 1], "8")

    def test_one_process_database_manifest_is_valid_with_explicit_override(self) -> None:
        config = read_json(PAPER_CONFIG)
        database = config["quant_database"]
        budget = config["budget"]
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_db = Path(temp_dir)
            module_names = []
            for layer_index in range(32):
                for projection in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ):
                    module_name = f"model.layers.{layer_index}.{projection}"
                    module_names.append(module_name)
                    module_dir = quant_db / module_name
                    module_dir.mkdir()
                    for level in database["bitwidths"]:
                        (module_dir / f"{level}.pth").touch()
            manifest = {
                "status": "complete",
                "model_name": config["model"],
                "tokenizer_name": config["model"],
                "tokenizer_is_fast": False,
                "attention_implementation": database["attention_implementation"],
                "bitwidth_options": database["bitwidths"],
                "group_size": database["group_size"],
                "perchannel": database["perchannel"],
                "symmetric": database["symmetric"],
                "activation_order": database["activation_order"],
                "calibration_data": database["calibration_data"],
                "calibration_tokens": database["calibration_tokens"],
                "calibration_sequence_length": database["sequence_length"],
                "calibration_token_count_loaded": database["calibration_tokens"],
                "calibration_token_count_used": database["calibration_tokens"],
                "calibration_logical_shard_count": database["torchrun_processes"],
                "configured_torchrun_processes": database["torchrun_processes"],
                "distributed_world_size": 1,
                "torchrun_process_override": True,
                "module_count": len(module_names),
                "total_parameters_dense": budget["reference_dense_parameters"],
                "quantized_weight_parameters": budget[
                    "reference_quantized_parameters"
                ],
                "fixed_parameters": budget["reference_fixed_parameters"],
            }
            manifest_path = quant_db / "quant_database_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_quant_database(quant_db, config, allow_unmanifested=False)

            manifest["torchrun_process_override"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "process-override provenance"):
                validate_quant_database(quant_db, config, allow_unmanifested=False)

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

    def test_prepare_db_dry_run_reports_process_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "not-created"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "prepare_db",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--torchrun-processes",
                    "1",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("configured_torchrun_processes=8", result.stdout)
            self.assertIn("effective_torchrun_processes=1", result.stdout)
            self.assertIn("torchrun_process_override=true", result.stdout)
            self.assertIn("--nproc-per-node=1", result.stdout)
            self.assertIn("--configured_torchrun_processes 8", result.stdout)
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
