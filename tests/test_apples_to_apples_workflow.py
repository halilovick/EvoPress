import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.calibration_utils import (
    calibration_token_digest,
    configured_calibration_partition,
)
from scripts.run_apples_to_apples import (
    PROJECTIONS,
    REPO_ROOT,
    build_command,
    default_quant_db,
    expected_search_compute,
    read_json,
    run_and_tee,
    sha256_file,
    validate_prepare_db_memory_mode,
    validate_config,
    validate_quant_database,
    validate_search_run_summary,
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
    @staticmethod
    def _write_fake_python(path: Path, *, exit_code: int = 0) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n"
            "print('wikitext2: 4.82', flush=True)\n"
            "print('c4: 7.72', flush=True)\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o755)

    def test_requirements_use_prebuilt_datalab_flash_attention_wheel(self) -> None:
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("\nflash-attn==2.8.3\n", requirements)
        self.assertIn(
            "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
            requirements,
        )
        self.assertIn('python_version == "3.11"', requirements)
        self.assertIn('sys_platform == "linux"', requirements)

    def test_legacy_mode_is_rejected_under_shared_16_gib_cgroup(self) -> None:
        constrained = {"limit_bytes": 16 * 1024**3}
        for effective_processes in (1, 8):
            with self.subTest(effective_processes=effective_processes):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"aggregate requirement for {effective_processes} process",
                ):
                    validate_prepare_db_memory_mode(
                        "legacy_cpu_offload", effective_processes, constrained
                    )
        validate_prepare_db_memory_mode(
            "disk_activation_cache", 1, constrained
        )
        validate_prepare_db_memory_mode(
            "legacy_cpu_offload",
            8,
            {"limit_bytes": 512 * 1024**3},
        )

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

    def test_calibration_digest_is_order_and_boundary_sensitive(self) -> None:
        first = [torch.tensor([[1, 2]]), torch.tensor([[3]])]
        same = [torch.tensor([[1, 2]]), torch.tensor([[3]])]
        reordered = [torch.tensor([[3]]), torch.tensor([[1, 2]])]
        different_boundaries = [torch.tensor([[1]]), torch.tensor([[2, 3]])]
        self.assertEqual(calibration_token_digest(first), calibration_token_digest(same))
        self.assertNotEqual(
            calibration_token_digest(first), calibration_token_digest(reordered)
        )
        self.assertNotEqual(
            calibration_token_digest(first),
            calibration_token_digest(different_boundaries),
        )

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
        database["expected_reconstruction_shapes"] = {
            projection: [1] for projection in PROJECTIONS
        }
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
                    container = (
                        "self_attn"
                        if projection in {"q_proj", "k_proj", "v_proj", "o_proj"}
                        else "mlp"
                    )
                    module_name = (
                        f"model.layers.{layer_index}.{container}.{projection}"
                    )
                    module_names.append(module_name)
                    module_dir = quant_db / module_name
                    module_dir.mkdir()
                    for level in database["bitwidths"]:
                        torch.save(
                            torch.tensor([float(level)], dtype=torch.float16),
                            module_dir / f"{level}.pth",
                        )
            level_file_sizes_bytes = {
                module_name: {
                    str(level): (quant_db / module_name / f"{level}.pth").stat().st_size
                    for level in database["bitwidths"]
                }
                for module_name in module_names
            }
            reconstruction_metadata = {
                module_name: {
                    str(level): {
                        "shape": [1],
                        "dtype": "float16",
                        "numel": 1,
                        "contiguous": True,
                        "file_size_bytes": (
                            quant_db / module_name / f"{level}.pth"
                        ).stat().st_size,
                        "file_sha256": sha256_file(
                            quant_db / module_name / f"{level}.pth"
                        ),
                    }
                    for level in database["bitwidths"]
                }
                for module_name in module_names
            }
            manifest = {
                "attempt_id": "fixture-attempt",
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
                "calibration_token_digest_algorithm": (
                    "sha256-v1-dtype-shape-boundaries"
                ),
                "calibration_token_digest_loaded": "0" * 64,
                "calibration_token_digest_used": "1" * 64,
                "calibration_token_counts_by_logical_shard": [
                    database["calibration_tokens"] // database["torchrun_processes"]
                ]
                * database["torchrun_processes"],
                "calibration_logical_shard_count": database["torchrun_processes"],
                "configured_torchrun_processes": database["torchrun_processes"],
                "distributed_world_size": 1,
                "torchrun_process_override": True,
                "database_memory_mode": "in_memory_activations",
                "module_count": len(module_names),
                "level_file_sizes_bytes": level_file_sizes_bytes,
                "reconstruction_metadata": reconstruction_metadata,
                "database_payload_bytes": sum(
                    size
                    for module_sizes in level_file_sizes_bytes.values()
                    for size in module_sizes.values()
                ),
                "total_parameters_dense": budget["reference_dense_parameters"],
                "quantized_weight_parameters": budget[
                    "reference_quantized_parameters"
                ],
                "fixed_parameters": budget["reference_fixed_parameters"],
            }
            manifest_path = quant_db / "quant_database_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_quant_database(quant_db, config, allow_unmanifested=False)

            unexpected_root_path = quant_db / "partial_manifest.tmp"
            unexpected_root_path.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "outside the exact module and manifest inventory",
            ):
                validate_quant_database(quant_db, config, allow_unmanifested=False)
            unexpected_root_path.unlink()

            manifest["torchrun_process_override"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "process-override provenance"):
                validate_quant_database(quant_db, config, allow_unmanifested=False)

            manifest["torchrun_process_override"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            unexpected_path = quant_db / module_names[0] / "partial.tmp"
            unexpected_path.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "exactly the five expected reconstruction files",
            ):
                validate_quant_database(quant_db, config, allow_unmanifested=False)
            unexpected_path.unlink()

            (quant_db / module_names[0] / "2.pth").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty, corrupt, or non-tensor"):
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

    def test_memory_safe_prepare_db_command_is_explicit_and_gpu_direct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "not-created"
            database_root = root / "database"
            cache_dir = root / "activation-cache"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "prepare_db",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--quant-db-root",
                    str(database_root),
                    "--torchrun-processes",
                    "1",
                    "--database-memory-mode",
                    "disk_activation_cache",
                    "--activation-cache-dir",
                    str(cache_dir),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("database_memory_mode=disk_activation_cache", result.stdout)
            self.assertIn(f"activation_cache_dir={cache_dir.resolve()}", result.stdout)
            self.assertIn("--load_model_to_gpu", result.stdout)
            self.assertIn(
                f"--activation_cache_dir {cache_dir.resolve()}", result.stdout
            )
            self.assertNotIn("--cpu_offload_modules", result.stdout)
            self.assertIn("--cpu_offload_activations", result.stdout)
            self.assertFalse(output_dir.exists())
            self.assertFalse(database_root.exists())
            self.assertFalse(cache_dir.exists())

    def test_memory_safe_prepare_db_requires_explicit_one_process(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "prepare_db",
                "--config",
                str(PAPER_CONFIG),
                "--database-memory-mode",
                "disk_activation_cache",
                "--activation-cache-dir",
                "/tmp/fixture-cache",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --torchrun-processes 1", result.stderr)

    def test_launcher_records_running_resources_and_completed_postflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_python = root / "fake-python"
            output_dir = root / "run"
            self._write_fake_python(fake_python)
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "dense",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--python",
                    str(fake_python),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("wikitext2: 4.82", result.stdout)
            self.assertIn("c4: 7.72", (output_dir / "run.log").read_text())
            self.assertTrue((output_dir / "resource_samples.jsonl").is_file())
            status = read_json(output_dir / "launcher_status.json")
            runtime = read_json(output_dir / "runtime.json")
            resolved = read_json(output_dir / "resolved_config.json")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(runtime["postflight_status"], "passed")
            self.assertEqual(runtime["exit_code"], 0)
            self.assertIn("resource_summary", runtime)
            self.assertEqual(runtime["attempt_id"], resolved["attempt_id"])
            self.assertEqual(
                runtime["resource_summary"]["attempt_id"],
                resolved["attempt_id"],
            )
            self.assertIn(
                "max_process_tree_rss_bytes_observed",
                runtime["resource_summary"],
            )
            resource_records = [
                json.loads(line)
                for line in (output_dir / "resource_samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreaterEqual(len(resource_records), 1)
            self.assertTrue(
                all(
                    record["attempt_id"] == resolved["attempt_id"]
                    for record in resource_records
                )
            )
            self.assertTrue(
                all("process_tree" in record for record in resource_records)
            )

    def test_launcher_records_failed_child_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_python = root / "fake-python"
            output_dir = root / "run"
            self._write_fake_python(fake_python, exit_code=7)
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "dense",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--python",
                    str(fake_python),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            status = read_json(output_dir / "launcher_status.json")
            runtime = read_json(output_dir / "runtime.json")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "child_process")
            self.assertEqual(runtime["exit_code"], 7)
            self.assertEqual(runtime["postflight_status"], "not_run")

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_run_and_tee_cleans_entire_process_group_on_output_failure(self) -> None:
        class ExplodingStdout:
            def write(self, value):
                del value
                raise RuntimeError("synthetic stdout failure")

            def flush(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "long-running.py"
            pid_file = root / "pids.txt"
            script.write_text(
                "import os, pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])\n"
                "pathlib.Path(sys.argv[1]).write_text("
                "f'{os.getpid()} {child.pid}', encoding='utf-8')\n"
                "print('ready', flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            with mock.patch(
                "scripts.run_apples_to_apples.sys.stdout", ExplodingStdout()
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic stdout failure"):
                    run_and_tee(
                        [sys.executable, str(script), str(pid_file)],
                        root / "run.log",
                        root / "resources.jsonl",
                        attempt_id="cleanup-fixture",
                    )
            process_group_id = int(pid_file.read_text(encoding="utf-8").split()[0])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.killpg(process_group_id, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail(f"Process group {process_group_id} survived launcher failure.")

    def test_run_and_tee_propagates_resource_sampler_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_path = root / "resources.jsonl"
            resource_path.write_text("do not overwrite\n", encoding="utf-8")
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "Resource sampler failed"):
                run_and_tee(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('ready', flush=True); time.sleep(60)",
                    ],
                    root / "run.log",
                    resource_path,
                    attempt_id="sampler-error-fixture",
                )
            self.assertLess(time.monotonic() - started, 15)
            self.assertEqual(
                resource_path.read_text(encoding="utf-8"),
                "do not overwrite\n",
            )

    def test_search_summary_is_required_and_strictly_validated(self) -> None:
        config = read_json(PAPER_CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "run_summary.json"
            with self.assertRaisesRegex(FileNotFoundError, "required run summary"):
                validate_search_run_summary(summary_path, "quant_only", config, 0)

            artifacts = {}
            for key, filename in (
                ("candidate_path", "candidate.json"),
                ("generation_log_path", "generations.csv"),
                ("config_path", "quant_config.json"),
            ):
                path = root / filename
                path.touch()
                artifacts[key] = str(path)
            expected_compute = expected_search_compute(config, "quant_only")
            target_bits = config["budget"][
                "reference_metadata_inclusive_target_bits"
            ]
            summary = {
                "search_type": "quant_only",
                "model_name": config["model"],
                "dataset_eval": config["search"]["eval_datasets"],
                "search_config": {
                    "seed": 0,
                    "generations": config["search"]["generations"],
                    "offspring": config["search"]["offspring"],
                    "fitness_fn": config["search"]["fitness"],
                    "candidate_evaluations_search_total": expected_compute[
                        "candidate_evaluations"
                    ],
                    "evaluation_tokens_search_total": expected_compute[
                        "candidate_tokens"
                    ],
                },
                "compression_config": {
                    "compression_budget_mode": config["budget"]["mode"],
                    "target_cost_bits": target_bits,
                    "database_module_count": config["quant_database"][
                        "expected_modules"
                    ],
                },
                "final_metrics": {
                    "compression_target_bits": target_bits,
                    "compression_realized_bits": target_bits,
                    "compression_difference_bits": 0,
                    "exact_budget_valid": True,
                    "final_calibration_kl": 0.1,
                    "wikitext2_ppl": 5.2,
                    "c4_ppl": 8.4,
                },
                "artifacts": artifacts,
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            validated = validate_search_run_summary(
                summary_path, "quant_only", config, 0
            )
            self.assertEqual(validated["search_type"], "quant_only")

            joint_compute = expected_search_compute(config, "joint")
            summary["search_type"] = "joint_depth_quant"
            summary["search_config"]["candidate_evaluations_search_total"] = (
                joint_compute["candidate_evaluations"]
            )
            summary["search_config"]["evaluation_tokens_search_total"] = (
                joint_compute["candidate_tokens"]
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            validated = validate_search_run_summary(
                summary_path, "joint", config, 0
            )
            self.assertEqual(validated["search_type"], "joint_depth_quant")

            summary["final_metrics"]["compression_realized_bits"] += 1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact-budget mismatch"):
                validate_search_run_summary(summary_path, "joint", config, 0)

    def test_launcher_refuses_even_empty_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "existing-empty"
            output_dir.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "dense",
                    "--config",
                    str(PAPER_CONFIG),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to reuse existing experiment directory", result.stderr)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_prepare_db_refuses_existing_empty_database_and_cache_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_root = root / "database"
            database_target = database_root / "Mistral-7B-v0.3" / "3bit"
            database_target.mkdir(parents=True)
            output_dir = root / "db-collision-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "prepare_db",
                    "--config",
                    str(PAPER_CONFIG),
                    "--quant-db-root",
                    str(database_root),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to reuse existing quantization database", result.stderr)
            self.assertFalse(output_dir.exists())

            database_target.rmdir()
            database_target.parent.rmdir()
            cache_target = root / "existing-empty-cache"
            cache_target.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "prepare_db",
                    "--config",
                    str(PAPER_CONFIG),
                    "--quant-db-root",
                    str(database_root),
                    "--output-dir",
                    str(output_dir),
                    "--torchrun-processes",
                    "1",
                    "--database-memory-mode",
                    "disk_activation_cache",
                    "--activation-cache-dir",
                    str(cache_target),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to reuse existing activation-cache", result.stderr)
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(cache_target.iterdir()), [])

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
