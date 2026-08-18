#!/usr/bin/env python3
"""Reproducible launcher for the EvoPress apples-to-apples experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "apples_to_apples"
    / "mistral7b_v03_paper_matched.json"
)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def expected_search_compute(config: dict[str, Any], method: str) -> dict[str, int]:
    if method not in {"quant_only", "joint"}:
        return {"candidate_evaluations": 0, "candidate_tokens": 0}
    search = config["search"]
    if method == "quant_only":
        initial_evaluations = (
            0 if search["skip_quant_uniform_initial_evaluation"] else 1
        )
    else:
        initial_evaluations = (
            0
            if search.get("skip_joint_single_initial_evaluation", False)
            else search["joint_initial_candidates"]
        )

    stage_counts = []
    candidate_pool = search["offspring"]
    for stage_index, survivor_count in enumerate(search["survivors"]):
        if stage_index == len(search["survivors"]) - 1:
            candidate_pool += 1  # incumbent parent, excluded from generated offspring
        stage_counts.append(candidate_pool)
        candidate_pool = survivor_count
    evaluations_per_generation = sum(stage_counts)
    tokens_per_generation = sum(
        count * tokens
        for count, tokens in zip(stage_counts, search["selection_tokens"])
    )
    return {
        "candidate_evaluations": (
            initial_evaluations
            + search["generations"] * evaluations_per_generation
        ),
        "candidate_tokens": (
            initial_evaluations * search["initial_tokens"]
            + search["generations"] * tokens_per_generation
        ),
    }


def validate_config(config: dict[str, Any]) -> None:
    for key in ("profile", "model", "quant_database", "budget", "search", "results_root"):
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    database = config["quant_database"]
    budget = config["budget"]
    search = config["search"]
    if sorted(database["bitwidths"]) != [2, 3, 4, 5, 6]:
        raise ValueError("Apples-to-apples database must provide bit-widths 2, 3, 4, 5, 6.")
    if database["group_size"] != 128:
        raise ValueError("Apples-to-apples GPTQ group size must be 128.")
    if database["expected_modules"] != 224:
        raise ValueError("Mistral apples-to-apples scope must contain 224 projection modules.")
    if budget["uniform_target_bitwidth"] != 3:
        raise ValueError("The common reference target must be uniform 3-bit GPTQ.")
    if budget["mode"] != "match_uniform_quantization_total":
        raise ValueError("The thesis configurations must use exact total-model budgeting.")
    if len(search["survivors"]) != len(search["selection_tokens"]):
        raise ValueError("Selection survivor and token schedules must have equal length.")
    if search["survivors"][-1] != 1:
        raise ValueError("The final selection stage must retain one candidate.")
    if search["group_rule"] != "size":
        raise ValueError("Paper-compatible quantization switches require size grouping.")
    if config["dtype"] != "float16" or budget["dense_dtype_bits"] != 16:
        raise ValueError("The Mistral reference accounting requires FP16 fixed parameters.")
    if budget["reference_quantized_parameters"] + budget["reference_fixed_parameters"] != budget["reference_dense_parameters"]:
        raise ValueError("Reference fixed and quantized parameter counts do not sum to dense parameters.")
    expected_dense_bits = budget["reference_dense_parameters"] * budget["dense_dtype_bits"]
    if budget["reference_dense_bits"] != expected_dense_bits:
        raise ValueError("Configured dense cost is inconsistent with the parameter count.")
    expected_weight_only = (
        budget["reference_fixed_parameters"] * budget["dense_dtype_bits"]
        + budget["reference_quantized_parameters"] * budget["uniform_target_bitwidth"]
    )
    if budget["reference_paper_weight_only_target_bits"] != expected_weight_only:
        raise ValueError("Configured weight-only target is internally inconsistent.")
    metadata_bits = (
        budget["reference_quantized_parameters"]
        // database["group_size"]
        * (budget["scale_bits"] + budget["zero_point_bits"])
    )
    expected_target = expected_weight_only + metadata_bits
    if budget["reference_metadata_inclusive_target_bits"] != expected_target:
        raise ValueError("Configured metadata-inclusive target is internally inconsistent.")
    quant_compute = expected_search_compute(config, "quant_only")
    joint_compute = expected_search_compute(config, "joint")
    if quant_compute != joint_compute:
        raise ValueError(
            "Quant-only and joint search-compute expectations differ: "
            f"quant_only={quant_compute}, joint={joint_compute}."
        )


def default_quant_db(config: dict[str, Any], root_override: str | None) -> Path:
    save_root = Path(root_override or config["quant_database"]["save_root"])
    if not save_root.is_absolute():
        save_root = REPO_ROOT / save_root
    model_basename = config["model"].rsplit("/", 1)[-1]
    bitwidth = config["quant_database"]["calibration_bitwidth"]
    return save_root / model_basename / f"{bitwidth}bit"


def validate_quant_database(
    path: Path,
    config: dict[str, Any],
    *,
    allow_unmanifested: bool,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Quantization database does not exist: {path}")
    database = config["quant_database"]
    budget = config["budget"]
    module_dirs = sorted(item for item in path.iterdir() if item.is_dir())
    if len(module_dirs) != database["expected_modules"]:
        raise ValueError(
            f"Expected {database['expected_modules']} quantized modules, found {len(module_dirs)}."
        )
    expected_levels = sorted(database["bitwidths"])
    failures = {}
    projection_counts: Counter[str] = Counter()
    for module_dir in module_dirs:
        levels = sorted(
            int(item.stem)
            for item in module_dir.glob("*.pth")
            if item.stem.isdigit()
        )
        if levels != expected_levels:
            failures[module_dir.name] = levels
        projection_counts[module_dir.name.rsplit(".", 1)[-1]] += 1
    if failures:
        raise ValueError(f"Database levels do not match 2--6 for every module: {failures}")
    expected_projection_counts = Counter({projection: 32 for projection in PROJECTIONS})
    if projection_counts != expected_projection_counts:
        raise ValueError(
            "Database is not the full seven-projection Mistral scope: "
            f"actual={dict(projection_counts)}."
        )

    manifest_path = path / "quant_database_manifest.json"
    if not manifest_path.is_file():
        if allow_unmanifested:
            return
        raise ValueError(
            f"Missing {manifest_path}. Regenerate with the apples launcher or pass "
            "--allow-unmanifested-db only after independently verifying provenance."
        )
    manifest = read_json(manifest_path)
    expected_values = {
        "status": "complete",
        "model_name": config["model"],
        "tokenizer_name": config["model"],
        "tokenizer_is_fast": False,
        "attention_implementation": database["attention_implementation"],
        "bitwidth_options": expected_levels,
        "group_size": database["group_size"],
        "perchannel": database["perchannel"],
        "symmetric": database["symmetric"],
        "activation_order": database["activation_order"],
        "calibration_data": database["calibration_data"],
        "calibration_tokens": database["calibration_tokens"],
        "calibration_sequence_length": database["sequence_length"],
        "calibration_token_count_loaded": database["calibration_tokens"],
        "calibration_logical_shard_count": database["torchrun_processes"],
        "configured_torchrun_processes": database["torchrun_processes"],
        "module_count": database["expected_modules"],
        "total_parameters_dense": budget["reference_dense_parameters"],
        "quantized_weight_parameters": budget["reference_quantized_parameters"],
        "fixed_parameters": budget["reference_fixed_parameters"],
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_values.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Quantization database manifest mismatch: {mismatches}")
    configured_processes = manifest.get("configured_torchrun_processes")
    effective_processes = manifest.get("distributed_world_size")
    if not isinstance(configured_processes, int) or configured_processes < 1:
        raise ValueError(
            "Manifest configured_torchrun_processes must be a positive integer."
        )
    if not isinstance(effective_processes, int) or effective_processes < 1:
        raise ValueError("Manifest distributed_world_size must be a positive integer.")
    if manifest.get("torchrun_process_override") != (
        configured_processes != effective_processes
    ):
        raise ValueError("Manifest torchrun process-override provenance is inconsistent.")
    loaded_tokens = manifest.get("calibration_token_count_loaded")
    used_tokens = manifest.get("calibration_token_count_used")
    if loaded_tokens != database["calibration_tokens"]:
        raise ValueError(
            "Manifest loaded calibration-token count does not match the request: "
            f"expected={database['calibration_tokens']}, actual={loaded_tokens}."
        )
    if not isinstance(used_tokens, int) or not 0 < used_tokens <= loaded_tokens:
        raise ValueError(
            "Manifest used calibration-token count must be positive and no larger "
            "than the loaded count."
        )


def common_eval_command(config: dict[str, Any], seed: int, python_bin: str) -> list[str]:
    search = config["search"]
    command = [
        python_bin,
        "eval_ppl.py",
        "--model_name_or_path",
        config["model"],
        "--eval_datasets",
        *search["eval_datasets"],
        "--eval_tokens",
        str(search["eval_tokens"]),
        "--sequence_length",
        str(search["eval_sequence_length"]),
        "--dtype",
        config["dtype"],
        "--attn_implementation",
        config["attention_implementation"],
        "--seed",
        str(seed),
    ]
    if config.get("use_fast_tokenizer", False):
        command.append("--use_fast_tokenizer")
    return command


def common_search_args(
    config: dict[str, Any],
    seed: int,
    quant_db: Path,
    output_dir: Path,
) -> list[str]:
    search = config["search"]
    database = config["quant_database"]
    budget = config["budget"]
    args = [
        "--model_name_or_path",
        config["model"],
        "--quant_weights_path",
        str(quant_db),
        "--target_bitwidth",
        str(budget["uniform_target_bitwidth"]),
        "--calibration_data",
        search["calibration_data"],
        "--calibration_tokens",
        str(search["calibration_tokens"]),
        "--calibration_sequence_length",
        str(search["sequence_length"]),
        "--eval_datasets",
        *search["eval_datasets"],
        "--eval_tokens",
        str(search["eval_tokens"]),
        "--eval_sequence_length",
        str(search["eval_sequence_length"]),
        "--eval_every",
        str(search["eval_every"]),
        "--generations",
        str(search["generations"]),
        "--offspring",
        str(search["offspring"]),
        "--survivors_per_selection",
        *[str(value) for value in search["survivors"]],
        "--tokens_per_selection",
        *[str(value) for value in search["selection_tokens"]],
        "--initial_tokens",
        str(search["initial_tokens"]),
        "--fitness_fn",
        search["fitness"],
        "--group_rule",
        search["group_rule"],
        "--step_size",
        str(search["step_size"]),
        "--compression_budget_mode",
        budget["mode"],
        "--quantization_group_size",
        str(database["group_size"]),
        "--budget_scale_bits",
        str(budget["scale_bits"]),
        "--budget_zero_point_bits",
        str(budget["zero_point_bits"]),
        "--budget_dense_dtype_bits",
        str(budget["dense_dtype_bits"]),
        "--expected_dense_model_bits",
        str(budget["reference_dense_bits"]),
        "--expected_target_cost_bits",
        str(budget["reference_metadata_inclusive_target_bits"]),
        "--expected_bitwidths",
        *[str(value) for value in database["bitwidths"]],
        "--expected_quantized_modules",
        str(database["expected_modules"]),
        "--dtype",
        config["dtype"],
        "--attn_implementation",
        config["attention_implementation"],
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
    ]
    if budget["include_quantization_metadata"]:
        args.append("--budget_include_quantization_metadata")
    if config.get("use_fast_tokenizer", False):
        args.append("--use_fast_tokenizer")
    return args


def build_command(
    method: str,
    config: dict[str, Any],
    seed: int,
    quant_db: Path,
    output_dir: Path,
    python_bin: str,
    torchrun_bin: str,
    quant_db_root: str | None,
    torchrun_processes: int | None = None,
) -> list[str]:
    database = config["quant_database"]
    search = config["search"]
    budget = config["budget"]
    if method == "prepare_db":
        effective_torchrun_processes = (
            database["torchrun_processes"]
            if torchrun_processes is None
            else torchrun_processes
        )
        if effective_torchrun_processes < 1:
            raise ValueError("Database generation requires at least one torchrun process.")
        save_root = Path(quant_db_root or database["save_root"])
        return [
            torchrun_bin,
            "--nnodes=1",
            f"--nproc-per-node={effective_torchrun_processes}",
            "quant.py",
            "--configured_torchrun_processes",
            str(database["torchrun_processes"]),
            "--model_name_or_path",
            config["model"],
            "--quantizable_modules",
            database["quantizable_modules"],
            "--pre_block_modules",
            "model.embed_tokens",
            "--block_modules",
            "model.layers",
            "--post_block_modules",
            "model.norm",
            "lm_head",
            "--calibration_data",
            database["calibration_data"],
            "--calibration_tokens",
            str(database["calibration_tokens"]),
            "--calibration_sequence_length",
            str(database["sequence_length"]),
            "--bitwidth_options",
            *[str(value) for value in database["bitwidths"]],
            "--calibration_bitwidth",
            str(database["calibration_bitwidth"]),
            "--group_size",
            str(database["group_size"]),
            "--rel_damp",
            str(database["relative_dampening"]),
            "--block_size",
            str(database["block_size"]),
            "--seed",
            str(seed),
            "--low_cpu_mem_usage",
            "--cpu_offload_modules",
            "--cpu_offload_activations",
            "--drop_saved_file_cache",
            "--verbose",
            "--dtype",
            config["dtype"],
            "--attn_implementation",
            database["attention_implementation"],
            "--save_dir",
            str(save_root),
            "--perchannel",
        ]
    if method == "dense":
        return common_eval_command(config, seed, python_bin)
    if method == "uniform3":
        return common_eval_command(config, seed, python_bin) + [
            "--quant_weights_path",
            str(quant_db),
            "--quant_default_level",
            str(budget["uniform_target_bitwidth"]),
        ]
    common = common_search_args(config, seed, quant_db, output_dir)
    if method == "quant_only":
        command = [python_bin, "evo_quant_search.py", *common]
        command += ["--initially_generated", str(search["quant_initial_candidates"])]
        if search["skip_quant_uniform_initial_evaluation"]:
            command.append("--skip_initial_uniform_evaluation")
        return command
    if method == "joint":
        command = [python_bin, "evo_joint_search.py", *common]
        command += [
            "--drop_sparsity",
            str(budget["joint_drop_sparsity"]),
            "--max_drop_mutations",
            str(search["max_drop_mutations"]),
            "--joint_mutation_mode",
            search["joint_mutation_mode"],
            "--initially_generated",
            str(search["joint_initial_candidates"]),
            "--population_size",
            str(search["population_size"]),
            "--crossover_probability",
            str(search["crossover_probability"]),
        ]
        if budget["drop_entire_block"]:
            command.append("--drop_entire_block")
        if search.get("skip_joint_single_initial_evaluation", False):
            command.append("--skip_initial_single_candidate_evaluation")
        return command
    raise ValueError(f"Unsupported method: {method}")


def parse_perplexities(log_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pattern = re.compile(r"^(wikitext2|c4|fineweb_edu):\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in log_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def eval_summary(
    method: str,
    config: dict[str, Any],
    seed: int,
    runtime_seconds: float,
    metrics: dict[str, float],
    output_dir: Path,
    quant_db: Path,
) -> dict[str, Any]:
    budget = config["budget"]
    is_dense = method == "dense"
    cost_bits = (
        budget["reference_dense_bits"]
        if is_dense
        else budget["reference_metadata_inclusive_target_bits"]
    )
    ratio = 1.0 if is_dense else budget["reference_metadata_inclusive_compression_ratio"]
    bitwidth_by_module = (
        {
            path.name: budget["uniform_target_bitwidth"]
            for path in sorted(quant_db.iterdir())
            if path.is_dir()
        }
        if method == "uniform3"
        else {}
    )
    candidate = {
        "candidate_type": "dense" if is_dense else "uniform_quantization",
        "dropped_modules": [],
        "attention_mask": [0] * 32,
        "mlp_mask": [0] * 32,
        "bitwidth_by_module": bitwidth_by_module,
        "candidate_vector_raw": None if is_dense else "uniform_3_bit",
    }
    candidate_path = output_dir / "final_candidate.json"
    write_json(candidate_path, candidate)
    return {
        "schema_version": 3,
        "run_name": output_dir.name,
        "timestamp_end": utc_now(),
        "git_commit": git_commit(),
        "model_name": config["model"],
        "dataset_calibration": None,
        "dataset_eval": config["search"]["eval_datasets"],
        "search_type": "dense" if is_dense else "uniform_quantization",
        "search_config": {"seed": seed, "candidate_evaluations_search_total": 0},
        "compression_config": {
            "target_cost_bits": cost_bits,
            "compression_budget_mode": budget["mode"],
            "target_average_bitwidth": None if is_dense else 3,
            "include_quantization_metadata": not is_dense,
        },
        "final_metrics": {
            "wikitext2_ppl": metrics.get("wikitext2"),
            "c4_ppl": metrics.get("c4"),
            "fineweb_ppl": metrics.get("fineweb_edu"),
            "best_search_fitness": None,
            "final_calibration_kl": None,
            "runtime_seconds": runtime_seconds,
            "compression_target_bits": cost_bits,
            "compression_realized_bits": cost_bits,
            "compression_difference_bits": 0,
            "estimated_compression_ratio": ratio,
            "estimated_weight_memory_mb": cost_bits / 8 / 1024**2,
        },
        "model_size_statistics": {
            "compression_cost_bits": cost_bits,
            "estimated_weight_memory_mb": cost_bits / 8 / 1024**2,
            "dense_weight_memory_mb": budget["reference_dense_bits"] / 8 / 1024**2,
            "estimated_compression_ratio": ratio,
        },
        "quantization_statistics": {
            "quantized_module_count": len(bitwidth_by_module),
            "bitwidth_by_module": bitwidth_by_module,
        },
        "artifacts": {
            "stdout_log_path": str(output_dir / "run.log"),
            "candidate_path": str(candidate_path),
            "resolved_config_path": str(output_dir / "resolved_config.json"),
        },
    }


def run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=["prepare_db", "dense", "uniform3", "quant_only", "joint"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant-db", type=Path, default=None)
    parser.add_argument("--quant-db-root", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument(
        "--torchrun-processes",
        type=int,
        default=None,
        help=(
            "Explicit prepare_db process-count override. The configured value remains "
            "unchanged and both configured/effective values are logged as provenance."
        ),
    )
    parser.add_argument("--allow-unmanifested-db", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.torchrun_processes is not None:
        if args.method != "prepare_db":
            parser.error("--torchrun-processes is valid only for prepare_db.")
        if args.torchrun_processes < 1:
            parser.error("--torchrun-processes must be at least 1.")

    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    quant_db = (
        args.quant_db.resolve()
        if args.quant_db is not None
        else default_quant_db(config, args.quant_db_root)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{args.method}_seed{args.seed}_{timestamp}"
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        results_root = Path(config["results_root"])
        if not results_root.is_absolute():
            results_root = REPO_ROOT / results_root
        output_dir = results_root / config["profile"] / args.method / run_id

    command = build_command(
        args.method,
        config,
        args.seed,
        quant_db,
        output_dir,
        args.python,
        args.torchrun,
        args.quant_db_root,
        args.torchrun_processes,
    )
    if args.dry_run:
        print(f"profile={config['profile']}")
        print(f"method={args.method}")
        print(f"seed={args.seed}")
        print(f"quant_db={quant_db}")
        print(f"output_dir={output_dir}")
        if args.method == "prepare_db":
            configured_processes = config["quant_database"]["torchrun_processes"]
            effective_processes = (
                configured_processes
                if args.torchrun_processes is None
                else args.torchrun_processes
            )
            print(f"configured_torchrun_processes={configured_processes}")
            print(f"effective_torchrun_processes={effective_processes}")
            print(
                "torchrun_process_override="
                f"{str(effective_processes != configured_processes).lower()}"
            )
        print(
            "expected_search_compute="
            f"{json.dumps(expected_search_compute(config, args.method), sort_keys=True)}"
        )
        print(f"command={shlex.join(command)}")
        return 0

    if args.method not in {"prepare_db", "dense"}:
        validate_quant_database(
            quant_db,
            config,
            allow_unmanifested=args.allow_unmanifested_db,
        )
    if args.method == "prepare_db" and quant_db.exists() and any(quant_db.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty quantization database: {quant_db}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "source_config": str(config_path),
        "resolved_at": utc_now(),
        "git_commit": git_commit(),
        "method": args.method,
        "seed": args.seed,
        "quant_database": str(quant_db),
        "output_dir": str(output_dir),
        "configuration": config,
        "runtime_overrides": {
            "configured_torchrun_processes": (
                config["quant_database"]["torchrun_processes"]
                if args.method == "prepare_db"
                else None
            ),
            "effective_torchrun_processes": (
                (
                    config["quant_database"]["torchrun_processes"]
                    if args.torchrun_processes is None
                    else args.torchrun_processes
                )
                if args.method == "prepare_db"
                else None
            ),
        },
        "expected_search_compute": expected_search_compute(config, args.method),
    }
    write_json(output_dir / "resolved_config.json", resolved)
    (output_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"exec {shlex.join(command)}\n",
        encoding="utf-8",
    )
    os.chmod(output_dir / "command.sh", 0o755)

    start_wall = time.time()
    start_monotonic = time.monotonic()
    exit_code = run_and_tee(command, output_dir / "run.log")
    runtime_seconds = time.monotonic() - start_monotonic
    runtime = {
        "started_unix_seconds": start_wall,
        "ended_unix_seconds": time.time(),
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
    }
    write_json(output_dir / "runtime.json", runtime)
    write_json(
        output_dir / "launcher_status.json",
        {
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "timestamp": utc_now(),
        },
    )
    if exit_code != 0:
        return exit_code

    if args.method in {"dense", "uniform3"}:
        log_text = (output_dir / "run.log").read_text(encoding="utf-8")
        metrics = parse_perplexities(log_text)
        missing = set(config["search"]["eval_datasets"]) - set(metrics)
        if missing:
            raise RuntimeError(f"Completed evaluation is missing datasets: {sorted(missing)}")
        write_json(
            output_dir / "run_summary.json",
            eval_summary(
                args.method,
                config,
                args.seed,
                runtime_seconds,
                metrics,
                output_dir,
                quant_db,
            ),
        )
    else:
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            summary["launcher_wall_clock"] = runtime
            summary["resolved_config_path"] = str(output_dir / "resolved_config.json")
            write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
