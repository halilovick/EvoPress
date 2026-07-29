#!/usr/bin/env python3
"""Summarize the matched G50 depth-warmstart experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_NAME = "mistralai/Mistral-7B-v0.3"
QUANT_WEIGHTS_PATH = (
    "outputs/experiments/quant_db_mistral_qproj_debug_bits234/"
    "quant_db/Mistral-7B-v0.3/3bit"
)
SEEDS = (0, 1, 2)
CONDITION_METADATA: dict[str, dict[str, str]] = {
    "standard_standard": {
        "label": "Standard initialization + standard mutation",
        "short_label": "Standard init\nStandard mutation",
        "initialization": "standard",
        "mutation": "standard",
        "base_pattern": (
            "thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
        "color": "#355f8a",
        "linestyle": "-",
    },
    "depthwarm_standard": {
        "label": "Depth→Joint warm start + standard mutation",
        "short_label": "Depth warm\nStandard mutation",
        "initialization": "depth_warm",
        "mutation": "standard",
        "base_pattern": (
            "thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
        "color": "#16827d",
        "linestyle": "-",
    },
    "standard_interaction": {
        "label": "Standard initialization + interaction-aware mutation",
        "short_label": "Standard init\nInteraction-aware",
        "initialization": "standard",
        "mutation": "interaction_aware",
        "base_pattern": (
            "thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
        "color": "#75558c",
        "linestyle": "--",
    },
    "depthwarm_interaction": {
        "label": "Depth→Joint warm start + interaction-aware mutation",
        "short_label": "Depth warm\nInteraction-aware",
        "initialization": "depth_warm",
        "mutation": "interaction_aware",
        "base_pattern": (
            "thesis_depthwarm_interactionaware_joint_"
            "mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
        "color": "#d17a2e",
        "linestyle": "--",
    },
}
CONDITION_ORDER = tuple(CONDITION_METADATA)
PAIR_DEFINITIONS = (
    {
        "comparison": "warm_vs_standard_initialization__standard_mutation",
        "method": "depthwarm_standard",
        "baseline": "standard_standard",
        "question": "Warm versus standard initialization under standard mutation",
    },
    {
        "comparison": "warm_vs_standard_initialization__interaction_mutation",
        "method": "depthwarm_interaction",
        "baseline": "standard_interaction",
        "question": (
            "Warm versus standard initialization under interaction-aware mutation"
        ),
    },
    {
        "comparison": "interaction_vs_standard_mutation__standard_initialization",
        "method": "standard_interaction",
        "baseline": "standard_standard",
        "question": (
            "Interaction-aware versus standard mutation under standard initialization"
        ),
    },
    {
        "comparison": "interaction_vs_standard_mutation__warm_initialization",
        "method": "depthwarm_interaction",
        "baseline": "depthwarm_standard",
        "question": (
            "Interaction-aware versus standard mutation under depth warm-starting"
        ),
    },
)
CHECKPOINT_GENERATIONS = (5, 10, 20, 30, 40, 50)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final comparison and four cost-normalized convergence "
            "views for the matched G50 depth-warmstart experiment."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-stem", default="depth_warmstart_g50")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate CSV and Markdown artifacts without importing matplotlib.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_finite(value: Any, description: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not numeric: {value!r}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{description} is not finite: {value!r}")
    return numeric


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return require_finite(value, "optional metric")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_succeeded(path: Path) -> bool:
    if not path.is_file():
        return False
    return "exit_code=0" in path.read_text(encoding="utf-8").splitlines()


def run_dir_is_complete(path: Path) -> bool:
    return all(
        (path / filename).is_file()
        for filename in (
            "run_summary.json",
            "final_candidate.json",
            "generation_log.csv",
            "runtime.txt",
        )
    ) and runtime_succeeded(path / "runtime.txt")


def resolve_run_dir(runs_root: Path, base_run_id: str) -> Path:
    candidates = [runs_root / base_run_id]
    candidates.extend(
        runs_root / f"{base_run_id}_retry{retry}" for retry in range(1, 21)
    )
    complete = [path for path in candidates if run_dir_is_complete(path)]
    if not complete:
        raise FileNotFoundError(
            f"No valid completed run found for {base_run_id} under {runs_root}"
        )
    return complete[-1]


def parse_generation_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing generation log: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    generations = [int(row["generation"]) for row in rows]
    if generations != list(range(1, 51)):
        raise ValueError(
            f"Expected exactly generations 1..50 in {path}, got {generations}"
        )
    return rows


def normalize_depth_candidate(path: Path) -> dict[str, list[bool]]:
    candidate = load_json(path)
    if candidate.get("candidate_type") != "depth_only":
        raise ValueError(f"Expected a depth_only stage-one candidate in {path}")
    attention = candidate.get("attention_mask")
    mlp = candidate.get("mlp_mask")
    if not isinstance(attention, list) or not isinstance(mlp, list):
        raise ValueError(f"Missing structured depth masks in {path}")
    depth = {
        "attn": [bool(value) for value in attention],
        "mlp": [bool(value) for value in mlp],
    }
    if len(depth["attn"]) != 32 or len(depth["mlp"]) != 32:
        raise ValueError(f"Expected 32-layer stage-one masks in {path}")
    if sum(depth["attn"]) != 8 or sum(depth["mlp"]) != 8:
        raise ValueError(f"Expected eight attention and MLP drops in {path}")
    return depth


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    try:
        layer_position = parts.index("layers") + 1
        layer = int(parts[layer_position])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Cannot infer decoder layer from {module_name!r}") from error
    if not 0 <= layer < 32:
        raise ValueError(f"Decoder layer is outside 0..31: {module_name!r}")
    return layer


def validate_final_candidate(path: Path) -> None:
    candidate = load_json(path)
    if candidate.get("candidate_type") != "joint_depth_quant":
        raise ValueError(f"Expected a joint_depth_quant candidate in {path}")
    attention = candidate.get("attention_mask")
    mlp = candidate.get("mlp_mask")
    assignments = candidate.get("bitwidth_by_module")
    if not isinstance(attention, list) or not isinstance(mlp, list):
        raise ValueError(f"Missing joint depth masks in {path}")
    if len(attention) != 32 or len(mlp) != 32:
        raise ValueError(f"Expected 32-layer final masks in {path}")
    if sum(bool(value) for value in attention) != 8:
        raise ValueError(f"Final attention drop count is not eight in {path}")
    if sum(bool(value) for value in mlp) != 8:
        raise ValueError(f"Final MLP drop count is not eight in {path}")
    if not isinstance(assignments, dict) or len(assignments) != 32:
        raise ValueError(f"Expected exactly 32 q_proj assignments in {path}")
    active_levels: list[int] = []
    seen_layers: set[int] = set()
    for module_name, level in assignments.items():
        if not module_name.endswith(".self_attn.q_proj"):
            raise ValueError(f"Unexpected quantization module in {path}: {module_name}")
        layer = layer_index(module_name)
        if layer in seen_layers:
            raise ValueError(f"Duplicated q_proj layer assignment in {path}: {layer}")
        seen_layers.add(layer)
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError(f"Non-integer bit-width in {path}: {level!r}")
        if not bool(attention[layer]):
            active_levels.append(level)
    if seen_layers != set(range(32)):
        raise ValueError(f"q_proj assignments do not cover all layers in {path}")
    if sum(active_levels) != 3 * len(active_levels):
        raise ValueError(
            f"Final active q_proj budget is not exactly 3 bits in {path}: "
            f"sum={sum(active_levels)}, count={len(active_levels)}"
        )


def saved_command_tokens(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved command: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("exec "):
            return shlex.split(line)[1:]
    raise ValueError(f"Could not find an exec command in {path}")


def command_option(tokens: Sequence[str], option: str) -> str | None:
    positions = [index for index, token in enumerate(tokens) if token == option]
    if not positions:
        return None
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one {option} in saved command")
    position = positions[0]
    if position + 1 >= len(tokens):
        raise ValueError(f"Missing value after {option} in saved command")
    return tokens[position + 1]


def validate_common_configuration(
    summary: Mapping[str, Any],
    condition: str,
    seed: int,
    run_dir: Path,
) -> None:
    if summary.get("model_name") != MODEL_NAME:
        raise ValueError(f"Model mismatch for {run_dir}")
    if summary.get("dataset_calibration") != "wikitext2":
        raise ValueError(f"Calibration dataset mismatch for {run_dir}")
    if summary.get("dataset_eval") != ["wikitext2"]:
        raise ValueError(f"Evaluation dataset mismatch for {run_dir}")
    search = summary["search_config"]
    expected_search = {
        "generations": 50,
        "offspring": 16,
        "initial_candidates": 32,
        "initial_tokens": 512,
        "selection_tokens": [512, 2048, 8192],
        "selection_survivors": [8, 2, 1],
        "fitness_fn": "kl",
        "sequence_length": 1024,
        "calibration_tokens": 8192,
        "eval_tokens": 524288,
        "eval_every": 5,
        "eval_tokens_loaded_by_dataset": {"wikitext2": 333824},
        "seed": seed,
    }
    for key, expected in expected_search.items():
        if search.get(key) != expected:
            raise ValueError(
                f"{run_dir} has {key}={search.get(key)!r}, expected {expected!r}"
            )
    compression = summary["compression_config"]
    expected_compression = {
        "target_depth_sparsity": 0.25,
        "target_average_bitwidth": 3.0,
        "group_rule": "size",
        "active_quant_budget": True,
        "drop_entire_block": False,
        "bits_available": [2, 3, 4],
        "quant_weights_path": QUANT_WEIGHTS_PATH,
    }
    for key, expected in expected_compression.items():
        if compression.get(key) != expected:
            raise ValueError(
                f"{run_dir} has {key}={compression.get(key)!r}, expected {expected!r}"
            )
    command = saved_command_tokens(run_dir / "command.sh")
    if "--joint_aware_mutation" in command:
        raise ValueError(f"Deprecated joint-aware mutation is active in {run_dir}")
    for option in (
        "--adaptive_mutation",
        "--coarse_to_fine_mutation",
        "--drop_entire_block",
    ):
        if option in command:
            raise ValueError(f"Unexpected {option} in {run_dir}")
    expected_command_options = {
        "--model_name_or_path": MODEL_NAME,
        "--quant_weights_path": QUANT_WEIGHTS_PATH,
        "--drop_sparsity": "0.25",
        "--target_bitwidth": "3.0",
        "--calibration_data": "wikitext2",
        "--calibration_tokens": "8192",
        "--calibration_sequence_length": "1024",
        "--eval_every": "5",
        "--eval_datasets": "wikitext2",
        "--eval_tokens": "524288",
        "--eval_sequence_length": "1024",
        "--generations": "50",
        "--offspring": "16",
        "--initially_generated": "32",
        "--initial_tokens": "512",
        "--fitness_fn": "kl",
        "--group_rule": "size",
        "--step_size": "1",
        "--max_drop_mutations": "3",
        "--dtype": "float16",
        "--attn_implementation": "sdpa",
        "--seed": str(seed),
    }
    for option, expected in expected_command_options.items():
        observed = command_option(command, option)
        if observed != expected:
            raise ValueError(
                f"{run_dir} has {option}={observed!r}, expected {expected!r}"
            )
    if "--active_quant_budget" not in command:
        raise ValueError(f"Active quantization budget is missing in {run_dir}")
    if "--use_fast_tokenizer" not in command:
        raise ValueError(f"Fast tokenizer setting is missing in {run_dir}")
    expected_mutation = CONDITION_METADATA[condition]["mutation"]
    if expected_mutation == "interaction_aware":
        if command_option(command, "--joint_mutation_mode") != "interaction_aware":
            raise ValueError(f"Interaction-aware mode is missing in {run_dir}")
    else:
        observed_mode = command_option(command, "--joint_mutation_mode")
        if observed_mode not in (None, "standard"):
            raise ValueError(f"Unexpected joint mutation mode in {run_dir}")
    warm = CONDITION_METADATA[condition]["initialization"] == "depth_warm"
    if warm:
        if command_option(command, "--sequential_mode") != "depth_to_joint_warm":
            raise ValueError(f"Depth warm-start command is missing in {run_dir}")
        if (
            command_option(command, "--sequential_quant_initialization_policy")
            != "strict"
        ):
            raise ValueError(f"Strict warm-start policy is missing in {run_dir}")
    elif any(
        option in command
        for option in (
            "--sequential_mode",
            "--stage1_run_dir",
            "--stage1_candidate",
        )
    ):
        raise ValueError(f"Standard initialization has a stage-one source in {run_dir}")


def validate_warm_provenance(
    summary: Mapping[str, Any],
    seed: int,
    runs_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    stage1_run_id = f"thesis_medium_depth_mistral_s0.25_g20_o16_seed{seed}"
    stage1_dir = resolve_run_dir(runs_root, stage1_run_id)
    stage1_summary = load_json(stage1_dir / "run_summary.json")
    if stage1_summary.get("search_type") != "depth_only":
        raise ValueError(f"Stage-one search is not depth_only: {stage1_dir}")
    if stage1_summary.get("model_name") != MODEL_NAME:
        raise ValueError(f"Stage-one model mismatch: {stage1_dir}")
    if stage1_summary.get("dataset_calibration") != "wikitext2":
        raise ValueError(f"Stage-one calibration mismatch: {stage1_dir}")
    if stage1_summary.get("dataset_eval") != ["wikitext2"]:
        raise ValueError(f"Stage-one evaluation mismatch: {stage1_dir}")
    stage1_search = stage1_summary["search_config"]
    expected_stage1_search = {
        "generations": 20,
        "offspring": 16,
        "initial_candidates": 32,
        "initial_tokens": 512,
        "selection_tokens": [512, 2048, 8192],
        "selection_survivors": [8, 2, 1],
        "fitness_fn": "kl",
        "sequence_length": 1024,
        "calibration_tokens": 8192,
        "eval_tokens": 524288,
        "eval_every": 5,
        "eval_tokens_loaded_by_dataset": {"wikitext2": 333824},
        "seed": seed,
    }
    for key, expected in expected_stage1_search.items():
        if stage1_search.get(key) != expected:
            raise ValueError(
                f"{stage1_dir} has {key}={stage1_search.get(key)!r}, "
                f"expected {expected!r}"
            )
    stage1_compression = stage1_summary["compression_config"]
    if stage1_compression.get("target_depth_sparsity") != 0.25:
        raise ValueError(f"Stage-one depth sparsity mismatch: {stage1_dir}")
    if stage1_compression.get("drop_entire_block") is not False:
        raise ValueError(f"Stage-one depth mode mismatch: {stage1_dir}")
    depth = normalize_depth_candidate(stage1_dir / "final_candidate.json")
    expected_component_hash = stable_json_hash(depth)
    expected_initial_parent_hash = stable_json_hash(
        {"drop": depth, "quant": [[3] * 32]}
    )
    if summary.get("sequential_mode") != "depth_to_joint_warm":
        raise ValueError(f"Warm sequential mode is missing in {run_dir}")
    if summary.get("stage1_search_type") != "depth_only":
        raise ValueError(f"Warm stage-one search type is invalid in {run_dir}")
    if summary.get("stage1_candidate_hash") != expected_component_hash:
        raise ValueError(f"Stage-one component hash mismatch in {run_dir}")
    if summary.get("initial_parent_hash") != expected_initial_parent_hash:
        raise ValueError(f"Warm initial-parent hash mismatch in {run_dir}")
    if summary.get("initial_feasible_candidate_count") != 1:
        raise ValueError(
            f"Warm initialization did not evaluate one candidate: {run_dir}"
        )
    if summary.get("active_budget_valid") is not True:
        raise ValueError(f"Warm final active budget is invalid: {run_dir}")
    if summary.get("depth_counts_valid") is not True:
        raise ValueError(f"Warm final depth counts are invalid: {run_dir}")
    if summary.get("initial_repair_changed_gene_count") != 0:
        raise ValueError(f"Warm initialization unexpectedly changed genes: {run_dir}")
    stage1_path = Path(str(summary.get("stage1_run_dir")))
    if stage1_path.name != stage1_dir.name:
        raise ValueError(
            f"Warm stage-one run mismatch: summary={stage1_path}, "
            f"expected={stage1_dir.name}"
        )
    command = saved_command_tokens(run_dir / "command.sh")
    command_stage1_path = Path(str(command_option(command, "--stage1_run_dir")))
    if command_stage1_path.name != stage1_dir.name:
        raise ValueError(
            f"Warm command stage-one run mismatch: command={command_stage1_path}, "
            f"expected={stage1_dir.name}"
        )
    return {
        "stage1_run_id": stage1_dir.name,
        "stage1_runtime_seconds": require_finite(
            stage1_summary["final_metrics"]["runtime_seconds"],
            f"{stage1_dir} runtime",
        ),
        "stage1_summary": stage1_summary,
        "stage1_candidate_hash": expected_component_hash,
        "initial_parent_hash": expected_initial_parent_hash,
    }


def selection_cost(
    search_config: Mapping[str, Any],
    generations: int | None = None,
) -> dict[str, int]:
    generation_count = (
        int(search_config["generations"]) if generations is None else generations
    )
    offspring = int(search_config["offspring"])
    initial_candidates = int(
        search_config.get(
            "initial_candidates_evaluated",
            search_config["initial_candidates"],
        )
    )
    initial_tokens = int(search_config["initial_tokens"])
    survivors = [int(value) for value in search_config["selection_survivors"]]
    tokens = [int(value) for value in search_config["selection_tokens"]]
    if len(survivors) != len(tokens) or not survivors or survivors[-1] != 1:
        raise ValueError("Invalid selection schedule in run summary")
    evaluated_candidates_per_stage: list[int] = []
    for step in range(len(tokens)):
        if step == 0:
            candidates = offspring
        else:
            candidates = survivors[step - 1]
        if step == len(tokens) - 1:
            candidates += 1  # persistent parent is re-evaluated for elitism
        evaluated_candidates_per_stage.append(candidates)
    per_generation_candidates = sum(evaluated_candidates_per_stage)
    per_generation_tokens = sum(
        candidates * token_count
        for candidates, token_count in zip(evaluated_candidates_per_stage, tokens)
    )
    return {
        "initial_candidate_evaluations": initial_candidates,
        "initial_evaluated_tokens": initial_candidates * initial_tokens,
        "candidate_evaluations_per_generation": per_generation_candidates,
        "evaluated_tokens_per_generation": per_generation_tokens,
        "candidate_evaluations": (
            initial_candidates + generation_count * per_generation_candidates
        ),
        "evaluated_tokens": (
            initial_candidates * initial_tokens
            + generation_count * per_generation_tokens
        ),
    }


def run_row_and_convergence(
    condition: str,
    seed: int,
    run_dir: Path,
    runs_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = load_json(run_dir / "run_summary.json")
    validate_common_configuration(summary, condition, seed, run_dir)
    validate_final_candidate(run_dir / "final_candidate.json")
    metadata = CONDITION_METADATA[condition]
    warm = metadata["initialization"] == "depth_warm"
    if warm:
        stage1 = validate_warm_provenance(summary, seed, runs_root, run_dir)
    else:
        if summary.get("sequential_mode") not in (None, "none"):
            raise ValueError(f"Standard initialization is not standard in {run_dir}")
        stage1 = {
            "stage1_run_id": None,
            "stage1_runtime_seconds": 0.0,
            "stage1_summary": None,
            "stage1_candidate_hash": None,
            "initial_parent_hash": None,
        }

    search = summary["search_config"]
    stage2_cost = selection_cost(search)
    if warm:
        stage1_cost = selection_cost(stage1["stage1_summary"]["search_config"])
    else:
        stage1_cost = {
            "candidate_evaluations": 0,
            "evaluated_tokens": 0,
        }
    metrics = summary["final_metrics"]
    stage2_runtime_seconds = require_finite(
        metrics["runtime_seconds"], f"{run_dir} stage-two runtime"
    )
    stage1_runtime_seconds = float(stage1["stage1_runtime_seconds"])
    final_eval_tokens = int(
        search.get("eval_tokens_loaded_by_dataset", {}).get("wikitext2", 0)
    )
    row = {
        "condition": condition,
        "condition_label": metadata["label"],
        "initialization": metadata["initialization"],
        "mutation": metadata["mutation"],
        "seed": seed,
        "run_id": run_dir.name,
        "source_summary": str(run_dir / "run_summary.json"),
        "stage1_run_id": stage1["stage1_run_id"],
        "stage1_candidate_hash": stage1["stage1_candidate_hash"],
        "initial_parent_hash": stage1["initial_parent_hash"],
        "wikitext2_ppl": require_finite(
            metrics["wikitext2_ppl"], f"{run_dir} WikiText2 PPL"
        ),
        "best_search_fitness": require_finite(
            metrics["best_search_fitness"], f"{run_dir} fitness"
        ),
        "final_calibration_kl": require_finite(
            metrics["final_calibration_kl"], f"{run_dir} final KL"
        ),
        "train_ppl": require_finite(metrics["train_ppl"], f"{run_dir} train PPL"),
        "stage1_runtime_minutes": stage1_runtime_seconds / 60,
        "stage2_runtime_minutes": stage2_runtime_seconds / 60,
        "total_runtime_minutes": (stage1_runtime_seconds + stage2_runtime_seconds) / 60,
        "stage1_candidate_evaluations": stage1_cost["candidate_evaluations"],
        "stage2_candidate_evaluations": stage2_cost["candidate_evaluations"],
        "total_candidate_evaluations": (
            stage1_cost["candidate_evaluations"] + stage2_cost["candidate_evaluations"]
        ),
        "stage1_evaluated_tokens": stage1_cost["evaluated_tokens"],
        "stage2_evaluated_tokens": stage2_cost["evaluated_tokens"],
        "total_evaluated_tokens": (
            stage1_cost["evaluated_tokens"] + stage2_cost["evaluated_tokens"]
        ),
        "final_wikitext2_eval_tokens_loaded": final_eval_tokens,
        "initial_candidate_evaluations": stage2_cost["initial_candidate_evaluations"],
        "candidate_evaluations_per_generation": stage2_cost[
            "candidate_evaluations_per_generation"
        ],
        "evaluated_tokens_per_generation": stage2_cost[
            "evaluated_tokens_per_generation"
        ],
        "estimated_compression_ratio": require_finite(
            metrics["estimated_compression_ratio"], f"{run_dir} compression"
        ),
        "average_bitwidth_active": require_finite(
            metrics["average_bitwidth_active"], f"{run_dir} active bits"
        ),
        "active_budget_valid": (summary.get("active_budget_valid") if warm else True),
        "depth_counts_valid": (summary.get("depth_counts_valid") if warm else True),
    }

    generation_rows = parse_generation_log(run_dir / "generation_log.csv")
    convergence: list[dict[str, Any]] = []
    previous_runtime = -1.0
    for generation_row in generation_rows:
        generation = int(generation_row["generation"])
        runtime_seconds = require_finite(
            generation_row["runtime_seconds_cumulative"],
            f"{run_dir} generation {generation} runtime",
        )
        if runtime_seconds < previous_runtime:
            raise ValueError(f"Non-monotonic cumulative runtime in {run_dir}")
        previous_runtime = runtime_seconds
        generation_cost = selection_cost(search, generation)
        ppl = optional_float(generation_row.get("wikitext2_ppl"))
        if generation == 50:
            ppl = row["wikitext2_ppl"]
        convergence.append(
            {
                "condition": condition,
                "condition_label": metadata["label"],
                "initialization": metadata["initialization"],
                "mutation": metadata["mutation"],
                "seed": seed,
                "run_id": run_dir.name,
                "generation": generation,
                "best_search_fitness": require_finite(
                    generation_row["best_search_fitness"],
                    f"{run_dir} generation {generation} fitness",
                ),
                "wikitext2_ppl": ppl,
                "stage2_candidate_evaluations_cumulative": generation_cost[
                    "candidate_evaluations"
                ],
                "stage2_evaluated_tokens_cumulative": generation_cost[
                    "evaluated_tokens"
                ],
                "total_candidate_evaluations_cumulative": (
                    stage1_cost["candidate_evaluations"]
                    + generation_cost["candidate_evaluations"]
                ),
                "total_evaluated_tokens_cumulative": (
                    stage1_cost["evaluated_tokens"]
                    + generation_cost["evaluated_tokens"]
                ),
                "stage2_runtime_minutes_cumulative": runtime_seconds / 60,
                "total_runtime_minutes_cumulative": (
                    stage1_runtime_seconds + runtime_seconds
                )
                / 60,
            }
        )
    return row, convergence


def build_rows(
    runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        pattern = CONDITION_METADATA[condition]["base_pattern"]
        for seed in SEEDS:
            run_dir = resolve_run_dir(runs_root, pattern.format(seed=seed))
            run_row, convergence_rows = run_row_and_convergence(
                condition,
                seed,
                run_dir,
                runs_root,
            )
            runs.append(run_row)
            convergence.extend(convergence_rows)
    return runs, convergence


def mean(values: Iterable[float | int]) -> float:
    return statistics.mean(float(value) for value in values)


def sample_std(values: Iterable[float | int]) -> float:
    return statistics.stdev(float(value) for value in values)


def aggregate_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        rows = [row for row in run_rows if row["condition"] == condition]
        if len(rows) != 3:
            raise ValueError(f"Expected three rows for {condition}, got {len(rows)}")
        aggregate.append(
            {
                "condition": condition,
                "condition_label": CONDITION_METADATA[condition]["label"],
                "initialization": CONDITION_METADATA[condition]["initialization"],
                "mutation": CONDITION_METADATA[condition]["mutation"],
                "runs": len(rows),
                "wikitext2_ppl_mean": mean(row["wikitext2_ppl"] for row in rows),
                "wikitext2_ppl_sample_std": sample_std(
                    row["wikitext2_ppl"] for row in rows
                ),
                "wikitext2_ppl_min": min(row["wikitext2_ppl"] for row in rows),
                "wikitext2_ppl_max": max(row["wikitext2_ppl"] for row in rows),
                "best_search_fitness_mean": mean(
                    row["best_search_fitness"] for row in rows
                ),
                "best_search_fitness_sample_std": sample_std(
                    row["best_search_fitness"] for row in rows
                ),
                "stage1_runtime_minutes_mean": mean(
                    row["stage1_runtime_minutes"] for row in rows
                ),
                "stage2_runtime_minutes_mean": mean(
                    row["stage2_runtime_minutes"] for row in rows
                ),
                "stage2_runtime_minutes_sample_std": sample_std(
                    row["stage2_runtime_minutes"] for row in rows
                ),
                "total_runtime_minutes_mean": mean(
                    row["total_runtime_minutes"] for row in rows
                ),
                "total_runtime_minutes_sample_std": sample_std(
                    row["total_runtime_minutes"] for row in rows
                ),
                "stage1_candidate_evaluations": int(
                    rows[0]["stage1_candidate_evaluations"]
                ),
                "stage2_candidate_evaluations": int(
                    rows[0]["stage2_candidate_evaluations"]
                ),
                "total_candidate_evaluations": int(
                    rows[0]["total_candidate_evaluations"]
                ),
                "stage1_evaluated_tokens": int(rows[0]["stage1_evaluated_tokens"]),
                "stage2_evaluated_tokens": int(rows[0]["stage2_evaluated_tokens"]),
                "total_evaluated_tokens": int(rows[0]["total_evaluated_tokens"]),
                "initial_candidate_evaluations": int(
                    rows[0]["initial_candidate_evaluations"]
                ),
                "final_wikitext2_eval_tokens_loaded": int(
                    rows[0]["final_wikitext2_eval_tokens_loaded"]
                ),
            }
        )
    return aggregate


def paired_delta_rows(
    run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_condition_seed = {(row["condition"], row["seed"]): row for row in run_rows}
    paired: list[dict[str, Any]] = []
    for definition in PAIR_DEFINITIONS:
        method = definition["method"]
        baseline = definition["baseline"]
        for seed in SEEDS:
            method_row = by_condition_seed[(method, seed)]
            baseline_row = by_condition_seed[(baseline, seed)]
            ppl_delta = method_row["wikitext2_ppl"] - baseline_row["wikitext2_ppl"]
            paired.append(
                {
                    "comparison": definition["comparison"],
                    "question": definition["question"],
                    "method": method,
                    "method_label": CONDITION_METADATA[method]["label"],
                    "baseline": baseline,
                    "baseline_label": CONDITION_METADATA[baseline]["label"],
                    "seed": seed,
                    "method_wikitext2_ppl": method_row["wikitext2_ppl"],
                    "baseline_wikitext2_ppl": baseline_row["wikitext2_ppl"],
                    "delta_ppl_method_minus_baseline": ppl_delta,
                    "relative_delta_percent": (
                        100 * ppl_delta / baseline_row["wikitext2_ppl"]
                    ),
                    "stage2_runtime_delta_minutes": (
                        method_row["stage2_runtime_minutes"]
                        - baseline_row["stage2_runtime_minutes"]
                    ),
                    "total_runtime_delta_minutes": (
                        method_row["total_runtime_minutes"]
                        - baseline_row["total_runtime_minutes"]
                    ),
                    "stage2_candidate_evaluations_delta": (
                        method_row["stage2_candidate_evaluations"]
                        - baseline_row["stage2_candidate_evaluations"]
                    ),
                    "total_candidate_evaluations_delta": (
                        method_row["total_candidate_evaluations"]
                        - baseline_row["total_candidate_evaluations"]
                    ),
                    "stage2_evaluated_tokens_delta": (
                        method_row["stage2_evaluated_tokens"]
                        - baseline_row["stage2_evaluated_tokens"]
                    ),
                    "total_evaluated_tokens_delta": (
                        method_row["total_evaluated_tokens"]
                        - baseline_row["total_evaluated_tokens"]
                    ),
                    "winner": (
                        "method"
                        if ppl_delta < 0
                        else "baseline"
                        if ppl_delta > 0
                        else "tie"
                    ),
                }
            )
    return paired


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    names = list(rows[0]) if fieldnames is None else list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def paired_summary(
    paired: Sequence[Mapping[str, Any]],
    comparison: str,
) -> tuple[float, float, int]:
    rows = [row for row in paired if row["comparison"] == comparison]
    deltas = [float(row["delta_ppl_method_minus_baseline"]) for row in rows]
    return (
        statistics.mean(deltas),
        statistics.stdev(deltas),
        sum(delta < 0 for delta in deltas),
    )


def checkpoint_means(
    convergence: Sequence[Mapping[str, Any]],
    condition: str,
    generation: int,
) -> tuple[float, float | None]:
    rows = [
        row
        for row in convergence
        if row["condition"] == condition and row["generation"] == generation
    ]
    fitness = mean(row["best_search_fitness"] for row in rows)
    ppls = [row["wikitext2_ppl"] for row in rows if row["wikitext2_ppl"] is not None]
    return fitness, mean(ppls) if ppls else None


def trend_statement(
    convergence: Sequence[Mapping[str, Any]],
    warm_condition: str,
    standard_condition: str,
    mutation_label: str,
) -> str:
    _, warm20 = checkpoint_means(convergence, warm_condition, 20)
    _, standard20 = checkpoint_means(convergence, standard_condition, 20)
    _, warm50 = checkpoint_means(convergence, warm_condition, 50)
    _, standard50 = checkpoint_means(convergence, standard_condition, 50)
    if None in (warm20, standard20, warm50, standard50):
        return (
            f"The periodic PPL checkpoints are incomplete for {mutation_label}; "
            "inspect the convergence CSV."
        )
    delta20 = float(warm20) - float(standard20)
    delta50 = float(warm50) - float(standard50)
    if delta20 < 0 and delta50 < 0:
        relation = (
            "persists but narrows"
            if abs(delta50) < abs(delta20)
            else "persists and grows"
        )
    elif delta20 < 0 <= delta50:
        relation = "is an early advantage that reverses by generation 50"
    elif delta20 >= 0 > delta50:
        relation = "emerges only later in the search"
    else:
        relation = "is not observed at either checkpoint"
    return (
        f"Under {mutation_label}, the warm-start delta is "
        f"{delta20:+.3f} PPL at generation 20 and {delta50:+.3f} at "
        f"generation 50; the advantage {relation}."
    )


def build_markdown(
    run_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
    output_stem: str,
) -> str:
    aggregate_by_condition = {row["condition"]: row for row in aggregate}
    run_by_condition_seed = {(row["condition"], row["seed"]): row for row in run_rows}
    pair_summaries = {
        definition["comparison"]: paired_summary(paired, definition["comparison"])
        for definition in PAIR_DEFINITIONS
    }
    warm_standard_delta = pair_summaries[
        "warm_vs_standard_initialization__standard_mutation"
    ]
    warm_interaction_delta = pair_summaries[
        "warm_vs_standard_initialization__interaction_mutation"
    ]
    lines = [
        "# Depth Warm-Start G50 Experiment",
        "",
        (
            "This report tests whether depth warm-starting provides only an "
            "early convergence benefit or remains useful after 50 generations, "
            "and whether it combines with interaction-aware mutation. Lower "
            "WikiText2 perplexity (PPL) and KL fitness are better."
        ),
        "",
        "## Matched Experimental Design",
        "",
        (
            "All four conditions use `mistralai/Mistral-7B-v0.3`, the same "
            "q-projection reconstruction database, 25% separate attention/MLP "
            "depth sparsity, an exact 3-bit active budget with size grouping, "
            "50 stage-two generations, 16 offspring, WikiText2 calibration, "
            "and the `[512, 2048, 8192]` token / `[8, 2, 1]` survivor schedule."
        ),
        "",
        (
            "The standard-initialization conditions evaluate 32 initial "
            "candidates. Depth-warm conditions import the seed-matched G20 "
            "depth-only result and evaluate one exact combined initial "
            "candidate. This difference is intentional and is included in the "
            "candidate/token accounting."
        ),
        "",
        "## Final Quality",
        "",
        (
            "| Condition | Seeds | WikiText2 PPL mean ± SD | Final KL mean ± SD | "
            "Stage 2 min | Total min incl. depth stage 1 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        row = aggregate_by_condition[condition]
        lines.append(
            f"| {row['condition_label']} | {row['runs']} | "
            f"{fmt(row['wikitext2_ppl_mean'])} ± "
            f"{fmt(row['wikitext2_ppl_sample_std'])} | "
            f"{fmt(row['best_search_fitness_mean'], 4)} ± "
            f"{fmt(row['best_search_fitness_sample_std'], 4)} | "
            f"{fmt(row['stage2_runtime_minutes_mean'], 2)} | "
            f"{fmt(row['total_runtime_minutes_mean'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Final PPL",
            "",
            (
                "| Seed | Standard init + standard | Depth warm + standard | "
                "Standard init + interaction | Depth warm + interaction |"
            ),
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in SEEDS:
        values = [
            fmt(run_by_condition_seed[(condition, seed)]["wikitext2_ppl"])
            for condition in CONDITION_ORDER
        ]
        lines.append(f"| {seed} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Paired Seed Deltas",
            "",
            (
                "Delta is `method PPL − baseline PPL`; negative values favor "
                "the method named first."
            ),
            "",
            "| Question | Mean delta ± SD | Seed wins |",
            "| --- | ---: | ---: |",
        ]
    )
    for definition in PAIR_DEFINITIONS:
        delta, std, wins = pair_summaries[definition["comparison"]]
        lines.append(
            f"| {definition['question']} | {fmt(delta, signed=True)} ± "
            f"{fmt(std)} | {wins}/3 |"
        )

    lines.extend(
        [
            "",
            "## Early Versus Late Convergence",
            "",
            trend_statement(
                convergence,
                "depthwarm_standard",
                "standard_standard",
                "standard mutation",
            ),
            "",
            trend_statement(
                convergence,
                "depthwarm_interaction",
                "standard_interaction",
                "interaction-aware mutation",
            ),
            "",
            (
                "The four generated convergence views show the same trajectories "
                "against generation, cumulative stage-two candidate evaluations, "
                "cumulative stage-two fitness-token exposures, and cumulative "
                "stage-two runtime."
            ),
            "",
            "### Mean Checkpoints",
            "",
            "| Condition | Generation | Best KL fitness | WikiText2 PPL |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition in CONDITION_ORDER:
        for generation in CHECKPOINT_GENERATIONS:
            fitness, ppl = checkpoint_means(convergence, condition, generation)
            lines.append(
                f"| {CONDITION_METADATA[condition]['label']} | {generation} | "
                f"{fmt(fitness, 4)} | {fmt(ppl) if ppl is not None else ''} |"
            )

    lines.extend(
        [
            "",
            "## Search Cost",
            "",
            (
                "`Evaluated tokens` below counts calibration-token exposures in "
                "fitness evaluations at every selection stage, including parent "
                "re-evaluation for final-stage elitism. It is not the number of "
                "unique dataset tokens and does not include periodic final-PPL "
                "evaluation tokens."
            ),
            "",
            (
                "| Condition | Initial candidates | Stage 2 candidate evals | "
                "Stage 2 evaluated tokens | Stage 1 candidate evals | "
                "Stage 1 evaluated tokens | Total candidate evals | "
                "Total evaluated tokens |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in CONDITION_ORDER:
        row = aggregate_by_condition[condition]
        lines.append(
            f"| {row['condition_label']} | "
            f"{row['initial_candidate_evaluations']} | "
            f"{row['stage2_candidate_evaluations']:,} | "
            f"{row['stage2_evaluated_tokens']:,} | "
            f"{row['stage1_candidate_evaluations']:,} | "
            f"{row['stage1_evaluated_tokens']:,} | "
            f"{row['total_candidate_evaluations']:,} | "
            f"{row['total_evaluated_tokens']:,} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- All 12 final candidates contain exactly eight dropped attention and eight dropped MLP components.",
            "- Every final candidate satisfies the exact active 3-bit q-projection sum.",
            "- Warm runs use the expected seed-matched depth-only stage-one artifact.",
            "- Stage-one depth-component hashes match the imported summaries.",
            "- Warm initial-parent hashes match `loaded depth + uniform 3-bit q_proj` exactly.",
            "- No run uses the deprecated `--joint_aware_mutation` path.",
            "- Standard and interaction-aware mutation conditions were verified from saved commands.",
            "",
            "## Interpretation",
            "",
            (
                f"Under standard mutation, the final paired warm-start delta is "
                f"{warm_standard_delta[0]:+.3f} ± "
                f"{warm_standard_delta[1]:.3f} PPL with "
                f"{warm_standard_delta[2]}/3 seed wins."
            ),
            (
                f"Under interaction-aware mutation, the final paired warm-start "
                f"delta is {warm_interaction_delta[0]:+.3f} ± "
                f"{warm_interaction_delta[1]:.3f} PPL with "
                f"{warm_interaction_delta[2]}/3 seed wins."
            ),
            (
                "Use the generation-20 versus generation-50 deltas and the "
                "cost-normalized curves together: generation alone does not "
                "account for the 31-candidate initialization difference, while "
                "total pipeline cost additionally includes the depth-only search."
            ),
            "",
            "## Limitations",
            "",
            "- Three seeds support descriptive paired comparisons, not strong significance claims.",
            "- Runtime comes from restarted TU Wien DataLab sessions and is an approximate wall-clock measure.",
            "- Evaluated-token exposure is a search-cost proxy; model execution cost also depends on caching and batch behavior.",
            "- The conclusion applies to Mistral-7B, WikiText2, 25% depth sparsity, and q_proj-only active 3-bit quantization.",
            "",
            "## Generated Artifacts",
            "",
            f"- `results/{output_stem}_runs.csv`",
            f"- `results/{output_stem}_summary.csv`",
            f"- `results/{output_stem}_paired_deltas.csv`",
            f"- `results/{output_stem}_convergence.csv`",
            f"- `results/{output_stem}_comparison.md`",
            f"- `results/{output_stem}_convergence_generation.png`",
            f"- `results/{output_stem}_convergence_candidate_evaluations.png`",
            f"- `results/{output_stem}_convergence_evaluated_tokens.png`",
            f"- `results/{output_stem}_convergence_stage2_runtime.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def create_convergence_plot(
    path: Path,
    convergence: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title_suffix: str,
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "evopress-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (fitness_ax, ppl_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
    for condition in CONDITION_ORDER:
        metadata = CONDITION_METADATA[condition]
        condition_rows = [row for row in convergence if row["condition"] == condition]
        by_generation: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            by_generation[int(row["generation"])].append(row)
        for seed in SEEDS:
            seed_rows = [row for row in condition_rows if int(row["seed"]) == seed]
            fitness_ax.plot(
                [float(row[x_key]) for row in seed_rows],
                [float(row["best_search_fitness"]) for row in seed_rows],
                color=metadata["color"],
                linestyle=metadata["linestyle"],
                alpha=0.16,
                linewidth=1,
            )
            eval_rows = [row for row in seed_rows if row["wikitext2_ppl"] is not None]
            ppl_ax.plot(
                [float(row[x_key]) for row in eval_rows],
                [float(row["wikitext2_ppl"]) for row in eval_rows],
                color=metadata["color"],
                linestyle=metadata["linestyle"],
                alpha=0.16,
                linewidth=1,
            )
        generations = sorted(by_generation)
        mean_x = [
            mean(row[x_key] for row in by_generation[generation])
            for generation in generations
        ]
        mean_fitness = [
            mean(row["best_search_fitness"] for row in by_generation[generation])
            for generation in generations
        ]
        fitness_ax.plot(
            mean_x,
            mean_fitness,
            color=metadata["color"],
            linestyle=metadata["linestyle"],
            linewidth=2.4,
            label=metadata["label"],
        )
        eval_generations = [
            generation
            for generation in generations
            if any(
                row["wikitext2_ppl"] is not None for row in by_generation[generation]
            )
        ]
        ppl_ax.plot(
            [
                mean(row[x_key] for row in by_generation[generation])
                for generation in eval_generations
            ],
            [
                mean(
                    row["wikitext2_ppl"]
                    for row in by_generation[generation]
                    if row["wikitext2_ppl"] is not None
                )
                for generation in eval_generations
            ],
            color=metadata["color"],
            linestyle=metadata["linestyle"],
            marker="o",
            markersize=4,
            linewidth=2.4,
            label=metadata["label"],
        )

    fitness_ax.set_yscale("log")
    fitness_ax.set_xlabel(x_label)
    fitness_ax.set_ylabel("Best KL search fitness (log scale)")
    fitness_ax.set_title("Search fitness")
    fitness_ax.grid(alpha=0.25)
    ppl_ax.set_xlabel(x_label)
    ppl_ax.set_ylabel("WikiText2 perplexity")
    ppl_ax.set_title("Periodic evaluation")
    ppl_ax.grid(alpha=0.25)
    handles, labels = fitness_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle(
        f"Mistral-7B depth warm-start G50 convergence — {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_rows, convergence = build_rows(args.runs_root)
    aggregate = aggregate_rows(run_rows)
    paired = paired_delta_rows(run_rows)
    stem = args.output_stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / f"{stem}_runs.csv", run_rows)
    write_csv(args.output_dir / f"{stem}_summary.csv", aggregate)
    write_csv(args.output_dir / f"{stem}_paired_deltas.csv", paired)
    write_csv(args.output_dir / f"{stem}_convergence.csv", convergence)
    markdown = build_markdown(
        run_rows,
        aggregate,
        paired,
        convergence,
        stem,
    )
    (args.output_dir / f"{stem}_comparison.md").write_text(markdown, encoding="utf-8")
    if not args.no_plots:
        plot_specs = (
            (
                "generation",
                "Generation",
                "generation",
                "generation",
            ),
            (
                "stage2_candidate_evaluations_cumulative",
                "Cumulative stage-two candidate evaluations",
                "candidate evaluations",
                "candidate_evaluations",
            ),
            (
                "stage2_evaluated_tokens_cumulative",
                "Cumulative stage-two evaluated tokens",
                "fitness-token exposures",
                "evaluated_tokens",
            ),
            (
                "stage2_runtime_minutes_cumulative",
                "Cumulative stage-two runtime (minutes)",
                "stage-two runtime",
                "stage2_runtime",
            ),
        )
        for x_key, x_label, title_suffix, filename_suffix in plot_specs:
            create_convergence_plot(
                args.output_dir / f"{stem}_convergence_{filename_suffix}.png",
                convergence,
                x_key=x_key,
                x_label=x_label,
                title_suffix=title_suffix,
            )

    print(f"Wrote {len(run_rows)} run rows.")
    print(f"Wrote {len(aggregate)} aggregate rows.")
    print(f"Wrote {len(paired)} paired delta rows.")
    print(f"Wrote {len(convergence)} convergence rows.")
    print(f"Output stem: {args.output_dir / stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
