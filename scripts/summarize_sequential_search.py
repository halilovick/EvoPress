#!/usr/bin/env python3
"""Build the full Mistral sequential-search comparison deliverables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEQUENTIAL_MODES = (
    "depth_to_quant_frozen",
    "depth_to_joint_warm",
    "quant_to_depth_frozen",
    "quant_to_joint_warm",
)
SEEDS = (0, 1, 2)
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

METHOD_METADATA: dict[str, dict[str, str]] = {
    "depth_uniform": {
        "label": "Depth + uniform 3-bit q_proj",
        "short_label": "Depth +\nuniform quant",
        "category": "baseline",
        "comparison_role": "target-matched composition baseline",
    },
    "independent": {
        "label": "Independent depth + searched q_proj quant",
        "short_label": "Independent",
        "category": "baseline",
        "comparison_role": "target-matched sequential composition baseline",
    },
    "joint_g20": {
        "label": "Standard joint G20",
        "short_label": "Joint G20",
        "category": "baseline",
        "comparison_role": "same stage-two generation/offspring schedule",
    },
    "seq_depth_to_quant_frozen": {
        "label": "Depth → Quantization, frozen",
        "short_label": "Depth → Quant\nfrozen",
        "category": "sequential",
        "comparison_role": "new sequential variant",
    },
    "seq_depth_to_joint_warm": {
        "label": "Depth → Joint, warm-started",
        "short_label": "Depth → Joint\nwarm",
        "category": "sequential",
        "comparison_role": "new sequential variant",
    },
    "seq_quant_to_depth_frozen": {
        "label": "Quantization → Depth, frozen",
        "short_label": "Quant → Depth\nfrozen",
        "category": "sequential",
        "comparison_role": "new sequential variant",
    },
    "seq_quant_to_joint_warm": {
        "label": "Quantization → Joint, warm-started",
        "short_label": "Quant → Joint\nwarm",
        "category": "sequential",
        "comparison_role": "new sequential variant",
    },
    "joint_g50": {
        "label": "Standard joint G50",
        "short_label": "Joint G50",
        "category": "reference",
        "comparison_role": "larger single-search compute reference",
    },
    "interaction_aware_g50": {
        "label": "Interaction-aware joint G50",
        "short_label": "Interaction-\naware G50",
        "category": "reference",
        "comparison_role": "non-matched operator reference",
    },
}

METHOD_ORDER = tuple(METHOD_METADATA)
SEQUENTIAL_METHODS = tuple(
    method for method in METHOD_ORDER if method.startswith("seq_")
)
PAIR_BASELINES = (
    "depth_uniform",
    "independent",
    "joint_g20",
    "joint_g50",
    "interaction_aware_g50",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the 4-mode x 3-seed Mistral sequential-search matrix "
            "and its existing baselines."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-stem", default="sequential_search")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate CSV and Markdown deliverables without importing matplotlib.",
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
    if value is None or value == "":
        return None
    return require_finite(value, "optional metric")


def read_single_evaluation_ppl(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matching = [row for row in rows if row.get("dataset") == "wikitext2"]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one WikiText2 evaluation row in {path}, found {len(matching)}"
        )
    return require_finite(matching[0].get("ppl"), f"{path} WikiText2 PPL")


def read_runtime_seconds(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing runtime artifact: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if values.get("exit_code") != "0":
        raise ValueError(f"Run did not complete successfully according to {path}")
    return require_finite(values.get("runtime_seconds"), f"{path} runtime")


def read_generation_statistics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing generation log: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_types: Counter[str] = Counter()
    accepted = 0
    attempts = 0
    no_ops = 0
    duplicates = 0
    infeasible = 0
    generated_types: Counter[str] = Counter()
    for row in rows:
        if row.get("accepted_parent_replacement", "").lower() == "true":
            accepted += 1
        selected_types[row.get("selected_parent_mutation_type") or "unknown"] += 1
        raw_summary = row.get("mutation_summary")
        if not raw_summary:
            continue
        summary = json.loads(raw_summary)
        offspring = summary.get("offspring_generation", {})
        attempts += int(offspring.get("attempts", 0))
        no_ops += int(offspring.get("no_op_mutations", 0))
        duplicates += int(offspring.get("duplicate_candidates", 0))
        infeasible += int(offspring.get("infeasible_candidates", 0))
        for mutation_type, count in summary.get(
            "generated_offspring_by_type", {}
        ).items():
            generated_types[mutation_type] += int(count)
    return {
        "accepted_parent_replacements": accepted,
        "selected_parent_depth": selected_types["depth"],
        "selected_parent_quantization": selected_types["quantization"],
        "selected_parent_sequential_quantization": selected_types[
            "sequential_quantization"
        ],
        "selected_parent_fixed_quant_depth": selected_types["fixed_quant_depth"],
        "selected_parent_interaction_aware": selected_types["interaction_aware"],
        "selected_parent_parent": selected_types["parent"],
        "offspring_attempts": attempts,
        "offspring_no_ops": no_ops,
        "offspring_duplicates": duplicates,
        "offspring_infeasible": infeasible,
        "generated_depth": generated_types["depth"],
        "generated_quantization": generated_types["quantization"],
        "generated_sequential_quantization": generated_types["sequential_quantization"],
        "generated_fixed_quant_depth": generated_types["fixed_quant_depth"],
        "generated_interaction_aware": generated_types["interaction_aware"],
    }


def structured_run_row(
    *,
    method: str,
    seed: int,
    run_dir: Path,
    stage1_runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    summary = load_json(run_dir / "run_summary.json")
    metrics = summary["final_metrics"]
    quantization = summary["quantization_statistics"]
    model_size = summary["model_size_statistics"]
    search = summary["search_config"]
    if summary.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Unexpected model for {run_dir}: {summary.get('model_name')!r}"
        )
    if int(search["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch for {run_dir}: expected {seed}, got {search['seed']}"
        )
    runtime_seconds = require_finite(metrics["runtime_seconds"], f"{run_dir} runtime")
    row: dict[str, Any] = {
        "method": method,
        "method_label": METHOD_METADATA[method]["label"],
        "category": METHOD_METADATA[method]["category"],
        "comparison_role": METHOD_METADATA[method]["comparison_role"],
        "seed": seed,
        "run_id": summary["run_name"],
        "source_artifact": str(run_dir / "run_summary.json"),
        "wikitext2_ppl": require_finite(
            metrics["wikitext2_ppl"], f"{run_dir} WikiText2 PPL"
        ),
        "best_search_fitness": optional_float(metrics.get("best_search_fitness")),
        "train_ppl": optional_float(metrics.get("train_ppl")),
        "final_calibration_kl": optional_float(metrics.get("final_calibration_kl")),
        "stage1_runtime_minutes": stage1_runtime_seconds / 60,
        "stage2_runtime_minutes": runtime_seconds / 60,
        "total_pipeline_runtime_minutes": (stage1_runtime_seconds + runtime_seconds)
        / 60,
        "estimated_compression_ratio": require_finite(
            metrics["estimated_compression_ratio"],
            f"{run_dir} compression ratio",
        ),
        "estimated_weight_memory_mb": require_finite(
            model_size["estimated_weight_memory_mb"],
            f"{run_dir} weight memory",
        ),
        "average_bitwidth_active": optional_float(
            quantization.get("average_bitwidth_active")
        ),
        "average_bitwidth_total": require_finite(
            metrics["average_bitwidth_total"],
            f"{run_dir} effective bits",
        ),
        "active_parameter_ratio": require_finite(
            metrics["active_parameter_ratio"],
            f"{run_dir} active parameter ratio",
        ),
        "generations": int(search["generations"]),
        "offspring": int(search["offspring"]),
        "initial_candidates_evaluated": int(
            search.get("initial_candidates_evaluated", search["initial_candidates"])
        ),
        "active_budget_valid": summary.get("active_budget_valid"),
        "depth_counts_valid": summary.get("depth_counts_valid"),
        "frozen_depth_unchanged": summary.get("frozen_depth_unchanged"),
        "frozen_quant_unchanged": summary.get("frozen_quant_unchanged"),
        "stage1_candidate_hash": summary.get("stage1_candidate_hash"),
        "fixed_quant_legal_swaps_initial": None,
        "fixed_quant_legal_swaps_final": None,
    }
    row.update(read_generation_statistics(run_dir / "generation_log.csv"))
    if method.startswith("seq_"):
        expected_mode = method.removeprefix("seq_")
        if summary.get("sequential_mode") != expected_mode:
            raise ValueError(
                f"Sequential mode mismatch for {run_dir}: "
                f"{summary.get('sequential_mode')!r}"
            )
        expected_search = {
            "generations": 20,
            "offspring": 16,
            "calibration_tokens": 8192,
            "sequence_length": 1024,
            "selection_tokens": [512, 2048, 8192],
            "selection_survivors": [8, 2, 1],
        }
        for key, expected in expected_search.items():
            if search.get(key) != expected:
                raise ValueError(
                    f"Unexpected {key} for {run_dir}: "
                    f"{search.get(key)!r} != {expected!r}"
                )
        if summary.get("active_budget_valid") is not True:
            raise ValueError(f"Active budget is invalid for {run_dir}")
        if summary.get("depth_counts_valid") is not True:
            raise ValueError(f"Depth counts are invalid for {run_dir}")
        if expected_mode == "depth_to_quant_frozen":
            if summary.get("frozen_depth_unchanged") is not True:
                raise ValueError(f"Frozen depth changed for {run_dir}")
        elif expected_mode == "quant_to_depth_frozen":
            if summary.get("frozen_quant_unchanged") is not True:
                raise ValueError(f"Frozen quantization changed for {run_dir}")
            legal = summary.get("fixed_quant_legal_swap_count") or {}
            row["fixed_quant_legal_swaps_initial"] = int(legal.get("initial_parent", 0))
            row["fixed_quant_legal_swaps_final"] = int(legal.get("final_parent", 0))
            if (
                row["fixed_quant_legal_swaps_initial"] <= 0
                or row["fixed_quant_legal_swaps_final"] <= 0
            ):
                raise ValueError(f"No legal frozen-quant depth swaps for {run_dir}")
        elif expected_mode == "quant_to_joint_warm":
            if summary.get("initial_component_changed_by_repair") is not False:
                raise ValueError(f"Strict warm start was repaired for {run_dir}")
            if summary.get("initial_repair_changed_gene_count") != 0:
                raise ValueError(f"Strict warm start changed genes for {run_dir}")
        if not summary.get("stage1_candidate_hash"):
            raise ValueError(f"Missing stage-one provenance hash for {run_dir}")
    return row


def medium_grid_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing medium-grid run table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if row["method"] in {"uniform", "independent"}:
            result[(row["method"], int(row["seed"]))] = row
    return result


def composition_row(
    *,
    method: str,
    seed: int,
    run_dir: Path,
    medium_row: Mapping[str, str],
) -> dict[str, Any]:
    ppl = read_single_evaluation_ppl(run_dir / "evaluation_metrics.csv")
    recorded_ppl = require_finite(
        medium_row["wikitext2_ppl"], f"{method} seed {seed} medium-grid PPL"
    )
    if not math.isclose(ppl, recorded_ppl, abs_tol=0.001):
        raise ValueError(
            f"PPL mismatch for {run_dir}: evaluation={ppl}, table={recorded_ppl}"
        )
    runtime_seconds = read_runtime_seconds(run_dir / "runtime.txt")
    table_runtime = require_finite(
        medium_row["runtime_seconds"], f"{method} seed {seed} runtime"
    )
    if not math.isclose(runtime_seconds, table_runtime, abs_tol=1):
        raise ValueError(
            f"Runtime mismatch for {run_dir}: file={runtime_seconds}, "
            f"table={table_runtime}"
        )
    source_seconds = require_finite(
        medium_row["source_search_runtime_seconds"],
        f"{method} seed {seed} source runtime",
    )
    row: dict[str, Any] = {
        "method": "depth_uniform" if method == "uniform" else method,
        "method_label": METHOD_METADATA[
            "depth_uniform" if method == "uniform" else method
        ]["label"],
        "category": "baseline",
        "comparison_role": METHOD_METADATA[
            "depth_uniform" if method == "uniform" else method
        ]["comparison_role"],
        "seed": seed,
        "run_id": medium_row["run_id"],
        "source_artifact": str(run_dir / "evaluation_metrics.csv"),
        "wikitext2_ppl": ppl,
        "best_search_fitness": None,
        "train_ppl": None,
        "final_calibration_kl": None,
        "stage1_runtime_minutes": source_seconds / 60,
        "stage2_runtime_minutes": runtime_seconds / 60,
        "total_pipeline_runtime_minutes": (
            require_finite(
                medium_row["total_pipeline_runtime_seconds"],
                f"{method} seed {seed} total runtime",
            )
            / 60
        ),
        "estimated_compression_ratio": require_finite(
            medium_row["estimated_compression_ratio"],
            f"{method} seed {seed} compression",
        ),
        "estimated_weight_memory_mb": require_finite(
            medium_row["estimated_weight_memory_mb"],
            f"{method} seed {seed} memory",
        ),
        "average_bitwidth_active": optional_float(
            medium_row["average_bitwidth_active"]
        ),
        "average_bitwidth_total": require_finite(
            medium_row["average_bitwidth_total"],
            f"{method} seed {seed} effective bits",
        ),
        "active_parameter_ratio": require_finite(
            medium_row["active_parameter_ratio"],
            f"{method} seed {seed} active ratio",
        ),
        "generations": None,
        "offspring": None,
        "initial_candidates_evaluated": None,
        "active_budget_valid": None,
        "depth_counts_valid": None,
        "frozen_depth_unchanged": None,
        "frozen_quant_unchanged": None,
        "stage1_candidate_hash": None,
        "fixed_quant_legal_swaps_initial": None,
        "fixed_quant_legal_swaps_final": None,
    }
    row.update(
        {
            key: None
            for key in (
                "accepted_parent_replacements",
                "selected_parent_depth",
                "selected_parent_quantization",
                "selected_parent_sequential_quantization",
                "selected_parent_fixed_quant_depth",
                "selected_parent_interaction_aware",
                "selected_parent_parent",
                "offspring_attempts",
                "offspring_no_ops",
                "offspring_duplicates",
                "offspring_infeasible",
                "generated_depth",
                "generated_quantization",
                "generated_sequential_quantization",
                "generated_fixed_quant_depth",
                "generated_interaction_aware",
            )
        }
    )
    return row


def build_run_rows(runs_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for direction in ("depth", "quant"):
        for seed in SEEDS:
            if direction == "depth":
                run_id = f"thesis_medium_depth_mistral_s0.25_g20_o16_seed{seed}"
            else:
                run_id = "thesis_medium_quant_mistral_qproj3.0_" f"g20_o16_seed{seed}"
            stage1_summaries[(direction, seed)] = load_json(
                runs_root / run_id / "run_summary.json"
            )

    for mode in SEQUENTIAL_MODES:
        method = f"seq_{mode}"
        direction = "depth" if mode.startswith("depth_") else "quant"
        for seed in SEEDS:
            run_id = (
                f"thesis_sequential_{mode}_"
                f"mistral_s0.25_qproj3.0_g20_o16_seed{seed}"
            )
            stage1_runtime = require_finite(
                stage1_summaries[(direction, seed)]["final_metrics"]["runtime_seconds"],
                f"{direction} stage-one seed {seed} runtime",
            )
            rows.append(
                structured_run_row(
                    method=method,
                    seed=seed,
                    run_dir=runs_root / run_id,
                    stage1_runtime_seconds=stage1_runtime,
                )
            )

    structured_baselines = {
        "joint_g20": ("thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed{seed}"),
        "joint_g50": (
            "thesis_compute_matched_joint_" "mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
        "interaction_aware_g50": (
            "thesis_interactionaware_joint_" "mistral_s0.25_qproj3.0_g50_o16_seed{seed}"
        ),
    }
    for method, pattern in structured_baselines.items():
        for seed in SEEDS:
            rows.append(
                structured_run_row(
                    method=method,
                    seed=seed,
                    run_dir=runs_root / pattern.format(seed=seed),
                )
            )

    medium_rows = medium_grid_rows(results_dir / "mistral_medium_runs.csv")
    composition_patterns = {
        "uniform": (
            "thesis_medium_depth_uniform_quant_" "mistral_s0.25_qproj3.0_seed{seed}"
        ),
        "independent": (
            "thesis_medium_independent_depth_quant_" "mistral_s0.25_qproj3.0_seed{seed}"
        ),
    }
    for method, pattern in composition_patterns.items():
        for seed in SEEDS:
            rows.append(
                composition_row(
                    method=method,
                    seed=seed,
                    run_dir=runs_root / pattern.format(seed=seed),
                    medium_row=medium_rows[(method, seed)],
                )
            )
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (order[row["method"]], row["seed"]))


def mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def sample_std(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.stdev(finite) if len(finite) >= 2 else None


def aggregate_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in run_rows if row["method"] == method]
        if len(method_rows) != len(SEEDS):
            raise ValueError(
                f"Expected {len(SEEDS)} rows for {method}, got {len(method_rows)}"
            )
        ppls = [row["wikitext2_ppl"] for row in method_rows]
        aggregate.append(
            {
                "method": method,
                "method_label": METHOD_METADATA[method]["label"],
                "category": METHOD_METADATA[method]["category"],
                "comparison_role": METHOD_METADATA[method]["comparison_role"],
                "runs": len(method_rows),
                "wikitext2_ppl_mean": mean(ppls),
                "wikitext2_ppl_sample_std": sample_std(ppls),
                "wikitext2_ppl_min": min(ppls),
                "wikitext2_ppl_max": max(ppls),
                "best_search_fitness_mean": mean(
                    row["best_search_fitness"] for row in method_rows
                ),
                "best_search_fitness_sample_std": sample_std(
                    row["best_search_fitness"] for row in method_rows
                ),
                "stage1_runtime_minutes_mean": mean(
                    row["stage1_runtime_minutes"] for row in method_rows
                ),
                "stage2_runtime_minutes_mean": mean(
                    row["stage2_runtime_minutes"] for row in method_rows
                ),
                "total_pipeline_runtime_minutes_mean": mean(
                    row["total_pipeline_runtime_minutes"] for row in method_rows
                ),
                "total_pipeline_runtime_minutes_sample_std": sample_std(
                    row["total_pipeline_runtime_minutes"] for row in method_rows
                ),
                "estimated_compression_ratio_mean": mean(
                    row["estimated_compression_ratio"] for row in method_rows
                ),
                "average_bitwidth_active_mean": mean(
                    row["average_bitwidth_active"] for row in method_rows
                ),
                "average_bitwidth_total_mean": mean(
                    row["average_bitwidth_total"] for row in method_rows
                ),
                "accepted_parent_replacements_mean": mean(
                    row["accepted_parent_replacements"] for row in method_rows
                ),
                "offspring_no_ops_total": sum(
                    row["offspring_no_ops"] or 0 for row in method_rows
                ),
                "offspring_duplicates_total": sum(
                    row["offspring_duplicates"] or 0 for row in method_rows
                ),
                "offspring_infeasible_total": sum(
                    row["offspring_infeasible"] or 0 for row in method_rows
                ),
            }
        )
    return aggregate


def paired_delta_rows(
    run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_method_seed = {(row["method"], row["seed"]): row for row in run_rows}
    paired: list[dict[str, Any]] = []
    for method in SEQUENTIAL_METHODS:
        for baseline in PAIR_BASELINES:
            for seed in SEEDS:
                method_row = by_method_seed[(method, seed)]
                baseline_row = by_method_seed[(baseline, seed)]
                method_ppl = method_row["wikitext2_ppl"]
                baseline_ppl = baseline_row["wikitext2_ppl"]
                delta = method_ppl - baseline_ppl
                paired.append(
                    {
                        "method": method,
                        "method_label": METHOD_METADATA[method]["label"],
                        "baseline": baseline,
                        "baseline_label": METHOD_METADATA[baseline]["label"],
                        "baseline_comparison_role": METHOD_METADATA[baseline][
                            "comparison_role"
                        ],
                        "seed": seed,
                        "method_wikitext2_ppl": method_ppl,
                        "baseline_wikitext2_ppl": baseline_ppl,
                        "delta_ppl_method_minus_baseline": delta,
                        "relative_delta_percent": 100 * delta / baseline_ppl,
                        "winner": (
                            "method"
                            if delta < 0
                            else "baseline" if delta > 0 else "tie"
                        ),
                    }
                )
    return paired


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or value == "":
        return ""
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def paired_summary(
    paired_rows: Sequence[Mapping[str, Any]],
    method: str,
    baseline: str,
) -> tuple[float, float, int]:
    rows = [
        row
        for row in paired_rows
        if row["method"] == method and row["baseline"] == baseline
    ]
    deltas = [row["delta_ppl_method_minus_baseline"] for row in rows]
    return (
        statistics.mean(deltas),
        statistics.stdev(deltas),
        sum(delta < 0 for delta in deltas),
    )


def build_markdown(
    run_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> str:
    aggregate_by_method = {row["method"]: row for row in aggregate}
    run_by_method_seed = {(row["method"], row["seed"]): row for row in run_rows}
    best_method = min(
        SEQUENTIAL_METHODS,
        key=lambda method: aggregate_by_method[method]["wikitext2_ppl_mean"],
    )
    best = aggregate_by_method[best_method]
    independent_delta, independent_delta_std, independent_wins = paired_summary(
        paired, best_method, "independent"
    )
    g20_delta, g20_delta_std, g20_wins = paired_summary(
        paired, best_method, "joint_g20"
    )
    g50_delta, g50_delta_std, g50_wins = paired_summary(
        paired, best_method, "joint_g50"
    )
    independent = aggregate_by_method["independent"]
    joint_g20 = aggregate_by_method["joint_g20"]
    joint_g50 = aggregate_by_method["joint_g50"]
    interaction_g50 = aggregate_by_method["interaction_aware_g50"]
    improvement_vs_independent_percent = (
        -100 * independent_delta / independent["wikitext2_ppl_mean"]
    )
    improvement_vs_g20_percent = -100 * g20_delta / joint_g20["wikitext2_ppl_mean"]
    runtime_reduction_vs_independent_percent = 100 * (
        1
        - best["total_pipeline_runtime_minutes_mean"]
        / independent["total_pipeline_runtime_minutes_mean"]
    )
    runtime_reduction_vs_g50_percent = 100 * (
        1
        - best["total_pipeline_runtime_minutes_mean"]
        / joint_g50["total_pipeline_runtime_minutes_mean"]
    )
    lines = [
        "# Sequential Initialization Search Comparison",
        "",
        (
            "This report compares the four sequential-initialization variants "
            "against the existing Mistral q-projection baselines. Lower "
            "WikiText2 perplexity (PPL) is better."
        ),
        "",
        "## Scope and Validation",
        "",
        (
            "All sequential runs use `mistralai/Mistral-7B-v0.3`, 25% separate "
            "attention/MLP depth sparsity, a 3-bit active q-projection budget, "
            "`group_rule=size`, 20 stage-two generations, 16 offspring, 32 "
            "requested initial candidates, and seeds 0–2."
        ),
        "",
        (
            "The generator verified all 12 sequential summaries, exact depth "
            "counts, active quantization budgets, stage-one provenance hashes, "
            "and frozen-component invariants. Quantization-first warm starts "
            "used strict initialization and changed zero imported genes before "
            "initial selection."
        ),
        "",
        (
            "Depth-first modes evaluate one exact imported-depth combined "
            "candidate during stage-two initialization. Quantization-first "
            "modes generate and evaluate 32 exact feasible depth masks. The "
            "ordinary joint G20 baseline also evaluates 32 initial candidates."
        ),
        "",
        "## Executive Result",
        "",
        (
            f"**{METHOD_METADATA[best_method]['label']} is the best sequential "
            f"variant at {fmt(best['wikitext2_ppl_mean'])} ± "
            f"{fmt(best['wikitext2_ppl_sample_std'])} PPL.**"
        ),
        "",
        (
            f"Its paired mean delta is {fmt(independent_delta, signed=True)} PPL "
            f"versus independent composition ({independent_wins}/3 seed wins), "
            f"{fmt(g20_delta, signed=True)} versus standard joint G20 "
            f"({g20_wins}/3 wins), and {fmt(g50_delta, signed=True)} versus "
            f"standard joint G50 ({g50_wins}/3 wins)."
        ),
        "",
        (
            f"This is a {improvement_vs_independent_percent:.2f}% mean PPL "
            f"improvement over independent composition and a "
            f"{improvement_vs_g20_percent:.2f}% improvement over standard "
            f"joint G20. Interaction-aware G50 reaches the lowest reference "
            f"mean ({interaction_g50['wikitext2_ppl_mean']:.3f}), but uses a "
            "different mutation operator and a 50-generation single-search "
            "budget."
        ),
        "",
        "## Aggregate Comparison",
        "",
        (
            "| Method | Role | Seeds | WikiText2 PPL mean ± SD | "
            "Stage 1 min | Stage 2/final job min | End-to-end min |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        row = aggregate_by_method[method]
        lines.append(
            "| "
            + " | ".join(
                (
                    row["method_label"],
                    row["comparison_role"],
                    str(row["runs"]),
                    (
                        f"{fmt(row['wikitext2_ppl_mean'])} ± "
                        f"{fmt(row['wikitext2_ppl_sample_std'])}"
                    ),
                    fmt(row["stage1_runtime_minutes_mean"], 2),
                    fmt(row["stage2_runtime_minutes_mean"], 2),
                    fmt(row["total_pipeline_runtime_minutes_mean"], 2),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Quality",
            "",
            (
                "| Seed | Depth→Quant frozen | Depth→Joint warm | "
                "Quant→Depth frozen | Quant→Joint warm | Independent | "
                "Joint G20 | Joint G50 | Interaction-aware G50 |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seed_methods = (
        "seq_depth_to_quant_frozen",
        "seq_depth_to_joint_warm",
        "seq_quant_to_depth_frozen",
        "seq_quant_to_joint_warm",
        "independent",
        "joint_g20",
        "joint_g50",
        "interaction_aware_g50",
    )
    for seed in SEEDS:
        values = [
            fmt(run_by_method_seed[(method, seed)]["wikitext2_ppl"])
            for method in seed_methods
        ]
        lines.append(f"| {seed} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Paired Mean Deltas",
            "",
            (
                "Delta is `sequential PPL − baseline PPL`; negative values favor "
                "the sequential method."
            ),
            "",
            ("| Sequential method | Baseline | Mean delta ± SD | " "Seed wins |"),
            "| --- | --- | ---: | ---: |",
        ]
    )
    for method in SEQUENTIAL_METHODS:
        for baseline in PAIR_BASELINES:
            delta_mean, delta_std, wins = paired_summary(paired, method, baseline)
            lines.append(
                f"| {METHOD_METADATA[method]['label']} | "
                f"{METHOD_METADATA[baseline]['label']} | "
                f"{fmt(delta_mean, signed=True)} ± {fmt(delta_std)} | "
                f"{wins}/3 |"
            )

    lines.extend(
        [
            "",
            "## Search-Dynamics Diagnostics",
            "",
            (
                "| Sequential method | Accepted replacements/run | No-op "
                "offspring | Duplicate offspring | Infeasible offspring |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in SEQUENTIAL_METHODS:
        row = aggregate_by_method[method]
        lines.append(
            f"| {row['method_label']} | "
            f"{fmt(row['accepted_parent_replacements_mean'], 1)} | "
            f"{row['offspring_no_ops_total']} | "
            f"{row['offspring_duplicates_total']} | "
            f"{row['offspring_infeasible_total']} |"
        )

    legal_counts = [
        run_by_method_seed[("seq_quant_to_depth_frozen", seed)][
            "fixed_quant_legal_swaps_initial"
        ]
        for seed in SEEDS
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- Depth → Joint warm-starting improves on independent "
                f"composition by {-independent_delta:.3f} PPL on average and "
                f"wins {independent_wins}/3 paired seeds. It also has the "
                f"lowest sample SD among the sequential variants."
            ),
            (
                "- Depth → Quantization frozen is effectively tied with the "
                "depth-plus-uniform and independent controls, indicating little "
                "additional gain from optimizing only q-projection assignments "
                "after fixing a strong depth solution."
            ),
            (
                "- Quantization → Depth frozen is the most variable variant. "
                f"Its initial parents expose {min(legal_counts)}–"
                f"{max(legal_counts)} legal contribution-compatible swaps, but "
                "the exact-budget frozen neighborhood still constrains which "
                "structural exchanges are reachable."
            ),
            (
                "- Quantization → Joint warm-starting is close to standard "
                "joint G20 on mean PPL, but does not show the consistent benefit "
                "observed for depth-first warm-starting."
            ),
            (
                f"- Standard joint G50 is {g50_delta:.3f} PPL better than the "
                "best sequential variant on average. It is a larger single "
                "search, not an exact nominal-budget match to a two-stage "
                "sequential pipeline."
            ),
            (
                "- Interaction-aware G50 is reported as a separate method "
                "reference. Its 50-generation budget and mutation operator "
                "differ from the G20 standard operator used in every sequential "
                "stage-two run."
            ),
            "",
            "## Compute Accounting",
            "",
            (
                "Sequential stage-two runtime alone is not the full cost. The "
                "end-to-end column adds the seed-matched stage-one search that "
                "produced the imported candidate. This treats warm starts and "
                "frozen baselines as complete pipelines even though the tracked "
                "experiments reused already-computed stage-one artifacts."
            ),
            "",
            (
                "Standard joint G20 matches the stage-two generation and "
                "offspring schedule, but it omits stage-one cost and its "
                "initialization count differs from the one-candidate "
                "depth-first variants. Standard joint G50 is a larger "
                "single-search compute reference. Neither is an exact match "
                "for the number and type of candidate evaluations in a "
                "two-stage sequential pipeline."
            ),
            "",
            (
                f"Depth → Joint warm-starting averages "
                f"{best['total_pipeline_runtime_minutes_mean']:.2f} end-to-end "
                f"minutes, compared with "
                f"{independent['total_pipeline_runtime_minutes_mean']:.2f} for "
                f"independent composition and "
                f"{joint_g50['total_pipeline_runtime_minutes_mean']:.2f} for "
                f"standard joint G50. These observed differences are "
                f"{runtime_reduction_vs_independent_percent:.1f}% and "
                f"{runtime_reduction_vs_g50_percent:.1f}% respectively, but "
                "hardware/session variation prevents treating them as precise "
                "operator-cost measurements."
            ),
            "",
            "## Methodological Limits",
            "",
            "- There are only three seeds; the report presents descriptive mean, sample SD, and paired deltas without significance claims.",
            "- Quality is currently measured on WikiText2 only. Broader calibration datasets and downstream LM-eval tasks are not included.",
            "- The quantization scope covers only `q_proj`, so it represents a small part of total model parameters.",
            "- Runtime comparisons span restarted TU Wien DataLab sessions and should be interpreted as approximate wall-clock evidence.",
            "- Compression and memory values are theoretical weight accounting, not measured compressed checkpoint or inference memory.",
            "",
            "## Presentation-Ready Conclusion",
            "",
            "- Sequential initialization is direction-dependent.",
            "- A strong depth solution is a useful warm start for joint refinement.",
            "- Freezing quantization restricts depth exploration and produces high seed variance.",
            "- Warm-starting from quantization alone does not materially improve standard joint G20.",
            "- Depth → Joint warm-starting is the best sequential variant, but standard joint G50 remains better on mean PPL.",
            "- Report stage-two and end-to-end search cost separately.",
            "",
            "## Generated Artifacts",
            "",
            "- `results/sequential_search_runs.csv`: seed-level metrics, provenance, invariants, runtime accounting, and mutation diagnostics.",
            "- `results/sequential_search_summary.csv`: method-level mean and sample standard deviation.",
            "- `results/sequential_search_paired_deltas.csv`: paired seed-level deltas for every sequential variant and baseline.",
            "- `results/sequential_search_comparison.png`: presentation-ready quality comparison.",
            "- `results/sequential_search_comparison.md`: this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_plot(
    path: Path,
    run_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "evopress-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_by_method = {row["method"]: row for row in aggregate}
    by_method = {
        method: [row for row in run_rows if row["method"] == method]
        for method in METHOD_ORDER
    }
    sequential_colors = {
        "seq_depth_to_quant_frozen": "#7aa6c2",
        "seq_depth_to_joint_warm": "#167c80",
        "seq_quant_to_depth_frozen": "#d98e32",
        "seq_quant_to_joint_warm": "#b85c74",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )

    ax = axes[0]
    x_positions = list(range(len(SEQUENTIAL_METHODS)))
    means = [
        aggregate_by_method[method]["wikitext2_ppl_mean"]
        for method in SEQUENTIAL_METHODS
    ]
    stds = [
        aggregate_by_method[method]["wikitext2_ppl_sample_std"]
        for method in SEQUENTIAL_METHODS
    ]
    colors = [sequential_colors[method] for method in SEQUENTIAL_METHODS]
    ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.82,
        edgecolor="#333333",
        linewidth=0.8,
        zorder=2,
    )
    offsets = (-0.10, 0.0, 0.10)
    for x, method in zip(x_positions, SEQUENTIAL_METHODS):
        for offset, row in zip(offsets, by_method[method]):
            ax.scatter(
                x + offset,
                row["wikitext2_ppl"],
                color="#1f1f1f",
                s=28,
                zorder=3,
            )
    ax.set_xticks(
        x_positions,
        [METHOD_METADATA[method]["short_label"] for method in SEQUENTIAL_METHODS],
    )
    ax.set_ylabel("WikiText2 perplexity (lower is better)")
    ax.set_title("A. Sequential initialization variants")
    ax.grid(axis="y", alpha=0.25, zorder=0)

    ax = axes[1]
    comparison_methods = (
        "depth_uniform",
        "independent",
        "joint_g20",
        "seq_depth_to_joint_warm",
        "joint_g50",
        "interaction_aware_g50",
    )
    comparison_colors = (
        "#b9bec5",
        "#8d99a6",
        "#54789c",
        sequential_colors["seq_depth_to_joint_warm"],
        "#355a82",
        "#76568c",
    )
    x_positions = list(range(len(comparison_methods)))
    means = [
        aggregate_by_method[method]["wikitext2_ppl_mean"]
        for method in comparison_methods
    ]
    stds = [
        aggregate_by_method[method]["wikitext2_ppl_sample_std"]
        for method in comparison_methods
    ]
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color=comparison_colors,
        alpha=0.84,
        edgecolor="#333333",
        linewidth=0.8,
        zorder=2,
    )
    for x, method in zip(x_positions, comparison_methods):
        for offset, row in zip(offsets, by_method[method]):
            ax.scatter(
                x + offset,
                row["wikitext2_ppl"],
                color="#1f1f1f",
                s=26,
                zorder=3,
            )
    for bar, value, method in zip(bars, means, comparison_methods):
        label_y = max(row["wikitext2_ppl"] for row in by_method[method]) + 0.10
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(
        x_positions,
        [METHOD_METADATA[method]["short_label"] for method in comparison_methods],
    )
    ax.set_title("B. Best sequential variant and existing references")
    ax.grid(axis="y", alpha=0.25, zorder=0)

    lower = min(
        row["wikitext2_ppl"] for row in run_rows if row["method"] in METHOD_ORDER
    )
    upper = max(
        row["wikitext2_ppl"] for row in run_rows if row["method"] in METHOD_ORDER
    )
    axes[0].set_ylim(max(0, lower - 0.7), upper + 1.1)
    comparison_values = [
        row["wikitext2_ppl"] for row in run_rows if row["method"] in comparison_methods
    ]
    axes[1].set_ylim(min(comparison_values) - 0.7, max(comparison_values) + 1.0)

    fig.suptitle(
        "Mistral-7B sequential initialization at 25% depth sparsity "
        "and 3-bit active q_proj budget",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        (
            "Bars show mean ± sample SD; black dots show seeds 0–2. "
            "G50 references use a larger single-search budget than G20 stage two."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_rows = build_run_rows(args.runs_root, args.results_dir)
    aggregate = aggregate_rows(run_rows)
    paired = paired_delta_rows(run_rows)
    stem = args.output_stem

    run_fields = [
        "method",
        "method_label",
        "category",
        "comparison_role",
        "seed",
        "run_id",
        "source_artifact",
        "wikitext2_ppl",
        "best_search_fitness",
        "train_ppl",
        "final_calibration_kl",
        "stage1_runtime_minutes",
        "stage2_runtime_minutes",
        "total_pipeline_runtime_minutes",
        "estimated_compression_ratio",
        "estimated_weight_memory_mb",
        "average_bitwidth_active",
        "average_bitwidth_total",
        "active_parameter_ratio",
        "generations",
        "offspring",
        "initial_candidates_evaluated",
        "active_budget_valid",
        "depth_counts_valid",
        "frozen_depth_unchanged",
        "frozen_quant_unchanged",
        "stage1_candidate_hash",
        "fixed_quant_legal_swaps_initial",
        "fixed_quant_legal_swaps_final",
        "accepted_parent_replacements",
        "selected_parent_depth",
        "selected_parent_quantization",
        "selected_parent_sequential_quantization",
        "selected_parent_fixed_quant_depth",
        "selected_parent_interaction_aware",
        "selected_parent_parent",
        "offspring_attempts",
        "offspring_no_ops",
        "offspring_duplicates",
        "offspring_infeasible",
        "generated_depth",
        "generated_quantization",
        "generated_sequential_quantization",
        "generated_fixed_quant_depth",
        "generated_interaction_aware",
    ]
    aggregate_fields = list(aggregate[0])
    paired_fields = list(paired[0])
    write_csv(args.results_dir / f"{stem}_runs.csv", run_rows, run_fields)
    write_csv(
        args.results_dir / f"{stem}_summary.csv",
        aggregate,
        aggregate_fields,
    )
    write_csv(
        args.results_dir / f"{stem}_paired_deltas.csv",
        paired,
        paired_fields,
    )
    markdown = build_markdown(run_rows, aggregate, paired)
    (args.results_dir / f"{stem}_comparison.md").write_text(markdown, encoding="utf-8")
    if not args.no_plots:
        create_plot(
            args.results_dir / f"{stem}_comparison.png",
            run_rows,
            aggregate,
        )

    print(f"Wrote {len(run_rows)} seed-level rows.")
    print(f"Wrote {len(aggregate)} aggregate rows.")
    print(f"Wrote {len(paired)} paired delta rows.")
    print(f"Output stem: {args.results_dir / stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
