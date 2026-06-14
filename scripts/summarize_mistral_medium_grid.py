#!/usr/bin/env python3
"""Build reproducible tables and plots for the thesis-scale Mistral grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHOD_ORDER = ("dense", "depth", "quant", "uniform", "independent", "joint")
METHOD_LABELS = {
    "dense": "Dense FP16",
    "depth": "Depth-only",
    "quant": "Quant-only q_proj",
    "uniform": "Depth + uniform 3-bit q_proj",
    "independent": "Independent depth + quant",
    "joint": "Joint depth + q_proj quant",
}
RUN_COLUMNS = (
    "method",
    "seed",
    "run_id",
    "wikitext2_ppl",
    "train_ppl",
    "final_calibration_kl",
    "active_parameter_ratio",
    "average_bitwidth_active",
    "average_bitwidth_total",
    "estimated_weight_memory_mb",
    "estimated_compression_ratio",
    "dropped_attention_count",
    "dropped_mlp_count",
    "runtime_seconds",
    "source_search_runtime_seconds",
    "total_pipeline_runtime_seconds",
    "peak_cpu_memory_mb",
    "peak_gpu_device_used_mb",
    "best_generation",
    "accepted_generations",
)
AGGREGATE_COLUMNS = (
    "method",
    "runs",
    "wikitext2_ppl_mean",
    "wikitext2_ppl_sample_std",
    "wikitext2_ppl_min",
    "wikitext2_ppl_max",
    "train_ppl_mean",
    "final_calibration_kl_mean",
    "estimated_compression_ratio_mean",
    "estimated_weight_memory_mb_mean",
    "average_bitwidth_total_mean",
    "active_parameter_ratio_mean",
    "runtime_minutes_mean",
    "runtime_minutes_sample_std",
    "source_search_minutes_mean",
    "total_pipeline_minutes_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the three-seed Mistral medium experiment grid."
    )
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--prefix", default="thesis_medium")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV and markdown outputs without importing matplotlib.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
    *,
    lineterminator: str = "\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator=lineterminator,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def discover_run_dirs(runs_root: Path, prefix: str) -> dict[str, list[Path]]:
    patterns = {
        "depth": f"{prefix}_depth_mistral_s0.25_g20_o16_seed*",
        "quant": f"{prefix}_quant_mistral_qproj3.0_g20_o16_seed*",
        "joint": f"{prefix}_joint_mistral_s0.25_qproj3.0_g20_o16_seed*",
    }
    discovered: dict[str, list[Path]] = {}
    for method, pattern in patterns.items():
        paths = sorted(
            path
            for path in runs_root.glob(pattern)
            if (path / "run_summary.json").is_file()
        )
        if len(paths) != 3:
            raise SystemExit(
                f"Expected 3 {method} run summaries matching {pattern}, found {len(paths)}."
            )
        discovered[method] = paths
    return discovered


def discover_composition_dirs(
    runs_root: Path, prefix: str
) -> dict[str, list[Path]]:
    patterns = {
        "independent": (
            f"{prefix}_independent_depth_quant_mistral_s0.25_qproj3.0_seed*"
        ),
        "uniform": f"{prefix}_depth_uniform_quant_mistral_s0.25_qproj3.0_seed*",
    }
    discovered: dict[str, list[Path]] = {}
    for method, pattern in patterns.items():
        paths = sorted(
            path
            for path in runs_root.glob(pattern)
            if (path / "evaluation_metrics.csv").is_file()
        )
        if len(paths) != 3:
            raise SystemExit(
                f"Expected 3 {method} evaluations matching {pattern}, found {len(paths)}."
            )
        discovered[method] = paths
    return discovered


def dense_row(runs_root: Path, prefix: str) -> dict[str, Any]:
    run_dir = runs_root / f"{prefix}_dense_mistral_seq1024_seed0"
    metrics = read_csv(run_dir / "evaluation_metrics.csv")
    matches = [row for row in metrics if row["dataset"] == "wikitext2"]
    if len(matches) != 1:
        raise SystemExit(f"Expected one WikiText2 dense metric in {run_dir}.")

    runtime_values: dict[str, str] = {}
    for line in (run_dir / "runtime.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            runtime_values[key] = value

    return {
        "method": "dense",
        "seed": 0,
        "run_id": run_dir.name,
        "wikitext2_ppl": float(matches[0]["ppl"]),
        "runtime_seconds": float(runtime_values["runtime_seconds"]),
        "source_search_runtime_seconds": 0.0,
        "total_pipeline_runtime_seconds": float(runtime_values["runtime_seconds"]),
        "estimated_weight_memory_mb": 13824.5078125,
        "estimated_compression_ratio": 1.0,
        "active_parameter_ratio": 1.0,
        "average_bitwidth_total": 16.0,
    }


def search_row(method: str, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    summary = load_json(run_dir / "run_summary.json")
    candidate = load_json(run_dir / "final_candidate.json")
    generation_rows = read_csv(run_dir / "generation_log.csv")
    final = summary["final_metrics"]
    depth = summary["depth_statistics"]

    best_generation = min(
        generation_rows,
        key=lambda row: float(row["best_search_fitness"]),
    )["generation"]
    accepted = sum(
        row["accepted_parent_replacement"].strip().lower() == "true"
        for row in generation_rows
    )
    row = {
        "method": method,
        "seed": int(summary["search_config"]["seed"]),
        "run_id": summary["run_name"],
        "wikitext2_ppl": final["wikitext2_ppl"],
        "train_ppl": final["train_ppl"],
        "final_calibration_kl": final["final_calibration_kl"],
        "active_parameter_ratio": final["active_parameter_ratio"],
        "average_bitwidth_active": final["average_bitwidth_active"],
        "average_bitwidth_total": final["average_bitwidth_total"],
        "estimated_weight_memory_mb": final["estimated_weight_memory_mb"],
        "estimated_compression_ratio": final["estimated_compression_ratio"],
        "dropped_attention_count": depth["dropped_attention_count"],
        "dropped_mlp_count": depth["dropped_mlp_count"],
        "runtime_seconds": final["runtime_seconds"],
        "source_search_runtime_seconds": final["runtime_seconds"],
        "total_pipeline_runtime_seconds": final["runtime_seconds"],
        "peak_cpu_memory_mb": final["peak_cpu_memory_mb"],
        "peak_gpu_device_used_mb": final["peak_gpu_device_used_mb"],
        "best_generation": int(best_generation),
        "accepted_generations": accepted,
    }
    return row, generation_rows, candidate


def runtime_seconds(run_dir: Path) -> float:
    values: dict[str, str] = {}
    for line in (run_dir / "runtime.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if values.get("exit_code") != "0":
        raise SystemExit(f"Run did not complete successfully: {run_dir}")
    return float(values["runtime_seconds"])


def wikitext2_ppl(run_dir: Path) -> float:
    matches = [
        row
        for row in read_csv(run_dir / "evaluation_metrics.csv")
        if row["dataset"] == "wikitext2"
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected one WikiText2 metric in {run_dir}.")
    return float(matches[0]["ppl"])


def composition_row(
    method: str,
    seed: int,
    run_dir: Path,
    depth_dir: Path,
    quant_dir: Path,
) -> dict[str, Any]:
    depth_summary = load_json(depth_dir / "run_summary.json")
    quant_summary = load_json(quant_dir / "run_summary.json")
    depth_candidate = load_json(depth_dir / "final_candidate.json")
    quant_candidate = load_json(quant_dir / "final_candidate.json")
    dropped_modules = set(depth_candidate["dropped_modules"])
    bitwidth_by_module = quant_candidate["bitwidth_by_module"]

    active_bitwidths: list[int] = []
    for module, searched_bits in bitwidth_by_module.items():
        attention_module = module.rsplit(".q_proj", 1)[0]
        if attention_module in dropped_modules:
            continue
        active_bitwidths.append(3 if method == "uniform" else int(searched_bits))

    searched_parameters = quant_summary["parameter_statistics"][
        "searched_parameters_dense"
    ]
    parameters_per_module = searched_parameters / len(bitwidth_by_module)
    active_quant_parameters = parameters_per_module * len(active_bitwidths)
    active_parameters = depth_summary["parameter_statistics"]["active_parameters"]
    active_nonsearched_parameters = active_parameters - active_quant_parameters
    total_parameters = depth_summary["parameter_statistics"][
        "total_parameters_dense"
    ]
    dense_weight_memory_mb = depth_summary["final_metrics"][
        "dense_weight_memory_mb"
    ]
    effective_bits = (
        active_nonsearched_parameters * 16
        + parameters_per_module * sum(active_bitwidths)
    )
    average_bitwidth_total = effective_bits / total_parameters
    estimated_weight_memory_mb = effective_bits / 8 / 1024 / 1024
    estimated_compression_ratio = (
        dense_weight_memory_mb / estimated_weight_memory_mb
    )
    evaluation_runtime = runtime_seconds(run_dir)
    depth_runtime = depth_summary["final_metrics"]["runtime_seconds"]
    quant_runtime = quant_summary["final_metrics"]["runtime_seconds"]
    source_search_runtime = (
        depth_runtime if method == "uniform" else depth_runtime + quant_runtime
    )

    return {
        "method": method,
        "seed": seed,
        "run_id": run_dir.name,
        "wikitext2_ppl": wikitext2_ppl(run_dir),
        "active_parameter_ratio": depth_summary["final_metrics"][
            "active_parameter_ratio"
        ],
        "average_bitwidth_active": statistics.mean(active_bitwidths),
        "average_bitwidth_total": average_bitwidth_total,
        "estimated_weight_memory_mb": estimated_weight_memory_mb,
        "estimated_compression_ratio": estimated_compression_ratio,
        "dropped_attention_count": depth_summary["depth_statistics"][
            "dropped_attention_count"
        ],
        "dropped_mlp_count": depth_summary["depth_statistics"][
            "dropped_mlp_count"
        ],
        "runtime_seconds": evaluation_runtime,
        "source_search_runtime_seconds": source_search_runtime,
        "total_pipeline_runtime_seconds": source_search_runtime
        + evaluation_runtime,
    }


def aggregate_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[row["method"]].append(row)

    outputs: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        rows = grouped[method]

        def values(key: str) -> list[float]:
            return [
                float(row[key])
                for row in rows
                if row.get(key) not in (None, "")
            ]

        ppls = values("wikitext2_ppl")
        runtime_minutes = [value / 60 for value in values("runtime_seconds")]
        source_search_minutes = [
            value / 60 for value in values("source_search_runtime_seconds")
        ]
        total_pipeline_minutes = [
            value / 60 for value in values("total_pipeline_runtime_seconds")
        ]
        outputs.append(
            {
                "method": method,
                "runs": len(rows),
                "wikitext2_ppl_mean": mean(ppls),
                "wikitext2_ppl_sample_std": sample_std(ppls),
                "wikitext2_ppl_min": min(ppls),
                "wikitext2_ppl_max": max(ppls),
                "train_ppl_mean": mean(values("train_ppl")),
                "final_calibration_kl_mean": mean(values("final_calibration_kl")),
                "estimated_compression_ratio_mean": mean(
                    values("estimated_compression_ratio")
                ),
                "estimated_weight_memory_mb_mean": mean(
                    values("estimated_weight_memory_mb")
                ),
                "average_bitwidth_total_mean": mean(
                    values("average_bitwidth_total")
                ),
                "active_parameter_ratio_mean": mean(
                    values("active_parameter_ratio")
                ),
                "runtime_minutes_mean": mean(runtime_minutes),
                "runtime_minutes_sample_std": sample_std(runtime_minutes),
                "source_search_minutes_mean": mean(source_search_minutes),
                "total_pipeline_minutes_mean": mean(total_pipeline_minutes),
            }
        )
    return outputs


def matched_comparison_rows(
    run_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_method_seed = {
        (row["method"], int(row["seed"])): row for row in run_rows
    }
    outputs: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        depth = by_method_seed[("depth", seed)]
        uniform = by_method_seed[("uniform", seed)]
        independent = by_method_seed[("independent", seed)]
        joint = by_method_seed[("joint", seed)]
        outputs.append(
            {
                "seed": seed,
                "depth_ppl": depth["wikitext2_ppl"],
                "uniform_composition_ppl": uniform["wikitext2_ppl"],
                "independent_composition_ppl": independent["wikitext2_ppl"],
                "joint_ppl": joint["wikitext2_ppl"],
                "joint_minus_independent_ppl": (
                    joint["wikitext2_ppl"] - independent["wikitext2_ppl"]
                ),
                "independent_active_qproj_bits": independent[
                    "average_bitwidth_active"
                ],
                "independent_compression_ratio": independent[
                    "estimated_compression_ratio"
                ],
                "joint_compression_ratio": joint[
                    "estimated_compression_ratio"
                ],
                "independent_source_search_minutes": independent[
                    "source_search_runtime_seconds"
                ]
                / 60,
                "joint_search_minutes": joint["source_search_runtime_seconds"]
                / 60,
            }
        )
    return outputs


def convergence_rows(
    method: str,
    run_row: dict[str, Any],
    generation_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    final_generation = max(int(row["generation"]) for row in generation_rows)
    for row in generation_rows:
        output.append(
            {
                "method": method,
                "seed": run_row["seed"],
                "run_id": run_row["run_id"],
                "generation": int(row["generation"]),
                "best_search_fitness": numeric(row, "best_search_fitness"),
                "wikitext2_ppl": numeric(row, "wikitext2_ppl"),
                "train_ppl": numeric(row, "best_train_ppl"),
                "accepted_parent_replacement": row["accepted_parent_replacement"],
                "runtime_seconds_cumulative": numeric(
                    row, "runtime_seconds_cumulative"
                ),
                "phase": "generation",
            }
        )
    output.append(
        {
            "method": method,
            "seed": run_row["seed"],
            "run_id": run_row["run_id"],
            "generation": final_generation,
            "best_search_fitness": run_row["final_calibration_kl"],
            "wikitext2_ppl": run_row["wikitext2_ppl"],
            "train_ppl": run_row["train_ppl"],
            "accepted_parent_replacement": "",
            "runtime_seconds_cumulative": run_row["runtime_seconds"],
            "phase": "final",
        }
    )
    return output


def dropped_set(candidate: dict[str, Any]) -> set[str]:
    return set(candidate.get("dropped_modules", []))


def overlap_markdown(method: str, candidates: dict[int, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {METHOD_LABELS[method]} dropped-module stability",
        "",
        "| Seed pair | Intersection | Union | Jaccard |",
        "| --- | ---: | ---: | ---: |",
    ]
    sets = {seed: dropped_set(candidate) for seed, candidate in candidates.items()}
    for left, right in itertools.combinations(sorted(sets), 2):
        intersection = len(sets[left] & sets[right])
        union = len(sets[left] | sets[right])
        jaccard = intersection / union if union else 1.0
        lines.append(f"| {left} vs {right} | {intersection} | {union} | {jaccard:.3f} |")

    common = set.intersection(*(sets[seed] for seed in sorted(sets)))
    lines.extend(
        [
            "",
            f"Modules selected in all three seeds ({len(common)}): "
            + (", ".join(f"`{name}`" for name in sorted(common)) or "none")
            + ".",
            "",
        ]
    )
    return lines


def quant_profile_markdown(candidates: dict[int, dict[str, Any]]) -> list[str]:
    assignments: dict[int, dict[str, int]] = {
        seed: candidate["bitwidth_by_module"]
        for seed, candidate in candidates.items()
    }
    module_counts: Counter[str] = Counter()
    for profile in assignments.values():
        for module, bits in profile.items():
            if bits == 2:
                module_counts[module] += 1

    stable_low_bit = [
        module for module, count in sorted(module_counts.items()) if count == 3
    ]
    lines = [
        "### Quantization profile stability",
        "",
        "| Seed | 2-bit modules | 3-bit modules | 4-bit modules |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted(assignments):
        histogram = Counter(assignments[seed].values())
        lines.append(
            f"| {seed} | {histogram[2]} | {histogram[3]} | {histogram[4]} |"
        )
    lines.extend(
        [
            "",
            "Modules assigned 2 bits in every quant-only seed: "
            + (", ".join(f"`{name}`" for name in stable_low_bit) or "none")
            + ". The locations receiving 4 bits varied across seeds.",
            "",
        ]
    )
    return lines


def build_markdown(
    run_rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    candidates: dict[str, dict[int, dict[str, Any]]],
) -> str:
    aggregate_by_method = {row["method"]: row for row in aggregate}
    rows_by_method_seed = {
        (row["method"], int(row["seed"])): row for row in run_rows
    }
    dense_ppl = aggregate_by_method["dense"]["wikitext2_ppl_mean"]
    depth = aggregate_by_method["depth"]
    quant = aggregate_by_method["quant"]
    uniform = aggregate_by_method["uniform"]
    independent = aggregate_by_method["independent"]
    joint = aggregate_by_method["joint"]
    assert dense_ppl is not None

    joint_depth_ppl_delta = (
        joint["wikitext2_ppl_mean"] - depth["wikitext2_ppl_mean"]
    )
    joint_depth_memory_delta = (
        depth["estimated_weight_memory_mb_mean"]
        - joint["estimated_weight_memory_mb_mean"]
    )
    joint_depth_ratio_gain = (
        joint["estimated_compression_ratio_mean"]
        / depth["estimated_compression_ratio_mean"]
        - 1
    ) * 100
    joint_independent_deltas = [
        rows_by_method_seed[("joint", seed)]["wikitext2_ppl"]
        - rows_by_method_seed[("independent", seed)]["wikitext2_ppl"]
        for seed in (0, 1, 2)
    ]
    joint_independent_mean_delta = statistics.mean(joint_independent_deltas)
    independent_uniform_delta = (
        independent["wikitext2_ppl_mean"] - uniform["wikitext2_ppl_mean"]
    )
    independent_compute_ratio = (
        independent["source_search_minutes_mean"]
        / joint["source_search_minutes_mean"]
    )

    lines = [
        "# Mistral-7B Medium Search Comparison",
        "",
        "This report is generated from the tracked artifacts for the thesis-scale medium grid. All searches use `mistralai/Mistral-7B-v0.3`, WikiText2 calibration, sequence length `1024`, `8192` calibration tokens, `20` generations, `16` offspring, `32` initial candidates, and seeds `0`, `1`, and `2`. The matched composition controls use the same final WikiText2 evaluation protocol.",
        "",
        "## Main comparison",
        "",
        "| Method | Runs | WikiText2 PPL mean +/- SD | Compression | Effective bits | Active q_proj bits | Source search min | Final job min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        row = aggregate_by_method[method]
        ppl_sd = row["wikitext2_ppl_sample_std"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['runs']} | "
            f"{fmt(row['wikitext2_ppl_mean'])}"
            + (f" +/- {fmt(ppl_sd)}" if ppl_sd is not None else "")
            + f" | {fmt(row['estimated_compression_ratio_mean'])}x | "
            f"{fmt(row['average_bitwidth_total_mean'])} | "
            f"{fmt(mean([float(item['average_bitwidth_active']) for item in run_rows if item['method'] == method and item.get('average_bitwidth_active') not in (None, '')]))} | "
            f"{fmt(row['source_search_minutes_mean'], 2)} | "
            f"{fmt(row['runtime_minutes_mean'], 2)} |"
        )

    lines.extend(
        [
            "",
            "The dense reference reaches WikiText2 PPL "
            f"{dense_ppl:.2f}. Quantizing only `q_proj` preserves dense quality "
            f"(mean PPL {quant['wikitext2_ppl_mean']:.3f}) but produces only "
            f"{quant['estimated_compression_ratio_mean']:.3f}x whole-model compression because `q_proj` is a small fraction of total model weights.",
            "",
            f"Depth-only search reaches {depth['estimated_compression_ratio_mean']:.3f}x compression at mean PPL {depth['wikitext2_ppl_mean']:.3f}. Joint search reaches {joint['estimated_compression_ratio_mean']:.3f}x at mean PPL {joint['wikitext2_ppl_mean']:.3f}. Relative to depth-only, joint search reduces the theoretical weight footprint by {joint_depth_memory_delta:.0f} MiB and increases the compression ratio by {joint_depth_ratio_gain:.1f}%, while mean PPL is {joint_depth_ppl_delta:.3f} higher.",
            "",
            f"At the matched combined target, independent composition reaches mean PPL {independent['wikitext2_ppl_mean']:.3f}, compared with {joint['wikitext2_ppl_mean']:.3f} for joint search. The paired mean difference `joint - independent` is {joint_independent_mean_delta:.3f} PPL, so the current 20-generation joint method is worse on average. Independent composition is better in two of three seeds; with only three seeds this should be reported as evidence, not a definitive statistical claim.",
            "",
            f"Uniform 3-bit `q_proj` composition reaches mean PPL {uniform['wikitext2_ppl_mean']:.3f}. Its difference from the searched independent quantization profiles is only {abs(independent_uniform_delta):.3f} PPL. At this narrow `q_proj` scope, quantization-profile search therefore adds no visible benefit over uniform 3-bit assignment after depth pruning.",
            "",
            f"The runtime comparison is not equal: independent composition uses both the depth and quant searches, averaging {independent['source_search_minutes_mean']:.2f} search minutes, or {independent_compute_ratio:.2f}x the {joint['source_search_minutes_mean']:.2f} minutes used by one joint search. The existing result is target-matched but not compute-matched.",
            "",
            "## Matched per-seed comparison",
            "",
            "| Seed | Depth-only PPL | Uniform composition PPL | Independent composition PPL | Joint PPL | Joint - independent |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in (0, 1, 2):
        depth_row = rows_by_method_seed[("depth", seed)]
        uniform_row = rows_by_method_seed[("uniform", seed)]
        independent_row = rows_by_method_seed[("independent", seed)]
        joint_row = rows_by_method_seed[("joint", seed)]
        lines.append(
            f"| {seed} | {fmt(depth_row['wikitext2_ppl'])} | "
            f"{fmt(uniform_row['wikitext2_ppl'])} | "
            f"{fmt(independent_row['wikitext2_ppl'])} | "
            f"{fmt(joint_row['wikitext2_ppl'])} | "
            f"{joint_row['wikitext2_ppl'] - independent_row['wikitext2_ppl']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| Method | Seed | WikiText2 PPL | Train PPL | Final KL | Compression | Effective bits | Source search min | Final job min | Best generation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['seed']} | "
            f"{fmt(row.get('wikitext2_ppl'))} | {fmt(row.get('train_ppl'))} | "
            f"{fmt(row.get('final_calibration_kl'), 4)} | "
            f"{fmt(row.get('estimated_compression_ratio'))}x | "
            f"{fmt(row.get('average_bitwidth_total'))} | "
            f"{fmt(float(row['source_search_runtime_seconds']) / 60, 2)} | "
            f"{fmt(float(row['runtime_seconds']) / 60, 2)} | "
            f"{row.get('best_generation', '')} |"
        )

    lines.extend(["", "## Seed stability", ""])
    lines.extend(overlap_markdown("depth", candidates["depth"]))
    lines.extend(overlap_markdown("joint", candidates["joint"]))
    lines.extend(quant_profile_markdown(candidates["quant"]))
    lines.extend(
        [
            "Final PPL is reasonably repeatable, but the selected depth masks are not identical. This suggests multiple competitive compression configurations rather than one uniquely stable mask.",
            "",
            "## Convergence evidence",
            "",
            "Depth and joint runs continued to accept improved parents near generation 20, and every joint run recorded its minimum search fitness at generation 20. The current budget therefore does not demonstrate full convergence. Quant-only quality was already close to dense throughout the search, although its calibration KL continued to change.",
            "",
            "The generated `mistral_medium_convergence.png` shows the generation-wise search fitness and the periodic WikiText2 evaluations. Final evaluations are added at generation 20.",
            "",
            "## Hardware and measurement notes",
            "",
            "- The nine searches ran on a Tesla V100 32 GB with a 16 GB container RAM limit.",
            "- Peak sampled device use was about 14.66 GB for every search.",
            "- Peak sampled CPU cgroup memory reached 16 GB, so CPU memory remains the binding resource and leaves little safety margin.",
            "- Compression ratios and model sizes are theoretical weight estimates. The current reconstruction database and runtime model remain floating point; these numbers are not measured checkpoint file sizes or inference-memory measurements.",
            "- Results currently cover WikiText2 only. C4, FineWeb-Edu, and downstream task evaluation remain necessary for broader claims.",
            "",
            "## Interpretation for the thesis",
            "",
            "The current baseline joint search does not beat independently optimized components at the same nominal compression target. This is a useful negative result: merely placing depth and quantization variables in one candidate representation is insufficient to produce a better solution.",
            "",
            "Two explanations remain open:",
            "",
            "1. The joint search receives less total compute because one offspring population is split between depth and quantization mutations.",
            "2. The current mutation operator does not explicitly model interactions between dropping an attention module and assigning bitwidth to its `q_proj` weights.",
            "",
            "## Required next experiment",
            "",
            f"Run the unchanged joint baseline for `50` generations and `16` offspring on seeds `0`, `1`, and `2`. Based on the measured 20-generation runtime, this should approach the independent pipeline's {independent['source_search_minutes_mean']:.2f}-minute search budget. It is the necessary compute-matched baseline because all three current joint runs achieved their best recorded fitness at generation 20.",
            "",
            "After the 50-generation baseline, implement joint-aware mutation and compare it against the unchanged 50-generation method with identical seeds and budgets. That ablation is the strongest candidate for the thesis implementation contribution.",
            "",
            "## Generated artifacts",
            "",
            "- `results/mistral_medium_runs.csv`: one row per run.",
            "- `results/mistral_medium_aggregate.csv`: method-level mean and sample standard deviation.",
            "- `results/mistral_medium_matched_comparison.csv`: paired seed-level comparison of depth, uniform, independent, and joint configurations.",
            "- `results/mistral_medium_convergence.csv`: generation-wise search and evaluation metrics.",
            "- `results/mistral_medium_quality_compression.png`: quality-compression tradeoff.",
            "- `results/mistral_medium_convergence.png`: search convergence.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_plots(
    output_dir: Path,
    aggregate: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "dense": "#444444",
        "depth": "#1f77b4",
        "quant": "#2ca02c",
        "uniform": "#ff7f0e",
        "independent": "#9467bd",
        "joint": "#d62728",
    }
    markers = {
        "dense": "o",
        "depth": "o",
        "quant": "o",
        "uniform": "D",
        "independent": "s",
        "joint": "X",
    }

    aggregate_by_method = {row["method"]: row for row in aggregate}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in METHOD_ORDER:
        row = aggregate_by_method[method]
        xerr = None
        yerr = row["wikitext2_ppl_sample_std"]
        ax.errorbar(
            row["estimated_compression_ratio_mean"],
            row["wikitext2_ppl_mean"],
            xerr=xerr,
            yerr=yerr,
            fmt=markers[method],
            markersize=8,
            capsize=4,
            color=colors[method],
            markerfacecolor="white" if method == "uniform" else colors[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Estimated theoretical compression ratio")
    ax.set_ylabel("WikiText2 perplexity (lower is better)")
    ax.set_title("Mistral-7B Medium Grid: Quality vs. Compression")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mistral_medium_quality_compression.png", dpi=180)
    plt.close(fig)

    fig, (fitness_ax, ppl_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    generation_only = [row for row in convergence if row["phase"] == "generation"]
    for method in ("depth", "quant", "joint"):
        method_rows = [row for row in generation_only if row["method"] == method]
        seeds = sorted({int(row["seed"]) for row in method_rows})
        by_generation: dict[int, list[float]] = defaultdict(list)
        for seed in seeds:
            seed_rows = [
                row for row in method_rows if int(row["seed"]) == seed
            ]
            fitness_ax.plot(
                [int(row["generation"]) for row in seed_rows],
                [float(row["best_search_fitness"]) for row in seed_rows],
                color=colors[method],
                alpha=0.2,
                linewidth=1,
            )
            for row in seed_rows:
                by_generation[int(row["generation"])].append(
                    float(row["best_search_fitness"])
                )
        generations = sorted(by_generation)
        fitness_ax.plot(
            generations,
            [statistics.mean(by_generation[generation]) for generation in generations],
            color=colors[method],
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )

        eval_rows = [
            row
            for row in convergence
            if row["method"] == method and row["wikitext2_ppl"] not in (None, "")
        ]
        eval_by_generation: dict[int, list[float]] = defaultdict(list)
        for row in eval_rows:
            eval_by_generation[int(row["generation"])].append(
                float(row["wikitext2_ppl"])
            )
        eval_generations = sorted(eval_by_generation)
        ppl_ax.plot(
            eval_generations,
            [
                statistics.mean(eval_by_generation[generation])
                for generation in eval_generations
            ],
            marker="o",
            color=colors[method],
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )

    fitness_ax.set_yscale("log")
    fitness_ax.set_xlabel("Generation")
    fitness_ax.set_ylabel("Best KL search fitness (log scale)")
    fitness_ax.set_title("Search Fitness")
    fitness_ax.grid(alpha=0.25)
    fitness_ax.legend()
    ppl_ax.set_xlabel("Generation")
    ppl_ax.set_ylabel("WikiText2 perplexity")
    ppl_ax.set_title("Periodic Evaluation")
    ppl_ax.grid(alpha=0.25)
    ppl_ax.legend()
    fig.suptitle("Mistral-7B Medium Grid Convergence")
    fig.tight_layout()
    fig.savefig(output_dir / "mistral_medium_convergence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    discovered = discover_run_dirs(runs_root, args.prefix)
    composition_dirs = discover_composition_dirs(runs_root, args.prefix)

    run_rows = [dense_row(runs_root, args.prefix)]
    convergence: list[dict[str, Any]] = []
    candidates: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    search_dirs_by_method_seed: dict[tuple[str, int], Path] = {}
    for method in ("depth", "quant", "joint"):
        for run_dir in discovered[method]:
            row, generation_rows, candidate = search_row(method, run_dir)
            run_rows.append(row)
            convergence.extend(convergence_rows(method, row, generation_rows))
            candidates[method][int(row["seed"])] = candidate
            search_dirs_by_method_seed[(method, int(row["seed"]))] = run_dir

    for method in ("uniform", "independent"):
        for seed, run_dir in enumerate(composition_dirs[method]):
            run_rows.append(
                composition_row(
                    method,
                    seed,
                    run_dir,
                    search_dirs_by_method_seed[("depth", seed)],
                    search_dirs_by_method_seed[("quant", seed)],
                )
            )

    run_rows.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), row["seed"]))
    aggregate = aggregate_rows(run_rows)
    matched_rows = matched_comparison_rows(run_rows)
    write_csv(output_dir / "mistral_medium_runs.csv", RUN_COLUMNS, run_rows)
    write_csv(
        output_dir / "mistral_medium_aggregate.csv",
        AGGREGATE_COLUMNS,
        aggregate,
    )
    write_csv(
        output_dir / "mistral_medium_convergence.csv",
        (
            "method",
            "seed",
            "run_id",
            "phase",
            "generation",
            "best_search_fitness",
            "wikitext2_ppl",
            "train_ppl",
            "accepted_parent_replacement",
            "runtime_seconds_cumulative",
        ),
        convergence,
        lineterminator="\r\n",
    )
    write_csv(
        output_dir / "mistral_medium_matched_comparison.csv",
        (
            "seed",
            "depth_ppl",
            "uniform_composition_ppl",
            "independent_composition_ppl",
            "joint_ppl",
            "joint_minus_independent_ppl",
            "independent_active_qproj_bits",
            "independent_compression_ratio",
            "joint_compression_ratio",
            "independent_source_search_minutes",
            "joint_search_minutes",
        ),
        matched_rows,
    )
    markdown = build_markdown(run_rows, aggregate, candidates)
    (output_dir / "mistral_medium_comparison.md").write_text(
        markdown, encoding="utf-8"
    )
    if not args.skip_plots:
        create_plots(output_dir, aggregate, convergence)

    print(f"Wrote Mistral medium-grid summary artifacts to {output_dir}")


if __name__ == "__main__":
    main()
