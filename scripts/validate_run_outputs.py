#!/usr/bin/env python3
"""Validate structured EvoPress search artifacts before expensive experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_FILES = [
    "run_summary.json",
    "generation_log.csv",
    "final_candidate.json",
]

REQUIRED_SUMMARY_KEYS = [
    "schema_version",
    "run_name",
    "timestamp_start",
    "timestamp_end",
    "git_commit",
    "model_name",
    "dataset_calibration",
    "dataset_eval",
    "search_type",
    "search_config",
    "compression_config",
    "final_metrics",
    "parameter_statistics",
    "depth_statistics",
    "quantization_statistics",
    "model_size_statistics",
    "artifacts",
    "metric_definitions",
]

REQUIRED_FINAL_METRICS = [
    "best_search_fitness",
    "final_calibration_kl",
    "wikitext2_ppl",
    "c4_ppl",
    "fineweb_ppl",
    "train_ppl",
    "active_parameters",
    "total_parameters_dense",
    "active_parameter_ratio",
    "average_bitwidth_active",
    "average_bitwidth_searched",
    "average_bitwidth_total",
    "estimated_weight_memory_mb",
    "dense_weight_memory_mb",
    "estimated_compression_ratio",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "peak_gpu_reserved_mb",
    "peak_cpu_memory_mb",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate run_summary.json, generation_log.csv, and final_candidate.json."
    )
    parser.add_argument("run_dir", help="Experiment directory under outputs/experiments.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def require_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def reject_nonfinite_numbers(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {path}: {value}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_nonfinite_numbers(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite_numbers(item, f"{path}[{index}]")


def validate_generation_log(path: Path, schema_version: int) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("generation_log.csv has no header.")
        required_columns = {
            "generation",
            "best_search_fitness",
            "fitness_fn",
            "eval_tokens_used",
            "active_parameters",
            "estimated_weight_memory_mb",
            "dropped_attention_count",
            "dropped_mlp_count",
            "runtime_seconds_cumulative",
        }
        if schema_version >= 2:
            required_columns.add("eval_tokens_by_dataset")
        missing = sorted(required_columns.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                f"generation_log.csv is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("generation_log.csv contains no generation rows.")
    observed = [int(row["generation"]) for row in rows]
    if observed != list(range(1, len(rows) + 1)):
        raise ValueError(f"Generation sequence is not contiguous from 1: {observed}")
    return len(rows)


def validate_consistency(summary: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    final_metrics = summary["final_metrics"]
    parameter_statistics = summary["parameter_statistics"]
    depth_statistics = summary["depth_statistics"]
    quantization_statistics = summary["quantization_statistics"]
    model_size_statistics = summary["model_size_statistics"]

    if summary.get("launcher_finalized_at"):
        if final_metrics.get("runtime_source") != "launcher_wall_clock":
            raise ValueError(
                "Launcher-finalized summary must use launcher_wall_clock runtime."
            )
        if final_metrics.get("search_process_runtime_seconds") is None:
            raise ValueError(
                "Launcher-finalized summary is missing search_process_runtime_seconds."
            )
        if final_metrics.get("peak_process_rss_mb") is None:
            raise ValueError(
                "Launcher-finalized summary is missing peak_process_rss_mb."
            )

        cgroup_peak = final_metrics.get("peak_cpu_cgroup_memory_mb")
        if cgroup_peak is not None and not math.isclose(
            summary["final_metrics"]["peak_cpu_memory_mb"],
            cgroup_peak,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "final_metrics.peak_cpu_memory_mb does not match "
                "peak_cpu_cgroup_memory_mb."
            )

    active = parameter_statistics["active_parameters"]
    dense = parameter_statistics["total_parameters_dense"]
    if active > dense:
        raise ValueError(f"active_parameters ({active}) exceeds dense parameters ({dense}).")
    if final_metrics["active_parameters"] != active:
        raise ValueError("final_metrics.active_parameters disagrees with parameter_statistics.")
    if final_metrics["total_parameters_dense"] != dense:
        raise ValueError("final_metrics.total_parameters_dense disagrees with parameter_statistics.")

    dropped_modules = candidate.get("dropped_modules", [])
    dropped_attention = depth_statistics["dropped_attention_count"]
    dropped_mlp = depth_statistics["dropped_mlp_count"]
    if depth_statistics["dropped_total_count"] != dropped_attention + dropped_mlp:
        raise ValueError("Depth dropped-module totals are inconsistent.")
    if len(dropped_modules) != depth_statistics["dropped_total_count"]:
        raise ValueError("final_candidate.json dropped_modules disagrees with depth statistics.")

    bitwidth_by_module = candidate.get("bitwidth_by_module", {})
    if len(bitwidth_by_module) != quantization_statistics["quantized_module_count"]:
        raise ValueError("Final candidate bitwidth assignments disagree with quantization statistics.")
    histogram_total = sum(quantization_statistics["bitwidth_histogram"].values())
    if histogram_total != quantization_statistics["active_quantized_module_count"]:
        raise ValueError("Active bitwidth histogram does not match active quantized module count.")

    dense_memory = model_size_statistics["dense_weight_memory_mb"]
    estimated_memory = model_size_statistics["estimated_weight_memory_mb"]
    ratio = model_size_statistics["estimated_compression_ratio"]
    if estimated_memory <= 0 or dense_memory <= 0:
        raise ValueError("Model-size estimates must be positive.")
    expected_ratio = dense_memory / estimated_memory
    if not math.isclose(ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Estimated compression ratio does not match the documented formula.")

    if summary["schema_version"] >= 2:
        expected_total_average = (
            estimated_memory
            / dense_memory
            * model_size_statistics["dense_dtype_bits"]
        )
        if not math.isclose(
            quantization_statistics["average_bitwidth_total"],
            expected_total_average,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "average_bitwidth_total does not treat dropped parameters as zero-bit."
            )

    is_compressed = (
        depth_statistics["dropped_total_count"] > 0
        or (
            quantization_statistics["average_bitwidth_active"] is not None
            and quantization_statistics["average_bitwidth_active"]
            < model_size_statistics["dense_dtype_bits"]
        )
    )
    if is_compressed and ratio <= 1:
        raise ValueError("Compressed candidate does not have an estimated compression ratio above 1.")


def validate_run_dir(run_dir: Path) -> int:
    missing_files = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing_files:
        raise ValueError(f"Missing required files: {', '.join(missing_files)}")

    summary = load_json(run_dir / "run_summary.json")
    candidate = load_json(run_dir / "final_candidate.json")
    require_keys(summary, REQUIRED_SUMMARY_KEYS, "run_summary.json")
    require_keys(summary["final_metrics"], REQUIRED_FINAL_METRICS, "final_metrics")
    reject_nonfinite_numbers(summary, "run_summary")
    reject_nonfinite_numbers(candidate, "final_candidate")

    generation_count = validate_generation_log(
        run_dir / "generation_log.csv",
        int(summary["schema_version"]),
    )
    validate_consistency(summary, candidate)
    return generation_count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    generation_count = validate_run_dir(run_dir)
    print(f"Validated structured run outputs: {run_dir}")
    print(f"Generation rows: {generation_count}")
    for filename in REQUIRED_FILES:
        print(f"OK: {filename}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
