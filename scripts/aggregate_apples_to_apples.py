#!/usr/bin/env python3
"""Aggregate completed apples-to-apples runs and paired seed differences."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "dense": "Dense",
    "uniform_quantization": "Uniform 3-bit",
    "quant_only": "EvoPress quant-only",
    "joint_depth_quant": "Joint depth+quant",
}
METHOD_ORDER = list(METHOD_LABELS)
METRICS = {
    "wikitext2_ppl": "Wiki2 PPL",
    "c4_ppl": "C4 PPL",
    "final_calibration_kl": "KL",
    "runtime_seconds": "Runtime (s)",
    "candidate_evaluations_search_total": "Evaluations",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def metric_value(summary: dict[str, Any], metric: str) -> float | None:
    if metric == "candidate_evaluations_search_total":
        value = summary.get("search_config", {}).get(metric)
    else:
        value = summary.get("final_metrics", {}).get(metric)
    return None if value is None else float(value)


def cost_bits(summary: dict[str, Any]) -> int | None:
    value = summary.get("final_metrics", {}).get("compression_realized_bits")
    if value is None:
        value = summary.get("model_size_statistics", {}).get("compression_cost_bits")
    return None if value is None else int(value)


def compression_ratio(summary: dict[str, Any]) -> float | None:
    value = summary.get("final_metrics", {}).get("estimated_compression_ratio")
    if value is None:
        value = summary.get("model_size_statistics", {}).get("estimated_compression_ratio")
    return None if value is None else float(value)


def seed(summary: dict[str, Any]) -> int:
    value = summary.get("search_config", {}).get("seed")
    if value is None:
        raise ValueError(f"Run summary has no seed: {summary.get('_path')}")
    return int(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def display(mean: float | None, std: float | None, *, integer: bool = False) -> str:
    if mean is None:
        return "-"
    if integer:
        return f"{mean:.0f} ± {std:.0f}" if std else f"{mean:.0f}"
    return f"{mean:.4f} ± {std:.4f}" if std else f"{mean:.4f}"


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.rglob("run_summary.json")):
        summary = read_json(path)
        search_type = summary.get("search_type")
        if search_type not in METHOD_LABELS:
            continue
        summary["_path"] = str(path)
        runs.append(summary)
    if not runs:
        raise ValueError(f"No apples-to-apples run summaries found under {root}")
    return runs


def validate_runs(runs: list[dict[str, Any]]) -> None:
    by_method_seed: dict[tuple[str, int], list[str]] = defaultdict(list)
    for run in runs:
        by_method_seed[(run["search_type"], seed(run))].append(run["_path"])
    duplicates = {key: paths for key, paths in by_method_seed.items() if len(paths) > 1}
    if duplicates:
        raise ValueError(
            "Multiple summaries exist for the same method and seed; aggregate an "
            f"explicit clean run directory instead: {duplicates}"
        )

    comparison_runs = [run for run in runs if run["search_type"] != "dense"]
    targets = {
        run.get("final_metrics", {}).get("compression_target_bits")
        for run in comparison_runs
    }
    targets.discard(None)
    if len(targets) != 1:
        raise ValueError(f"Compressed runs do not share one target: {targets}")
    for run in comparison_runs:
        difference = run.get("final_metrics", {}).get("compression_difference_bits")
        if difference != 0:
            raise ValueError(
                f"Run is not exact-budget feasible ({difference} bits): {run['_path']}"
            )


def aggregate_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["search_type"]].append(run)
    rows = []
    for method in METHOD_ORDER:
        method_runs = grouped.get(method, [])
        if not method_runs:
            continue
        costs = sorted({cost_bits(run) for run in method_runs})
        ratios = [value for run in method_runs if (value := compression_ratio(run)) is not None]
        row: dict[str, Any] = {
            "method": METHOD_LABELS[method],
            "search_type": method,
            "seeds": ",".join(str(seed(run)) for run in sorted(method_runs, key=seed)),
            "runs": len(method_runs),
            "budget_bits": costs[0] if len(costs) == 1 else None,
            "compression_ratio_mean": mean_std(ratios)[0],
            "compression_ratio_std": mean_std(ratios)[1],
        }
        for metric in METRICS:
            values = [
                value
                for run in method_runs
                if (value := metric_value(run, metric)) is not None
            ]
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(values)
        rows.append(row)
    return rows


def paired_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quant = {seed(run): run for run in runs if run["search_type"] == "quant_only"}
    joint = {seed(run): run for run in runs if run["search_type"] == "joint_depth_quant"}
    common_seeds = sorted(set(quant) & set(joint))
    rows = []
    for run_seed in common_seeds:
        row: dict[str, Any] = {"seed": run_seed}
        for metric in (
            "wikitext2_ppl",
            "c4_ppl",
            "final_calibration_kl",
            "runtime_seconds",
        ):
            quant_value = metric_value(quant[run_seed], metric)
            joint_value = metric_value(joint[run_seed], metric)
            row[f"quant_only_{metric}"] = quant_value
            row[f"joint_{metric}"] = joint_value
            row[f"joint_minus_quant_only_{metric}"] = (
                joint_value - quant_value
                if quant_value is not None and joint_value is not None
                else None
            )
        rows.append(row)
    if rows:
        mean_row: dict[str, Any] = {"seed": "mean"}
        for metric in (
            "wikitext2_ppl",
            "c4_ppl",
            "final_calibration_kl",
            "runtime_seconds",
        ):
            key = f"joint_minus_quant_only_{metric}"
            values = [float(row[key]) for row in rows if row[key] is not None]
            mean_row[key] = statistics.mean(values) if values else None
        rows.append(mean_row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Budget (bits) | Compression Ratio | Wiki2 PPL | C4 PPL | KL | Runtime (s) | Evaluations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {budget} | {ratio} | {wiki} | {c4} | {kl} | {runtime} | {evals} |".format(
                method=row["method"],
                budget=row["budget_bits"] if row["budget_bits"] is not None else "-",
                ratio=display(row["compression_ratio_mean"], row["compression_ratio_std"]),
                wiki=display(row["wikitext2_ppl_mean"], row["wikitext2_ppl_std"]),
                c4=display(row["c4_ppl_mean"], row["c4_ppl_std"]),
                kl=display(row["final_calibration_kl_mean"], row["final_calibration_kl_std"]),
                runtime=display(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                evals=display(
                    row["candidate_evaluations_search_total_mean"],
                    row["candidate_evaluations_search_total_std"],
                    integer=True,
                ),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = [
        output_dir / "comparison.csv",
        output_dir / "comparison.md",
        output_dir / "paired_seed_differences.csv",
        output_dir / "aggregation.json",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite aggregation outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    validate_runs(runs)
    comparison = aggregate_rows(runs)
    paired = paired_rows(runs)
    write_csv(output_paths[0], comparison)
    output_paths[1].write_text(markdown_table(comparison), encoding="utf-8")
    write_csv(output_paths[2], paired)
    write_json(
        output_paths[3],
        {
            "input_root": str(root),
            "run_summaries": [run["_path"] for run in runs],
            "comparison": comparison,
            "paired_seed_differences": paired,
            "difference_definition": "joint - quantization_only; lower PPL and KL are better",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
