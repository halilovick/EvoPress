#!/usr/bin/env python3
"""Aggregate replay attribution matrices across seeds or runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


METRICS = (
    "wiki2_ppl",
    "c4_ppl",
    "fineweb_ppl",
    "compression_ratio",
    "active_params",
    "avg_active_bitwidth",
    "repair_num_changes",
)

SUMMARY_COLUMNS = (
    "depth_label",
    "quant_label",
    "rows_total",
    "rows_completed",
    "rows_failed",
    "wiki2_ppl_mean",
    "wiki2_ppl_std",
    "wiki2_ppl_min",
    "wiki2_ppl_max",
    "c4_ppl_mean",
    "c4_ppl_std",
    "c4_ppl_min",
    "c4_ppl_max",
    "fineweb_ppl_mean",
    "fineweb_ppl_std",
    "fineweb_ppl_min",
    "fineweb_ppl_max",
    "compression_ratio_mean",
    "compression_ratio_std",
    "compression_ratio_min",
    "compression_ratio_max",
    "active_params_mean",
    "active_params_std",
    "active_params_min",
    "active_params_max",
    "avg_active_bitwidth_mean",
    "avg_active_bitwidth_std",
    "avg_active_bitwidth_min",
    "avg_active_bitwidth_max",
    "repair_num_changes_mean",
    "repair_num_changes_std",
    "repair_num_changes_min",
    "repair_num_changes_max",
    "failed_sources",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evo_joint_attribution.py matrix CSV files."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--output_md",
        default=None,
        help="Markdown output path. Defaults to the CSV path with .md suffix.",
    )
    return parser.parse_args(argv)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def read_rows(inputs: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_path in inputs:
        path = Path(input_path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def metric_summary(values: Sequence[float]) -> dict[str, float | str]:
    if not values:
        return {"mean": "", "std": "", "min": "", "max": ""}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else "",
        "min": min(values),
        "max": max(values),
    }


def aggregate(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("depth_label", ""), row.get("quant_label", ""))].append(row)

    summaries: list[dict[str, Any]] = []
    for (depth_label, quant_label), group_rows in sorted(grouped.items()):
        completed = [row for row in group_rows if row.get("status") == "completed"]
        failed = [row for row in group_rows if row.get("status") != "completed"]
        summary: dict[str, Any] = {
            "depth_label": depth_label,
            "quant_label": quant_label,
            "rows_total": len(group_rows),
            "rows_completed": len(completed),
            "rows_failed": len(failed),
            "failed_sources": "; ".join(
                f"{row.get('_source_file', '')}: {row.get('error_message', '')}"
                for row in failed
            ),
        }
        for metric in METRICS:
            values = [
                numeric
                for numeric in (parse_float(row.get(metric)) for row in completed)
                if numeric is not None
            ]
            stats = metric_summary(values)
            for stat_name, stat_value in stats.items():
                summary[f"{metric}_{stat_name}"] = stat_value
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def fmt(value: Any) -> str:
    numeric = parse_float(value)
    if numeric is None:
        return ""
    if abs(numeric) >= 1000:
        return f"{numeric:.2f}"
    return f"{numeric:.4g}"


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Joint Compression Attribution Aggregate",
        "",
        "| Depth source | Quant source | Runs | WikiText2 PPL mean | C4 PPL mean | FineWeb-Edu PPL mean | Compression mean | Avg active bits | Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('depth_label', '')} | "
            f"{row.get('quant_label', '')} | "
            f"{row.get('rows_completed', '')}/{row.get('rows_total', '')} | "
            f"{fmt(row.get('wiki2_ppl_mean'))} | "
            f"{fmt(row.get('c4_ppl_mean'))} | "
            f"{fmt(row.get('fineweb_ppl_mean'))} | "
            f"{fmt(row.get('compression_ratio_mean'))} | "
            f"{fmt(row.get('avg_active_bitwidth_mean'))} | "
            f"{row.get('rows_failed', '')} |"
        )
    lines += [
        "",
        "The aggregate includes only rows with `status=completed` in metric means.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = aggregate(read_rows(args.inputs))
    output = Path(args.output)
    output_md = Path(args.output_md) if args.output_md else output.with_suffix(".md")
    write_csv(output, rows)
    write_markdown(output_md, rows)
    print(f"Wrote aggregate CSV: {output}")
    print(f"Wrote aggregate Markdown: {output_md}")


if __name__ == "__main__":
    main()
