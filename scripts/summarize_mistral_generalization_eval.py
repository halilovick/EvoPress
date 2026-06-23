#!/usr/bin/env python3
"""Summarize cross-dataset Mistral generalization evaluations."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


METHOD_ORDER = ("dense", "depth", "independent", "joint_g50")
METHOD_LABELS = {
    "dense": "Dense FP16",
    "depth": "Depth-only",
    "independent": "Independent depth + q_proj quant",
    "joint_g50": "Joint G50 depth + q_proj quant",
}
RUN_COLUMNS = (
    "method",
    "method_label",
    "seed",
    "run_id",
    "dataset",
    "ppl",
)
AGGREGATE_COLUMNS = (
    "method",
    "method_label",
    "dataset",
    "runs",
    "ppl_mean",
    "ppl_sample_std",
    "ppl_min",
    "ppl_max",
)
PAIRED_COLUMNS = (
    "dataset",
    "seed",
    "depth_ppl",
    "independent_ppl",
    "joint_g50_ppl",
    "joint_minus_depth",
    "joint_minus_independent",
    "independent_minus_depth",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize results/runs/generalization_* Mistral evaluations "
            "across WikiText2, C4, and FineWeb-Edu."
        )
    )
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-prefix", default="generalization")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "c4", "fineweb_edu"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_ORDER),
        choices=METHOD_ORDER,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Summarize available runs instead of failing on missing artifacts.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def run_id_for(
    method: str,
    seed: int,
    run_prefix: str,
    sequence_length: int,
) -> str:
    if method == "dense":
        return f"{run_prefix}_dense_mistral_multidataset_seq{sequence_length}_seed0"
    if method == "depth":
        return f"{run_prefix}_depth_mistral_s0.25_multidataset_seed{seed}"
    if method == "independent":
        return (
            f"{run_prefix}_independent_depth_quant_mistral_"
            f"s0.25_qproj3.0_multidataset_seed{seed}"
        )
    if method == "joint_g50":
        return (
            f"{run_prefix}_joint_g50_mistral_"
            f"s0.25_qproj3.0_multidataset_seed{seed}"
        )
    raise ValueError(f"Unsupported method: {method}")


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, float] = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset in metrics:
            raise ValueError(f"Duplicate dataset in {path}: {dataset}")
        metrics[dataset] = float(row["ppl"])
    return metrics


def load_run_rows(
    runs_root: Path,
    *,
    run_prefix: str,
    sequence_length: int,
    seeds: Sequence[int],
    datasets: Sequence[str],
    methods: Sequence[str],
    allow_missing: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_seeds = [0] if method == "dense" else list(seeds)
        for seed in method_seeds:
            run_id = run_id_for(method, seed, run_prefix, sequence_length)
            metrics_path = runs_root / run_id / "evaluation_metrics.csv"
            if not metrics_path.is_file():
                if allow_missing:
                    continue
                raise SystemExit(f"Missing evaluation metrics: {metrics_path}")
            metrics = read_metrics(metrics_path)
            missing = [dataset for dataset in datasets if dataset not in metrics]
            if missing and not allow_missing:
                raise SystemExit(
                    f"Missing datasets in {metrics_path}: {', '.join(missing)}"
                )
            for dataset in datasets:
                if dataset not in metrics:
                    continue
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": seed,
                        "run_id": run_id,
                        "dataset": dataset,
                        "ppl": metrics[dataset],
                    }
                )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"])].append(float(row["ppl"]))

    aggregate: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_datasets = sorted(
            dataset for row_method, dataset in grouped if row_method == method
        )
        for dataset in method_datasets:
            values = grouped[(method, dataset)]
            aggregate.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "dataset": dataset,
                    "runs": len(values),
                    "ppl_mean": mean(values),
                    "ppl_sample_std": sample_std(values),
                    "ppl_min": min(values),
                    "ppl_max": max(values),
                }
            )
    return aggregate


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], row["seed"], row["dataset"]): float(row["ppl"])
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows if row["method"] != "dense"})
    datasets = sorted({row["dataset"] for row in rows})
    paired: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            depth = by_key.get(("depth", seed, dataset))
            independent = by_key.get(("independent", seed, dataset))
            joint = by_key.get(("joint_g50", seed, dataset))
            if depth is None or independent is None or joint is None:
                continue
            paired.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "depth_ppl": depth,
                    "independent_ppl": independent,
                    "joint_g50_ppl": joint,
                    "joint_minus_depth": joint - depth,
                    "joint_minus_independent": joint - independent,
                    "independent_minus_depth": independent - depth,
                }
            )
    return paired


def dataset_order_key(dataset: str, datasets: Sequence[str]) -> tuple[int, str]:
    try:
        return (list(datasets).index(dataset), dataset)
    except ValueError:
        return (len(datasets), dataset)


def write_markdown(
    path: Path,
    run_rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    datasets: Sequence[str],
) -> None:
    by_method_dataset = {
        (row["method"], row["dataset"]): row for row in aggregate
    }
    dense_by_dataset = {
        row["dataset"]: row for row in aggregate if row["method"] == "dense"
    }
    paired_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        paired_by_dataset[row["dataset"]].append(row)

    lines = [
        "# Mistral Generalization Evaluation",
        "",
        "This summary evaluates the same Mistral-7B compression configurations "
        "on multiple held-out datasets. The purpose is to check whether the "
        "joint-search result is only optimized for WikiText2 or whether it "
        "also transfers to C4 and FineWeb-Edu.",
        "",
        "## Aggregate Perplexity",
        "",
        "| Method | Dataset | Runs | Mean PPL | Std | Min | Max | Delta vs dense |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in sorted(
        {row["dataset"] for row in aggregate},
        key=lambda item: dataset_order_key(item, datasets),
    ):
        dense_mean = dense_by_dataset.get(dataset, {}).get("ppl_mean")
        for method in METHOD_ORDER:
            row = by_method_dataset.get((method, dataset))
            if not row:
                continue
            delta = (
                float(row["ppl_mean"]) - float(dense_mean)
                if dense_mean is not None
                else None
            )
            lines.append(
                "| {method} | {dataset} | {runs} | {mean} | {std} | "
                "{min_} | {max_} | {delta} |".format(
                    method=METHOD_LABELS[method],
                    dataset=dataset,
                    runs=row["runs"],
                    mean=fmt(row["ppl_mean"]),
                    std=fmt(row["ppl_sample_std"]),
                    min_=fmt(row["ppl_min"]),
                    max_=fmt(row["ppl_max"]),
                    delta=fmt(delta),
                )
            )

    lines.extend(
        [
            "",
            "## Paired Comparison",
            "",
            "| Dataset | Seed | Depth PPL | Independent PPL | Joint G50 PPL | "
            "Joint - depth | Joint - independent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in sorted(
        paired_by_dataset,
        key=lambda item: dataset_order_key(item, datasets),
    ):
        for row in sorted(paired_by_dataset[dataset], key=lambda item: item["seed"]):
            lines.append(
                "| {dataset} | {seed} | {depth} | {independent} | {joint} | "
                "{joint_depth} | {joint_independent} |".format(
                    dataset=dataset,
                    seed=row["seed"],
                    depth=fmt(row["depth_ppl"]),
                    independent=fmt(row["independent_ppl"]),
                    joint=fmt(row["joint_g50_ppl"]),
                    joint_depth=fmt(row["joint_minus_depth"]),
                    joint_independent=fmt(row["joint_minus_independent"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- If joint G50 is better than independent on all or most datasets, "
            "the thesis claim is stronger: joint selection improves transfer, "
            "not just WikiText2 fit.",
            "- If joint G50 only wins on WikiText2, present it as evidence that "
            "the current objective overfits the calibration/evaluation setup and "
            "needs broader calibration data.",
            "- If depth-only is close to or better than the combined methods, the "
            "next experiment should increase the quantized module scope or adjust "
            "the compression target, because q_proj-only quantization contributes "
            "limited compression.",
            "",
            "## Source Runs",
            "",
            "| Method | Seed | Run ID | Dataset | PPL |",
            "| --- | ---: | --- | --- | ---: |",
        ]
    )
    for row in sorted(
        run_rows,
        key=lambda item: (
            dataset_order_key(item["dataset"], datasets),
            METHOD_ORDER.index(item["method"]),
            int(item["seed"]),
        ),
    ):
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['seed']} | "
            f"`{row['run_id']}` | {row['dataset']} | {fmt(row['ppl'])} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path,
    aggregate: list[dict[str, Any]],
    datasets: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    observed_datasets = [
        dataset
        for dataset in datasets
        if any(row["dataset"] == dataset for row in aggregate)
    ]
    x_positions = list(range(len(observed_datasets)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(METHOD_ORDER):
        values = []
        for dataset in observed_datasets:
            matches = [
                row
                for row in aggregate
                if row["method"] == method and row["dataset"] == dataset
            ]
            values.append(matches[0]["ppl_mean"] if matches else 0.0)
        offsets = [
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width
            for x in x_positions
        ]
        ax.bar(offsets, values, width=width, label=METHOD_LABELS[method])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(observed_datasets)
    ax.set_ylabel("Perplexity")
    ax.set_title("Mistral Generalization Evaluation")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    run_rows = load_run_rows(
        runs_root,
        run_prefix=args.run_prefix,
        sequence_length=args.sequence_length,
        seeds=args.seeds,
        datasets=args.datasets,
        methods=args.methods,
        allow_missing=args.allow_missing,
    )
    if not run_rows:
        raise SystemExit("No generalization evaluation rows were found.")

    aggregate = aggregate_rows(run_rows)
    paired = paired_rows(run_rows)

    write_csv(
        output_dir / "mistral_generalization_eval.csv",
        RUN_COLUMNS,
        run_rows,
    )
    write_csv(
        output_dir / "mistral_generalization_aggregate.csv",
        AGGREGATE_COLUMNS,
        aggregate,
    )
    write_csv(
        output_dir / "mistral_generalization_paired_deltas.csv",
        PAIRED_COLUMNS,
        paired,
    )
    write_markdown(
        output_dir / "mistral_generalization_eval.md",
        run_rows,
        aggregate,
        paired,
        args.datasets,
    )
    if not args.skip_plots:
        write_plot(
            output_dir / "mistral_generalization_eval.png",
            aggregate,
            args.datasets,
        )

    print(f"Wrote {len(run_rows)} run rows.")
    print(f"Wrote {len(aggregate)} aggregate rows.")
    print(f"Wrote {len(paired)} paired comparison rows.")


if __name__ == "__main__":
    main()
