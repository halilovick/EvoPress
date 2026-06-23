#!/usr/bin/env python3
"""Summarize Mistral LM Evaluation Harness comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
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
PREFERRED_METRICS = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,none",
    "f1,none",
)
SCORE_COLUMNS = (
    "method",
    "method_label",
    "seed",
    "run_id",
    "task",
    "metric",
    "score",
)
AGGREGATE_COLUMNS = (
    "method",
    "method_label",
    "task",
    "metric",
    "runs",
    "score_mean",
    "score_sample_std",
    "score_min",
    "score_max",
)
PAIRED_COLUMNS = (
    "task",
    "metric",
    "seed",
    "depth_score",
    "independent_score",
    "joint_g50_score",
    "joint_minus_depth",
    "joint_minus_independent",
    "independent_minus_depth",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize lmeval_* downstream-task runs."
    )
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-prefix", default="lmeval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["arc_easy", "piqa", "winogrande"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_ORDER),
        choices=METHOD_ORDER,
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def run_id_for(method: str, seed: int, run_prefix: str) -> str:
    if method == "dense":
        return f"{run_prefix}_dense_mistral_tasks_seed0"
    if method == "depth":
        return f"{run_prefix}_depth_mistral_s0.25_tasks_seed{seed}"
    if method == "independent":
        return (
            f"{run_prefix}_independent_depth_quant_mistral_"
            f"s0.25_qproj3.0_tasks_seed{seed}"
        )
    if method == "joint_g50":
        return (
            f"{run_prefix}_joint_g50_mistral_"
            f"s0.25_qproj3.0_tasks_seed{seed}"
        )
    raise ValueError(f"Unsupported method: {method}")


def choose_metric(task: str, metrics: dict[str, Any]) -> tuple[str, float]:
    for metric in PREFERRED_METRICS:
        if metric in metrics:
            return metric, float(metrics[metric])
    for metric, value in metrics.items():
        if metric.endswith("_stderr") or metric.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)):
            return metric, float(value)
    raise ValueError(f"No scalar score metric found for task {task}.")


def load_scores(
    runs_root: Path,
    *,
    run_prefix: str,
    seeds: Sequence[int],
    tasks: Sequence[str],
    methods: Sequence[str],
    allow_missing: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_seeds = [0] if method == "dense" else list(seeds)
        for seed in method_seeds:
            run_id = run_id_for(method, seed, run_prefix)
            path = runs_root / run_id / "lmeval_results.json"
            if not path.is_file():
                if allow_missing:
                    continue
                raise SystemExit(f"Missing LM-eval results: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            results = data.get("results", {})
            missing = [task for task in tasks if task not in results]
            if missing and not allow_missing:
                raise SystemExit(
                    f"Missing tasks in {path}: {', '.join(missing)}"
                )
            for task in tasks:
                if task not in results:
                    continue
                metric, score = choose_metric(task, results[task])
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": seed,
                        "run_id": run_id,
                        "task": task,
                        "metric": metric,
                        "score": score,
                    }
                )
    return rows


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


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


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["task"], row["metric"])].append(
            float(row["score"])
        )

    aggregate: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        keys = sorted(
            key for key in grouped if key[0] == method
        )
        for _, task, metric in keys:
            values = grouped[(method, task, metric)]
            aggregate.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "metric": metric,
                    "runs": len(values),
                    "score_mean": mean(values),
                    "score_sample_std": sample_std(values),
                    "score_min": min(values),
                    "score_max": max(values),
                }
            )
    return aggregate


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], row["seed"], row["task"]): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows if row["method"] != "dense"})
    tasks = sorted({row["task"] for row in rows})
    paired: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            depth = by_key.get(("depth", seed, task))
            independent = by_key.get(("independent", seed, task))
            joint = by_key.get(("joint_g50", seed, task))
            if depth is None or independent is None or joint is None:
                continue
            paired.append(
                {
                    "task": task,
                    "metric": joint["metric"],
                    "seed": seed,
                    "depth_score": depth["score"],
                    "independent_score": independent["score"],
                    "joint_g50_score": joint["score"],
                    "joint_minus_depth": joint["score"] - depth["score"],
                    "joint_minus_independent": (
                        joint["score"] - independent["score"]
                    ),
                    "independent_minus_depth": (
                        independent["score"] - depth["score"]
                    ),
                }
            )
    return paired


def macro_scores(aggregate: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in aggregate:
        grouped[row["method"]].append(float(row["score_mean"]))
    return {
        method: statistics.mean(values)
        for method, values in grouped.items()
        if values
    }


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    macros = macro_scores(aggregate)
    lines = [
        "# Mistral Downstream LM-Eval Comparison",
        "",
        "This summary evaluates whether the Mistral joint compression result "
        "transfers from perplexity to downstream multiple-choice tasks. Higher "
        "scores are better.",
        "",
        "## Macro Average",
        "",
        "| Method | Macro score |",
        "| --- | ---: |",
    ]
    for method in METHOD_ORDER:
        if method in macros:
            lines.append(
                f"| {METHOD_LABELS[method]} | {fmt(macros[method])} |"
            )

    lines.extend(
        [
            "",
            "## Task Averages",
            "",
            "| Method | Task | Metric | Runs | Mean | Std | Min | Max |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['task']} | "
            f"{row['metric']} | {row['runs']} | {fmt(row['score_mean'])} | "
            f"{fmt(row['score_sample_std'])} | {fmt(row['score_min'])} | "
            f"{fmt(row['score_max'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| Task | Seed | Metric | Depth | Independent | Joint G50 | "
            "Joint - depth | Joint - independent |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['metric']} | "
            f"{fmt(row['depth_score'])} | {fmt(row['independent_score'])} | "
            f"{fmt(row['joint_g50_score'])} | "
            f"{fmt(row['joint_minus_depth'])} | "
            f"{fmt(row['joint_minus_independent'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- If joint G50 improves macro score over independent composition, "
            "the joint-search advantage transfers to task accuracy.",
            "- If perplexity improves but task scores do not, report the result "
            "as a limitation and keep downstream alignment as future work.",
            "- Limited LM-eval runs are smoke tests only. Final reported task "
            "metrics should not use `--limit`.",
            "",
            "## Source Runs",
            "",
            "| Method | Seed | Run ID | Task | Metric | Score |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['seed']} | "
            f"`{row['run_id']}` | {row['task']} | {row['metric']} | "
            f"{fmt(row['score'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, aggregate: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    tasks = sorted({row["task"] for row in aggregate})
    x_positions = list(range(len(tasks)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(METHOD_ORDER):
        values = []
        for task in tasks:
            matches = [
                row
                for row in aggregate
                if row["method"] == method and row["task"] == task
            ]
            values.append(matches[0]["score_mean"] if matches else 0.0)
        offsets = [
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width
            for x in x_positions
        ]
        ax.bar(offsets, values, width=width, label=METHOD_LABELS[method])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Score")
    ax.set_title("Mistral Downstream LM-Eval")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_scores(
        Path(args.runs_root),
        run_prefix=args.run_prefix,
        seeds=args.seeds,
        tasks=args.tasks,
        methods=args.methods,
        allow_missing=args.allow_missing,
    )
    if not rows:
        raise SystemExit("No LM-eval score rows were found.")
    aggregate = aggregate_rows(rows)
    paired = paired_rows(rows)
    output_dir = Path(args.output_dir)

    write_csv(output_dir / "mistral_lmeval_task_scores.csv", SCORE_COLUMNS, rows)
    write_csv(
        output_dir / "mistral_lmeval_aggregate.csv",
        AGGREGATE_COLUMNS,
        aggregate,
    )
    write_csv(
        output_dir / "mistral_lmeval_paired_deltas.csv",
        PAIRED_COLUMNS,
        paired,
    )
    write_markdown(
        output_dir / "mistral_lmeval_comparison.md",
        rows,
        aggregate,
        paired,
    )
    if not args.skip_plots:
        write_plot(output_dir / "mistral_lmeval_comparison.png", aggregate)

    print(f"Wrote {len(rows)} task score rows.")
    print(f"Wrote {len(aggregate)} aggregate rows.")
    print(f"Wrote {len(paired)} paired comparison rows.")


if __name__ == "__main__":
    main()
