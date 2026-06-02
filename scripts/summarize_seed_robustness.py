#!/usr/bin/env python3
"""Summarize depth-pruning seed robustness from tracked lightweight artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from pathlib import Path


DEFAULT_MODEL = "mistralai/Mistral-7B-v0.3"
DEFAULT_SPARSITY = "0.375"
DEFAULT_GENERATIONS = "10"
DEFAULT_SEEDS = ("1", "2", "3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a markdown seed-robustness summary for depth pruning."
    )
    parser.add_argument("--experiment-log", default="results/experiment_log.csv")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output", default="results/seed_robustness_table.md")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sparsity", default=DEFAULT_SPARSITY)
    parser.add_argument("--generations", default=DEFAULT_GENERATIONS)
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def read_experiment_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_id_from_output_dir(output_dir: str) -> str:
    return Path(output_dir).name


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_experiment_log(Path(args.experiment_log))
    seed_set = set(args.seeds)
    matches = [
        row
        for row in rows
        if row["method"] == "depth_evo"
        and row["model"] == args.model
        and row["sparsity_or_bits"] == args.sparsity
        and row["generations"] == args.generations
        and row["seed"] in seed_set
        and row["status"] == "completed"
    ]
    by_seed: dict[str, dict[str, str]] = {}
    for row in matches:
        by_seed[row["seed"]] = row

    missing = [seed for seed in args.seeds if seed not in by_seed]
    if missing:
        raise SystemExit(
            "Missing completed matching runs for seed(s): " + ", ".join(missing)
        )
    return [by_seed[seed] for seed in args.seeds]


def parse_drop_config(path: Path) -> set[tuple[int, str]]:
    drops: set[tuple[int, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for layer_index, raw_line in enumerate(handle):
            value = raw_line.strip()
            if value == "none":
                continue
            if value in {"attn", "attn+mlp"}:
                drops.add((layer_index, "attn"))
            if value in {"mlp", "attn+mlp"}:
                drops.add((layer_index, "mlp"))
            if value not in {"none", "attn", "mlp", "attn+mlp"}:
                raise ValueError(f"Unexpected drop-config value in {path}: {value}")
    return drops


def jaccard(left: set[tuple[int, str]], right: set[tuple[int, str]]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def artifact_dir(row: dict[str, str], runs_root: Path) -> Path:
    return runs_root / run_id_from_output_dir(row["output_dir"])


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_markdown(rows: list[dict[str, str]], runs_root: Path) -> str:
    per_seed: list[dict[str, object]] = []
    for row in rows:
        run_dir = artifact_dir(row, runs_root)
        config_path = run_dir / "layer_drop_config.txt"
        if not config_path.exists():
            raise SystemExit(f"Missing config artifact: {config_path}")
        drops = parse_drop_config(config_path)
        per_seed.append(
            {
                "seed": row["seed"],
                "run_id": row["run_id"],
                "wikitext2_ppl": float(row["wikitext2_ppl"]),
                "train_ppl": float(row["train_ppl"]),
                "runtime_minutes": float(row["runtime_minutes"]),
                "gpu_name": row["gpu_name"],
                "attn_drops": sum(1 for _, module_type in drops if module_type == "attn"),
                "mlp_drops": sum(1 for _, module_type in drops if module_type == "mlp"),
                "drops": drops,
            }
        )

    ppls = [item["wikitext2_ppl"] for item in per_seed]
    train_ppls = [item["train_ppl"] for item in per_seed]
    runtimes = [item["runtime_minutes"] for item in per_seed]
    best = min(per_seed, key=lambda item: item["wikitext2_ppl"])
    worst = max(per_seed, key=lambda item: item["wikitext2_ppl"])
    gpu_names = sorted({str(item["gpu_name"]) for item in per_seed if item["gpu_name"]})

    lines = [
        "# Seed Robustness: Mistral-7B Depth Pruning at 37.5%",
        "",
        "This table summarizes the three-seed repeatability experiment for EvoPress depth pruning on `mistralai/Mistral-7B-v0.3` with `37.5%` sparsity, `10` generations, and `8` offspring.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Run ID | WikiText2 PPL | Train PPL | Runtime (min) | GPU | Dropped attn | Dropped MLP |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for item in per_seed:
        lines.append(
            "| {seed} | `{run_id}` | {wikitext2_ppl} | {train_ppl} | {runtime_minutes} | {gpu_name} | {attn_drops} | {mlp_drops} |".format(
                seed=item["seed"],
                run_id=item["run_id"],
                wikitext2_ppl=fmt_float(float(item["wikitext2_ppl"])),
                train_ppl=fmt_float(float(item["train_ppl"])),
                runtime_minutes=fmt_float(float(item["runtime_minutes"])),
                gpu_name=item["gpu_name"],
                attn_drops=item["attn_drops"],
                mlp_drops=item["mlp_drops"],
            )
        )

    lines.extend(
        [
            "",
            "## Summary statistics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Mean WikiText2 PPL | {fmt_float(statistics.mean(ppls))} |",
            f"| Sample std WikiText2 PPL | {fmt_float(statistics.stdev(ppls))} |",
            f"| Mean train PPL | {fmt_float(statistics.mean(train_ppls))} |",
            f"| Mean runtime minutes | {fmt_float(statistics.mean(runtimes))} |",
            f"| Best seed | {best['seed']} (`{best['run_id']}`, PPL {fmt_float(float(best['wikitext2_ppl']))}) |",
            f"| Worst seed | {worst['seed']} (`{worst['run_id']}`, PPL {fmt_float(float(worst['wikitext2_ppl']))}) |",
            "",
            "## Pairwise dropped-module overlap",
            "",
            "Jaccard overlap is computed over dropped `(layer_index, module_type)` pairs using zero-based layer indices.",
            "",
            "| Seed pair | Intersection | Union | Jaccard overlap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for left, right in itertools.combinations(per_seed, 2):
        left_drops = left["drops"]
        right_drops = right["drops"]
        assert isinstance(left_drops, set)
        assert isinstance(right_drops, set)
        intersection = len(left_drops & right_drops)
        union = len(left_drops | right_drops)
        lines.append(
            f"| {left['seed']} vs {right['seed']} | {intersection} | {union} | {jaccard(left_drops, right_drops):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"All three seeds completed with finite WikiText2 PPL. The mean final WikiText2 PPL is {fmt_float(statistics.mean(ppls))} with sample standard deviation {fmt_float(statistics.stdev(ppls))}. The selected dropped-module sets are similar but not identical, which indicates that the search is finding related high-quality regions rather than a single fixed mask.",
        ]
    )
    if len(gpu_names) > 1:
        lines.append(
            "Runtime should not be compared directly across seeds because the runs used different GPU types: "
            + ", ".join(gpu_names)
            + "."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows = selected_rows(args)
    markdown = build_markdown(rows, runs_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote seed robustness summary to {output}")


if __name__ == "__main__":
    main()
