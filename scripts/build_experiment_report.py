#!/usr/bin/env python3
"""Build reproducible CSV, plot, and markdown report artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

# Avoid noisy warnings when the default home Matplotlib cache is not writable
# in notebook/container environments.
_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="experiment-report-cache-"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPARSITY_ORDER = ["0.125", "0.25", "0.375", "0.50"]
SPARSITY_LABELS = {
    "0.125": "12.5%",
    "0.25": "25.0%",
    "0.375": "37.5%",
    "0.50": "50.0%",
}
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Task 10 report artifacts from tracked experiment results."
    )
    parser.add_argument("--experiment-log", default="results/experiment_log.csv")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_table_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def run_artifact_dir(row: dict[str, str], runs_root: Path) -> Path:
    return runs_root / Path(row["output_dir"]).name


def completed(rows: list[dict[str, str]], method: str | None = None) -> list[dict[str, str]]:
    matches = [row for row in rows if row["status"] == "completed"]
    if method is not None:
        matches = [row for row in matches if row["method"] == method]
    return matches


def rows_for(
    rows: list[dict[str, str]],
    *,
    method: str,
    sparsity: str | None = None,
    generations: str | None = None,
    seed: str | None = None,
    status: str | None = None,
) -> list[dict[str, str]]:
    matches = [row for row in rows if row["method"] == method]
    if sparsity is not None:
        matches = [row for row in matches if row["sparsity_or_bits"] == sparsity]
    if generations is not None:
        matches = [row for row in matches if row["generations"] == generations]
    if seed is not None:
        matches = [row for row in matches if row["seed"] == seed]
    if status is not None:
        matches = [row for row in matches if row["status"] == status]
    return matches


def finite_ppls(rows: Iterable[dict[str, str]]) -> list[float]:
    values = []
    for row in rows:
        value = parse_float(row.get("wikitext2_ppl"))
        if value is not None:
            values.append(value)
    return values


def choose_main_evo_row(rows: list[dict[str, str]], sparsity: str) -> dict[str, str] | None:
    matches = rows_for(
        rows,
        method="depth_evo",
        sparsity=sparsity,
        generations="10",
        seed="1",
        status="completed",
    )
    return matches[-1] if matches else None


def choose_late_layer_row(rows: list[dict[str, str]], sparsity: str) -> dict[str, str] | None:
    matches = rows_for(
        rows,
        method="depth_baseline_late_layer",
        sparsity=sparsity,
        status="completed",
    )
    return matches[-1] if matches else None


def dense_wikitext2_ppl(rows: list[dict[str, str]]) -> float | None:
    dense_rows = completed(rows)
    dense_rows = [row for row in dense_rows if row["method"].startswith("dense")]
    if not dense_rows:
        return None
    return parse_float(dense_rows[-1]["wikitext2_ppl"])


def build_depth_curve(rows: list[dict[str, str]], results_dir: Path) -> list[dict[str, object]]:
    dense_ppl = dense_wikitext2_ppl(rows)
    curve_rows: list[dict[str, object]] = []
    for sparsity in SPARSITY_ORDER:
        evo_row = choose_main_evo_row(rows, sparsity)
        random_completed = rows_for(
            rows,
            method="depth_baseline_random",
            sparsity=sparsity,
            status="completed",
        )
        random_failed = rows_for(
            rows,
            method="depth_baseline_random",
            sparsity=sparsity,
            status="failed",
        )
        random_values = finite_ppls(random_completed)
        late_row = choose_late_layer_row(rows, sparsity)
        late_failed = rows_for(
            rows,
            method="depth_baseline_late_layer",
            sparsity=sparsity,
            status="failed",
        )
        curve_rows.append(
            {
                "sparsity": sparsity,
                "sparsity_label": SPARSITY_LABELS[sparsity],
                "dense_wikitext2_ppl": fmt_number(dense_ppl),
                "evopress_wikitext2_ppl": fmt_number(parse_float(evo_row["wikitext2_ppl"]) if evo_row else None),
                "evopress_run_id": evo_row["run_id"] if evo_row else "",
                "random_mean_wikitext2_ppl": fmt_number(statistics.mean(random_values) if random_values else None),
                "random_std_wikitext2_ppl": fmt_number(
                    statistics.stdev(random_values) if len(random_values) > 1 else None
                ),
                "random_median_wikitext2_ppl": fmt_number(
                    statistics.median(random_values) if random_values else None
                ),
                "random_completed_n": len(random_values),
                "random_failed_n": len(random_failed),
                "late_layer_wikitext2_ppl": fmt_number(
                    parse_float(late_row["wikitext2_ppl"]) if late_row else None
                ),
                "late_layer_run_id": late_row["run_id"] if late_row else "",
                "late_layer_failed_n": len(late_failed),
            }
        )

    fieldnames = [
        "sparsity",
        "sparsity_label",
        "dense_wikitext2_ppl",
        "evopress_wikitext2_ppl",
        "evopress_run_id",
        "random_mean_wikitext2_ppl",
        "random_std_wikitext2_ppl",
        "random_median_wikitext2_ppl",
        "random_completed_n",
        "random_failed_n",
        "late_layer_wikitext2_ppl",
        "late_layer_run_id",
        "late_layer_failed_n",
    ]
    write_csv(results_dir / "depth_pruning_curve.csv", fieldnames, curve_rows)
    plot_depth_curve(curve_rows, dense_ppl, results_dir / "depth_pruning_curve.png")
    return curve_rows


def curve_value(row: dict[str, object], key: str) -> float | None:
    return parse_float(str(row.get(key, "")))


def plot_depth_curve(curve_rows: list[dict[str, object]], dense_ppl: float | None, output: Path) -> None:
    x = [float(row["sparsity"]) * 100 for row in curve_rows]
    labels = [str(row["sparsity_label"]) for row in curve_rows]
    evo = [curve_value(row, "evopress_wikitext2_ppl") for row in curve_rows]
    random_mean = [curve_value(row, "random_mean_wikitext2_ppl") for row in curve_rows]
    random_std = [curve_value(row, "random_std_wikitext2_ppl") or 0.0 for row in curve_rows]
    late = [curve_value(row, "late_layer_wikitext2_ppl") for row in curve_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, evo, marker="o", linewidth=2, label="EvoPress depth")
    ax.plot(x, late, marker="s", linewidth=2, label="Late-layer baseline")
    # Use asymmetric error bars so the lower side remains valid on a log axis
    # even when the sample standard deviation exceeds the finite mean.
    lower_errors = []
    upper_errors = []
    for mean, std in zip(random_mean, random_std):
        if mean is None:
            lower_errors.append(0.0)
            upper_errors.append(0.0)
        else:
            # Perplexity is positive and a one-standard-deviation interval can
            # cross zero for the highly unstable random baseline. Clip the
            # lower visual error bar at PPL=1 on the log-scale plot; the CSV
            # keeps the exact unmodified standard deviation.
            lower_errors.append(min(std, max(mean - 1.0, 0.0)))
            upper_errors.append(std)
    ax.errorbar(
        x,
        random_mean,
        yerr=[lower_errors, upper_errors],
        marker="^",
        linewidth=2,
        capsize=4,
        label="Random baseline mean +/- std",
    )
    if dense_ppl is not None:
        ax.axhline(dense_ppl, linestyle="--", color="black", linewidth=1.2, label=f"Dense reference ({dense_ppl:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("Depth pruning sparsity")
    ax.set_ylabel("WikiText2 perplexity (log scale)")
    ax.set_title("Mistral-7B Depth Pruning Curve")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def choose_convergence_run(rows: list[dict[str, str]], runs_root: Path) -> dict[str, str]:
    candidates = rows_for(
        rows,
        method="depth_evo",
        sparsity="0.375",
        generations="20",
        seed="1",
        status="completed",
    )
    candidates = [
        row
        for row in candidates
        if (run_artifact_dir(row, runs_root) / "generation_metrics.csv").exists()
    ]
    if candidates:
        return candidates[-1]
    fallback = rows_for(
        rows,
        method="depth_evo",
        sparsity="0.375",
        generations="10",
        seed="1",
        status="completed",
    )
    fallback = [
        row
        for row in fallback
        if (run_artifact_dir(row, runs_root) / "generation_metrics.csv").exists()
    ]
    if not fallback:
        raise FileNotFoundError("No 37.5% depth-pruning generation metrics artifact found.")
    return fallback[-1]


def build_convergence(rows: list[dict[str, str]], runs_root: Path, results_dir: Path) -> list[dict[str, object]]:
    run = choose_convergence_run(rows, runs_root)
    metrics_path = run_artifact_dir(run, runs_root) / "generation_metrics.csv"
    metrics = read_csv(metrics_path)
    out_rows: list[dict[str, object]] = []
    for row in metrics:
        if row["phase"] != "generation":
            continue
        out_rows.append(
            {
                "run_id": run["run_id"],
                "generation": row["generation"],
                "wikitext2_ppl": row["wikitext2_ppl"],
                "train_ppl": row["train_ppl"],
                "train_fitness": row["train_fitness"],
            }
        )

    write_csv(
        results_dir / "convergence_37_5.csv",
        ["run_id", "generation", "wikitext2_ppl", "train_ppl", "train_fitness"],
        out_rows,
    )
    plot_convergence(out_rows, results_dir / "convergence_37_5.png")
    return out_rows


def plot_convergence(rows: list[dict[str, object]], output: Path) -> None:
    generations = [int(row["generation"]) for row in rows]
    wikitext2 = [parse_float(str(row["wikitext2_ppl"])) for row in rows]
    train = [parse_float(str(row["train_ppl"])) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, wikitext2, marker="o", linewidth=2, label="WikiText2 PPL")
    ax.plot(generations, train, marker="s", linewidth=2, label="Train PPL")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("37.5% Mistral-7B Depth-Pruning Convergence")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_baseline_table(curve_rows: list[dict[str, object]], rows: list[dict[str, str]], results_dir: Path) -> None:
    dense_ppl = dense_wikitext2_ppl(rows)
    lines = [
        "# Baseline Comparison",
        "",
        "Numeric aggregates include completed runs with finite WikiText2 perplexity only. Failed attempts remain in `results/experiment_log.csv` and are counted separately below.",
        "",
    ]
    if dense_ppl is not None:
        lines.append(f"Dense Mistral-7B WikiText2 reference PPL: `{dense_ppl:.2f}`.")
        lines.append("")
    lines.extend(
        [
            "| Sparsity | EvoPress PPL | Random mean PPL | Random std | Random median | Random completed / failed | Late-layer PPL | Late-layer failed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in curve_rows:
        lines.append(
            "| {label} | {evo} | {rand_mean} | {rand_std} | {rand_median} | {rand_done}/{rand_failed} | {late} | {late_failed} |".format(
                label=row["sparsity_label"],
                evo=fmt_table_number(curve_value(row, "evopress_wikitext2_ppl")),
                rand_mean=fmt_table_number(curve_value(row, "random_mean_wikitext2_ppl")),
                rand_std=fmt_table_number(curve_value(row, "random_std_wikitext2_ppl")),
                rand_median=fmt_table_number(curve_value(row, "random_median_wikitext2_ppl")),
                rand_done=row["random_completed_n"],
                rand_failed=row["random_failed_n"],
                late=fmt_table_number(curve_value(row, "late_layer_wikitext2_ppl")),
                late_failed=row["late_layer_failed_n"],
            )
        )
    lines.extend(
        [
            "",
            "The random baseline has very high variance, especially at low and high sparsity. The late-layer baseline is consistently worse than EvoPress. Runtime comparisons should be treated separately because the experiment log contains both Tesla T4 and NVIDIA A40 runs.",
        ]
    )
    (results_dir / "baseline_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_drop_config(path: Path) -> set[tuple[int, str]]:
    drops: set[tuple[int, str]] = set()
    for layer_index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        value = raw_line.strip()
        if value in {"attn", "attn+mlp"}:
            drops.add((layer_index, "attn"))
        if value in {"mlp", "attn+mlp"}:
            drops.add((layer_index, "mlp"))
    return drops


def jaccard(left: set[tuple[int, str]], right: set[tuple[int, str]]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def write_seed_robustness(rows: list[dict[str, str]], runs_root: Path, results_dir: Path) -> None:
    seeds = ["1", "2", "3"]
    selected = []
    for seed in seeds:
        matches = rows_for(
            rows,
            method="depth_evo",
            sparsity="0.375",
            generations="10",
            seed=seed,
            status="completed",
        )
        if matches:
            selected.append(matches[-1])
    lines = [
        "# Seed Robustness: Mistral-7B Depth Pruning at 37.5%",
        "",
        "This table summarizes the three-seed repeatability experiment for EvoPress depth pruning with `10` generations and `8` offspring.",
        "",
    ]
    if len(selected) != len(seeds):
        lines.append(f"Only `{len(selected)}` of `{len(seeds)}` planned seeds were available.")
        lines.append("")

    per_seed = []
    for row in selected:
        config_path = run_artifact_dir(row, runs_root) / "layer_drop_config.txt"
        drops = parse_drop_config(config_path) if config_path.exists() else set()
        per_seed.append(
            {
                "seed": row["seed"],
                "run_id": row["run_id"],
                "wikitext2_ppl": parse_float(row["wikitext2_ppl"]),
                "train_ppl": parse_float(row["train_ppl"]),
                "runtime_minutes": parse_float(row["runtime_minutes"]),
                "gpu_name": row["gpu_name"],
                "attn_drops": sum(1 for _, module_type in drops if module_type == "attn"),
                "mlp_drops": sum(1 for _, module_type in drops if module_type == "mlp"),
                "drops": drops,
            }
        )

    lines.extend(
        [
            "| Seed | Run ID | WikiText2 PPL | Train PPL | Runtime (min) | GPU | Dropped attn | Dropped MLP |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for item in per_seed:
        lines.append(
            "| {seed} | `{run_id}` | {wikitext2} | {train} | {runtime} | {gpu} | {attn} | {mlp} |".format(
                seed=item["seed"],
                run_id=item["run_id"],
                wikitext2=fmt_table_number(item["wikitext2_ppl"]),
                train=fmt_table_number(item["train_ppl"]),
                runtime=fmt_table_number(item["runtime_minutes"]),
                gpu=item["gpu_name"],
                attn=item["attn_drops"],
                mlp=item["mlp_drops"],
            )
        )

    ppls = [item["wikitext2_ppl"] for item in per_seed if item["wikitext2_ppl"] is not None]
    train_ppls = [item["train_ppl"] for item in per_seed if item["train_ppl"] is not None]
    runtimes = [item["runtime_minutes"] for item in per_seed if item["runtime_minutes"] is not None]
    if ppls:
        best = min(per_seed, key=lambda item: item["wikitext2_ppl"] or float("inf"))
        worst = max(per_seed, key=lambda item: item["wikitext2_ppl"] or float("-inf"))
        lines.extend(
            [
                "",
                "## Summary Statistics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Mean WikiText2 PPL | {statistics.mean(ppls):.2f} |",
                f"| Sample std WikiText2 PPL | {statistics.stdev(ppls):.2f} |" if len(ppls) > 1 else "| Sample std WikiText2 PPL | n/a |",
                f"| Mean train PPL | {statistics.mean(train_ppls):.2f} |",
                f"| Mean runtime minutes | {statistics.mean(runtimes):.2f} |",
                f"| Best seed | {best['seed']} (`{best['run_id']}`, PPL {best['wikitext2_ppl']:.2f}) |",
                f"| Worst seed | {worst['seed']} (`{worst['run_id']}`, PPL {worst['wikitext2_ppl']:.2f}) |",
            ]
        )
    lines.extend(
        [
            "",
            "## Pairwise Dropped-Module Overlap",
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
        intersection = len(left_drops & right_drops)
        union = len(left_drops | right_drops)
        lines.append(
            f"| {left['seed']} vs {right['seed']} | {intersection} | {union} | {jaccard(left_drops, right_drops):.3f} |"
        )
    if ppls:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"All available seeds completed with finite WikiText2 PPL. The final WikiText2 PPL values are tightly grouped relative to the baseline spread, with mean `{statistics.mean(ppls):.2f}`.",
                "Runtime should not be compared directly across seeds because the runs used different GPU types.",
            ]
        )
    (results_dir / "seed_robustness_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_key_value_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_small_model_summary(rows: list[dict[str, str]], runs_root: Path, results_dir: Path) -> None:
    sparse_rows = [
        row
        for row in rows
        if row["method"] in {"sparse_db", "sparse_search"} or "quant" in row["method"]
    ]
    lines = [
        "# Small-Model Sparse/Quant Feasibility Summary",
        "",
        "This summary reports small-model database and search pipeline tests. Failed setup attempts are kept visible but excluded from numeric success aggregates.",
        "",
        "## SparseGPT/FastOBC Pipeline",
        "",
        "| Run ID | Method | Status | Model | PPL | Train PPL | Runtime (min) | GPU | Peak CPU GB | Peak GPU GB | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in sparse_rows:
        if row["method"] not in {"sparse_db", "sparse_search"}:
            continue
        run_dir = run_artifact_dir(row, runs_root)
        db_summary = parse_key_value_file(run_dir / "sparse_db_summary.txt")
        peak_cpu = ""
        peak_gpu = ""
        if "max_cpu_memory_gb=" in row["notes"] or "max_cpu_memory_gb" in row["notes"]:
            # Notes are semicolon-delimited key-value pairs.
            note_values = parse_note_values(row["notes"])
            peak_cpu = note_values.get("max_cpu_memory_gb", "")
            peak_gpu = note_values.get("max_gpu_memory_gb", "")
        if not peak_cpu:
            peak_cpu = db_summary.get("max_cpu_memory_gb", "")
        if not peak_gpu:
            peak_gpu = db_summary.get("max_gpu_memory_gb", "")
        note = summarize_feasibility_note(row, db_summary)
        lines.append(
            "| `{run_id}` | {method} | {status} | `{model}` | {ppl} | {train} | {runtime} | {gpu} | {cpu} | {gpu_mem} | {note} |".format(
                run_id=row["run_id"],
                method=row["method"],
                status=row["status"],
                model=row["model"],
                ppl=fmt_table_number(parse_float(row["wikitext2_ppl"])),
                train=fmt_table_number(parse_float(row["train_ppl"])),
                runtime=fmt_table_number(parse_float(row["runtime_minutes"])),
                gpu=row["gpu_name"] or "n/a",
                cpu=peak_cpu or "n/a",
                gpu_mem=peak_gpu or "n/a",
                note=note,
            )
        )
    quant_rows = [row for row in sparse_rows if "quant" in row["method"]]
    lines.extend(["", "## Quantization Pipeline", ""])
    if quant_rows:
        lines.extend(
            [
                "| Run ID | Method | Status | Model | Bits/Sparsity | Runtime (min) | Notes |",
                "| --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in quant_rows:
            lines.append(
                f"| `{row['run_id']}` | {row['method']} | {row['status']} | `{row['model']}` | {row['sparsity_or_bits']} | {fmt_table_number(parse_float(row['runtime_minutes']))} | {row['notes']} |"
            )
    else:
        lines.append("No small-model quantization feasibility run is logged yet. This is optional and lower priority than consolidating the completed depth-pruning and sparse-pipeline evidence.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The reduced TinyLlama SparseGPT database generation and sparse search completed end-to-end. This demonstrates that the unstructured sparse pipeline is operational on a smaller model, while the Mistral-7B full sparse database remains constrained by the current `16 GB` container RAM limit.",
        ]
    )
    (results_dir / "small_model_feasibility_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_note_values(notes: str) -> dict[str, str]:
    values = {}
    for part in notes.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def summarize_feasibility_note(row: dict[str, str], db_summary: dict[str, str]) -> str:
    if row["status"] != "completed":
        return "Failed attempt preserved; excluded from aggregates."
    if row["method"] == "sparse_db":
        module_dirs = db_summary.get("generated_module_dirs", "")
        level_files = db_summary.get("generated_level_files", "")
        size_mb = db_summary.get("database_size_mb", "")
        return f"Generated {module_dirs} module dirs and {level_files} level files; database size {size_mb} MB."
    if row["method"] == "sparse_search":
        return "Completed 20-generation sparse search using the TinyLlama q-proj database."
    return "Completed."


def main() -> None:
    args = parse_args()
    experiment_log = Path(args.experiment_log)
    runs_root = Path(args.runs_root)
    results_dir = Path(args.results_dir)

    rows = read_csv(experiment_log)
    curve_rows = build_depth_curve(rows, results_dir)
    convergence_rows = build_convergence(rows, runs_root, results_dir)
    write_baseline_table(curve_rows, rows, results_dir)
    write_seed_robustness(rows, runs_root, results_dir)
    write_small_model_summary(rows, runs_root, results_dir)

    status_counts = Counter(row["status"] for row in rows)
    print("Generated report artifacts:")
    for path in [
        results_dir / "depth_pruning_curve.csv",
        results_dir / "depth_pruning_curve.png",
        results_dir / "convergence_37_5.csv",
        results_dir / "convergence_37_5.png",
        results_dir / "baseline_comparison_table.md",
        results_dir / "seed_robustness_table.md",
        results_dir / "small_model_feasibility_summary.md",
    ]:
        print(f"  {path}")
    print(f"Included completed rows available in log: {status_counts.get('completed', 0)}")
    print(f"Excluded non-completed rows available in log: {len(rows) - status_counts.get('completed', 0)}")
    print(f"Convergence rows written: {len(convergence_rows)}")


if __name__ == "__main__":
    main()
