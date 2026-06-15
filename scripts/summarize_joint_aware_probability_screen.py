#!/usr/bin/env python3
"""Summarize the matched TinyLlama joint-aware probability screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("baseline", "p025")
LABELS = {
    "baseline": "Unchanged joint search (p=0)",
    "p025": "Joint-aware search (p=0.25)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the TinyLlama p=0 versus p=0.25 screen."
    )
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values)


def discover(runs_root: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = defaultdict(dict)
    labels = {"baseline": "p0", "p025": "p025"}
    for variant, run_label in labels.items():
        for seed in (0, 1, 2):
            run_dir = (
                runs_root
                / f"screen_jointaware_tiny_{run_label}_g20_o8_seed{seed}"
            )
            required = (
                run_dir / "run_summary.json",
                run_dir / "generation_log.csv",
                run_dir / "final_candidate.json",
            )
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise SystemExit(
                    "Missing screening artifacts: "
                    + ", ".join(str(path) for path in missing)
                )
            discovered[variant][seed] = run_dir
    return discovered


def load_run(
    variant: str,
    seed: int,
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(run_dir / "run_summary.json")
    generations = read_csv(run_dir / "generation_log.csv")
    final = summary["final_metrics"]
    compression = summary["compression_config"]

    if int(summary["search_config"]["seed"]) != seed:
        raise SystemExit(f"Unexpected seed in {run_dir}.")
    if bool(compression["joint_aware_mutation"]) != (variant == "p025"):
        raise SystemExit(f"Unexpected mutation setting in {run_dir}.")
    if len(generations) != 20:
        raise SystemExit(f"Expected 20 generations in {run_dir}.")

    generated: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    accepted = 0
    convergence: list[dict[str, Any]] = []
    for generation in generations:
        mutation = json.loads(generation["mutation_summary"])
        generated.update(
            mutation.get(
                "generated_offspring_by_type",
                mutation.get("accepted_offspring_by_type", {}),
            )
        )
        selected_type = generation.get(
            "selected_parent_mutation_type", ""
        )
        if selected_type:
            selected[selected_type] += 1
        was_accepted = (
            generation["accepted_parent_replacement"].strip().lower()
            == "true"
        )
        accepted += was_accepted
        convergence.append(
            {
                "variant": variant,
                "seed": seed,
                "run_id": summary["run_name"],
                "generation": int(generation["generation"]),
                "best_search_fitness": float(
                    generation["best_search_fitness"]
                ),
                "wikitext2_ppl": (
                    float(generation["wikitext2_ppl"])
                    if generation["wikitext2_ppl"]
                    else ""
                ),
                "train_ppl": (
                    float(generation["best_train_ppl"])
                    if generation["best_train_ppl"]
                    else ""
                ),
                "selected_parent_mutation_type": selected_type,
                "accepted_parent_replacement": was_accepted,
                "phase": "generation",
            }
        )

    convergence.append(
        {
            "variant": variant,
            "seed": seed,
            "run_id": summary["run_name"],
            "generation": 20,
            "best_search_fitness": final["final_calibration_kl"],
            "wikitext2_ppl": final["wikitext2_ppl"],
            "train_ppl": final["train_ppl"],
            "selected_parent_mutation_type": "",
            "accepted_parent_replacement": "",
            "phase": "final",
        }
    )

    return (
        {
            "variant": variant,
            "seed": seed,
            "run_id": summary["run_name"],
            "wikitext2_ppl": final["wikitext2_ppl"],
            "train_ppl": final["train_ppl"],
            "final_calibration_kl": final["final_calibration_kl"],
            "runtime_seconds": final["runtime_seconds"],
            "estimated_compression_ratio": final[
                "estimated_compression_ratio"
            ],
            "average_bitwidth_active": final["average_bitwidth_active"],
            "accepted_generations": accepted,
            "generated_depth": generated["depth"],
            "generated_quantization": generated["quantization"],
            "generated_joint_aware": generated["joint_aware"],
            "selected_depth": selected["depth"],
            "selected_quantization": selected["quantization"],
            "selected_joint_aware": selected["joint_aware"],
            "retained_parent": selected["parent"],
        },
        convergence,
    )


def paired_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["variant"], int(row["seed"])): row for row in run_rows
    }
    output = []
    for seed in (0, 1, 2):
        baseline = by_key[("baseline", seed)]
        aware = by_key[("p025", seed)]
        output.append(
            {
                "seed": seed,
                "baseline_wikitext2_ppl": baseline["wikitext2_ppl"],
                "p025_wikitext2_ppl": aware["wikitext2_ppl"],
                "p025_minus_baseline_ppl": (
                    aware["wikitext2_ppl"] - baseline["wikitext2_ppl"]
                ),
                "baseline_final_calibration_kl": baseline[
                    "final_calibration_kl"
                ],
                "p025_final_calibration_kl": aware[
                    "final_calibration_kl"
                ],
                "p025_minus_baseline_kl": (
                    aware["final_calibration_kl"]
                    - baseline["final_calibration_kl"]
                ),
                "baseline_runtime_seconds": baseline["runtime_seconds"],
                "p025_runtime_seconds": aware["runtime_seconds"],
                "p025_minus_baseline_runtime_seconds": (
                    aware["runtime_seconds"] - baseline["runtime_seconds"]
                ),
            }
        )
    return output


def mutation_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        rows = [row for row in run_rows if row["variant"] == variant]
        for mutation_type in ("depth", "quantization", "joint_aware"):
            generated = sum(
                int(row[f"generated_{mutation_type}"]) for row in rows
            )
            selected = sum(
                int(row[f"selected_{mutation_type}"]) for row in rows
            )
            output.append(
                {
                    "variant": variant,
                    "mutation_type": mutation_type,
                    "generated_offspring": generated,
                    "selected_as_parent": selected,
                    "selection_rate": (
                        selected / generated if generated else ""
                    ),
                }
            )
    return output


def build_markdown(
    run_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> str:
    baseline_ppl = [
        float(row["wikitext2_ppl"])
        for row in run_rows
        if row["variant"] == "baseline"
    ]
    aware_ppl = [
        float(row["wikitext2_ppl"])
        for row in run_rows
        if row["variant"] == "p025"
    ]
    ppl_deltas = [
        float(row["p025_minus_baseline_ppl"]) for row in pairs
    ]
    baseline_kl = [
        float(row["final_calibration_kl"])
        for row in run_rows
        if row["variant"] == "baseline"
    ]
    aware_kl = [
        float(row["final_calibration_kl"])
        for row in run_rows
        if row["variant"] == "p025"
    ]
    kl_deltas = [
        float(row["p025_minus_baseline_kl"]) for row in pairs
    ]
    wins = sum(delta < 0 for delta in ppl_deltas)
    aware_mutation = next(
        row
        for row in mutations
        if row["variant"] == "p025"
        and row["mutation_type"] == "joint_aware"
    )
    aware_rows = [row for row in run_rows if row["variant"] == "p025"]
    accepted_total = sum(int(row["accepted_generations"]) for row in aware_rows)

    lines = [
        "# TinyLlama Joint-Aware Probability Screen",
        "",
        "This matched screen compares the unchanged joint depth-plus-quantization search (`p=0`) with joint-aware mutation probability `0.25`. Both variants use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2.",
        "",
        "## Result",
        "",
        "| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| {LABELS['baseline']} | 3 | {mean(baseline_ppl):.3f} +/- {sample_std(baseline_ppl):.3f} | {mean(baseline_kl):.4f} +/- {sample_std(baseline_kl):.4f} | {mean([float(row['runtime_seconds']) for row in run_rows if row['variant'] == 'baseline']):.1f} s |",
        f"| {LABELS['p025']} | 3 | {mean(aware_ppl):.3f} +/- {sample_std(aware_ppl):.3f} | {mean(aware_kl):.4f} +/- {sample_std(aware_kl):.4f} | {mean([float(row['runtime_seconds']) for row in run_rows if row['variant'] == 'p025']):.1f} s |",
        "",
        f"The paired mean difference `p=0.25 - baseline` is {mean(ppl_deltas):+.3f} PPL with sample SD {sample_std(ppl_deltas):.3f}. The joint-aware variant wins {wins} of 3 seeds, but its mean PPL and seed variance are both worse. Mean final calibration KL changes by {mean(kl_deltas):+.4f}.",
        "",
        "## Paired seeds",
        "",
        "| Seed | Baseline PPL | p=0.25 PPL | PPL delta | Baseline KL | p=0.25 KL | KL delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['baseline_wikitext2_ppl']:.3f} | "
            f"{row['p025_wikitext2_ppl']:.3f} | "
            f"{row['p025_minus_baseline_ppl']:+.3f} | "
            f"{row['baseline_final_calibration_kl']:.4f} | "
            f"{row['p025_final_calibration_kl']:.4f} | "
            f"{row['p025_minus_baseline_kl']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mutation selection",
            "",
            "| Variant | Mutation | Generated | Selected as parent | Selection rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in mutations:
        rate = (
            f"{float(row['selection_rate']) * 100:.1f}%"
            if row["selection_rate"] != ""
            else "n/a"
        )
        lines.append(
            f"| {LABELS[row['variant']]} | {row['mutation_type']} | "
            f"{row['generated_offspring']} | {row['selected_as_parent']} | "
            f"{rate} |"
        )

    lines.extend(
        [
            "",
            f"The `p=0.25` runs generated {aware_mutation['generated_offspring']} joint-aware offspring. Only {aware_mutation['selected_as_parent']} became the selected parent, a {float(aware_mutation['selection_rate']) * 100:.1f}% proposal-level selection rate and {aware_mutation['selected_as_parent']}/{accepted_total} accepted replacements. This provenance records the immediate winning mutation type; it does not reconstruct the full ancestry of later candidates.",
            "",
            "## Decision",
            "",
            "Do not promote `p=0.25` directly to a three-seed Mistral G50 experiment. The screen is mixed rather than consistently positive: it wins two seeds, loses seed 0 by 0.727 PPL, slightly worsens the mean, and increases variance. Combined with the negative Mistral `p=0.5` ablation, there is not enough evidence that probability tuning alone improves the operator.",
            "",
            "Keep the unchanged G50 Mistral search as the primary method. The next algorithmic iteration should improve the coupled move itself or use adaptive scheduling, and should first pass this same inexpensive TinyLlama screen before Mistral promotion.",
            "",
            "## Generated artifacts",
            "",
            "- `results/joint_aware_probability_screen.csv`: paired seed-level metrics.",
            "- `results/joint_aware_probability_screen_mutations.csv`: generated and selected mutation counts.",
            "- `results/joint_aware_probability_screen_convergence.csv`: generation-wise metrics and selected mutation provenance.",
            "- `results/joint_aware_probability_screen.png`: paired final PPL and mean convergence.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_plot(
    output_dir: Path,
    pairs: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#8c1d40", "p025": "#17becf"}
    fig, (paired_ax, convergence_ax) = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )

    for row in pairs:
        paired_ax.plot(
            [0, 1],
            [
                row["baseline_wikitext2_ppl"],
                row["p025_wikitext2_ppl"],
            ],
            marker="o",
            alpha=0.75,
            label=f"Seed {row['seed']}",
        )
    paired_ax.set_xticks([0, 1], ["p=0", "p=0.25"])
    paired_ax.set_ylabel("Final WikiText2 perplexity")
    paired_ax.set_title("Paired Final Results")
    paired_ax.grid(alpha=0.25)
    paired_ax.legend()

    for variant in VARIANTS:
        by_generation: dict[int, list[float]] = defaultdict(list)
        for row in convergence:
            if (
                row["variant"] == variant
                and row["phase"] == "generation"
                and row["wikitext2_ppl"] != ""
            ):
                by_generation[int(row["generation"])].append(
                    float(row["wikitext2_ppl"])
                )
        generations = sorted(by_generation)
        convergence_ax.plot(
            generations,
            [mean(by_generation[generation]) for generation in generations],
            marker="o",
            color=colors[variant],
            label=LABELS[variant],
        )
    convergence_ax.set_xlabel("Generation")
    convergence_ax.set_ylabel("Mean WikiText2 perplexity")
    convergence_ax.set_title("Periodic Evaluation")
    convergence_ax.grid(alpha=0.25)
    convergence_ax.legend()

    fig.suptitle("TinyLlama Joint-Aware Probability Screen")
    fig.tight_layout()
    fig.savefig(
        output_dir / "joint_aware_probability_screen.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    discovered = discover(runs_root)

    run_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in (0, 1, 2):
            row, generation_rows = load_run(
                variant,
                seed,
                discovered[variant][seed],
            )
            run_rows.append(row)
            convergence.extend(generation_rows)

    pairs = paired_rows(run_rows)
    mutations = mutation_rows(run_rows)
    write_csv(
        output_dir / "joint_aware_probability_screen.csv",
        (
            "seed",
            "baseline_wikitext2_ppl",
            "p025_wikitext2_ppl",
            "p025_minus_baseline_ppl",
            "baseline_final_calibration_kl",
            "p025_final_calibration_kl",
            "p025_minus_baseline_kl",
            "baseline_runtime_seconds",
            "p025_runtime_seconds",
            "p025_minus_baseline_runtime_seconds",
        ),
        pairs,
    )
    write_csv(
        output_dir / "joint_aware_probability_screen_mutations.csv",
        (
            "variant",
            "mutation_type",
            "generated_offspring",
            "selected_as_parent",
            "selection_rate",
        ),
        mutations,
    )
    write_csv(
        output_dir / "joint_aware_probability_screen_convergence.csv",
        (
            "variant",
            "seed",
            "run_id",
            "phase",
            "generation",
            "best_search_fitness",
            "wikitext2_ppl",
            "train_ppl",
            "selected_parent_mutation_type",
            "accepted_parent_replacement",
        ),
        convergence,
    )
    (
        output_dir / "joint_aware_probability_screen.md"
    ).write_text(
        build_markdown(run_rows, pairs, mutations),
        encoding="utf-8",
    )
    if not args.skip_plots:
        create_plot(output_dir, pairs, convergence)

    print(f"Wrote joint-aware screening summary to {output_dir}")


if __name__ == "__main__":
    main()
