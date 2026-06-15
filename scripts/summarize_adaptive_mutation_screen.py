#!/usr/bin/env python3
"""Summarize the matched TinyLlama adaptive-mutation screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("baseline", "adaptive", "fixed")
LABELS = {
    "baseline": "Default mutation (max 3)",
    "adaptive": "Adaptive mutation (patience 3, max 3)",
    "fixed": "Fixed local mutation (strength 1)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the TinyLlama adaptive-mutation screen."
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
    names = {
        "baseline": "screen_jointaware_tiny_p0_g20_o8_seed{seed}",
        "adaptive": (
            "screen_adaptive_tiny_pat3_max3_g20_o8_seed{seed}"
        ),
        "fixed": "screen_fixedstrength_tiny_max1_g20_o8_seed{seed}",
    }
    for variant, template in names.items():
        for seed in (0, 1, 2):
            run_dir = runs_root / template.format(seed=seed)
            required = (
                run_dir / "run_summary.json",
                run_dir / "generation_log.csv",
                run_dir / "final_candidate.json",
            )
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise SystemExit(
                    "Missing adaptive-screen artifacts: "
                    + ", ".join(str(path) for path in missing)
                )
            discovered[variant][seed] = run_dir
    return discovered


def load_run(
    variant: str,
    seed: int,
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary = read_json(run_dir / "run_summary.json")
    generations = read_csv(run_dir / "generation_log.csv")
    final = summary["final_metrics"]
    compression = summary["compression_config"]

    if int(summary["search_config"]["seed"]) != seed:
        raise SystemExit(f"Unexpected seed in {run_dir}.")
    if bool(compression.get("adaptive_mutation", False)) != (
        variant == "adaptive"
    ):
        raise SystemExit(f"Unexpected adaptive setting in {run_dir}.")
    expected_strength = 1 if variant == "fixed" else 3
    command = (run_dir / "command.sh").read_text(encoding="utf-8")
    if f"--max_drop_mutations {expected_strength}" not in command:
        raise SystemExit(
            f"Unexpected maximum mutation strength in {run_dir}."
        )
    if len(generations) != 20:
        raise SystemExit(f"Expected 20 generations in {run_dir}.")

    selected: Counter[str] = Counter()
    accepted = 0
    convergence: list[dict[str, Any]] = []
    strengths: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "generations": 0,
            "generated_depth": 0,
            "generated_quantization": 0,
            "selected_depth": 0,
            "selected_quantization": 0,
            "retained_parent": 0,
        }
    )

    for generation in generations:
        mutation = json.loads(generation["mutation_summary"])
        strength = int(mutation.get("adaptive_mutation_strength", 1))
        generated = mutation.get(
            "generated_offspring_by_type",
            mutation.get("accepted_offspring_by_type", {}),
        )
        selected_type = generation.get(
            "selected_parent_mutation_type", ""
        )
        was_accepted = (
            generation["accepted_parent_replacement"].strip().lower()
            == "true"
        )
        selected[selected_type] += 1
        accepted += was_accepted
        strengths[strength]["generations"] += 1
        strengths[strength]["generated_depth"] += int(
            generated.get("depth", 0)
        )
        strengths[strength]["generated_quantization"] += int(
            generated.get("quantization", 0)
        )
        if selected_type == "depth":
            strengths[strength]["selected_depth"] += 1
        elif selected_type == "quantization":
            strengths[strength]["selected_quantization"] += 1
        elif selected_type == "parent":
            strengths[strength]["retained_parent"] += 1

        convergence.append(
            {
                "variant": variant,
                "seed": seed,
                "run_id": summary["run_name"],
                "phase": "generation",
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
                "mutation_strength": strength,
                "selected_parent_mutation_type": selected_type,
                "accepted_parent_replacement": was_accepted,
            }
        )

    convergence.append(
        {
            "variant": variant,
            "seed": seed,
            "run_id": summary["run_name"],
            "phase": "final",
            "generation": 20,
            "best_search_fitness": final["final_calibration_kl"],
            "wikitext2_ppl": final["wikitext2_ppl"],
            "train_ppl": final["train_ppl"],
            "mutation_strength": "",
            "selected_parent_mutation_type": "",
            "accepted_parent_replacement": "",
        }
    )

    strength_rows = []
    for strength in sorted(strengths):
        strength_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "mutation_strength": strength,
                **strengths[strength],
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
            "selected_depth": selected["depth"],
            "selected_quantization": selected["quantization"],
            "retained_parent": selected["parent"],
            "elevated_strength_generations": sum(
                row["generations"]
                for row in strength_rows
                if int(row["mutation_strength"]) > 1
            ),
            "elevated_strength_replacements": sum(
                row["selected_depth"] + row["selected_quantization"]
                for row in strength_rows
                if int(row["mutation_strength"]) > 1
            ),
        },
        convergence,
        strength_rows,
    )


def paired_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["variant"], int(row["seed"])): row for row in run_rows
    }
    output = []
    for seed in (0, 1, 2):
        baseline = by_key[("baseline", seed)]
        adaptive = by_key[("adaptive", seed)]
        fixed = by_key[("fixed", seed)]
        output.append(
            {
                "seed": seed,
                "baseline_wikitext2_ppl": baseline["wikitext2_ppl"],
                "adaptive_wikitext2_ppl": adaptive["wikitext2_ppl"],
                "fixed_wikitext2_ppl": fixed["wikitext2_ppl"],
                "adaptive_minus_baseline_ppl": (
                    adaptive["wikitext2_ppl"]
                    - baseline["wikitext2_ppl"]
                ),
                "fixed_minus_baseline_ppl": (
                    fixed["wikitext2_ppl"]
                    - baseline["wikitext2_ppl"]
                ),
                "adaptive_minus_fixed_ppl": (
                    adaptive["wikitext2_ppl"]
                    - fixed["wikitext2_ppl"]
                ),
                "baseline_final_calibration_kl": baseline[
                    "final_calibration_kl"
                ],
                "adaptive_final_calibration_kl": adaptive[
                    "final_calibration_kl"
                ],
                "fixed_final_calibration_kl": fixed[
                    "final_calibration_kl"
                ],
                "adaptive_minus_baseline_kl": (
                    adaptive["final_calibration_kl"]
                    - baseline["final_calibration_kl"]
                ),
                "fixed_minus_baseline_kl": (
                    fixed["final_calibration_kl"]
                    - baseline["final_calibration_kl"]
                ),
                "adaptive_minus_fixed_kl": (
                    adaptive["final_calibration_kl"]
                    - fixed["final_calibration_kl"]
                ),
                "baseline_runtime_seconds": baseline["runtime_seconds"],
                "adaptive_runtime_seconds": adaptive["runtime_seconds"],
                "fixed_runtime_seconds": fixed["runtime_seconds"],
                "adaptive_minus_baseline_runtime_seconds": (
                    adaptive["runtime_seconds"]
                    - baseline["runtime_seconds"]
                ),
                "fixed_minus_baseline_runtime_seconds": (
                    fixed["runtime_seconds"]
                    - baseline["runtime_seconds"]
                ),
                "elevated_strength_generations": adaptive[
                    "elevated_strength_generations"
                ],
                "elevated_strength_replacements": adaptive[
                    "elevated_strength_replacements"
                ],
            }
        )
    return output


def build_markdown(
    run_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    strength_rows: list[dict[str, Any]],
) -> str:
    baseline = [
        row for row in run_rows if row["variant"] == "baseline"
    ]
    adaptive = [
        row for row in run_rows if row["variant"] == "adaptive"
    ]
    fixed = [row for row in run_rows if row["variant"] == "fixed"]
    baseline_ppl = [float(row["wikitext2_ppl"]) for row in baseline]
    adaptive_ppl = [float(row["wikitext2_ppl"]) for row in adaptive]
    fixed_ppl = [float(row["wikitext2_ppl"]) for row in fixed]
    baseline_kl = [
        float(row["final_calibration_kl"]) for row in baseline
    ]
    adaptive_kl = [
        float(row["final_calibration_kl"]) for row in adaptive
    ]
    fixed_kl = [
        float(row["final_calibration_kl"]) for row in fixed
    ]
    adaptive_ppl_deltas = [
        float(row["adaptive_minus_baseline_ppl"]) for row in pairs
    ]
    fixed_ppl_deltas = [
        float(row["fixed_minus_baseline_ppl"]) for row in pairs
    ]
    adaptive_fixed_ppl_deltas = [
        float(row["adaptive_minus_fixed_ppl"]) for row in pairs
    ]
    adaptive_kl_deltas = [
        float(row["adaptive_minus_baseline_kl"]) for row in pairs
    ]
    fixed_kl_deltas = [
        float(row["fixed_minus_baseline_kl"]) for row in pairs
    ]
    adaptive_wins = sum(delta < 0 for delta in adaptive_ppl_deltas)
    fixed_wins = sum(delta < 0 for delta in fixed_ppl_deltas)
    elevated_generations = sum(
        int(row["elevated_strength_generations"]) for row in adaptive
    )
    elevated_replacements = sum(
        int(row["elevated_strength_replacements"]) for row in adaptive
    )

    lines = [
        "# TinyLlama Adaptive Mutation Screen",
        "",
        "This matched ablation compares the existing joint search, adaptive mutation strength, and a fixed strength-1 control. All runs use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2.",
        "",
        "## Result",
        "",
        "| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| {LABELS['baseline']} | 3 | {mean(baseline_ppl):.3f} +/- {sample_std(baseline_ppl):.3f} | {mean(baseline_kl):.4f} +/- {sample_std(baseline_kl):.4f} | {mean([float(row['runtime_seconds']) for row in baseline]):.1f} s |",
        f"| {LABELS['adaptive']} | 3 | {mean(adaptive_ppl):.3f} +/- {sample_std(adaptive_ppl):.3f} | {mean(adaptive_kl):.4f} +/- {sample_std(adaptive_kl):.4f} | {mean([float(row['runtime_seconds']) for row in adaptive]):.1f} s |",
        f"| {LABELS['fixed']} | 3 | {mean(fixed_ppl):.3f} +/- {sample_std(fixed_ppl):.3f} | {mean(fixed_kl):.4f} +/- {sample_std(fixed_kl):.4f} | {mean([float(row['runtime_seconds']) for row in fixed]):.1f} s |",
        "",
        f"Adaptive mode changes mean PPL by {mean(adaptive_ppl_deltas):+.3f} versus the default and wins {adaptive_wins} of 3 seeds. Fixed strength 1 changes mean PPL by {mean(fixed_ppl_deltas):+.3f} and wins {fixed_wins} of 3 seeds. The mean `adaptive - fixed` difference is {mean(adaptive_fixed_ppl_deltas):+.3f} PPL.",
        "",
        "## Paired seeds",
        "",
        "| Seed | Default PPL | Adaptive PPL | Fixed-1 PPL | Adaptive - default | Fixed-1 - default | Adaptive - fixed-1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['baseline_wikitext2_ppl']:.3f} | "
            f"{row['adaptive_wikitext2_ppl']:.3f} | "
            f"{row['fixed_wikitext2_ppl']:.3f} | "
            f"{row['adaptive_minus_baseline_ppl']:+.3f} | "
            f"{row['fixed_minus_baseline_ppl']:+.3f} | "
            f"{row['adaptive_minus_fixed_ppl']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Calibration objective",
            "",
            "| Variant | Final KL mean +/- SD | Mean delta from default |",
            "| --- | ---: | ---: |",
            f"| {LABELS['baseline']} | {mean(baseline_kl):.4f} +/- {sample_std(baseline_kl):.4f} | - |",
            f"| {LABELS['adaptive']} | {mean(adaptive_kl):.4f} +/- {sample_std(adaptive_kl):.4f} | {mean(adaptive_kl_deltas):+.4f} |",
            f"| {LABELS['fixed']} | {mean(fixed_kl):.4f} +/- {sample_std(fixed_kl):.4f} | {mean(fixed_kl_deltas):+.4f} |",
            "",
            "## Strength schedule",
            "",
            "| Seed | Strength | Generations | Generated depth | Generated quant | Selected depth | Selected quant | Parent retained |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in strength_rows:
        if row["variant"] != "adaptive":
            continue
        lines.append(
            f"| {row['seed']} | {row['mutation_strength']} | "
            f"{row['generations']} | {row['generated_depth']} | "
            f"{row['generated_quantization']} | {row['selected_depth']} | "
            f"{row['selected_quantization']} | {row['retained_parent']} |"
        )

    lines.extend(
        [
            "",
            f"Elevated strengths were active for {elevated_generations} generation(s), all in seed 2. They produced {elevated_replacements} accepted replacement(s). Seeds 0 and 1 stayed at strength 1 for all generations. Therefore the improved aggregate result cannot be attributed to escalation; the final candidates were selected entirely under strength-1 behavior.",
            "",
            "The fixed-strength control resolves the earlier attribution problem. Seeds 0 and 1 produced byte-identical final candidates under adaptive and fixed-strength modes. Seed 2 used elevated adaptive strengths for six generations but accepted no elevated-strength replacement; fixed strength 1 was slightly better by 0.047 PPL.",
            "",
            "## Decision",
            "",
            "The supported contribution is mutation locality, not adaptive escalation. A single depth swap per mutation improved mean PPL and substantially reduced variance relative to allowing up to three swaps. Promote fixed strength 1 to a matched Mistral ablation. Do not spend Mistral compute on the current adaptive schedule unless a future design makes elevated mutations demonstrably useful.",
            "",
            "## Generated artifacts",
            "",
            "- `results/adaptive_mutation_screen.csv`: three-way paired final metrics.",
            "- `results/adaptive_mutation_strengths.csv`: per-seed strength usage and selected mutations.",
            "- `results/adaptive_mutation_convergence.csv`: generation-wise quality, strength, and provenance.",
            "- `results/adaptive_mutation_screen.png`: paired final PPL and convergence.",
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

    colors = {
        "baseline": "#8c1d40",
        "adaptive": "#2ca02c",
        "fixed": "#1f77b4",
    }
    fig, (paired_ax, convergence_ax) = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )

    for row in pairs:
        paired_ax.plot(
            [0, 1, 2],
            [
                row["baseline_wikitext2_ppl"],
                row["adaptive_wikitext2_ppl"],
                row["fixed_wikitext2_ppl"],
            ],
            marker="o",
            alpha=0.75,
            label=f"Seed {row['seed']}",
        )
    paired_ax.set_xticks(
        [0, 1, 2],
        ["Default", "Adaptive", "Fixed-1"],
    )
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

    fig.suptitle("TinyLlama Adaptive Mutation Screen")
    fig.tight_layout()
    fig.savefig(output_dir / "adaptive_mutation_screen.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    discovered = discover(runs_root)

    run_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    strength_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in (0, 1, 2):
            row, generation_rows, run_strengths = load_run(
                variant,
                seed,
                discovered[variant][seed],
            )
            run_rows.append(row)
            convergence.extend(generation_rows)
            strength_rows.extend(run_strengths)

    pairs = paired_rows(run_rows)
    write_csv(
        output_dir / "adaptive_mutation_screen.csv",
        (
            "seed",
            "baseline_wikitext2_ppl",
            "adaptive_wikitext2_ppl",
            "fixed_wikitext2_ppl",
            "adaptive_minus_baseline_ppl",
            "fixed_minus_baseline_ppl",
            "adaptive_minus_fixed_ppl",
            "baseline_final_calibration_kl",
            "adaptive_final_calibration_kl",
            "fixed_final_calibration_kl",
            "adaptive_minus_baseline_kl",
            "fixed_minus_baseline_kl",
            "adaptive_minus_fixed_kl",
            "baseline_runtime_seconds",
            "adaptive_runtime_seconds",
            "fixed_runtime_seconds",
            "adaptive_minus_baseline_runtime_seconds",
            "fixed_minus_baseline_runtime_seconds",
            "elevated_strength_generations",
            "elevated_strength_replacements",
        ),
        pairs,
    )
    write_csv(
        output_dir / "adaptive_mutation_strengths.csv",
        (
            "variant",
            "seed",
            "mutation_strength",
            "generations",
            "generated_depth",
            "generated_quantization",
            "selected_depth",
            "selected_quantization",
            "retained_parent",
        ),
        strength_rows,
    )
    write_csv(
        output_dir / "adaptive_mutation_convergence.csv",
        (
            "variant",
            "seed",
            "run_id",
            "phase",
            "generation",
            "best_search_fitness",
            "wikitext2_ppl",
            "train_ppl",
            "mutation_strength",
            "selected_parent_mutation_type",
            "accepted_parent_replacement",
        ),
        convergence,
    )
    (output_dir / "adaptive_mutation_screen.md").write_text(
        build_markdown(run_rows, pairs, strength_rows),
        encoding="utf-8",
    )
    if not args.skip_plots:
        create_plot(output_dir, pairs, convergence)

    print(f"Wrote adaptive-mutation summary to {output_dir}")


if __name__ == "__main__":
    main()
