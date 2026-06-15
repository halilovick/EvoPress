#!/usr/bin/env python3
"""Summarize the matched Mistral max-3 versus fixed-1 mutation ablation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("max3", "fixed1")
LABELS = {
    "max3": "Default mutation (max 3)",
    "fixed1": "Fixed local mutation (strength 1)",
}
RUN_TEMPLATES = {
    "max3": (
        "thesis_compute_matched_joint_mistral_"
        "s0.25_qproj3.0_g50_o16_seed{seed}"
    ),
    "fixed1": (
        "thesis_fixedstrength_joint_mistral_"
        "s0.25_qproj3.0_g50_o16_seed{seed}"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the Mistral fixed mutation-strength ablation."
    )
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument(
        "--experiment-log",
        default="results/experiment_log.csv",
    )
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


def command_arguments(path: Path) -> list[str]:
    exec_line = next(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("exec ")
    )
    return shlex.split(exec_line)[1:]


def command_signature(path: Path) -> tuple[list[str], int]:
    arguments = command_arguments(path)
    normalized: list[str] = []
    mutation_strength = -1
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--max_drop_mutations":
            mutation_strength = int(arguments[index + 1])
            index += 2
            continue
        if argument == "--output_dir":
            index += 2
            continue
        if argument == "--seed":
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return normalized, mutation_strength


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def discover(runs_root: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = defaultdict(dict)
    for variant, template in RUN_TEMPLATES.items():
        for seed in (0, 1, 2):
            run_dir = runs_root / template.format(seed=seed)
            required = (
                run_dir / "run_summary.json",
                run_dir / "generation_log.csv",
                run_dir / "final_candidate.json",
                run_dir / "command.sh",
            )
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise SystemExit(
                    "Missing Mistral mutation-ablation artifacts: "
                    + ", ".join(str(path) for path in missing)
                )
            discovered[variant][seed] = run_dir
    return discovered


def hardware_by_run(experiment_log: Path) -> dict[str, str]:
    if not experiment_log.is_file():
        return {}
    return {
        row["run_id"]: row["gpu_name"]
        for row in read_csv(experiment_log)
        if row.get("run_id")
    }


def load_run(
    variant: str,
    seed: int,
    run_dir: Path,
    hardware: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(run_dir / "run_summary.json")
    generations = read_csv(run_dir / "generation_log.csv")
    candidate = read_json(run_dir / "final_candidate.json")
    final = summary["final_metrics"]
    search = summary["search_config"]
    compression = summary["compression_config"]
    _, mutation_strength = command_signature(run_dir / "command.sh")
    expected_strength = 3 if variant == "max3" else 1

    if int(search["seed"]) != seed:
        raise SystemExit(f"Unexpected seed in {run_dir}.")
    if mutation_strength != expected_strength:
        raise SystemExit(f"Unexpected mutation strength in {run_dir}.")
    if int(search["generations"]) != 50 or int(search["offspring"]) != 16:
        raise SystemExit(f"Unexpected search budget in {run_dir}.")
    if len(generations) != 50:
        raise SystemExit(f"Expected 50 generations in {run_dir}.")
    if float(compression["target_depth_sparsity"]) != 0.25:
        raise SystemExit(f"Unexpected depth target in {run_dir}.")
    if float(compression["target_average_bitwidth"]) != 3.0:
        raise SystemExit(f"Unexpected bitwidth target in {run_dir}.")

    accepted = sum(
        row.get("accepted_parent_replacement", "").lower() == "true"
        for row in generations
    )
    convergence: list[dict[str, Any]] = []
    for row in generations:
        convergence.append(
            {
                "variant": variant,
                "seed": seed,
                "run_id": summary["run_name"],
                "phase": "generation",
                "generation": int(row["generation"]),
                "best_search_fitness": float(row["best_search_fitness"]),
                "wikitext2_ppl": (
                    float(row["wikitext2_ppl"])
                    if row["wikitext2_ppl"]
                    else ""
                ),
                "train_ppl": (
                    float(row["best_train_ppl"])
                    if row["best_train_ppl"]
                    else ""
                ),
                "accepted_parent_replacement": row.get(
                    "accepted_parent_replacement", ""
                ),
            }
        )
    convergence.append(
        {
            "variant": variant,
            "seed": seed,
            "run_id": summary["run_name"],
            "phase": "final",
            "generation": 50,
            "best_search_fitness": final["final_calibration_kl"],
            "wikitext2_ppl": final["wikitext2_ppl"],
            "train_ppl": final["train_ppl"],
            "accepted_parent_replacement": "",
        }
    )

    return (
        {
            "variant": variant,
            "seed": seed,
            "run_id": summary["run_name"],
            "mutation_strength": mutation_strength,
            "gpu_name": hardware.get(summary["run_name"], ""),
            "wikitext2_ppl": final["wikitext2_ppl"],
            "train_ppl": final["train_ppl"],
            "final_calibration_kl": final["final_calibration_kl"],
            "runtime_seconds": final["runtime_seconds"],
            "estimated_compression_ratio": final[
                "estimated_compression_ratio"
            ],
            "average_bitwidth_active": final["average_bitwidth_active"],
            "accepted_generations": accepted,
        },
        convergence,
        candidate,
    )


def verify_matched_commands(
    discovered: dict[str, dict[int, Path]],
) -> None:
    for seed in (0, 1, 2):
        baseline, baseline_strength = command_signature(
            discovered["max3"][seed] / "command.sh"
        )
        fixed, fixed_strength = command_signature(
            discovered["fixed1"][seed] / "command.sh"
        )
        if baseline != fixed:
            raise SystemExit(
                f"Search commands differ beyond mutation strength for seed {seed}."
            )
        if (baseline_strength, fixed_strength) != (3, 1):
            raise SystemExit(f"Unexpected mutation strengths for seed {seed}.")


def paired_rows(
    run_rows: list[dict[str, Any]],
    candidates: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["variant"], int(row["seed"])): row for row in run_rows
    }
    output = []
    for seed in (0, 1, 2):
        baseline = by_key[("max3", seed)]
        fixed = by_key[("fixed1", seed)]
        baseline_candidate = candidates["max3"][seed]
        fixed_candidate = candidates["fixed1"][seed]
        baseline_attn = {
            index
            for index, dropped in enumerate(
                baseline_candidate["attention_mask"]
            )
            if dropped
        }
        fixed_attn = {
            index
            for index, dropped in enumerate(
                fixed_candidate["attention_mask"]
            )
            if dropped
        }
        baseline_mlp = {
            index
            for index, dropped in enumerate(baseline_candidate["mlp_mask"])
            if dropped
        }
        fixed_mlp = {
            index
            for index, dropped in enumerate(fixed_candidate["mlp_mask"])
            if dropped
        }
        baseline_bits = baseline_candidate["bitwidth_by_module"]
        fixed_bits = fixed_candidate["bitwidth_by_module"]
        output.append(
            {
                "seed": seed,
                "max3_wikitext2_ppl": baseline["wikitext2_ppl"],
                "fixed1_wikitext2_ppl": fixed["wikitext2_ppl"],
                "fixed1_minus_max3_ppl": (
                    fixed["wikitext2_ppl"] - baseline["wikitext2_ppl"]
                ),
                "max3_final_calibration_kl": baseline[
                    "final_calibration_kl"
                ],
                "fixed1_final_calibration_kl": fixed[
                    "final_calibration_kl"
                ],
                "fixed1_minus_max3_kl": (
                    fixed["final_calibration_kl"]
                    - baseline["final_calibration_kl"]
                ),
                "max3_runtime_seconds": baseline["runtime_seconds"],
                "fixed1_runtime_seconds": fixed["runtime_seconds"],
                "attention_mask_jaccard": jaccard(
                    baseline_attn, fixed_attn
                ),
                "mlp_mask_jaccard": jaccard(baseline_mlp, fixed_mlp),
                "equal_quant_assignments": sum(
                    baseline_bits[module] == fixed_bits[module]
                    for module in baseline_bits
                ),
                "quant_modules": len(baseline_bits),
                "quant_bit_l1_distance": sum(
                    abs(baseline_bits[module] - fixed_bits[module])
                    for module in baseline_bits
                ),
            }
        )
    return output


def build_markdown(
    run_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> str:
    by_variant = {
        variant: [row for row in run_rows if row["variant"] == variant]
        for variant in VARIANTS
    }
    ppl = {
        variant: [float(row["wikitext2_ppl"]) for row in rows]
        for variant, rows in by_variant.items()
    }
    kl = {
        variant: [float(row["final_calibration_kl"]) for row in rows]
        for variant, rows in by_variant.items()
    }
    deltas = [float(row["fixed1_minus_max3_ppl"]) for row in pairs]
    kl_deltas = [float(row["fixed1_minus_max3_kl"]) for row in pairs]
    wins = sum(delta < 0 for delta in deltas)
    max3_gpu = sorted(
        {row["gpu_name"] for row in by_variant["max3"] if row["gpu_name"]}
    )
    fixed_gpu = sorted(
        {row["gpu_name"] for row in by_variant["fixed1"] if row["gpu_name"]}
    )

    lines = [
        "# Mistral-7B Fixed Mutation-Strength Ablation",
        "",
        "This matched experiment tests whether the fixed strength-1 depth mutation that improved the TinyLlama joint search transfers to Mistral-7B. Both variants use 25% depth sparsity, an active 3-bit `q_proj` budget, 50 generations, 16 offspring, 32 initial candidates, 8192 WikiText2 calibration tokens, sequence length 1024, and seeds 0-2. Command validation confirms that mutation strength is the only search parameter changed.",
        "",
        "## Result",
        "",
        "| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD |",
        "| --- | ---: | ---: | ---: |",
        f"| {LABELS['max3']} | 3 | {statistics.mean(ppl['max3']):.3f} +/- {statistics.stdev(ppl['max3']):.3f} | {statistics.mean(kl['max3']):.4f} +/- {statistics.stdev(kl['max3']):.4f} |",
        f"| {LABELS['fixed1']} | 3 | {statistics.mean(ppl['fixed1']):.3f} +/- {statistics.stdev(ppl['fixed1']):.3f} | {statistics.mean(kl['fixed1']):.4f} +/- {statistics.stdev(kl['fixed1']):.4f} |",
        "",
        f"The paired mean difference `fixed-1 - max-3` is {statistics.mean(deltas):+.3f} PPL with sample SD {statistics.stdev(deltas):.3f}. Fixed strength 1 wins {wins} of 3 seeds. Mean final calibration KL changes by {statistics.mean(kl_deltas):+.4f}, which is effectively neutral.",
        "",
        "## Paired seeds",
        "",
        "| Seed | Max-3 PPL | Fixed-1 PPL | PPL delta | Max-3 KL | Fixed-1 KL | KL delta | Attention Jaccard | MLP Jaccard | Equal quant bits |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['max3_wikitext2_ppl']:.3f} | "
            f"{row['fixed1_wikitext2_ppl']:.3f} | "
            f"{row['fixed1_minus_max3_ppl']:+.3f} | "
            f"{row['max3_final_calibration_kl']:.4f} | "
            f"{row['fixed1_final_calibration_kl']:.4f} | "
            f"{row['fixed1_minus_max3_kl']:+.4f} | "
            f"{row['attention_mask_jaccard']:.3f} | "
            f"{row['mlp_mask_jaccard']:.3f} | "
            f"{row['equal_quant_assignments']}/{row['quant_modules']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The TinyLlama locality result does not transfer as a clear Mistral improvement. Fixed strength 1 slightly improves the three-seed mean and variance, but it loses seeds 0 and 1 and gains mainly through seed 2. The effect size is small relative to seed variation, while the optimized masks and quantization profiles remain materially different.",
            "",
            "This is useful scale-dependent evidence. Small local mutations appear helpful in TinyLlama's smaller search space, but Mistral's larger depth configuration may benefit from occasional multi-swap moves. Neither fixed strength 1 nor unrestricted max-3 is uniformly superior from the current three seeds.",
            "",
            "## Runtime caveat",
            "",
            f"The max-3 runs used {', '.join(max3_gpu) or 'unrecorded hardware'}, while fixed-1 used {', '.join(fixed_gpu) or 'unrecorded hardware'}. Therefore the shorter fixed-1 wall-clock runtime must not be attributed to mutation strength.",
            "",
            "## Decision",
            "",
            "Keep the unchanged max-3 G50 search as the primary Mistral method because fixed strength 1 does not consistently improve it. Do not run more seeds for this binary comparison yet. The next useful algorithmic experiment is a scheduled locality operator that starts with broader mutations and decays toward strength 1, with stagnation-based expansion only if provenance shows expanded moves being selected. Screen that design on TinyLlama before another three-seed Mistral run.",
            "",
            "## Generated artifacts",
            "",
            "- `results/mistral_fixed_mutation_ablation.csv`: paired final metrics and candidate overlap.",
            "- `results/mistral_fixed_mutation_convergence.csv`: generation-wise search and evaluation metrics.",
            "- `results/mistral_fixed_mutation_ablation.png`: paired final PPL and mean convergence.",
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

    colors = {"max3": "#8c1d40", "fixed1": "#1f77b4"}
    fig, (paired_ax, convergence_ax) = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )
    for row in pairs:
        paired_ax.plot(
            [0, 1],
            [row["max3_wikitext2_ppl"], row["fixed1_wikitext2_ppl"]],
            marker="o",
            alpha=0.75,
            label=f"Seed {row['seed']}",
        )
    paired_ax.set_xticks([0, 1], ["Max-3", "Fixed-1"])
    paired_ax.set_ylabel("Final WikiText2 perplexity")
    paired_ax.set_title("Paired Final Results")
    paired_ax.grid(alpha=0.25)
    paired_ax.legend()

    for variant in VARIANTS:
        by_generation: dict[int, list[float]] = defaultdict(list)
        for row in convergence:
            if row["variant"] != variant or row["wikitext2_ppl"] == "":
                continue
            by_generation[int(row["generation"])].append(
                float(row["wikitext2_ppl"])
            )
        generations = sorted(by_generation)
        convergence_ax.plot(
            generations,
            [
                statistics.mean(by_generation[generation])
                for generation in generations
            ],
            marker="o",
            color=colors[variant],
            label=LABELS[variant],
        )
    convergence_ax.set_xlabel("Generation")
    convergence_ax.set_ylabel("Mean WikiText2 perplexity")
    convergence_ax.set_title("Periodic Evaluation")
    convergence_ax.grid(alpha=0.25)
    convergence_ax.legend()

    fig.suptitle("Mistral-7B Mutation Locality Ablation")
    fig.tight_layout()
    fig.savefig(
        output_dir / "mistral_fixed_mutation_ablation.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    discovered = discover(runs_root)
    verify_matched_commands(discovered)
    hardware = hardware_by_run(Path(args.experiment_log))

    run_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    candidates: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for variant in VARIANTS:
        for seed in (0, 1, 2):
            row, generation_rows, candidate = load_run(
                variant,
                seed,
                discovered[variant][seed],
                hardware,
            )
            run_rows.append(row)
            convergence.extend(generation_rows)
            candidates[variant][seed] = candidate

    pairs = paired_rows(run_rows, candidates)
    write_csv(
        output_dir / "mistral_fixed_mutation_ablation.csv",
        (
            "seed",
            "max3_wikitext2_ppl",
            "fixed1_wikitext2_ppl",
            "fixed1_minus_max3_ppl",
            "max3_final_calibration_kl",
            "fixed1_final_calibration_kl",
            "fixed1_minus_max3_kl",
            "max3_runtime_seconds",
            "fixed1_runtime_seconds",
            "attention_mask_jaccard",
            "mlp_mask_jaccard",
            "equal_quant_assignments",
            "quant_modules",
            "quant_bit_l1_distance",
        ),
        pairs,
    )
    write_csv(
        output_dir / "mistral_fixed_mutation_convergence.csv",
        (
            "variant",
            "seed",
            "run_id",
            "phase",
            "generation",
            "best_search_fitness",
            "wikitext2_ppl",
            "train_ppl",
            "accepted_parent_replacement",
        ),
        convergence,
    )
    (output_dir / "mistral_fixed_mutation_ablation.md").write_text(
        build_markdown(run_rows, pairs),
        encoding="utf-8",
    )
    if not args.skip_plots:
        create_plot(output_dir, pairs, convergence)
    print(f"Wrote Mistral fixed-mutation summary to {output_dir}")


if __name__ == "__main__":
    main()
