#!/usr/bin/env python3
"""Harvest results from the crossover hyperparameter sweep (G20, cheap
S-protocol) into one mean +/- SD table per (crossover_type,
crossover_probability) cell, across 3 seeds.

Schema confirmed against a real completed run_summary.json
(joint_g20_component_xover_pop4_seed0, schema_version 2):
  - final_metrics.wikitext2_ppl          -> eval PPL
  - final_metrics.runtime_seconds        -> launcher wall-clock runtime
  - final_metrics.best_search_fitness    -> internal KL fitness of best candidate
  - final_calibration_kl is NOT a top-level key; it's final_metrics.final_calibration_kl
  - crossover_offspring_attempted/accepted, crossover_repaired_proposals,
    crossover_repair_changed_gene_count, crossover_duplicates,
    crossover_infeasible_candidates are all top-level ints
  - mutation_offspring_accepted is top-level (for comparison against
    crossover_offspring_accepted -- how much of the population turnover each
    generation mechanism is actually responsible for)

Reads:
  - The 4 pre-existing cells from
    thesis_results/datalab_raw/cheap_population_crossover/
    (joint_g20_{component,layer_bundle,local_exchange}_xover_pop4_seed0,
     joint_g20_population_pop4_seed0)
  - The new cells produced by run_crossover_sweep_g20.sh, under
    outputs/experiments/sweep_crossover_g20/joint_g20_<type>_<prob>_pop4_seed<seed>/

Usage (run from the repo root on DataLab, where outputs/ lives):
    python thesis_results/datalab_raw/launchers/crossover_sweep_g20/harvest_crossover_sweep.py \
        --sweep_dir outputs/experiments/sweep_crossover_g20 \
        --existing_dir thesis_results/datalab_raw/cheap_population_crossover \
        --out_csv thesis_results/crossover_sweep_g20_summary.csv

Prints a markdown table to stdout and writes per-seed + aggregated CSVs.
Missing/incomplete cells are reported, not silently dropped.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

TYPES = ["component", "layer_bundle", "local_exchange"]
TYPE_LABEL = {"component": "component", "layer_bundle": "layerbundle", "local_exchange": "localexchange"}
PROBS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
PROB_LABEL = {0.0: "p000", 0.1: "p010", 0.25: "p025", 0.5: "p050", 0.75: "p075", 1.0: "p100"}
SEEDS = [0, 1, 2]

EXISTING_DIR_NAMES = {
    # (type_or_None_for_pc0, prob) -> existing dir name, seed 0 only
    ("component", 0.25): "joint_g20_component_xover_pop4_seed0",
    ("layer_bundle", 0.25): "joint_g20_layerbundle_xover_pop4_seed0",
    ("local_exchange", 0.25): "joint_g20_localexchange_xover_pop4_seed0",
    (None, 0.0): "joint_g20_population_pop4_seed0",
}

PER_SEED_FIELDS = [
    "crossover_type", "crossover_probability", "seed",
    "wikitext2_ppl", "best_search_fitness", "final_calibration_kl",
    "crossover_offspring_attempted", "crossover_offspring_accepted",
    "crossover_repaired_proposals", "crossover_repair_changed_gene_count",
    "crossover_duplicates", "crossover_infeasible_candidates",
    "mutation_offspring_accepted",
    "runtime_seconds", "path",
]

AGG_FIELDS = [
    "crossover_type", "crossover_probability", "n_seeds",
    "mean_wikitext2_ppl", "sd_wikitext2_ppl",
    "mean_crossover_offspring_accepted", "mean_crossover_repaired_proposals",
]


def find_summary_file(cell_dir: Path):
    candidate = cell_dir / "run_summary.json"
    return candidate if candidate.exists() else None


def extract_row(summary_path: Path):
    try:
        data = json.loads(summary_path.read_text())
    except Exception as exc:
        return None, f"unparseable ({exc})"

    fm = data.get("final_metrics", {})
    ppl = fm.get("wikitext2_ppl")
    if ppl is None:
        return None, "final_metrics.wikitext2_ppl missing (run likely incomplete)"

    row = {
        "crossover_type": data.get("crossover_type"),
        "crossover_probability": data.get("crossover_probability"),
        "seed": data.get("search_config", {}).get("seed"),
        "wikitext2_ppl": ppl,
        "best_search_fitness": fm.get("best_search_fitness"),
        "final_calibration_kl": fm.get("final_calibration_kl"),
        "crossover_offspring_attempted": data.get("crossover_offspring_attempted"),
        "crossover_offspring_accepted": data.get("crossover_offspring_accepted"),
        "crossover_repaired_proposals": data.get("crossover_repaired_proposals"),
        "crossover_repair_changed_gene_count": data.get("crossover_repair_changed_gene_count"),
        "crossover_duplicates": data.get("crossover_duplicates"),
        "crossover_infeasible_candidates": data.get("crossover_infeasible_candidates"),
        "mutation_offspring_accepted": data.get("mutation_offspring_accepted"),
        "runtime_seconds": fm.get("runtime_seconds"),
    }
    return row, None


def cell_path_for(sweep_dir: Path, existing_dir: Path, xtype, prob, seed):
    key = (xtype, prob) if prob != 0.0 else (None, prob)
    if seed == 0 and key in EXISTING_DIR_NAMES:
        return existing_dir / EXISTING_DIR_NAMES[key]
    tlabel = TYPE_LABEL[xtype] if xtype else "component"
    plabel = PROB_LABEL[prob]
    return sweep_dir / f"joint_g20_{tlabel}_{plabel}_pop4_seed{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default="outputs/experiments/sweep_crossover_g20", type=Path)
    parser.add_argument("--existing_dir", default="thesis_results/datalab_raw/cheap_population_crossover", type=Path)
    parser.add_argument("--out_csv", default="thesis_results/crossover_sweep_g20_summary.csv", type=Path)
    args = parser.parse_args()

    rows = []
    missing = []

    cells = [(None, 0.0)] + [(t, p) for p in PROBS if p != 0.0 for t in TYPES]
    for xtype, prob in cells:
        for seed in SEEDS:
            cell_dir = cell_path_for(args.sweep_dir, args.existing_dir, xtype, prob, seed)
            label = f"{xtype or 'n/a'} / p_c={prob} / seed{seed}"
            if not cell_dir.exists():
                missing.append((label, str(cell_dir), "directory does not exist yet"))
                continue
            summary = find_summary_file(cell_dir)
            if summary is None:
                missing.append((label, str(cell_dir), "no run_summary.json (still running or crashed before writing it)"))
                continue
            row, err = extract_row(summary)
            if err:
                missing.append((label, str(cell_dir), err))
                continue
            row["path"] = str(cell_dir)
            # Report the display label's type/prob even if p_c=0 run has an
            # arbitrary (inert) crossover_type recorded.
            rows.append(row)

    # Aggregate per (type, prob) -- use the display grouping, not the raw
    # recorded crossover_type for p_c=0 cells (which is inert/arbitrary there).
    agg = {}
    for r in rows:
        prob = r["crossover_probability"]
        xtype = "n/a (p_c=0)" if prob == 0.0 else r["crossover_type"]
        key = (xtype, prob)
        agg.setdefault(key, []).append(r)

    agg_rows = []
    for (xtype, prob), group in sorted(agg.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        ppls = [r["wikitext2_ppl"] for r in group]
        n = len(ppls)
        mean = statistics.mean(ppls)
        sd = statistics.stdev(ppls) if n > 1 else float("nan")
        accepted = [r["crossover_offspring_accepted"] for r in group if r["crossover_offspring_accepted"] is not None]
        repaired = [r["crossover_repaired_proposals"] for r in group if r["crossover_repaired_proposals"] is not None]
        agg_rows.append({
            "crossover_type": xtype,
            "crossover_probability": prob,
            "n_seeds": n,
            "mean_wikitext2_ppl": round(mean, 4),
            "sd_wikitext2_ppl": round(sd, 4) if n > 1 else "",
            "mean_crossover_offspring_accepted": round(statistics.mean(accepted), 1) if accepted else "",
            "mean_crossover_repaired_proposals": round(statistics.mean(repaired), 1) if repaired else "",
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    agg_csv = args.out_csv.with_name(args.out_csv.stem + "_aggregated.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    print(f"\n=== Crossover sweep summary: {len(rows)} complete cells, {len(missing)} missing/incomplete ===\n")
    print("| Type | p_c | n_seeds | mean WT2 PPL | SD | mean xover accepted | mean repaired |")
    print("|---|---|---|---|---|---|---|")
    for r in agg_rows:
        sd_str = r["sd_wikitext2_ppl"] if r["sd_wikitext2_ppl"] != "" else "-"
        print(f"| {r['crossover_type']} | {r['crossover_probability']} | {r['n_seeds']} | "
              f"{r['mean_wikitext2_ppl']:.4f} | {sd_str} | {r['mean_crossover_offspring_accepted']} | "
              f"{r['mean_crossover_repaired_proposals']} |")

    if missing:
        print(f"\n--- {len(missing)} missing/incomplete cells ---")
        for label, path, reason in missing:
            print(f"  {label}: {reason} ({path})")

    print(f"\nPer-seed CSV: {args.out_csv}")
    print(f"Aggregated CSV: {agg_csv}")

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
