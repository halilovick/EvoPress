#!/usr/bin/env python3
"""Figures for the final results presentation (CPU only, no model needed).

Every number is read from the stored run artifacts:

* ``run_summary.json`` of the full-scale searches (final WikiText-2 / C4
  perplexity and calibration KL);
* ``results/exact_lmeval/fullspace_lmeval/lmeval_results.csv`` (zero-shot
  accuracy: acc_norm for ARC-Easy and PIQA, acc for WinoGrande).

Writes vector PDFs plus the CSV of plotted values to ``--output_dir``:

* ``rq1_removal_vs_precision.pdf``: WikiText-2 perplexity at the same storage
  for uniform 3-bit, quantization-only search and joint search at 12.5% and
  25% removal, with per-seed points and mean zero-shot accuracy;
* ``rq2_search_variants.pdf``: final WikiText-2 perplexity per seed of the
  standard joint search and the six search-design variants at 25% removal.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = "thesis_results/datalab_raw"
PM = f"{RAW}/paper_matched"
EXT = f"{RAW}/fullspace_extensions"

RUNS = {
    "E1": [f"{PM}/uniform3/uniform3_seed0_20260819"],
    "E2": [f"{PM}/quant_only/quant_only_seed0_attempt12_resume_20260826",
           f"{PM}/quant_only/quant_only_seed1_attempt4_resume_20260828",
           f"{PM}/quant_only/quant_only_seed2_attempt2_resume_20260830"],
    "J12": [f"{EXT}/fullspace_joint_s0125_g150_seed{s}_attempt1" for s in range(3)],
    "E3": [f"{PM}/joint/joint_seed0_attempt1_20260831",
           f"{PM}/joint/joint_seed1_attempt3_resume_20260902",
           f"{PM}/joint/joint_seed2_attempt1_20260903"],
    "IA": [f"{EXT}/fullspace_joint_s025_ia_g150_seed{s}_attempt1" for s in range(3)],
    "DW": [f"{EXT}/fullspace_depthwarm_g150_seed{s}_attempt1" for s in range(3)],
    "P4": [f"{EXT}/fullspace_joint_s025_pop4_g150_seed{s}_attempt1" for s in range(3)],
    "LB": [f"{EXT}/fullspace_joint_s025_layerbundle_pop4_g150_seed{s}_attempt1" for s in range(2)],
    "LX": [f"{EXT}/fullspace_joint_s025_localxover_pop4_g150_seed0_attempt1"],
    "AO": [f"{EXT}/fullspace_joint_s025_activeonly_g150_seed{s}_attempt1" for s in range(3)],
}
LMEVAL_IDS = {"E1": ["own_uniform3"], "E2": [f"own_E2s{s}" for s in range(3)],
              "J12": [f"own_J12s{s}" for s in range(3)], "E3": [f"own_E3s{s}" for s in range(3)]}

# Reference palette (dataviz skill): one accent hue, neutrals for everything else.
BLUE = "#2a78d6"
GRAY = "#b9b8b2"
DARK_GRAY = "#52514e"
INK = "#0b0b0b"
GRID = "#e4e3df"


def final_metrics(rel: str) -> dict:
    data = json.loads((REPO / rel / "run_summary.json").read_text(encoding="utf-8"))
    return data.get("final_metrics", data)


def w2(label: str) -> list[float]:
    return [float(final_metrics(rel)["wikitext2_ppl"]) for rel in RUNS[label]]


def accuracy() -> dict[str, list[float]]:
    rows = list(csv.DictReader((REPO / "results/exact_lmeval/fullspace_lmeval/lmeval_results.csv").open()))
    per_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_model[row["id"]].append(float(row["score"]))
    out = {}
    for label, ids in LMEVAL_IDS.items():
        out[label] = [100 * statistics.mean(per_model[i]) for i in ids if len(per_model[i]) == 3]
    return out


def style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.tick_params(colors=DARK_GRAY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def rq1_figure(out: Path, table: list[dict]) -> None:
    labels = ["E1", "E2", "J12", "E3"]
    names = ["Uniform\n3-bit", "Bit search,\nnothing removed", "Joint search,\n12.5% removed",
             "Joint search,\n25% removed"]
    acc = accuracy()
    values = {lab: w2(lab) for lab in labels}
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, lab in enumerate(labels):
        vals = values[lab]
        mean = statistics.mean(vals)
        color = BLUE if lab == "E2" else GRAY
        ax.bar(i, mean, width=0.62, color=color, edgecolor="white", linewidth=2, zorder=2)
        if len(vals) > 1:
            ax.scatter([i] * len(vals), vals, s=9, color=INK, zorder=3, linewidths=0)
        ax.text(i, max(vals) + 0.18, f"{mean:.2f}", ha="center", va="bottom", fontsize=9,
                color=INK, fontweight="bold")
        a = statistics.mean(acc[lab])
        table.append({"figure": "rq1", "label": lab, "wikitext2_mean": f"{mean:.3f}",
                      "wikitext2_seeds": " ".join(f"{v:.4f}" for v in vals),
                      "zero_shot_mean_pct": f"{a:.1f}", "n_seeds": len(vals)})
        names[i] = f"{names[i]}\nacc. {a:.1f}%"
    ax.set_xticks(range(len(labels)), names, fontsize=7.5, color=DARK_GRAY)
    ax.set_ylabel("WikiText-2 perplexity (lower is better)", fontsize=7.5, color=DARK_GRAY)
    ax.set_ylim(0, 10.2)
    ax.tick_params(axis="y", labelsize=7.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "rq1_removal_vs_precision.pdf")
    fig.savefig(out / "rq1_removal_vs_precision.png", dpi=200)
    plt.close(fig)


def rq2_figure(out: Path, table: list[dict]) -> None:
    rows = [
        ("E3", "Standard search (EvoPress design)"),
        ("AO", "Mutation only on kept projections"),
        ("P4", "Population of four (+33% tokens)"),
        ("DW", "Warm start from a depth-only mask"),
        ("LX", "Population + local-exchange crossover (1 seed)"),
        ("LB", "Population + layer-bundle crossover (2 seeds)"),
        ("IA", "Interaction-aware mutation"),
    ]
    ref = w2("E3")
    fig, ax = plt.subplots(figsize=(4.9, 2.7))
    ax.axvspan(min(ref), max(ref), color="#eaf1fb", zorder=0, linewidth=0)
    for y, (lab, name) in enumerate(reversed(rows)):
        vals = w2(lab)
        mean = statistics.mean(vals)
        color = BLUE if lab == "E3" else DARK_GRAY
        ax.scatter(vals, [y] * len(vals), s=22, color=color, zorder=3, linewidths=0)
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=color, linewidth=2, zorder=3)
        n = len(vals)
        ax.text(11.4, y, f"{mean:.2f}", va="center", ha="left", fontsize=7.5, color=INK,
                fontweight="bold" if lab == "E3" else "normal")
        table.append({"figure": "rq2", "label": lab, "wikitext2_mean": f"{mean:.3f}",
                      "wikitext2_seeds": " ".join(f"{v:.4f}" for v in vals), "n_seeds": n})
    ax.set_yticks(range(len(rows)), [name for _, name in reversed(rows)], fontsize=7.5, color=DARK_GRAY)
    ax.set_xlim(8.3, 11.3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Final WikiText-2 perplexity (lower is better)", fontsize=7.5, color=DARK_GRAY)
    ax.tick_params(axis="x", labelsize=7.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.tick_params(colors=DARK_GRAY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.savefig(out / "rq2_search_variants.pdf")
    fig.savefig(out / "rq2_search_variants.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", default=str(REPO / "thesis_results/presentation_figures"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    table: list[dict] = []
    rq1_figure(out, table)
    rq2_figure(out, table)
    fields = ["figure", "label", "n_seeds", "wikitext2_mean", "wikitext2_seeds", "zero_shot_mean_pct"]
    with (out / "plotted_values.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote figures and plotted_values.csv to {out}")


if __name__ == "__main__":
    main()
