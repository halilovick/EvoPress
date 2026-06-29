#!/usr/bin/env python3
"""Create a consolidated q_proj-vs-attention Mistral evidence summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


DATASETS = ("wikitext2", "c4", "fineweb_edu")
METHODS = ("dense", "depth", "independent", "joint_g50")
OUTPUT_COLUMNS = (
    "scope",
    "method",
    "label",
    "compression_ratio",
    "wikitext2_ppl",
    "c4_ppl",
    "fineweb_edu_ppl",
    "lmeval_macro",
    "delta_wikitext2_vs_depth",
    "delta_wikitext2_vs_independent",
    "delta_lmeval_vs_depth",
    "delta_lmeval_vs_independent",
    "notes",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Mistral q_proj and attention-scope evidence."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-stem", default="mistral_scope_comparison")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def read_generalization(path: Path) -> dict[tuple[str, str], float]:
    rows = read_csv(path)
    return {
        (row["method"], row["dataset"]): float(row["ppl_mean"])
        for row in rows
    }


def read_lmeval_macros(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(float(row["score_mean"]))
    return {
        method: statistics.mean(values)
        for method, values in grouped.items()
        if values
    }


def run_summary_metric(path: Path, key: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    final_metrics = data.get("final_metrics", {})
    if key not in final_metrics:
        raise KeyError(f"Missing {key} in {path}")
    return float(final_metrics[key])


def compression_mean(runs_root: Path, pattern: str, seeds: Sequence[int]) -> float:
    values = []
    for seed in seeds:
        path = runs_root / pattern.format(seed=seed) / "run_summary.json"
        values.append(run_summary_metric(path, "estimated_compression_ratio"))
    return statistics.mean(values)


def medium_metric(results_dir: Path, method: str, column: str) -> float:
    rows = read_csv(results_dir / "mistral_medium_aggregate.csv")
    for row in rows:
        if row["method"] == method:
            return float(row[column])
    raise KeyError(f"Missing method={method} column={column}")


def build_rows(results_dir: Path, runs_root: Path) -> list[dict[str, Any]]:
    qproj_gen = read_generalization(results_dir / "mistral_generalization_aggregate.csv")
    attn_gen = read_generalization(
        results_dir / "mistral_attention_generalization_aggregate.csv"
    )
    qproj_lm = read_lmeval_macros(results_dir / "mistral_lmeval_aggregate.csv")
    attn_lm = read_lmeval_macros(results_dir / "mistral_attention_lmeval_aggregate.csv")

    depth_compression = compression_mean(
        runs_root,
        "thesis_medium_depth_mistral_s0.25_g20_o16_seed{seed}",
        [0, 1, 2],
    )
    qproj_joint_compression = compression_mean(
        runs_root,
        "thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed{seed}",
        [0, 1, 2],
    )
    attention_joint_compression = compression_mean(
        runs_root,
        "thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed{seed}",
        [0, 1, 2],
    )
    qproj_quant_compression = compression_mean(
        runs_root,
        "thesis_medium_quant_mistral_qproj3.0_g20_o16_seed{seed}",
        [0, 1, 2],
    )
    attention_quant_compression = compression_mean(
        runs_root,
        "thesis_attention_quant_mistral_attention3.0_g20_o16_seed{seed}",
        [0, 1, 2],
    )

    rows: list[dict[str, Any]] = [
        {
            "scope": "baseline",
            "method": "dense",
            "label": "Dense FP16",
            "compression_ratio": 1.0,
            "generalization": qproj_gen,
            "lmeval_macro": qproj_lm.get("dense"),
            "notes": "Uncompressed reference.",
        },
        {
            "scope": "depth",
            "method": "depth",
            "label": "Depth-only 25%",
            "compression_ratio": depth_compression,
            "generalization": qproj_gen,
            "lmeval_macro": qproj_lm.get("depth"),
            "notes": "Depth pruning baseline reused by both composition scopes.",
        },
        {
            "scope": "q_proj",
            "method": "quant_only",
            "label": "q_proj quant-only",
            "compression_ratio": qproj_quant_compression,
            "wikitext2_ppl": medium_metric(
                results_dir, "quant", "wikitext2_ppl_mean"
            ),
            "lmeval_macro": None,
            "notes": "Quant-only search; no multi-dataset or LM-eval run.",
        },
        {
            "scope": "q_proj",
            "method": "independent",
            "label": "Depth + independent q_proj quant",
            "compression_ratio": medium_metric(
                results_dir, "independent", "estimated_compression_ratio_mean"
            ),
            "generalization": qproj_gen,
            "lmeval_macro": qproj_lm.get("independent"),
            "notes": "Depth and quantization selected separately.",
        },
        {
            "scope": "q_proj",
            "method": "joint_g50",
            "label": "Joint G50 depth + q_proj quant",
            "compression_ratio": qproj_joint_compression,
            "generalization": qproj_gen,
            "lmeval_macro": qproj_lm.get("joint_g50"),
            "notes": "Best q_proj combined search setting.",
        },
        {
            "scope": "attention",
            "method": "quant_only",
            "label": "Attention q/k/v/o quant-only",
            "compression_ratio": attention_quant_compression,
            "wikitext2_ppl": mean(
                [
                    run_summary_metric(
                        runs_root
                        / f"thesis_attention_quant_mistral_attention3.0_g20_o16_seed{seed}"
                        / "run_summary.json",
                        "wikitext2_ppl",
                    )
                    for seed in (0, 1, 2)
                ]
            ),
            "lmeval_macro": None,
            "notes": "Quant-only search; no multi-dataset or LM-eval run.",
        },
        {
            "scope": "attention",
            "method": "independent",
            "label": "Depth + independent attention quant",
            "compression_ratio": attention_joint_compression,
            "generalization": attn_gen,
            "lmeval_macro": attn_lm.get("independent"),
            "notes": (
                "Compression ratio uses the matched attention joint target; "
                "composition eval artifacts do not contain a run_summary.json."
            ),
        },
        {
            "scope": "attention",
            "method": "joint_g50",
            "label": "Joint G50 depth + attention quant",
            "compression_ratio": attention_joint_compression,
            "generalization": attn_gen,
            "lmeval_macro": attn_lm.get("joint_g50"),
            "notes": "Broader q/k/v/o attention scope at the same depth target.",
        },
    ]

    depth_wikitext = qproj_gen[("depth", "wikitext2")]
    depth_lmeval = qproj_lm["depth"]
    independent_by_scope = {
        "q_proj": {
            "wikitext2": qproj_gen[("independent", "wikitext2")],
            "lmeval": qproj_lm["independent"],
        },
        "attention": {
            "wikitext2": attn_gen[("independent", "wikitext2")],
            "lmeval": attn_lm["independent"],
        },
    }

    normalized: list[dict[str, Any]] = []
    for row in rows:
        generalization = row.pop("generalization", None)
        method = row["method"]
        scope = row["scope"]
        if generalization:
            for dataset in DATASETS:
                row[f"{dataset}_ppl"] = generalization.get((method, dataset))
        row.setdefault("wikitext2_ppl", None)
        row.setdefault("c4_ppl", None)
        row.setdefault("fineweb_edu_ppl", None)

        row["delta_wikitext2_vs_depth"] = (
            row["wikitext2_ppl"] - depth_wikitext
            if row["wikitext2_ppl"] is not None
            else None
        )
        row["delta_lmeval_vs_depth"] = (
            row["lmeval_macro"] - depth_lmeval
            if row["lmeval_macro"] is not None
            else None
        )
        if scope in independent_by_scope and row["method"] != "independent":
            row["delta_wikitext2_vs_independent"] = (
                row["wikitext2_ppl"] - independent_by_scope[scope]["wikitext2"]
                if row["wikitext2_ppl"] is not None
                else None
            )
            row["delta_lmeval_vs_independent"] = (
                row["lmeval_macro"] - independent_by_scope[scope]["lmeval"]
                if row["lmeval_macro"] is not None
                else None
            )
        else:
            row["delta_wikitext2_vs_independent"] = None
            row["delta_lmeval_vs_independent"] = None
        normalized.append(row)
    return normalized


def write_csv_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        f"{row[column]:.6g}"
                        if isinstance(row.get(column), float)
                        else row.get(column, "")
                    )
                    for column in OUTPUT_COLUMNS
                }
            )


def row_by(rows: list[dict[str, Any]], scope: str, method: str) -> dict[str, Any]:
    for row in rows:
        if row["scope"] == scope and row["method"] == method:
            return row
    raise KeyError((scope, method))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    qproj_joint = row_by(rows, "q_proj", "joint_g50")
    qproj_ind = row_by(rows, "q_proj", "independent")
    attn_joint = row_by(rows, "attention", "joint_g50")
    attn_ind = row_by(rows, "attention", "independent")
    depth = row_by(rows, "depth", "depth")

    lines = [
        "# Mistral Scope Comparison Summary",
        "",
        "This summary consolidates the Mistral-7B evidence for the thesis "
        "question: whether joint search helps when combining structural depth "
        "pruning with quantization, and how the conclusion changes when the "
        "quantization scope is expanded from `q_proj` to all attention "
        "projections `q/k/v/o`.",
        "",
        "## Main Comparison",
        "",
        "| Scope | Method | Compression | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | LM-eval macro |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['label']} | "
            f"{fmt(row['compression_ratio'])}x | "
            f"{fmt(row['wikitext2_ppl'])} | "
            f"{fmt(row['c4_ppl'])} | "
            f"{fmt(row['fineweb_edu_ppl'])} | "
            f"{fmt(row['lmeval_macro'])} |"
        )

    lines.extend(
        [
            "",
            "## Core Findings",
            "",
            f"- `q_proj` joint G50 is the best quality-compression point found so far: "
            f"{fmt(qproj_joint['compression_ratio'])}x compression, "
            f"WikiText2 PPL {fmt(qproj_joint['wikitext2_ppl'])}, "
            f"C4 PPL {fmt(qproj_joint['c4_ppl'])}, "
            f"FineWeb-Edu PPL {fmt(qproj_joint['fineweb_edu_ppl'])}, "
            f"and LM-eval macro {fmt(qproj_joint['lmeval_macro'])}.",
            f"- `q_proj` joint reduces perplexity relative to the matched independent "
            f"composition by {fmt(qproj_ind['wikitext2_ppl'] - qproj_joint['wikitext2_ppl'])} "
            f"WikiText2 PPL, {fmt(qproj_ind['c4_ppl'] - qproj_joint['c4_ppl'])} C4 PPL, "
            f"and {fmt(qproj_ind['fineweb_edu_ppl'] - qproj_joint['fineweb_edu_ppl'])} "
            f"FineWeb-Edu PPL. Its LM-eval macro is slightly lower: "
            f"{fmt(qproj_joint['lmeval_macro'] - qproj_ind['lmeval_macro'])}.",
            f"- Attention-scope joint reaches higher compression "
            f"({fmt(attn_joint['compression_ratio'])}x) but with lower quality: "
            f"WikiText2 PPL {fmt(attn_joint['wikitext2_ppl'])} and LM-eval macro "
            f"{fmt(attn_joint['lmeval_macro'])}.",
            f"- Within the broader attention scope, joint still reduces perplexity "
            f"relative to the matched independent composition by "
            f"{fmt(attn_ind['wikitext2_ppl'] - attn_joint['wikitext2_ppl'])} "
            f"WikiText2 PPL, {fmt(attn_ind['c4_ppl'] - attn_joint['c4_ppl'])} C4 PPL, "
            f"{fmt(attn_ind['fineweb_edu_ppl'] - attn_joint['fineweb_edu_ppl'])} "
            f"FineWeb-Edu PPL, and {fmt(attn_joint['lmeval_macro'] - attn_ind['lmeval_macro'])} "
            f"LM-eval macro.",
            f"- Depth-only remains a strong baseline: {fmt(depth['compression_ratio'])}x "
            f"compression, WikiText2 PPL {fmt(depth['wikitext2_ppl'])}, and "
            f"LM-eval macro {fmt(depth['lmeval_macro'])}. Attention-scope compression "
            f"does not beat this quality baseline, but it targets stronger compression.",
            "",
            "## Thesis Interpretation",
            "",
            "The evidence supports a nuanced claim rather than a simple win. Joint "
            "search consistently helps relative to independently combining depth "
            "and quantization at the same scope, especially in perplexity. However, "
            "expanding the quantization scope from `q_proj` to full attention "
            "projections increases compression while reducing model quality. "
            "The strongest thesis framing is therefore a compression-quality "
            "tradeoff: EvoPress-style joint search can recover part of the quality "
            "lost by broader combined compression, but the chosen scope and target "
            "compression strongly determine whether the result is competitive with "
            "depth-only pruning.",
            "",
            "## Generated Artifacts",
            "",
            f"- `{path.with_suffix('.csv')}`",
            f"- `{path}`",
            "",
            "## Source Summaries",
            "",
            "- `results/mistral_medium_aggregate.csv`",
            "- `results/mistral_generalization_aggregate.csv`",
            "- `results/mistral_attention_generalization_aggregate.csv`",
            "- `results/mistral_lmeval_aggregate.csv`",
            "- `results/mistral_attention_lmeval_aggregate.csv`",
            "- structured `run_summary.json` files under `results/runs/`",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    runs_root = Path(args.runs_root)
    rows = build_rows(results_dir, runs_root)
    output_base = results_dir / args.output_stem
    write_csv_output(output_base.with_suffix(".csv"), rows)
    write_markdown(output_base.with_suffix(".md"), rows)
    print(f"Wrote {len(rows)} comparison rows.")
    print(f"Wrote {output_base.with_suffix('.csv')}.")
    print(f"Wrote {output_base.with_suffix('.md')}.")


if __name__ == "__main__":
    main()
