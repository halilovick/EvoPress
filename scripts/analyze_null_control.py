#!/usr/bin/env python3
"""Compare a repaired shared-gene exchange's real RQ3 cost against its
deficit-matched null control (Tier B-ctrl, see ``plan_exact_replays.py``).

For each Tier B / Tier B-ctrl direction (mask M crossed against other mask
N), reports:

    delta_real = mean_metric(x_owner_exclusive_{M}_Q{N}_r*) - metric(own_{M})
    delta_null = mean_metric(rnd_{M}_vs_{N}_r*)            - metric(own_{M})
    excess     = delta_real - delta_null

``excess`` near zero means the real shared-gene exchange's cost is explained
by perturbing about the same number of the mask owner's genes by about the
same magnitude, regardless of where the perturbation came from -- i.e. the
negative RQ3 interaction of replay-findings-2026-10-09.md, point 9, is an
upper bound on the interaction, not evidence of one, exactly as that note
already suspected. A clearly positive ``excess`` (the real cross costs more
than deficit-matched random noise) is evidence of a genuine mask x
quantization interaction beyond that confound.

Metrics: mean negative log-likelihood on WikiText-2 (log perplexity) and the
calibration KL divergence, matching replay-findings-2026-10-09.md and
scripts/summarize_exact_replays.py. Torch-free; reads replay_results.csv
written by evo_exact_replay.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

METRICS = (("wikitext2_ppl", True), ("calibration_kl", False))  # (column, is_ppl)

DIRECTIONS = [
    ("E3s0", "E3s1"), ("E3s1", "E3s0"),
    ("E3s0", "E3s2"), ("E3s2", "E3s0"),
    ("E3s1", "E3s2"), ("E3s2", "E3s1"),
    ("J12s0", "J12s2"), ("J12s2", "J12s0"),
    ("J12s1", "J12s2"), ("J12s2", "J12s1"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", required=True)
    return parser.parse_args(argv)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(results_dir: Path) -> dict[str, dict]:
    rows = {}
    with (results_dir / "replay_results.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "completed":
                rows[row["id"]] = row
    return rows


def metric_value(row, column, is_ppl):
    value = as_float(row.get(column))
    if value is None:
        return None
    return math.log(value) if is_ppl else value


def gather(rows, ids, column, is_ppl):
    values = [metric_value(rows[i], column, is_ppl) for i in ids if i in rows]
    return [v for v in values if v is not None]


def analyze(rows: dict[str, dict]) -> list[dict]:
    records = []
    for mask, other in DIRECTIONS:
        own_id = f"own_{mask}"
        if own_id not in rows:
            continue
        real_ids = [f"x_owner_exclusive_{mask}_Q{other}_r{s}" for s in (0, 1, 2)]
        null_ids = [f"rnd_{mask}_vs_{other}_r{s}" for s in (0, 1, 2)]
        record = {"mask": mask, "other": other}
        real_genes = gather(rows, real_ids, "repair_changed_genes", False)
        null_genes = gather(rows, null_ids, "repair_changed_genes", False)
        record["real_genes"] = round(statistics.mean(real_genes), 1) if real_genes else None
        record["null_genes"] = round(statistics.mean(null_genes), 1) if null_genes else None
        for column, is_ppl in METRICS:
            own = metric_value(rows[own_id], column, is_ppl)
            real_vals = gather(rows, real_ids, column, is_ppl)
            null_vals = gather(rows, null_ids, column, is_ppl)
            if own is None or not real_vals or not null_vals:
                continue
            delta_real = statistics.mean(real_vals) - own
            delta_null = statistics.mean(null_vals) - own
            record[f"{column}_delta_real"] = delta_real
            record[f"{column}_delta_null"] = delta_null
            record[f"{column}_excess"] = delta_real - delta_null
            record[f"{column}_real_sd"] = statistics.pstdev(real_vals) if len(real_vals) > 1 else 0.0
            record[f"{column}_null_sd"] = statistics.pstdev(null_vals) if len(null_vals) > 1 else 0.0
        records.append(record)
    return records


def fmt(value):
    return "" if value is None else f"{value:+.4f}"


def write(results_dir: Path, records: list[dict]) -> None:
    if records:
        columns = sorted({key for r in records for key in r}, key=lambda k: (k not in ("mask", "other"), k))
        with (results_dir / "null_control_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

    lines = [
        "# RQ3 repair-cost control: real shared-gene exchange vs. deficit-matched random fill",
        "",
        "delta = mean(crossed or null) - own, in log PPL / KL (lower is better, so a positive delta is a loss).",
        "excess = delta_real - delta_null. Near 0: the real exchange's cost is explained by perturbing about",
        "this many genes by about this much, regardless of source (confirms the upper-bound reading of",
        "replay-findings-2026-10-09.md, point 9). Clearly positive: the real exchange costs more than",
        "equally-sized random noise -- evidence of a genuine mask x quantization interaction.",
        "",
        "| Mask | vs | genes (real/null) | W2 NLL δreal | δnull | excess | KL δreal | δnull | excess |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| {r['mask']} | {r['other']} | {r.get('real_genes', '-')}/{r.get('null_genes', '-')} | "
            f"{fmt(r.get('wikitext2_ppl_delta_real'))} | {fmt(r.get('wikitext2_ppl_delta_null'))} | "
            f"{fmt(r.get('wikitext2_ppl_excess'))} | "
            f"{fmt(r.get('calibration_kl_delta_real'))} | {fmt(r.get('calibration_kl_delta_null'))} | "
            f"{fmt(r.get('calibration_kl_excess'))} |"
        )

    excesses = [r["wikitext2_ppl_excess"] for r in records if r.get("wikitext2_ppl_excess") is not None]
    if excesses:
        sd = statistics.pstdev(excesses) if len(excesses) > 1 else 0.0
        lines += ["", f"Mean W2 NLL excess over {len(excesses)} directions: "
                       f"{statistics.mean(excesses):+.4f} (population SD {sd:.4f})."]
    (results_dir / "null_control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    rows = load(results_dir)
    records = analyze(rows)
    write(results_dir, records)
    print(f"directions analyzed: {len(records)} of {len(DIRECTIONS)}")


if __name__ == "__main__":
    main()
