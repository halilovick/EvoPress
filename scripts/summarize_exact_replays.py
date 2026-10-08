#!/usr/bin/env python3
"""Summarise exact-budget replay results (RQ1 baselines and RQ3 contrasts).

Reads ``replay_results.csv`` written by ``evo_exact_replay.py`` and the plan, and
writes ``rq3_contrasts.csv``, ``rq1_baselines.csv`` and ``summary.md`` next to it.
Torch-free.

Contrasts use loss-like metrics (lower is better): mean negative
log-likelihood on WikiText-2 and C4 (log perplexity) and, when available, the
calibration KL divergence. Crossed candidates evaluated with several repair
seeds enter with their mean; the range across seeds is reported separately as
the repair sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_replay import interaction_terms  # noqa: E402

METRICS = ("wikitext2_nll", "c4_nll", "calibration_kl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", required=True)
    return parser.parse_args(argv)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(results_dir: Path):
    plan = json.loads((results_dir / "plan_used.json").read_text(encoding="utf-8"))
    rows = {}
    with (results_dir / "replay_results.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "completed":
                rows[row["id"]] = row
    return plan, rows


def metric_values(rows, ids, metric):
    values = [as_float(rows[i][metric]) for i in ids if i in rows]
    return [v for v in values if v is not None]


def rq3_contrasts(plan, rows):
    pairs = {}
    for job in plan["jobs"]:
        pair = job.get("pair")
        if not pair:
            continue
        key = (pair["a"], pair["b"], pair["fill"], job["tier"])
        pairs.setdefault(key, {}).setdefault(pair["mask"], []).append(job["id"])
    out = []
    for (a, b, fill, tier), crossed in sorted(pairs.items()):
        own_a, own_b = f"own_{a}", f"own_{b}"
        record = {"a": a, "b": b, "fill": fill, "tier": tier}
        genes = [as_float(rows[i]["repair_changed_genes"]) for ids in crossed.values() for i in ids if i in rows]
        record["repair_genes_max"] = max(genes) if genes else None
        complete = own_a in rows and own_b in rows and a in crossed and b in crossed
        for metric in METRICS:
            j_aa = metric_values(rows, [own_a], metric)
            j_bb = metric_values(rows, [own_b], metric)
            j_ab = metric_values(rows, crossed.get(a, []), metric)
            j_ba = metric_values(rows, crossed.get(b, []), metric)
            if not (complete and j_aa and j_bb and j_ab and j_ba):
                continue
            terms = interaction_terms(j_aa[0], j_bb[0], statistics.mean(j_ab), statistics.mean(j_ba))
            record[f"{metric}_I"] = terms["I"]
            record[f"{metric}_delta_a"] = terms["delta_a"]
            record[f"{metric}_delta_b"] = terms["delta_b"]
            record[f"{metric}_repair_range"] = max(
                (max(v) - min(v)) if len(v) > 1 else 0.0 for v in (j_ab, j_ba)
            )
        out.append(record)
    return out


def rq1_rows(plan, rows):
    out = []
    for job in plan["jobs"]:
        if job["tier"] not in ("0", "C", "D") or job["id"] not in rows:
            continue
        row = rows[job["id"]]
        out.append({
            "id": job["id"], "purpose": job["purpose"], "precision": job["precision"],
            "wikitext2_ppl": row["wikitext2_ppl"], "c4_ppl": row["c4_ppl"],
            "calibration_kl": row["calibration_kl"], "repair_changed_genes": row["repair_changed_genes"],
        })
    return out


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row}, key=lambda k: (k not in ("a", "b", "fill", "tier", "id"), k))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "" if value is None else f"{value:+.4f}"


def main(argv=None):
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    plan, rows = load(results_dir)
    contrasts = rq3_contrasts(plan, rows)
    baselines = rq1_rows(plan, rows)
    write_csv(results_dir / "rq3_contrasts.csv", contrasts)
    write_csv(results_dir / "rq1_baselines.csv", baselines)
    lines = [
        "# Exact-budget replay summary", "",
        f"Completed jobs: {len(rows)} of {len(plan['jobs'])}.", "",
        "## RQ3: interaction contrasts (lower metric is better; I = -(delta_A + delta_B))", "",
        "| A | B | fill | tier | W2 NLL I | delta_A | delta_B | C4 NLL I | repair genes (max) | repair range W2 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in contrasts:
        lines.append(
            f"| {r['a']} | {r['b']} | {r['fill']} | {r['tier']} | {fmt(r.get('wikitext2_nll_I'))} | "
            f"{fmt(r.get('wikitext2_nll_delta_a'))} | {fmt(r.get('wikitext2_nll_delta_b'))} | "
            f"{fmt(r.get('c4_nll_I'))} | {r.get('repair_genes_max')} | {fmt(r.get('wikitext2_nll_repair_range'))} |"
        )
    lines += ["", "## RQ1 baselines and references", "", "| Job | Precision | W2 PPL | C4 PPL | repair genes |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in baselines:
        lines.append(f"| `{r['id']}` | {r['precision']} | {r['wikitext2_ppl']} | {r['c4_ppl']} | {r['repair_changed_genes']} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"contrasts={len(contrasts)} baselines={len(baselines)}")


if __name__ == "__main__":
    main()
