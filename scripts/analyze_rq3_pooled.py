#!/usr/bin/env python3
"""Pooled analysis of the RQ3 shared-gene exchange against its null control.

Reads one or more replay result directories (``replay_results.csv``), finds every
direction that has both real jobs ``x_owner_exclusive_{M}_Q{N}_r*`` and null jobs
``rnd_{M}_vs_{N}_r*``, and computes per direction

    excess = mean(real) - mean(null)      (log WikiText-2 PPL and calibration KL)

The owner's own baseline cancels, so no ``own_*`` job is needed.  The pooled
statistics use a **pair-level** bootstrap (both directions of a pair are
resampled together) and a **model-level** bootstrap (all pairs containing a
resampled model), because the same final model appears in several pairs.
Equivalence is reported against margins in log PPL (0.01 = about 1% perplexity).

Groups: ``noIA`` (no interaction-aware candidate in the pair) and ``IA``.
For IA pairs the sum of the two directions is also reported; an antisymmetric
pattern (sum near zero) indicates a main effect of the bit profile, not an
interaction.  Torch-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path

REAL = re.compile(r"^x_owner_(?:exclusive|all)_(?P<m>\w+?)_Q(?P<o>\w+?)_r(?P<s>\d+)$")
NULL = re.compile(r"^rnd_(?P<m>\w+?)_vs_(?P<o>\w+?)_r(?P<s>\d+)$")
METRICS = (("wikitext2_ppl", True), ("calibration_kl", False))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--margins", nargs="+", type=float, default=[0.01, 0.02])
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def value(row, column, is_ppl):
    try:
        v = float(row[column])
    except (KeyError, TypeError, ValueError):
        return None
    return math.log(v) if is_ppl else v


def load_directions(dirs):
    real, null, fallback = {}, {}, set()
    for d in dirs:
        with (Path(d) / "replay_results.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["status"] != "completed":
                    continue
                m = REAL.match(row["id"])
                if m:
                    real.setdefault((m["m"], m["o"]), {})[int(m["s"])] = row
                    if "_all_" in row["id"]:
                        fallback.add((m["m"], m["o"]))
                    continue
                m = NULL.match(row["id"])
                if m:
                    null.setdefault((m["m"], m["o"]), {})[int(m["s"])] = row
    out = []
    for key in sorted(set(real) & set(null)):
        rec = {"mask": key[0], "other": key[1], "n_real": len(real[key]), "n_null": len(null[key]),
               "group": "IA" if ("IA" in key[0] or "IA" in key[1]) else "noIA",
               "fallback": key in fallback}
        for column, is_ppl in METRICS:
            r = [value(x, column, is_ppl) for x in real[key].values()]
            n = [value(x, column, is_ppl) for x in null[key].values()]
            r, n = [x for x in r if x is not None], [x for x in n if x is not None]
            if not r or not n:
                continue
            rec[f"{column}_excess"] = statistics.mean(r) - statistics.mean(n)
            se2 = (statistics.variance(r) / len(r) if len(r) > 1 else 0.0) + \
                  (statistics.variance(n) / len(n) if len(n) > 1 else 0.0)
            rec[f"{column}_se"] = math.sqrt(se2)
        if "wikitext2_ppl_excess" in rec:
            out.append(rec)
    return out


def pairs_of(records):
    pairs = {}
    for r in records:
        pairs.setdefault(frozenset((r["mask"], r["other"])), []).append(r)
    return pairs


def boot_ci(units, stat, n, rng):
    """units: list of lists of per-direction values; resamples units."""
    draws = []
    for _ in range(n):
        sample = [units[rng.randrange(len(units))] for _ in units]
        draws.append(stat(sample))
    draws.sort()
    return draws[int(0.05 * n)], draws[int(0.95 * n)], draws[int(0.025 * n)], draws[int(0.975 * n)]


def pooled(records, column, margins, n_boot, rng):
    pairs = list(pairs_of(records).values())
    units = [[r[f"{column}_excess"] for r in p] for p in pairs]
    flat = [v for u in units for v in u]
    mean = statistics.mean(flat)
    stat = lambda sample: statistics.mean([v for u in sample for v in u])
    lo90, hi90, lo95, hi95 = boot_ci(units, stat, n_boot, rng)
    # Signed means can hide antisymmetric effects (+x in one direction, -x in the other), so
    # equivalence is judged on the mean ABSOLUTE excess, not on the signed mean.
    abs_stat = lambda sample: statistics.mean([abs(v) for u in sample for v in u])
    abs_mean = abs_stat(units)
    _, abs_hi90, _, _ = boot_ci(units, abs_stat, n_boot, rng)
    # model-level bootstrap: resample models, keep pairs whose both models are drawn
    models = sorted({m for p in pairs for r in p[:1] for m in (r["mask"], r["other"])})
    pair_models = [(set((p[0]["mask"], p[0]["other"])), [r[f"{column}_excess"] for r in p]) for p in pairs]
    mdraws = []
    for _ in range(n_boot):
        drawn = [models[rng.randrange(len(models))] for _ in models]
        counts = {}
        for m in drawn:
            counts[m] = counts.get(m, 0) + 1
        vals = []
        for ms, ex in pair_models:
            w = 1
            for m in ms:
                w *= counts.get(m, 0)
            vals += ex * w if w else []
        if vals:
            mdraws.append(statistics.mean(vals))
    mdraws.sort()
    m_lo, m_hi = (mdraws[int(0.05 * len(mdraws))], mdraws[int(0.95 * len(mdraws))]) if mdraws else (None, None)
    res = {"n_dirs": len(flat), "n_pairs": len(pairs), "mean": mean,
           "sd_dirs": statistics.pstdev(flat), "mean_abs": abs_mean, "mean_abs_ci95_upper_pair": abs_hi90,
           "ci90_pair": (lo90, hi90), "ci95_pair": (lo95, hi95),
           "ci90_model": (m_lo, m_hi), "max_abs": max(abs(v) for v in flat)}
    res["equivalent_at"] = {str(mg): abs_hi90 < mg for mg in margins}
    res["dirs_within"] = {str(mg): sum(abs(v) < mg for v in flat) for mg in margins}
    return res


def main(argv=None):
    a = parse_args(argv)
    rng = random.Random(a.seed)
    records = load_directions(a.results_dirs)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in records for k in r}, key=lambda k: (k not in ("mask", "other", "group"), k))
    with (out / "pooled_directions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    summary = {}
    for group, recs in (("noIA", [r for r in records if r["group"] == "noIA"]),
                        ("IA", [r for r in records if r["group"] == "IA"]),
                        ("all", records)):
        if not recs:
            continue
        summary[group] = {}
        for column, _ in METRICS:
            summary[group][column] = pooled(recs, column, a.margins, a.bootstrap, rng)
    antis = []
    for key, p in pairs_of(records).items():
        if len(p) == 2:
            antis.append({"pair": "x".join(sorted(key)), "group": p[0]["group"],
                          "sum": p[0]["wikitext2_ppl_excess"] + p[1]["wikitext2_ppl_excess"],
                          "abs_each": sorted(abs(r["wikitext2_ppl_excess"]) for r in p)})
    summary["antisymmetry"] = antis
    summary["fallback_directions"] = [f"{r['mask']}<-{r['other']}" for r in records if r["fallback"]]
    (out / "pooled_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    L = ["# RQ3 pooled analysis: shared-gene exchange minus deficit-matched null", "",
         "excess = mean(real) - mean(null), log WikiText-2 PPL (0.01 = about 1% perplexity) and calibration KL.",
         "Equivalence is judged on the mean ABSOLUTE excess: a signed mean lets opposite directions cancel.",
         "CI: bootstrap over pairs (both directions resampled together); model-level CI resamples final models.", ""]
    for group in ("noIA", "IA", "all"):
        if group not in summary:
            continue
        for column, _ in METRICS:
            s = summary[group][column]
            L.append(f"**{group} / {column}**: {s['n_dirs']} directions, {s['n_pairs']} pairs; mean excess "
                     f"{s['mean']:+.4f}; 90% CI (pairs) [{s['ci90_pair'][0]:+.4f}, {s['ci90_pair'][1]:+.4f}]; "
                     f"90% CI (models) [{s['ci90_model'][0]:+.4f}, {s['ci90_model'][1]:+.4f}]; "
                     f"mean |excess| {s['mean_abs']:.4f} (95% upper bound {s['mean_abs_ci95_upper_pair']:.4f}); "
                     f"largest |excess| {s['max_abs']:.4f}; directions within margin: " +
                     ", ".join(f"±{m}: {n}/{s['n_dirs']}" for m, n in s["dirs_within"].items()) +
                     "; mean |excess| upper bound below margin: " +
                     ", ".join(f"±{m}: {'yes' if v else 'no'}" for m, v in s["equivalent_at"].items()))
            L.append("")
    L += ["## Direction pairs: sum of the two directions (W2 log PPL)", "",
          "| Pair | group | sum | |excess| (each) |", "| --- | --- | ---: | --- |"]
    for r in sorted(antis, key=lambda r: (r["group"], r["pair"])):
        L.append(f"| {r['pair']} | {r['group']} | {r['sum']:+.4f} | "
                 f"{r['abs_each'][0]:.4f} / {r['abs_each'][1]:.4f} |")
    if summary["fallback_directions"]:
        L += ["", "Directions whose real job fell back to repair scope 'all': " +
              ", ".join(summary["fallback_directions"])]
    (out / "pooled_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"directions={len(records)} pairs={len(pairs_of(records))}")


if __name__ == "__main__":
    main()
