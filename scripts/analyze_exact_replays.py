#!/usr/bin/env python3
"""Thesis tables from the exact-budget replay batch (CPU only, standard library).

Reads ``replay_results.csv`` of a replay run and the full-space ledger and writes
``analysis/`` next to the results:

  rq1_baselines.md / .csv     joint search vs heuristic, independent and
                              allocation-swapped candidates at the same budget,
                              plus 16-bit references
  rq3_contrasts.md / .csv     interaction contrasts per pair and crossing type
  reproduction.md             replayed finals vs recorded ledger values

Metrics: WikiText-2 / C4 perplexity, their mean NLL (log perplexity), and the
calibration KL divergence (the search objective on all 524,288 calibration
tokens). Contrasts follow I = (J_AA + J_BB) - (J_AB + J_BA) = -(delta_A + delta_B).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_replay import interaction_terms  # noqa: E402

LEDGER_OF_OWN = {
    "own_E3s0": ("E3_joint_s025_standard_g150", "0"),
    "own_E3s1": ("E3_joint_s025_standard_g150", "1"),
    "own_E3s2": ("E3_joint_s025_standard_g150", "2"),
    "own_J12s0": ("joint_s0125_g150", "0"),
    "own_J12s1": ("joint_s0125_g150", "1"),
    "own_J12s2": ("joint_s0125_g150", "2"),
    "own_DWs0": ("depthwarm_g150", "0"),
    "own_uniform3": ("E1_uniform3", "0"),
}


def load(path):
    return {row["id"]: row for row in csv.DictReader(open(path, encoding="utf-8"))}


def f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else None)


def fmt(mean, sd=None, digits=3):
    if mean is None:
        return "–"
    text = f"{mean:.{digits}f}"
    return text + (f" ± {sd:.{digits}f}" if sd is not None else "")


def group(rows, ids, label, sparsity, note=""):
    present = [rows[i] for i in ids if i in rows]
    out = {"label": label, "s": sparsity, "n": len(present), "note": note}
    for key in ("wikitext2_ppl", "c4_ppl", "calibration_kl"):
        mean, sd = stats([f(r, key) for r in present])
        out[key] = mean
        out[key + "_sd"] = sd
        out[key + "_values"] = [f(r, key) for r in present]
    return out


def rq1_table(rows, ledger):
    def ledger_group(condition, label, s):
        lrows = [r for r in ledger if r["condition"] == condition]
        out = {"label": label, "s": s, "n": len(lrows), "note": "ledger (search run)"}
        for key, lkey in (("wikitext2_ppl", "wikitext2_ppl"), ("c4_ppl", "c4_ppl"), ("calibration_kl", "final_calibration_kl")):
            vals = [float(r[lkey]) for r in lrows if r[lkey] not in ("", None)]
            mean, sd = stats(vals)
            out[key], out[key + "_sd"], out[key + "_values"] = mean, sd, vals
        return out

    seeds = (0, 1, 2)
    table = [
        ledger_group("E0_dense", "Dense 16-bit (E0)", "–"),
        group(rows, ["own_uniform3"], "Uniform 3-bit (E1), replayed", 0.0),
        ledger_group("E2_quant_only_g150", "Quantization-only search (E2)", 0.0),
        # s = 0.125
        group(rows, [f"own_J12s{i}" for i in seeds], "Joint search (J12), replayed", 0.125),
        group(rows, [f"att_J12s{i}_QE2s{i}_shift" for i in seeds], "Joint masks + E2 profile shifted", 0.125),
        group(rows, [f"att_J12s{i}_nu" for i in seeds], "Joint masks + near-uniform", 0.125),
        group(rows, ["h12_bi_score_nu"], "Block-influence mask + near-uniform", 0.125),
        group(rows, [f"h12_random_s{i}_nu" for i in seeds], "Random masks + near-uniform", 0.125),
        group(rows, ["h12_late_layer_nu"], "Last sublayers + near-uniform", 0.125),
        group(rows, ["h12_late_layer_keep_last_nu"], "Last sublayers before final + near-uniform", 0.125),
        # s = 0.25
        group(rows, [f"own_E3s{i}" for i in seeds], "Joint search (E3), replayed", 0.25),
        group(rows, [f"att_E3s{i}_QE2s{i}_shift" for i in seeds], "Joint masks + E2 profile shifted", 0.25),
        group(rows, [f"att_E3s{i}_nu" for i in seeds], "Joint masks + near-uniform", 0.25),
        group(rows, [f"ind_DO25s{i}_QE2s{i}_shift" for i in seeds], "Independent: depth-only masks + E2 profile shifted", 0.25),
        group(rows, [f"ind_DO25s{i}_nu" for i in seeds], "Depth-only masks + near-uniform", 0.25),
        group(rows, ["h25_bi_score_nu"], "Block-influence mask + near-uniform", 0.25),
        group(rows, [f"h25_random_s{i}_nu" for i in seeds], "Random masks + near-uniform", 0.25),
        group(rows, ["h25_late_layer_nu"], "Last sublayers + near-uniform", 0.25),
        group(rows, ["h25_late_layer_keep_last_nu"], "Last sublayers before final + near-uniform", 0.25),
    ]
    fp16 = [
        group(rows, [f"fp16_J12s{i}" for i in seeds], "Joint masks, 16-bit", 0.125, "not at budget"),
        group(rows, ["fp16_h12_bi_score"], "Block-influence mask, 16-bit", 0.125, "not at budget"),
        group(rows, ["fp16_h12_late_layer_keep_last"], "Last sublayers before final, 16-bit", 0.125, "not at budget"),
        group(rows, [f"fp16_E3s{i}" for i in seeds], "Joint masks, 16-bit", 0.25, "not at budget"),
        group(rows, [f"fp16_DO25s{i}" for i in seeds], "Depth-only masks, 16-bit", 0.25, "not at budget"),
        group(rows, ["fp16_h25_bi_score"], "Block-influence mask, 16-bit", 0.25, "not at budget"),
        group(rows, ["fp16_h25_late_layer_keep_last"], "Last sublayers before final, 16-bit", 0.25, "not at budget"),
    ]
    return table, fp16


def big(mean, sd, values):
    """Use per-value listing when the spread is huge (random masks)."""
    if mean is None:
        return "–"
    if len(values) > 1 and max(values) > 3 * min(values):
        return " / ".join(f"{v:.4g}" for v in values)
    return fmt(mean, sd, 3 if mean < 100 else 1)


def write_rq1(out, table, fp16):
    lines = [
        "# RQ1 at the exact budget (T = 26,982,023,168 bits)",
        "",
        "Mean ± population SD over the listed candidates; individual values when they span more than a factor of 3.",
        "W2 = WikiText-2 test PPL, C4 = C4 validation PPL (length 8,192); KL = calibration KL to the dense model.",
        "",
        "| s | Configuration | n | W2 PPL | C4 PPL | Calib. KL |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    rows_csv = []
    for block in (table, fp16):
        for g in block:
            lines.append(
                f"| {g['s']} | {g['label']}{' (' + g['note'] + ')' if g['note'] and 'ledger' not in g['note'] else ''} | {g['n']} | "
                f"{big(g['wikitext2_ppl'], g['wikitext2_ppl_sd'], g['wikitext2_ppl_values'])} | "
                f"{big(g['c4_ppl'], g['c4_ppl_sd'], g['c4_ppl_values'])} | "
                f"{fmt(g['calibration_kl'], g['calibration_kl_sd'], 4)} |"
            )
            rows_csv.append({k: v for k, v in g.items() if not k.endswith("_values")})
        if block is table:
            lines += ["", "16-bit references (same masks, all active projections in 16-bit; not at the budget):", "",
                      "| s | Configuration | n | W2 PPL | C4 PPL | Calib. KL |", "| --- | --- | ---: | --- | --- | --- |"]
    (out / "rq1_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "rq1_baselines.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_csv)


PAIRS = [
    # (a, b, fill, scope label, tier)
    ("E3s1", "DWs0", "owner", "repair-free", "A"),
    ("J12s0", "J12s1", "owner", "repair-free", "A"),
    ("E3s1", "DWs0", "donor", "plain, production repair", "A"),
    ("J12s0", "J12s1", "donor", "plain, production repair", "A"),
    ("E3s0", "E3s1", "owner", "exclusive repair", "B"),
    ("E3s0", "E3s2", "owner", "exclusive repair", "B"),
    ("E3s1", "E3s2", "owner", "exclusive repair", "B"),
    ("J12s0", "J12s2", "owner", "exclusive repair", "B"),
    ("J12s1", "J12s2", "owner", "exclusive repair", "B"),
]


def crossed_ids(rows, mask, bits, fill):
    prefix = f"x_{fill}_"
    return [i for i in rows if i.startswith(prefix) and i.split("_r")[0].endswith(f"_{mask}_Q{bits}")]


def metric(row, key):
    value = f(row, key)
    if value is None:
        return None
    return math.log(value) if key.endswith("_ppl") else value


def rq3_table(rows):
    records = []
    for a, b, fill, label, tier in PAIRS:
        ab = crossed_ids(rows, a, b, fill)
        ba = crossed_ids(rows, b, a, fill)
        rec = {"a": a, "b": b, "crossing": "shared-gene exchange" if fill == "owner" else "plain (D_A, Q_B)",
               "repair": label, "tier": tier, "repair_seeds": len(ab),
               "repaired_genes": max(int(rows[i]["repair_changed_genes"] or 0) for i in ab + ba)}
        for key in ("wikitext2_ppl", "c4_ppl", "calibration_kl"):
            j_aa = metric(rows[f"own_{a}"], key)
            j_bb = metric(rows[f"own_{b}"], key)
            v_ab = [metric(rows[i], key) for i in ab]
            v_ba = [metric(rows[i], key) for i in ba]
            terms = interaction_terms(j_aa, j_bb, statistics.mean(v_ab), statistics.mean(v_ba))
            name = {"wikitext2_ppl": "w2_nll", "c4_ppl": "c4_nll", "calibration_kl": "kl"}[key]
            rec[f"{name}_I"] = terms["I"]
            rec[f"{name}_dA"] = terms["delta_a"]
            rec[f"{name}_dB"] = terms["delta_b"]
            rec[f"{name}_repair_range"] = max(max(v) - min(v) for v in (v_ab, v_ba))
        records.append(rec)
    return records


def write_rq3(out, records):
    lines = [
        "# RQ3: interaction contrasts at the exact budget",
        "",
        "I = (J_AA + J_BB) - (J_AB + J_BA) = -(delta_A + delta_B); J = mean NLL (log PPL) or calibration KL; lower is better.",
        "Negative I: each mask does better with its own allocation than additivity predicts.",
        "Repaired crossings use the mean over 3 repair seeds; 'range' is the largest spread over repair seeds.",
        "",
        "| Pair | Crossing | Repair (max genes) | W2 NLL I (δA, δB) | C4 NLL I | KL I (δA, δB) | range W2 / KL |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r['a']} × {r['b']} | {r['crossing']} | {r['repair']} ({r['repaired_genes']}) | "
            f"{r['w2_nll_I']:+.4f} ({r['w2_nll_dA']:+.4f}, {r['w2_nll_dB']:+.4f}) | {r['c4_nll_I']:+.4f} | "
            f"{r['kl_I']:+.4f} ({r['kl_dA']:+.4f}, {r['kl_dB']:+.4f}) | "
            f"{r['w2_nll_repair_range']:.4f} / {r['kl_repair_range']:.4f} |"
        )
    (out / "rq3_contrasts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "rq3_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_reproduction(out, rows, ledger):
    index = {(r["condition"], r["seed"]): r for r in ledger}
    lines = ["# Reproduction of completed finals in the replay harness", "",
             "| Job | W2 replay / recorded | C4 replay / recorded | KL replay / recorded |", "| --- | --- | --- | --- |"]
    for job, key in LEDGER_OF_OWN.items():
        a, b = rows[job], index[key]
        lines.append(f"| {job} | {a['wikitext2_ppl']} / {b['wikitext2_ppl']} | {a['c4_ppl']} / {b['c4_ppl']} | "
                     f"{a['calibration_kl']} / {b['final_calibration_kl'] or '–'} |")
    (out / "reproduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", default="results/exact_replays/replay_20261008")
    parser.add_argument("--ledger", default="thesis_results/fullspace_ledger.csv")
    args = parser.parse_args(argv)
    results = Path(args.results_dir)
    rows = load(results / "replay_results.csv")
    ledger = list(csv.DictReader(open(args.ledger, encoding="utf-8")))
    out = results / "analysis"
    out.mkdir(exist_ok=True)
    table, fp16 = rq1_table(rows, ledger)
    write_rq1(out, table, fp16)
    write_rq3(out, rq3_table(rows))
    write_reproduction(out, rows, ledger)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
