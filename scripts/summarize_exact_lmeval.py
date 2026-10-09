#!/usr/bin/env python3
"""Verify and summarize the zero-shot LM-eval of the full-space final models.

CPU only. Reads ``<results_dir>/jobs/*/result.json`` written by
``evo_exact_lmeval.py`` and writes ``<results_dir>/analysis/``:

  verification.md   checks: completeness, settings, full task sets, realized
                    storage cost, bit-width hashes against the replay batch,
                    dense model against the screening LM-eval of the same model
  lmeval_by_model.csv / .md   per-model scores and mean +- population SD per
                    method (E0, E1, E2, J12, E3)

Exit code 1 if a verification check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_lmeval import FULL_TASK_SIZES, incomplete_tasks, task_scores  # noqa: E402

TARGET_BITS = 26982023168
TASKS = ("arc_easy", "piqa", "winogrande")
METHODS = (
    ("E0", "Dense, 16-bit", ["dense"]),
    ("E1", "Uniform 3-bit", ["own_uniform3"]),
    ("E2", "Quantization-only search", ["own_E2s0", "own_E2s1", "own_E2s2"]),
    ("J12", "Joint search, s = 0.125", ["own_J12s0", "own_J12s1", "own_J12s2"]),
    ("E3", "Joint search, s = 0.25", ["own_E3s0", "own_E3s1", "own_E3s2"]),
)
# Six independent compositions (depth-only mask of the screening protocol + E2
# profile shifted to the budget), the same candidates as in the replay batches.
METHODS_INDEPENDENT = (
    ("IND12", "Independent composition, s = 0.125", [f"ind_DO12s{i}_QE2s{i}_shift" for i in range(3)]),
    ("IND25", "Independent composition, s = 0.25", [f"ind_DO25s{i}_QE2s{i}_shift" for i in range(3)]),
)
SETS = {
    "fullspace": {"methods": METHODS, "prefix": "own_",
                  "removed": {"E0": (0, 0), "E1": (0, 0), "E2": (0, 0), "J12": (4, 4), "E3": (8, 8)}},
    "independent": {"methods": METHODS_INDEPENDENT, "prefix": "ind_",
                    "removed": {"IND12": (4, 4), "IND25": (8, 8)}},
}
DEFAULT_REPLAY_DIRS = ["results/exact_replays/replay_20261008", "results/exact_replays/replay_depth125"]
DEFAULT_REFERENCE_SETTINGS = "results/exact_lmeval/fullspace_lmeval/settings.json"
DEFAULT_REPLAY_DIR = "results/exact_replays/replay_20261008"
DEFAULT_SCREENING_DENSE = "results/runs/lmeval_dense_mistral_tasks_seed0_retry4/lmeval_results.json"
DENSE_TOLERANCE = 0.01  # absolute; two screening runs of the dense model differ by up to 0.0005


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((results_dir / "jobs").glob("*/result.json")):
        out[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def replay_hashes(replay_dir: Path, prefix: str = "own_") -> dict[str, str]:
    hashes = {}
    for path in (replay_dir / "jobs").glob(f"{prefix}*/result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        value = (data.get("details") or {}).get("bits_sha256")
        if value:
            hashes[path.parent.name] = value
    return hashes


def verify(results: dict[str, dict[str, Any]], replay: dict[str, str], screening_dense: dict[str, Any] | None,
           methods=METHODS, want=None, reference_settings: dict[str, Any] | None = None, require_replay=False):
    checks: list[tuple[str, bool, str]] = []
    want = want or SETS["fullspace"]["removed"]
    expected = [job for _, _, jobs in methods for job in jobs]
    missing = [j for j in expected if results.get(j, {}).get("status") != "completed"]
    n = len(expected)
    checks.append((f"all {n} jobs completed", not missing,
                   f"missing or failed: {missing}" if missing else f"{n}/{n}"))
    limits = {j: r["settings"].get("limit") for j, r in results.items() if r.get("status") == "completed"}
    checks.append(("no sample limit", all(v is None for v in limits.values()),
                   ", ".join(f"{j}={v}" for j, v in limits.items() if v is not None) or "none"))
    settings = {json.dumps({k: v for k, v in r["settings"].items() if k != "quant_db"}, sort_keys=True)
                for r in results.values() if r.get("status") == "completed"}
    checks.append(("identical settings for all jobs", len(settings) <= 1, f"{len(settings)} distinct"))
    if reference_settings is not None:
        ref = {k: v for k, v in reference_settings.items() if k != "quant_db"}
        differ = sorted({k for r in results.values() if r.get("status") == "completed"
                         for k in ref if r["settings"].get(k) != ref[k]})
        checks.append(("settings identical to the 11-model LM-eval", not differ,
                       f"differing keys: {differ}" if differ else "identical"))
    short = {j: incomplete_tasks(r["scores"]) for j, r in results.items()
             if r.get("status") == "completed" and incomplete_tasks(r["scores"])}
    checks.append(("full task sets " + str(FULL_TASK_SIZES), not short, str(short) if short else "all full"))
    bad_cost = {j: r["details"].get("cost_bits") for j, r in results.items()
                if r.get("status") == "completed" and j != "dense" and r["details"].get("cost_bits") != TARGET_BITS}
    checks.append((f"realized cost = {TARGET_BITS:,} bits", not bad_cost, str(bad_cost) if bad_cost else "all quantized jobs"))
    compared, mismatched = [], []
    for job, digest in replay.items():
        if job in results and results[job].get("status") == "completed":
            compared.append(job)
            if results[job]["details"].get("bits_sha256") != digest:
                mismatched.append(job)
    not_compared = [j for j in expected if j not in compared and j.startswith(("own_", "ind_"))
                    and results.get(j, {}).get("status") == "completed"]
    ok = not mismatched and (not require_replay or not not_compared)
    checks.append(("bit-widths identical to the replay batch", ok,
                   f"compared {len(compared)}: {sorted(compared)}; mismatched: {mismatched}"
                   + (f"; without replay record: {not_compared}" if require_replay and not_compared else "")))
    removed = {}
    for code, _, jobs in methods:
        for job in jobs:
            if job in results and results[job].get("status") == "completed":
                d = results[job]["details"]
                removed[job] = (len(d.get("removed_attn", [])), len(d.get("removed_mlp", [])))
    wrong = {j: v for code, _, jobs in methods for j in jobs if j in removed and removed[j] != want[code]}
    checks.append(("removed sublayers (attn, MLP) as expected", not wrong, str(wrong) if wrong else "ok"))
    if screening_dense is not None and results.get("dense", {}).get("status") == "completed":
        ref = task_scores(screening_dense, TASKS)
        ours = results["dense"]["scores"]
        diffs = {t: round(ours[t]["score"] - ref[t]["score"], 4) for t in TASKS}
        ok = all(abs(v) <= DENSE_TOLERANCE for v in diffs.values())
        checks.append((f"dense vs screening dense LM-eval (|diff| <= {DENSE_TOLERANCE})", ok, str(diffs)))
    return checks


def summarize(results: dict[str, dict[str, Any]], methods=METHODS):
    per_model, per_method = [], []
    for code, label, jobs in methods:
        done = [j for j in jobs if results.get(j, {}).get("status") == "completed"]
        for job in done:
            scores = results[job]["scores"]
            row = {"method": code, "id": job}
            for task in TASKS:
                row[task] = scores[task]["score"]
                row[f"{task}_stderr"] = scores[task]["stderr"]
                row[f"{task}_metric"] = scores[task]["metric"]
            row["mean3"] = st.mean(scores[t]["score"] for t in TASKS)
            per_model.append(row)
        if not done:
            continue
        agg = {"method": code, "label": label, "n": len(done)}
        for key in (*TASKS, "mean3"):
            values = [r[key] for r in per_model if r["method"] == code]
            agg[key] = st.mean(values)
            agg[f"{key}_sd"] = st.pstdev(values) if len(values) > 1 else 0.0
        per_method.append(agg)
    return per_model, per_method


def verify_candidates(results, replay, methods, want):
    """Checks of a dry run (status 'validated'): candidates only, no scores."""
    expected = [job for _, _, jobs in methods for job in jobs]
    status = {j: results.get(j, {}).get("status") for j in expected}
    bad = {j: v for j, v in status.items() if v != "validated"}
    checks = [(f"all {len(expected)} candidates built and validated", not bad, str(bad) if bad else "ok")]
    cost = {j: results[j]["details"].get("cost_bits") for j in expected
            if j in results and results[j].get("details") and results[j]["details"].get("cost_bits") != TARGET_BITS}
    checks.append((f"realized cost = {TARGET_BITS:,} bits", not cost, str(cost) if cost else "all"))
    mismatched = [j for j in expected if j in results and results[j].get("details")
                  and results[j]["details"].get("bits_sha256") != replay.get(j)]
    checks.append(("bit-widths identical to the replay batches", not mismatched,
                   f"mismatched or without replay record: {mismatched}" if mismatched else "all"))
    wrong = {}
    for code, _, jobs in methods:
        for j in jobs:
            d = (results.get(j) or {}).get("details") or {}
            got = (len(d.get("removed_attn", [])), len(d.get("removed_mlp", [])))
            if got != want[code]:
                wrong[j] = got
    checks.append(("removed sublayers (attn, MLP) as expected", not wrong, str(wrong) if wrong else "ok"))
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--set", default="fullspace", choices=sorted(SETS),
                        help="fullspace: the 11 models E0-E3; independent: the 6 independent compositions.")
    parser.add_argument("--replay_dir", nargs="+", default=None,
                        help="Replay batches with the reference bit-width hashes.")
    parser.add_argument("--screening_dense", default=DEFAULT_SCREENING_DENSE)
    parser.add_argument("--reference_settings", default=DEFAULT_REFERENCE_SETTINGS)
    parser.add_argument("--dry_run", action="store_true", help="Check a dry run (candidates only).")
    args = parser.parse_args(argv)
    spec = SETS[args.set]
    methods, want = spec["methods"], spec["removed"]
    replay_dirs = args.replay_dir or ([DEFAULT_REPLAY_DIR] if args.set == "fullspace" else DEFAULT_REPLAY_DIRS)
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    replay = {}
    for d in replay_dirs:
        if Path(d).is_dir():
            for job, digest in replay_hashes(Path(d), spec["prefix"]).items():
                if job in replay and replay[job] != digest:
                    raise SystemExit(f"Replay batches disagree on {job}.")
                replay[job] = digest
    if args.dry_run:
        checks = verify_candidates(results, replay, methods, want)
        for name, ok, detail in checks:
            print(f"- [{'x' if ok else ' '}] {name}: {detail}")
        return 0 if all(ok for _, ok, _ in checks) else 1
    dense_path = Path(args.screening_dense)
    screening_dense = json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.is_file() else None
    reference = None
    if args.set != "fullspace" and Path(args.reference_settings).is_file():
        reference = json.loads(Path(args.reference_settings).read_text(encoding="utf-8"))
    checks = verify(results, replay, screening_dense, methods=methods, want=want,
                    reference_settings=reference, require_replay=args.set != "fullspace")
    if args.set != "fullspace" and reference is None:
        checks.append(("settings identical to the 11-model LM-eval", False, "reference settings not found"))
    per_model, per_method = summarize(results, methods)

    out = results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# LM-eval verification", ""] + [
        f"- [{'x' if ok else ' '}] {name}: {detail}" for name, ok, detail in checks
    ]
    (out / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if per_model:
        with (out / "lmeval_by_model.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_model[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(per_model)
    md = [f"# Zero-shot accuracy (%) ({args.set} set)", "",
          "ARC-Easy and PIQA: acc_norm; WinoGrande: acc. Mean +- population SD over seeds.", "",
          "| Method | n | ARC-Easy | PIQA | WinoGrande | Mean |", "| --- | ---: | --- | --- | --- | --- |"]
    for agg in per_method:
        cells = [f"{100 * agg[k]:.2f} ± {100 * agg[k + '_sd']:.2f}" if agg["n"] > 1 else f"{100 * agg[k]:.2f}"
                 for k in (*TASKS, "mean3")]
        md.append(f"| {agg['label']} ({agg['method']}) | {agg['n']} | " + " | ".join(cells) + " |")
    n_methods = len(md)
    md += ["", "Per model:", "", "| Model | ARC-Easy | PIQA | WinoGrande | Mean |", "| --- | --- | --- | --- | --- |"]
    for row in per_model:
        md.append(f"| {row['id']} | " + " | ".join(f"{100 * row[k]:.2f}" for k in (*TASKS, "mean3")) + " |")
    (out / "lmeval_by_model.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print()
    print("\n".join(md[:n_methods]))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
