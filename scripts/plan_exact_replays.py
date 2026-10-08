#!/usr/bin/env python3
"""Plan evaluation-only replays under the exact (group-wise) storage budget.

This script needs no GPU, model weights or PyTorch. It reads the final
candidates of completed runs, analyses which crossed candidates satisfy the
group-wise budget without repair, and writes a replay plan for
``evo_exact_replay.py``. Nothing is evaluated here.

Outputs (in ``--output_dir``):
  pair_analysis.csv / .md   feasibility of every crossed pair, with and without
                            owner fill
  replay_plan.json          the jobs, grouped in tiers
  replay_plan.md            human-readable job list
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_replay import (  # noqa: E402
    build_replay_bits,
    crossed_pair_report,
    fill_bits,
    level_deficits,
    load_final_candidate,
    repair_capacity,
    mistral_module_sizes,
    size_groups,
    validate_group_budget,
)

RAW = "thesis_results/datalab_raw"

# Completed full-space final candidates (exact budget), keyed by thesis label.
FULLSPACE_CANDIDATES = {
    # 25% structural sparsity
    "E3s0": f"{RAW}/paper_matched/joint/joint_seed0_attempt1_20260831/final_candidate.json",
    "E3s1": f"{RAW}/paper_matched/joint/joint_seed1_attempt3_resume_20260902/final_candidate.json",
    "E3s2": f"{RAW}/paper_matched/joint/joint_seed2_attempt1_20260903/final_candidate.json",
    "IAs0": f"{RAW}/fullspace_extensions/fullspace_joint_s025_ia_g150_seed0_attempt1/final_candidate.json",
    "IAs1": f"{RAW}/fullspace_extensions/fullspace_joint_s025_ia_g150_seed1_attempt1/final_candidate.json",
    "IAs2": f"{RAW}/fullspace_extensions/fullspace_joint_s025_ia_g150_seed2_attempt1/final_candidate.json",
    "P4s0": f"{RAW}/fullspace_extensions/fullspace_joint_s025_pop4_g150_seed0_attempt1/final_candidate.json",
    "P4s1": f"{RAW}/fullspace_extensions/fullspace_joint_s025_pop4_g150_seed1_attempt1/final_candidate.json",
    "DWs0": f"{RAW}/fullspace_extensions/fullspace_depthwarm_g150_seed0_attempt1/final_candidate.json",
    "LXs0": f"{RAW}/fullspace_extensions/fullspace_joint_s025_localxover_pop4_g150_seed0_attempt1/final_candidate.json",
    # 12.5% structural sparsity
    "J12s0": f"{RAW}/fullspace_extensions/fullspace_joint_s0125_g150_seed0_attempt1/final_candidate.json",
    "J12s1": f"{RAW}/fullspace_extensions/fullspace_joint_s0125_g150_seed1_attempt1/final_candidate.json",
    "J12s2": f"{RAW}/fullspace_extensions/fullspace_joint_s0125_g150_seed2_attempt1/final_candidate.json",
    # Quantization-only search (no removal)
    "E2s0": f"{RAW}/paper_matched/quant_only/quant_only_seed0_attempt12_resume_20260826/final_candidate.json",
    "E2s1": f"{RAW}/paper_matched/quant_only/quant_only_seed1_attempt4_resume_20260828/final_candidate.json",
    "E2s2": f"{RAW}/paper_matched/quant_only/quant_only_seed2_attempt2_resume_20260830/final_candidate.json",
}

# Independently searched 16-bit depth-only masks (cheap protocol: WikiText2,
# 8,192 calibration tokens, G20/O16). Used only as masks.
DEPTH_ONLY_MASKS = {
    "DO25s0": "results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json",
    "DO25s1": "results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1/final_candidate.json",
    "DO25s2": "results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/final_candidate.json",
}
# Optional 12.5% depth-only masks produced by stage 0 of
# scripts/run_exact_replays.sh with the same cheap protocol.
DEPTH_ONLY_MASKS_125 = {
    f"DO12s{seed}": f"results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed{seed}/final_candidate.json"
    for seed in range(3)
}

SPARSITY_OF = {
    **{k: 0.25 for k in ("E3s0", "E3s1", "E3s2", "IAs0", "IAs1", "IAs2", "P4s0", "P4s1", "DWs0", "LXs0")},
    **{k: 0.125 for k in ("J12s0", "J12s1", "J12s2")},
    **{k: 0.0 for k in ("E2s0", "E2s1", "E2s2")},
}

REPAIR_SEEDS = (0, 1, 2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--include_depth_only_125",
        action="store_true",
        help="Add 12.5%% independent-composition jobs (requires stage-0 depth masks).",
    )
    return parser.parse_args(argv)


def load_sources(root: Path, include_125: bool) -> dict[str, dict[str, Any]]:
    sources = {}
    registry = dict(FULLSPACE_CANDIDATES)
    registry.update(DEPTH_ONLY_MASKS)
    if include_125:
        registry.update(DEPTH_ONLY_MASKS_125)
    for label, rel in registry.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate {label}: {path}")
        cand = load_final_candidate(path)
        cand["source"] = rel
        cand["label"] = label
        sources[label] = cand
    return sources


def pair_analysis(sources, groups) -> list[dict[str, Any]]:
    rows = []
    searched = [k for k in FULLSPACE_CANDIDATES if SPARSITY_OF[k] > 0]
    for a, b in itertools.combinations(searched, 2):
        if SPARSITY_OF[a] != SPARSITY_OF[b]:
            continue
        report = crossed_pair_report(a, sources[a], b, sources[b], groups)
        ab_owner = level_deficits(groups, fill_bits(sources[a], sources[b], sources[a]["drop"], groups, "owner"), sources[a]["drop"])
        ba_owner = level_deficits(groups, fill_bits(sources[b], sources[a], sources[b]["drop"], groups, "owner"), sources[b]["drop"])
        report.update(
            {
                "sparsity": SPARSITY_OF[a],
                "owner_deficits_DA_QB": ab_owner,
                "owner_deficits_DB_QA": ba_owner,
                "owner_repair_free_pair": not any(ab_owner) and not any(ba_owner),
                "owner_abs_levels": sum(abs(x) for x in ab_owner + ba_owner),
                "donor_abs_levels": sum(abs(x) for x in report["deficits_DA_QB"] + report["deficits_DB_QA"]),
            }
        )
        rows.append(report)
    return rows


def job(job_id, tier, rq, purpose, mask, bits=None, repair=None, precision="quantized", pair=None):
    entry = {
        "id": job_id,
        "tier": tier,
        "rq": rq,
        "purpose": purpose,
        "mask": mask,
        "precision": precision,
    }
    if bits is not None:
        entry["bits"] = bits
    if repair is not None:
        entry["repair"] = repair
    if pair is not None:
        entry["pair"] = pair
    return entry


def crossed_jobs(a, b, fill, scope, tier, purpose, deficits_ab, deficits_ba):
    """Jobs for (D_A, Q_B) and (D_B, Q_A); repaired jobs get several seeds."""
    jobs = []
    for mask_label, bit_label, deficits in ((a, b, deficits_ab), (b, a, deficits_ba)):
        seeds = (0,) if not any(deficits) else REPAIR_SEEDS
        for seed in seeds:
            jobs.append(
                job(
                    f"x_{fill}_{scope}_{mask_label}_Q{bit_label}_r{seed}",
                    tier,
                    "RQ3",
                    purpose,
                    {"source": mask_label},
                    {"source": bit_label, "fill": fill},
                    {"scope": scope if any(deficits) else "none", "seed": seed},
                    pair={"a": a, "b": b, "mask": mask_label, "bits": bit_label, "fill": fill},
                )
            )
    return jobs


def build_plan(sources, groups, analysis, include_125: bool) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    own = ["E3s0", "E3s1", "E3s2", "J12s0", "J12s1", "J12s2", "DWs0"]
    # Tier 0: re-evaluate finals in the replay harness (reproduction check and
    # the J(D_A, Q_A) terms of every contrast).
    for label in own:
        jobs.append(job(f"own_{label}", "0", "control", "re-evaluate completed final candidate",
                        {"source": label}, {"source": label, "fill": "donor"}, {"scope": "none", "seed": 0}))
    jobs.append(job("own_uniform3", "0", "control", "re-evaluate E1 uniform 3-bit",
                    {"rule": "none"}, {"rule": "uniform", "bitwidth": 3}, {"scope": "none", "seed": 0}))

    by_pair = {(r["a"], r["b"]): r for r in analysis}
    # Tier A: pairs that are repair-free under owner fill (shared-gene exchange).
    free = [r for r in analysis if r["owner_repair_free_pair"]]
    for r in free:
        jobs += crossed_jobs(r["a"], r["b"], "owner", "exclusive", "A",
                             "shared-gene exchange, repair-free", r["owner_deficits_DA_QB"], r["owner_deficits_DB_QA"])
    # Same pairs with plain component crossing (definition 3.10), production repair.
    for r in free:
        jobs += crossed_jobs(r["a"], r["b"], "donor", "all", "A",
                             "plain crossing (D_A, Q_B) of the same pair", r["deficits_DA_QB"], r["deficits_DB_QA"])
    # Tier B: remaining same-variant seed pairs (E3, 12.5%), owner fill,
    # repair restricted to the mask's exclusive genes, three repair seeds.
    for a, b in (("E3s0", "E3s1"), ("E3s0", "E3s2"), ("E3s1", "E3s2"), ("J12s0", "J12s2"), ("J12s1", "J12s2")):
        r = by_pair[(a, b)]
        if r["owner_repair_free_pair"]:
            continue
        jobs += crossed_jobs(a, b, "owner", "exclusive", "B", "shared-gene exchange, minimal repair",
                             r["owner_deficits_DA_QB"], r["owner_deficits_DB_QA"])

    # Tier C: RQ1 baselines under the exact budget.
    for sparsity, k in ((0.25, 8), (0.125, 4)):
        tag = "25" if sparsity == 0.25 else "12"
        rules = [("late_layer", 0), ("late_layer_keep_last", 0), ("random", 0), ("random", 1), ("random", 2), ("bi_score", 0)]
        for rule, seed in rules:
            mask = {"rule": rule, "k_attn": k, "k_mlp": k, "seed": seed}
            suffix = f"{rule}" + (f"_s{seed}" if rule == "random" else "")
            jobs.append(job(f"h{tag}_{suffix}_nu", "C", "RQ1", "heuristic mask + near-uniform precision",
                            mask, {"rule": "near_uniform", "order": "spread"}, {"scope": "none", "seed": 0}))
    # Independent composition: depth-only mask x quantization-only profile.
    compositions = [("DO25s0", "E2s0"), ("DO25s1", "E2s1"), ("DO25s2", "E2s2")]
    if include_125:
        compositions += [("DO12s0", "E2s0"), ("DO12s1", "E2s1"), ("DO12s2", "E2s2")]
    for mask_label, bit_label in compositions:
        jobs.append(job(f"ind_{mask_label}_Q{bit_label}_shift", "C", "RQ1",
                        "independent composition: depth-only mask, quantization-only profile shifted to budget",
                        {"source": mask_label}, {"source": bit_label, "fill": "donor", "transform": "shift"},
                        {"scope": "all", "seed": 0}))
        jobs.append(job(f"ind_{mask_label}_nu", "C", "RQ1", "depth-only mask + near-uniform precision",
                        {"source": mask_label}, {"rule": "near_uniform", "order": "spread"}, {"scope": "none", "seed": 0}))
    # Allocation attribution: joint masks with alternative precision.
    for mask_label in ("E3s0", "E3s1", "E3s2", "J12s0", "J12s1", "J12s2"):
        e2 = "E2s" + mask_label[-1]
        jobs.append(job(f"att_{mask_label}_nu", "C", "RQ1/RQ3", "joint mask + near-uniform precision",
                        {"source": mask_label}, {"rule": "near_uniform", "order": "spread"}, {"scope": "none", "seed": 0}))
        jobs.append(job(f"att_{mask_label}_Q{e2}_shift", "C", "RQ1/RQ3", "joint mask + quantization-only profile shifted",
                        {"source": mask_label}, {"source": e2, "fill": "donor", "transform": "shift"}, {"scope": "all", "seed": 0}))

    # Tier D: measured 16-bit depth-only references (not at the budget).
    fp16_masks = ["E3s0", "E3s1", "E3s2", "J12s0", "J12s1", "J12s2", "DO25s0", "DO25s1", "DO25s2"]
    if include_125:
        fp16_masks += ["DO12s0", "DO12s1", "DO12s2"]
    for mask_label in fp16_masks:
        jobs.append(job(f"fp16_{mask_label}", "D", "RQ1 reference", "16-bit depth-only evaluation of a mask",
                        {"source": mask_label}, precision="fp16"))
    for tag, k in (("25", 8), ("12", 4)):
        for rule in ("late_layer_keep_last", "bi_score"):
            jobs.append(job(f"fp16_h{tag}_{rule}", "D", "RQ1 reference", "16-bit heuristic depth-only evaluation",
                            {"rule": rule, "k_attn": k, "k_mlp": k, "seed": 0}, precision="fp16"))

    # Static validation of every job whose bits can be built without the model.
    for entry in jobs:
        mask = entry["mask"]
        if "source" in mask and entry["precision"] == "quantized":
            drop = sources[mask["source"]]["drop"]
            bits, eligibility = build_replay_bits(entry, drop, sources, groups)
            deficits = level_deficits(groups, bits, drop)
            entry["static_deficits"] = deficits
            if eligibility is not None and not all(repair_capacity(groups, bits, eligibility, deficits)):
                # The mask's exclusive genes cannot absorb the correction;
                # fall back to the production repair over all active genes.
                entry["repair"]["scope"] = "all"
                entry["repair"]["fallback_from"] = "exclusive"
                entry["id"] = entry["id"].replace("_exclusive_", "_all_")

    return {
        "version": 1,
        "reference_bitwidth": 3,
        "levels": [2, 3, 4, 5, 6],
        "repair_seeds": list(REPAIR_SEEDS),
        "sources": {label: cand["source"] for label, cand in sources.items()},
        "jobs": jobs,
    }


def write_outputs(out: Path, analysis, plan) -> None:
    out.mkdir(parents=True, exist_ok=True)
    columns = [
        "sparsity", "a", "b", "jaccard", "attn_shared", "mlp_shared", "hamming_mask", "hamming_bits",
        "active_in_both_bits_differ", "deficits_DA_QB", "deficits_DB_QA", "repair_free_pair",
        "owner_deficits_DA_QB", "owner_deficits_DB_QA", "owner_repair_free_pair",
        "donor_abs_levels", "owner_abs_levels",
    ]
    with (out / "pair_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in analysis:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})
    lines = [
        "# Crossed-pair feasibility under the group-wise exact budget",
        "",
        "Deficits are the level corrections (k/v, q/o, MLP group) that a repair must apply.",
        "`donor`: plain crossing (D_A, Q_B). `owner`: shared-gene exchange (donor bits only on",
        "projections active under both masks).",
        "",
        "| s | A | B | Jaccard | donor (D_A,Q_B) | donor (D_B,Q_A) | owner (D_A,.) | owner (D_B,.) | owner repair-free |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(analysis, key=lambda r: (r["sparsity"], r["owner_abs_levels"], r["donor_abs_levels"])):
        lines.append(
            f"| {r['sparsity']} | {r['a']} | {r['b']} | {r['jaccard']:.2f} | {r['deficits_DA_QB']} | "
            f"{r['deficits_DB_QA']} | {r['owner_deficits_DA_QB']} | {r['owner_deficits_DB_QA']} | "
            f"{'yes' if r['owner_repair_free_pair'] else 'no'} |"
        )
    n_free = sum(r["repair_free_pair"] for r in analysis)
    n_owner = sum(r["owner_repair_free_pair"] for r in analysis)
    lines += ["", f"Pairs analysed: {len(analysis)}. Repair-free under plain crossing: {n_free}. "
              f"Repair-free under shared-gene exchange: {n_owner}."]
    (out / "pair_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "replay_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    tiers = {}
    for entry in plan["jobs"]:
        tiers.setdefault(entry["tier"], []).append(entry)
    md = ["# Exact-budget replay plan", "", f"Jobs: {len(plan['jobs'])}", ""]
    for tier in sorted(tiers):
        md += [f"## Tier {tier} ({len(tiers[tier])} jobs)", "", "| Job | RQ | Purpose | Repair |", "| --- | --- | --- | --- |"]
        for entry in tiers[tier]:
            repair = entry.get("repair") or {}
            md.append(f"| `{entry['id']}` | {entry['rq']} | {entry['purpose']} | {repair.get('scope', '-')} |")
        md.append("")
    (out / "replay_plan.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    root = Path(args.repo_root)
    groups = size_groups(mistral_module_sizes())
    sources = load_sources(root, args.include_depth_only_125)
    for label in FULLSPACE_CANDIDATES:
        validate_group_budget(groups, sources[label]["bits"], sources[label]["drop"])
    analysis = pair_analysis(sources, groups)
    plan = build_plan(sources, groups, analysis, args.include_depth_only_125)
    write_outputs(Path(args.output_dir), analysis, plan)
    print(f"pairs={len(analysis)} owner_repair_free={sum(r['owner_repair_free_pair'] for r in analysis)} "
          f"jobs={len(plan['jobs'])}")


if __name__ == "__main__":
    main()
