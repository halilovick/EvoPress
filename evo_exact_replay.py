#!/usr/bin/env python3
"""Evaluation-only replays under the exact (group-wise) storage budget.

Executes the jobs of a plan written by ``scripts/plan_exact_replays.py``. Each
job assembles one compressed model (a mask and a bit-width assignment), repairs
it to the exact budget with the production repair of the searches where
needed, validates the realized storage cost against the target, and evaluates
held-out perplexity (and, optionally, the calibration KL divergence to the dense
model, i.e. the search objective on the full calibration set).

No search is performed. Evaluation settings default to those of the
paper-matched final evaluations (WikiText-2 and C4, 524,288 evaluation tokens,
sequence length 8,192; FineWeb-Edu calibration data).

Job order within one process: block-influence scores (if a job needs them),
then 16-bit depth-only jobs (dense weights), then quantized jobs. Completed jobs
are skipped on a rerun unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import trange

from evo_joint_attribution import (
    build_depth_module_names,
    initialize_model_quant_state,
    json_safe,
    normalize_eval_datasets,
    prepare_model,
    quant_groups,
)
from evo_joint_search import apply_joint_state, load_drop_state
from src.common_utils import fix_seed
from src.compression_budget import (
    CompressionBudgetError,
    _repair_delta_within_groups,
    candidate_compression_cost,
    repair_quant_state_to_budget,
    uniform_quantization_target_cost,
    validate_exact_budget,
)
from src.data_utils import get_data
from src.exact_replay import (
    build_replay_bits,
    heuristic_mask,
    level_deficits,
    load_final_candidate,
    module_is_active,
    normalize_drop,
    score_mask,
)
from src.metrics import compute_kl_div, compute_perplexity
from src.teacher_logits_cache import DiskTensorCache

RESULT_COLUMNS = (
    "id", "tier", "rq", "status", "precision", "mask_label", "bits_label", "fill", "repair_scope",
    "repair_seed", "pre_repair_deficits", "repair_changed_genes", "repair_levels_moved",
    "wikitext2_ppl", "c4_ppl", "wikitext2_nll", "c4_nll", "calibration_kl",
    "cost_bits", "target_bits", "removed_attn", "removed_mlp", "bits_sha256", "seconds", "error",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-budget evaluation-only replays.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--quant_db", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tiers", nargs="*", default=None, help="Run only these tiers.")
    parser.add_argument("--jobs", nargs="*", default=None, help="Run only these job ids.")
    parser.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--eval_tokens", type=int, default=524288)
    parser.add_argument("--eval_sequence_length", type=int, default=8192)
    parser.add_argument("--calibration_data", default="fineweb_edu")
    parser.add_argument("--calibration_tokens", type=int, default=524288)
    parser.add_argument("--calibration_sequence_length", type=int, default=8192)
    parser.add_argument(
        "--calibration_kl",
        action="store_true",
        help="Also compute KL to the dense model on the full calibration set "
        "(caches dense logits on disk, about 34 GB for 524,288 tokens).",
    )
    parser.add_argument("--bi_tokens", type=int, default=131072, help="Calibration tokens for block-influence scores.")
    parser.add_argument("--target_bitwidth", type=int, default=3)
    parser.add_argument("--quantization_group_size", type=int, default=128)
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--zero_point_bits", type=int, default=16)
    parser.add_argument("--expected_target_cost_bits", type=int, default=26982023168)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build, repair and validate candidates without evaluation.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Candidate construction.
# ---------------------------------------------------------------------------


def to_quant_state(grouped_layer_names, bits: Mapping[str, int]) -> list[list[int]]:
    return [[int(bits[name]) for name in group] for group in grouped_layer_names]


def to_bits(grouped_layer_names, quant_state) -> dict[str, int]:
    return {
        name: int(level)
        for group, levels in zip(grouped_layer_names, quant_state)
        for name, level in zip(group, levels)
    }


def bits_hash(bits: Mapping[str, int]) -> str:
    payload = json.dumps(sorted(bits.items()), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_mask(job, sources, num_layers, bi_scores) -> dict[str, list[bool]]:
    spec = job["mask"]
    if "source" in spec:
        return normalize_drop(sources[spec["source"]]["drop"])
    rule = spec["rule"]
    if rule == "none":
        return {"attn": [False] * num_layers, "mlp": [False] * num_layers}
    if rule == "bi_score":
        if bi_scores is None:
            raise RuntimeError("Block-influence scores were not computed.")
        return score_mask(bi_scores, int(spec["k_attn"]), int(spec["k_mlp"]))
    return heuristic_mask(rule, num_layers, int(spec["k_attn"]), int(spec["k_mlp"]), int(spec.get("seed", 0)))


def repair_exclusive(model, grouped_layer_names, quant_db, quant_state, drop, eligibility, rng, reference_bitwidth, mu):
    """Group-wise repair restricted to the eligible (exclusive) sublayers.

    Uses the production subset-sum repair ``_repair_delta_within_groups`` with an
    eligibility drop state; the target level sums are those of the real mask.
    """
    repaired = copy.deepcopy(quant_state)
    bits = to_bits(grouped_layer_names, repaired)
    deficits = level_deficits(grouped_layer_names, bits, drop, reference_bitwidth, mu)
    for group_id, (group, delta) in enumerate(zip(grouped_layer_names, deficits)):
        if delta == 0:
            continue
        size = model.get_submodule(group[0]).weight.numel()
        _repair_delta_within_groups(
            model, grouped_layer_names, quant_db, repaired, eligibility,
            int(delta) * int(size), {group_id}, rng,
        )
    return repaired


def build_candidate(job, args, ctx) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (candidate, details) with the realized cost already validated."""
    from fractions import Fraction

    grouped = ctx["grouped_layer_names"]
    mu = Fraction(args.scale_bits + args.zero_point_bits, args.quantization_group_size)
    drop = resolve_mask(job, ctx["sources"], ctx["num_layers"], ctx.get("bi_scores"))
    details: dict[str, Any] = {
        "removed_attn": [i for i, v in enumerate(drop["attn"]) if v],
        "removed_mlp": [i for i, v in enumerate(drop["mlp"]) if v],
    }
    if job["precision"] == "fp16":
        details.update({"repair_changed_genes": 0, "repair_levels_moved": 0})
        return {"drop": drop, "quant": None}, details

    bits, eligibility = build_replay_bits(job, drop, ctx["sources"], grouped, args.target_bitwidth, mu)
    missing = [name for group in grouped for name in group if name not in bits]
    if missing:
        raise ValueError(f"Bit-width assignment misses {len(missing)} modules, e.g. {missing[:3]}")
    pre_deficits = level_deficits(grouped, bits, drop, args.target_bitwidth, mu)
    details["pre_repair_deficits"] = pre_deficits
    state = to_quant_state(grouped, bits)
    repair = job.get("repair") or {"scope": "none", "seed": 0}
    rng = random.Random(int(repair.get("seed", 0)))
    scope = repair.get("scope", "none")
    if any(pre_deficits):
        if scope == "none":
            raise CompressionBudgetError(f"Job {job['id']} needs repair but its repair scope is 'none'.")
        if scope == "exclusive":
            state = repair_exclusive(model=ctx["model"], grouped_layer_names=grouped, quant_db=args.quant_db,
                                     quant_state=state, drop=drop, eligibility=eligibility, rng=rng,
                                     reference_bitwidth=args.target_bitwidth, mu=mu)
        else:
            state = repair_quant_state_to_budget(
                ctx["model"], grouped, args.quant_db, state, drop, ctx["target_cost_bits"],
                preserve_equal_size_group_costs=True, uniform_reference_bitwidth=args.target_bitwidth,
                rng=rng, **ctx["cost_kwargs"],
            )
    repaired_bits = to_bits(grouped, state)
    changed = {
        name: [bits[name], repaired_bits[name]]
        for name in repaired_bits
        if repaired_bits[name] != bits[name]
    }
    candidate = {"drop": drop, "quant": state}
    cost = candidate_compression_cost(ctx["model"], candidate, grouped_layer_names=grouped, **ctx["cost_kwargs"])
    validate_exact_budget(cost, ctx["target_cost_bits"], context=f"replay job {job['id']}")
    details.update(
        {
            "repair_changed_genes": len(changed),
            "repair_levels_moved": sum(abs(b - a) for a, b in changed.values()),
            "repair_changes": changed,
            "repair_changed_active_only": all(module_is_active(name, drop) for name in changed),
            "cost_bits": int(cost["total_cost_bits"]),
            "bits_sha256": bits_hash(repaired_bits),
            "bits": repaired_bits,
        }
    )
    return candidate, details


# ---------------------------------------------------------------------------
# Block-influence scores (ShortGPT-style, adapted to sublayers).
# ---------------------------------------------------------------------------


@torch.no_grad()
def block_influence_scores(model, layers, data) -> dict[str, list[float]]:
    """``1 - mean cos`` between the residual stream before and after each sublayer.

    Attention: input of the decoder layer vs. input of the post-attention norm.
    MLP: input of the post-attention norm vs. output of the decoder layer.
    Computed on the dense model.
    """
    num = len(layers)
    sums = {"attn": [0.0] * num, "mlp": [0.0] * num}
    count = 0
    cache: dict[int, dict[str, torch.Tensor]] = {}
    handles = []

    def layer_pre(i):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            cache.setdefault(i, {})["in"] = hidden.detach()
        return hook

    def norm_pre(i):
        def hook(module, args):
            cache.setdefault(i, {})["mid"] = args[0].detach()
        return hook

    def layer_post(i):
        def hook(module, args, kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            entry = cache.pop(i)
            cos_attn = torch.nn.functional.cosine_similarity(entry["in"].float(), entry["mid"].float(), dim=-1)
            cos_mlp = torch.nn.functional.cosine_similarity(entry["mid"].float(), out.detach().float(), dim=-1)
            sums["attn"][i] += float((1.0 - cos_attn).sum())
            sums["mlp"][i] += float((1.0 - cos_mlp).sum())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(layer_pre(i), with_kwargs=True))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(norm_pre(i)))
        handles.append(layer.register_forward_hook(layer_post(i), with_kwargs=True))
    device = next(model.parameters()).device
    try:
        for sample in data:
            model(sample.to(device))
            count += sample.numel()
    finally:
        for handle in handles:
            handle.remove()
    return {kind: [value / count for value in values] for kind, values in sums.items()}


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def select_jobs(plan, args) -> list[dict[str, Any]]:
    jobs = plan["jobs"]
    if args.tiers:
        jobs = [job for job in jobs if job["tier"] in set(args.tiers)]
    if args.jobs:
        jobs = [job for job in jobs if job["id"] in set(args.jobs)]
    fp16 = [job for job in jobs if job["precision"] == "fp16"]
    quant = [job for job in jobs if job["precision"] != "fp16"]
    return fp16 + quant


def result_row(job, result) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    details = result.get("details") or {}
    repair = job.get("repair") or {}
    bits_spec = job.get("bits") or {}
    row = {
        "id": job["id"], "tier": job["tier"], "rq": job["rq"], "status": result["status"],
        "precision": job["precision"],
        "mask_label": job["mask"].get("source") or job["mask"].get("rule"),
        "bits_label": bits_spec.get("source") or bits_spec.get("rule", ""),
        "fill": bits_spec.get("fill", ""), "repair_scope": repair.get("scope", ""),
        "repair_seed": repair.get("seed", ""),
        "pre_repair_deficits": json.dumps(details.get("pre_repair_deficits", "")),
        "repair_changed_genes": details.get("repair_changed_genes", ""),
        "repair_levels_moved": details.get("repair_levels_moved", ""),
        "wikitext2_ppl": metrics.get("wikitext2", ""), "c4_ppl": metrics.get("c4", ""),
        "wikitext2_nll": math.log(metrics["wikitext2"]) if metrics.get("wikitext2") else "",
        "c4_nll": math.log(metrics["c4"]) if metrics.get("c4") else "",
        "calibration_kl": metrics.get("calibration_kl", ""),
        "cost_bits": details.get("cost_bits", ""), "target_bits": result.get("target_bits", ""),
        "removed_attn": json.dumps(details.get("removed_attn", [])),
        "removed_mlp": json.dumps(details.get("removed_mlp", [])),
        "bits_sha256": details.get("bits_sha256", ""), "seconds": result.get("seconds", ""),
        "error": result.get("error", ""),
    }
    return row


def write_results(output_dir: Path, plan) -> None:
    rows = []
    for job in plan["jobs"]:
        path = output_dir / "jobs" / job["id"] / "result.json"
        if path.is_file():
            rows.append(result_row(job, json.loads(path.read_text(encoding="utf-8"))))
    with (output_dir / "replay_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.eval_datasets = normalize_eval_datasets(args.eval_datasets)
    fix_seed(args.seed)
    root = Path(args.repo_root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (output_dir / "plan_used.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    sources = {label: load_final_candidate(root / rel) for label, rel in plan["sources"].items()}

    jobs = select_jobs(plan, args)
    pending = []
    for job in jobs:
        result_path = output_dir / "jobs" / job["id"] / "result.json"
        if result_path.is_file() and not args.overwrite:
            if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "completed":
                continue
        pending.append(job)
    print(f"Jobs selected: {len(jobs)}; pending: {len(pending)}")
    if not pending:
        write_results(output_dir, plan)
        return

    # prepare_model reads base_model/dtype/attn_implementation/use_fast_tokenizer.
    model, tokenizer, layers = prepare_model(args)
    _, grouped = quant_groups(model, Path(args.quant_db), "size")
    initialize_model_quant_state(model, grouped)
    attention_names, mlp_names = build_depth_module_names(model, layers)
    cost_kwargs = {
        "attention_module_names": attention_names, "mlp_module_names": mlp_names,
        "dense_dtype_bits": 16, "group_size": args.quantization_group_size,
        "include_quantization_metadata": True,
        "scale_bits": args.scale_bits, "zero_point_bits": args.zero_point_bits,
    }
    target = int(uniform_quantization_target_cost(model, grouped, args.target_bitwidth, **cost_kwargs)["total_cost_bits"])
    if args.expected_target_cost_bits and target != args.expected_target_cost_bits:
        raise ValueError(f"Target cost {target} differs from the expected {args.expected_target_cost_bits}.")
    ctx = {
        "model": model, "grouped_layer_names": grouped, "sources": sources,
        "num_layers": len(layers), "cost_kwargs": cost_kwargs, "target_cost_bits": target,
    }

    calibration = None
    if args.calibration_kl or any(job["mask"].get("rule") == "bi_score" for job in pending):
        calibration = get_data(args.calibration_data, args.calibration_tokens,
                               args.calibration_sequence_length, tokenizer, train=True)
    if any(job["mask"].get("rule") == "bi_score" for job in pending):
        subset, used = [], 0
        for sample in calibration:
            if used >= args.bi_tokens:
                break
            subset.append(sample)
            used += sample.numel()
        ctx["bi_scores"] = block_influence_scores(model, layers, subset)
        (output_dir / "block_influence_scores.json").write_text(
            json.dumps({"tokens": used, "scores": ctx["bi_scores"]}, indent=2) + "\n", encoding="utf-8")

    target_logits = None
    if args.calibration_kl:
        target_logits = DiskTensorCache.temporary(prefix="evopress-replay-teacher-logits")
        device = next(model.parameters()).device
        for i in trange(len(calibration), desc="Dense calibration logits", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration[i].to(device)).logits)

    eval_data = {
        name: get_data(name, args.eval_tokens, args.eval_sequence_length, tokenizer, train=False)
        for name in args.eval_datasets
    }

    for job in pending:
        job_dir = output_dir / "jobs" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result: dict[str, Any] = {"job": job, "target_bits": target}
        try:
            candidate, details = build_candidate(job, args, ctx)
            result["details"] = details
            metrics: dict[str, float] = {}
            if not args.dry_run:
                if job["precision"] == "fp16":
                    if any(level is not None for group in model.state for level in group):
                        raise RuntimeError("16-bit jobs must run before any quantized weights are loaded.")
                    load_drop_state(model, layers, candidate["drop"])
                else:
                    apply_joint_state(model, layers, grouped, candidate, args.quant_db)
                for name, data in eval_data.items():
                    metrics[name] = float(compute_perplexity(model, data))
                if target_logits is not None:
                    metrics["calibration_kl"] = float(compute_kl_div(model, calibration, target_logits))
            result["metrics"] = metrics
            result["status"] = "completed" if not args.dry_run else "validated"
        except Exception as exc:  # noqa: BLE001 - record and continue with the next job.
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.time() - start
        (job_dir / "result.json").write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[{result['status']}] {job['id']} {result.get('metrics', {})} {result.get('error', '')}", flush=True)
        write_results(output_dir, plan)

    if target_logits is not None:
        target_logits.cleanup()
    write_results(output_dir, plan)


if __name__ == "__main__":
    main()
