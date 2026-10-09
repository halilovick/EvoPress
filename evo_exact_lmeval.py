#!/usr/bin/env python3
"""Zero-shot downstream evaluation of exact-budget models with LM-eval.

Executes the jobs of a plan written by ``scripts/plan_exact_lmeval.py``. The
model is loaded once; every job assembles one compressed model with the code of
``evo_exact_replay.py`` (the same candidate construction, budget validation and
weight loading that reproduced the recorded perplexities exactly) and evaluates
it with the LM Evaluation Harness on the in-memory model.

The LM-eval settings default to those of the screening LM-eval runs
(``lmeval.py``, ``scripts/run_mistral_lmeval_comparison.sh``): tasks
``arc_easy``, ``piqa`` and ``winogrande``, ``num_fewshot = 0``, batch size 4,
16-bit weights, SDPA attention (the transformers default used there), the fast
tokenizer of the base model, no sample limit and the harness's default seeds.
The settings are written to ``settings.json``; a rerun with different settings
into the same output directory is refused, so all models of one directory are
evaluated identically.

Order within one process: the dense job first (before any quantized weights are
loaded), then the quantized jobs. Completed jobs are skipped on a rerun unless
``--overwrite`` is given. ``--limit`` is for smoke tests only and is only
accepted for an output directory whose name contains ``smoke``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from evo_exact_replay import build_candidate, select_jobs
from evo_joint_attribution import (
    build_depth_module_names,
    initialize_model_quant_state,
    json_safe,
    prepare_model,
    quant_groups,
)
from evo_joint_search import apply_joint_state, load_drop_state
from src.common_utils import fix_seed
from src.compression_budget import uniform_quantization_target_cost
from src.exact_lmeval import (
    check_settings,
    incomplete_tasks,
    require_smoke_dir_for_limit,
    settings_of,
    task_scores,
    write_csv,
)
from src.exact_replay import load_final_candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot LM-eval of exact-budget models.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--quant_db", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--jobs", nargs="*", default=None, help="Run only these job ids.")
    parser.add_argument("--tasks", default="arc_easy,piqa,winogrande")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=float, default=None, help="Smoke tests only.")
    parser.add_argument("--target_bitwidth", type=int, default=3)
    parser.add_argument("--quantization_group_size", type=int, default=128)
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--zero_point_bits", type=int, default=16)
    parser.add_argument("--expected_target_cost_bits", type=int, default=26982023168)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build and validate candidates, no evaluation.")
    args = parser.parse_args(argv)
    args.use_fast_tokenizer = True  # read by prepare_model
    return args


def make_lm(model, args):
    """Wrap the in-memory model in the harness's HF model class."""
    import inspect

    from lm_eval.models.huggingface import HFLM

    kwargs = {"pretrained": model, "tokenizer": args.base_model, "batch_size": args.batch_size,
              "backend": "causal", "use_fast_tokenizer": True}
    parameters = inspect.signature(HFLM.__init__).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in parameters}
    return HFLM(**kwargs)


def evaluate(lm, args, tasks: Sequence[str]) -> dict[str, Any]:
    from lmeval import make_task_manager, simple_evaluate_compat

    task_manager = make_task_manager(argparse.Namespace(verbosity="INFO", include_path=None))
    return simple_evaluate_compat(
        task_manager,
        model=lm,
        tasks=list(tasks),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    require_smoke_dir_for_limit(output_dir, args.limit)
    tasks = [t for t in args.tasks.split(",") if t]
    fix_seed(args.seed)
    root = Path(args.repo_root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    sources = {label: load_final_candidate(root / rel) for label, rel in plan["sources"].items()}

    try:
        import lm_eval
    except ImportError as exc:  # pragma: no cover - environment check
        raise SystemExit(f"lm_eval is not installed: {exc}") from exc
    try:
        from importlib.metadata import version

        lm_eval_version = version("lm_eval")
    except Exception:  # noqa: BLE001
        lm_eval_version = getattr(lm_eval, "__version__", "unknown")

    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)
    settings = settings_of(args, lm_eval_version)
    check_settings(output_dir, settings, args.overwrite)
    (output_dir / "plan_used.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    jobs = select_jobs(plan, argparse.Namespace(tiers=None, jobs=args.jobs))
    pending = []
    for job in jobs:
        result_path = output_dir / "jobs" / job["id"] / "result.json"
        if result_path.is_file() and not args.overwrite:
            if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "completed":
                continue
        pending.append(job)
    print(f"Jobs selected: {len(jobs)}; pending: {len(pending)}; lm_eval {lm_eval_version}", flush=True)
    if not pending:
        write_csv(output_dir, plan)
        return

    model, _tokenizer, layers = prepare_model(args)
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
    lm = None if args.dry_run else make_lm(model, args)

    for job in pending:
        job_dir = output_dir / "jobs" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result: dict[str, Any] = {"job": job, "target_bits": target, "settings": settings}
        try:
            if (job.get("repair") or {}).get("scope", "none") != "none":
                raise ValueError("LM-eval jobs must not be repaired.")
            candidate, details = build_candidate(job, args, ctx)
            details.pop("bits", None)
            result["details"] = details
            if not args.dry_run:
                if job["precision"] == "fp16":
                    if any(level is not None for group in model.state for level in group):
                        raise RuntimeError("The dense job must run before any quantized weights are loaded.")
                    load_drop_state(model, layers, candidate["drop"])
                else:
                    apply_joint_state(model, layers, grouped, candidate, args.quant_db)
                raw = evaluate(lm, args, tasks)
                raw.pop("samples", None)
                (job_dir / "lmeval_results.json").write_text(
                    json.dumps(json_safe(raw), indent=2) + "\n", encoding="utf-8")
                scores = task_scores(raw, tasks)
                if args.limit is None and incomplete_tasks(scores):
                    raise RuntimeError(f"Incomplete task sets: {incomplete_tasks(scores)}")
                result["scores"] = scores
            result["status"] = "completed" if not args.dry_run else "validated"
        except Exception as exc:  # noqa: BLE001 - record and continue with the next job.
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.time() - start
        (job_dir / "result.json").write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        summary = {t: round(s["score"], 4) for t, s in (result.get("scores") or {}).items()}
        print(f"[{result['status']}] {job['id']} {summary} {result.get('error', '')} "
              f"({result['seconds']:.0f} s)", flush=True)
        write_csv(output_dir, plan)

    write_csv(output_dir, plan)


if __name__ == "__main__":
    main()
