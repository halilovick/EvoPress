"""Torch-free helpers for the zero-shot LM-eval of exact-budget models.

Used by ``evo_exact_lmeval.py`` (GPU) and ``scripts/summarize_exact_lmeval.py``
(CPU); tested in ``tests/test_exact_lmeval_pure.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

# Full evaluation sets of the harness's default splits (ARC-Easy test, PIQA and
# WinoGrande validation), as recorded by the screening LM-eval runs.
FULL_TASK_SIZES = {"arc_easy": 2376, "piqa": 1838, "winogrande": 1267}
# Same preference as scripts/summarize_mistral_lmeval_comparison.py.
PREFERRED_METRICS = ("acc_norm,none", "acc,none")
CSV_COLUMNS = (
    "id", "task", "metric", "score", "stderr", "acc", "acc_stderr", "acc_norm", "acc_norm_stderr",
    "n_samples", "limit", "removed_attn", "removed_mlp", "cost_bits", "bits_sha256", "seconds",
)
# Keys that must agree between runs written to the same output directory.
SETTINGS_KEYS = (
    "base_model", "tasks", "num_fewshot", "batch_size", "limit", "dtype",
    "attn_implementation", "tokenizer", "lm_eval_version", "target_bitwidth",
)


def require_smoke_dir_for_limit(output_dir: Path, limit: float | None) -> None:
    if limit is not None and "smoke" not in Path(output_dir).name:
        raise SystemExit("--limit is for smoke tests only; use an --output_dir whose name contains 'smoke'.")


def settings_of(args: argparse.Namespace, lm_eval_version: str) -> dict[str, Any]:
    return {
        "base_model": args.base_model,
        "tasks": [t for t in args.tasks.split(",") if t],
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "tokenizer": f"{args.base_model} (fast)",
        "lm_eval_version": lm_eval_version,
        "quant_db": str(args.quant_db),
        "target_bitwidth": args.target_bitwidth,
    }


def check_settings(output_dir: Path, settings: Mapping[str, Any], overwrite: bool) -> None:
    """Refuse to mix evaluation settings within one output directory."""
    path = Path(output_dir) / "settings.json"
    if path.is_file() and not overwrite:
        stored = json.loads(path.read_text(encoding="utf-8"))
        diff = {k: (stored.get(k), settings.get(k)) for k in SETTINGS_KEYS if stored.get(k) != settings.get(k)}
        if diff:
            raise SystemExit(f"Settings differ from {path}: {diff}. Use a new --output_dir.")
    path.write_text(json.dumps(dict(settings), indent=2) + "\n", encoding="utf-8")


def _float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def task_scores(results: Mapping[str, Any], tasks: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-task scores from a harness result dictionary.

    The reported score is ``acc_norm`` where the task defines it, otherwise
    ``acc`` (the rule of the screening LM-eval summary); both are kept.
    """
    scores = {}
    n_samples = results.get("n-samples", {})
    for task in tasks:
        metrics = results["results"][task]
        metric = next(m for m in PREFERRED_METRICS if m in metrics)
        scores[task] = {
            "metric": metric.split(",")[0],
            "score": _float(metrics, metric),
            "stderr": _float(metrics, metric.replace(",", "_stderr,", 1)),
            "acc": _float(metrics, "acc,none"),
            "acc_stderr": _float(metrics, "acc_stderr,none"),
            "acc_norm": _float(metrics, "acc_norm,none"),
            "acc_norm_stderr": _float(metrics, "acc_norm_stderr,none"),
            "n_samples": int((n_samples.get(task) or {}).get("effective", -1)),
        }
    return scores


def incomplete_tasks(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Tasks whose evaluated sample count differs from the full set."""
    return {
        task: int(entry["n_samples"])
        for task, entry in scores.items()
        if task in FULL_TASK_SIZES and int(entry["n_samples"]) != FULL_TASK_SIZES[task]
    }


def csv_rows(output_dir: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for job in plan["jobs"]:
        path = Path(output_dir) / "jobs" / job["id"] / "result.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        details = result.get("details", {})
        for task, entry in result["scores"].items():
            rows.append({
                "id": job["id"], "task": task, "metric": entry["metric"], "score": entry["score"],
                "stderr": entry["stderr"], "acc": entry["acc"], "acc_stderr": entry["acc_stderr"],
                "acc_norm": entry["acc_norm"], "acc_norm_stderr": entry["acc_norm_stderr"],
                "n_samples": entry["n_samples"], "limit": result["settings"]["limit"],
                "removed_attn": json.dumps(details.get("removed_attn", [])),
                "removed_mlp": json.dumps(details.get("removed_mlp", [])),
                "cost_bits": details.get("cost_bits", ""), "bits_sha256": details.get("bits_sha256", ""),
                "seconds": round(float(result.get("seconds", 0.0)), 1),
            })
    return rows


def write_csv(output_dir: Path, plan: Mapping[str, Any]) -> None:
    with (Path(output_dir) / "lmeval_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows(output_dir, plan))
