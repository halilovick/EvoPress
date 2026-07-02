#!/usr/bin/env python3
"""Replay-based attribution for joint depth-pruning + quantization candidates.

This script recombines a depth mask from one EvoPress run with a quantization
assignment from another run, replays the combined compressed model, and writes
per-combination metrics. It is intended as an analysis tool, not as a search
algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evo_joint_search import (
    apply_joint_state,
    candidate_bits,
    get_layer_drop_config,
    quantizable_weights,
    repair_active_quant_budget,
)
from src.common_utils import fix_seed
from src.data_utils import get_data
from src.metrics import compute_perplexity
from src.model_utils import (
    dummy_initialize,
    get_attn_layer_name,
    get_layers,
    get_mlp_layer_name,
    group_layers,
    layer_order_fn,
)
from src.run_reporting import (
    build_depth_details,
    build_final_candidate,
    compute_compression_metrics,
    flatten_quant_state,
    module_name,
)


MATRIX_COLUMNS = (
    "depth_label",
    "quant_label",
    "depth_source",
    "quant_source",
    "wiki2_ppl",
    "c4_ppl",
    "fineweb_ppl",
    "compression_ratio",
    "active_params",
    "avg_active_bitwidth",
    "budget_valid_before_repair",
    "budget_repaired",
    "repair_num_changes",
    "status",
    "error_message",
)


DATASET_ALIASES = {
    "wiki2": "wikitext2",
    "fineweb": "fineweb_edu",
}


@dataclass
class CandidateSource:
    label: str
    source: str
    drop: dict[str, list[bool]] | None = None
    quant_state: list[list[int]] | None = None
    bitwidth_by_module: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
    candidate_type: str | None = None
    is_uniform: bool = False
    uniform_bits: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recombine EvoPress depth masks and quantization assignments, "
            "replay the resulting compressed model, and write attribution "
            "metrics."
        )
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--quant_db", required=True)
    parser.add_argument("--output_dir", required=True)

    # Single recombination mode.
    parser.add_argument("--depth_source")
    parser.add_argument("--quant_source")
    parser.add_argument("--depth_label", default="depth")
    parser.add_argument("--quant_label", default="quant")

    # Batch attribution-matrix mode.
    parser.add_argument(
        "--depth_sources",
        nargs="+",
        help="Batch sources in label=path form.",
    )
    parser.add_argument(
        "--quant_sources",
        nargs="+",
        help="Batch sources in label=path or label=uniform:BITS form.",
    )

    parser.add_argument("--eval_datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--eval_tokens", type=int, default=131072)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--group_rule", default="size", choices=["size", "name", "none"])
    parser.add_argument("--target_bitwidth", type=float, default=None)
    parser.add_argument("--active_quant_budget", action="store_true")
    parser.add_argument("--repair_active_budget", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument(
        "--dry_validate",
        action="store_true",
        help=(
            "Validate and write combined candidates without replaying/evaluating "
            "perplexity. The model is still loaded for module metadata."
        ),
    )
    return parser.parse_args(argv)


def normalize_eval_datasets(values: Sequence[str]) -> list[str]:
    datasets: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            datasets.append(DATASET_ALIASES.get(item, item))
    return datasets


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "source"


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sibling_run_summary(path: Path) -> dict[str, Any]:
    summary_path = path.parent / "run_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return read_json(summary_path)
    except json.JSONDecodeError:
        return {}


def normalize_drop_state(drop: Mapping[str, Sequence[Any]]) -> dict[str, list[bool]]:
    if "attn" not in drop or "mlp" not in drop:
        raise ValueError("Drop state must contain 'attn' and 'mlp'.")
    attn = [bool(value) for value in drop["attn"]]
    mlp = [bool(value) for value in drop["mlp"]]
    if len(attn) != len(mlp):
        raise ValueError("Attention and MLP masks must have the same length.")
    return {"attn": attn, "mlp": mlp}


def drop_from_masks(attention_mask: Sequence[Any], mlp_mask: Sequence[Any]) -> dict[str, list[bool]]:
    return normalize_drop_state(
        {
            "attn": [bool(int(value)) for value in attention_mask],
            "mlp": [bool(int(value)) for value in mlp_mask],
        }
    )


def parse_drop_config(path: Path) -> dict[str, list[bool]]:
    attn: list[bool] = []
    mlp: list[bool] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        if value not in {"none", "attn", "mlp", "attn+mlp"}:
            raise ValueError(f"Unsupported drop config entry in {path}: {value}")
        attn.append(value in {"attn", "attn+mlp"})
        mlp.append(value in {"mlp", "attn+mlp"})
    if not attn:
        raise ValueError(f"No drop entries found in {path}")
    return {"attn": attn, "mlp": mlp}


def parse_quant_config(path: Path) -> dict[str, int]:
    bitwidths: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported quant config entry in {path}: {line}")
        module, level = line.split(":", 1)
        bitwidths[module.strip()] = int(level.strip())
    if not bitwidths:
        raise ValueError(f"No quant entries found in {path}")
    return bitwidths


def load_candidate(label: str, source: str) -> CandidateSource:
    if source.startswith("uniform:"):
        bits = int(source.split(":", 1)[1])
        return CandidateSource(
            label=label,
            source=source,
            metadata={"source_kind": "uniform"},
            is_uniform=True,
            uniform_bits=bits,
        )

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate source does not exist: {source}")

    metadata = {"source_path": str(path)}
    metadata.update({"run_summary": sibling_run_summary(path)})

    if path.suffix == ".json":
        data = read_json(path)
        raw = data.get("candidate_vector_raw")
        drop = None
        quant_state = None
        bitwidth_by_module = None

        if isinstance(data.get("drop"), Mapping):
            drop = normalize_drop_state(data["drop"])
        elif isinstance(raw, Mapping) and isinstance(raw.get("drop"), Mapping):
            drop = normalize_drop_state(raw["drop"])
        elif isinstance(raw, Mapping) and "attn" in raw and "mlp" in raw:
            drop = normalize_drop_state(raw)
        elif "attention_mask" in data and "mlp_mask" in data:
            drop = drop_from_masks(data["attention_mask"], data["mlp_mask"])

        if isinstance(data.get("quant"), list):
            quant_state = [[int(level) for level in group] for group in data["quant"]]
        elif isinstance(raw, Mapping) and isinstance(raw.get("quant"), list):
            quant_state = [[int(level) for level in group] for group in raw["quant"]]
        elif isinstance(raw, list):
            quant_state = [[int(level) for level in group] for group in raw]

        if isinstance(data.get("bitwidth_by_module"), Mapping):
            bitwidth_by_module = {
                str(module): int(level)
                for module, level in data["bitwidth_by_module"].items()
            }

        return CandidateSource(
            label=label,
            source=source,
            drop=drop,
            quant_state=quant_state,
            bitwidth_by_module=bitwidth_by_module,
            metadata=metadata,
            candidate_type=data.get("candidate_type"),
        )

    if path.name.endswith("drop_config.txt") or path.name == "layer_drop_config.txt":
        return CandidateSource(
            label=label,
            source=source,
            drop=parse_drop_config(path),
            metadata=metadata,
            candidate_type="depth_config_txt",
        )

    if path.name.endswith("quant_config.txt") or path.name == "quant_configuration.txt":
        return CandidateSource(
            label=label,
            source=source,
            bitwidth_by_module=parse_quant_config(path),
            metadata=metadata,
            candidate_type="quant_config_txt",
        )

    raise ValueError(f"Unsupported candidate source format: {source}")


def parse_labeled_sources(values: Sequence[str] | None) -> dict[str, CandidateSource]:
    if not values:
        return {}
    sources: dict[str, CandidateSource] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected label=source, got: {item}")
        label, source = item.split("=", 1)
        label = safe_label(label)
        if label in sources:
            raise ValueError(f"Duplicate source label: {label}")
        sources[label] = load_candidate(label, source)
    return sources


def infer_target_bitwidth(
    cli_target: float | None,
    quant_source: CandidateSource,
) -> float | None:
    if cli_target is not None:
        return cli_target
    if quant_source.is_uniform and quant_source.uniform_bits is not None:
        return float(quant_source.uniform_bits)
    run_summary = (quant_source.metadata or {}).get("run_summary") or {}
    compression_config = run_summary.get("compression_config") or {}
    value = compression_config.get("target_average_bitwidth")
    return float(value) if value is not None else None


def quant_state_from_source(
    source: CandidateSource,
    grouped_layer_names: Sequence[Sequence[str]],
) -> list[list[int]]:
    if source.is_uniform:
        if source.uniform_bits is None:
            raise ValueError("Uniform quant source has no bitwidth.")
        return [[int(source.uniform_bits) for _ in group] for group in grouped_layer_names]

    if source.quant_state is not None:
        if len(source.quant_state) != len(grouped_layer_names):
            raise ValueError(
                f"Quant source {source.label} has {len(source.quant_state)} groups; "
                f"expected {len(grouped_layer_names)}."
            )
        normalized: list[list[int]] = []
        for group_id, (levels, names) in enumerate(zip(source.quant_state, grouped_layer_names)):
            if len(levels) != len(names):
                raise ValueError(
                    f"Quant source {source.label} group {group_id} has {len(levels)} "
                    f"entries; expected {len(names)}."
                )
            normalized.append([int(level) for level in levels])
        return normalized

    if source.bitwidth_by_module is None:
        raise ValueError(f"Quant source {source.label} does not contain quantization data.")

    missing = [
        name
        for group in grouped_layer_names
        for name in group
        if name not in source.bitwidth_by_module
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Quant source {source.label} is missing {len(missing)} modules "
            f"from the quantization database scope. First missing: {preview}"
        )
    return [
        [int(source.bitwidth_by_module[name]) for name in group]
        for group in grouped_layer_names
    ]


def validate_drop_state(drop: dict[str, list[bool]], num_layers: int, label: str) -> None:
    if len(drop["attn"]) != num_layers or len(drop["mlp"]) != num_layers:
        raise ValueError(
            f"Depth source {label} has mask lengths "
            f"attn={len(drop['attn'])}, mlp={len(drop['mlp'])}; expected {num_layers}."
        )


def validate_quant_db(
    quant_db: Path,
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
) -> None:
    missing: list[str] = []
    for group, levels in zip(grouped_layer_names, quant_state):
        for layer_name, level in zip(group, levels):
            weight_path = quant_db / layer_name / f"{int(level)}.pth"
            if not weight_path.is_file():
                missing.append(str(weight_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Quantization database is missing {len(missing)} required files. "
            f"First missing files:\n{preview}"
        )


def count_quant_changes(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> int:
    return sum(
        int(before[group_id][i] != after[group_id][i])
        for group_id in range(len(before))
        for i in range(len(before[group_id]))
    )


def active_average_bitwidth(
    model: torch.nn.Module,
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    drop_state: dict[str, list[bool]],
) -> float | None:
    active_weights = quantizable_weights(model, grouped_layer_names, drop_state)
    if active_weights == 0:
        return None
    return candidate_bits(model, grouped_layer_names, quant_state, drop_state) / active_weights


def repair_or_validate_budget(
    model: torch.nn.Module,
    grouped_layer_names: Sequence[Sequence[str]],
    quant_db: Path,
    candidate: dict[str, Any],
    target_bitwidth: float | None,
    *,
    active_quant_budget: bool,
    repair_active_budget: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    details: dict[str, Any] = {
        "active_quant_budget": active_quant_budget,
        "target_bitwidth": target_bitwidth,
        "budget_checked": False,
        "budget_valid_before_repair": None,
        "active_average_bitwidth_before_repair": None,
        "active_average_bitwidth_after_repair": None,
        "budget_repaired": False,
        "repair_num_changes": 0,
    }

    if not active_quant_budget:
        return candidate, details
    if target_bitwidth is None:
        raise ValueError(
            "Active quantization budget validation requires --target_bitwidth "
            "or a source run_summary.json with compression_config.target_average_bitwidth."
        )

    before_avg = active_average_bitwidth(
        model,
        grouped_layer_names,
        candidate["quant"],
        candidate["drop"],
    )
    valid_before = (
        before_avg is not None
        and math.isclose(before_avg, target_bitwidth, rel_tol=0.0, abs_tol=1e-9)
    )
    details.update(
        {
            "budget_checked": True,
            "budget_valid_before_repair": valid_before,
            "active_average_bitwidth_before_repair": before_avg,
        }
    )

    if not valid_before:
        if not repair_active_budget:
            raise ValueError(
                f"Active average bitwidth {before_avg} does not match target "
                f"{target_bitwidth}. Pass --repair_active_budget to repair."
            )
        repaired_quant = repair_active_quant_budget(
            grouped_layer_names,
            str(quant_db),
            candidate["quant"],
            candidate["drop"],
            target_bitwidth,
        )
        details["repair_num_changes"] = count_quant_changes(candidate["quant"], repaired_quant)
        details["budget_repaired"] = details["repair_num_changes"] > 0
        candidate = {"drop": candidate["drop"], "quant": repaired_quant}

    after_avg = active_average_bitwidth(
        model,
        grouped_layer_names,
        candidate["quant"],
        candidate["drop"],
    )
    details["active_average_bitwidth_after_repair"] = after_avg
    return candidate, details


def prepare_model(args: argparse.Namespace):
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=args.use_fast_tokenizer)
    layers = get_layers(model)
    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model)))
    return model, tokenizer, layers


def quant_groups(model: torch.nn.Module, quant_db: Path, group_rule: str) -> tuple[list[str], Sequence[Sequence[str]]]:
    layer_names = [
        item.name
        for item in quant_db.iterdir()
        if item.is_dir()
    ]
    if not layer_names:
        raise ValueError(f"No module directories found in quantization database: {quant_db}")
    layer_names = sorted(layer_names, key=layer_order_fn)
    return layer_names, group_layers(model, layer_names, group_rule)


def initialize_model_quant_state(model: torch.nn.Module, grouped_layer_names: Sequence[Sequence[str]]) -> None:
    model.state = [[None] * len(names) for names in grouped_layer_names]


def build_depth_module_names(model: torch.nn.Module, layers) -> tuple[list[str], list[str]]:
    attention_names = [
        module_name(model, getattr(layer, get_attn_layer_name(model)))
        for layer in layers
    ]
    mlp_names = [
        module_name(model, getattr(layer, get_mlp_layer_name(model)))
        for layer in layers
    ]
    return attention_names, mlp_names


def dataset_metrics(
    model: torch.nn.Module,
    tokenizer,
    datasets: Sequence[str],
    eval_tokens: int,
    sequence_length: int,
    batch_size: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for dataset_name in datasets:
        data = get_data(
            dataset_name,
            eval_tokens,
            sequence_length,
            tokenizer,
            train=False,
        )
        metrics[dataset_name] = float(compute_perplexity(model, data, batch_size=batch_size))
    return metrics


def matrix_row_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    compression = summary.get("compression_metrics") or {}
    final_metrics = compression.get("final_metrics") or {}
    repair = summary.get("repair_details") or {}
    return {
        "depth_label": summary.get("depth_label"),
        "quant_label": summary.get("quant_label"),
        "depth_source": summary.get("depth_source"),
        "quant_source": summary.get("quant_source"),
        "wiki2_ppl": metrics.get("wikitext2"),
        "c4_ppl": metrics.get("c4"),
        "fineweb_ppl": metrics.get("fineweb_edu"),
        "compression_ratio": final_metrics.get("estimated_compression_ratio"),
        "active_params": final_metrics.get("active_parameters"),
        "avg_active_bitwidth": final_metrics.get("average_bitwidth_active"),
        "budget_valid_before_repair": repair.get("budget_valid_before_repair"),
        "budget_repaired": repair.get("budget_repaired"),
        "repair_num_changes": repair.get("repair_num_changes"),
        "status": summary.get("status"),
        "error_message": summary.get("error_message"),
    }


def write_summary_markdown(output_dir: Path, summary: Mapping[str, Any]) -> None:
    row = matrix_row_from_summary(summary)
    lines = [
        "# Joint Compression Attribution Replay",
        "",
        f"- status: `{summary.get('status')}`",
        f"- depth source: `{summary.get('depth_label')}` / `{summary.get('depth_source')}`",
        f"- quant source: `{summary.get('quant_label')}` / `{summary.get('quant_source')}`",
        f"- model: `{summary.get('base_model')}`",
        f"- quant database: `{summary.get('quant_db')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| WikiText2 PPL | {row.get('wiki2_ppl', '')} |",
        f"| C4 PPL | {row.get('c4_ppl', '')} |",
        f"| FineWeb-Edu PPL | {row.get('fineweb_ppl', '')} |",
        f"| Compression ratio | {row.get('compression_ratio', '')} |",
        f"| Active parameters | {row.get('active_params', '')} |",
        f"| Average active bitwidth | {row.get('avg_active_bitwidth', '')} |",
        "",
        "## Active Budget",
        "",
        f"- valid before repair: `{row.get('budget_valid_before_repair')}`",
        f"- repaired: `{row.get('budget_repaired')}`",
        f"- repair changes: `{row.get('repair_num_changes')}`",
    ]
    if summary.get("error_message"):
        lines += ["", "## Error", "", str(summary["error_message"])]
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combination_outputs(output_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    write_json(output_dir / "attribution_summary.json", summary)
    write_csv(output_dir / "attribution_summary.csv", MATRIX_COLUMNS, [matrix_row_from_summary(summary)])
    write_summary_markdown(output_dir, summary)
    return matrix_row_from_summary(summary)


def fail_summary(
    args: argparse.Namespace,
    depth_source: CandidateSource,
    quant_source: CandidateSource,
    output_dir: Path,
    error: BaseException,
) -> dict[str, Any]:
    summary = {
        "status": "failed",
        "error_message": str(error),
        "base_model": args.base_model,
        "quant_db": args.quant_db,
        "depth_label": depth_source.label,
        "quant_label": quant_source.label,
        "depth_source": depth_source.source,
        "quant_source": quant_source.source,
        "metrics": {},
        "compression_metrics": {},
        "repair_details": {},
    }
    return write_combination_outputs(output_dir, summary)


def run_combination(
    args: argparse.Namespace,
    *,
    model,
    tokenizer,
    layers,
    grouped_layer_names,
    attention_names,
    mlp_names,
    depth_source: CandidateSource,
    quant_source: CandidateSource,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_output_dir(output_dir, args.overwrite)
    if depth_source.drop is None:
        raise ValueError(f"Depth source {depth_source.label} does not contain a depth mask.")
    validate_drop_state(depth_source.drop, len(layers), depth_source.label)

    quant_state = quant_state_from_source(quant_source, grouped_layer_names)
    validate_quant_db(Path(args.quant_db), grouped_layer_names, quant_state)

    target_bitwidth = infer_target_bitwidth(args.target_bitwidth, quant_source)
    combined = {
        "drop": depth_source.drop,
        "quant": quant_state,
    }
    combined, repair_details = repair_or_validate_budget(
        model,
        grouped_layer_names,
        Path(args.quant_db),
        combined,
        target_bitwidth,
        active_quant_budget=args.active_quant_budget,
        repair_active_budget=args.repair_active_budget,
    )

    write_json(output_dir / "combined_candidate.json", combined)
    drop_config = get_layer_drop_config(combined["drop"])
    (output_dir / "combined_drop_config.txt").write_text(
        "\n".join(drop_config) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "combined_quant_config.txt").open("w", encoding="utf-8") as handle:
        for group_id, group in enumerate(grouped_layer_names):
            for layer_name, level in zip(group, combined["quant"][group_id]):
                handle.write(f"{layer_name}: {int(level)}\n")

    metrics: dict[str, float] = {}
    if not args.dry_validate:
        apply_joint_state(model, layers, grouped_layer_names, combined, args.quant_db)
        metrics = dataset_metrics(
            model,
            tokenizer,
            normalize_eval_datasets(args.eval_datasets),
            args.eval_tokens,
            args.sequence_length,
            args.eval_batch_size,
        )

    depth_details = build_depth_details(attention_names, mlp_names, combined["drop"])
    bitwidths = flatten_quant_state(grouped_layer_names, combined["quant"])
    compression = compute_compression_metrics(model, depth_details, bitwidths)
    final_candidate = build_final_candidate(
        "joint_depth_quant_attribution",
        depth_details,
        bitwidths,
        combined,
    )
    write_json(output_dir / "final_candidate.json", final_candidate)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "compression_metrics.json", compression)

    summary = {
        "status": "completed",
        "error_message": "",
        "base_model": args.base_model,
        "quant_db": args.quant_db,
        "depth_label": depth_source.label,
        "quant_label": quant_source.label,
        "depth_source": depth_source.source,
        "quant_source": quant_source.source,
        "depth_candidate_type": depth_source.candidate_type,
        "quant_candidate_type": quant_source.candidate_type,
        "target_bitwidth": target_bitwidth,
        "active_quant_budget": args.active_quant_budget,
        "repair_active_budget": args.repair_active_budget,
        "repair_details": repair_details,
        "metrics": metrics,
        "compression_metrics": {
            "parameter_statistics": compression["parameter_statistics"],
            "quantization_statistics": compression["quantization_statistics"],
            "model_size_statistics": compression["model_size_statistics"],
            "final_metrics": {
                "estimated_compression_ratio": compression["model_size_statistics"].get("estimated_compression_ratio"),
                "active_parameters": compression["parameter_statistics"].get("active_parameters"),
                "average_bitwidth_active": compression["quantization_statistics"].get("average_bitwidth_active"),
                "average_bitwidth_searched": compression["quantization_statistics"].get("average_bitwidth_searched"),
                "average_bitwidth_total": compression["quantization_statistics"].get("average_bitwidth_total"),
            },
        },
        "artifacts": {
            "combined_candidate": str(output_dir / "combined_candidate.json"),
            "metrics": str(output_dir / "metrics.json"),
            "compression_metrics": str(output_dir / "compression_metrics.json"),
            "summary": str(output_dir / "attribution_summary.json"),
        },
    }
    return write_combination_outputs(output_dir, summary)


def write_matrix_markdown(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Joint Compression Attribution Matrix",
        "",
        "| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('depth_label', '')} | "
            f"{row.get('quant_label', '')} | "
            f"{row.get('wiki2_ppl', '')} | "
            f"{row.get('c4_ppl', '')} | "
            f"{row.get('fineweb_ppl', '')} | "
            f"{row.get('compression_ratio', '')} | "
            f"{row.get('status', '')} |"
        )
    lines += [
        "",
        "Lower perplexity is better. The matrix is post-hoc: it tests replayed",
        "recombinations and does not prove causal mechanisms by itself.",
    ]
    (output_dir / "attribution_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single(args: argparse.Namespace) -> None:
    if not args.depth_source or not args.quant_source:
        raise SystemExit("Single mode requires --depth_source and --quant_source.")
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    depth_source = load_candidate(args.depth_label, args.depth_source)
    quant_source = load_candidate(args.quant_label, args.quant_source)

    model, tokenizer, layers = prepare_model(args)
    _, grouped_layer_names = quant_groups(model, Path(args.quant_db), args.group_rule)
    initialize_model_quant_state(model, grouped_layer_names)
    attention_names, mlp_names = build_depth_module_names(model, layers)

    row = run_combination(
        args,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        grouped_layer_names=grouped_layer_names,
        attention_names=attention_names,
        mlp_names=mlp_names,
        depth_source=depth_source,
        quant_source=quant_source,
        output_dir=output_dir,
    )
    print(json.dumps(row, indent=2, sort_keys=True))


def run_batch(args: argparse.Namespace) -> None:
    if not args.depth_sources or not args.quant_sources:
        raise SystemExit("Batch mode requires --depth_sources and --quant_sources.")
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    depth_sources = parse_labeled_sources(args.depth_sources)
    quant_sources = parse_labeled_sources(args.quant_sources)

    model, tokenizer, layers = prepare_model(args)
    _, grouped_layer_names = quant_groups(model, Path(args.quant_db), args.group_rule)
    initialize_model_quant_state(model, grouped_layer_names)
    attention_names, mlp_names = build_depth_module_names(model, layers)

    rows: list[dict[str, Any]] = []
    combinations_dir = output_dir / "combinations"
    combinations_dir.mkdir(parents=True, exist_ok=True)

    for depth_label, depth_source in depth_sources.items():
        for quant_label, quant_source in quant_sources.items():
            combo_dir = combinations_dir / f"{safe_label(depth_label)}__{safe_label(quant_label)}"
            try:
                row = run_combination(
                    args,
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    grouped_layer_names=grouped_layer_names,
                    attention_names=attention_names,
                    mlp_names=mlp_names,
                    depth_source=depth_source,
                    quant_source=quant_source,
                    output_dir=combo_dir,
                )
            except Exception as exc:  # noqa: BLE001 - batch mode must continue.
                combo_dir.mkdir(parents=True, exist_ok=True)
                row = fail_summary(args, depth_source, quant_source, combo_dir, exc)
            rows.append(row)

    write_csv(output_dir / "attribution_matrix.csv", MATRIX_COLUMNS, rows)
    write_json(output_dir / "attribution_matrix.json", {"rows": rows})
    write_matrix_markdown(output_dir, rows)
    print(f"Wrote attribution matrix: {output_dir / 'attribution_matrix.csv'}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.eval_datasets = normalize_eval_datasets(args.eval_datasets)
    fix_seed(args.seed)

    if args.depth_sources or args.quant_sources:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
