"""Structured reporting helpers for EvoPress search runs."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


GENERATION_COLUMNS = [
    "generation",
    "best_search_fitness",
    "fitness_fn",
    "best_calibration_kl",
    "best_train_ppl",
    "wikitext2_ppl",
    "c4_ppl",
    "fineweb_edu_ppl",
    "eval_tokens_used",
    "num_offspring",
    "num_survivors_stage_1",
    "num_survivors_stage_2",
    "num_survivors_stage_3",
    "survivors_per_selection",
    "tokens_per_selection",
    "active_parameters",
    "average_bitwidth_active",
    "estimated_weight_memory_mb",
    "dropped_attention_count",
    "dropped_mlp_count",
    "mutation_summary",
    "accepted_parent_replacement",
    "runtime_seconds_cumulative",
    "peak_gpu_memory_mb",
]

MODEL_SIZE_NOTE = (
    "The estimated model size is theoretical and based on assigned bitwidths. "
    "It does not represent the on-disk size of the current floating-point "
    "reconstruction database."
)

METRIC_DEFINITIONS = {
    "active_parameters": (
        "Theoretical parameters used by inference. Parameters in bypassed "
        "attention/MLP modules are excluded even though their tensors remain allocated."
    ),
    "average_bitwidth_active": (
        "Parameter-weighted average over active searched weights only."
    ),
    "average_bitwidth_searched": (
        "Parameter-weighted average over all searched weights, including assignments "
        "inside modules later bypassed by depth pruning."
    ),
    "average_bitwidth_total": (
        "Theoretical effective bits divided by active total-model parameters. "
        "Active non-searched parameters use the dense dtype bitwidth."
    ),
    "estimated_compression_ratio": (
        "Dense theoretical weight memory divided by estimated compressed weight memory."
    ),
    "peak_gpu_memory_mb": "Peak tensor memory allocated by PyTorch in this process.",
    "peak_cpu_memory_mb": "Peak resident set size reported by the operating system for this process.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit(repo_root: str | os.PathLike[str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.max_memory_allocated() / 1024**2,
        torch.cuda.max_memory_reserved() / 1024**2,
    )


def peak_cpu_memory_mb() -> float | None:
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    if peak_rss <= 0:
        return None
    if platform.system() == "Darwin":
        return peak_rss / 1024**2
    return peak_rss / 1024


def infer_dense_dtype_bits(model: torch.nn.Module, default: int = 16) -> int:
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return default
    return parameter.element_size() * 8


def module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise ValueError("Module is not registered under the supplied model.")


def flatten_quant_state(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]] | None,
) -> dict[str, int]:
    if quant_state is None:
        return {}
    if len(grouped_layer_names) != len(quant_state):
        raise ValueError("Quantization state does not match the number of module groups.")

    bitwidths: dict[str, int] = {}
    for names, levels in zip(grouped_layer_names, quant_state):
        if len(names) != len(levels):
            raise ValueError("Quantization state group length does not match module names.")
        for name, level in zip(names, levels):
            bitwidths[name] = int(level)
    return bitwidths


def build_depth_details(
    attention_module_names: Sequence[str],
    mlp_module_names: Sequence[str],
    drop_state: Mapping[str, Sequence[bool]] | None,
) -> dict[str, Any]:
    if len(attention_module_names) != len(mlp_module_names):
        raise ValueError("Attention and MLP module lists must have equal length.")

    if drop_state is None:
        attention_mask = [0] * len(attention_module_names)
        mlp_mask = [0] * len(mlp_module_names)
    else:
        attention_mask = [int(value) for value in drop_state["attn"]]
        mlp_mask = [int(value) for value in drop_state["mlp"]]

    if len(attention_mask) != len(attention_module_names) or len(mlp_mask) != len(mlp_module_names):
        raise ValueError("Depth masks do not match the model layer count.")

    dropped_attention_layers = [index for index, value in enumerate(attention_mask) if value]
    dropped_mlp_layers = [index for index, value in enumerate(mlp_mask) if value]
    dropped_attention_modules = [
        attention_module_names[index] for index in dropped_attention_layers
    ]
    dropped_mlp_modules = [mlp_module_names[index] for index in dropped_mlp_layers]
    dropped_modules = dropped_attention_modules + dropped_mlp_modules
    kept_modules = [
        name
        for index, name in enumerate(attention_module_names)
        if not attention_mask[index]
    ] + [
        name
        for index, name in enumerate(mlp_module_names)
        if not mlp_mask[index]
    ]

    return {
        "num_layers": len(attention_module_names),
        "num_attention_modules": len(attention_module_names),
        "num_mlp_modules": len(mlp_module_names),
        "dropped_attention_count": len(dropped_attention_modules),
        "dropped_mlp_count": len(dropped_mlp_modules),
        "dropped_total_count": len(dropped_modules),
        "dropped_attention_layers": dropped_attention_layers,
        "dropped_mlp_layers": dropped_mlp_layers,
        "dropped_attention_modules": dropped_attention_modules,
        "dropped_mlp_modules": dropped_mlp_modules,
        "dropped_modules": dropped_modules,
        "kept_modules": kept_modules,
        "attention_mask": attention_mask,
        "mlp_mask": mlp_mask,
    }


def _is_under_module(parameter_name: str, module_names: Iterable[str]) -> bool:
    return any(
        parameter_name == module_name or parameter_name.startswith(f"{module_name}.")
        for module_name in module_names
    )


def _projection_type(module_name_value: str) -> str:
    return module_name_value.rsplit(".", 1)[-1]


def compute_compression_metrics(
    model: torch.nn.Module,
    depth_details: Mapping[str, Any],
    bitwidth_by_module: Mapping[str, int] | None = None,
    dense_dtype_bits: int | None = None,
) -> dict[str, dict[str, Any]]:
    bitwidth_by_module = dict(bitwidth_by_module or {})
    dense_dtype_bits = dense_dtype_bits or infer_dense_dtype_bits(model)
    dropped_modules = set(depth_details.get("dropped_modules", []))

    parameter_sizes = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
    }
    total_parameters = sum(parameter_sizes.values())

    quant_weight_parameters: dict[str, tuple[str, int, int]] = {}
    for quant_module_name, bitwidth in bitwidth_by_module.items():
        module = model.get_submodule(quant_module_name)
        if not hasattr(module, "weight"):
            raise ValueError(f"Quantized module has no weight parameter: {quant_module_name}")
        parameter_name = f"{quant_module_name}.weight"
        if parameter_name not in parameter_sizes:
            raise ValueError(f"Quantized weight is not a named model parameter: {parameter_name}")
        quant_weight_parameters[parameter_name] = (
            quant_module_name,
            int(bitwidth),
            parameter_sizes[parameter_name],
        )

    dropped_parameters = sum(
        num_parameters
        for name, num_parameters in parameter_sizes.items()
        if _is_under_module(name, dropped_modules)
    )
    active_parameters = total_parameters - dropped_parameters

    searched_parameters_dense = sum(item[2] for item in quant_weight_parameters.values())
    searched_parameters_active = sum(
        num_parameters
        for parameter_name, (_, _, num_parameters) in quant_weight_parameters.items()
        if not _is_under_module(parameter_name, dropped_modules)
    )
    nonsearched_parameters = total_parameters - searched_parameters_dense
    active_nonsearched_parameters = active_parameters - searched_parameters_active

    assigned_searched_bits = 0
    active_searched_bits = 0
    total_effective_bits = 0
    module_histogram: Counter[str] = Counter()
    parameter_histogram: Counter[str] = Counter()
    projection_bits: dict[str, int] = defaultdict(int)
    projection_parameters: dict[str, int] = defaultdict(int)
    active_quantized_modules = 0

    for parameter_name, num_parameters in parameter_sizes.items():
        if _is_under_module(parameter_name, dropped_modules):
            continue

        quant_entry = quant_weight_parameters.get(parameter_name)
        if quant_entry is None:
            total_effective_bits += num_parameters * dense_dtype_bits
            continue

        quant_module_name, bitwidth, _ = quant_entry
        weighted_bits = num_parameters * bitwidth
        active_searched_bits += weighted_bits
        total_effective_bits += weighted_bits
        active_quantized_modules += 1
        module_histogram[str(bitwidth)] += 1
        parameter_histogram[str(bitwidth)] += num_parameters
        projection = _projection_type(quant_module_name)
        projection_bits[projection] += weighted_bits
        projection_parameters[projection] += num_parameters

    for _, bitwidth, num_parameters in quant_weight_parameters.values():
        assigned_searched_bits += num_parameters * bitwidth

    average_bitwidth_searched = (
        assigned_searched_bits / searched_parameters_dense
        if searched_parameters_dense
        else None
    )
    average_bitwidth_active = (
        active_searched_bits / searched_parameters_active
        if searched_parameters_active
        else None
    )
    average_bitwidth_total = (
        total_effective_bits / active_parameters if active_parameters else None
    )

    dense_weight_memory_mb = total_parameters * dense_dtype_bits / 8 / 1024**2
    estimated_weight_memory_mb = total_effective_bits / 8 / 1024**2
    searched_weight_memory_mb = active_searched_bits / 8 / 1024**2
    nonsearched_weight_memory_mb = (
        active_nonsearched_parameters * dense_dtype_bits / 8 / 1024**2
    )
    estimated_compression_ratio = (
        dense_weight_memory_mb / estimated_weight_memory_mb
        if estimated_weight_memory_mb
        else None
    )

    active_parameter_ratio = (
        active_parameters / total_parameters if total_parameters else None
    )
    average_by_projection = {
        projection: projection_bits[projection] / projection_parameters[projection]
        for projection in sorted(projection_bits)
    }

    parameter_statistics = {
        "total_parameters_dense": total_parameters,
        "active_parameters": active_parameters,
        "dropped_parameters": dropped_parameters,
        "active_parameter_ratio": active_parameter_ratio,
        "searched_parameters_dense": searched_parameters_dense,
        "searched_parameters_active": searched_parameters_active,
        "nonsearched_parameters": nonsearched_parameters,
        "active_nonsearched_parameters": active_nonsearched_parameters,
        "active_parameters_note": METRIC_DEFINITIONS["active_parameters"],
    }
    quantization_statistics = {
        "quantized_module_count": len(bitwidth_by_module),
        "active_quantized_module_count": active_quantized_modules,
        "bitwidth_histogram": dict(sorted(module_histogram.items())),
        "bitwidth_parameter_histogram": dict(sorted(parameter_histogram.items())),
        "average_bitwidth_by_projection_type": average_by_projection,
        "bitwidth_by_module": dict(sorted(bitwidth_by_module.items())),
        "average_bitwidth_active": average_bitwidth_active,
        "average_bitwidth_searched": average_bitwidth_searched,
        "average_bitwidth_total": average_bitwidth_total,
    }
    model_size_statistics = {
        "estimated_weight_memory_mb": estimated_weight_memory_mb,
        "dense_weight_memory_mb": dense_weight_memory_mb,
        "estimated_compression_ratio": estimated_compression_ratio,
        "searched_weight_memory_mb": searched_weight_memory_mb,
        "nonsearched_weight_memory_mb": nonsearched_weight_memory_mb,
        "dense_dtype_bits": dense_dtype_bits,
        "model_size_note": MODEL_SIZE_NOTE,
    }
    return {
        "parameter_statistics": parameter_statistics,
        "quantization_statistics": quantization_statistics,
        "model_size_statistics": model_size_statistics,
    }


def build_final_candidate(
    candidate_type: str,
    depth_details: Mapping[str, Any],
    bitwidth_by_module: Mapping[str, int] | None,
    raw_candidate: Any,
) -> dict[str, Any]:
    return {
        "candidate_type": candidate_type,
        "dropped_modules": list(depth_details.get("dropped_modules", [])),
        "kept_modules": list(depth_details.get("kept_modules", [])),
        "attention_mask": list(depth_details.get("attention_mask", [])),
        "mlp_mask": list(depth_details.get("mlp_mask", [])),
        "bitwidth_by_module": dict(sorted((bitwidth_by_module or {}).items())),
        "candidate_vector_raw": raw_candidate,
    }


def available_bitwidths(quant_weights_path: str | os.PathLike[str] | None) -> list[int]:
    if not quant_weights_path:
        return []
    root = Path(quant_weights_path)
    if not root.is_dir():
        return []
    bitwidths = {
        int(path.stem)
        for path in root.glob("*/*.pth")
        if path.stem.lstrip("-").isdigit()
    }
    return sorted(bitwidths)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
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


def write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class RunReporter:
    def __init__(
        self,
        output_dir: str | os.PathLike[str] | None,
        search_type: str,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.enabled = output_dir is not None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.search_type = search_type
        self.repo_root = repo_root
        self.timestamp_start = utc_now()
        self.start_monotonic = time.monotonic()
        self.run_name = self.output_dir.name if self.output_dir is not None else None
        self.generation_log_path = (
            self.output_dir / "generation_log.csv" if self.output_dir is not None else None
        )
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            reset_peak_gpu_memory()

    def runtime_seconds(self) -> float:
        return time.monotonic() - self.start_monotonic

    def append_generation(self, row: Mapping[str, Any]) -> None:
        if not self.enabled or self.generation_log_path is None:
            return
        normalized = {column: row.get(column) for column in GENERATION_COLUMNS}
        normalized["survivors_per_selection"] = json.dumps(
            normalized["survivors_per_selection"], separators=(",", ":")
        )
        normalized["tokens_per_selection"] = json.dumps(
            normalized["tokens_per_selection"], separators=(",", ":")
        )
        normalized["mutation_summary"] = json.dumps(
            normalized["mutation_summary"], separators=(",", ":")
        )
        write_header = not self.generation_log_path.exists()
        with self.generation_log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GENERATION_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)

    def write_final_candidate(self, candidate: Mapping[str, Any]) -> str | None:
        if not self.enabled or self.output_dir is None:
            return None
        path = self.output_dir / "final_candidate.json"
        write_json(path, candidate)
        return str(path)

    def write_summary(
        self,
        *,
        model_name: str,
        dataset_calibration: str,
        dataset_eval: Sequence[str],
        search_config: Mapping[str, Any],
        compression_config: Mapping[str, Any],
        final_metrics: Mapping[str, Any],
        parameter_statistics: Mapping[str, Any],
        depth_statistics: Mapping[str, Any],
        quantization_statistics: Mapping[str, Any],
        model_size_statistics: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> str | None:
        if not self.enabled or self.output_dir is None:
            return None

        peak_allocated_mb, peak_reserved_mb = peak_gpu_memory()
        merged_final_metrics = {
            "best_search_fitness": None,
            "final_calibration_kl": None,
            "wikitext2_ppl": None,
            "c4_ppl": None,
            "fineweb_ppl": None,
            "train_ppl": None,
            "active_parameters": parameter_statistics.get("active_parameters"),
            "total_parameters_dense": parameter_statistics.get("total_parameters_dense"),
            "active_parameter_ratio": parameter_statistics.get("active_parameter_ratio"),
            "average_bitwidth_active": quantization_statistics.get("average_bitwidth_active"),
            "average_bitwidth_searched": quantization_statistics.get("average_bitwidth_searched"),
            "average_bitwidth_total": quantization_statistics.get("average_bitwidth_total"),
            "estimated_weight_memory_mb": model_size_statistics.get("estimated_weight_memory_mb"),
            "dense_weight_memory_mb": model_size_statistics.get("dense_weight_memory_mb"),
            "estimated_compression_ratio": model_size_statistics.get("estimated_compression_ratio"),
            "runtime_seconds": self.runtime_seconds(),
            "peak_gpu_memory_mb": peak_allocated_mb,
            "peak_gpu_reserved_mb": peak_reserved_mb,
            "peak_cpu_memory_mb": peak_cpu_memory_mb(),
        }
        merged_final_metrics.update(final_metrics)

        summary = {
            "schema_version": 1,
            "run_name": self.run_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": utc_now(),
            "git_commit": get_git_commit(self.repo_root),
            "model_name": model_name,
            "dataset_calibration": dataset_calibration,
            "dataset_eval": list(dataset_eval),
            "search_type": self.search_type,
            "search_config": dict(search_config),
            "compression_config": dict(compression_config),
            "final_metrics": merged_final_metrics,
            "parameter_statistics": dict(parameter_statistics),
            "depth_statistics": dict(depth_statistics),
            "quantization_statistics": dict(quantization_statistics),
            "model_size_statistics": dict(model_size_statistics),
            "artifacts": dict(artifacts),
            "metric_definitions": METRIC_DEFINITIONS,
        }
        path = self.output_dir / "run_summary.json"
        write_json(path, summary)
        return str(path)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
