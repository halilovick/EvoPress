"""Exact, shared compression-cost accounting for EvoPress candidates.

The original EvoPress quantization search constrains the parameter-weighted
average assigned bit-width.  That is sufficient when every candidate keeps the
same modules active, but it is not sufficient for joint depth/quantization
search.  This module defines a total-model cost that can be applied to both
candidate types.

The cost is theoretical packed storage, not the size of the ``.pth`` level
database.  The database stores dequantized floating-point reconstructions for
fast candidate evaluation.
"""

from __future__ import annotations

import copy
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


class CompressionBudgetError(ValueError):
    """Raised when a candidate or requested budget is not representable."""


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def flatten_quant_state(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
) -> dict[str, int]:
    if len(grouped_layer_names) != len(quant_state):
        raise CompressionBudgetError("Quantization state group count mismatch.")
    assignments: dict[str, int] = {}
    for names, levels in zip(grouped_layer_names, quant_state):
        if len(names) != len(levels):
            raise CompressionBudgetError("Quantization state group length mismatch.")
        for name, level in zip(names, levels):
            if isinstance(level, bool) or not isinstance(level, int):
                raise CompressionBudgetError(
                    f"Bit-width for {name} must be an integer, got {level!r}."
                )
            if name in assignments:
                raise CompressionBudgetError(f"Duplicate quantized module: {name}")
            assignments[name] = int(level)
    return assignments


def _normalize_drop_state(
    candidate: Any,
    attention_module_names: Sequence[str],
    mlp_module_names: Sequence[str],
) -> tuple[dict[str, list[bool]] | None, set[str]]:
    if not isinstance(candidate, Mapping):
        return None, set()

    raw_drop = candidate.get("drop")
    if raw_drop is None and "attention_mask" in candidate and "mlp_mask" in candidate:
        raw_drop = {
            "attn": candidate["attention_mask"],
            "mlp": candidate["mlp_mask"],
        }
    if raw_drop is None:
        return None, set(candidate.get("dropped_modules", []))
    if not isinstance(raw_drop, Mapping) or set(raw_drop) != {"attn", "mlp"}:
        raise CompressionBudgetError(
            "Depth state must contain exactly 'attn' and 'mlp' masks."
        )
    if len(attention_module_names) != len(mlp_module_names):
        raise CompressionBudgetError("Attention and MLP module lists must match.")

    attn = [bool(value) for value in raw_drop["attn"]]
    mlp = [bool(value) for value in raw_drop["mlp"]]
    if len(attn) != len(attention_module_names) or len(mlp) != len(mlp_module_names):
        raise CompressionBudgetError("Depth masks do not match the model layer count.")
    dropped = {
        name for name, is_dropped in zip(attention_module_names, attn) if is_dropped
    }
    dropped.update(
        name for name, is_dropped in zip(mlp_module_names, mlp) if is_dropped
    )
    return {"attn": attn, "mlp": mlp}, dropped


def _normalize_assignments(
    candidate: Any,
    grouped_layer_names: Sequence[Sequence[str]] | None,
) -> dict[str, int]:
    if isinstance(candidate, Mapping):
        assignments = candidate.get("bitwidth_by_module")
        if assignments is not None:
            if not isinstance(assignments, Mapping):
                raise CompressionBudgetError("bitwidth_by_module must be a mapping.")
            return {str(name): int(level) for name, level in assignments.items()}
        if "quant" in candidate:
            if grouped_layer_names is None:
                raise CompressionBudgetError(
                    "grouped_layer_names is required for a raw joint candidate."
                )
            return flatten_quant_state(grouped_layer_names, candidate["quant"])
    if grouped_layer_names is None:
        raise CompressionBudgetError(
            "grouped_layer_names is required for a raw quantization candidate."
        )
    return flatten_quant_state(grouped_layer_names, candidate)


def _is_under_module(parameter_name: str, module_names: set[str]) -> bool:
    return any(
        parameter_name == module_name or parameter_name.startswith(f"{module_name}.")
        for module_name in module_names
    )


def _metadata_value_count(module: torch.nn.Module, group_size: int) -> int:
    if not hasattr(module, "weight") or module.weight.ndim != 2:
        raise CompressionBudgetError(
            "GPTQ metadata accounting currently requires two-dimensional linear weights."
        )
    out_features, in_features = module.weight.shape
    if group_size <= 0:
        raise CompressionBudgetError("group_size must be positive.")
    if in_features % group_size != 0:
        raise CompressionBudgetError(
            f"Input width {in_features} is not divisible by group size {group_size}."
        )
    return int(out_features) * (int(in_features) // group_size)


def candidate_compression_cost(
    model: torch.nn.Module,
    candidate: Any,
    *,
    grouped_layer_names: Sequence[Sequence[str]] | None = None,
    attention_module_names: Sequence[str] = (),
    mlp_module_names: Sequence[str] = (),
    dense_dtype_bits: int = 16,
    group_size: int | None = None,
    include_quantization_metadata: bool = False,
    scale_bits: int = 16,
    zero_point_bits: int = 16,
) -> dict[str, Any]:
    """Return exact theoretical packed-storage cost for either candidate type.

    Active searched weights cost ``numel * assigned_bitwidth``.  Active weights
    outside the quantization scope cost ``numel * dense_dtype_bits``.  Parameters
    under a removed attention/MLP module cost zero.  If requested, groupwise
    GPTQ scale and zero-point metadata is charged once per output channel and
    input group.
    """

    if dense_dtype_bits <= 0:
        raise CompressionBudgetError("dense_dtype_bits must be positive.")
    if include_quantization_metadata and group_size is None:
        raise CompressionBudgetError(
            "group_size is required when quantization metadata is included."
        )
    if scale_bits < 0 or zero_point_bits < 0:
        raise CompressionBudgetError("Metadata bit-widths must be non-negative.")

    assignments = _normalize_assignments(candidate, grouped_layer_names)
    _, dropped_modules = _normalize_drop_state(
        candidate,
        attention_module_names,
        mlp_module_names,
    )
    parameter_sizes = {name: parameter.numel() for name, parameter in model.named_parameters()}
    total_parameters = sum(parameter_sizes.values())

    quantized_parameters: dict[str, tuple[str, int, int]] = {}
    for module_name, bitwidth in assignments.items():
        if bitwidth <= 0:
            raise CompressionBudgetError(
                f"Bit-width for {module_name} must be positive, got {bitwidth}."
            )
        module = model.get_submodule(module_name)
        parameter_name = f"{module_name}.weight"
        if parameter_name not in parameter_sizes:
            raise CompressionBudgetError(
                f"Quantized module weight is not a named parameter: {parameter_name}"
            )
        quantized_parameters[parameter_name] = (
            module_name,
            bitwidth,
            parameter_sizes[parameter_name],
        )

    dense_model_bits = total_parameters * dense_dtype_bits
    fixed_precision_bits = 0
    quantized_weight_bits = 0
    metadata_bits = 0
    active_parameters = 0
    removed_parameters = 0
    searched_parameters_dense = sum(item[2] for item in quantized_parameters.values())
    searched_parameters_active = 0

    for parameter_name, num_parameters in parameter_sizes.items():
        if _is_under_module(parameter_name, dropped_modules):
            removed_parameters += num_parameters
            continue
        active_parameters += num_parameters
        quantized = quantized_parameters.get(parameter_name)
        if quantized is None:
            fixed_precision_bits += num_parameters * dense_dtype_bits
            continue

        module_name, bitwidth, _ = quantized
        searched_parameters_active += num_parameters
        quantized_weight_bits += num_parameters * bitwidth
        if include_quantization_metadata:
            module = model.get_submodule(module_name)
            metadata_bits += _metadata_value_count(module, int(group_size)) * (
                scale_bits + zero_point_bits
            )

    total_cost_bits = fixed_precision_bits + quantized_weight_bits + metadata_bits
    if total_cost_bits % 8:
        total_cost_bytes: int | float = total_cost_bits / 8
    else:
        total_cost_bytes = total_cost_bits // 8
    compression_ratio = dense_model_bits / total_cost_bits if total_cost_bits else None
    assigned_average_active = (
        quantized_weight_bits / searched_parameters_active
        if searched_parameters_active
        else None
    )

    return {
        "accounting_version": 1,
        "total_cost_bits": total_cost_bits,
        "total_cost_bytes": total_cost_bytes,
        "total_cost_mib": total_cost_bits / 8 / 1024**2,
        "total_cost_gib": total_cost_bits / 8 / 1024**3,
        "dense_model_bits": dense_model_bits,
        "dense_model_bytes": dense_model_bits // 8,
        "dense_model_mib": dense_model_bits / 8 / 1024**2,
        "compression_ratio": compression_ratio,
        "fixed_precision_bits": fixed_precision_bits,
        "quantized_weight_bits": quantized_weight_bits,
        "quantization_metadata_bits": metadata_bits,
        "removed_dense_equivalent_bits": removed_parameters * dense_dtype_bits,
        "total_parameters_dense": total_parameters,
        "active_parameters": active_parameters,
        "removed_parameters": removed_parameters,
        "searched_parameters_dense": searched_parameters_dense,
        "searched_parameters_active": searched_parameters_active,
        "fixed_parameters_active": active_parameters - searched_parameters_active,
        "assigned_average_bitwidth_active": assigned_average_active,
        "dense_dtype_bits": dense_dtype_bits,
        "group_size": group_size,
        "include_quantization_metadata": include_quantization_metadata,
        "scale_bits": scale_bits if include_quantization_metadata else 0,
        "zero_point_bits": zero_point_bits if include_quantization_metadata else 0,
        "cost_formula": (
            "sum(active searched weight numel * assigned bits) + "
            "sum(active unsearched parameter numel * dense dtype bits) + "
            "sum(active GPTQ groups * (scale bits + zero-point bits)); "
            "removed module parameters and metadata contribute zero"
        ),
    }


def uniform_quantization_target_cost(
    model: torch.nn.Module,
    grouped_layer_names: Sequence[Sequence[str]],
    target_bitwidth: int,
    **cost_kwargs: Any,
) -> dict[str, Any]:
    if isinstance(target_bitwidth, bool) or not isinstance(target_bitwidth, int):
        raise CompressionBudgetError(
            "The uniform reference target bit-width must be an integer."
        )
    candidate = [
        [target_bitwidth for _ in group]
        for group in grouped_layer_names
    ]
    return candidate_compression_cost(
        model,
        candidate,
        grouped_layer_names=grouped_layer_names,
        **cost_kwargs,
    )


def validate_exact_budget(
    realized_cost: Mapping[str, Any],
    target_cost_bits: int,
    *,
    context: str = "candidate",
) -> bool:
    realized_value = realized_cost.get("total_cost_bits")
    if realized_value is None:
        realized_value = realized_cost.get("compression_cost_bits")
    if realized_value is None:
        raise CompressionBudgetError(
            "Realized cost mapping has neither total_cost_bits nor compression_cost_bits."
        )
    realized = int(realized_value)
    difference = realized - int(target_cost_bits)
    if difference:
        percentage = 100.0 * difference / int(target_cost_bits)
        raise CompressionBudgetError(
            f"{context} violates the exact compression budget: "
            f"target_bits={target_cost_bits}, realized_bits={realized}, "
            f"difference_bits={difference}, difference_percent={percentage:.12g}."
        )
    return True


def _module_is_active(module_name: str, drop_state: Mapping[str, Sequence[bool]] | None) -> bool:
    if drop_state is None:
        return True
    match = _LAYER_PATTERN.search(module_name)
    if match is None:
        return True
    layer_id = int(match.group(1))
    if ".self_attn." in module_name:
        return not bool(drop_state["attn"][layer_id])
    if ".mlp." in module_name:
        return not bool(drop_state["mlp"][layer_id])
    return True


def available_module_bitwidths(
    quant_weights_path: str | os.PathLike[str],
    module_name: str,
) -> list[int]:
    module_dir = Path(quant_weights_path) / module_name
    if not module_dir.is_dir():
        raise CompressionBudgetError(f"Missing quantization module directory: {module_dir}")
    levels = sorted(
        int(path.stem)
        for path in module_dir.glob("*.pth")
        if path.stem.isdigit()
    )
    if not levels:
        raise CompressionBudgetError(f"No quantization levels found for {module_name}.")
    return levels


def _repair_delta_within_groups(
    model: torch.nn.Module,
    grouped_layer_names: Sequence[Sequence[str]],
    quant_weights_path: str | os.PathLike[str],
    repaired: list[list[int]],
    drop_state: Mapping[str, Sequence[bool]] | None,
    signed_delta: int,
    allowed_group_ids: set[int],
    rng: random.Random | Any,
) -> None:
    if signed_delta == 0:
        return
    direction = 1 if signed_delta > 0 else -1
    required_bits = abs(signed_delta)
    module_options: list[tuple[int, int, str, list[tuple[int, int]]]] = []
    possible_gains: list[int] = []
    for group_id, group in enumerate(grouped_layer_names):
        if group_id not in allowed_group_ids:
            continue
        for position, module_name in enumerate(group):
            if not _module_is_active(module_name, drop_state):
                continue
            current_level = int(repaired[group_id][position])
            num_parameters = model.get_submodule(module_name).weight.numel()
            choices: list[tuple[int, int]] = []
            for level in available_module_bitwidths(quant_weights_path, module_name):
                level_delta = direction * (level - current_level)
                if level_delta <= 0:
                    continue
                gain = num_parameters * level_delta
                if gain <= required_bits:
                    choices.append((level, gain))
                    possible_gains.append(gain)
            if choices:
                rng.shuffle(choices)
                module_options.append((group_id, position, module_name, choices))

    if not possible_gains:
        raise CompressionBudgetError(
            "No active quantization assignment can move toward the requested budget."
        )
    unit = math.gcd(*possible_gains)
    if required_bits % unit:
        raise CompressionBudgetError(
            "Requested budget difference is not exactly representable by available "
            f"active level changes: difference_bits={signed_delta}, gcd_bits={unit}."
        )
    required_units = required_bits // unit
    rng.shuffle(module_options)
    paths: dict[int, tuple[tuple[int, int, int], ...]] = {0: ()}
    for group_id, position, _module_name, choices in module_options:
        updated = dict(paths)
        for subtotal, path in paths.items():
            for new_level, gain_bits in choices:
                total = subtotal + gain_bits // unit
                if total > required_units or total in updated:
                    continue
                updated[total] = path + ((group_id, position, new_level),)
        paths = updated
        if required_units in paths:
            break
    if required_units not in paths:
        attainable = min(paths, key=lambda value: abs(required_units - value))
        residual_bits = (required_units - attainable) * unit * direction
        raise CompressionBudgetError(
            "Unable to repair the candidate to the exact budget with available "
            f"levels: difference_bits={signed_delta}, closest_residual_bits={residual_bits}."
        )
    for group_id, position, new_level in paths[required_units]:
        repaired[group_id][position] = new_level


def repair_quant_state_to_budget(
    model: torch.nn.Module,
    grouped_layer_names: Sequence[Sequence[str]],
    quant_weights_path: str | os.PathLike[str],
    quant_state: Sequence[Sequence[int]],
    drop_state: Mapping[str, Sequence[bool]] | None,
    target_cost_bits: int,
    *,
    attention_module_names: Sequence[str],
    mlp_module_names: Sequence[str],
    dense_dtype_bits: int = 16,
    group_size: int | None = None,
    include_quantization_metadata: bool = False,
    scale_bits: int = 16,
    zero_point_bits: int = 16,
    preserve_equal_size_group_costs: bool = False,
    uniform_reference_bitwidth: int | None = None,
    rng: random.Random | Any = random,
) -> list[list[int]]:
    """Change active bit-width assignments to hit an exact total-model cost.

    A bounded multiple-choice subset-sum is solved over available database
    levels.  The state space is normalized by the GCD of possible cost changes,
    which keeps the Mistral 7B problem small (one unit is the smallest projection
    size).  Assignments inside removed modules are deliberately ignored.
    """

    repaired = copy.deepcopy(quant_state)
    candidate = {"drop": drop_state, "quant": repaired} if drop_state is not None else repaired
    cost_kwargs = {
        "attention_module_names": attention_module_names,
        "mlp_module_names": mlp_module_names,
        "dense_dtype_bits": dense_dtype_bits,
        "group_size": group_size,
        "include_quantization_metadata": include_quantization_metadata,
        "scale_bits": scale_bits,
        "zero_point_bits": zero_point_bits,
    }
    if preserve_equal_size_group_costs:
        if uniform_reference_bitwidth is None:
            raise CompressionBudgetError(
                "uniform_reference_bitwidth is required for group-preserving repair."
            )
        for group_id, group in enumerate(grouped_layer_names):
            module_sizes = {
                model.get_submodule(module_name).weight.numel()
                for module_name in group
            }
            if len(module_sizes) != 1:
                raise CompressionBudgetError(
                    "Group-preserving repair requires equal-size module groups."
                )
            baseline_weight_bits = sum(module_sizes) * len(group) * uniform_reference_bitwidth
            metadata_per_module = 0
            if include_quantization_metadata:
                first_module = model.get_submodule(group[0])
                metadata_per_module = _metadata_value_count(first_module, int(group_size)) * (
                    scale_bits + zero_point_bits
                )
            active_positions = [
                position
                for position, module_name in enumerate(group)
                if _module_is_active(module_name, drop_state)
            ]
            target_group_weight_bits = (
                baseline_weight_bits
                + metadata_per_module * len(group)
                - metadata_per_module * len(active_positions)
            )
            current_group_weight_bits = sum(
                model.get_submodule(group[position]).weight.numel()
                * int(repaired[group_id][position])
                for position in active_positions
            )
            _repair_delta_within_groups(
                model,
                grouped_layer_names,
                quant_weights_path,
                repaired,
                drop_state,
                target_group_weight_bits - current_group_weight_bits,
                {group_id},
                rng,
            )
    else:
        current_cost = candidate_compression_cost(
            model,
            candidate,
            grouped_layer_names=grouped_layer_names,
            **cost_kwargs,
        )
        signed_delta = int(target_cost_bits) - int(current_cost["total_cost_bits"])
        _repair_delta_within_groups(
            model,
            grouped_layer_names,
            quant_weights_path,
            repaired,
            drop_state,
            signed_delta,
            set(range(len(grouped_layer_names))),
            rng,
        )

    repaired_candidate = (
        {"drop": drop_state, "quant": repaired}
        if drop_state is not None
        else repaired
    )
    repaired_cost = candidate_compression_cost(
        model,
        repaired_candidate,
        grouped_layer_names=grouped_layer_names,
        **cost_kwargs,
    )
    validate_exact_budget(repaired_cost, target_cost_bits, context="repaired candidate")
    return repaired


def inspect_quantization_database(
    quant_weights_path: str | os.PathLike[str],
    *,
    expected_module_names: Sequence[str] | None = None,
    expected_bitwidths: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(quant_weights_path)
    if not root.is_dir():
        raise CompressionBudgetError(f"Quantization database does not exist: {root}")
    module_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if expected_module_names is not None:
        expected = set(expected_module_names)
        actual = set(module_names)
        if expected != actual:
            raise CompressionBudgetError(
                "Quantization database scope mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
            )
    levels_by_module = {
        name: available_module_bitwidths(root, name) for name in module_names
    }
    if expected_bitwidths is not None:
        expected_levels = sorted(int(level) for level in expected_bitwidths)
        failures = {
            name: levels
            for name, levels in levels_by_module.items()
            if levels != expected_levels
        }
        if failures:
            raise CompressionBudgetError(
                f"Quantization database level mismatch: {failures}"
            )
    return {
        "path": str(root),
        "module_count": len(module_names),
        "module_names": module_names,
        "levels_by_module": levels_by_module,
        "common_bitwidths": (
            levels_by_module[module_names[0]] if module_names else []
        ),
    }
