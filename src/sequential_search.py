"""Sequential-initialization helpers for the joint EvoPress search.

This module is intentionally independent of torch/transformers so candidate
adapters and exact frozen-quantization feasibility can be tested on CPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


SEQUENTIAL_MODES = (
    "none",
    "depth_to_quant_frozen",
    "depth_to_joint_warm",
    "quant_to_depth_frozen",
    "quant_to_joint_warm",
)
DEPTH_FIRST_MODES = {
    "depth_to_quant_frozen",
    "depth_to_joint_warm",
}
QUANT_FIRST_MODES = {
    "quant_to_depth_frozen",
    "quant_to_joint_warm",
}
FROZEN_MODES = {
    "depth_to_quant_frozen",
    "quant_to_depth_frozen",
}

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class SequentialSearchError(ValueError):
    """Raised when a sequential configuration or candidate is incompatible."""


class FeasibilitySearchLimitError(SequentialSearchError):
    """Raised when exact feasibility search reaches its configured state bound."""


@dataclass
class Stage1Artifacts:
    candidate_path: Path
    summary_path: Path | None
    run_dir: Path | None
    candidate: dict[str, Any]
    summary: dict[str, Any] | None


@dataclass
class Stage1Import:
    component: dict[str, list[bool]] | list[list[int]]
    component_hash: str
    candidate_path: str
    summary_path: str | None
    run_dir: str | None
    search_type: str
    model_name: str | None
    source_group_rule: str | None
    source_target_bitwidth: float | None


@dataclass(frozen=True)
class ComponentContribution:
    level_sums: tuple[int, ...]
    module_counts: tuple[int, ...]

    def combined(self, other: "ComponentContribution") -> "ComponentContribution":
        return ComponentContribution(
            tuple(a + b for a, b in zip(self.level_sums, other.level_sums)),
            tuple(a + b for a, b in zip(self.module_counts, other.module_counts)),
        )


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SequentialSearchError(
                f"Duplicate JSON key in stage-one artifact: {key}"
            )
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise SequentialSearchError(
            f"Unable to read stage-one JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SequentialSearchError(
            f"Stage-one JSON artifact must contain an object: {path}"
        )
    return value


def resolve_stage1_artifacts(
    stage1_run_dir: str | os.PathLike[str] | None,
    stage1_candidate: str | os.PathLike[str] | None,
) -> Stage1Artifacts:
    if bool(stage1_run_dir) == bool(stage1_candidate):
        raise SequentialSearchError(
            "Specify exactly one of --stage1_run_dir or --stage1_candidate."
        )

    run_dir: Path | None = None
    if stage1_run_dir:
        run_dir = Path(stage1_run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise SequentialSearchError(
                f"Stage-one run directory does not exist: {run_dir}"
            )
        candidate_path = run_dir / "final_candidate.json"
        summary_path = run_dir / "run_summary.json"
        if not candidate_path.is_file() and summary_path.is_file():
            summary = _read_json(summary_path)
            artifact_value = summary.get("artifacts", {}).get("candidate_path")
            if artifact_value:
                artifact_path = Path(str(artifact_value)).expanduser()
                alternatives = [
                    artifact_path,
                    run_dir / artifact_path.name,
                ]
                candidate_path = next(
                    (path.resolve() for path in alternatives if path.is_file()),
                    candidate_path,
                )
        if not candidate_path.is_file():
            raise SequentialSearchError(
                f"No structured final_candidate.json was found in stage-one run: {run_dir}"
            )
    else:
        candidate_path = Path(str(stage1_candidate)).expanduser().resolve()
        if not candidate_path.is_file():
            raise SequentialSearchError(
                f"Stage-one candidate path does not exist: {candidate_path}"
            )
        run_dir = candidate_path.parent
        summary_path = run_dir / "run_summary.json"

    candidate = _read_json(candidate_path)
    summary = _read_json(summary_path) if summary_path.is_file() else None
    return Stage1Artifacts(
        candidate_path=candidate_path.resolve(),
        summary_path=summary_path.resolve() if summary is not None else None,
        run_dir=run_dir.resolve() if run_dir is not None else None,
        candidate=candidate,
        summary=summary,
    )


def _summary_value(
    summary: Mapping[str, Any] | None,
    section: str,
    key: str,
) -> Any:
    if summary is None:
        return None
    section_value = summary.get(section, {})
    return section_value.get(key) if isinstance(section_value, Mapping) else None


def _validate_model_identity(
    summary: Mapping[str, Any] | None,
    expected_model_name: str,
) -> str | None:
    source_model = summary.get("model_name") if summary is not None else None
    if source_model is not None and str(source_model) != expected_model_name:
        raise SequentialSearchError(
            "Stage-one model identity mismatch: "
            f"source={source_model!r}, requested={expected_model_name!r}."
        )
    return str(source_model) if source_model is not None else None


def _normalize_mask(value: Any, label: str) -> list[bool]:
    if not isinstance(value, list):
        raise SequentialSearchError(f"{label} must be a JSON list.")
    normalized: list[bool] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            normalized.append(item)
        elif isinstance(item, int) and item in (0, 1):
            normalized.append(bool(item))
        else:
            raise SequentialSearchError(
                f"{label}[{index}] must be boolean or binary integer, got {item!r}."
            )
    return normalized


def validate_depth_counts(
    drop_state: Mapping[str, Sequence[bool]],
    num_layers: int,
    drop_count: int,
    drop_entire_block: bool,
) -> bool:
    if set(drop_state) != {"attn", "mlp"}:
        raise SequentialSearchError(
            "Depth state must contain exactly 'attn' and 'mlp' masks."
        )
    attn = list(drop_state["attn"])
    mlp = list(drop_state["mlp"])
    if len(attn) != num_layers or len(mlp) != num_layers:
        raise SequentialSearchError(
            "Depth mask layer-count mismatch: "
            f"expected={num_layers}, attention={len(attn)}, mlp={len(mlp)}."
        )
    if any(type(value) is not bool for value in attn + mlp):
        raise SequentialSearchError("In-memory depth masks must contain booleans.")
    if sum(attn) != drop_count or sum(mlp) != drop_count:
        raise SequentialSearchError(
            "Depth drop-count mismatch: "
            f"requested={drop_count}, attention={sum(attn)}, mlp={sum(mlp)}."
        )
    if drop_entire_block and attn != mlp:
        raise SequentialSearchError(
            "Whole-block mode requires identical attention and MLP masks."
        )
    return True


def load_stage1_depth_candidate(
    artifacts: Stage1Artifacts,
    *,
    expected_model_name: str,
    num_layers: int,
    drop_count: int,
    drop_entire_block: bool,
) -> Stage1Import:
    candidate = artifacts.candidate
    candidate_type = candidate.get("candidate_type")
    summary_type = artifacts.summary.get("search_type") if artifacts.summary else None
    if candidate_type != "depth_only" or (
        summary_type is not None and summary_type != "depth_only"
    ):
        raise SequentialSearchError(
            "Depth-first sequential modes require a depth_only stage-one result; "
            f"candidate_type={candidate_type!r}, search_type={summary_type!r}."
        )

    attention_mask = _normalize_mask(candidate.get("attention_mask"), "attention_mask")
    mlp_mask = _normalize_mask(candidate.get("mlp_mask"), "mlp_mask")
    depth = {"attn": attention_mask, "mlp": mlp_mask}

    raw = candidate.get("candidate_vector_raw")
    if isinstance(raw, Mapping) and "attn" in raw and "mlp" in raw:
        raw_depth = {
            "attn": _normalize_mask(raw["attn"], "candidate_vector_raw.attn"),
            "mlp": _normalize_mask(raw["mlp"], "candidate_vector_raw.mlp"),
        }
        if raw_depth != depth:
            raise SequentialSearchError(
                "Structured depth masks disagree with candidate_vector_raw."
            )

    validate_depth_counts(depth, num_layers, drop_count, drop_entire_block)
    source_layers = _summary_value(artifacts.summary, "depth_statistics", "num_layers")
    if source_layers is not None and int(source_layers) != num_layers:
        raise SequentialSearchError(
            f"Stage-one decoder-layer count mismatch: source={source_layers}, target={num_layers}."
        )
    source_whole_block = _summary_value(
        artifacts.summary,
        "compression_config",
        "drop_entire_block",
    )
    if source_whole_block is not None and bool(source_whole_block) != drop_entire_block:
        raise SequentialSearchError(
            "Stage-one whole-block setting does not match --drop_entire_block."
        )
    source_drop_two = _summary_value(
        artifacts.summary,
        "compression_config",
        "drop_two_consecutive",
    )
    if source_drop_two:
        raise SequentialSearchError(
            "Stage-one drop_two_consecutive candidates are not compatible with "
            "the joint search representation."
        )
    model_name = _validate_model_identity(artifacts.summary, expected_model_name)
    component = copy.deepcopy(depth)
    return Stage1Import(
        component=component,
        component_hash=stable_json_hash(component),
        candidate_path=str(artifacts.candidate_path),
        summary_path=str(artifacts.summary_path) if artifacts.summary_path else None,
        run_dir=str(artifacts.run_dir) if artifacts.run_dir else None,
        search_type="depth_only",
        model_name=model_name,
        source_group_rule=None,
        source_target_bitwidth=None,
    )


def _flatten_group_names(
    grouped_layer_names: Sequence[Sequence[str]],
) -> list[str]:
    names = [name for group in grouped_layer_names for name in group]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SequentialSearchError(
            f"Target quantization groups contain duplicated modules: {duplicates}"
        )
    return names


def load_stage1_quant_candidate(
    artifacts: Stage1Artifacts,
    *,
    expected_model_name: str,
    num_layers: int,
    grouped_layer_names: Sequence[Sequence[str]],
    group_rule: str,
    target_bitwidth: float,
    quant_weights_path: str | os.PathLike[str],
) -> Stage1Import:
    candidate = artifacts.candidate
    candidate_type = candidate.get("candidate_type")
    summary_type = artifacts.summary.get("search_type") if artifacts.summary else None
    if candidate_type != "quant_only" or (
        summary_type is not None and summary_type != "quant_only"
    ):
        raise SequentialSearchError(
            "Quantization-first sequential modes require a quant_only stage-one result; "
            f"candidate_type={candidate_type!r}, search_type={summary_type!r}."
        )

    source_layers = _summary_value(artifacts.summary, "depth_statistics", "num_layers")
    if source_layers is None and isinstance(candidate.get("attention_mask"), list):
        source_layers = len(candidate["attention_mask"])
    if source_layers is not None and int(source_layers) != num_layers:
        raise SequentialSearchError(
            f"Stage-one decoder-layer count mismatch: source={source_layers}, target={num_layers}."
        )

    model_name = _validate_model_identity(artifacts.summary, expected_model_name)
    source_group_rule = _summary_value(
        artifacts.summary,
        "compression_config",
        "group_rule",
    )
    if source_group_rule is not None and source_group_rule != group_rule:
        raise SequentialSearchError(
            "Stage-one grouping mismatch: "
            f"source={source_group_rule!r}, target={group_rule!r}."
        )
    source_target = _summary_value(
        artifacts.summary,
        "compression_config",
        "target_average_bitwidth",
    )
    if source_target is not None and not math.isclose(
        float(source_target),
        float(target_bitwidth),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SequentialSearchError(
            "Stage-one target bit-width mismatch: "
            f"source={source_target}, target={target_bitwidth}."
        )

    assignments = candidate.get("bitwidth_by_module")
    if not isinstance(assignments, Mapping):
        raise SequentialSearchError(
            "Structured quantization candidate must contain bitwidth_by_module."
        )
    target_names = _flatten_group_names(grouped_layer_names)
    target_set = set(target_names)
    source_set = set(assignments)
    missing = sorted(target_set - source_set)
    extra = sorted(source_set - target_set)
    if missing or extra:
        raise SequentialSearchError(
            "Stage-one quantization scope mismatch: "
            f"missing_modules={missing}, extra_modules={extra}."
        )

    quant_state: list[list[int]] = []
    quant_root = Path(quant_weights_path)
    for group in grouped_layer_names:
        levels: list[int] = []
        for module in group:
            level = assignments[module]
            if isinstance(level, bool) or not isinstance(level, int):
                raise SequentialSearchError(
                    f"Bit-width for {module} must be an integer, got {level!r}."
                )
            weight_path = quant_root / module / f"{level}.pth"
            if not weight_path.is_file():
                raise SequentialSearchError(
                    f"Missing reconstruction for imported assignment: {weight_path}"
                )
            levels.append(level)
        quant_state.append(levels)

    component = copy.deepcopy(quant_state)
    imported_module_assignments = {
        name: int(assignments[name]) for name in sorted(target_names)
    }
    return Stage1Import(
        component=component,
        component_hash=stable_json_hash(imported_module_assignments),
        candidate_path=str(artifacts.candidate_path),
        summary_path=str(artifacts.summary_path) if artifacts.summary_path else None,
        run_dir=str(artifacts.run_dir) if artifacts.run_dir else None,
        search_type="quant_only",
        model_name=model_name,
        source_group_rule=(
            str(source_group_rule) if source_group_rule is not None else None
        ),
        source_target_bitwidth=(
            float(source_target) if source_target is not None else None
        ),
    )


def module_structural_component(module_name: str) -> tuple[str | None, int | None]:
    match = _LAYER_PATTERN.search(module_name)
    if match is None:
        return None, None
    layer_id = int(match.group(1))
    if ".self_attn." in module_name:
        return "attn", layer_id
    if ".mlp." in module_name:
        return "mlp", layer_id
    return None, None


def quant_module_is_active(
    module_name: str,
    drop_state: Mapping[str, Sequence[bool]],
) -> bool:
    component_type, layer_id = module_structural_component(module_name)
    if component_type is None or layer_id is None:
        return True
    if layer_id >= len(drop_state[component_type]):
        raise SequentialSearchError(
            f"Quantized module refers to out-of-range decoder layer: {module_name}"
        )
    return not bool(drop_state[component_type][layer_id])


def compute_active_group_level_sums(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    drop_state: Mapping[str, Sequence[bool]],
) -> tuple[list[int], list[int]]:
    if len(grouped_layer_names) != len(quant_state):
        raise SequentialSearchError("Quantization state group count mismatch.")
    level_sums: list[int] = []
    active_counts: list[int] = []
    for names, levels in zip(grouped_layer_names, quant_state):
        if len(names) != len(levels):
            raise SequentialSearchError("Quantization state group length mismatch.")
        active_levels = [
            int(level)
            for name, level in zip(names, levels)
            if quant_module_is_active(name, drop_state)
        ]
        level_sums.append(sum(active_levels))
        active_counts.append(len(active_levels))
    return level_sums, active_counts


def compute_target_active_group_level_sums(
    active_counts: Sequence[int],
    target_bitwidth: float,
) -> list[Fraction]:
    target = Fraction(str(target_bitwidth))
    return [count * target for count in active_counts]


def validate_active_quant_budget(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    drop_state: Mapping[str, Sequence[bool]],
    target_bitwidth: float,
) -> bool:
    level_sums, active_counts = compute_active_group_level_sums(
        grouped_layer_names,
        quant_state,
        drop_state,
    )
    targets = compute_target_active_group_level_sums(
        active_counts,
        target_bitwidth,
    )
    failures = []
    for group_id, (actual, count, target) in enumerate(
        zip(level_sums, active_counts, targets)
    ):
        if count and Fraction(actual, 1) != target:
            failures.append(
                {
                    "group": group_id,
                    "actual_level_sum": actual,
                    "active_modules": count,
                    "target_level_sum": str(target),
                }
            )
    if failures:
        raise SequentialSearchError(
            f"Active quantization budget is not exact: {failures}"
        )
    return True


def compute_component_contributions(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    num_layers: int,
) -> dict[str, list[ComponentContribution]]:
    num_groups = len(grouped_layer_names)
    sums = {
        "attn": [[0] * num_groups for _ in range(num_layers)],
        "mlp": [[0] * num_groups for _ in range(num_layers)],
    }
    counts = {
        "attn": [[0] * num_groups for _ in range(num_layers)],
        "mlp": [[0] * num_groups for _ in range(num_layers)],
    }
    if len(grouped_layer_names) != len(quant_state):
        raise SequentialSearchError("Quantization state group count mismatch.")
    for group_id, (names, levels) in enumerate(zip(grouped_layer_names, quant_state)):
        if len(names) != len(levels):
            raise SequentialSearchError("Quantization state group length mismatch.")
        for name, level in zip(names, levels):
            component_type, layer_id = module_structural_component(name)
            if component_type is None or layer_id is None:
                continue
            if layer_id >= num_layers:
                raise SequentialSearchError(
                    f"Quantized module refers to out-of-range decoder layer: {name}"
                )
            sums[component_type][layer_id][group_id] += int(level)
            counts[component_type][layer_id][group_id] += 1
    return {
        component_type: [
            ComponentContribution(tuple(levels), tuple(module_counts))
            for levels, module_counts in zip(
                sums[component_type],
                counts[component_type],
            )
        ]
        for component_type in ("attn", "mlp")
    }


def _scaled_deviation(
    contribution: ComponentContribution,
    target: Fraction,
) -> tuple[int, ...]:
    return tuple(
        level_sum * target.denominator - count * target.numerator
        for level_sum, count in zip(
            contribution.level_sums,
            contribution.module_counts,
        )
    )


def _feasibility_diagnostics(
    contributions: Mapping[str, Sequence[ComponentContribution]],
    full_deviation: tuple[int, ...],
    drop_count: int,
    target_bitwidth: float,
    drop_entire_block: bool,
) -> str:
    def unique_vectors(component_type: str) -> list[tuple[int, ...]]:
        target = Fraction(str(target_bitwidth))
        return sorted(
            {_scaled_deviation(item, target) for item in contributions[component_type]}
        )

    return (
        f"requested_drop_count={drop_count}, target_bitwidth={target_bitwidth}, "
        f"required_removed_deviation={full_deviation}, "
        f"whole_block={drop_entire_block}, "
        f"attn_attainable_component_deviations={unique_vectors('attn')}, "
        f"mlp_attainable_component_deviations={unique_vectors('mlp')}"
    )


def generate_exact_feasible_depth_states(
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    *,
    num_layers: int,
    drop_count: int,
    target_bitwidth: float,
    drop_entire_block: bool,
    requested_candidates: int,
    max_states: int,
) -> list[dict[str, list[bool]]]:
    if requested_candidates < 1:
        raise SequentialSearchError("requested_candidates must be at least 1.")
    if max_states < 1:
        raise SequentialSearchError("max_states must be at least 1.")
    if drop_count < 0 or drop_count > num_layers:
        raise SequentialSearchError(
            f"Requested drop count {drop_count} is outside [0, {num_layers}]."
        )

    contributions = compute_component_contributions(
        grouped_layer_names,
        quant_state,
        num_layers,
    )
    target = Fraction(str(target_bitwidth))
    full_sums = [sum(int(level) for level in levels) for levels in quant_state]
    full_counts = [len(levels) for levels in quant_state]
    full_deviation = tuple(
        level_sum * target.denominator - count * target.numerator
        for level_sum, count in zip(full_sums, full_counts)
    )

    if drop_entire_block:
        items = [
            (
                "block",
                layer_id,
                _scaled_deviation(
                    contributions["attn"][layer_id].combined(
                        contributions["mlp"][layer_id]
                    ),
                    target,
                ),
            )
            for layer_id in range(num_layers)
        ]
    else:
        items = [
            (
                component_type,
                layer_id,
                _scaled_deviation(
                    contributions[component_type][layer_id],
                    target,
                ),
            )
            for component_type in ("attn", "mlp")
            for layer_id in range(num_layers)
        ]

    suffix_attn = [0] * (len(items) + 1)
    suffix_mlp = [0] * (len(items) + 1)
    suffix_block = [0] * (len(items) + 1)
    for position in range(len(items) - 1, -1, -1):
        kind = items[position][0]
        suffix_attn[position] = suffix_attn[position + 1] + int(kind == "attn")
        suffix_mlp[position] = suffix_mlp[position + 1] + int(kind == "mlp")
        suffix_block[position] = suffix_block[position + 1] + int(kind == "block")

    explored_states = 0

    @lru_cache(maxsize=None)
    def can_complete(
        position: int,
        attn_needed: int,
        mlp_needed: int,
        block_needed: int,
        remaining: tuple[int, ...],
    ) -> bool:
        nonlocal explored_states
        explored_states += 1
        if explored_states > max_states:
            raise FeasibilitySearchLimitError(
                "Exact depth-feasibility search exceeded "
                f"max_initialization_attempts={max_states}. "
                + _feasibility_diagnostics(
                    contributions,
                    full_deviation,
                    drop_count,
                    target_bitwidth,
                    drop_entire_block,
                )
            )
        if min(attn_needed, mlp_needed, block_needed) < 0:
            return False
        if (
            attn_needed > suffix_attn[position]
            or mlp_needed > suffix_mlp[position]
            or block_needed > suffix_block[position]
        ):
            return False
        if position == len(items):
            return (
                attn_needed == 0
                and mlp_needed == 0
                and block_needed == 0
                and all(value == 0 for value in remaining)
            )

        kind, _, deviation = items[position]
        next_counts = (
            attn_needed - int(kind == "attn"),
            mlp_needed - int(kind == "mlp"),
            block_needed - int(kind == "block"),
        )
        next_remaining = tuple(
            value - delta for value, delta in zip(remaining, deviation)
        )
        if can_complete(position + 1, *next_counts, next_remaining):
            return True
        return can_complete(
            position + 1,
            attn_needed,
            mlp_needed,
            block_needed,
            remaining,
        )

    required_counts = (
        (0, 0, drop_count) if drop_entire_block else (drop_count, drop_count, 0)
    )
    if not can_complete(0, *required_counts, full_deviation):
        raise SequentialSearchError(
            "No exact depth mask is feasible for the frozen quantization profile. "
            + _feasibility_diagnostics(
                contributions,
                full_deviation,
                drop_count,
                target_bitwidth,
                drop_entire_block,
            )
        )

    solutions: list[dict[str, list[bool]]] = []
    solution_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def append_solution(
        dropped_attn: tuple[int, ...],
        dropped_mlp: tuple[int, ...],
    ) -> None:
        key = (tuple(sorted(dropped_attn)), tuple(sorted(dropped_mlp)))
        if key in solution_keys:
            return
        attn_mask = [False] * num_layers
        mlp_mask = [False] * num_layers
        for index in key[0]:
            attn_mask[index] = True
        for index in key[1]:
            mlp_mask[index] = True
        state = {"attn": attn_mask, "mlp": mlp_mask}
        validate_depth_counts(
            state,
            num_layers,
            drop_count,
            drop_entire_block,
        )
        validate_active_quant_budget(
            grouped_layer_names,
            quant_state,
            state,
            target_bitwidth,
        )
        solution_keys.add(key)
        solutions.append(state)

    # Sample exact paths through the memoized feasibility oracle first. This
    # avoids returning many lexicographically adjacent masks when a zero-
    # contribution component type (often MLP for attention-only databases)
    # has a very large combinatorial solution set.
    max_random_paths = min(max_states, max(100, requested_candidates * 100))
    for _ in range(max_random_paths):
        if len(solutions) >= requested_candidates:
            break
        position = 0
        attn_needed, mlp_needed, block_needed = required_counts
        remaining = full_deviation
        dropped_attn: tuple[int, ...] = ()
        dropped_mlp: tuple[int, ...] = ()
        while position < len(items):
            kind, layer_id, deviation = items[position]
            next_counts = (
                attn_needed - int(kind == "attn"),
                mlp_needed - int(kind == "mlp"),
                block_needed - int(kind == "block"),
            )
            next_remaining = tuple(
                value - delta for value, delta in zip(remaining, deviation)
            )
            choices = []
            if can_complete(position + 1, *next_counts, next_remaining):
                choices.append("drop")
            if can_complete(
                position + 1,
                attn_needed,
                mlp_needed,
                block_needed,
                remaining,
            ):
                choices.append("keep")
            decision = random.choice(choices)
            if decision == "drop":
                attn_needed, mlp_needed, block_needed = next_counts
                remaining = next_remaining
                if kind in {"attn", "block"}:
                    dropped_attn += (layer_id,)
                if kind in {"mlp", "block"}:
                    dropped_mlp += (layer_id,)
            position += 1
        append_solution(dropped_attn, dropped_mlp)

    def collect(
        position: int,
        attn_needed: int,
        mlp_needed: int,
        block_needed: int,
        remaining: tuple[int, ...],
        dropped_attn: tuple[int, ...],
        dropped_mlp: tuple[int, ...],
    ) -> None:
        if len(solutions) >= requested_candidates:
            return
        if position == len(items):
            if (
                attn_needed == 0
                and mlp_needed == 0
                and block_needed == 0
                and all(value == 0 for value in remaining)
            ):
                append_solution(dropped_attn, dropped_mlp)
            return

        kind, layer_id, deviation = items[position]
        next_counts = (
            attn_needed - int(kind == "attn"),
            mlp_needed - int(kind == "mlp"),
            block_needed - int(kind == "block"),
        )
        next_remaining = tuple(
            value - delta for value, delta in zip(remaining, deviation)
        )
        if can_complete(position + 1, *next_counts, next_remaining):
            collect(
                position + 1,
                *next_counts,
                next_remaining,
                (
                    dropped_attn + (layer_id,)
                    if kind in {"attn", "block"}
                    else dropped_attn
                ),
                (
                    dropped_mlp + (layer_id,)
                    if kind in {"mlp", "block"}
                    else dropped_mlp
                ),
            )
        if len(solutions) >= requested_candidates:
            return
        if can_complete(
            position + 1,
            attn_needed,
            mlp_needed,
            block_needed,
            remaining,
        ):
            collect(
                position + 1,
                attn_needed,
                mlp_needed,
                block_needed,
                remaining,
                dropped_attn,
                dropped_mlp,
            )

    collect(0, *required_counts, full_deviation, (), ())
    if len(solutions) < requested_candidates:
        raise SequentialSearchError(
            "Exact depth-feasibility search found fewer unique candidates than requested: "
            f"requested={requested_candidates}, generated={len(solutions)}. "
            + _feasibility_diagnostics(
                contributions,
                full_deviation,
                drop_count,
                target_bitwidth,
                drop_entire_block,
            )
        )
    return copy.deepcopy(solutions)


def enumerate_legal_fixed_quant_depth_swaps(
    drop_state: Mapping[str, Sequence[bool]],
    grouped_layer_names: Sequence[Sequence[str]],
    quant_state: Sequence[Sequence[int]],
    *,
    target_bitwidth: float,
    drop_entire_block: bool,
) -> list[tuple[str, int, int]]:
    num_layers = len(drop_state["attn"])
    contributions = compute_component_contributions(
        grouped_layer_names,
        quant_state,
        num_layers,
    )
    legal: list[tuple[str, int, int]] = []
    component_types = ("block",) if drop_entire_block else ("attn", "mlp")
    for component_type in component_types:
        mask_type = "attn" if component_type == "block" else component_type
        kept = [
            index for index, dropped in enumerate(drop_state[mask_type]) if not dropped
        ]
        dropped = [index for index, value in enumerate(drop_state[mask_type]) if value]
        for remove_index in kept:
            for restore_index in dropped:
                if component_type == "block":
                    remove_contribution = contributions["attn"][remove_index].combined(
                        contributions["mlp"][remove_index]
                    )
                    restore_contribution = contributions["attn"][
                        restore_index
                    ].combined(contributions["mlp"][restore_index])
                else:
                    remove_contribution = contributions[component_type][remove_index]
                    restore_contribution = contributions[component_type][restore_index]
                if remove_contribution != restore_contribution:
                    continue

                proposed = {
                    "attn": list(drop_state["attn"]),
                    "mlp": list(drop_state["mlp"]),
                }
                proposed[mask_type][remove_index] = True
                proposed[mask_type][restore_index] = False
                if component_type == "block":
                    proposed["mlp"] = list(proposed["attn"])
                try:
                    validate_active_quant_budget(
                        grouped_layer_names,
                        quant_state,
                        proposed,
                        target_bitwidth,
                    )
                except SequentialSearchError:
                    continue
                legal.append((component_type, remove_index, restore_index))
    return legal


def mutate_fixed_quant_depth_candidate(
    candidate: Mapping[str, Any],
    grouped_layer_names: Sequence[Sequence[str]],
    *,
    target_bitwidth: float,
    drop_entire_block: bool,
    max_mutations: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if max_mutations < 1:
        raise SequentialSearchError("max_mutations must be at least 1.")
    original = copy.deepcopy(candidate)
    offspring = copy.deepcopy(candidate)
    num_layers = len(original["drop"]["attn"])
    original_drop_count = sum(original["drop"]["attn"])
    validate_depth_counts(
        original["drop"],
        num_layers,
        original_drop_count,
        drop_entire_block,
    )
    initial_legal = enumerate_legal_fixed_quant_depth_swaps(
        offspring["drop"],
        grouped_layer_names,
        offspring["quant"],
        target_bitwidth=target_bitwidth,
        drop_entire_block=drop_entire_block,
    )
    if not initial_legal:
        return None, {
            "legal_swap_count": 0,
            "applied_swaps": [],
            "reason": "no_legal_fixed_quant_depth_swap",
        }

    num_mutations = min(
        random.randint(1, max_mutations),
        random.randint(1, max_mutations),
    )
    applied: list[tuple[str, int, int]] = []
    for _ in range(num_mutations):
        legal = enumerate_legal_fixed_quant_depth_swaps(
            offspring["drop"],
            grouped_layer_names,
            offspring["quant"],
            target_bitwidth=target_bitwidth,
            drop_entire_block=drop_entire_block,
        )
        if not legal:
            break
        component_type, remove_index, restore_index = random.choice(legal)
        mask_type = "attn" if component_type == "block" else component_type
        offspring["drop"][mask_type][remove_index] = True
        offspring["drop"][mask_type][restore_index] = False
        if component_type == "block":
            offspring["drop"]["mlp"] = copy.deepcopy(offspring["drop"]["attn"])
        applied.append((component_type, remove_index, restore_index))

    if offspring["quant"] != original["quant"]:
        raise SequentialSearchError(
            "Frozen-quantization mutation changed the quantization component."
        )
    validate_depth_counts(
        offspring["drop"],
        num_layers,
        original_drop_count,
        drop_entire_block,
    )
    validate_active_quant_budget(
        grouped_layer_names,
        offspring["quant"],
        offspring["drop"],
        target_bitwidth,
    )
    return offspring, {
        "legal_swap_count": len(initial_legal),
        "applied_swaps": applied,
        "reason": None,
    }


def changed_quant_gene_names(
    grouped_layer_names: Sequence[Sequence[str]],
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> list[str]:
    if len(grouped_layer_names) != len(before) or len(before) != len(after):
        raise SequentialSearchError("Cannot compare mismatched quantization states.")
    changed: list[str] = []
    for names, before_levels, after_levels in zip(
        grouped_layer_names,
        before,
        after,
    ):
        if len(names) != len(before_levels) or len(names) != len(after_levels):
            raise SequentialSearchError(
                "Cannot compare mismatched quantization groups."
            )
        changed.extend(
            name
            for name, old, new in zip(names, before_levels, after_levels)
            if old != new
        )
    return changed


def sequential_mode_metadata(mode: str) -> tuple[str | None, str | None, str | None]:
    values = {
        "none": (None, None, None),
        "depth_to_quant_frozen": (
            "depth_to_quantization",
            "frozen",
            "depth",
        ),
        "depth_to_joint_warm": ("depth_to_joint", "warm", None),
        "quant_to_depth_frozen": (
            "quantization_to_depth",
            "frozen",
            "quantization",
        ),
        "quant_to_joint_warm": ("quantization_to_joint", "warm", None),
    }
    if mode not in values:
        raise SequentialSearchError(f"Unknown sequential mode: {mode}")
    return values[mode]


def build_sequential_summary_metadata(
    *,
    mode: str,
    stage1_import: Stage1Import | None,
    quant_initialization_policy: str,
    initial_repair_changed_gene_names: Sequence[str],
    initial_candidate_count: int,
    initial_parent: Mapping[str, Any],
    final_parent: Mapping[str, Any],
    initial_fixed_quant_legal_swap_count: int | None,
    final_fixed_quant_legal_swap_count: int | None,
    active_budget_valid: bool | None,
    depth_counts_valid: bool,
) -> dict[str, Any]:
    direction, variant, frozen_component = sequential_mode_metadata(mode)
    return {
        "sequential_mode": mode,
        "sequential_direction": direction,
        "sequential_variant": variant,
        "stage1_run_dir": stage1_import.run_dir if stage1_import else None,
        "stage1_candidate_path": (
            stage1_import.candidate_path if stage1_import else None
        ),
        "stage1_candidate_hash": (
            stage1_import.component_hash if stage1_import else None
        ),
        "stage1_search_type": stage1_import.search_type if stage1_import else None,
        "frozen_component": frozen_component,
        "sequential_quant_initialization_policy": quant_initialization_policy,
        "initial_component_changed_by_repair": (
            mode == "quant_to_joint_warm"
            and quant_initialization_policy == "repair"
            and bool(initial_repair_changed_gene_names)
        ),
        "initial_repair_changed_gene_count": len(initial_repair_changed_gene_names),
        "initial_repair_changed_gene_names": list(initial_repair_changed_gene_names),
        "initial_feasible_candidate_count": initial_candidate_count,
        "initial_parent_hash": stable_json_hash(initial_parent),
        "fixed_quant_legal_swap_count": (
            {
                "initial_parent": initial_fixed_quant_legal_swap_count,
                "final_parent": final_fixed_quant_legal_swap_count,
            }
            if mode == "quant_to_depth_frozen"
            else None
        ),
        "frozen_depth_unchanged": (
            final_parent["drop"] == initial_parent["drop"]
            if mode in FROZEN_MODES
            else None
        ),
        "frozen_quant_unchanged": (
            final_parent["quant"] == initial_parent["quant"]
            if mode in FROZEN_MODES
            else None
        ),
        "active_budget_valid": active_budget_valid,
        "depth_counts_valid": depth_counts_valid,
    }


def validate_sequential_cli(args: Any) -> None:
    mode = getattr(args, "sequential_mode", "none")
    if mode not in SEQUENTIAL_MODES:
        raise SequentialSearchError(f"Unknown sequential mode: {mode}")
    run_dir = getattr(args, "stage1_run_dir", None)
    candidate = getattr(args, "stage1_candidate", None)
    policy = getattr(args, "sequential_quant_initialization_policy", "strict")
    max_initialization_attempts = getattr(args, "max_initialization_attempts", 1)
    max_offspring_attempts = getattr(args, "max_offspring_attempts", 1)
    if max_initialization_attempts < 1 or max_offspring_attempts < 1:
        raise SequentialSearchError(
            "--max_initialization_attempts and --max_offspring_attempts must be positive."
        )

    if mode == "none":
        if run_dir or candidate:
            raise SequentialSearchError(
                "Stage-one source options require a non-'none' --sequential_mode."
            )
        if policy != "strict":
            raise SequentialSearchError(
                "--sequential_quant_initialization_policy applies only to "
                "quant_to_joint_warm."
            )
        return

    if bool(run_dir) == bool(candidate):
        raise SequentialSearchError(
            "Sequential modes require exactly one of --stage1_run_dir or "
            "--stage1_candidate."
        )

    frozen = mode in FROZEN_MODES
    if frozen and getattr(args, "joint_mutation_mode", "standard") != "standard":
        raise SequentialSearchError(
            "Frozen sequential modes reject interaction-aware mutation."
        )
    if frozen and getattr(args, "joint_aware_mutation", False):
        raise SequentialSearchError(
            "Frozen sequential modes reject --joint_aware_mutation."
        )
    if mode == "depth_to_quant_frozen" and getattr(
        args,
        "coarse_to_fine_mutation",
        False,
    ):
        raise SequentialSearchError(
            "Coarse-to-fine depth strength is invalid when depth is frozen."
        )
    if mode == "quant_to_depth_frozen" and getattr(
        args,
        "adaptive_mutation",
        False,
    ):
        raise SequentialSearchError(
            "Adaptive quantization mutation strength is invalid when quantization is frozen."
        )
    if mode in QUANT_FIRST_MODES:
        if not getattr(args, "active_quant_budget", False):
            raise SequentialSearchError(
                "Strict quantization-first modes require --active_quant_budget."
            )
        if getattr(args, "group_rule", None) != "size":
            raise SequentialSearchError(
                "Strict quantization-first modes require --group_rule size."
            )
    if policy == "repair" and mode != "quant_to_joint_warm":
        raise SequentialSearchError(
            "Quantization initialization policy 'repair' is allowed only for "
            "quant_to_joint_warm."
        )
    if mode == "quant_to_depth_frozen" and policy != "strict":
        raise SequentialSearchError(
            "quant_to_depth_frozen requires strict quantization initialization."
        )


def validate_frozen_component(
    candidate: Mapping[str, Any],
    imported_component: Any,
    frozen_component: str,
) -> bool:
    key = {"depth": "drop", "quantization": "quant"}.get(frozen_component)
    if key is None:
        raise SequentialSearchError(f"Unknown frozen component: {frozen_component}")
    if candidate[key] != imported_component:
        raise SequentialSearchError(
            f"Frozen {frozen_component} invariant was violated."
        )
    return True
