import argparse
import copy
import json
import math
import os
import random
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common_utils import fix_seed
from src.data_utils import get_data
from src.metrics import compute_kl_div, compute_perplexity
from src.model_utils import (
    dummy_initialize,
    get_attn_layer_name,
    get_layers,
    get_mlp_layer_name,
    group_layers,
    layer_order_fn,
    make_dummy_forward,
    restore_forward,
)
from src.run_reporting import (
    RunReporter,
    available_bitwidths,
    build_depth_details,
    build_final_candidate,
    compute_compression_metrics,
    flatten_quant_state,
    module_name,
    peak_gpu_memory,
)
from src.sequential_search import (
    DEPTH_FIRST_MODES,
    QUANT_FIRST_MODES,
    SEQUENTIAL_MODES,
    build_sequential_summary_metadata,
    changed_quant_gene_names,
    enumerate_legal_fixed_quant_depth_swaps,
    generate_exact_feasible_depth_states,
    load_stage1_depth_candidate,
    load_stage1_quant_candidate,
    mutate_fixed_quant_depth_candidate,
    resolve_stage1_artifacts,
    sequential_mode_metadata,
    stable_json_hash,
    validate_active_quant_budget,
    validate_depth_counts,
    validate_frozen_component,
    validate_sequential_cli,
)


def get_layer_drop_config(removed_state) -> List[str]:
    num_blocks = len(removed_state["attn"])
    drop_config = ["none"] * num_blocks

    for i in range(num_blocks):
        if removed_state["attn"][i] and removed_state["mlp"][i]:
            drop_config[i] = "attn+mlp"
        elif removed_state["attn"][i]:
            drop_config[i] = "attn"
        elif removed_state["mlp"][i]:
            drop_config[i] = "mlp"

    return drop_config


def load_drop_state(model, layers, removed_state):
    """
    Apply depth-pruning state by patching attention/MLP forwards.
    True means the sub-block is skipped with a dummy forward.
    """
    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model))

            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            else:
                restore_forward(subblock)


def load_quant_layers(
    model: AutoModelForCausalLM,
    grouped_layer_names: Tuple[Sequence[str]],
    new_state: Tuple[Sequence[int]],
    quant_weights_path: str,
):
    """
    Apply quantization state by loading dequantized candidate weights from disk.
    The state format follows evo_quant_search.py: one list of bitwidths per group.
    """
    assert hasattr(model, "state")

    for group_id in range(len(grouped_layer_names)):
        for layer_name, new_level, old_level in zip(
            grouped_layer_names[group_id],
            new_state[group_id],
            model.state[group_id],
        ):
            if new_level != old_level:
                layer = model.get_submodule(layer_name)
                weight_path = os.path.join(quant_weights_path, layer_name, f"{new_level}.pth")
                layer.weight.data = torch.load(weight_path, map_location=layer.weight.device).to(layer.weight.dtype)

    model.state = copy.deepcopy(new_state)


def apply_joint_state(model, layers, grouped_layer_names, candidate, quant_weights_path):
    """
    Apply both compression components.

    Order:
    1. Load quantized/dequantized weights.
    2. Apply drop masks.

    If a dropped attention/MLP contains a quantized q_proj, the loaded q_proj weights
    are simply unused during that candidate's forward pass.
    """
    load_quant_layers(model, grouped_layer_names, candidate["quant"], quant_weights_path)
    load_drop_state(model, layers, candidate["drop"])


def compute_fitness(model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    if fitness_fn == "kl":
        return compute_kl_div(model, data, target_logits)
    raise ValueError(f"Unsupported fitness_fn: {fitness_fn}")


def sample_minibatch(calibration_data, target_logits, num_tokens, fitness_fn):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0

    while tokens_used < num_tokens:
        minibatch_id = random.randint(0, len(calibration_data) - 1)

        if minibatch_id in minibatch_ids:
            continue

        minibatch_ids.append(minibatch_id)
        sample = calibration_data[minibatch_id]

        remaining = num_tokens - tokens_used

        if sample.shape[1] > remaining:
            calibration_minibatch.append(sample[:, :remaining])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, :remaining])
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(sample)
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += sample.shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    return calibration_minibatch, target_logits_minibatch


def selection(
    model,
    layers,
    grouped_layer_names,
    quant_weights_path,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    fitness_fn: str,
    target_logits=None,
):
    calibration_minibatch, target_logits_minibatch = sample_minibatch(
        calibration_data,
        target_logits,
        num_tokens,
        fitness_fn,
    )

    fitnesses = []

    for candidate in candidates:
        apply_joint_state(model, layers, grouped_layer_names, candidate, quant_weights_path)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, target_logits_minibatch)
        fitnesses.append(fitness)

    best_ids = np.argsort(fitnesses)[:num_survive]

    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def selected_candidate_metadata(
    selected_candidates,
    candidate_pool,
    metadata,
):
    if len(candidate_pool) != len(metadata):
        raise ValueError("Candidate pool and metadata must have equal lengths.")

    selected_metadata = []
    for selected_candidate in selected_candidates:
        matches = [
            index
            for index, candidate in enumerate(candidate_pool)
            if candidate == selected_candidate
        ]
        if len(matches) != 1:
            raise ValueError(
                "Each selected candidate must match exactly one candidate in "
                "the source pool."
            )
        selected_metadata.append(metadata[matches[0]])
    return selected_metadata


def crossover_is_enabled(population_size: int, crossover_probability: float) -> bool:
    return population_size > 1 and crossover_probability > 0.0


def effective_survivors_per_selection(
    configured_survivors: Sequence[int],
    population_size: int,
) -> List[int]:
    """Keep intermediate stages unchanged and resize only the final stage."""
    if not configured_survivors:
        raise ValueError("At least one selection stage is required.")
    if population_size < 1:
        raise ValueError("population_size must be at least 1.")

    effective_survivors = list(configured_survivors)
    effective_survivors[-1] = population_size
    return effective_survivors


def unique_candidate_count(candidates) -> int:
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return len(unique_candidates)


def validate_persistent_population(
    population,
    population_size: int,
    *,
    context: str,
) -> None:
    unique_count = unique_candidate_count(population)
    if len(population) != population_size or unique_count != population_size:
        raise RuntimeError(
            f"Unable to retain {population_size} unique feasible candidates "
            f"for {context}: retained={len(population)}, unique={unique_count}."
        )


def add_persistent_parents_for_elitism(
    candidates,
    candidate_metadata,
    persistent_population,
):
    if len(candidates) != len(candidate_metadata):
        raise ValueError("Candidate pool and metadata must have equal lengths.")

    final_candidates = list(candidates)
    final_metadata = list(candidate_metadata)
    for persistent_parent in persistent_population:
        if persistent_parent not in final_candidates:
            final_candidates.append(persistent_parent)
            final_metadata.append("parent")
    return final_candidates, final_metadata


def select_distinct_parents(population):
    unique_population = []
    for candidate in population:
        if candidate not in unique_population:
            unique_population.append(candidate)
    if len(unique_population) < 2:
        raise ValueError(
            "Component crossover requires at least two distinct persistent parents."
        )
    return tuple(random.sample(unique_population, 2))


def select_mutation_parent(population):
    if len(population) == 1:
        # Avoid consuming randomness on the legacy single-parent path.
        return population[0]
    return random.choice(population)


def candidate_is_duplicate(candidate, *candidate_collections) -> bool:
    return any(candidate in candidates for candidates in candidate_collections)


def adaptive_mutation_strength(
    stagnation_generations: int,
    patience: int,
    max_strength: int,
) -> int:
    if stagnation_generations < 0:
        raise ValueError("stagnation_generations must be non-negative.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if max_strength < 1:
        raise ValueError("max_strength must be at least 1.")
    return min(max_strength, 1 + stagnation_generations // patience)


def coarse_to_fine_mutation_strength(
    generation: int,
    total_generations: int,
    start_strength: int,
    end_strength: int,
) -> int:
    if total_generations < 1:
        raise ValueError("total_generations must be at least 1.")
    if generation < 1 or generation > total_generations:
        raise ValueError("generation must be within the search range.")
    if end_strength < 1:
        raise ValueError("end_strength must be at least 1.")
    if start_strength < end_strength:
        raise ValueError(
            "start_strength must be greater than or equal to end_strength."
        )
    num_strengths = start_strength - end_strength + 1
    stage = min(
        num_strengths - 1,
        (generation * num_strengths - 1) // total_generations,
    )
    return start_strength - stage


def make_random_drop_state(num_blocks: int, blocks_to_remove: int, drop_entire_block: bool):
    removed_state = {
        "attn": [False] * num_blocks,
        "mlp": [False] * num_blocks,
    }

    attn_remove_ind = random.sample(range(num_blocks), blocks_to_remove)
    for idx in attn_remove_ind:
        removed_state["attn"][idx] = True

    if drop_entire_block:
        removed_state["mlp"] = copy.deepcopy(removed_state["attn"])
    else:
        mlp_remove_ind = random.sample(range(num_blocks), blocks_to_remove)
        for idx in mlp_remove_ind:
            removed_state["mlp"][idx] = True

    return removed_state


def mutate_drop_state(drop_state, drop_entire_block: bool, max_mutations: int):
    offspring = copy.deepcopy(drop_state)

    num_blocks = len(offspring["attn"])
    num_flips = min(random.randint(1, max_mutations), random.randint(1, max_mutations))

    for _ in range(num_flips):
        subblock_type = "attn" if drop_entire_block or random.randint(0, 1) == 0 else "mlp"

        # Pick one currently kept position and drop it.
        remove_ind = random.randint(0, num_blocks - 1)
        while offspring[subblock_type][remove_ind]:
            remove_ind = random.randint(0, num_blocks - 1)

        # Pick one currently dropped position and restore it.
        add_ind = random.randint(0, num_blocks - 1)
        while not offspring[subblock_type][add_ind]:
            add_ind = random.randint(0, num_blocks - 1)

        offspring[subblock_type][remove_ind] = True
        offspring[subblock_type][add_ind] = False

    if drop_entire_block:
        offspring["mlp"] = copy.deepcopy(offspring["attn"])

    return offspring


def count_drop_state_changes(before, after) -> int:
    return sum(
        int(before[subblock_type][i] != after[subblock_type][i])
        for subblock_type in ("attn", "mlp")
        for i in range(len(before[subblock_type]))
    )


def changed_drop_layer_ids(before, after) -> set[int]:
    return {
        i
        for subblock_type in ("attn", "mlp")
        for i in range(len(before[subblock_type]))
        if before[subblock_type][i] != after[subblock_type][i]
    }


LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def quant_layer_is_active(layer_name: str, drop_state) -> bool:
    if drop_state is None:
        return True

    match = LAYER_INDEX_RE.search(layer_name)
    if match is None:
        return True

    layer_id = int(match.group(1))
    if ".self_attn." in layer_name:
        return not drop_state["attn"][layer_id]
    if ".mlp." in layer_name:
        return not drop_state["mlp"][layer_id]
    return True


def candidate_bits(model, grouped_layer_names, quant_state, drop_state=None):
    total = 0
    for group_id, group in enumerate(grouped_layer_names):
        for i, layer_name in enumerate(group):
            if not quant_layer_is_active(layer_name, drop_state):
                continue
            total += model.get_submodule(layer_name).weight.numel() * quant_state[group_id][i]
    return total


def quantizable_weights(model, grouped_layer_names, drop_state=None):
    return sum(
        model.get_submodule(layer_name).weight.numel()
        for group in grouped_layer_names
        for layer_name in group
        if quant_layer_is_active(layer_name, drop_state)
    )


def repair_active_quant_budget(
    grouped_layer_names,
    quant_weights_path,
    quant_state,
    drop_state,
    target_bitwidth: float,
    step_size: int = 1,
):
    """Restore the target average independently within each equal-size group."""
    repaired = copy.deepcopy(quant_state)

    for group_id, group in enumerate(grouped_layer_names):
        active_ids = [
            i for i, layer_name in enumerate(group) if quant_layer_is_active(layer_name, drop_state)
        ]
        if not active_ids:
            continue

        target_level_sum = len(active_ids) * target_bitwidth
        rounded_target = round(target_level_sum)
        if not math.isclose(target_level_sum, rounded_target, abs_tol=1e-9):
            raise ValueError(
                "Active quantization budgeting requires a target bitwidth that "
                "is exactly representable within every active size group."
            )

        current_level_sum = sum(repaired[group_id][i] for i in active_ids)

        while current_level_sum < rounded_target:
            candidates = [
                i
                for i in active_ids
                if os.path.exists(
                    os.path.join(
                        quant_weights_path,
                        group[i],
                        f"{repaired[group_id][i] + step_size}.pth",
                    )
                )
            ]
            if not candidates:
                raise RuntimeError("Unable to restore the active quantization bit budget.")
            candidate_id = random.choice(candidates)
            repaired[group_id][candidate_id] += step_size
            current_level_sum += step_size

        while current_level_sum > rounded_target:
            candidates = [
                i
                for i in active_ids
                if os.path.exists(
                    os.path.join(
                        quant_weights_path,
                        group[i],
                        f"{repaired[group_id][i] - step_size}.pth",
                    )
                )
            ]
            if not candidates:
                raise RuntimeError("Unable to restore the active quantization bit budget.")
            candidate_id = random.choice(candidates)
            repaired[group_id][candidate_id] -= step_size
            current_level_sum -= step_size

    return repaired


def make_initial_quant_state(model, grouped_layer_names, quant_weights_path, target_bitwidth: float):
    """
    Integer target:
        all layers start at that bitwidth.

    Fractional target:
        start from ceil(target) and randomly decrease layers until target average is reached.
        This mirrors evo_quant_search.py's initialization idea.
    """
    if int(target_bitwidth) == target_bitwidth:
        bit = int(target_bitwidth)
        return [[bit for _ in group] for group in grouped_layer_names]

    quantizable_weights = sum(
        model.get_submodule(layer_name).weight.numel()
        for group in grouped_layer_names
        for layer_name in group
    )
    target_bits = int(quantizable_weights * target_bitwidth)

    start_bit = math.ceil(target_bitwidth)
    candidate = [[start_bit for _ in group] for group in grouped_layer_names]
    current_bits = quantizable_weights * start_bit

    while current_bits > target_bits:
        group_id = random.choices(range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names])[0]
        group = grouped_layer_names[group_id]

        decr_ids = []
        for i, layer_name in enumerate(group):
            level = candidate[group_id][i]
            next_level = level - 1
            if os.path.exists(os.path.join(quant_weights_path, layer_name, f"{next_level}.pth")):
                decr_ids.append(i)

        if not decr_ids:
            raise RuntimeError("No valid way to decrease quantization level during initialization.")

        decr_id = random.choice(decr_ids)
        candidate[group_id][decr_id] -= 1
        current_bits -= model.get_submodule(group[decr_id]).weight.numel()

    return candidate


def mutate_quant_state(
    model,
    grouped_layer_names,
    quant_weights_path,
    quant_state,
    step_size: int = 1,
    drop_state=None,
    preferred_layer_ids=None,
):
    """
    Simple budget-preserving mutation:
    within a randomly chosen group, decrease one layer's bitwidth and increase another's.

    For the first prototype, this is intended for same-size groups such as q_proj-only.
    """
    offspring = copy.deepcopy(quant_state)

    valid_groups = []
    for group_id, group in enumerate(grouped_layer_names):
        decr_ids = []
        incr_ids = []
        for i, layer_name in enumerate(group):
            if not quant_layer_is_active(layer_name, drop_state):
                continue

            level = offspring[group_id][i]
            if os.path.exists(os.path.join(quant_weights_path, layer_name, f"{level - step_size}.pth")):
                decr_ids.append(i)
            if os.path.exists(os.path.join(quant_weights_path, layer_name, f"{level + step_size}.pth")):
                incr_ids.append(i)

        valid_pairs = [
            (decr_id, incr_id)
            for decr_id in decr_ids
            for incr_id in incr_ids
            if decr_id != incr_id
        ]
        if preferred_layer_ids:
            valid_pairs = [
                (decr_id, incr_id)
                for decr_id, incr_id in valid_pairs
                if layer_index(group[decr_id]) in preferred_layer_ids
                or layer_index(group[incr_id]) in preferred_layer_ids
            ]
        if valid_pairs:
            valid_groups.append((group_id, valid_pairs))

    if not valid_groups:
        return offspring

    group_id, valid_pairs = random.choices(
        valid_groups,
        weights=[len(pairs) for _, pairs in valid_groups],
    )[0]
    decr_id, incr_id = random.choice(valid_pairs)

    offspring[group_id][decr_id] -= step_size
    offspring[group_id][incr_id] += step_size

    return offspring


def count_quant_state_changes(before, after) -> int:
    return sum(
        int(before[group_id][i] != after[group_id][i])
        for group_id in range(len(before))
        for i in range(len(before[group_id]))
    )


def component_crossover(parent_a, parent_b, *, use_depth_from_a=None):
    """Recombine one parent's depth mask with the other parent's quant state."""
    if parent_a == parent_b:
        raise ValueError("Component crossover parents must be distinct.")
    if use_depth_from_a is None:
        use_depth_from_a = bool(random.getrandbits(1))

    if use_depth_from_a:
        child = {
            "drop": copy.deepcopy(parent_a["drop"]),
            "quant": copy.deepcopy(parent_b["quant"]),
        }
        source_details = {"depth_parent": "a", "quant_parent": "b"}
    else:
        child = {
            "drop": copy.deepcopy(parent_b["drop"]),
            "quant": copy.deepcopy(parent_a["quant"]),
        }
        source_details = {"depth_parent": "b", "quant_parent": "a"}

    return child, source_details


def validate_quant_reconstruction_files(
    grouped_layer_names,
    quant_weights_path,
    quant_state,
) -> None:
    if len(grouped_layer_names) != len(quant_state):
        raise ValueError("Quantization state does not match the number of groups.")

    missing_files = []
    for group_id, group in enumerate(grouped_layer_names):
        if len(group) != len(quant_state[group_id]):
            raise ValueError(
                "Quantization state group length does not match module names."
            )
        for layer_name, level in zip(group, quant_state[group_id]):
            weight_path = os.path.join(
                quant_weights_path,
                layer_name,
                f"{level}.pth",
            )
            if not os.path.isfile(weight_path):
                missing_files.append(weight_path)

    if missing_files:
        raise FileNotFoundError(
            "Crossover candidate references unavailable quantization "
            f"reconstruction files: {missing_files[:3]}"
        )


def try_component_crossover(
    parent_a,
    parent_b,
    *,
    grouped_layer_names,
    quant_weights_path,
    target_bitwidth: float,
    total_blocks: int,
    blocks_to_remove: int,
    active_quant_budget: bool,
    step_size: int = 1,
    drop_entire_block: bool = False,
    use_depth_from_a=None,
):
    """Build, repair, and validate one component-crossover proposal."""
    details = {
        "classification": "component crossover",
        "repair_changed_gene_count": 0,
        "rejection_reason": None,
    }
    try:
        child, source_details = component_crossover(
            parent_a,
            parent_b,
            use_depth_from_a=use_depth_from_a,
        )
        details.update(source_details)
        validate_depth_counts(
            child["drop"],
            total_blocks,
            blocks_to_remove,
            drop_entire_block,
        )

        if active_quant_budget:
            initial_quant = copy.deepcopy(child["quant"])
            child["quant"] = repair_active_quant_budget(
                grouped_layer_names,
                quant_weights_path,
                child["quant"],
                child["drop"],
                target_bitwidth,
                step_size,
            )
            details["repair_changed_gene_count"] = count_quant_state_changes(
                initial_quant,
                child["quant"],
            )
            validate_active_quant_budget(
                grouped_layer_names,
                child["quant"],
                child["drop"],
                target_bitwidth,
            )

        validate_quant_reconstruction_files(
            grouped_layer_names,
            quant_weights_path,
            child["quant"],
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        details["rejection_reason"] = str(error)
        return None, details

    if details["repair_changed_gene_count"] > 0:
        details["classification"] = "component crossover + repair"
    return child, details


def layer_index(layer_name: str) -> Optional[int]:
    match = LAYER_INDEX_RE.search(layer_name)
    return int(match.group(1)) if match is not None else None


def mutate_joint_aware_candidate(
    model,
    grouped_layer_names,
    quant_weights_path,
    candidate,
    target_bitwidth: float,
    step_size: int = 1,
    drop_entire_block: bool = False,
):
    """
    Couple a depth swap with a quantization exchange involving the restored layer.

    The attention drop count is unchanged: one active attention module is dropped
    and one dropped module is restored. The active quantization budget is then
    repaired and a same-group bitwidth exchange explicitly touches the restored
    attention layer.
    """
    offspring = copy.deepcopy(candidate)
    kept_attention = [
        i for i, is_dropped in enumerate(offspring["drop"]["attn"]) if not is_dropped
    ]
    dropped_attention = [
        i for i, is_dropped in enumerate(offspring["drop"]["attn"]) if is_dropped
    ]
    if not kept_attention or not dropped_attention:
        return offspring

    newly_dropped = random.choice(kept_attention)
    newly_restored = random.choice(dropped_attention)
    offspring["drop"]["attn"][newly_dropped] = True
    offspring["drop"]["attn"][newly_restored] = False
    if drop_entire_block:
        offspring["drop"]["mlp"] = copy.deepcopy(offspring["drop"]["attn"])

    offspring["quant"] = repair_active_quant_budget(
        grouped_layer_names,
        quant_weights_path,
        offspring["quant"],
        offspring["drop"],
        target_bitwidth,
        step_size,
    )
    exchanged_quant_state = mutate_quant_state(
        model,
        grouped_layer_names,
        quant_weights_path,
        offspring["quant"],
        step_size,
        offspring["drop"],
        preferred_layer_ids={newly_restored},
    )
    if exchanged_quant_state != offspring["quant"]:
        offspring["quant"] = exchanged_quant_state

    return offspring


def mutate_interaction_aware_candidate(
    model,
    grouped_layer_names,
    quant_weights_path,
    candidate,
    target_bitwidth: float,
    step_size: int = 1,
    drop_entire_block: bool = False,
    max_drop_mutations: int = 1,
    quant_mutations: int = 1,
):
    """
    Coordinated joint mutation for depth pruning + quantization.

    The operator first changes the depth mask using the same budget-preserving
    swap style as the standard depth mutation. It then repairs the active
    quantization budget so dropped modules do not consume the active average
    bitwidth. Finally, it tries to perform a bitwidth exchange involving a
    layer touched by the depth mutation, falling back to any active layer if no
    touched-layer exchange is available.
    """
    offspring = copy.deepcopy(candidate)
    original_drop = copy.deepcopy(offspring["drop"])
    original_quant = copy.deepcopy(offspring["quant"])

    offspring["drop"] = mutate_drop_state(
        offspring["drop"],
        drop_entire_block,
        max_drop_mutations,
    )
    touched_layer_ids = changed_drop_layer_ids(original_drop, offspring["drop"])

    repaired_quant = repair_active_quant_budget(
        grouped_layer_names,
        quant_weights_path,
        offspring["quant"],
        offspring["drop"],
        target_bitwidth,
        step_size,
    )
    budget_repair_changes = count_quant_state_changes(offspring["quant"], repaired_quant)
    offspring["quant"] = repaired_quant

    preferred_quant_changed = False
    fallback_quant_changed = False
    for _ in range(max(1, quant_mutations)):
        before_quant = copy.deepcopy(offspring["quant"])
        mutated_quant = mutate_quant_state(
            model,
            grouped_layer_names,
            quant_weights_path,
            offspring["quant"],
            step_size,
            offspring["drop"],
            preferred_layer_ids=touched_layer_ids,
        )
        if mutated_quant != offspring["quant"]:
            offspring["quant"] = mutated_quant
            preferred_quant_changed = True
            continue

        mutated_quant = mutate_quant_state(
            model,
            grouped_layer_names,
            quant_weights_path,
            offspring["quant"],
            step_size,
            offspring["drop"],
        )
        if mutated_quant != offspring["quant"]:
            offspring["quant"] = mutated_quant
            fallback_quant_changed = True
            continue

        offspring["quant"] = before_quant

    details = {
        "depth_mask_entries_changed": count_drop_state_changes(
            original_drop,
            offspring["drop"],
        ),
        "quant_assignments_changed": count_quant_state_changes(
            original_quant,
            offspring["quant"],
        ),
        "touched_layer_ids": sorted(touched_layer_ids),
        "budget_repair_quant_changes": budget_repair_changes,
        "preferred_quant_exchange_used": preferred_quant_changed,
        "fallback_quant_exchange_used": fallback_quant_changed,
    }
    return offspring, details


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prototype joint EvoPress search: depth pruning + quantization.")

    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--tokenizer_name", default=None, type=str)

    parser.add_argument("--calibration_data", required=True, type=str)
    parser.add_argument("--calibration_tokens", default=2048, type=int)
    parser.add_argument("--calibration_sequence_length", default=None, type=int)

    parser.add_argument("--eval_datasets", nargs="+", default=["wikitext2"], type=str)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--eval_tokens", default=8192, type=int)
    parser.add_argument("--eval_sequence_length", default=None, type=int)

    parser.add_argument("--drop_sparsity", required=True, type=float)
    parser.add_argument("--drop_entire_block", action="store_true")
    parser.add_argument("--max_drop_mutations", default=3, type=int)

    parser.add_argument("--quant_weights_path", required=True, type=str)
    parser.add_argument("--target_bitwidth", required=True, type=float)
    parser.add_argument("--group_rule", default="none", choices=["size", "name", "none"])
    parser.add_argument("--step_size", default=1, type=int)
    parser.add_argument(
        "--active_quant_budget",
        action="store_true",
        help="Keep the target bit average over projections still active after depth pruning.",
    )
    parser.add_argument(
        "--joint_mutation_mode",
        "--joint-mutation-mode",
        default="standard",
        choices=["standard", "interaction_aware"],
        help=(
            "Mutation policy for joint search. 'standard' proposes depth-only "
            "or quantization-only offspring. 'interaction_aware' changes the "
            "depth mask, repairs the active quantization budget, and then "
            "attempts a quantization exchange on touched active layers."
        ),
    )
    parser.add_argument(
        "--joint_aware_mutation",
        action="store_true",
        help=(
            "Add coupled depth/quant offspring that swap an attention module and "
            "exchange bitwidth with the newly restored layer."
        ),
    )
    parser.add_argument(
        "--joint_aware_probability",
        default=0.5,
        type=float,
        help="Probability of a coupled offspring when joint-aware mutation is enabled.",
    )
    parser.add_argument(
        "--adaptive_mutation",
        action="store_true",
        help=(
            "Increase depth-swap and quant-exchange counts after consecutive "
            "generations that retain the parent."
        ),
    )
    parser.add_argument(
        "--adaptive_mutation_patience",
        default=3,
        type=int,
        help="Retained-parent generations required before increasing mutation strength.",
    )
    parser.add_argument(
        "--adaptive_mutation_max_strength",
        default=3,
        type=int,
        help="Maximum depth swaps or quant exchanges per adaptive offspring.",
    )
    parser.add_argument(
        "--coarse_to_fine_mutation",
        action="store_true",
        help=(
            "Decrease the maximum depth swaps from a broad initial value to "
            "a local final value over the search. Quantization offspring "
            "remain one exchange."
        ),
    )
    parser.add_argument(
        "--coarse_to_fine_start_strength",
        default=3,
        type=int,
        help="Maximum depth swaps during the first schedule stage.",
    )
    parser.add_argument(
        "--coarse_to_fine_end_strength",
        default=1,
        type=int,
        help="Maximum depth swaps during the final schedule stage.",
    )

    parser.add_argument("--fitness_fn", default="kl", choices=["ppl", "kl"])

    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--offspring", required=True, type=int)
    parser.add_argument("--initially_generated", required=True, type=int)
    parser.add_argument("--initial_tokens", required=True, type=int)
    parser.add_argument("--survivors_per_selection", nargs="+", required=True, type=int)
    parser.add_argument("--tokens_per_selection", nargs="+", required=True, type=int)
    parser.add_argument(
        "--population_size",
        "--population-size",
        default=1,
        type=int,
        help="Number of unique candidates retained between generations.",
    )
    parser.add_argument(
        "--crossover_probability",
        "--crossover-probability",
        default=0.0,
        type=float,
        help="Probability that an offspring proposal uses component crossover.",
    )
    parser.add_argument(
        "--crossover_type",
        "--crossover-type",
        default="component",
        choices=["component"],
        help="Crossover operator. This pilot supports component crossover only.",
    )

    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--output_dir", default="./outputs/joint_search_tiny")
    parser.add_argument(
        "--sequential_mode",
        default="none",
        choices=SEQUENTIAL_MODES,
        help=(
            "Optional stage-two initialization/freezing policy. The default "
            "'none' preserves the existing joint-search behavior."
        ),
    )
    parser.add_argument(
        "--stage1_run_dir",
        default=None,
        help="Completed depth-only or quant-only run containing final_candidate.json.",
    )
    parser.add_argument(
        "--stage1_candidate",
        default=None,
        help="Direct path to a structured stage-one final_candidate.json.",
    )
    parser.add_argument(
        "--sequential_quant_initialization_policy",
        default="strict",
        choices=["strict", "repair"],
        help=(
            "Quantization-first initialization policy. 'repair' is allowed "
            "only for quant_to_joint_warm."
        ),
    )
    parser.add_argument(
        "--max_initialization_attempts",
        default=100000,
        type=int,
        help=(
            "Bound for random initialization attempts or exact-feasibility "
            "search states."
        ),
    )
    parser.add_argument(
        "--max_offspring_attempts",
        default=10000,
        type=int,
        help="Maximum mutation proposals used to construct one generation's unique offspring.",
    )

    return parser.parse_args(argv)


def validate_crossover_configuration(args) -> None:
    if args.population_size < 1:
        raise ValueError("--population_size must be at least 1.")
    if not 0.0 <= args.crossover_probability <= 1.0:
        raise ValueError("--crossover_probability must be between 0 and 1.")
    if args.crossover_probability > 0.0 and args.population_size < 2:
        raise ValueError(
            "--crossover_probability greater than 0 requires "
            "--population_size at least 2."
        )
    if args.population_size > args.initially_generated:
        raise ValueError(
            "--population_size cannot exceed --initially_generated because "
            "the initial population must contain unique feasible candidates."
        )

    population_extension_requested = (
        args.population_size != 1 or args.crossover_probability != 0.0
    )
    if args.sequential_mode != "none" and population_extension_requested:
        raise ValueError(
            "Component crossover and persistent populations are supported only "
            "with --sequential_mode none in this pilot."
        )


def validate_joint_search_args(args):
    if len(args.survivors_per_selection) != len(args.tokens_per_selection):
        raise ValueError(
            "--survivors_per_selection and --tokens_per_selection must have equal lengths."
        )
    if args.survivors_per_selection[-1] != 1:
        raise ValueError("The final selection stage must have one survivor.")
    if args.active_quant_budget and args.group_rule != "size":
        raise ValueError("--active_quant_budget requires --group_rule size.")
    if args.joint_mutation_mode == "interaction_aware" and not args.active_quant_budget:
        raise ValueError("--joint_mutation_mode interaction_aware requires --active_quant_budget.")
    if args.joint_mutation_mode == "interaction_aware" and args.joint_aware_mutation:
        raise ValueError(
            "--joint_mutation_mode interaction_aware and --joint_aware_mutation "
            "must be ablated separately."
        )
    if args.joint_aware_mutation and not args.active_quant_budget:
        raise ValueError("--joint_aware_mutation requires --active_quant_budget.")
    if args.adaptive_mutation and args.joint_aware_mutation:
        raise ValueError(
            "--adaptive_mutation and --joint_aware_mutation must be ablated separately."
        )
    if args.coarse_to_fine_mutation and (
        args.adaptive_mutation or args.joint_aware_mutation
    ):
        raise ValueError(
            "--coarse_to_fine_mutation must be ablated separately from "
            "adaptive and joint-aware mutation."
        )
    if not 0.0 <= args.joint_aware_probability <= 1.0:
        raise ValueError("--joint_aware_probability must be between 0 and 1.")
    if args.adaptive_mutation_patience < 1:
        raise ValueError("--adaptive_mutation_patience must be at least 1.")
    if args.adaptive_mutation_max_strength < 1:
        raise ValueError("--adaptive_mutation_max_strength must be at least 1.")
    if args.coarse_to_fine_end_strength < 1:
        raise ValueError("--coarse_to_fine_end_strength must be at least 1.")
    if (
        args.coarse_to_fine_start_strength
        < args.coarse_to_fine_end_strength
    ):
        raise ValueError(
            "--coarse_to_fine_start_strength must be greater than or equal "
            "to --coarse_to_fine_end_strength."
        )
    validate_crossover_configuration(args)
    validate_sequential_cli(args)


def main():
    args = parse_args()
    validate_joint_search_args(args)
    effective_selection_survivors = effective_survivors_per_selection(
        args.survivors_per_selection,
        args.population_size,
    )
    crossover_enabled = crossover_is_enabled(
        args.population_size,
        args.crossover_probability,
    )
    stage1_artifacts = (
        resolve_stage1_artifacts(args.stage1_run_dir, args.stage1_candidate)
        if args.sequential_mode != "none"
        else None
    )
    reporter = RunReporter(
        args.output_dir,
        search_type="joint_depth_quant",
        repo_root=os.path.dirname(os.path.abspath(__file__)),
    )

    fix_seed(args.seed)

    device = "cuda"
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings,
        8192,
    )

    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.calibration_sequence_length,
        tokenizer,
        train=True,
    )

    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)

    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,
                args.eval_sequence_length,
                tokenizer,
                train=False,
            )
        )
    eval_tokens_by_dataset = {
        name: sum(sample.numel() for sample in dataset)
        for name, dataset in zip(args.eval_datasets, eval_datasets)
    }

    target_logits = []
    if args.fitness_fn == "kl":
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    # Prepare depth-pruning part.
    layers = get_layers(model)
    attention_module_names = [
        module_name(model, getattr(layer, get_attn_layer_name(model))) for layer in layers
    ]
    mlp_module_names = [
        module_name(model, getattr(layer, get_mlp_layer_name(model))) for layer in layers
    ]
    total_blocks = len(layers)
    blocks_to_remove = int(args.drop_sparsity * total_blocks)
    print(f"Total blocks: {total_blocks}")
    print(f"Drop budget: {blocks_to_remove} blocks/sub-blocks")

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model)))

    # Prepare quantization part.
    layer_names = []
    for layer_name in os.listdir(args.quant_weights_path):
        if os.path.isdir(os.path.join(args.quant_weights_path, layer_name)):
            layer_names.append(layer_name)

    layer_names = sorted(layer_names, key=layer_order_fn)
    grouped_layer_names = group_layers(model, layer_names, args.group_rule)

    print("Quant groups:")
    for group in grouped_layer_names:
        print(group)

    model.state = [[None] * len(names) for names in grouped_layer_names]

    full_quantizable_weights = quantizable_weights(model, grouped_layer_names)
    print(f"Quant budget scope: {'active' if args.active_quant_budget else 'all'}")
    print(f"Joint mutation mode: {args.joint_mutation_mode}")
    print(f"Sequential mode: {args.sequential_mode}")
    print(f"Persistent population size: {args.population_size}")
    print(f"Crossover enabled: {crossover_enabled}")
    print(
        "Effective survivors per selection: "
        f"{effective_selection_survivors}"
    )

    stage1_import = None
    if args.sequential_mode in DEPTH_FIRST_MODES:
        stage1_import = load_stage1_depth_candidate(
            stage1_artifacts,
            expected_model_name=args.model_name_or_path,
            num_layers=total_blocks,
            drop_count=blocks_to_remove,
            drop_entire_block=args.drop_entire_block,
        )
    elif args.sequential_mode in QUANT_FIRST_MODES:
        stage1_import = load_stage1_quant_candidate(
            stage1_artifacts,
            expected_model_name=args.model_name_or_path,
            num_layers=total_blocks,
            grouped_layer_names=grouped_layer_names,
            group_rule=args.group_rule,
            target_bitwidth=args.target_bitwidth,
            quant_weights_path=args.quant_weights_path,
        )

    # Initial joint population.
    initial_candidates = []
    initialization_attempts = 0
    initial_repair_changed_gene_names = []
    if args.sequential_mode == "none":
        while len(initial_candidates) < args.initially_generated:
            initialization_attempts += 1
            if initialization_attempts > args.max_initialization_attempts:
                raise RuntimeError(
                    "Unable to generate requested unique initial candidates: "
                    f"requested={args.initially_generated}, "
                    f"generated={len(initial_candidates)}, "
                    f"attempts={initialization_attempts - 1}, "
                    f"sequential_mode={args.sequential_mode}."
                )
            candidate = {
                "drop": make_random_drop_state(
                    total_blocks,
                    blocks_to_remove,
                    args.drop_entire_block,
                ),
                "quant": make_initial_quant_state(
                    model,
                    grouped_layer_names,
                    args.quant_weights_path,
                    args.target_bitwidth,
                ),
            }
            if args.active_quant_budget:
                candidate["quant"] = repair_active_quant_budget(
                    grouped_layer_names,
                    args.quant_weights_path,
                    candidate["quant"],
                    candidate["drop"],
                    args.target_bitwidth,
                    args.step_size,
                )

            if candidate in initial_candidates:
                continue
            initial_candidates.append(candidate)

    elif args.sequential_mode in DEPTH_FIRST_MODES:
        initial_quant = make_initial_quant_state(
            model,
            grouped_layer_names,
            args.quant_weights_path,
            args.target_bitwidth,
        )
        repaired_quant = copy.deepcopy(initial_quant)
        if args.active_quant_budget:
            repaired_quant = repair_active_quant_budget(
                grouped_layer_names,
                args.quant_weights_path,
                repaired_quant,
                stage1_import.component,
                args.target_bitwidth,
                args.step_size,
            )
        initial_repair_changed_gene_names = changed_quant_gene_names(
            grouped_layer_names,
            initial_quant,
            repaired_quant,
        )
        candidate = {
            "drop": copy.deepcopy(stage1_import.component),
            "quant": repaired_quant,
        }
        validate_depth_counts(
            candidate["drop"],
            total_blocks,
            blocks_to_remove,
            args.drop_entire_block,
        )
        if args.active_quant_budget:
            validate_active_quant_budget(
                grouped_layer_names,
                candidate["quant"],
                candidate["drop"],
                args.target_bitwidth,
            )
        validate_frozen_component(candidate, stage1_import.component, "depth")
        initial_candidates = [candidate]

    elif (
        args.sequential_mode in QUANT_FIRST_MODES
        and args.sequential_quant_initialization_policy == "strict"
    ):
        feasible_depth_states = generate_exact_feasible_depth_states(
            grouped_layer_names,
            stage1_import.component,
            num_layers=total_blocks,
            drop_count=blocks_to_remove,
            target_bitwidth=args.target_bitwidth,
            drop_entire_block=args.drop_entire_block,
            requested_candidates=args.initially_generated,
            max_states=args.max_initialization_attempts,
        )
        initial_candidates = [
            {
                "drop": copy.deepcopy(drop_state),
                "quant": copy.deepcopy(stage1_import.component),
            }
            for drop_state in feasible_depth_states
        ]

    elif args.sequential_mode == "quant_to_joint_warm":
        while len(initial_candidates) < args.initially_generated:
            initialization_attempts += 1
            if initialization_attempts > args.max_initialization_attempts:
                raise RuntimeError(
                    "Unable to generate requested repaired quantization-first "
                    "initial candidates: "
                    f"requested={args.initially_generated}, "
                    f"generated={len(initial_candidates)}, "
                    f"attempts={initialization_attempts - 1}, "
                    f"sequential_mode={args.sequential_mode}."
                )
            drop_state = make_random_drop_state(
                total_blocks,
                blocks_to_remove,
                args.drop_entire_block,
            )
            candidate = {
                "drop": drop_state,
                "quant": repair_active_quant_budget(
                    grouped_layer_names,
                    args.quant_weights_path,
                    copy.deepcopy(stage1_import.component),
                    drop_state,
                    args.target_bitwidth,
                    args.step_size,
                ),
            }
            validate_active_quant_budget(
                grouped_layer_names,
                candidate["quant"],
                candidate["drop"],
                args.target_bitwidth,
            )
            if candidate in initial_candidates:
                continue
            initial_candidates.append(candidate)
    else:
        raise ValueError(f"Unsupported sequential mode: {args.sequential_mode}")

    for candidate in initial_candidates:
        validate_depth_counts(
            candidate["drop"],
            total_blocks,
            blocks_to_remove,
            args.drop_entire_block,
        )
        if args.sequential_mode != "none" and args.active_quant_budget:
            validate_active_quant_budget(
                grouped_layer_names,
                candidate["quant"],
                candidate["drop"],
                args.target_bitwidth,
            )
        if args.sequential_mode == "depth_to_quant_frozen":
            validate_frozen_component(candidate, stage1_import.component, "depth")
        if args.sequential_mode in QUANT_FIRST_MODES and (
            args.sequential_quant_initialization_policy == "strict"
        ):
            validate_frozen_component(
                candidate,
                stage1_import.component,
                "quantization",
            )

    population, train_fitnesses = selection(
        model=model,
        layers=layers,
        grouped_layer_names=grouped_layer_names,
        quant_weights_path=args.quant_weights_path,
        candidates=initial_candidates,
        num_survive=args.population_size,
        calibration_data=calibration_data,
        num_tokens=args.initial_tokens,
        fitness_fn=args.fitness_fn,
        target_logits=target_logits,
    )
    validate_persistent_population(
        population,
        args.population_size,
        context="initial selection",
    )

    parent = population[0]
    train_fitness = train_fitnesses[0]
    initial_parent = copy.deepcopy(parent)
    if (
        args.sequential_mode == "quant_to_joint_warm"
        and args.sequential_quant_initialization_policy == "repair"
    ):
        initial_repair_changed_gene_names = changed_quant_gene_names(
            grouped_layer_names,
            stage1_import.component,
            parent["quant"],
        )
    if args.sequential_mode in DEPTH_FIRST_MODES:
        validate_frozen_component(parent, stage1_import.component, "depth")
    if args.sequential_mode in QUANT_FIRST_MODES and (
        args.sequential_quant_initialization_policy == "strict"
    ):
        validate_frozen_component(parent, stage1_import.component, "quantization")

    initial_fixed_quant_legal_swap_count = None
    if args.sequential_mode == "quant_to_depth_frozen":
        initial_fixed_quant_legal_swap_count = len(
            enumerate_legal_fixed_quant_depth_swaps(
                parent["drop"],
                grouped_layer_names,
                parent["quant"],
                target_bitwidth=args.target_bitwidth,
                drop_entire_block=args.drop_entire_block,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)

    stagnation_generations = 0
    crossover_offspring_attempted_total = 0
    crossover_offspring_accepted_total = 0
    crossover_repair_changed_gene_count_total = 0
    mutation_offspring_accepted_total = 0
    for generation in range(args.generations):
        generation_population = copy.deepcopy(population)
        generation_parent = copy.deepcopy(parent)
        generation_train_fitness = train_fitness
        if args.adaptive_mutation:
            mutation_strength = adaptive_mutation_strength(
                stagnation_generations,
                args.adaptive_mutation_patience,
                args.adaptive_mutation_max_strength,
            )
            depth_mutation_limit = mutation_strength
            quant_mutation_count = mutation_strength
        elif args.coarse_to_fine_mutation:
            mutation_strength = coarse_to_fine_mutation_strength(
                generation + 1,
                args.generations,
                args.coarse_to_fine_start_strength,
                args.coarse_to_fine_end_strength,
            )
            depth_mutation_limit = mutation_strength
            quant_mutation_count = 1
        else:
            mutation_strength = 1
            depth_mutation_limit = args.max_drop_mutations
            quant_mutation_count = 1
        print(f"Generation {generation + 1}/{args.generations}")
        print(f"Train fitness: {train_fitness:.4e}")
        print("Drop config:")
        print(get_layer_drop_config(parent["drop"]))
        print("Quant state:")
        for group in parent["quant"]:
            print(group)
        budget_drop_state = parent["drop"] if args.active_quant_budget else None
        budget_weights = quantizable_weights(model, grouped_layer_names, budget_drop_state)
        print(
            f"Quant bit average: "
            f"{candidate_bits(model, grouped_layer_names, parent['quant'], budget_drop_state) / budget_weights:.4e}"
        )
        if args.active_quant_budget:
            print(
                f"Full quant bit average: "
                f"{candidate_bits(model, grouped_layer_names, parent['quant']) / full_quantizable_weights:.4e}"
            )

        apply_joint_state(model, layers, grouped_layer_names, parent, args.quant_weights_path)

        generation_eval_metrics = {}
        ppl_train = None
        if generation % args.eval_every == 0:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                generation_eval_metrics[eval_dataset_name] = ppl_eval

            ppl_train = compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")

        offspring_list = []
        offspring_mutation_types = []
        mutation_counts = {
            "depth": 0,
            "quantization": 0,
            "joint_aware": 0,
            "interaction_aware": 0,
            "sequential_quantization": 0,
            "fixed_quant_depth": 0,
            "component crossover": 0,
            "component crossover + repair": 0,
        }
        depth_change_totals = {
            "depth": 0,
            "quantization": 0,
            "joint_aware": 0,
            "interaction_aware": 0,
            "sequential_quantization": 0,
            "fixed_quant_depth": 0,
            "component crossover": 0,
            "component crossover + repair": 0,
        }
        quant_change_totals = {
            "depth": 0,
            "quantization": 0,
            "joint_aware": 0,
            "interaction_aware": 0,
            "sequential_quantization": 0,
            "fixed_quant_depth": 0,
            "component crossover": 0,
            "component crossover + repair": 0,
        }
        interaction_aware_totals = {
            "budget_repair_quant_changes": 0,
            "preferred_quant_exchanges_used": 0,
            "fallback_quant_exchanges_used": 0,
        }
        offspring_attempts = 0
        no_op_mutations = 0
        duplicate_candidates = 0
        infeasible_candidates = 0
        fixed_quant_legal_swap_counts = []
        crossover_offspring_attempted = 0
        crossover_offspring_accepted = 0
        crossover_duplicates = 0
        crossover_infeasible_candidates = 0
        crossover_repair_changed_gene_count = 0
        mutation_offspring_accepted = 0

        while len(offspring_list) < args.offspring:
            offspring_attempts += 1
            if offspring_attempts > args.max_offspring_attempts:
                raise RuntimeError(
                    "Unable to generate requested unique offspring within the "
                    "configured bound: "
                    f"requested={args.offspring}, "
                    f"generated={len(offspring_list)}, "
                    f"attempts={offspring_attempts - 1}, "
                    f"no_op_mutations={no_op_mutations}, "
                    f"duplicate_candidates={duplicate_candidates}, "
                    f"infeasible_candidates={infeasible_candidates}, "
                    f"sequential_mode={args.sequential_mode}, "
                    f"frozen_component={sequential_mode_metadata(args.sequential_mode)[2]}, "
                    "fixed_quant_legal_swap_count="
                    f"{fixed_quant_legal_swap_counts[-1] if fixed_quant_legal_swap_counts else None}."
                )
            use_component_crossover = (
                crossover_enabled
                and random.random() < args.crossover_probability
            )
            interaction_details = None
            if use_component_crossover:
                crossover_offspring_attempted += 1
                crossover_offspring_attempted_total += 1
                parent_a, parent_b = select_distinct_parents(population)
                reference_parent = parent_a
                offspring, crossover_details = try_component_crossover(
                    parent_a,
                    parent_b,
                    grouped_layer_names=grouped_layer_names,
                    quant_weights_path=args.quant_weights_path,
                    target_bitwidth=args.target_bitwidth,
                    total_blocks=total_blocks,
                    blocks_to_remove=blocks_to_remove,
                    active_quant_budget=args.active_quant_budget,
                    step_size=args.step_size,
                    drop_entire_block=args.drop_entire_block,
                )
                if offspring is None:
                    infeasible_candidates += 1
                    crossover_infeasible_candidates += 1
                    continue
                crossover_repair_changed_gene_count += crossover_details[
                    "repair_changed_gene_count"
                ]
                crossover_repair_changed_gene_count_total += crossover_details[
                    "repair_changed_gene_count"
                ]
                mutation_type = crossover_details["classification"]
            else:
                reference_parent = select_mutation_parent(population)
                offspring = copy.deepcopy(reference_parent)
                use_joint_aware = (
                    args.joint_aware_mutation
                    and random.random() < args.joint_aware_probability
                )
                if args.sequential_mode == "depth_to_quant_frozen":
                    mutation_type = "sequential_quantization"
                    for _ in range(quant_mutation_count):
                        offspring["quant"] = mutate_quant_state(
                            model,
                            grouped_layer_names,
                            args.quant_weights_path,
                            offspring["quant"],
                            args.step_size,
                            (
                                offspring["drop"]
                                if args.active_quant_budget
                                else None
                            ),
                        )
                elif args.sequential_mode == "quant_to_depth_frozen":
                    mutation_type = "fixed_quant_depth"
                    offspring, fixed_quant_details = mutate_fixed_quant_depth_candidate(
                        offspring,
                        grouped_layer_names,
                        target_bitwidth=args.target_bitwidth,
                        drop_entire_block=args.drop_entire_block,
                        max_mutations=depth_mutation_limit,
                    )
                    fixed_quant_legal_swap_counts.append(
                        fixed_quant_details["legal_swap_count"]
                    )
                    if offspring is None:
                        no_op_mutations += 1
                        continue
                elif args.joint_mutation_mode == "interaction_aware":
                    mutation_type = "interaction_aware"
                    offspring, interaction_details = mutate_interaction_aware_candidate(
                        model,
                        grouped_layer_names,
                        args.quant_weights_path,
                        offspring,
                        args.target_bitwidth,
                        args.step_size,
                        args.drop_entire_block,
                        depth_mutation_limit,
                        quant_mutation_count,
                    )
                elif use_joint_aware:
                    mutation_type = "joint_aware"
                    offspring = mutate_joint_aware_candidate(
                        model,
                        grouped_layer_names,
                        args.quant_weights_path,
                        offspring,
                        args.target_bitwidth,
                        args.step_size,
                        args.drop_entire_block,
                    )
                elif random.random() < 0.5:
                    mutation_type = "depth"
                    offspring["drop"] = mutate_drop_state(
                        offspring["drop"],
                        args.drop_entire_block,
                        depth_mutation_limit,
                    )
                    if args.active_quant_budget:
                        offspring["quant"] = repair_active_quant_budget(
                            grouped_layer_names,
                            args.quant_weights_path,
                            offspring["quant"],
                            offspring["drop"],
                            args.target_bitwidth,
                            args.step_size,
                        )
                else:
                    mutation_type = "quantization"
                    for _ in range(quant_mutation_count):
                        offspring["quant"] = mutate_quant_state(
                            model,
                            grouped_layer_names,
                            args.quant_weights_path,
                            offspring["quant"],
                            args.step_size,
                            (
                                offspring["drop"]
                                if args.active_quant_budget
                                else None
                            ),
                        )

                if args.sequential_mode != "none":
                    validate_depth_counts(
                        offspring["drop"],
                        total_blocks,
                        blocks_to_remove,
                        args.drop_entire_block,
                    )
                    if args.active_quant_budget:
                        validate_active_quant_budget(
                            grouped_layer_names,
                            offspring["quant"],
                            offspring["drop"],
                            args.target_bitwidth,
                        )
                    if args.sequential_mode == "depth_to_quant_frozen":
                        validate_frozen_component(
                            offspring,
                            stage1_import.component,
                            "depth",
                        )
                    elif args.sequential_mode == "quant_to_depth_frozen":
                        validate_frozen_component(
                            offspring,
                            stage1_import.component,
                            "quantization",
                        )

                if offspring == reference_parent:
                    no_op_mutations += 1
                    continue

            if candidate_is_duplicate(offspring, population, offspring_list):
                duplicate_candidates += 1
                if use_component_crossover:
                    crossover_duplicates += 1
                continue

            offspring_list.append(offspring)
            offspring_mutation_types.append(mutation_type)
            mutation_counts[mutation_type] += 1
            depth_change_totals[mutation_type] += count_drop_state_changes(
                reference_parent["drop"],
                offspring["drop"],
            )
            quant_change_totals[mutation_type] += count_quant_state_changes(
                reference_parent["quant"],
                offspring["quant"],
            )
            if use_component_crossover:
                crossover_offspring_accepted += 1
                crossover_offspring_accepted_total += 1
            else:
                mutation_offspring_accepted += 1
                mutation_offspring_accepted_total += 1
            if interaction_details is not None:
                interaction_aware_totals[
                    "budget_repair_quant_changes"
                ] += interaction_details["budget_repair_quant_changes"]
                interaction_aware_totals[
                    "preferred_quant_exchanges_used"
                ] += int(interaction_details["preferred_quant_exchange_used"])
                interaction_aware_totals[
                    "fallback_quant_exchanges_used"
                ] += int(interaction_details["fallback_quant_exchange_used"])

        for selection_stage, (num_survive, num_tokens) in enumerate(
            zip(effective_selection_survivors, args.tokens_per_selection)
        ):
            is_final_selection_stage = (
                selection_stage == len(effective_selection_survivors) - 1
            )
            legacy_elitism_stage = (
                args.population_size == 1
                and args.survivors_per_selection[selection_stage]
                == args.survivors_per_selection[-1]
            )
            if is_final_selection_stage or legacy_elitism_stage:
                offspring_list, offspring_mutation_types = (
                    add_persistent_parents_for_elitism(
                        offspring_list,
                        offspring_mutation_types,
                        generation_population,
                    )
                )

            if (
                is_final_selection_stage
                and unique_candidate_count(offspring_list) < args.population_size
            ):
                raise RuntimeError(
                    "Unable to retain the requested unique feasible persistent "
                    "population before final-stage selection: "
                    f"requested={args.population_size}, "
                    f"available_unique={unique_candidate_count(offspring_list)}."
                )

            candidate_pool = offspring_list
            mutation_type_pool = offspring_mutation_types
            offspring_list, train_fitnesses = selection(
                model=model,
                layers=layers,
                grouped_layer_names=grouped_layer_names,
                quant_weights_path=args.quant_weights_path,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )
            offspring_mutation_types = selected_candidate_metadata(
                offspring_list,
                candidate_pool,
                mutation_type_pool,
            )
            if args.sequential_mode != "none":
                for survivor in offspring_list:
                    validate_depth_counts(
                        survivor["drop"],
                        total_blocks,
                        blocks_to_remove,
                        args.drop_entire_block,
                    )
                    if args.active_quant_budget:
                        validate_active_quant_budget(
                            grouped_layer_names,
                            survivor["quant"],
                            survivor["drop"],
                            args.target_bitwidth,
                        )
                    if args.sequential_mode == "depth_to_quant_frozen":
                        validate_frozen_component(
                            survivor,
                            stage1_import.component,
                            "depth",
                        )
                    elif args.sequential_mode == "quant_to_depth_frozen":
                        validate_frozen_component(
                            survivor,
                            stage1_import.component,
                            "quantization",
                        )

            if is_final_selection_stage:
                validate_persistent_population(
                    offspring_list,
                    args.population_size,
                    context=f"generation {generation + 1} final selection",
                )

        population = offspring_list
        parent = population[0]
        train_fitness = train_fitnesses[0]
        selected_parent_mutation_type = offspring_mutation_types[0]
        accepted_parent_replacement = parent != generation_parent
        if accepted_parent_replacement:
            stagnation_generations = 0
        else:
            stagnation_generations += 1

        generation_depth_details = build_depth_details(
            attention_module_names,
            mlp_module_names,
            generation_parent["drop"],
        )
        generation_bitwidths = flatten_quant_state(
            grouped_layer_names,
            generation_parent["quant"],
        )
        generation_compression = compute_compression_metrics(
            model,
            generation_depth_details,
            generation_bitwidths,
        )
        survivors = list(args.survivors_per_selection)
        if args.sequential_mode == "depth_to_quant_frozen":
            mutation_summary_type = "sequential_quantization_only"
        elif args.sequential_mode == "quant_to_depth_frozen":
            mutation_summary_type = "sequential_fixed_quant_depth_only"
        elif args.joint_mutation_mode == "interaction_aware":
            mutation_summary_type = "interaction_aware_depth_quantization"
        elif args.joint_aware_mutation:
            mutation_summary_type = "joint_aware_depth_quantization"
        else:
            mutation_summary_type = "mixed_depth_quantization"

        reporter.append_generation(
            {
                "generation": generation + 1,
                "best_search_fitness": generation_train_fitness,
                "fitness_fn": args.fitness_fn,
                "best_calibration_kl": None,
                "best_train_ppl": ppl_train,
                "wikitext2_ppl": generation_eval_metrics.get("wikitext2"),
                "c4_ppl": generation_eval_metrics.get("c4"),
                "fineweb_edu_ppl": generation_eval_metrics.get("fineweb_edu"),
                "eval_tokens_used": (
                    sum(eval_tokens_by_dataset[name] for name in generation_eval_metrics)
                    if generation_eval_metrics
                    else 0
                ),
                "eval_tokens_by_dataset": {
                    name: eval_tokens_by_dataset[name] for name in generation_eval_metrics
                },
                "num_offspring": args.offspring,
                "num_survivors_stage_1": survivors[0] if len(survivors) > 0 else None,
                "num_survivors_stage_2": survivors[1] if len(survivors) > 1 else None,
                "num_survivors_stage_3": survivors[2] if len(survivors) > 2 else None,
                "survivors_per_selection": survivors,
                "tokens_per_selection": list(args.tokens_per_selection),
                "active_parameters": generation_compression["parameter_statistics"][
                    "active_parameters"
                ],
                "average_bitwidth_active": generation_compression["quantization_statistics"][
                    "average_bitwidth_active"
                ],
                "estimated_weight_memory_mb": generation_compression["model_size_statistics"][
                    "estimated_weight_memory_mb"
                ],
                "dropped_attention_count": generation_depth_details[
                    "dropped_attention_count"
                ],
                "dropped_mlp_count": generation_depth_details["dropped_mlp_count"],
                "mutation_summary": {
                    "type": mutation_summary_type,
                    "sequential_mode": args.sequential_mode,
                    "joint_mutation_mode": args.joint_mutation_mode,
                    "parent_before_generation_hash": stable_json_hash(
                        generation_parent
                    ),
                    "parent_after_generation_hash": stable_json_hash(parent),
                    "generated_offspring_by_type": mutation_counts,
                    "depth_mask_entries_changed_by_type": depth_change_totals,
                    "quant_assignments_changed_by_type": quant_change_totals,
                    "depth_mask_entries_changed_total": sum(
                        depth_change_totals.values()
                    ),
                    "quant_assignments_changed_total": sum(
                        quant_change_totals.values()
                    ),
                    "interaction_aware_details": interaction_aware_totals,
                    "crossover_diagnostics": {
                        "crossover_offspring": crossover_offspring_accepted,
                        "mutation_offspring": mutation_offspring_accepted,
                        "crossover_offspring_attempted": (
                            crossover_offspring_attempted
                        ),
                        "crossover_duplicates": crossover_duplicates,
                        "crossover_infeasible_candidates": (
                            crossover_infeasible_candidates
                        ),
                        "crossover_repair_changed_gene_count": (
                            crossover_repair_changed_gene_count
                        ),
                        "persistent_population_size": len(population),
                        "unique_population_size": unique_candidate_count(
                            population
                        ),
                        "best_population_fitness": train_fitness,
                        "configured_survivors_per_selection": survivors,
                        "effective_survivors_per_selection": list(
                            effective_selection_survivors
                        ),
                        "effective_final_survivor_count": (
                            effective_selection_survivors[-1]
                        ),
                    },
                    "offspring_generation": {
                        "attempts": offspring_attempts,
                        "no_op_mutations": no_op_mutations,
                        "duplicate_candidates": duplicate_candidates,
                        "infeasible_candidates": infeasible_candidates,
                        "fixed_quant_legal_swap_counts": (
                            fixed_quant_legal_swap_counts
                        ),
                    },
                    "maximum_depth_mutations_per_offspring": depth_mutation_limit,
                    "quantization_step_size": args.step_size,
                    "quantization_mutations_per_offspring": quant_mutation_count,
                    "joint_aware_probability": (
                        args.joint_aware_probability
                        if args.joint_aware_mutation
                        else 0.0
                    ),
                    "adaptive_mutation": args.adaptive_mutation,
                    "adaptive_mutation_strength": mutation_strength,
                    "adaptive_mutation_patience": (
                        args.adaptive_mutation_patience
                    ),
                    "adaptive_mutation_max_strength": (
                        args.adaptive_mutation_max_strength
                    ),
                    "coarse_to_fine_mutation": (
                        args.coarse_to_fine_mutation
                    ),
                    "coarse_to_fine_start_strength": (
                        args.coarse_to_fine_start_strength
                    ),
                    "coarse_to_fine_end_strength": (
                        args.coarse_to_fine_end_strength
                    ),
                    "stagnation_generations_after_selection": (
                        stagnation_generations
                    ),
                },
                "selected_parent_mutation_type": (
                    selected_parent_mutation_type
                ),
                "accepted_parent_replacement": accepted_parent_replacement,
                "runtime_seconds_cumulative": reporter.runtime_seconds(),
                "peak_gpu_memory_mb": peak_gpu_memory()[0],
            }
        )

    final_depth_counts_valid = validate_depth_counts(
        parent["drop"],
        total_blocks,
        blocks_to_remove,
        args.drop_entire_block,
    )
    final_active_budget_valid = None
    if args.active_quant_budget:
        final_active_budget_valid = validate_active_quant_budget(
            grouped_layer_names,
            parent["quant"],
            parent["drop"],
            args.target_bitwidth,
        )
    if args.sequential_mode == "depth_to_quant_frozen":
        validate_frozen_component(parent, stage1_import.component, "depth")
    elif args.sequential_mode == "quant_to_depth_frozen":
        validate_frozen_component(parent, stage1_import.component, "quantization")

    final_fixed_quant_legal_swap_count = None
    if args.sequential_mode == "quant_to_depth_frozen":
        final_fixed_quant_legal_swap_count = len(
            enumerate_legal_fixed_quant_depth_swaps(
                parent["drop"],
                grouped_layer_names,
                parent["quant"],
                target_bitwidth=args.target_bitwidth,
                drop_entire_block=args.drop_entire_block,
            )
        )

    # Save final joint configuration.
    drop_config = get_layer_drop_config(parent["drop"])

    with open(os.path.join(args.output_dir, "joint_drop_config.txt"), "w") as f:
        for line in drop_config:
            f.write(line + "\n")

    with open(os.path.join(args.output_dir, "joint_quant_config.txt"), "w") as f:
        for group_id, group in enumerate(grouped_layer_names):
            for layer_name, level in zip(group, parent["quant"][group_id]):
                f.write(f"{layer_name}: {level}\n")

    with open(os.path.join(args.output_dir, "joint_config.json"), "w") as f:
        json.dump(parent, f, indent=2)

    print("Final joint configuration saved to:")
    print(args.output_dir)

    print("Final drop config:")
    print(drop_config)

    print("Final quant state:")
    for group in parent["quant"]:
        print(group)
    budget_drop_state = parent["drop"] if args.active_quant_budget else None
    budget_weights = quantizable_weights(model, grouped_layer_names, budget_drop_state)
    print(
        f"Final quant bit average: "
        f"{candidate_bits(model, grouped_layer_names, parent['quant'], budget_drop_state) / budget_weights:.4e}"
    )
    if args.active_quant_budget:
        print(
            f"Final full quant bit average: "
            f"{candidate_bits(model, grouped_layer_names, parent['quant']) / full_quantizable_weights:.4e}"
        )
    print(f"Final dropped attention modules: {sum(parent['drop']['attn'])}")
    print(f"Final dropped MLP modules: {sum(parent['drop']['mlp'])}")

    # Final evaluation.
    apply_joint_state(model, layers, grouped_layer_names, parent, args.quant_weights_path)

    final_eval_metrics = {}
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        final_eval_metrics[eval_dataset_name] = ppl_eval

    ppl_train = compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")

    final_calibration_kl = None
    if args.fitness_fn == "kl":
        final_calibration_kl = compute_kl_div(model, calibration_data, target_logits)

    final_depth_details = build_depth_details(
        attention_module_names,
        mlp_module_names,
        parent["drop"],
    )
    final_bitwidths = flatten_quant_state(grouped_layer_names, parent["quant"])
    final_compression = compute_compression_metrics(
        model,
        final_depth_details,
        final_bitwidths,
    )
    final_candidate = build_final_candidate(
        "joint_depth_quant",
        final_depth_details,
        final_bitwidths,
        parent,
    )
    final_candidate_path = reporter.write_final_candidate(final_candidate)
    sequential_summary = build_sequential_summary_metadata(
        mode=args.sequential_mode,
        stage1_import=stage1_import,
        quant_initialization_policy=(
            args.sequential_quant_initialization_policy
        ),
        initial_repair_changed_gene_names=initial_repair_changed_gene_names,
        initial_candidate_count=len(initial_candidates),
        initial_parent=initial_parent,
        final_parent=parent,
        initial_fixed_quant_legal_swap_count=(
            initial_fixed_quant_legal_swap_count
        ),
        final_fixed_quant_legal_swap_count=(
            final_fixed_quant_legal_swap_count
        ),
        active_budget_valid=final_active_budget_valid,
        depth_counts_valid=final_depth_counts_valid,
    )
    crossover_summary = {
        "population_size": args.population_size,
        "crossover_probability": args.crossover_probability,
        "crossover_type": args.crossover_type,
        "crossover_enabled": crossover_enabled,
        "crossover_offspring_attempted": crossover_offspring_attempted_total,
        "crossover_offspring_accepted": crossover_offspring_accepted_total,
        "crossover_repair_changed_gene_count": (
            crossover_repair_changed_gene_count_total
        ),
        "mutation_offspring_accepted": mutation_offspring_accepted_total,
        "population_unique_count": unique_candidate_count(population),
        "effective_final_survivor_count": effective_selection_survivors[-1],
    }
    reporter.write_summary(
        model_name=args.model_name_or_path,
        dataset_calibration=args.calibration_data,
        dataset_eval=args.eval_datasets,
        search_config={
            "generations": args.generations,
            "offspring": args.offspring,
            "initial_candidates": args.initially_generated,
            "initial_tokens": args.initial_tokens,
            "selection_tokens": list(args.tokens_per_selection),
            "selection_survivors": list(args.survivors_per_selection),
            "fitness_fn": args.fitness_fn,
            "sequence_length": args.calibration_sequence_length,
            "calibration_tokens": args.calibration_tokens,
            "eval_tokens": args.eval_tokens,
            "eval_tokens_loaded_by_dataset": eval_tokens_by_dataset,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "sequential_mode": args.sequential_mode,
            "max_initialization_attempts": args.max_initialization_attempts,
            "max_offspring_attempts": args.max_offspring_attempts,
            "initial_candidates_evaluated": len(initial_candidates),
            "population_size": args.population_size,
            "crossover_probability": args.crossover_probability,
            "crossover_type": args.crossover_type,
            "crossover_enabled": crossover_enabled,
            "configured_survivors_per_selection": list(
                args.survivors_per_selection
            ),
            "effective_survivors_per_selection": list(
                effective_selection_survivors
            ),
            "effective_final_survivor_count": (
                effective_selection_survivors[-1]
            ),
        },
        compression_config={
            "target_depth_sparsity": args.drop_sparsity,
            "target_average_bitwidth": args.target_bitwidth,
            "bits_available": available_bitwidths(args.quant_weights_path),
            "group_size": None,
            "group_rule": args.group_rule,
            "active_quant_budget": args.active_quant_budget,
            "joint_mutation_mode": args.joint_mutation_mode,
            "joint_aware_mutation": args.joint_aware_mutation,
            "joint_aware_probability": (
                args.joint_aware_probability
                if args.joint_aware_mutation
                else 0.0
            ),
            "adaptive_mutation": args.adaptive_mutation,
            "adaptive_mutation_patience": args.adaptive_mutation_patience,
            "adaptive_mutation_max_strength": (
                args.adaptive_mutation_max_strength
            ),
            "coarse_to_fine_mutation": args.coarse_to_fine_mutation,
            "coarse_to_fine_start_strength": (
                args.coarse_to_fine_start_strength
            ),
            "coarse_to_fine_end_strength": (
                args.coarse_to_fine_end_strength
            ),
            "drop_entire_block": args.drop_entire_block,
            "quant_weights_path": args.quant_weights_path,
        },
        final_metrics={
            "best_search_fitness": train_fitness,
            "final_calibration_kl": final_calibration_kl,
            "wikitext2_ppl": final_eval_metrics.get("wikitext2"),
            "c4_ppl": final_eval_metrics.get("c4"),
            "fineweb_ppl": final_eval_metrics.get("fineweb_edu"),
            "train_ppl": ppl_train,
        },
        parameter_statistics=final_compression["parameter_statistics"],
        depth_statistics={
            key: value
            for key, value in final_depth_details.items()
            if key not in {"kept_modules", "attention_mask", "mlp_mask"}
        },
        quantization_statistics=final_compression["quantization_statistics"],
        model_size_statistics=final_compression["model_size_statistics"],
        artifacts={
            "candidate_path": final_candidate_path,
            "generation_log_path": os.path.join(args.output_dir, "generation_log.csv"),
            "config_path": os.path.join(args.output_dir, "joint_config.json"),
            "depth_config_path": os.path.join(args.output_dir, "joint_drop_config.txt"),
            "quant_config_path": os.path.join(args.output_dir, "joint_quant_config.txt"),
            "stdout_log_path": os.path.join(args.output_dir, "run.log"),
            "stage1_candidate_path": (
                stage1_import.candidate_path if stage1_import else None
            ),
        },
        extra_summary={**sequential_summary, **crossover_summary},
    )


if __name__ == "__main__":
    main()
