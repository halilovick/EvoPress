import argparse
import copy
import json
import math
import os
import random
import re
from typing import List, Optional, Sequence, Tuple, Union

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

        valid_pairs = [(decr_id, incr_id) for decr_id in decr_ids for incr_id in incr_ids if decr_id != incr_id]
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


def parse_args():
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

    parser.add_argument("--fitness_fn", default="kl", choices=["ppl", "kl"])

    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--offspring", required=True, type=int)
    parser.add_argument("--initially_generated", required=True, type=int)
    parser.add_argument("--initial_tokens", required=True, type=int)
    parser.add_argument("--survivors_per_selection", nargs="+", required=True, type=int)
    parser.add_argument("--tokens_per_selection", nargs="+", required=True, type=int)

    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--output_dir", default="./outputs/joint_search_tiny")

    return parser.parse_args()


def main():
    args = parse_args()
    reporter = RunReporter(
        args.output_dir,
        search_type="joint_depth_quant",
        repo_root=os.path.dirname(os.path.abspath(__file__)),
    )

    assert len(args.survivors_per_selection) == len(args.tokens_per_selection)
    assert args.survivors_per_selection[-1] == 1
    if args.active_quant_budget and args.group_rule != "size":
        raise ValueError("--active_quant_budget requires --group_rule size.")

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

    # Initial joint population.
    initial_candidates = []
    while len(initial_candidates) < args.initially_generated:
        candidate = {
            "drop": make_random_drop_state(total_blocks, blocks_to_remove, args.drop_entire_block),
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

    population, train_fitnesses = selection(
        model=model,
        layers=layers,
        grouped_layer_names=grouped_layer_names,
        quant_weights_path=args.quant_weights_path,
        candidates=initial_candidates,
        num_survive=1,
        calibration_data=calibration_data,
        num_tokens=args.initial_tokens,
        fitness_fn=args.fitness_fn,
        target_logits=target_logits,
    )

    parent = population[0]
    train_fitness = train_fitnesses[0]

    os.makedirs(args.output_dir, exist_ok=True)

    for generation in range(args.generations):
        generation_parent = copy.deepcopy(parent)
        generation_train_fitness = train_fitness
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
        mutation_counts = {"depth": 0, "quantization": 0}

        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(parent)

            if random.random() < 0.5:
                mutation_type = "depth"
                offspring["drop"] = mutate_drop_state(
                    offspring["drop"],
                    args.drop_entire_block,
                    args.max_drop_mutations,
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
                offspring["quant"] = mutate_quant_state(
                    model,
                    grouped_layer_names,
                    args.quant_weights_path,
                    offspring["quant"],
                    args.step_size,
                    offspring["drop"] if args.active_quant_budget else None,
                )

            if offspring in offspring_list or offspring == parent:
                continue

            offspring_list.append(offspring)
            mutation_counts[mutation_type] += 1

        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                if parent not in offspring_list:
                    offspring_list.append(parent)

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

        parent = offspring_list[0]
        train_fitness = train_fitnesses[0]

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
                    "type": "mixed_depth_quantization",
                    "accepted_offspring_by_type": mutation_counts,
                    "maximum_depth_mutations_per_offspring": args.max_drop_mutations,
                    "quantization_step_size": args.step_size,
                },
                "accepted_parent_replacement": parent != generation_parent,
                "runtime_seconds_cumulative": reporter.runtime_seconds(),
                "peak_gpu_memory_mb": peak_gpu_memory()[0],
            }
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
        },
        compression_config={
            "target_depth_sparsity": args.drop_sparsity,
            "target_average_bitwidth": args.target_bitwidth,
            "bits_available": available_bitwidths(args.quant_weights_path),
            "group_size": None,
            "group_rule": args.group_rule,
            "active_quant_budget": args.active_quant_budget,
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
        },
    )


if __name__ == "__main__":
    main()
