import argparse
import random
import copy
import os
import math
from tqdm import trange
from typing import List, Tuple, Sequence, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.metrics import compute_perplexity, compute_kl_div, compute_sparse_kl_div
from src.model_utils import (
    get_attn_layer_name,
    get_layers,
    get_mlp_layer_name,
    group_layers,
    layer_order_fn,
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


def load_layers(
    model: AutoModelForCausalLM,
    grouped_layer_names: Tuple[Sequence[str]],
    new_state: Tuple[Sequence[int]],
    quant_weights_path: str,
):
    assert hasattr(model, "state")
    num_groups = len(grouped_layer_names)
    for i in range(num_groups):
        for layer_name, new_level, old_level in zip(grouped_layer_names[i], new_state[i], model.state[i]):
            if new_level != old_level:
                layer = model.get_submodule(layer_name)
                layer.weight.data = torch.load(
                    os.path.join(quant_weights_path, layer_name, f"{new_level}.pth"), map_location=layer.weight.device
                ).to(layer.weight.dtype)
    # Update model state
    model.state = new_state


def compute_fitness(model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    elif fitness_fn == "kl":
        return compute_kl_div(model, data, target_logits)
    elif fitness_fn == "sparse_kl":
        return compute_sparse_kl_div(model, data, target_logits)


def selection(
    model,
    grouped_layer_names,
    quant_weights_path: str,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    fitness_fn: str = "ppl",
    target_logits: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor]]] = None,
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0
    while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
        minibatch_id = random.randint(0, len(calibration_data) - 1)
        if minibatch_id in minibatch_ids:  # avoid duplicates
            continue
        minibatch_ids.append(minibatch_id)
        if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
            calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
            elif fitness_fn == "sparse_kl":
                target_logits_minibatch.append(
                    (
                        target_logits[minibatch_id][0][:, : num_tokens - tokens_used],  # TopK indices
                        target_logits[minibatch_id][1][:, : num_tokens - tokens_used],  # TopK values
                    )
                )
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(calibration_data[minibatch_id])
            if fitness_fn in ["kl", "sparse_kl"]:
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += calibration_data[minibatch_id].shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    fitnesses = []
    for candidate in candidates:
        load_layers(model, grouped_layer_names, candidate, quant_weights_path)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, target_logits_minibatch)
        fitnesses.append(fitness)
    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=524288, type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl", "sparse_kl"], default="kl", help="Fitness function.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search params
    parser.add_argument("--generations", type=int, required=True, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated in each generation")
    parser.add_argument(
        "--target_bitwidth",
        type=float,
        required=True,
        help="Base level for all layers. If no integer, initialize random with this average",
    )
    parser.add_argument("--quant_weights_path", type=str, required=True, help="Path to quantized weights")
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--tokens_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of calibration tokens at each stage of selection",
    )
    parser.add_argument(
        "--initially_generated",
        type=int,
        help="Only for non-integer initial level: Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        help="Only for non-integer initial level: Number of calibration tokens used for the initial generation",
    )
    parser.add_argument(
        "--group_rule",
        type=str,
        default="size",
        choices=["size", "name", "none"],
        help="Layer grouping rule. Mutations are performed only within a group.",
    )
    parser.add_argument(
        "--kl_topk",
        type=int,
        default=10,
        help="TopK logits in KL-divergence (for sparse_kl fitness function)",
    )
    # TODO infer automatically from configuration
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size between adjacent levels",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument(
        "--configuration_name",
        type=str,
        default=None,
        help="Filename for the final quantization configuration. Defaults to the legacy generated name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for structured run_summary.json, generation_log.csv, and final_candidate.json.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reporter = RunReporter(
        args.output_dir,
        search_type="quant_only",
        repo_root=os.path.dirname(os.path.abspath(__file__)),
    )
    # Sanity checks
    assert len(args.survivors_per_selection) == len(args.tokens_per_selection), "Must have same number of stages"
    assert args.survivors_per_selection[-1] == 1, "Last stage should have only one survivor"
    if int(args.target_bitwidth) != args.target_bitwidth:
        assert args.initially_generated is not None, "Need initially_generated for non-integer initial level"
        assert args.initial_tokens is not None, "Need initial_tokens for non-integer initial level"
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # init device
    device = f"cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache
    transformer_layers = get_layers(model)
    attention_module_names = [
        module_name(model, getattr(layer, get_attn_layer_name(model)))
        for layer in transformer_layers
    ]
    mlp_module_names = [
        module_name(model, getattr(layer, get_mlp_layer_name(model)))
        for layer in transformer_layers
    ]
    no_depth_details = build_depth_details(
        attention_module_names,
        mlp_module_names,
        None,
    )
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # Load eval datasets
    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)
    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.eval_sequence_length,
                tokenizer,
                train=False,
            )
        )
    target_logits = []
    if args.fitness_fn == "kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    elif args.fitness_fn == "sparse_kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                logits = model(calibration_data[i].to(device)).logits.cpu()
                topk_values, topk_indices = logits.topk(k=args.kl_topk, dim=-1)
                target_logits.append((topk_values, topk_indices))

    # Prepare layers and initial state
    layer_names = []
    for layer_name in os.listdir(args.quant_weights_path):
        if os.path.isdir(os.path.join(args.quant_weights_path, layer_name)):
            layer_names.append(layer_name)
    # Sort layers
    layer_names = sorted(layer_names, key=layer_order_fn)
    # Group layers
    grouped_layer_names = group_layers(model, layer_names, args.group_rule)
    print(grouped_layer_names)
    num_groups = len(grouped_layer_names)
    # Loaded state
    model.state = [[None] * len(names) for names in grouped_layer_names]

    target_bits = 0
    quantizable_weights = 0
    for group_id in range(len(grouped_layer_names)):
        for i, layer_name in enumerate(grouped_layer_names[group_id]):
            target_bits += int(model.get_submodule(layer_name).weight.numel() * args.target_bitwidth)
            quantizable_weights += model.get_submodule(layer_name).weight.numel()

    # Initialization
    if (
        int(args.target_bitwidth) == args.target_bitwidth
    ):  # TODO: What if target bitwidth is integer, but not available (e.g. 4/8 with 5bit average)
        parent = [[int(args.target_bitwidth) for _ in names] for names in grouped_layer_names]
        initial_population, train_fitnesses = selection(
            model=model,
            grouped_layer_names=grouped_layer_names,
            quant_weights_path=args.quant_weights_path,
            candidates=[parent],
            num_survive=1,
            calibration_data=calibration_data,
            num_tokens=args.initial_tokens or args.tokens_per_selection[0],
            fitness_fn=args.fitness_fn,
            target_logits=target_logits,
        )
        parent = initial_population[0]
        train_fitness = train_fitnesses[0]
    else:
        candidates = []
        for _ in range(args.initially_generated):
            # Start with all bitwidths rounded up and decrease bitwidths randomly until target bitwidth achieved

            candidate = [[math.ceil(args.target_bitwidth) for _ in names] for names in grouped_layer_names]
            candidate_bits = quantizable_weights * math.ceil(args.target_bitwidth)

            while candidate_bits > target_bits:
                # Select random group, proportional to the number of layers in a group
                group_id = random.choices(
                    range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                )[0]
                group = grouped_layer_names[group_id]

                decr_ids = []
                for i, layer_name in enumerate(group):
                    level = candidate[group_id][i]
                    if os.path.exists(
                        os.path.join(args.quant_weights_path, layer_name, f"{level - args.step_size}.pth")
                    ):
                        decr_ids.append(i)
                assert len(decr_ids) > 0, "There is no way to decrease compression level."
                decr_id = random.choice(decr_ids)

                candidate[group_id][decr_id] -= args.step_size
                candidate_bits -= model.get_submodule(group[decr_id]).weight.numel() * args.step_size

            candidates.append(candidate)

        candidates, train_fitnesses = selection(
            model=model,
            grouped_layer_names=grouped_layer_names,
            quant_weights_path=args.quant_weights_path,
            candidates=candidates,
            num_survive=1,
            calibration_data=calibration_data,
            num_tokens=args.initial_tokens,
            fitness_fn=args.fitness_fn,
            target_logits=target_logits,
        )
        train_fitness = train_fitnesses[0]
        parent = candidates[0]


    log_dict = {}
    for generation in range(args.generations):
        generation_parent = copy.deepcopy(parent)
        generation_train_fitness = train_fitness
        parent_bits = 0
        for group_id in range(len(grouped_layer_names)):
            for i, layer_name in enumerate(grouped_layer_names[group_id]):
                parent_bits += model.get_submodule(layer_name).weight.numel() * parent[group_id][i]
                
        print(f"Generation {generation + 1}/{args.generations}")
        print(f"Current search point:")
        for group in parent:
            print(group)
        print(f"Parent bits: {parent_bits}")
        print(f"Bit average: {parent_bits/quantizable_weights:.4e}")
        print(f"Train fitness: {train_fitness:.4e}")

        load_layers(model, grouped_layer_names, parent, args.quant_weights_path)

        # Evaluate current search point
        generation_eval_metrics = {}
        ppl_train = None
        if generation % args.eval_every == 0:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
                generation_eval_metrics[eval_dataset_name] = ppl_eval
            ppl_train = compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")
            log_dict["ppl_train"] = ppl_train
        if args.log_wandb:
            wandb.log(log_dict)

        offspring_list = []

        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, 3), random.randint(1, 3))  # bias towards lower values

            if args.group_rule == "none":  # there can be mutations between layers of different sizes
                offspring_bits = parent_bits
                bits_added = 0
                bits_removed = 0

                for _ in range(num_flips):  # increase levels
                    # Select random group, proportional to the number of layers in a group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]
                    group = grouped_layer_names[group_id]

                    incr_ids = []
                    for i, layer_name in enumerate(group):
                        level = offspring[group_id][i]
                        if os.path.exists(
                            os.path.join(args.quant_weights_path, layer_name, f"{level + args.step_size}.pth")
                        ):
                            incr_ids.append(i)
                    assert len(incr_ids) > 0, "There is no way to increase compression level."
                    incr_id = random.choice(incr_ids)

                    offspring[group_id][incr_id] += args.step_size
                    offspring_bits += model.get_submodule(group[incr_id]).weight.numel() * args.step_size
                    bits_added += model.get_submodule(group[incr_id]).weight.numel() * args.step_size

                number_level_changes = num_flips
                while offspring_bits > target_bits:  # Decrease levels until target bitwidth satisfied
                    number_level_changes += 1

                    # Select random group, proportional to the number of layers in a group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]
                    group = grouped_layer_names[group_id]

                    decr_ids = []
                    for i, layer_name in enumerate(group):
                        level = offspring[group_id][i]
                        if os.path.exists(
                            os.path.join(args.quant_weights_path, layer_name, f"{level - args.step_size}.pth")
                        ):
                            decr_ids.append(i)
                    assert len(decr_ids) > 0, "There is no way to decrease compression level."
                    decr_id = random.choice(decr_ids)

                    offspring[group_id][decr_id] -= args.step_size
                    offspring_bits -= model.get_submodule(group[decr_id]).weight.numel() * args.step_size
                    bits_removed += model.get_submodule(group[decr_id]).weight.numel() * args.step_size

                if number_level_changes > 10:  # Avoid too many mutations
                    continue

                if bits_added / max(1, bits_removed) < 0.8:  # Avoid offspring with too few bits
                    continue

            else:  # only mutations between layers of same size/type
                for _ in range(num_flips):
                    # Select random group, proportional to the number of layers in a group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]
                    group = grouped_layer_names[group_id]

                    # Positions where compression can be decreased
                    decr_ids = []
                    for i, layer_name in enumerate(group):
                        level = offspring[group_id][i]
                        if os.path.exists(
                            os.path.join(args.quant_weights_path, layer_name, f"{level - args.step_size}.pth")
                        ):
                            decr_ids.append(i)
                    assert len(decr_ids) > 0, "There is no way to decrease compression level."
                    decr_id = random.choice(decr_ids)
                    # Positions where compression can be increased
                    incr_ids = []
                    for i, layer_name in enumerate(group):
                        level = offspring[group_id][i]
                        if os.path.exists(
                            os.path.join(args.quant_weights_path, layer_name, f"{level + args.step_size}.pth")
                        ):
                            incr_ids.append(i)
                    assert len(incr_ids) > 0, "There is no way to increase compression level."
                    incr_id = random.choice(incr_ids)

                    offspring[group_id][decr_id] -= args.step_size
                    offspring[group_id][incr_id] += args.step_size

            if offspring in offspring_list or offspring in [parent]:  # Avoid duplicates
                continue
            offspring_list.append(offspring)

        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                if parent not in offspring_list:  # Elitist EA
                    offspring_list.append(parent)
            offspring_list, train_fitnesses = selection(
                model=model,
                grouped_layer_names=grouped_layer_names,
                quant_weights_path=args.quant_weights_path,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )
        # In the end we have lists with a single element (only 1 survivor in last selection step)
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        print(f"Train fitnesses: {train_fitness:.2e}")
        log_dict["train_fitness"] = train_fitness

        generation_bitwidths = flatten_quant_state(
            grouped_layer_names,
            generation_parent,
        )
        generation_compression = compute_compression_metrics(
            model,
            no_depth_details,
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
                "eval_tokens_used": args.eval_tokens if generation_eval_metrics else 0,
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
                "dropped_attention_count": 0,
                "dropped_mlp_count": 0,
                "mutation_summary": {
                    "type": "quantization_level_switch",
                    "offspring_generated": args.offspring,
                    "step_size": args.step_size,
                    "group_rule": args.group_rule,
                },
                "accepted_parent_replacement": parent != generation_parent,
                "runtime_seconds_cumulative": reporter.runtime_seconds(),
                "peak_gpu_memory_mb": peak_gpu_memory()[0],
            }
        )
    # Save final configuration
    configuration_name = args.configuration_name or f"evo-{args.fitness_fn}-configuration-{args.target_bitwidth}.txt"
    with open(os.path.join(args.quant_weights_path, configuration_name), "w") as f:
        for i in range(num_groups):
            f.write(
                "\n".join([f"{layer_name}: {level}" for layer_name, level in zip(grouped_layer_names[i], parent[i])])
            )
            if i != num_groups - 1:
                f.write("\n")
    # Log final configuration
    print("Final configuration:")
    for group in parent:
        print(group)
    # Final evaluation
    load_layers(model, grouped_layer_names, parent, args.quant_weights_path)
    final_eval_metrics = {}
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
        final_eval_metrics[eval_dataset_name] = ppl_eval
    ppl_train = compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")
    log_dict["ppl_train"] = ppl_train
    if args.log_wandb:
        wandb.log(log_dict)

    final_calibration_kl = None
    if args.fitness_fn == "kl":
        final_calibration_kl = compute_kl_div(model, calibration_data, target_logits)

    final_bitwidths = flatten_quant_state(grouped_layer_names, parent)
    final_compression = compute_compression_metrics(
        model,
        no_depth_details,
        final_bitwidths,
    )
    final_candidate = build_final_candidate(
        "quant_only",
        no_depth_details,
        final_bitwidths,
        parent,
    )
    final_candidate_path = reporter.write_final_candidate(final_candidate)
    output_dir = args.output_dir
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
            "eval_every": args.eval_every,
            "seed": args.seed,
        },
        compression_config={
            "target_depth_sparsity": 0.0,
            "target_average_bitwidth": args.target_bitwidth,
            "bits_available": available_bitwidths(args.quant_weights_path),
            "group_size": None,
            "group_rule": args.group_rule,
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
            for key, value in no_depth_details.items()
            if key not in {"kept_modules", "attention_mask", "mlp_mask"}
        },
        quantization_statistics=final_compression["quantization_statistics"],
        model_size_statistics=final_compression["model_size_statistics"],
        artifacts={
            "candidate_path": final_candidate_path,
            "generation_log_path": (
                os.path.join(output_dir, "generation_log.csv") if output_dir else None
            ),
            "config_path": (
                os.path.join(output_dir, "quant_configuration.txt") if output_dir else configuration_name
            ),
            "stdout_log_path": os.path.join(output_dir, "run.log") if output_dir else None,
        },
    )


if __name__ == "__main__":
    main()
