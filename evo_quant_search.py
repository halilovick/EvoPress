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
from src.compression_budget import (
    candidate_compression_cost,
    inspect_quantization_database,
    uniform_quantization_target_cost,
    validate_exact_budget,
)
from src.metrics import compute_perplexity, compute_kl_div, compute_sparse_kl_div
from src.teacher_logits_cache import DiskTensorCache
from src.search_checkpoint import (
    load_search_checkpoint,
    restore_rng_state,
    save_search_checkpoint,
    validate_checkpoint_identity,
)
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
    parser.add_argument(
        "--compression_budget_mode",
        default="legacy_average_bitwidth",
        choices=["legacy_average_bitwidth", "match_uniform_quantization_total"],
        help=(
            "Use the legacy searched-weight average or derive and validate the "
            "exact total-model cost of the uniform target."
        ),
    )
    parser.add_argument("--quantization_group_size", default=None, type=int)
    parser.add_argument(
        "--budget_include_quantization_metadata",
        action="store_true",
    )
    parser.add_argument("--budget_scale_bits", default=16, type=int)
    parser.add_argument("--budget_zero_point_bits", default=16, type=int)
    parser.add_argument("--budget_dense_dtype_bits", default=16, type=int)
    parser.add_argument("--expected_dense_model_bits", default=None, type=int)
    parser.add_argument("--expected_target_cost_bits", default=None, type=int)
    parser.add_argument("--expected_bitwidths", nargs="+", type=int, default=None)
    parser.add_argument("--expected_quantized_modules", type=int, default=None)
    parser.add_argument(
        "--skip_initial_uniform_evaluation",
        action="store_true",
        help="Match the original EvoPress integer-target initialization exactly.",
    )
    parser.add_argument(
        "--max_offspring_attempts",
        default=1000000,
        type=int,
        help="Bound duplicate/infeasible proposals while constructing one generation.",
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
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume from a generation-level search_checkpoint.pt. "
            "Model/data/teacher logits are rebuilt before RNG/search state is restored."
        ),
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
    exact_total_budget = (
        args.compression_budget_mode == "match_uniform_quantization_total"
    )
    if exact_total_budget and args.group_rule != "size":
        raise ValueError(
            "Exact paper comparison requires --group_rule size."
        )
    if exact_total_budget and int(args.target_bitwidth) != args.target_bitwidth:
        raise ValueError("The uniform exact-budget reference must use an integral bit-width.")
    if exact_total_budget and args.budget_include_quantization_metadata:
        if args.quantization_group_size is None:
            raise ValueError(
                "--quantization_group_size is required when metadata is included."
            )
    if args.max_offspring_attempts < args.offspring:
        raise ValueError("--max_offspring_attempts must be at least --offspring.")
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
    eval_tokens_by_dataset = {
        name: sum(sample.numel() for sample in dataset)
        for name, dataset in zip(args.eval_datasets, eval_datasets)
    }
    target_logits = []
    if args.fitness_fn == "kl":
        # Dense KL is unchanged mathematically. Only storage changes: each
        # teacher-logit tensor is serialized losslessly and released from RAM.
        target_logits = DiskTensorCache.temporary(
            prefix="evopress-quant-teacher-logits",
        )
        print(f"Dense teacher logits cache: {target_logits.root}")
        for i in trange(
            0,
            len(calibration_data),
            desc="Computing target logits (calib)",
            leave=False,
        ):
            with torch.no_grad():
                target_logits.append(
                    model(calibration_data[i].to(device)).logits
                )

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
    if args.expected_quantized_modules is not None and len(layer_names) != args.expected_quantized_modules:
        raise ValueError(
            "Quantization database module-count mismatch: "
            f"expected={args.expected_quantized_modules}, actual={len(layer_names)}."
        )
    database_audit = inspect_quantization_database(
        args.quant_weights_path,
        expected_module_names=layer_names,
        expected_bitwidths=args.expected_bitwidths,
    )
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

    budget_cost_kwargs = {
        "attention_module_names": attention_module_names,
        "mlp_module_names": mlp_module_names,
        "dense_dtype_bits": args.budget_dense_dtype_bits,
        "group_size": args.quantization_group_size,
        "include_quantization_metadata": args.budget_include_quantization_metadata,
        "scale_bits": args.budget_scale_bits,
        "zero_point_bits": args.budget_zero_point_bits,
    }
    uniform_reference_cost = None
    exact_target_cost_bits = None
    if exact_total_budget:
        uniform_reference_cost = uniform_quantization_target_cost(
            model,
            grouped_layer_names,
            int(args.target_bitwidth),
            **budget_cost_kwargs,
        )
        exact_target_cost_bits = int(uniform_reference_cost["total_cost_bits"])
        if (
            args.expected_dense_model_bits is not None
            and uniform_reference_cost["dense_model_bits"]
            != args.expected_dense_model_bits
        ):
            raise ValueError(
                "Live model dense cost does not match the configured reference: "
                f"expected={args.expected_dense_model_bits}, "
                f"actual={uniform_reference_cost['dense_model_bits']}."
            )
        if (
            args.expected_target_cost_bits is not None
            and exact_target_cost_bits != args.expected_target_cost_bits
        ):
            raise ValueError(
                "Live model uniform target does not match the configured reference: "
                f"expected={args.expected_target_cost_bits}, "
                f"actual={exact_target_cost_bits}."
            )
        print(f"Exact common compression target: {exact_target_cost_bits} bits")
        print(
            "Exact common compression target size: "
            f"{uniform_reference_cost['total_cost_mib']:.6f} MiB"
        )
        print(
            "Exact common compression ratio: "
            f"{uniform_reference_cost['compression_ratio']:.12f}x"
        )

    # Initialization
    if (
        int(args.target_bitwidth) == args.target_bitwidth
    ):  # TODO: What if target bitwidth is integer, but not available (e.g. 4/8 with 5bit average)
        parent = [[int(args.target_bitwidth) for _ in names] for names in grouped_layer_names]
        if args.skip_initial_uniform_evaluation:
            initial_candidate_evaluations = 0
            initial_evaluation_tokens = 0
            train_fitness = float("inf")
        else:
            initial_eval_token_count = args.initial_tokens or args.tokens_per_selection[0]
            initial_population, train_fitnesses = selection(
                model=model,
                grouped_layer_names=grouped_layer_names,
                quant_weights_path=args.quant_weights_path,
                candidates=[parent],
                num_survive=1,
                calibration_data=calibration_data,
                num_tokens=initial_eval_token_count,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )
            parent = initial_population[0]
            train_fitness = train_fitnesses[0]
            initial_candidate_evaluations = 1
            initial_evaluation_tokens = initial_eval_token_count
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

        initial_generated_count = len(candidates)
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
        initial_candidate_evaluations = initial_generated_count
        initial_evaluation_tokens = initial_generated_count * args.initial_tokens

    if exact_total_budget:
        initial_cost = candidate_compression_cost(
            model,
            parent,
            grouped_layer_names=grouped_layer_names,
            **budget_cost_kwargs,
        )
        validate_exact_budget(
            initial_cost,
            exact_target_cost_bits,
            context="initial quantization-only candidate",
        )


    log_dict = {}
    offspring_attempts_total = 0
    candidate_evaluations_search_cumulative = initial_candidate_evaluations
    evaluation_tokens_search_cumulative = initial_evaluation_tokens

    checkpoint_identity = {
        "model_name_or_path": args.model_name_or_path,
        "quant_weights_path": os.path.realpath(args.quant_weights_path),
        "seed": args.seed,
        "generations": args.generations,
        "offspring": args.offspring,
        "survivors_per_selection": list(args.survivors_per_selection),
        "tokens_per_selection": list(args.tokens_per_selection),
        "fitness_fn": args.fitness_fn,
        "target_bitwidth": float(args.target_bitwidth),
        "group_rule": args.group_rule,
        "step_size": args.step_size,
        "compression_budget_mode": args.compression_budget_mode,
        "quantization_group_size": args.quantization_group_size,
        "expected_target_cost_bits": args.expected_target_cost_bits,
        "grouped_layer_names": [
            list(group)
            for group in grouped_layer_names
        ],
    }

    checkpoint_path = (
        os.path.join(args.output_dir, "search_checkpoint.pt")
        if args.output_dir is not None
        else None
    )
    resume_source_checkpoint = None
    resumed_from_generation = 0
    start_generation = 0

    if args.resume_checkpoint is not None:
        if args.output_dir is None:
            raise ValueError(
                "--resume_checkpoint requires --output_dir so new checkpoints "
                "can be persisted."
            )

        resume_source_checkpoint = os.path.realpath(
            args.resume_checkpoint
        )
        checkpoint = load_search_checkpoint(
            resume_source_checkpoint,
            expected_search_type="quant_only",
        )
        validate_checkpoint_identity(
            checkpoint,
            checkpoint_identity,
        )

        state = checkpoint["state"]
        parent = copy.deepcopy(state["parent"])
        train_fitness = float(state["train_fitness"])
        initial_candidate_evaluations = int(
            state["initial_candidate_evaluations"]
        )
        initial_evaluation_tokens = int(
            state["initial_evaluation_tokens"]
        )
        offspring_attempts_total = int(
            state["offspring_attempts_total"]
        )
        candidate_evaluations_search_cumulative = int(
            state["candidate_evaluations_search_cumulative"]
        )
        evaluation_tokens_search_cumulative = int(
            state["evaluation_tokens_search_cumulative"]
        )

        start_generation = int(
            checkpoint["completed_generation"]
        )
        resumed_from_generation = start_generation

        if not 0 <= start_generation <= args.generations:
            raise ValueError(
                "Checkpoint completed generation is outside the configured "
                f"range: {start_generation}/{args.generations}."
            )

        if exact_total_budget:
            resumed_cost = candidate_compression_cost(
                model,
                parent,
                grouped_layer_names=grouped_layer_names,
                **budget_cost_kwargs,
            )
            validate_exact_budget(
                resumed_cost,
                exact_target_cost_bits,
                context="resumed quantization-only parent",
            )

        reporter.set_runtime_offset_seconds(
            float(state["runtime_seconds_cumulative"])
        )

        # Restore only after all model/data/cache/startup work has finished.
        restore_rng_state(checkpoint["rng_state"])

        print(
            "Resuming quantization search from completed generation "
            f"{start_generation}; next generation is "
            f"{start_generation + 1 if start_generation < args.generations else 'final evaluation'}."
        )

    def quant_checkpoint_state():
        return {
            "parent": copy.deepcopy(parent),
            "train_fitness": float(train_fitness),
            "initial_candidate_evaluations": int(
                initial_candidate_evaluations
            ),
            "initial_evaluation_tokens": int(
                initial_evaluation_tokens
            ),
            "offspring_attempts_total": int(
                offspring_attempts_total
            ),
            "candidate_evaluations_search_cumulative": int(
                candidate_evaluations_search_cumulative
            ),
            "evaluation_tokens_search_cumulative": int(
                evaluation_tokens_search_cumulative
            ),
            "runtime_seconds_cumulative": float(
                reporter.runtime_seconds()
            ),
        }

    if checkpoint_path is not None:
        # Generation 0 is useful too: a crash during the first generation can
        # restart from the exact post-initialization RNG/search state.
        save_search_checkpoint(
            checkpoint_path,
            search_type="quant_only",
            completed_generation=start_generation,
            identity=checkpoint_identity,
            state=quant_checkpoint_state(),
        )

    for generation in range(start_generation, args.generations):
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
        offspring_attempts = 0

        while len(offspring_list) < args.offspring:
            offspring_attempts += 1
            if offspring_attempts > args.max_offspring_attempts:
                raise RuntimeError(
                    "Unable to generate requested unique quantization offspring: "
                    f"requested={args.offspring}, generated={len(offspring_list)}, "
                    f"attempts={offspring_attempts - 1}."
                )
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
            if exact_total_budget:
                offspring_cost = candidate_compression_cost(
                    model,
                    offspring,
                    grouped_layer_names=grouped_layer_names,
                    **budget_cost_kwargs,
                )
                validate_exact_budget(
                    offspring_cost,
                    exact_target_cost_bits,
                    context="quantization-only offspring",
                )
            offspring_list.append(offspring)

        stage_candidate_evaluations = []
        stage_evaluation_tokens = []
        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                if parent not in offspring_list:  # Elitist EA
                    offspring_list.append(parent)
            stage_candidate_evaluations.append(len(offspring_list))
            stage_evaluation_tokens.append(len(offspring_list) * num_tokens)
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
        offspring_attempts_total += offspring_attempts
        candidate_evaluations_search_cumulative += sum(stage_candidate_evaluations)
        evaluation_tokens_search_cumulative += sum(stage_evaluation_tokens)
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
            quantization_group_size=(
                args.quantization_group_size if exact_total_budget else None
            ),
            include_quantization_metadata=(
                args.budget_include_quantization_metadata if exact_total_budget else False
            ),
            scale_bits=args.budget_scale_bits,
            zero_point_bits=args.budget_zero_point_bits,
        )
        generation_cost_bits = generation_compression["model_size_statistics"][
            "compression_cost_bits"
        ]
        survivors = list(args.survivors_per_selection)
        reporter.append_generation(
            {
                "generation": generation + 1,
                "best_search_fitness": train_fitness,
                "fitness_fn": args.fitness_fn,
                "best_calibration_kl": (
                    train_fitness if args.fitness_fn == "kl" else None
                ),
                "parent_search_fitness_before_generation": (
                    generation_train_fitness
                ),
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
                "compression_cost_bits": generation_cost_bits,
                "compression_target_bits": exact_target_cost_bits,
                "compression_difference_bits": (
                    generation_cost_bits - exact_target_cost_bits
                    if exact_target_cost_bits is not None
                    else None
                ),
                "dropped_attention_count": 0,
                "dropped_mlp_count": 0,
                "mutation_summary": {
                    "type": "quantization_level_switch",
                    "offspring_generated": args.offspring,
                    "step_size": args.step_size,
                    "group_rule": args.group_rule,
                },
                "accepted_parent_replacement": parent != generation_parent,
                "offspring_attempts": offspring_attempts,
                "candidate_evaluations_stage_1": (
                    stage_candidate_evaluations[0]
                    if len(stage_candidate_evaluations) > 0
                    else None
                ),
                "candidate_evaluations_stage_2": (
                    stage_candidate_evaluations[1]
                    if len(stage_candidate_evaluations) > 1
                    else None
                ),
                "candidate_evaluations_stage_3": (
                    stage_candidate_evaluations[2]
                    if len(stage_candidate_evaluations) > 2
                    else None
                ),
                "evaluation_tokens_stage_1": (
                    stage_evaluation_tokens[0]
                    if len(stage_evaluation_tokens) > 0
                    else None
                ),
                "evaluation_tokens_stage_2": (
                    stage_evaluation_tokens[1]
                    if len(stage_evaluation_tokens) > 1
                    else None
                ),
                "evaluation_tokens_stage_3": (
                    stage_evaluation_tokens[2]
                    if len(stage_evaluation_tokens) > 2
                    else None
                ),
                "candidate_evaluations_search_cumulative": (
                    candidate_evaluations_search_cumulative
                ),
                "evaluation_tokens_search_cumulative": (
                    evaluation_tokens_search_cumulative
                ),
                "runtime_seconds_cumulative": reporter.runtime_seconds(),
                "peak_gpu_memory_mb": peak_gpu_memory()[0],
            }
        )

        if checkpoint_path is not None:
            save_search_checkpoint(
                checkpoint_path,
                search_type="quant_only",
                completed_generation=generation + 1,
                identity=checkpoint_identity,
                state=quant_checkpoint_state(),
            )

    # Save final configuration
    configuration_name = args.configuration_name or f"evo-{args.fitness_fn}-configuration-{args.target_bitwidth}.txt"
    configuration_path = (
        os.path.join(args.output_dir, "quant_configuration.txt")
        if args.output_dir
        else os.path.join(args.quant_weights_path, configuration_name)
    )
    with open(configuration_path, "w") as f:
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
        quantization_group_size=(
            args.quantization_group_size if exact_total_budget else None
        ),
        include_quantization_metadata=(
            args.budget_include_quantization_metadata if exact_total_budget else False
        ),
        scale_bits=args.budget_scale_bits,
        zero_point_bits=args.budget_zero_point_bits,
        dense_dtype_bits=(
            args.budget_dense_dtype_bits if exact_total_budget else None
        ),
    )
    final_cost_bits = final_compression["model_size_statistics"]["compression_cost_bits"]
    final_exact_budget_valid = None
    if exact_total_budget:
        final_exact_budget_valid = validate_exact_budget(
            final_compression["model_size_statistics"],
            exact_target_cost_bits,
            context="final quantization-only candidate",
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
            "eval_tokens_loaded_by_dataset": eval_tokens_by_dataset,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "resume_source_checkpoint": resume_source_checkpoint,
            "resumed_from_generation": resumed_from_generation,
            "skip_initial_uniform_evaluation": args.skip_initial_uniform_evaluation,
            "initial_candidate_evaluations": initial_candidate_evaluations,
            "initial_evaluation_tokens": initial_evaluation_tokens,
            "offspring_attempts_total": offspring_attempts_total,
            "offspring_generated_total": args.generations * args.offspring,
            "candidate_evaluations_search_total": (
                candidate_evaluations_search_cumulative
            ),
            "evaluation_tokens_search_total": evaluation_tokens_search_cumulative,
        },
        compression_config={
            "target_depth_sparsity": 0.0,
            "target_average_bitwidth": args.target_bitwidth,
            "bits_available": available_bitwidths(args.quant_weights_path),
            "group_size": args.quantization_group_size,
            "group_rule": args.group_rule,
            "compression_budget_mode": args.compression_budget_mode,
            "preserve_equal_size_group_costs": exact_total_budget,
            "target_cost_bits": exact_target_cost_bits,
            "expected_dense_model_bits": args.expected_dense_model_bits,
            "expected_target_cost_bits": args.expected_target_cost_bits,
            "target_cost_bytes": (
                uniform_reference_cost["total_cost_bytes"]
                if uniform_reference_cost is not None
                else None
            ),
            "target_cost_mib": (
                uniform_reference_cost["total_cost_mib"]
                if uniform_reference_cost is not None
                else None
            ),
            "target_compression_ratio": (
                uniform_reference_cost["compression_ratio"]
                if uniform_reference_cost is not None
                else None
            ),
            "target_paper_weight_only_cost_bits": (
                uniform_reference_cost["fixed_precision_bits"]
                + uniform_reference_cost["quantized_weight_bits"]
                if uniform_reference_cost is not None
                else None
            ),
            "include_quantization_metadata": (
                args.budget_include_quantization_metadata
            ),
            "scale_bits": args.budget_scale_bits,
            "zero_point_bits": args.budget_zero_point_bits,
            "database_module_count": database_audit["module_count"],
            "quant_weights_path": args.quant_weights_path,
        },
        final_metrics={
            "best_search_fitness": train_fitness,
            "final_calibration_kl": final_calibration_kl,
            "wikitext2_ppl": final_eval_metrics.get("wikitext2"),
            "c4_ppl": final_eval_metrics.get("c4"),
            "fineweb_ppl": final_eval_metrics.get("fineweb_edu"),
            "train_ppl": ppl_train,
            "compression_target_bits": exact_target_cost_bits,
            "compression_realized_bits": final_cost_bits,
            "compression_difference_bits": (
                final_cost_bits - exact_target_cost_bits
                if exact_target_cost_bits is not None
                else None
            ),
            "compression_difference_percent": (
                100.0 * (final_cost_bits - exact_target_cost_bits) / exact_target_cost_bits
                if exact_target_cost_bits is not None
                else None
            ),
            "exact_budget_valid": final_exact_budget_valid,
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
                configuration_path
            ),
            "stdout_log_path": os.path.join(output_dir, "run.log") if output_dir else None,
            "checkpoint_path": checkpoint_path,
            "resume_source_checkpoint": resume_source_checkpoint,
        },
    )


if __name__ == "__main__":
    main()
