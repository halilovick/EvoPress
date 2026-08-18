# Quantized weight database generation step.
#
# This script does not perform the EvoPress evolutionary search itself.
# Instead, it prepares the candidate quantization database used later by
# evo_quant_search.py. For each selected linear projection layer, it collects
# calibration activations, estimates a Hessian/input covariance matrix, and uses
# a GPTQ/FastOBQ-style backend to generate quantized versions of the layer at
# several bitwidths.
#
# The saved files are organized as:
#   save_dir/<model_name>/<calibration_bitwidth>bit/<layer_name>/<bit>.pth
#
# Each file contains the dequantized float reconstruction of the quantized
# weight tensor for that bitwidth. During database generation, the model is
# processed block by block. After a layer is quantized, the calibration bitwidth
# version is inserted back into the model as QLinear so later blocks are
# calibrated on activations coming from already-quantized previous blocks.
#
# The later evolutionary search chooses one saved bitwidth per layer while
# satisfying a global average bitwidth budget.

import os
import argparse
import importlib.metadata
import platform
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from src import dist_utils
from src.calibration_utils import configured_calibration_partition
from src.common_utils import fix_seed
from src.data_utils import get_data
from src.quantizer import Quantizer
from src.run_reporting import get_git_commit, utc_now, write_json


def installed_package_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot quantization with parallel GPTQ.")
    parser.add_argument(
        "--configured_torchrun_processes",
        type=int,
        default=None,
        help=(
            "Process count from the experiment profile. It defines the logical "
            "calibration prefix and is recorded so a smaller physical-process "
            "override remains visible in the database manifest."
        ),
    )
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being quantized",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--quantizable_modules",
        type=str,
        required=True,
        help="Regex for modules to quantize",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    parser.add_argument(
        "--post_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules after transformer blocks",
    )
    ## Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=int(2**23), type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--bitwidth_options",
        nargs="+",
        type=int,
        required=True,
        help="List of bitwidths to quantize the model.",
    )
    parser.add_argument(
        "--calibration_bitwidth",
        type=int,
        required=True,
        help="Quantization bitwidth loaded to produce hessian. Must be in bitwidth_options.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Whether to permute in activation order.",
    )
    parser.add_argument("--sym", action="store_true", help="Whether to use symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Log to W&B")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument(
        "--drop_saved_file_cache",
        action="store_true",
        help="Flush and evict each saved candidate file from Linux page cache to reduce cgroup RAM pressure.",
    )
    parser.add_argument("--new_eval", action="store_true", help="whether to use new evaluation setup.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse model.")
    args = parser.parse_args()
    if (
        args.configured_torchrun_processes is not None
        and args.configured_torchrun_processes < 1
    ):
        parser.error("--configured_torchrun_processes must be at least 1.")
    return args


def main():
    args = parse_args()
    fix_seed(args.seed)
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    visible_cuda_device_count = torch.cuda.device_count()
    if rank >= visible_cuda_device_count:
        raise RuntimeError(
            f"Distributed rank {rank} cannot map to cuda:{rank}; only "
            f"{visible_cuda_device_count} CUDA device(s) are visible."
        )
    configured_torchrun_processes = args.configured_torchrun_processes or world_size
    # init device
    device = f"cuda:{rank}"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # init W&B logger
    if args.log_wandb and dist_utils.is_main():
        wandb.init(config=args)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    dense_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_revision = getattr(model.config, "_commit_hash", None)
    print(model)
    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_revision is None and hasattr(tokenizer, "init_kwargs"):
        tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    calibration_sequence_count_loaded = len(calibration_data)
    calibration_token_count_loaded = sum(
        int(input_ids.shape[-1]) for input_ids in calibration_data
    )
    # Preserve the configured multi-process calibration semantics even when a
    # smaller physical world size is explicitly used. Upstream splits with
    # floor division, so it drops the tail that cannot be divided equally among
    # the configured workers. Truncating to that same prefix before re-sharding
    # makes a one-process run use the same calibration examples as an eight-
    # process run; only floating-point accumulation/reduction order can differ.
    logical_shard_count = configured_torchrun_processes
    calibration_sequence_count_used, partition_start, partition_end = (
        configured_calibration_partition(
            calibration_sequence_count_loaded,
            logical_shard_count,
            world_size,
            rank,
        )
    )
    calibration_data = calibration_data[:calibration_sequence_count_used]
    calibration_token_count_used = sum(
        int(input_ids.shape[-1]) for input_ids in calibration_data
    )
    sequences_per_effective_rank = calibration_sequence_count_used // world_size
    calibration_token_counts_by_effective_rank = [
        sum(
            int(input_ids.shape[-1])
            for input_ids in calibration_data[
                effective_rank
                * sequences_per_effective_rank : (effective_rank + 1)
                * sequences_per_effective_rank
            ]
        )
        for effective_rank in range(world_size)
    ]
    sequences_per_logical_shard = (
        calibration_sequence_count_used // logical_shard_count
    )
    calibration_token_counts_by_logical_shard = [
        sum(
            int(input_ids.shape[-1])
            for input_ids in calibration_data[
                logical_rank
                * sequences_per_logical_shard : (logical_rank + 1)
                * sequences_per_logical_shard
            ]
        )
        for logical_rank in range(logical_shard_count)
    ]
    # Take an equal slice for each effective worker.
    if dist_utils.is_dist_available_and_initialized():
        calibration_data = calibration_data[partition_start:partition_end]
    calibration_sequence_count_per_rank = len(calibration_data)
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    # Quantizer
    if args.calibration_bitwidth not in args.bitwidth_options:
        raise ValueError(f"Calibration bitwidth {args.calibration_bitwidth} is not in bitwidth_options.")
    # Move calibration_bitwidth to last position (last bitwidth is used for hessian)
    args.bitwidth_options = [bits for bits in args.bitwidth_options if bits != args.calibration_bitwidth] + [
        args.calibration_bitwidth
    ]
    dist_utils.print_on_main(f"Bitwidth options: {args.bitwidth_options}")
    # Override save dir name
    args.save_dir = os.path.join(
        args.save_dir, args.model_name_or_path.split("/")[-1], f"{args.calibration_bitwidth}bit"
    )
    quantizer = Quantizer(
        model,
        calibration_data,
        quantizable_modules=args.quantizable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        obq_kwargs=dict(
            rel_damp=args.rel_damp,
            block_size=args.block_size,
            perchannel=args.perchannel,
            group_size=args.group_size,
            sym=args.sym,
            act_order=args.act_order,
        ),
        save_dir=args.save_dir,
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        drop_saved_file_cache=args.drop_saved_file_cache,
        verbose=args.verbose,
    )
    tokenizer_provenance = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_revision": tokenizer_revision,
    }
    execution_provenance = {
        "configured_torchrun_processes": configured_torchrun_processes,
        "distributed_world_size": world_size,
        "torchrun_process_override": configured_torchrun_processes != world_size,
        "visible_cuda_device_count": visible_cuda_device_count,
        "visible_cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(visible_cuda_device_count)
        ],
        "calibration_sequence_count_loaded": calibration_sequence_count_loaded,
        "calibration_sequence_count_used": calibration_sequence_count_used,
        "calibration_sequence_count_per_rank": calibration_sequence_count_per_rank,
        "calibration_logical_shard_count": logical_shard_count,
        "calibration_token_count_loaded": calibration_token_count_loaded,
        "calibration_token_count_used": calibration_token_count_used,
        "calibration_token_count_rank0": calibration_token_counts_by_effective_rank[0],
        "calibration_token_counts_by_effective_rank": (
            calibration_token_counts_by_effective_rank
        ),
        "calibration_token_counts_by_logical_shard": (
            calibration_token_counts_by_logical_shard
        ),
        "calibration_dropped_sequence_count": (
            calibration_sequence_count_loaded - calibration_sequence_count_used
        ),
        "calibration_dropped_token_count": (
            calibration_token_count_loaded - calibration_token_count_used
        ),
        "software_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "transformers": installed_package_version("transformers"),
            "datasets": installed_package_version("datasets"),
            "flash_attn": installed_package_version("flash-attn"),
        },
    }
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        manifest_path = os.path.join(args.save_dir, "quant_database_manifest.json")
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "generating",
                "timestamp_start": utc_now(),
                "git_commit": get_git_commit(os.path.dirname(os.path.abspath(__file__))),
                "model_name": args.model_name_or_path,
                "model_revision": model_revision,
                **tokenizer_provenance,
                **execution_provenance,
                "total_parameters_dense": dense_parameter_count,
                "attention_implementation": args.attn_implementation,
                "quantizable_modules_regex": args.quantizable_modules,
                "bitwidth_options": sorted(args.bitwidth_options),
                "calibration_bitwidth": args.calibration_bitwidth,
                "group_size": args.group_size,
                "perchannel": args.perchannel,
                "symmetric": args.sym,
                "activation_order": args.act_order,
                "relative_dampening": args.rel_damp,
                "block_size": args.block_size,
                "calibration_data": args.calibration_data,
                "calibration_tokens": args.calibration_tokens,
                "calibration_sequence_length": args.calibration_sequence_length,
                "seed": args.seed,
                "database_representation": (
                    "dequantized floating-point reconstruction tensors; not packed storage"
                ),
            },
        )

    dist.barrier()

    t1 = time.perf_counter()
    quantizer.quantize(args.bitwidth_options, args.calibration_bitwidth)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Quantization took {(t2 - t1)} s.")
    if dist_utils.is_main():
        module_names = sorted(
            name
            for name in os.listdir(args.save_dir)
            if os.path.isdir(os.path.join(args.save_dir, name))
        )
        levels_by_module = {
            name: sorted(
                int(filename[:-4])
                for filename in os.listdir(os.path.join(args.save_dir, name))
                if filename.endswith(".pth") and filename[:-4].isdigit()
            )
            for name in module_names
        }
        quantized_weight_parameters = sum(
            int(model.get_submodule(name).in_features)
            * int(model.get_submodule(name).out_features)
            for name in module_names
        )
        write_json(
            os.path.join(args.save_dir, "quant_database_manifest.json"),
            {
                "schema_version": 1,
                "status": "complete",
                "timestamp_end": utc_now(),
                "git_commit": get_git_commit(os.path.dirname(os.path.abspath(__file__))),
                "model_name": args.model_name_or_path,
                "model_revision": model_revision,
                **tokenizer_provenance,
                **execution_provenance,
                "total_parameters_dense": dense_parameter_count,
                "attention_implementation": args.attn_implementation,
                "quantized_weight_parameters": quantized_weight_parameters,
                "fixed_parameters": dense_parameter_count - quantized_weight_parameters,
                "quantizable_modules_regex": args.quantizable_modules,
                "bitwidth_options": sorted(args.bitwidth_options),
                "calibration_bitwidth": args.calibration_bitwidth,
                "group_size": args.group_size,
                "perchannel": args.perchannel,
                "symmetric": args.sym,
                "activation_order": args.act_order,
                "relative_dampening": args.rel_damp,
                "block_size": args.block_size,
                "calibration_data": args.calibration_data,
                "calibration_tokens": args.calibration_tokens,
                "calibration_sequence_length": args.calibration_sequence_length,
                "seed": args.seed,
                "module_count": len(module_names),
                "module_names": module_names,
                "levels_by_module": levels_by_module,
                "runtime_seconds": t2 - t1,
                "database_representation": (
                    "dequantized floating-point reconstruction tensors; not packed storage"
                ),
            },
        )


if __name__ == "__main__":
    main()
