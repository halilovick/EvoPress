# Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py

import argparse
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from lm_eval import evaluator, utils
from lm_eval.utils import make_table
try:
    from lm_eval.tasks import TaskManager
except ImportError:
    TaskManager = None

try:
    from lm_eval.tasks import include_path, initialize_tasks
except ImportError:
    include_path = None
    initialize_tasks = None

try:
    from lm_eval.api.registry import ALL_TASKS
except ImportError:
    ALL_TASKS = None

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.model_utils import drop_layers_from_config


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`."),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    # Loading params
    parser.add_argument("--drop_layer_config", type=str, default=None, help="Path to layer dropping configuration.")
    # Sparsification params
    parser.add_argument(
        "--sparse_weights_path",
        type=str,
        default=None,
        help="Path to sparse weights",
    )
    parser.add_argument(
        "--sparse_config_path",
        type=str,
        default=None,
        help="Path to sparsification config",
    )
    parser.add_argument(
        "--sparse_default_level",
        type=int,
        default=0,
        help="Default sparsity level",
    )
    # Quantization params
    parser.add_argument(
        "--quant_weights_path",
        type=str,
        default=None,
        help="Path to quantized weights",
    )
    parser.add_argument(
        "--quant_config_path",
        type=str,
        default=None,
        help="Path to quantization config",
    )
    parser.add_argument(
        "--quant_default_level",
        type=int,
        default=0,
        help="Default quantization level",
    )
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")

    return parser.parse_args()


# Compressed model loader
def load_compressed_weights(
    model: AutoModelForCausalLM,
    compressed_weights_path: Union[str, os.PathLike],
    compressed_config_path: Optional[str] = None,
    default_level: int = 0,
):
    # Load weights from configuration if provided
    if compressed_config_path:
        with open(os.path.join(compressed_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(compressed_weights_path, layer_name, f"{int(level)}.pth"),
                    map_location=layer.weight.device,
                ).to(orig_dtype)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(compressed_weights_path)):
            if not os.path.isdir(os.path.join(compressed_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(compressed_weights_path, layer_name, f"{default_level}.pth"),
                map_location=layer.weight.device,
            ).to(orig_dtype)
    return model


def make_task_manager(args: argparse.Namespace):
    if TaskManager is None:
        return None

    attempts = (
        lambda: TaskManager(args.verbosity, include_path=args.include_path),
        lambda: TaskManager(verbosity=args.verbosity, include_path=args.include_path),
        lambda: TaskManager(args.verbosity),
        lambda: TaskManager(verbosity=args.verbosity),
        lambda: TaskManager(),
    )
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def all_task_names(task_manager) -> list[str] | None:
    if task_manager is not None:
        for attribute in ("all_tasks", "task_names"):
            value = getattr(task_manager, attribute, None)
            if value is not None:
                return list(value)
        task_index = getattr(task_manager, "task_index", None)
        if isinstance(task_index, dict):
            return list(task_index)
    if ALL_TASKS is not None:
        return list(ALL_TASKS)
    return None


def resolve_tasks(args: argparse.Namespace, eval_logger, task_manager) -> list:
    available_tasks = all_task_names(task_manager)

    if args.tasks is None:
        if available_tasks is None:
            raise ValueError(
                "This lm-eval version does not expose a task registry. "
                "Pass explicit --tasks instead of leaving it empty."
            )
        return available_tasks

    if args.tasks == "list":
        if available_tasks is None:
            raise ValueError("This lm-eval version does not expose task listing.")
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(sorted(available_tasks)))
        )
        sys.exit()

    if os.path.isdir(args.tasks):
        import glob

        task_names = []
        yaml_path = os.path.join(args.tasks, "*.yaml")
        for yaml_file in glob.glob(yaml_path):
            config = utils.load_yaml_config(yaml_file)
            task_names.append(config)
        return task_names

    tasks_list = args.tasks.split(",")
    task_names = []
    if available_tasks is not None:
        task_names = utils.pattern_match(tasks_list, available_tasks)
    else:
        task_names = [task for task in tasks_list if not os.path.isfile(task)]

    for task in [task for task in tasks_list if task not in task_names]:
        if os.path.isfile(task):
            config = utils.load_yaml_config(task)
            task_names.append(config)

    if available_tasks is not None:
        task_missing = [
            task for task in tasks_list if task not in task_names and "*" not in task
        ]
        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks list` "
                "for list of available tasks, or '--verbosity DEBUG' to "
                "troubleshoot task registration issues."
            )

    return task_names


def simple_evaluate_compat(task_manager, **kwargs):
    signature = inspect.signature(evaluator.simple_evaluate)
    supports_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if task_manager is not None and "task_manager" in signature.parameters:
        kwargs["task_manager"] = task_manager
    if not supports_arbitrary_kwargs:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return evaluator.simple_evaluate(**kwargs)


def get_eval_logger(verbosity: str) -> logging.Logger:
    eval_logger = getattr(utils, "eval_logger", None)
    if eval_logger is None:
        eval_logger = logging.getLogger("lm_eval")
    eval_logger.setLevel(getattr(logging, verbosity))
    return eval_logger


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    # Backup original from_pretrained
    from_pretrained_orig = AutoModelForCausalLM.from_pretrained

    def from_pretrained_overriden(*from_args, **from_kwargs):
        model = from_pretrained_orig(*from_args, **from_kwargs)
        if args.sparse_weights_path:
            model = load_compressed_weights(
                model,
                args.sparse_weights_path,
                args.sparse_config_path,
                args.sparse_default_level,
            )
        if args.quant_weights_path:
            model = load_compressed_weights(
                model,
                args.quant_weights_path,
                args.quant_config_path,
                args.quant_default_level,
            )
        if args.drop_layer_config:
            drop_layers_from_config(model, args.drop_layer_config)
        return model

    # Override init
    AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

    eval_logger = get_eval_logger(args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    task_manager = make_task_manager(args)
    if task_manager is None and initialize_tasks is not None:
        initialize_tasks(args.verbosity)
    if task_manager is None and args.include_path is not None and include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    task_names = resolve_tasks(args, eval_logger, task_manager)

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(f"File already exists at {path}. Results will be overwritten.")
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = simple_evaluate_compat(
        task_manager,
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(re.sub("/|=", "__", args.model_args), task_name)
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.log_wandb:
            wandb.log(results)


if __name__ == "__main__":
    cli_evaluate()
