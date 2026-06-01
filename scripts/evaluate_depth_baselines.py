#!/usr/bin/env python3
"""Evaluate a cheap depth-pruning baseline matched to an EvoPress config."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_depth_baseline_config import (
    VALID_METHODS,
    build_baseline_config,
    count_removed_modules,
    read_config,
    write_config,
)
from src.common_utils import fix_seed
from src.data_utils import get_data
from src.metrics import compute_perplexity
from src.model_utils import drop_layers


METRIC_COLUMNS = [
    "run_id",
    "sparsity",
    "method",
    "seed",
    "calibration_seed",
    "wikitext2_ppl",
    "train_ppl",
    "dropped_attn_modules",
    "dropped_mlp_modules",
    "reference_config",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a matched depth-pruning baseline.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--reference_config", required=True)
    parser.add_argument("--sparsity", required=True)
    parser.add_argument("--method", required=True, choices=sorted(VALID_METHODS))
    parser.add_argument("--calibration_data", default="wikitext2")
    parser.add_argument("--calibration_tokens", type=int, default=8192)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1, help="Baseline config seed.")
    parser.add_argument("--calibration_seed", type=int, default=1, help="Fixed train-data sampling seed.")
    parser.add_argument("--eval_tokens", type=int, default=524288)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--protect_layer_zero", action="store_true")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def write_metrics(metrics_path: Path, row: dict[str, str]) -> None:
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config_path = Path(args.reference_config)
    reference_config = read_config(reference_config_path)
    baseline_config = build_baseline_config(
        reference_config,
        method=args.method,
        seed=args.seed,
        protect_layer_zero=args.protect_layer_zero,
    )
    config_path = output_dir / "layer_drop_config.txt"
    write_config(config_path, baseline_config)
    dropped_attn_modules, dropped_mlp_modules = count_removed_modules(baseline_config)

    fix_seed(args.calibration_seed)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast_tokenizer)

    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.sequence_length,
        tokenizer,
        train=True,
    )
    wikitext2_data = get_data(
        "wikitext2",
        args.eval_tokens,
        args.sequence_length,
        tokenizer,
        train=False,
    )

    drop_layers(model, baseline_config)
    print(f"Baseline method: {args.method}")
    print(f"Reference config: {reference_config_path}")
    print(f"Dropped attention modules: {dropped_attn_modules}")
    print(f"Dropped MLP modules: {dropped_mlp_modules}")

    wikitext2_ppl = compute_perplexity(model, wikitext2_data)
    print(f"wikitext2: {wikitext2_ppl:.2f}")
    train_ppl = compute_perplexity(model, calibration_data)
    print(f"full train ppl: {train_ppl:.2e}")

    if not math.isfinite(wikitext2_ppl) or not math.isfinite(train_ppl):
        raise ValueError("Baseline evaluation produced non-finite perplexity.")

    write_metrics(
        output_dir / "baseline_metrics.csv",
        {
            "run_id": args.run_id,
            "sparsity": args.sparsity,
            "method": args.method,
            "seed": str(args.seed),
            "calibration_seed": str(args.calibration_seed),
            "wikitext2_ppl": f"{wikitext2_ppl:.2f}",
            "train_ppl": f"{train_ppl:.2e}",
            "dropped_attn_modules": str(dropped_attn_modules),
            "dropped_mlp_modules": str(dropped_mlp_modules),
            "reference_config": str(reference_config_path),
            "output_dir": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
