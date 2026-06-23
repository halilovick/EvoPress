#!/usr/bin/env python3
"""Emit a tiny LM-eval style JSON result for launcher tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--model_args")
    parser.add_argument("--tasks")
    parser.add_argument("--batch_size")
    parser.add_argument("--device")
    parser.add_argument("--num_fewshot")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_batch_size", default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--quant_weights_path", default=None)
    parser.add_argument("--quant_config_path", default=None)
    parser.add_argument("--quant_default_level", default=None)
    parser.add_argument("--drop_layer_config", default=None)
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    result = {
        "results": {
            task: {
                "acc,none": 0.5,
                "acc_stderr,none": 0.01,
            }
            for task in tasks
        },
        "config": {
            "model": args.model,
            "model_args": args.model_args,
            "batch_sizes": [args.batch_size],
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"fixture wrote {output_path}")


if __name__ == "__main__":
    main()
