#!/usr/bin/env python3
"""Emit deterministic depth-baseline artifacts for launcher lifecycle tests."""

import argparse
import csv
from pathlib import Path


def count_removed_modules(config: list[str]) -> tuple[int, int]:
    attn_count = sum(value in {"attn", "attn+mlp"} for value in config)
    mlp_count = sum(value in {"mlp", "attn+mlp"} for value in config)
    return attn_count, mlp_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_config", required=True)
    parser.add_argument("--sparsity", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--calibration_seed", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", required=True)
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = [line.strip() for line in Path(args.reference_config).read_text(encoding="utf-8").splitlines()]
    (output_dir / "layer_drop_config.txt").write_text("\n".join(config) + "\n", encoding="utf-8")
    dropped_attn_modules, dropped_mlp_modules = count_removed_modules(config)

    with (output_dir / "baseline_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": args.run_id,
                "sparsity": args.sparsity,
                "method": args.method,
                "seed": args.seed,
                "calibration_seed": args.calibration_seed,
                "wikitext2_ppl": "12.34",
                "train_ppl": "1.11e+01",
                "dropped_attn_modules": dropped_attn_modules,
                "dropped_mlp_modules": dropped_mlp_modules,
                "reference_config": args.reference_config,
                "output_dir": args.output_dir,
            }
        )

    print(f"Baseline method: {args.method}")
    print(f"Dropped attention modules: {dropped_attn_modules}")
    print(f"Dropped MLP modules: {dropped_mlp_modules}")
    print("wikitext2: 12.34")
    print("full train ppl: 1.11e+01")


if __name__ == "__main__":
    main()
