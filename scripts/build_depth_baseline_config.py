#!/usr/bin/env python3
"""Build a depth-pruning baseline config matched to an EvoPress config."""

import argparse
import random
from pathlib import Path
from typing import Sequence


VALID_CONFIG_VALUES = {"none", "attn", "mlp", "attn+mlp"}
VALID_METHODS = {"random", "late_layer", "early_layer"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a matched depth-pruning baseline config.")
    parser.add_argument("--reference-config", required=True, help="EvoPress layer_drop_config.txt to match.")
    parser.add_argument("--method", required=True, choices=sorted(VALID_METHODS))
    parser.add_argument("--seed", type=int, default=1, help="Random baseline seed.")
    parser.add_argument("--protect-layer-zero", action="store_true", help="Exclude layer 0 from candidate removals.")
    parser.add_argument("--output", required=True, help="Baseline layer_drop_config.txt to write.")
    return parser.parse_args(argv)


def read_config(config_path: Path) -> list[str]:
    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")
    config = [line.strip() for line in config_path.read_text(encoding="utf-8").splitlines()]
    if not config:
        raise ValueError(f"Config file is empty: {config_path}")
    invalid = [value for value in config if value not in VALID_CONFIG_VALUES]
    if invalid:
        raise ValueError(f"Invalid config values in {config_path}: {invalid}")
    return config


def count_removed_modules(config: Sequence[str]) -> tuple[int, int]:
    attn_count = sum(value in {"attn", "attn+mlp"} for value in config)
    mlp_count = sum(value in {"mlp", "attn+mlp"} for value in config)
    return attn_count, mlp_count


def choose_indices(
    method: str,
    num_layers: int,
    count: int,
    rng: random.Random,
    protect_layer_zero: bool,
) -> list[int]:
    candidates = list(range(1 if protect_layer_zero else 0, num_layers))
    if count > len(candidates):
        raise ValueError(
            f"Cannot remove {count} modules from {len(candidates)} eligible layers "
            f"with protect_layer_zero={protect_layer_zero}."
        )
    if method == "random":
        return sorted(rng.sample(candidates, count))
    if method == "late_layer":
        return candidates[-count:] if count else []
    if method == "early_layer":
        return candidates[:count]
    raise ValueError(f"Unsupported baseline method: {method}")


def build_baseline_config(
    reference_config: Sequence[str],
    method: str,
    seed: int,
    protect_layer_zero: bool = False,
) -> list[str]:
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported baseline method: {method}")

    attn_count, mlp_count = count_removed_modules(reference_config)
    rng = random.Random(seed)
    attn_indices = choose_indices(method, len(reference_config), attn_count, rng, protect_layer_zero)
    mlp_indices = choose_indices(method, len(reference_config), mlp_count, rng, protect_layer_zero)

    baseline_config = ["none"] * len(reference_config)
    for layer_id in attn_indices:
        baseline_config[layer_id] = "attn"
    for layer_id in mlp_indices:
        baseline_config[layer_id] = "attn+mlp" if baseline_config[layer_id] == "attn" else "mlp"
    return baseline_config


def write_config(config_path: Path, config: Sequence[str]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(config) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference_config = read_config(Path(args.reference_config))
    baseline_config = build_baseline_config(
        reference_config,
        method=args.method,
        seed=args.seed,
        protect_layer_zero=args.protect_layer_zero,
    )
    write_config(Path(args.output), baseline_config)
    attn_count, mlp_count = count_removed_modules(baseline_config)
    print(f"method={args.method}")
    print(f"seed={args.seed}")
    print(f"layers={len(baseline_config)}")
    print(f"dropped_attn_modules={attn_count}")
    print(f"dropped_mlp_modules={mlp_count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
