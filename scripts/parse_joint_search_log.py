#!/usr/bin/env python3
"""Parse generation-wise metrics from evo_joint_search.py stdout."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Sequence


COLUMNS = [
    "run_id",
    "phase",
    "generation",
    "train_fitness",
    "wikitext2_ppl",
    "train_ppl",
    "quant_bit_average",
    "dropped_attn_modules",
    "dropped_mlp_modules",
    "drop_config",
    "quant_state",
]

NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GENERATION_RE = re.compile(r"^Generation\s+(\d+)/(\d+)$")
TRAIN_FITNESS_RE = re.compile(rf"^Train fitness:\s*({NUMBER})$", re.IGNORECASE)
QUANT_AVERAGE_RE = re.compile(rf"^Quant bit average:\s*({NUMBER})$", re.IGNORECASE)
FINAL_QUANT_AVERAGE_RE = re.compile(rf"^Final quant bit average:\s*({NUMBER})$", re.IGNORECASE)
FINAL_DROPPED_ATTN_RE = re.compile(r"^Final dropped attention modules:\s*(\d+)$")
FINAL_DROPPED_MLP_RE = re.compile(r"^Final dropped MLP modules:\s*(\d+)$")
WIKITEXT2_RE = re.compile(rf"^wikitext2:\s*({NUMBER})$", re.IGNORECASE)
TRAIN_PPL_RE = re.compile(rf"^ppl_train:\s*({NUMBER})$", re.IGNORECASE)
DROP_VALUES = {"none", "attn", "mlp", "attn+mlp"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evo_joint_search.py stdout into generation-wise CSV metrics."
    )
    parser.add_argument("--log", required=True, help="Joint-search stdout log to parse.")
    parser.add_argument("--output", required=True, help="CSV file to write.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. Defaults to the output file's parent directory name.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial generation metrics instead of failing when generations are missing.",
    )
    return parser.parse_args(argv)


def new_row(run_id: str, phase: str, generation: int | None = None) -> dict[str, str]:
    return {
        "run_id": run_id,
        "phase": phase,
        "generation": "" if generation is None else str(generation),
        "train_fitness": "",
        "wikitext2_ppl": "",
        "train_ppl": "",
        "quant_bit_average": "",
        "dropped_attn_modules": "",
        "dropped_mlp_modules": "",
        "drop_config": "",
        "quant_state": "",
    }


def parse_drop_config(text: str) -> list[str]:
    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid drop configuration: {text}") from exc
    if not isinstance(values, list) or any(value not in DROP_VALUES for value in values):
        raise ValueError(f"Expected a list of depth-drop values: {text}")
    return values


def parse_quant_group(text: str) -> list[int]:
    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid quantization state: {text}") from exc
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise ValueError(f"Expected a list of integer bitwidths: {text}")
    return values


def encode(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def drop_counts(config: list[str]) -> tuple[int, int]:
    attn = sum(value in {"attn", "attn+mlp"} for value in config)
    mlp = sum(value in {"mlp", "attn+mlp"} for value in config)
    return attn, mlp


def parse_log(log_file: Path, run_id: str, allow_incomplete: bool = False) -> tuple[list[dict[str, str]], int]:
    if not log_file.is_file():
        raise ValueError(f"Log file does not exist: {log_file}")

    generation_rows: dict[int, dict[str, str]] = {}
    generation_quant_groups: dict[int, list[list[int]]] = {}
    expected_generations: int | None = None
    current_generation: int | None = None
    expect_generation_drop = False
    reading_generation_quant = False
    saw_final_configuration = False
    expect_final_drop = False
    reading_final_quant = False
    final_quant_groups: list[list[int]] = []
    final_row = new_row(run_id, "final")

    for raw_line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line:
            continue

        generation_match = GENERATION_RE.match(line)
        if generation_match:
            generation = int(generation_match.group(1))
            total = int(generation_match.group(2))
            if expected_generations is not None and total != expected_generations:
                raise ValueError(f"Inconsistent generation totals: saw {expected_generations} and {total}.")
            if generation in generation_rows:
                raise ValueError(f"Duplicate generation in log: {generation}")
            expected_generations = total
            current_generation = generation
            generation_rows[generation] = new_row(run_id, "generation", generation)
            generation_quant_groups[generation] = []
            expect_generation_drop = False
            reading_generation_quant = False
            continue

        if line == "Drop config:" and current_generation is not None:
            expect_generation_drop = True
            reading_generation_quant = False
            continue

        if line == "Quant state:" and current_generation is not None:
            reading_generation_quant = True
            expect_generation_drop = False
            continue

        if line == "Final drop config:":
            saw_final_configuration = True
            current_generation = None
            expect_final_drop = True
            reading_final_quant = False
            continue

        if line == "Final quant state:":
            reading_final_quant = True
            expect_final_drop = False
            continue

        if line.startswith("["):
            if expect_generation_drop and current_generation is not None:
                config = parse_drop_config(line)
                row = generation_rows[current_generation]
                row["drop_config"] = encode(config)
                attn, mlp = drop_counts(config)
                row["dropped_attn_modules"] = str(attn)
                row["dropped_mlp_modules"] = str(mlp)
                expect_generation_drop = False
                continue
            if reading_generation_quant and current_generation is not None:
                generation_quant_groups[current_generation].append(parse_quant_group(line))
                continue
            if expect_final_drop:
                config = parse_drop_config(line)
                final_row["drop_config"] = encode(config)
                attn, mlp = drop_counts(config)
                final_row["dropped_attn_modules"] = str(attn)
                final_row["dropped_mlp_modules"] = str(mlp)
                expect_final_drop = False
                continue
            if reading_final_quant:
                final_quant_groups.append(parse_quant_group(line))
                continue

        if reading_generation_quant and current_generation is not None:
            reading_generation_quant = False
            groups = generation_quant_groups[current_generation]
            if groups:
                generation_rows[current_generation]["quant_state"] = encode(groups)

        if reading_final_quant:
            reading_final_quant = False
            if final_quant_groups:
                final_row["quant_state"] = encode(final_quant_groups)

        train_fitness_match = TRAIN_FITNESS_RE.match(line)
        if train_fitness_match and current_generation is not None:
            generation_rows[current_generation]["train_fitness"] = train_fitness_match.group(1)
            continue

        quant_average_match = QUANT_AVERAGE_RE.match(line)
        if quant_average_match and current_generation is not None:
            generation_rows[current_generation]["quant_bit_average"] = quant_average_match.group(1)
            continue

        final_quant_average_match = FINAL_QUANT_AVERAGE_RE.match(line)
        if final_quant_average_match:
            final_row["quant_bit_average"] = final_quant_average_match.group(1)
            continue

        final_attn_match = FINAL_DROPPED_ATTN_RE.match(line)
        if final_attn_match:
            final_row["dropped_attn_modules"] = final_attn_match.group(1)
            continue

        final_mlp_match = FINAL_DROPPED_MLP_RE.match(line)
        if final_mlp_match:
            final_row["dropped_mlp_modules"] = final_mlp_match.group(1)
            continue

        wikitext2_match = WIKITEXT2_RE.match(line)
        if wikitext2_match:
            if saw_final_configuration:
                final_row["wikitext2_ppl"] = wikitext2_match.group(1)
            elif current_generation is not None:
                generation_rows[current_generation]["wikitext2_ppl"] = wikitext2_match.group(1)
            continue

        train_ppl_match = TRAIN_PPL_RE.match(line)
        if train_ppl_match:
            if saw_final_configuration:
                final_row["train_ppl"] = train_ppl_match.group(1)
            elif current_generation is not None:
                generation_rows[current_generation]["train_ppl"] = train_ppl_match.group(1)

    if current_generation is not None and generation_quant_groups[current_generation]:
        generation_rows[current_generation]["quant_state"] = encode(generation_quant_groups[current_generation])
    if final_quant_groups:
        final_row["quant_state"] = encode(final_quant_groups)

    if expected_generations is None:
        raise ValueError("No generation markers were found in the log.")

    observed_generations = sorted(generation_rows)
    expected_sequence = list(range(1, expected_generations + 1))
    if observed_generations != expected_sequence:
        message = (
            f"Incomplete generation coverage: parsed {len(observed_generations)}/"
            f"{expected_generations}; observed generations: {observed_generations}."
        )
        if not allow_incomplete:
            raise ValueError(f"{message} Use --allow-incomplete to write partial metrics.")
        print(f"Warning: {message}", file=sys.stderr)

    for generation in observed_generations:
        row = generation_rows[generation]
        missing = [
            field
            for field in (
                "train_fitness",
                "quant_bit_average",
                "dropped_attn_modules",
                "dropped_mlp_modules",
                "drop_config",
                "quant_state",
            )
            if not row[field]
        ]
        if missing:
            raise ValueError(f"Generation {generation} is missing required parsed fields: {', '.join(missing)}.")

    rows = [generation_rows[generation] for generation in observed_generations]
    if saw_final_configuration:
        rows.append(final_row)
    return rows, expected_generations


def write_csv(output_file: Path, rows: list[dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_file = Path(args.output)
    run_id = args.run_id or output_file.parent.name
    rows, expected_generations = parse_log(Path(args.log), run_id, args.allow_incomplete)
    write_csv(output_file, rows)

    generation_count = sum(row["phase"] == "generation" for row in rows)
    print(f"Parsed {generation_count}/{expected_generations} generations for {run_id}.")
    final_rows = [row for row in rows if row["phase"] == "final"]
    if final_rows:
        final_row = final_rows[0]
        print(f"Final WikiText2 PPL: {final_row['wikitext2_ppl'] or 'not found'}")
        print(f"Final train PPL: {final_row['train_ppl'] or 'not found'}")
        print(f"Final quant bit average: {final_row['quant_bit_average'] or 'not found'}")
    else:
        print("Final evaluation: not found")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
