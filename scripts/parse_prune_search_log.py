#!/usr/bin/env python3
"""Parse generation-wise metrics from evo_prune_search.py stdout."""

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
    "selection_train_fitness",
    "wikitext2_ppl",
    "train_ppl",
    "search_point",
]

NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GENERATION_RE = re.compile(r"^Generation\s+(\d+)/(\d+)$")
CURRENT_SEARCH_POINT_RE = re.compile(r"^Current search point:\s*(\[[^\]]*\])$")
TRAIN_FITNESS_RE = re.compile(rf"^Train fitness:\s*({NUMBER})$", re.IGNORECASE)
SELECTION_TRAIN_FITNESS_RE = re.compile(rf"^Train fitnesses:\s*({NUMBER})$", re.IGNORECASE)
WIKITEXT2_RE = re.compile(rf"^wikitext2:\s*({NUMBER})$", re.IGNORECASE)
TRAIN_PPL_RE = re.compile(rf"^ppl_train:\s*({NUMBER})$", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evo_prune_search.py stdout into generation-wise CSV metrics."
    )
    parser.add_argument("--log", required=True, help="Sparse-search stdout log to parse.")
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
        "selection_train_fitness": "",
        "wikitext2_ppl": "",
        "train_ppl": "",
        "search_point": "",
    }


def normalize_int_list(values_text: str) -> str:
    try:
        values = ast.literal_eval(values_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid integer list: {values_text}") from exc
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise ValueError(f"Expected a list of integers: {values_text}")
    return json.dumps(values, separators=(",", ":"))


def parse_log(log_file: Path, run_id: str, allow_incomplete: bool = False) -> tuple[list[dict[str, str]], int | None]:
    if not log_file.is_file():
        raise ValueError(f"Log file does not exist: {log_file}")

    generation_rows: dict[int, dict[str, str]] = {}
    expected_generations: int | None = None
    current_generation: int | None = None
    saw_final_configuration = False
    expect_final_search_point = False
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
                raise ValueError(
                    f"Inconsistent generation totals: saw {expected_generations} and {total}."
                )
            if generation in generation_rows:
                raise ValueError(f"Duplicate generation in log: {generation}")
            expected_generations = total
            current_generation = generation
            saw_final_configuration = False
            expect_final_search_point = False
            generation_rows[generation] = new_row(run_id, "generation", generation)
            continue

        if line == "Final configuration:":
            saw_final_configuration = True
            expect_final_search_point = True
            current_generation = None
            continue

        if expect_final_search_point and line.startswith("["):
            final_row["search_point"] = normalize_int_list(line)
            expect_final_search_point = False
            continue

        search_point_match = CURRENT_SEARCH_POINT_RE.match(line)
        if search_point_match and current_generation is not None:
            generation_rows[current_generation]["search_point"] = normalize_int_list(search_point_match.group(1))
            continue

        train_fitness_match = TRAIN_FITNESS_RE.match(line)
        if train_fitness_match and current_generation is not None:
            generation_rows[current_generation]["train_fitness"] = train_fitness_match.group(1)
            continue

        selection_train_fitness_match = SELECTION_TRAIN_FITNESS_RE.match(line)
        if selection_train_fitness_match and current_generation is not None:
            generation_rows[current_generation]["selection_train_fitness"] = selection_train_fitness_match.group(1)
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

    if expected_generations is None:
        if allow_incomplete:
            return [], None
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
        missing = [field for field in ("train_fitness", "search_point") if not row[field]]
        if missing:
            raise ValueError(
                f"Generation {generation} is missing required parsed fields: {', '.join(missing)}."
            )

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
    if expected_generations is None:
        print(f"Parsed {generation_count} generations for {run_id}.")
    else:
        print(f"Parsed {generation_count}/{expected_generations} generations for {run_id}.")
    final_rows = [row for row in rows if row["phase"] == "final"]
    if final_rows:
        final_row = final_rows[0]
        print(f"Final WikiText2 PPL: {final_row['wikitext2_ppl'] or 'not found'}")
        print(f"Final train PPL: {final_row['train_ppl'] or 'not found'}")
    else:
        print("Final evaluation: not found")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
