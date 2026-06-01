#!/usr/bin/env python3
"""Append one validated row to results/depth_baseline_runs.csv."""

import argparse
import csv
from pathlib import Path
from typing import Sequence


COLUMNS = [
    "sparsity",
    "method",
    "seed",
    "wikitext2_ppl",
    "train_ppl",
    "runtime_minutes",
    "notes",
    "output_dir",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one depth-baseline result row.")
    parser.add_argument("--log-file", default="results/depth_baseline_runs.csv")
    for column in COLUMNS:
        parser.add_argument(f"--{column.replace('_', '-')}", required=True)
    return parser.parse_args(argv)


def validate_or_initialize_log(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists() or log_file.stat().st_size == 0:
        with log_file.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return
    with log_file.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != COLUMNS:
        raise ValueError(f"Unexpected CSV header in {log_file}. Refusing to append because the schema has drifted.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    row = {column: str(getattr(args, column)) for column in COLUMNS}
    log_file = Path(args.log_file)
    validate_or_initialize_log(log_file)
    with log_file.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
    print(f"Appended depth-baseline row to {log_file}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
