#!/usr/bin/env python3
"""Append one validated experiment row to the EvoPress experiment log."""

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Sequence


COLUMNS = [
    "date",
    "run_id",
    "method",
    "model",
    "sparsity_or_bits",
    "generations",
    "offspring",
    "calibration_data",
    "sequence_length",
    "calibration_tokens",
    "fitness_fn",
    "attention_impl",
    "dtype",
    "seed",
    "wikitext2_ppl",
    "train_ppl",
    "runtime_minutes",
    "gpu_name",
    "gpu_vram_gb",
    "cpu_ram_limit_gb",
    "status",
    "notes",
    "output_dir",
]

REQUIRED_COLUMNS = ["run_id", "method", "model", "status", "output_dir"]
VALID_STATUSES = ["planned", "running", "completed", "failed", "skipped"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one row to results/experiment_log.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-file",
        default="results/experiment_log.csv",
        help="CSV file to append to.",
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Experiment date in ISO format.")
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--method", required=True, help="Experiment method, for example depth_evo or dense.")
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument("--sparsity-or-bits", default="")
    parser.add_argument("--generations", default="")
    parser.add_argument("--offspring", default="")
    parser.add_argument("--calibration-data", default="")
    parser.add_argument("--sequence-length", default="")
    parser.add_argument("--calibration-tokens", default="")
    parser.add_argument("--fitness-fn", default="")
    parser.add_argument("--attention-impl", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--wikitext2-ppl", default="")
    parser.add_argument("--train-ppl", default="")
    parser.add_argument("--runtime-minutes", default="")
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--gpu-vram-gb", default="")
    parser.add_argument("--cpu-ram-limit-gb", default="")
    parser.add_argument("--status", required=True, choices=VALID_STATUSES)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Run artifact directory, conventionally outputs/experiments/<run_id>/.",
    )
    return parser.parse_args(argv)


def row_from_args(args: argparse.Namespace) -> dict[str, str]:
    row = {column: str(getattr(args, column)) for column in COLUMNS}
    missing = [column for column in REQUIRED_COLUMNS if not row[column].strip()]
    if missing:
        raise ValueError(f"Missing required values: {', '.join(missing)}")
    return row


def validate_or_initialize_log(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists() or log_file.stat().st_size == 0:
        with log_file.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return

    with log_file.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != COLUMNS:
        raise ValueError(
            f"Unexpected CSV header in {log_file}. "
            "Refusing to append because the experiment-log schema has drifted."
        )


def append_row(log_file: Path, row: dict[str, str]) -> None:
    validate_or_initialize_log(log_file)
    Path(row["output_dir"]).mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_file = Path(args.log_file)
    append_row(log_file, row_from_args(args))
    print(f"Appended experiment row to {log_file}")


if __name__ == "__main__":
    main()
