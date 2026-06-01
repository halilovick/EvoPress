#!/usr/bin/env python3
"""Parse dataset perplexities from eval_ppl.py stdout."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Sequence


COLUMNS = ["run_id", "dataset", "ppl"]
NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PPL_RE = re.compile(rf"^([A-Za-z0-9_.-]+):\s*({NUMBER})$", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert eval_ppl.py stdout into a CSV file.")
    parser.add_argument("--log", required=True, help="Evaluation stdout log to parse.")
    parser.add_argument("--output", required=True, help="CSV file to write.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. Defaults to the output file's parent directory name.",
    )
    parser.add_argument(
        "--required-datasets",
        nargs="+",
        default=["wikitext2"],
        help="Datasets that must be present in the parsed output.",
    )
    return parser.parse_args(argv)


def parse_log(log_file: Path, run_id: str, required_datasets: Sequence[str]) -> list[dict[str, str]]:
    if not log_file.is_file():
        raise ValueError(f"Log file does not exist: {log_file}")

    rows: list[dict[str, str]] = []
    seen_datasets: set[str] = set()
    for raw_line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        match = PPL_RE.match(line)
        if not match:
            continue
        dataset, ppl = match.groups()
        if dataset in seen_datasets:
            raise ValueError(f"Duplicate perplexity for dataset: {dataset}")
        if not math.isfinite(float(ppl)):
            raise ValueError(f"Non-finite perplexity for dataset {dataset}: {ppl}")
        seen_datasets.add(dataset)
        rows.append({"run_id": run_id, "dataset": dataset, "ppl": ppl})

    missing = [dataset for dataset in required_datasets if dataset not in seen_datasets]
    if missing:
        raise ValueError(f"Missing required dataset perplexities: {', '.join(missing)}")
    return rows


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
    rows = parse_log(Path(args.log), run_id, args.required_datasets)
    write_csv(output_file, rows)
    for row in rows:
        print(f"{row['dataset']}: {row['ppl']}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
