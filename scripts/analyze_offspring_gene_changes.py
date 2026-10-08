#!/usr/bin/env python3
"""Average genes changed per offspring type, from generation logs (CPU only).

Reads the per-generation JSON diagnostics of the imported full-space runs and
reports, per run and offspring type, the number of accepted offspring and the
mean numbers of changed bit-width genes and mask entries relative to the parent.
Under the exact budget, the standard joint quantization child draws its level
exchange from all projections of a size group, including inactive ones, and is
then repaired; with a fraction f of inactive projections the expected number of
changed genes is about 2(1-f)^2 + 3*2f(1-f) + 2f^2, which this table checks.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

csv.field_size_limit(10**9)


def run_rows(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # The depth-warm G150 log records every generation twice (rows differ only
    # in cumulative runtime); keep the last row of every generation.
    by_generation = {}
    for row in rows:
        by_generation[row.get("generation")] = row
    rows = list(by_generation.values())
    if not rows:
        return {}
    key = next((k for k in rows[0] if (rows[0][k] or "").startswith('{"type"')), None)
    if key is None:
        return {}
    totals: dict[str, list[float]] = {}
    for row in rows:
        diag = json.loads(row[key])
        generated = diag.get("generated_offspring_by_type", {})
        quant = diag.get("quant_assignments_changed_by_type", {})
        depth = diag.get("depth_mask_entries_changed_by_type", {})
        for kind, count in generated.items():
            if not count:
                continue
            entry = totals.setdefault(kind, [0, 0, 0])
            entry[0] += count
            entry[1] += quant.get(kind, 0)
            entry[2] += depth.get(kind, 0)
    return totals


def expected_quant_genes(inactive_fraction: float) -> float:
    f = inactive_fraction
    return 2 * (1 - f) ** 2 + 3 * 2 * f * (1 - f) + 2 * f**2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_dir", default="thesis_results/datalab_raw")
    parser.add_argument("--output", default="thesis_results/offspring_gene_changes.md")
    args = parser.parse_args(argv)
    raw = Path(args.raw_dir)
    paths = sorted(raw.glob("paper_matched/joint/*/generation_log.csv")) + sorted(
        raw.glob("fullspace_extensions/*/generation_log.csv"))
    lines = [
        "# Genes changed per accepted offspring (full-space runs)", "",
        "Source: generation logs in `thesis_results/datalab_raw/`. Means over all generations of the",
        "logged attempt. Resumed runs contribute only the generations of the imported attempt.", "",
        f"Expected quantization-child genes if exchanges may pick inactive genes: "
        f"{expected_quant_genes(0.25):.3f} at 25% removal, {expected_quant_genes(0.125):.3f} at 12.5% "
        "(2.000 if only active genes were used).", "",
        "| Run directory | Offspring type | Offspring | Bit genes / child | Mask entries / child |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for path in paths:
        for kind, (count, genes, masks) in sorted(run_rows(path).items()):
            lines.append(f"| {path.parent.name} | {kind} | {count} | {genes / count:.3f} | {masks / count:.3f} |")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
