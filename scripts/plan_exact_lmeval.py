#!/usr/bin/env python3
"""Plan the zero-shot downstream evaluation of the full-space final models.

This script needs no GPU, model weights or PyTorch. It writes a job list in the
format of ``scripts/plan_exact_replays.py`` for ``evo_exact_lmeval.py``:

* ``dense``: the 16-bit model (E0);
* ``own_uniform3``: uniform 3-bit quantization (E1);
* ``own_E2s{0,1,2}``, ``own_J12s{0,1,2}``, ``own_E3s{0,1,2}``: the completed
  final candidates of quantization-only search and of joint search at
  s = 0.125 and s = 0.25, exactly as stored (no repair).

Every quantized job is checked statically against the group-wise exact budget;
a job that would need repair is rejected. Nothing is evaluated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plan_exact_replays import FULLSPACE_CANDIDATES, job  # noqa: E402
from src.exact_replay import (  # noqa: E402
    build_replay_bits,
    level_deficits,
    load_final_candidate,
    mistral_module_sizes,
    size_groups,
)

DEFAULT_SOURCES = ("E2s0", "E2s1", "E2s2", "J12s0", "J12s1", "J12s2", "E3s0", "E3s1", "E3s2")
EXTRA_SOURCES = ("IAs0", "IAs1", "IAs2", "DWs0", "P4s0", "P4s1", "LXs0")


def build_plan(root: Path, labels: list[str]) -> dict[str, Any]:
    groups = size_groups(mistral_module_sizes())
    sources: dict[str, dict[str, Any]] = {}
    for label in labels:
        rel = FULLSPACE_CANDIDATES[label]
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate {label}: {path}")
        cand = load_final_candidate(path)
        cand["source"] = rel
        sources[label] = cand

    jobs = [
        job("dense", "L", "RQ1 reference", "dense 16-bit model (E0)", {"rule": "none"}, precision="fp16"),
        job("own_uniform3", "L", "RQ1", "uniform 3-bit quantization (E1)", {"rule": "none"},
            {"rule": "uniform", "bitwidth": 3}, {"scope": "none", "seed": 0}),
    ]
    for label in labels:
        jobs.append(job(f"own_{label}", "L", "RQ1", "completed final candidate, as stored",
                        {"source": label}, {"source": label, "fill": "donor"}, {"scope": "none", "seed": 0}))

    for entry in jobs:
        if entry["precision"] != "quantized":
            continue
        mask = entry["mask"]
        drop = sources[mask["source"]]["drop"] if "source" in mask else {"attn": [False] * 32, "mlp": [False] * 32}
        bits, _ = build_replay_bits(entry, drop, sources, groups)
        deficits = level_deficits(groups, bits, drop)
        if any(deficits):
            raise ValueError(f"Job {entry['id']} violates the exact budget (deficits {deficits}).")
        entry["static_deficits"] = deficits
        entry["removed"] = [sum(drop["attn"]), sum(drop["mlp"])]

    return {
        "version": 1,
        "purpose": "zero-shot LM-eval of full-space final models",
        "reference_bitwidth": 3,
        "levels": [2, 3, 4, 5, 6],
        "sources": {label: cand["source"] for label, cand in sources.items()},
        "jobs": jobs,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_variants", action="store_true",
                        help="Also add the RQ2 variants (IA, DW, P4, LX); not part of the default 11 models.")
    args = parser.parse_args(argv)
    labels = list(DEFAULT_SOURCES) + (list(EXTRA_SOURCES) if args.include_variants else [])
    plan = build_plan(Path(args.repo_root), labels)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lmeval_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    lines = ["# Zero-shot LM-eval plan (full-space final models)", "", f"Jobs: {len(plan['jobs'])}", "",
             "| Job | Precision | Removed (attn, MLP) | Source |", "| --- | --- | --- | --- |"]
    for entry in plan["jobs"]:
        source = plan["sources"].get(entry["mask"].get("source", ""), "-")
        lines.append(f"| `{entry['id']}` | {entry['precision']} | {entry.get('removed', [0, 0])} | {source} |")
    (out / "lmeval_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"jobs={len(plan['jobs'])} written to {out / 'lmeval_plan.json'}")


if __name__ == "__main__":
    main()
