#!/usr/bin/env python3
"""Check and print the results of selected exact-budget replay jobs (CPU only).

Checks per job: status completed; for quantized jobs, realized cost equal to
the target; removed sublayers per type equal to ``--expect_removed``. Prints
WikiText-2 / C4 perplexity and calibration KL. Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET_BITS = 26982023168


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--expect_removed", nargs=2, type=int, metavar=("ATTN", "MLP"), default=None)
    args = parser.parse_args(argv)
    ok = True
    print(f"{'job':28s} {'status':10s} {'removed':8s} {'repaired':>8s} {'W2':>9s} {'C4':>9s} {'KL':>8s}  cost")
    for job in args.jobs:
        path = Path(args.results_dir) / "jobs" / job / "result.json"
        if not path.is_file():
            print(f"{job:28s} MISSING")
            ok = False
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        d, m = r.get("details") or {}, r.get("metrics") or {}
        removed = (len(d.get("removed_attn", [])), len(d.get("removed_mlp", [])))
        quantized = r["job"]["precision"] != "fp16"
        cost_ok = (d.get("cost_bits") == TARGET_BITS) if quantized else True
        removed_ok = args.expect_removed is None or list(removed) == args.expect_removed
        job_ok = r.get("status") == "completed" and cost_ok and removed_ok
        ok &= job_ok
        fmt = lambda v, n: f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"  # noqa: E731
        print(f"{job:28s} {r.get('status', '?'):10s} {str(removed):8s} {str(d.get('repair_changed_genes', '-')):>8s} "
              f"{fmt(m.get('wikitext2'), 3):>9s} {fmt(m.get('c4'), 3):>9s} {fmt(m.get('calibration_kl'), 4):>8s}  "
              f"{'= T' if quantized and cost_ok else ('16-bit' if not quantized else 'WRONG')}"
              f"{'' if job_ok else '   <-- CHECK FAILED ' + r.get('error', '')}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
