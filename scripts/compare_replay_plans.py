#!/usr/bin/env python3
"""Compare two replay plans: list added, removed and changed jobs and sources.

Used to check that an extended plan (e.g. with the 12.5% depth-only masks)
leaves every job of the completed batch unchanged. Exit code 1 if a job or
source of the old plan was removed or changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args(argv)
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    old_jobs = {job["id"]: job for job in old["jobs"]}
    new_jobs = {job["id"]: job for job in new["jobs"]}
    added = [j for j in new_jobs if j not in old_jobs]
    removed = [j for j in old_jobs if j not in new_jobs]
    changed = [j for j in old_jobs if j in new_jobs and old_jobs[j] != new_jobs[j]]
    source_changes = [k for k, v in old["sources"].items() if new["sources"].get(k) != v]
    new_sources = [k for k in new["sources"] if k not in old["sources"]]
    print(f"old jobs: {len(old_jobs)}; new jobs: {len(new_jobs)}")
    print(f"added ({len(added)}): {' '.join(added)}")
    print(f"removed ({len(removed)}): {' '.join(removed)}")
    print(f"changed ({len(changed)}): {' '.join(changed)}")
    print(f"new sources: {' '.join(new_sources)}; changed sources: {' '.join(source_changes)}")
    ok = not removed and not changed and not source_changes
    print("OK: all jobs of the old plan are unchanged" if ok else "ERROR: the old plan was modified")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
