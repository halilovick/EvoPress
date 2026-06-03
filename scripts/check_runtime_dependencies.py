#!/usr/bin/env python3
"""Fail fast when experiment runtime dependencies are missing."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from typing import Sequence


DEFAULT_PACKAGES = [
    "datasets",
    "numpy",
    "torch",
    "transformers",
    "tqdm",
    "accelerate",
    "sentencepiece",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that required Python packages can be imported before starting an experiment."
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=DEFAULT_PACKAGES,
        help="Import names to check.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Also require torch.cuda.is_available().",
    )
    return parser.parse_args(argv)


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    missing: list[str] = []
    imported: list[str] = []
    for package_name in args.packages:
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError:
            missing.append(package_name)
        else:
            imported.append(package_name)

    if missing:
        print("Missing required Python package(s): " + ", ".join(missing), file=sys.stderr)
        print("Install dependencies before launching experiments:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            print("CUDA is not available to torch in this environment.", file=sys.stderr)
            return 1

    print("Runtime dependency check passed:")
    for package_name in imported:
        print(f"  {package_name} {package_version(package_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
