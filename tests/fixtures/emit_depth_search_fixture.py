#!/usr/bin/env python3
"""Emit a deterministic depth-search log for launcher lifecycle tests."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_config_dir", required=True)
    args, _ = parser.parse_known_args()

    output_dir = Path(args.drop_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ["none", "mlp", "attn", "attn+mlp"]
    (output_dir / "layer_drop_config.txt").write_text("\n".join(config) + "\n", encoding="utf-8")

    fixture = Path(__file__).with_name("depth_search_sample.txt")
    print(fixture.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
