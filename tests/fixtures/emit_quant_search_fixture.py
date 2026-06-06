#!/usr/bin/env python3
"""Emit deterministic quant-search output and save a final configuration."""

import sys
from pathlib import Path


def option_value(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


quant_weights_path = Path(option_value("--quant_weights_path"))
configuration_name = option_value("--configuration_name")
(quant_weights_path / configuration_name).write_text(
    "\n".join(
        [
            "model.layers.0.self_attn.q_proj: 2",
            "model.layers.1.self_attn.q_proj: 4",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print("Generation 1/2")
print("Current search point:")
print("[3, 3]")
print("Parent bits: 600")
print("Bit average: 3.0000e+00")
print("Train fitness: inf")
print("wikitext2: 9.50")
print("ppl_train: 9.80")
print("Train fitnesses: 2.50e-02")
print("Generation 2/2")
print("Current search point:")
print("[2, 4]")
print("Parent bits: 600")
print("Bit average: 3.0000e+00")
print("Train fitness: 2.5000e-02")
print("wikitext2: 9.30")
print("ppl_train: 9.60")
print("Train fitnesses: 1.25e-02")
print("Final configuration:")
print("[2, 4]")
print("wikitext2: 9.10")
print("ppl_train: 9.40")
