#!/usr/bin/env python3
"""Create a tiny fake quantization database for launcher lifecycle tests."""

import sys
from pathlib import Path


def option_value(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def option_values(name: str) -> list[str]:
    index = sys.argv.index(name) + 1
    values: list[str] = []
    while index < len(sys.argv) and not sys.argv[index].startswith("--"):
        values.append(sys.argv[index])
        index += 1
    return values


model_name = option_value("--model_name_or_path").split("/")[-1]
save_root = Path(option_value("--save_dir"))
calibration_bitwidth = option_value("--calibration_bitwidth")
bitwidths = option_values("--bitwidth_options")
quant_db_dir = save_root / model_name / f"{calibration_bitwidth}bit"

for layer_id in range(2):
    module_dir = quant_db_dir / f"model.layers.{layer_id}.self_attn.q_proj"
    module_dir.mkdir(parents=True, exist_ok=True)
    for bitwidth in bitwidths:
        (module_dir / f"{bitwidth}.pth").write_bytes(b"fixture")

print("Processing model.layers.1.self_attn.q_proj")
print("Quantization took 0.01 s.")
