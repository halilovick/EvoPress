#!/usr/bin/env python3
"""Emit deterministic joint-search output and save joint configurations."""

import json
import sys
from pathlib import Path


def option_value(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


output_dir = Path(option_value("--output_dir"))
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "joint_drop_config.txt").write_text("none\nattn\nnone\nmlp\n", encoding="utf-8")
(output_dir / "joint_quant_config.txt").write_text(
    "model.layers.0.self_attn.q_proj: 2\nmodel.layers.1.self_attn.q_proj: 4\n",
    encoding="utf-8",
)
(output_dir / "joint_config.json").write_text(
    json.dumps(
        {
            "drop": {"attn": [False, True, False, False], "mlp": [False, False, False, True]},
            "quant": [[2, 4]],
        }
    ),
    encoding="utf-8",
)

print("Generation 1/2")
print("Train fitness: 3.0000e-02")
print("Drop config:")
print("['none', 'attn', 'mlp', 'none']")
print("Quant state:")
print("[3, 3]")
print("Quant bit average: 3.0000e+00")
print("wikitext2: 12.40")
print("ppl_train: 12.80")
print("Generation 2/2")
print("Train fitness: 1.5000e-02")
print("Drop config:")
print("['none', 'attn', 'none', 'mlp']")
print("Quant state:")
print("[2, 4]")
print("Quant bit average: 3.0000e+00")
print("wikitext2: 11.90")
print("ppl_train: 12.20")
print("Final joint configuration saved to:")
print(output_dir)
print("Final drop config:")
print("['none', 'attn', 'none', 'mlp']")
print("Final quant state:")
print("[2, 4]")
print("Final quant bit average: 3.0000e+00")
print("Final dropped attention modules: 1")
print("Final dropped MLP modules: 1")
print("wikitext2: 11.70")
print("ppl_train: 12.00")
