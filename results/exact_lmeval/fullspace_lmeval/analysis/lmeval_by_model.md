# Zero-shot accuracy (%) of the full-space models

ARC-Easy and PIQA: acc_norm; WinoGrande: acc. Mean +- population SD over seeds.

| Method | n | ARC-Easy | PIQA | WinoGrande | Mean |
| --- | ---: | --- | --- | --- | --- |
| Dense, 16-bit (E0) | 1 | 80.18 | 82.05 | 74.27 | 78.83 |
| Uniform 3-bit (E1) | 1 | 67.76 | 75.35 | 58.09 | 67.07 |
| Quantization-only search (E2) | 3 | 74.44 ± 0.23 | 78.35 ± 0.15 | 68.43 ± 0.56 | 73.74 ± 0.18 |
| Joint search, s = 0.125 (J12) | 3 | 68.64 ± 2.35 | 76.95 ± 0.99 | 65.19 ± 1.98 | 70.26 ± 1.74 |
| Joint search, s = 0.25 (E3) | 3 | 62.72 ± 0.64 | 74.19 ± 0.43 | 62.51 ± 1.80 | 66.48 ± 0.92 |

Per model:

| Model | ARC-Easy | PIQA | WinoGrande | Mean |
| --- | --- | --- | --- | --- |
| dense | 80.18 | 82.05 | 74.27 | 78.83 |
| own_uniform3 | 67.76 | 75.35 | 58.09 | 67.07 |
| own_E2s0 | 74.12 | 78.56 | 68.90 | 73.86 |
| own_E2s1 | 74.58 | 78.24 | 67.64 | 73.49 |
| own_E2s2 | 74.62 | 78.24 | 68.75 | 73.87 |
| own_J12s0 | 65.32 | 75.57 | 62.51 | 67.80 |
| own_J12s1 | 70.41 | 77.86 | 65.82 | 71.36 |
| own_J12s2 | 70.20 | 77.42 | 67.25 | 71.62 |
| own_E3s0 | 62.08 | 74.32 | 61.56 | 65.99 |
| own_E3s1 | 63.59 | 74.65 | 65.04 | 67.76 |
| own_E3s2 | 62.50 | 73.61 | 60.93 | 65.68 |
