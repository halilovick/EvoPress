# Seed Robustness: Mistral-7B Depth Pruning at 37.5%

This table summarizes the three-seed repeatability experiment for EvoPress depth pruning with `10` generations and `8` offspring.

| Seed | Run ID | WikiText2 PPL | Train PPL | Runtime (min) | GPU | Dropped attn | Dropped MLP |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `depth_mistral7b_s0.375_seed1` | 51.69 | 48.20 | 46.08 | Tesla T4 | 12 | 12 |
| 2 | `depth_mistral7b_s0.375_seed2` | 40.19 | 28.50 | 9.38 | NVIDIA A40 | 12 | 12 |
| 3 | `depth_mistral7b_s0.375_seed3` | 47.91 | 41.90 | 9.35 | NVIDIA A40 | 12 | 12 |

## Summary Statistics

| Metric | Value |
| --- | ---: |
| Mean WikiText2 PPL | 46.60 |
| Sample std WikiText2 PPL | 5.86 |
| Mean train PPL | 39.53 |
| Mean runtime minutes | 21.60 |
| Best seed | 2 (`depth_mistral7b_s0.375_seed2`, PPL 40.19) |
| Worst seed | 1 (`depth_mistral7b_s0.375_seed1`, PPL 51.69) |

## Pairwise Dropped-Module Overlap

Jaccard overlap is computed over dropped `(layer_index, module_type)` pairs using zero-based layer indices.

| Seed pair | Intersection | Union | Jaccard overlap |
| --- | ---: | ---: | ---: |
| 1 vs 2 | 13 | 35 | 0.371 |
| 1 vs 3 | 13 | 35 | 0.371 |
| 2 vs 3 | 14 | 34 | 0.412 |

## Interpretation

All available seeds completed with finite WikiText2 PPL. The final WikiText2 PPL values are tightly grouped relative to the baseline spread, with mean `46.60`.
Runtime should not be compared directly across seeds because the runs used different GPU types.
