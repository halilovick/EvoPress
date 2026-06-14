# Mistral-7B Medium Search Comparison

This report is generated from the tracked structured artifacts for the thesis-scale medium grid. All search runs use `mistralai/Mistral-7B-v0.3`, WikiText2 calibration, sequence length `1024`, `8192` calibration tokens, `20` generations, `16` offspring, `32` initial candidates, and seeds `0`, `1`, and `2`.

## Main comparison

| Method | Runs | WikiText2 PPL mean +/- SD | PPL range | Compression ratio | Effective bits/parameter | Active parameter ratio | Estimated weight MiB | Runtime min mean +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 1 | 5.960 | 5.960-5.960 | 1.000x | 16.000 | 1.000 | 13824.5 | 1.67 |
| Depth-only | 3 | 11.818 +/- 0.621 | 11.203-12.445 | 1.317x | 12.148 | 0.759 | 10496.5 | 9.58 +/- 0.04 |
| Quant-only q_proj | 3 | 5.938 +/- 0.000 | 5.938-5.938 | 1.064x | 15.037 | 1.000 | 12992.5 | 13.15 +/- 0.03 |
| Joint depth + q_proj quant | 3 | 12.607 +/- 0.675 | 12.156-13.383 | 1.400x | 11.426 | 0.759 | 9872.5 | 10.15 +/- 0.03 |

The dense reference reaches WikiText2 PPL 5.96. Quantizing only `q_proj` preserves dense quality (mean PPL 5.938) but produces only 1.064x whole-model compression because `q_proj` is a small fraction of total model weights.

Depth-only search reaches 1.317x compression at mean PPL 11.818. Joint search reaches 1.400x at mean PPL 12.607. Relative to depth-only, joint search reduces the theoretical weight footprint by 624 MiB and increases the compression ratio by 6.3%, while mean PPL is 0.789 higher.

This is evidence that the joint implementation can optimize a combined candidate at Mistral-7B scale. It is not yet evidence that joint search outperforms independent composition: the matched independent depth-plus-quant control has not been run for this medium grid.

## Per-seed results

| Method | Seed | WikiText2 PPL | Train PPL | Final KL | Compression | Effective bits | Runtime min | Best generation | Accepted generations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 0 | 5.960 |  |  | 1.000x | 16.000 | 1.67 |  |  |
| Depth-only | 0 | 11.805 | 12.156 | 0.6851 | 1.317x | 12.148 | 9.53 | 20 | 13 |
| Depth-only | 1 | 12.445 | 11.719 | 0.6992 | 1.317x | 12.148 | 9.58 | 19 | 12 |
| Depth-only | 2 | 11.203 | 10.039 | 0.6079 | 1.317x | 12.148 | 9.62 | 17 | 14 |
| Quant-only q_proj | 0 | 5.938 | 6.168 | 0.0057 | 1.064x | 15.037 | 13.13 | 18 | 3 |
| Quant-only q_proj | 1 | 5.938 | 6.008 | 0.0050 | 1.064x | 15.037 | 13.13 | 1 | 4 |
| Quant-only q_proj | 2 | 5.938 | 5.332 | 0.0050 | 1.064x | 15.037 | 13.18 | 17 | 4 |
| Joint depth + q_proj quant | 0 | 13.383 | 13.508 | 0.7939 | 1.400x | 11.426 | 10.15 | 20 | 18 |
| Joint depth + q_proj quant | 1 | 12.281 | 11.922 | 0.6992 | 1.400x | 11.426 | 10.18 | 20 | 19 |
| Joint depth + q_proj quant | 2 | 12.156 | 11.117 | 0.7129 | 1.400x | 11.426 | 10.12 | 20 | 19 |

## Seed stability

### Depth-only dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 6 | 26 | 0.231 |
| 0 vs 2 | 7 | 25 | 0.280 |
| 1 vs 2 | 9 | 23 | 0.391 |

Modules selected in all three seeds (6): `model.layers.12.mlp`, `model.layers.13.mlp`, `model.layers.14.self_attn`, `model.layers.15.mlp`, `model.layers.16.self_attn`, `model.layers.28.self_attn`.

### Joint depth + q_proj quant dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 7 | 25 | 0.280 |
| 0 vs 2 | 7 | 25 | 0.280 |
| 1 vs 2 | 7 | 25 | 0.280 |

Modules selected in all three seeds (3): `model.layers.14.self_attn`, `model.layers.25.self_attn`, `model.layers.28.self_attn`.

### Quantization profile stability

| Seed | 2-bit modules | 3-bit modules | 4-bit modules |
| ---: | ---: | ---: | ---: |
| 0 | 2 | 28 | 2 |
| 1 | 2 | 28 | 2 |
| 2 | 2 | 28 | 2 |

Modules assigned 2 bits in every quant-only seed: `model.layers.0.self_attn.q_proj`, `model.layers.1.self_attn.q_proj`. The locations receiving 4 bits varied across seeds.

Final PPL is reasonably repeatable, but the selected depth masks are not identical. This suggests multiple competitive compression configurations rather than one uniquely stable mask.

## Convergence evidence

Depth and joint runs continued to accept improved parents near generation 20, and every joint run recorded its minimum search fitness at generation 20. The current budget therefore does not demonstrate full convergence. Quant-only quality was already close to dense throughout the search, although its calibration KL continued to change.

The generated `mistral_medium_convergence.png` shows the generation-wise search fitness and the periodic WikiText2 evaluations. Final evaluations are added at generation 20.

## Hardware and measurement notes

- The nine searches ran on a Tesla V100 32 GB with a 16 GB container RAM limit.
- Peak sampled device use was about 14.66 GB for every search.
- Peak sampled CPU cgroup memory reached 16 GB, so CPU memory remains the binding resource and leaves little safety margin.
- Compression ratios and model sizes are theoretical weight estimates. The current reconstruction database and runtime model remain floating point; these numbers are not measured checkpoint file sizes or inference-memory measurements.
- Results currently cover WikiText2 only. C4, FineWeb-Edu, and downstream task evaluation remain necessary for broader claims.

## Required next comparison

Evaluate the independently selected depth mask and independently selected `q_proj` quantization profile together for seeds 0, 1, and 2. This produces the missing matched-target control:

```text
joint depth+quant search
vs.
independent depth search + independent quant search, composed afterward
```

Use the same WikiText2 evaluation length and each seed's medium-grid artifacts. Also report the active quantization average after dropped attention modules are excluded, because composing independent profiles can shift the active bit budget away from exactly 3.0 bits.

If the independently composed control is weaker than joint search, that supports the value of coupled optimization. If it is equal or better, the result motivates the planned thesis extension: a more interaction-aware joint mutation operator. After this control, extend the baseline joint search to 50 generations because the 20-generation curves are still improving.

## Generated artifacts

- `results/mistral_medium_runs.csv`: one row per run.
- `results/mistral_medium_aggregate.csv`: method-level mean and sample standard deviation.
- `results/mistral_medium_convergence.csv`: generation-wise search and evaluation metrics.
- `results/mistral_medium_quality_compression.png`: quality-compression tradeoff.
- `results/mistral_medium_convergence.png`: search convergence.
