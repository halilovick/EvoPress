# Mistral-7B Medium Search Comparison

This report is generated from the tracked artifacts for the thesis-scale medium grid. All searches use `mistralai/Mistral-7B-v0.3`, WikiText2 calibration, sequence length `1024`, `8192` calibration tokens, `20` generations, `16` offspring, `32` initial candidates, and seeds `0`, `1`, and `2`. The matched composition controls use the same final WikiText2 evaluation protocol.

## Main comparison

| Method | Runs | WikiText2 PPL mean +/- SD | Compression | Effective bits | Active q_proj bits | Source search min | Final job min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 1 | 5.960 | 1.000x | 16.000 |  | 0.00 | 1.67 |
| Depth-only | 3 | 11.818 +/- 0.621 | 1.317x | 12.148 |  | 9.58 | 9.58 |
| Quant-only q_proj | 3 | 5.938 +/- 0.000 | 1.064x | 15.037 | 3.000 | 13.15 | 13.15 |
| Depth + uniform 3-bit q_proj | 3 | 11.953 +/- 0.586 | 1.400x | 11.426 | 3.000 | 9.58 | 1.41 |
| Independent depth + quant | 3 | 11.947 +/- 0.585 | 1.400x | 11.425 | 2.972 | 22.73 | 1.45 |
| Joint depth + q_proj quant | 3 | 12.607 +/- 0.675 | 1.400x | 11.426 | 3.000 | 10.15 | 10.15 |

The dense reference reaches WikiText2 PPL 5.96. Quantizing only `q_proj` preserves dense quality (mean PPL 5.938) but produces only 1.064x whole-model compression because `q_proj` is a small fraction of total model weights.

Depth-only search reaches 1.317x compression at mean PPL 11.818. Joint search reaches 1.400x at mean PPL 12.607. Relative to depth-only, joint search reduces the theoretical weight footprint by 624 MiB and increases the compression ratio by 6.3%, while mean PPL is 0.789 higher.

At the matched combined target, independent composition reaches mean PPL 11.947, compared with 12.607 for joint search. The paired mean difference `joint - independent` is 0.660 PPL, so the current 20-generation joint method is worse on average. Independent composition is better in two of three seeds; with only three seeds this should be reported as evidence, not a definitive statistical claim.

Uniform 3-bit `q_proj` composition reaches mean PPL 11.953. Its difference from the searched independent quantization profiles is only 0.007 PPL. At this narrow `q_proj` scope, quantization-profile search therefore adds no visible benefit over uniform 3-bit assignment after depth pruning.

The runtime comparison is not equal: independent composition uses both the depth and quant searches, averaging 22.73 search minutes, or 2.24x the 10.15 minutes used by one joint search. The existing result is target-matched but not compute-matched.

## Matched per-seed comparison

| Seed | Depth-only PPL | Uniform composition PPL | Independent composition PPL | Joint PPL | Joint - independent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.805 | 11.990 | 11.970 | 13.383 | +1.413 |
| 1 | 12.445 | 12.520 | 12.520 | 12.281 | -0.239 |
| 2 | 11.203 | 11.350 | 11.350 | 12.156 | +0.806 |

## Per-seed results

| Method | Seed | WikiText2 PPL | Train PPL | Final KL | Compression | Effective bits | Source search min | Final job min | Best generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 0 | 5.960 |  |  | 1.000x | 16.000 | 0.00 | 1.67 |  |
| Depth-only | 0 | 11.805 | 12.156 | 0.6851 | 1.317x | 12.148 | 9.53 | 9.53 | 20 |
| Depth-only | 1 | 12.445 | 11.719 | 0.6992 | 1.317x | 12.148 | 9.58 | 9.58 | 19 |
| Depth-only | 2 | 11.203 | 10.039 | 0.6079 | 1.317x | 12.148 | 9.62 | 9.62 | 17 |
| Quant-only q_proj | 0 | 5.938 | 6.168 | 0.0057 | 1.064x | 15.037 | 13.13 | 13.13 | 18 |
| Quant-only q_proj | 1 | 5.938 | 6.008 | 0.0050 | 1.064x | 15.037 | 13.13 | 13.13 | 1 |
| Quant-only q_proj | 2 | 5.938 | 5.332 | 0.0050 | 1.064x | 15.037 | 13.18 | 13.18 | 17 |
| Depth + uniform 3-bit q_proj | 0 | 11.990 |  |  | 1.400x | 11.426 | 9.53 | 1.42 |  |
| Depth + uniform 3-bit q_proj | 1 | 12.520 |  |  | 1.400x | 11.426 | 9.58 | 1.40 |  |
| Depth + uniform 3-bit q_proj | 2 | 11.350 |  |  | 1.400x | 11.426 | 9.62 | 1.40 |  |
| Independent depth + quant | 0 | 11.970 |  |  | 1.400x | 11.426 | 22.67 | 1.55 |  |
| Independent depth + quant | 1 | 12.520 |  |  | 1.401x | 11.424 | 22.72 | 1.40 |  |
| Independent depth + quant | 2 | 11.350 |  |  | 1.401x | 11.424 | 22.80 | 1.40 |  |
| Joint depth + q_proj quant | 0 | 13.383 | 13.508 | 0.7939 | 1.400x | 11.426 | 10.15 | 10.15 | 20 |
| Joint depth + q_proj quant | 1 | 12.281 | 11.922 | 0.6992 | 1.400x | 11.426 | 10.18 | 10.18 | 20 |
| Joint depth + q_proj quant | 2 | 12.156 | 11.117 | 0.7129 | 1.400x | 11.426 | 10.12 | 10.12 | 20 |

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

## Interpretation for the thesis

The current baseline joint search does not beat independently optimized components at the same nominal compression target. This is a useful negative result: merely placing depth and quantization variables in one candidate representation is insufficient to produce a better solution.

Two explanations remain open:

1. The joint search receives less total compute because one offspring population is split between depth and quantization mutations.
2. The current mutation operator does not explicitly model interactions between dropping an attention module and assigning bitwidth to its `q_proj` weights.

## Required next experiment

Run the unchanged joint baseline for `50` generations and `16` offspring on seeds `0`, `1`, and `2`. Based on the measured 20-generation runtime, this should approach the independent pipeline's 22.73-minute search budget. It is the necessary compute-matched baseline because all three current joint runs achieved their best recorded fitness at generation 20.

After the 50-generation baseline, implement joint-aware mutation and compare it against the unchanged 50-generation method with identical seeds and budgets. That ablation is the strongest candidate for the thesis implementation contribution.

## Generated artifacts

- `results/mistral_medium_runs.csv`: one row per run.
- `results/mistral_medium_aggregate.csv`: method-level mean and sample standard deviation.
- `results/mistral_medium_matched_comparison.csv`: paired seed-level comparison of depth, uniform, independent, and joint configurations.
- `results/mistral_medium_convergence.csv`: generation-wise search and evaluation metrics.
- `results/mistral_medium_quality_compression.png`: quality-compression tradeoff.
- `results/mistral_medium_convergence.png`: search convergence.
