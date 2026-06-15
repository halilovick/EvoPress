# Mistral-7B Medium Search Comparison

This report is generated from the tracked artifacts for the thesis-scale Mistral grid. All searches use `mistralai/Mistral-7B-v0.3`, WikiText2 calibration, sequence length `1024`, `8192` calibration tokens, `16` offspring, `32` initial candidates, and seeds `0`, `1`, and `2`. The baseline searches use `20` generations; the compute-matched unchanged and joint-aware searches use `50`. All matched controls use the same final WikiText2 evaluation protocol.

## Main comparison

| Method | Runs | WikiText2 PPL mean +/- SD | Compression | Effective bits | Active q_proj bits | Source search min | Final job min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 1 | 5.960 | 1.000x | 16.000 |  | 0.00 | 1.67 |
| Depth-only | 3 | 11.818 +/- 0.621 | 1.317x | 12.148 |  | 9.58 | 9.58 |
| Quant-only q_proj | 3 | 5.938 +/- 0.000 | 1.064x | 15.037 | 3.000 | 13.15 | 13.15 |
| Depth + uniform 3-bit q_proj | 3 | 11.953 +/- 0.586 | 1.400x | 11.426 | 3.000 | 9.58 | 1.41 |
| Independent depth + quant | 3 | 11.947 +/- 0.585 | 1.400x | 11.425 | 2.972 | 22.73 | 1.45 |
| Joint G20 depth + q_proj quant | 3 | 12.607 +/- 0.675 | 1.400x | 11.426 | 3.000 | 10.15 | 10.15 |
| Joint G50 compute-matched | 3 | 11.242 +/- 0.285 | 1.400x | 11.426 | 3.000 | 22.58 | 22.58 |
| Joint-aware G50 (p=0.5) | 3 | 11.547 +/- 0.617 | 1.400x | 11.426 | 3.000 | 26.43 | 26.43 |

The dense reference reaches WikiText2 PPL 5.96. Quantizing only `q_proj` preserves dense quality (mean PPL 5.938) but produces only 1.064x whole-model compression because `q_proj` is a small fraction of total model weights.

Depth-only search reaches 1.317x compression at mean PPL 11.818. Joint search reaches 1.400x at mean PPL 12.607. Relative to depth-only, joint search reduces the theoretical weight footprint by 624 MiB and increases the compression ratio by 6.3%, while mean PPL is 0.789 higher.

At the matched combined target, independent composition reaches mean PPL 11.947, compared with 12.607 for joint search. The paired mean difference `joint - independent` is 0.660 PPL, so the current 20-generation joint method is worse on average. Independent composition is better in two of three seeds; with only three seeds this should be reported as evidence, not a definitive statistical claim.

Uniform 3-bit `q_proj` composition reaches mean PPL 11.953. Its difference from the searched independent quantization profiles is only 0.007 PPL. At this narrow `q_proj` scope, quantization-profile search therefore adds no visible benefit over uniform 3-bit assignment after depth pruning.

The runtime comparison is not equal: independent composition uses both the depth and quant searches, averaging 22.73 search minutes, or 2.24x the 10.15 minutes used by one joint search. The existing result is target-matched but not compute-matched.

The compute-matched G50 joint search changes the conclusion. It reaches mean PPL 11.242, improving over G20 by 1.365 PPL and outperforming independent composition by 0.704 PPL on average. G50 is better in all three paired seeds. Its mean search runtime is 22.58 minutes versus 22.73 minutes for the independent pipeline, a difference of 0.14 minutes.

The first joint-aware mutation ablation does not improve the unchanged G50 search. At the same target and nominal search budget, joint-aware mutation reaches mean PPL 11.547 +/- 0.617, compared with 11.242 +/- 0.285 for the baseline. The paired mean difference `joint-aware - baseline` is +0.305 PPL; joint-aware wins in 1 of 3 seeds. This is useful negative evidence: coupling every proposed depth exchange to a bit-budget exchange with probability `0.5` did not produce a better search operator.

## Matched per-seed comparison

| Seed | Depth-only | Uniform | Independent | Joint G20 | Joint G50 | Joint-aware G50 | Aware - G50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.805 | 11.990 | 11.970 | 13.383 | 11.469 | 11.156 | -0.312 |
| 1 | 12.445 | 12.520 | 12.520 | 12.281 | 10.922 | 11.227 | +0.305 |
| 2 | 11.203 | 11.350 | 11.350 | 12.156 | 11.336 | 12.258 | +0.922 |

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
| Joint G20 depth + q_proj quant | 0 | 13.383 | 13.508 | 0.7939 | 1.400x | 11.426 | 10.15 | 10.15 | 20 |
| Joint G20 depth + q_proj quant | 1 | 12.281 | 11.922 | 0.6992 | 1.400x | 11.426 | 10.18 | 10.18 | 20 |
| Joint G20 depth + q_proj quant | 2 | 12.156 | 11.117 | 0.7129 | 1.400x | 11.426 | 10.12 | 10.12 | 20 |
| Joint G50 compute-matched | 0 | 11.469 | 11.781 | 0.6641 | 1.400x | 11.426 | 22.47 | 22.47 | 49 |
| Joint G50 compute-matched | 1 | 10.922 | 10.398 | 0.5591 | 1.400x | 11.426 | 22.68 | 22.68 | 50 |
| Joint G50 compute-matched | 2 | 11.336 | 10.102 | 0.6152 | 1.400x | 11.426 | 22.60 | 22.60 | 50 |
| Joint-aware G50 (p=0.5) | 0 | 11.156 | 11.555 | 0.6357 | 1.400x | 11.426 | 23.37 | 23.37 | 47 |
| Joint-aware G50 (p=0.5) | 1 | 11.227 | 10.898 | 0.6025 | 1.400x | 11.426 | 23.47 | 23.47 | 45 |
| Joint-aware G50 (p=0.5) | 2 | 12.258 | 11.070 | 0.7227 | 1.400x | 11.426 | 32.47 | 32.47 | 50 |

## Seed stability

### Depth-only dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 6 | 26 | 0.231 |
| 0 vs 2 | 7 | 25 | 0.280 |
| 1 vs 2 | 9 | 23 | 0.391 |

Modules selected in all three seeds (6): `model.layers.12.mlp`, `model.layers.13.mlp`, `model.layers.14.self_attn`, `model.layers.15.mlp`, `model.layers.16.self_attn`, `model.layers.28.self_attn`.

### Joint G20 depth + q_proj quant dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 7 | 25 | 0.280 |
| 0 vs 2 | 7 | 25 | 0.280 |
| 1 vs 2 | 7 | 25 | 0.280 |

Modules selected in all three seeds (3): `model.layers.14.self_attn`, `model.layers.25.self_attn`, `model.layers.28.self_attn`.

### Joint G50 compute-matched dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 8 | 24 | 0.333 |
| 0 vs 2 | 7 | 25 | 0.280 |
| 1 vs 2 | 10 | 22 | 0.455 |

Modules selected in all three seeds (5): `model.layers.13.mlp`, `model.layers.14.self_attn`, `model.layers.22.self_attn`, `model.layers.28.self_attn`, `model.layers.9.self_attn`.

### Joint-aware G50 (p=0.5) dropped-module stability

| Seed pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| 0 vs 1 | 9 | 23 | 0.391 |
| 0 vs 2 | 6 | 26 | 0.231 |
| 1 vs 2 | 7 | 25 | 0.280 |

Modules selected in all three seeds (5): `model.layers.13.mlp`, `model.layers.14.self_attn`, `model.layers.22.self_attn`, `model.layers.23.self_attn`, `model.layers.28.self_attn`.

### Quantization profile stability

| Seed | 2-bit modules | 3-bit modules | 4-bit modules |
| ---: | ---: | ---: | ---: |
| 0 | 2 | 28 | 2 |
| 1 | 2 | 28 | 2 |
| 2 | 2 | 28 | 2 |

Modules assigned 2 bits in every quant-only seed: `model.layers.0.self_attn.q_proj`, `model.layers.1.self_attn.q_proj`. The locations receiving 4 bits varied across seeds.

Final PPL is reasonably repeatable, but the selected depth masks are not identical. This suggests multiple competitive compression configurations rather than one uniquely stable mask.

## Convergence evidence

Depth and G20 joint runs continued to accept improved parents near generation 20, and every G20 joint run recorded its minimum search fitness at generation 20. Extending the unchanged search to 50 generations materially improved all three seeds. Some late G50 improvements still occurred around generations 47-50, so the curves should be described as substantially improved rather than proven fully converged. The joint-aware runs also continued improving late, but their final distribution was worse and more variable than the unchanged G50 baseline. Quant-only quality was already close to dense throughout its search.

The generated `mistral_medium_convergence.png` shows generation-wise search fitness and periodic WikiText2 evaluations. Final evaluations are added at each run's terminal generation.

## Hardware and measurement notes

- The runs were executed across restarted TU Wien Datalab sessions, where accelerator availability can vary. Runtime comparisons should therefore prioritize matched settings and be interpreted cautiously.
- The unchanged G50 runs averaged about 22.58 minutes. Joint-aware seeds 0 and 1 took about 23.4 minutes, while seed 2 took 32.5 minutes and reported 29.3 GB sampled device use; this makes the aggregate joint-aware runtime unsuitable as a clean operator-overhead estimate.
- Peak sampled CPU cgroup memory reached 16 GB, so CPU memory remains the binding resource and leaves little safety margin.
- Compression ratios and model sizes are theoretical weight estimates. The current reconstruction database and runtime model remain floating point; these numbers are not measured checkpoint file sizes or inference-memory measurements.
- Results currently cover WikiText2 only. C4, FineWeb-Edu, and downstream task evaluation remain necessary for broader claims.

## Interpretation for the thesis

At 20 generations, joint search does not beat independently optimized components at the same compression target. Once search compute is matched, the unchanged 50-generation joint search beats independent composition in every seed. This demonstrates that coupled search is beneficial, but it needs enough generations because its offspring budget is shared across two mutation subspaces.

The joint-aware ablation narrows the method claim. A simple interaction-aware mutation is not automatically better: the `p=0.5` operator worsens mean PPL by 0.305 and doubles the sample standard deviation relative to unchanged G50. The thesis contribution should distinguish the demonstrated benefit of sufficient joint search compute from the unsupported claim that this first coupled mutation design improves search.

## Next thesis contribution

Keep the unchanged G50 method as the current primary result. Before another expensive run, extend the structured logs to record the mutation type that produced the selected parent, rather than only the number of generated offspring of each type. This will show whether joint-aware offspring are actually selected and whether their utility changes over generations.

Then run a bounded probability ablation, starting with `joint_aware_probability=0.25` rather than repeating `0.5`. A lower probability preserves more of the successful baseline exploration while testing whether occasional coupled moves help. Promote it to a three-seed G50 comparison only if a cheaper G20 screening run shows a consistent signal.

## Generated artifacts

- `results/mistral_medium_runs.csv`: one row per run.
- `results/mistral_medium_aggregate.csv`: method-level mean and sample standard deviation.
- `results/mistral_medium_matched_comparison.csv`: paired seed-level comparison of depth, uniform, independent, and joint configurations.
- `results/mistral_joint_aware_ablation.csv`: paired unchanged-G50 versus joint-aware-G50 results, runtimes, and generated mutation counts.
- `results/mistral_medium_convergence.csv`: generation-wise search and evaluation metrics.
- `results/mistral_medium_quality_compression.png`: quality-compression tradeoff.
- `results/mistral_medium_convergence.png`: search convergence.
