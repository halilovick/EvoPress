# Sequential Initialization Search Comparison

This report compares five 20-generation sequential conditions, including both mutation operators for quantization-first warm starts, plus the completed 50-generation depth-warm interaction-aware reference. Lower WikiText2 perplexity (PPL) is better.

## Scope and Validation

All runs use `mistralai/Mistral-7B-v0.3`, 25% separate attention/MLP depth sparsity, a 3-bit active q-projection budget, `group_rule=size`, 16 offspring, and seeds 0–2. All sequential conditions use 20 stage-two generations except Depth → Joint warm + interaction-aware, which is explicitly labeled as a G50 reference.

The generator verified 15 G20 sequential summaries and the three depth-warm interaction-aware G50 summaries, including exact depth counts, active quantization budgets, stage-one provenance hashes, and frozen-component invariants. Both quantization-first warm-start conditions used strict initialization and changed zero imported genes before initial selection.

Depth-first modes evaluate one exact imported-depth combined candidate during stage-two initialization. Quantization-first modes generate and evaluate 32 exact feasible depth masks. The ordinary joint G20 baseline also evaluates 32 initial candidates.

## Executive Result

**Depth → Joint warm + standard (G20) is the best G20 sequential condition at 11.526 ± 0.223 PPL.**

Its paired mean delta is -0.421 PPL versus independent composition (3/3 seed wins), -1.081 versus standard joint G20 (3/3 wins), and +0.284 versus standard joint G50 (1/3 wins).

This is a 3.52% mean PPL improvement over independent composition and a 8.57% improvement over standard joint G20. Standard-initialization interaction-aware G50 reaches 11.086 mean PPL; its matched depth-warm counterpart reaches 11.172. Both are explicitly separated from the G20 conditions.

## Aggregate Comparison

| Method | Role | Seeds | WikiText2 PPL mean ± SD | Stage 1 min | Stage 2/final job min | End-to-end min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Depth + uniform 3-bit q_proj | target-matched composition baseline | 3 | 11.953 ± 0.586 | 9.58 | 1.41 | 10.98 |
| Independent depth + searched q_proj quant | target-matched sequential composition baseline | 3 | 11.947 ± 0.585 | 22.73 | 1.45 | 24.18 |
| Standard joint G20 | same stage-two generation/offspring schedule | 3 | 12.607 ± 0.675 | 0.00 | 10.15 | 10.15 |
| Depth → Quantization, frozen | new sequential variant | 3 | 11.932 ± 0.586 | 9.58 | 9.22 | 18.80 |
| Depth → Joint warm + standard (G20) | depth warm start with standard mutation | 3 | 11.526 ± 0.223 | 9.58 | 8.98 | 18.56 |
| Depth → Joint warm + interaction-aware (G50) | depth warm start with interaction-aware mutation at G50 | 3 | 11.172 ± 0.196 | 9.58 | 20.69 | 30.27 |
| Quantization → Depth, frozen | new sequential variant | 3 | 14.003 ± 2.748 | 13.15 | 8.64 | 21.79 |
| Quantization → Joint warm + standard (G20) | quantization warm start with standard mutation | 3 | 12.573 ± 0.439 | 13.15 | 9.01 | 22.16 |
| Quantization → Joint warm + interaction-aware (G20) | quantization warm start with interaction-aware mutation | 3 | 11.846 ± 0.532 | 13.15 | 11.39 | 24.54 |
| Standard joint G50 | larger single-search compute reference | 3 | 11.242 ± 0.285 | 0.00 | 22.58 | 22.58 |
| Interaction-aware joint G50 | non-matched operator reference | 3 | 11.086 ± 0.174 | 0.00 | 21.19 | 21.19 |

## Per-Seed Quality

| Seed | Depth→Quant frozen | Depth→Joint standard G20 | Depth→Joint interaction G50 | Quant→Depth frozen | Quant→Joint standard G20 | Quant→Joint interaction G20 | Independent | Joint G20 | Joint G50 | Interaction-aware G50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.922 | 11.508 | 10.984 | 11.031 | 13.047 | 12.398 | 11.970 | 13.383 | 11.469 | 10.898 |
| 1 | 12.523 | 11.758 | 11.375 | 16.453 | 12.180 | 11.805 | 12.520 | 12.281 | 10.922 | 11.117 |
| 2 | 11.352 | 11.312 | 11.156 | 14.523 | 12.492 | 11.336 | 11.350 | 12.156 | 11.336 | 11.242 |

## Paired Mean Deltas

Delta is `sequential PPL − baseline PPL`; negative values favor the sequential method.

| Sequential method | Baseline | Mean delta ± SD | Seed wins |
| --- | --- | ---: | ---: |
| Depth → Quantization, frozen | Depth + uniform 3-bit q_proj | -0.021 ± 0.041 | 1/3 |
| Depth → Quantization, frozen | Independent depth + searched q_proj quant | -0.014 ± 0.029 | 1/3 |
| Depth → Quantization, frozen | Standard joint G20 | -0.674 ± 0.859 | 2/3 |
| Depth → Quantization, frozen | Standard joint G50 | +0.690 ± 0.819 | 0/3 |
| Depth → Quantization, frozen | Interaction-aware joint G50 | +0.846 ± 0.666 | 0/3 |
| Depth → Joint warm + standard (G20) | Depth + uniform 3-bit q_proj | -0.427 ± 0.365 | 3/3 |
| Depth → Joint warm + standard (G20) | Independent depth + searched q_proj quant | -0.421 ± 0.364 | 3/3 |
| Depth → Joint warm + standard (G20) | Standard joint G20 | -1.081 ± 0.706 | 3/3 |
| Depth → Joint warm + standard (G20) | Standard joint G50 | +0.284 ± 0.479 | 1/3 |
| Depth → Joint warm + standard (G20) | Interaction-aware joint G50 | +0.440 ± 0.321 | 0/3 |
| Quantization → Depth, frozen | Depth + uniform 3-bit q_proj | +2.049 ± 2.633 | 1/3 |
| Quantization → Depth, frozen | Independent depth + searched q_proj quant | +2.056 ± 2.621 | 1/3 |
| Quantization → Depth, frozen | Standard joint G20 | +1.396 ± 3.368 | 1/3 |
| Quantization → Depth, frozen | Standard joint G50 | +2.760 ± 3.007 | 1/3 |
| Quantization → Depth, frozen | Interaction-aware joint G50 | +2.917 ± 2.621 | 0/3 |
| Quantization → Joint warm + standard (G20) | Depth + uniform 3-bit q_proj | +0.620 ± 0.832 | 1/3 |
| Quantization → Joint warm + standard (G20) | Independent depth + searched q_proj quant | +0.626 ± 0.838 | 1/3 |
| Quantization → Joint warm + standard (G20) | Standard joint G20 | -0.034 ± 0.341 | 2/3 |
| Quantization → Joint warm + standard (G20) | Standard joint G50 | +1.331 ± 0.220 | 0/3 |
| Quantization → Joint warm + standard (G20) | Interaction-aware joint G50 | +1.487 ± 0.580 | 0/3 |
| Quantization → Joint warm + interaction-aware (G20) | Depth + uniform 3-bit q_proj | -0.107 ± 0.568 | 2/3 |
| Quantization → Joint warm + interaction-aware (G20) | Independent depth + searched q_proj quant | -0.100 ± 0.577 | 2/3 |
| Quantization → Joint warm + interaction-aware (G20) | Standard joint G20 | -0.760 ± 0.259 | 3/3 |
| Quantization → Joint warm + interaction-aware (G20) | Standard joint G50 | +0.604 ± 0.524 | 0/3 |
| Quantization → Joint warm + interaction-aware (G20) | Interaction-aware joint G50 | +0.760 ± 0.706 | 0/3 |

## Targeted Warm-Start and Mutation Comparisons

| Comparison | Mean paired delta ± SD | Seed wins | Interpretation |
| --- | ---: | ---: | --- |
| Quant warm interaction-aware G20 − quant warm standard G20 | -0.727 ± 0.396 | 3/3 | Same initialization direction and stage-two budget; mutation operator changes. |
| Depth-warm interaction-aware G50 − standard-initialization interaction-aware G50 | +0.086 ± 0.172 | 1/3 | Same interaction-aware operator and G50 schedule; initialization changes. |

## Search-Dynamics Diagnostics

| Sequential method | Accepted replacements/run | No-op offspring | Duplicate offspring | Infeasible offspring |
| --- | ---: | ---: | ---: | ---: |
| Depth → Quantization, frozen | 11.3 | 0 | 10 | 0 |
| Depth → Joint warm + standard (G20) | 9.3 | 0 | 7 | 0 |
| Quantization → Depth, frozen | 13.0 | 2 | 7 | 0 |
| Quantization → Joint warm + standard (G20) | 16.3 | 1 | 6 | 0 |
| Quantization → Joint warm + interaction-aware (G20) | 14.3 | 0 | 0 | 0 |

## Interpretation

- Depth → Joint warm + standard G20 improves on independent composition by 0.421 PPL on average and wins 3/3 paired seeds. It also has the lowest sample SD among the sequential variants.
- Depth → Quantization frozen is effectively tied with the depth-plus-uniform and independent controls, indicating little additional gain from optimizing only q-projection assignments after fixing a strong depth solution.
- Quantization → Depth frozen is the most variable variant. Its initial parents expose 326–352 legal contribution-compatible swaps, but the exact-budget frozen neighborhood still constrains which structural exchanges are reachable.
- Quantization → Joint warm + interaction-aware G20 reaches 11.846 mean PPL versus 12.573 for the same warm start with standard mutation. The paired delta is -0.727 PPL with 3/3 seed wins.
- Standard joint G50 is 0.284 PPL better than the best sequential variant on average. It is a larger single search, not an exact nominal-budget match to a two-stage sequential pipeline.
- Under the same interaction-aware G50 operator and schedule, depth warm-starting changes mean PPL by +0.086 relative to standard initialization and wins 1/3 seeds. This comparison is kept separate from the G20 matrix.

## Compute Accounting

Sequential stage-two runtime alone is not the full cost. The end-to-end column adds the seed-matched stage-one search that produced the imported candidate. This treats warm starts and frozen baselines as complete pipelines even though the tracked experiments reused already-computed stage-one artifacts.

Standard joint G20 matches the stage-two generation and offspring schedule, but it omits stage-one cost and its initialization count differs from the one-candidate depth-first variants. Standard joint G50 is a larger single-search compute reference. Neither is an exact match for the number and type of candidate evaluations in a two-stage sequential pipeline.

Depth → Joint warm-starting averages 18.56 end-to-end minutes, compared with 24.18 for independent composition and 22.58 for standard joint G50. These observed differences are 23.2% and 17.8% respectively, but hardware/session variation prevents treating them as precise operator-cost measurements.

## Methodological Limits

- There are only three seeds; the report presents descriptive mean, sample SD, and paired deltas without significance claims.
- Quality is currently measured on WikiText2 only. Broader calibration datasets and downstream LM-eval tasks are not included.
- The quantization scope covers only `q_proj`, so it represents a small part of total model parameters.
- Runtime comparisons span restarted TU Wien DataLab sessions and should be interpreted as approximate wall-clock evidence.
- Compression and memory values are theoretical weight accounting, not measured compressed checkpoint or inference memory.

## Presentation-Ready Conclusion

- Sequential initialization is direction-dependent.
- A strong depth solution is a useful warm start for joint refinement.
- Freezing quantization restricts depth exploration and produces high seed variance.
- Interaction-aware mutation improves the quantization-first warm G20 result relative to its standard-mutation counterpart.
- Depth → Joint warm + standard remains the best G20 sequential condition; the G50 interaction-aware comparisons are reported separately.
- Report stage-two and end-to-end search cost separately.

## Generated Artifacts

- `results/sequential_search_runs.csv`: seed-level metrics, provenance, invariants, runtime accounting, and mutation diagnostics.
- `results/sequential_search_summary.csv`: method-level mean and sample standard deviation.
- `results/sequential_search_paired_deltas.csv`: paired seed-level deltas for every G20 sequential condition, baseline, and targeted mutation comparison.
- `results/sequential_search_comparison.png`: presentation-ready quality comparison.
- `results/sequential_search_comparison.md`: this report.
