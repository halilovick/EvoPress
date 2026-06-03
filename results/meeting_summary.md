# EvoPress Thesis Progress Meeting Summary

## 1. Scope and Experimental Question

This experiment phase focused on producing a coherent master-thesis progress story for EvoPress under the available TU Wien Datalab constraints. The goal was not to reproduce every full-scale paper result, but to establish whether EvoPress depth pruning produces useful Mistral-7B compression behavior, whether it outperforms simple depth-pruning baselines, whether the search is repeatable, and whether the heavier unstructured sparse pipeline is feasible in the current environment.

The main model for depth pruning was `mistralai/Mistral-7B-v0.3`. The small-model sparse-pipeline feasibility test used `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## 2. Environment and Hardware Constraints

The experiments were run on TU Wien Datalab with changing GPU allocations. Logged runs include both `Tesla T4` and `NVIDIA A40` hardware. Runtime comparisons across all runs should therefore be treated carefully. Perplexity comparisons remain valid because each experiment row records model, dataset, sparsity, dtype, attention implementation, seed, hardware, and output directory.

The current hardware snapshot from `2026-06-03` documents an `NVIDIA A40` allocation with `0 MiB / 46068 MiB` VRAM in use at snapshot time. PyTorch reported approximately `44.42 GiB` GPU memory. The important limiting factor is the container memory limit: `/sys/fs/cgroup/memory.max` is `17179869184` bytes, or exactly `16.00 GB`. This `16.00 GB` CPU/container RAM limit is also recorded across the experiment log.

Hardware evidence is stored in:

- `results/hardware_snapshot_2026-06-03_a40.txt`
- `results/hardware_bottleneck_summary.md`

## 3. Successfully Completed Experiments

The following experiment groups were completed and logged:

- Dense Mistral-7B WikiText2 reference evaluation.
- Mistral-7B EvoPress depth-pruning grid at `12.5%`, `25.0%`, `37.5%`, and `50.0%` sparsity.
- `37.5%` Mistral-7B convergence extension to `20` generations.
- Random and late-layer depth-pruning baselines.
- Three-seed EvoPress repeatability test at `37.5%` sparsity.
- TinyLlama SparseGPT/FastOBC `q_proj` database generation.
- TinyLlama sparse-search run using the generated sparse database.

Failed setup attempts are preserved in `results/experiment_log.csv` and excluded from numeric aggregates. The report builder counted `26` completed rows and `11` non-completed rows.

## 4. Main Mistral-7B Depth-Pruning Curve

The dense Mistral-7B WikiText2 reference PPL was `5.35`. EvoPress depth pruning produced the following WikiText2 PPL values for the main seed-1, 10-generation grid:

| Sparsity | EvoPress WikiText2 PPL |
| --- | ---: |
| 12.5% | 6.75 |
| 25.0% | 14.70 |
| 37.5% | 51.69 |
| 50.0% | 371.75 |

The generated curve is stored in:

- `results/depth_pruning_curve.csv`
- `results/depth_pruning_curve.png`

The result shows the expected compression-quality tradeoff: perplexity increases as more attention and MLP modules are removed, but the degradation is controlled compared with simple baselines.

## 5. Dense and Simple-Baseline Comparison

EvoPress outperformed both random dropping and the late-layer heuristic at every tested sparsity.

| Sparsity | EvoPress PPL | Random mean PPL | Random median PPL | Late-layer PPL |
| --- | ---: | ---: | ---: | ---: |
| 12.5% | 6.75 | 12693.60 | 21.50 | 61.16 |
| 25.0% | 14.70 | 5435.12 | 2490.00 | 541.00 |
| 37.5% | 51.69 | 5456.67 | 4476.00 | 1419.00 |
| 50.0% | 371.75 | 15516.00 | 15516.00 | 4992.00 |

The random baseline was highly unstable. At `50.0%`, one random seed collapsed to non-finite perplexity and is recorded as a failed run. The comparison table is stored in `results/baseline_comparison_table.md`.

## 6. 37.5% Convergence Behavior

The `37.5%` EvoPress depth-pruning run improved substantially when extended to `20` generations:

| Generation | WikiText2 PPL | Train PPL |
| ---: | ---: | ---: |
| 1 | 277.25 | 267.00 |
| 5 | 80.69 | 74.60 |
| 10 | 51.69 | 48.20 |
| 15 | 33.78 | 32.10 |
| 20 | 26.00 | 24.50 |

This demonstrates that the 10-generation setting was still improving and that additional search budget can materially improve final perplexity. The convergence artifacts are stored in:

- `results/convergence_37_5.csv`
- `results/convergence_37_5.png`

## 7. Three-Seed Repeatability Result

The three-seed `37.5%` repeatability test completed with finite WikiText2 PPL values:

| Seed | WikiText2 PPL | Train PPL |
| ---: | ---: | ---: |
| 1 | 51.69 | 48.20 |
| 2 | 40.19 | 28.50 |
| 3 | 47.91 | 41.90 |

Mean WikiText2 PPL was `46.60` with sample standard deviation `5.86`. Pairwise dropped-module Jaccard overlap ranged from `0.371` to `0.412`. The selected masks are similar but not identical, indicating that the search is finding related high-quality regions rather than a single fixed mask.

The repeatability table is stored in `results/seed_robustness_table.md`.

## 8. Small-Model SparseGPT Feasibility Result

Full Mistral-7B SparseGPT database generation was not retried immediately after earlier crashes. Instead, the smaller TinyLlama pipeline was tested first.

The TinyLlama SparseGPT/FastOBC database generation completed for `q_proj` modules:

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Scope: `q_proj` only
- Sparsity: `50%`
- Levels: `3`, producing levels `-3` through `3`
- Generated module directories: `22`
- Generated level files: `154`
- Database size: `1233 MB`
- Runtime: `1.68 min`
- Peak CPU memory: `10.31 GB`
- Peak GPU memory: `0.89 GB`

The subsequent TinyLlama sparse search completed:

- Generations: `20`
- Offspring: `8`
- Final WikiText2 PPL: `9.00`
- Final train PPL: `9.83`
- Runtime: `6.18 min`
- Peak CPU memory: `6.80 GB`
- Peak GPU memory: `2.68 GB`

This confirms that the sparse database and sparse search pipeline works end-to-end on a smaller model in the current environment.

## 9. Optional GPTQ Feasibility Result

No small-model GPTQ feasibility run was performed in this phase. This is intentionally left as optional because the completed depth-pruning and SparseGPT feasibility experiments already provide a coherent progress story.

## 10. Limitations

The main limitations are:

- Runtime comparisons are confounded by variable GPU allocation across Datalab sessions.
- Full Mistral-7B SparseGPT/FastOBC database generation has not been successfully completed in the current `16 GB` CPU/container RAM environment.
- The TinyLlama sparse-pipeline test was intentionally reduced to `q_proj` modules only and should not be interpreted as a full Mistral sparse-compression result.
- Some failed rows are setup failures caused by missing dependencies after Datalab restarts; these are logged but excluded from numeric aggregates.
- The random baseline mean values are dominated by very unstable runs, so medians and failure counts should be reported alongside means.

## 11. Next Hardware Requirement Estimate

The current evidence indicates that GPU VRAM is not the primary bottleneck for the inspected allocation. The A40 snapshot showed approximately `44.4 GiB` usable VRAM and no active GPU process. The consistent constraint is the `16.00 GB` container RAM limit.

For full Mistral-7B unstructured sparse database generation, the next hardware requirement should prioritize substantially more CPU/container RAM. A measured retry on at least `32 GB` CPU RAM would be useful, but `64 GB` or more is the more defensible target for full Mistral-7B database generation across all projection and MLP modules at larger calibration scale. Disk space should also be monitored because sparse and quantized database artifacts can become large.

## 12. Next Experimental Steps

The immediate next step is to use the generated artifacts for the thesis-progress meeting:

- `results/depth_pruning_curve.png`
- `results/convergence_37_5.png`
- `results/baseline_comparison_table.md`
- `results/seed_robustness_table.md`
- `results/small_model_feasibility_summary.md`
- `results/hardware_bottleneck_summary.md`

After the meeting, the next technical step should be chosen based on supervisor feedback. The strongest candidates are:

- run full Mistral SparseGPT/FastOBC generation only on a higher-RAM allocation;
- broaden the TinyLlama SparseGPT test beyond `q_proj` if a larger sparse-pipeline demonstration is needed;
- optionally run the TinyLlama GPTQ debug pipeline if quantization coverage becomes necessary.
