# Detailed EvoPress Experiment Report

## Purpose of This Report

This document explains what has been done so far in the EvoPress experiments, why each experiment was run, what the results mean, and how the results fit into a master-thesis progress story.

The central question was practical: under the available TU Wien Datalab environment, can EvoPress produce meaningful compression results for a large language model, and can we document clearly why some heavier experiments are not feasible in the current container?

The answer from the current evidence is yes. EvoPress depth pruning works on Mistral-7B, beats simple baselines, improves with more search generations, shows reasonable seed robustness, and the smaller-model SparseGPT pipeline works end-to-end. The main unresolved limitation is not GPU VRAM; it is the `16 GB` CPU/container RAM limit.

## 1. What Was Set Up First

Before running the main experiments, the repository needed reliable experiment bookkeeping. This matters because thesis experiments are only useful if every result can be traced back to exact parameters, logs, and output files.

The following infrastructure was added:

- `results/experiment_log.csv`
- `scripts/append_experiment_log.py`
- `scripts/parse_depth_search_log.py`
- parameterized depth-pruning launchers
- dense-evaluation launcher
- baseline-evaluation launchers
- sparse database and sparse-search launchers
- `scripts/check_runtime_dependencies.py`
- `scripts/build_experiment_report.py`

The most important file is `results/experiment_log.csv`. Each run records the date, run ID, method, model, sparsity or bits, generations, offspring, calibration setup, dtype, attention implementation, seed, WikiText2 perplexity, train perplexity, runtime, GPU, GPU VRAM, CPU RAM limit, status, notes, and output directory.

This means failed runs are not hidden. They are preserved and labeled. This is important because some failures were setup failures, such as forgetting to reinstall `requirements.txt` after a Datalab restart, while other failures, such as a random-pruning collapse to non-finite perplexity, are scientifically meaningful.

## 2. The Experimental Environment

The experiments were run on TU Wien Datalab. The GPU allocation changed depending on availability. Some runs used a `Tesla T4` and others used an `NVIDIA A40`.

This affects runtime comparisons. For example, a run on an A40 can be much faster than a run on a T4. Therefore, runtime numbers should not be used as direct algorithmic comparisons unless the hardware is the same.

Perplexity comparisons are still meaningful because the model, data, sparsity, dtype, attention implementation, seed, and method are all logged.

The hardware snapshot from `2026-06-03` shows:

- GPU: `NVIDIA A40`
- GPU memory visible to PyTorch: approximately `44.42 GiB`
- `nvidia-smi`: `0 MiB / 46068 MiB` used at snapshot time
- cgroup memory limit: `17179869184` bytes
- cgroup memory limit in GiB: `16.00 GB`

The key point is that `free -h` shows a large host memory pool, but the container itself is limited to `16 GB`. For the EvoPress depth-pruning experiments, this is enough. For full Mistral-7B SparseGPT database generation, it is likely not enough.

## 3. Dense Mistral-7B Reference

Before compression results can be interpreted, a dense reference is needed. The dense `mistralai/Mistral-7B-v0.3` WikiText2 evaluation produced:

| Model | WikiText2 PPL |
| --- | ---: |
| Dense Mistral-7B | 5.35 |

Perplexity is lower-is-better. A compressed model is expected to have worse perplexity than the dense model, but the question is how quickly the quality degrades as compression increases and whether EvoPress degrades more gracefully than simple baselines.

## 4. Main Experiment: Mistral-7B Depth-Pruning Curve

The main depth-pruning experiment ran EvoPress on `mistralai/Mistral-7B-v0.3` at four sparsity levels:

- `12.5%`
- `25.0%`
- `37.5%`
- `50.0%`

The common settings were:

- calibration data: WikiText2
- sequence length: `2048`
- calibration tokens: `8192`
- generations: `10`
- offspring: `8`
- fitness function: KL divergence
- dtype: `float16`
- attention implementation: `sdpa`
- seed: `1`

The result was:

| Sparsity | EvoPress WikiText2 PPL |
| --- | ---: |
| 12.5% | 6.75 |
| 25.0% | 14.70 |
| 37.5% | 51.69 |
| 50.0% | 371.75 |

This is the main compression-versus-quality curve. It shows that quality decreases as more modules are removed, which is expected. The important point is that the decrease is much better controlled than with the simple baselines.

The plot is stored in `results/depth_pruning_curve.png`.

## 5. What Depth Pruning Means Here

In these experiments, depth pruning removes selected attention and MLP modules from the transformer stack. At each sparsity level, the launcher records how many attention and MLP modules are dropped.

For the main grid:

- `12.5%`: `4` attention modules and `4` MLP modules dropped
- `25.0%`: `8` attention modules and `8` MLP modules dropped
- `37.5%`: `12` attention modules and `12` MLP modules dropped
- `50.0%`: `16` attention modules and `16` MLP modules dropped

The EvoPress search chooses which modules to drop. This is the key difference from a baseline that drops modules randomly or simply removes late layers.

## 6. Baseline Experiments

Baselines are necessary because a compression method only looks useful if it beats simple alternatives.

Two cheap baselines were implemented and evaluated:

- random dropping baseline
- late-layer dropping baseline

The random baseline removes the same number of attention and MLP modules as the EvoPress configuration but chooses them randomly. It was run with multiple seeds. The late-layer baseline removes modules from deeper layers first.

The comparison was:

| Sparsity | EvoPress PPL | Random mean PPL | Random median PPL | Late-layer PPL |
| --- | ---: | ---: | ---: | ---: |
| 12.5% | 6.75 | 12693.60 | 21.50 | 61.16 |
| 25.0% | 14.70 | 5435.12 | 2490.00 | 541.00 |
| 37.5% | 51.69 | 5456.67 | 4476.00 | 1419.00 |
| 50.0% | 371.75 | 15516.00 | 15516.00 | 4992.00 |

This is one of the strongest results in the current experiment set. EvoPress is better than both baselines at every sparsity level.

The random baseline is especially unstable. For example, at `12.5%`, one random seed produced a reasonable-looking result while another produced a very large PPL. At `50.0%`, one random run collapsed to non-finite perplexity. That collapse is useful evidence: it shows that simply removing the right number of modules is not enough. Which modules are removed matters.

The baseline table is stored in `results/baseline_comparison_table.md`.

## 7. Convergence Extension

The original 10-generation `37.5%` EvoPress run had WikiText2 PPL `51.69`. To check whether the search was still improving, a longer `20`-generation run was performed at the same sparsity.

Key convergence points:

| Generation | WikiText2 PPL | Train PPL |
| ---: | ---: | ---: |
| 1 | 277.25 | 267.00 |
| 5 | 80.69 | 74.60 |
| 10 | 51.69 | 48.20 |
| 15 | 33.78 | 32.10 |
| 20 | 26.00 | 24.50 |

This result is important because it shows that generation 10 was not necessarily the end of useful optimization. More generations reduced WikiText2 PPL from `51.69` to `26.00`.

In thesis terms, this supports the claim that EvoPress is doing a real search rather than just getting a lucky initial mask. The convergence plot is stored in `results/convergence_37_5.png`.

## 8. Seed Robustness

A single successful seed is not enough. To check repeatability, the `37.5%` setting was run with seeds `1`, `2`, and `3`, using `10` generations and `8` offspring.

Results:

| Seed | WikiText2 PPL | Train PPL |
| ---: | ---: | ---: |
| 1 | 51.69 | 48.20 |
| 2 | 40.19 | 28.50 |
| 3 | 47.91 | 41.90 |

Summary:

- mean WikiText2 PPL: `46.60`
- sample standard deviation: `5.86`
- best seed: `2`
- worst seed: `1`

The selected dropped-module sets were similar but not identical. Pairwise Jaccard overlap over dropped `(layer_index, module_type)` pairs ranged from `0.371` to `0.412`.

This means the search is reasonably stable in terms of final quality, even though different seeds can choose different but related pruning masks.

The seed robustness table is stored in `results/seed_robustness_table.md`.

## 9. SparseGPT/FastOBC Feasibility

The full Mistral-7B SparseGPT database generation had previously crashed in this environment, likely because of CPU RAM pressure. Instead of repeatedly trying the full Mistral setup, the experiment plan switched to a smaller-model feasibility test.

The selected model was:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The SparseGPT database test used:

- target sparsity: `50%`
- module scope: `q_proj` only
- calibration data: WikiText2
- calibration tokens: `4096`
- sequence length: `1024`
- levels: `3`
- dtype: `float16`
- attention implementation: `sdpa`

The database generation completed successfully:

| Metric | Value |
| --- | ---: |
| Module directories | 22 |
| Level files | 154 |
| Database size | 1233 MB |
| Runtime | 1.68 min |
| Peak CPU memory | 10.31 GB |
| Peak GPU memory | 0.89 GB |

The `154` level files are exactly what is expected: TinyLlama has `22` transformer layers and the test generated seven levels per layer, from `-3` through `3`.

This result proves that the SparseGPT/FastOBC database-generation path is operational in the repository and Datalab environment for a smaller model and reduced module scope.

## 10. Sparse Search on TinyLlama

After the TinyLlama sparse database was generated, a small sparse search was run:

- generations: `20`
- offspring: `8`
- fitness function: KL divergence
- calibration data: WikiText2
- evaluation data: WikiText2

The search completed successfully:

| Metric | Value |
| --- | ---: |
| Final WikiText2 PPL | 9.00 |
| Final train PPL | 9.83 |
| Runtime | 6.18 min |
| Peak CPU memory | 6.80 GB |
| Peak GPU memory | 2.68 GB |

The final sparse configuration had `22` entries, one for each `q_proj` module. The selected levels ranged from `-2` to `3`, and the total level sum was `0`, which means the global sparsity budget was preserved.

This is important because it demonstrates the full smaller-model unstructured sparse pipeline:

1. generate sparse weight database;
2. run evolutionary sparse allocation search;
3. evaluate WikiText2 perplexity;
4. save final sparse configuration.

The small-model feasibility summary is stored in `results/small_model_feasibility_summary.md`.

## 11. Dependency and Runtime Issues

Several failed rows in the experiment log came from Datalab restarts where `requirements.txt` had not yet been installed again. These failures are not scientific failures of EvoPress. They are preserved in the log as setup failures.

To prevent this from happening repeatedly, `scripts/check_runtime_dependencies.py` was added and wired into the main launchers. It checks that required packages are importable before a real experiment starts:

- `datasets`
- `numpy`
- `torch`
- `transformers`
- `tqdm`
- `accelerate`
- `sentencepiece`

It also checks CUDA availability when required. This helps fail fast before creating another misleading run directory.

## 12. What the Results Mean for the Thesis

The current results support the following thesis-progress claims:

1. EvoPress depth pruning is feasible for Mistral-7B in the available environment.
2. EvoPress depth pruning produces a clear compression-quality curve.
3. EvoPress is substantially better than random and late-layer module dropping.
4. The evolutionary search keeps improving with additional generations.
5. The method is reasonably repeatable across three seeds at the tested setting.
6. The current Datalab container is sufficient for depth pruning but not suitable for full Mistral-7B SparseGPT database generation.
7. The SparseGPT/FastOBC pipeline itself is understood and executable on a smaller model.

This is already a coherent experimental story for a supervision meeting.

## 13. What Should Be Shown in the Meeting

The most important files to show are:

- `results/depth_pruning_curve.png`
- `results/convergence_37_5.png`
- `results/baseline_comparison_table.md`
- `results/seed_robustness_table.md`
- `results/small_model_feasibility_summary.md`
- `results/hardware_bottleneck_summary.md`

The recommended story is:

1. Start with the dense Mistral-7B baseline PPL of `5.35`.
2. Show the EvoPress depth-pruning curve.
3. Show that EvoPress beats random and late-layer baselines.
4. Show the convergence extension from PPL `51.69` at generation 10 to `26.00` at generation 20.
5. Show three-seed robustness with mean PPL `46.60` and std `5.86`.
6. Explain that full Mistral SparseGPT database generation is constrained by the `16 GB` container RAM limit.
7. Show that the TinyLlama SparseGPT database and sparse search completed end-to-end.

## 14. Limitations to State Clearly

The limitations should be stated explicitly:

- The Datalab GPU allocation changed between runs, so runtime is not directly comparable across all experiments.
- The full Mistral SparseGPT/FastOBC database was not completed in this environment.
- The successful SparseGPT pipeline test is on TinyLlama and `q_proj` only, so it is a feasibility demonstration, not a full Mistral sparse-compression result.
- Some failed runs were caused by missing dependencies after Datalab restarts and should be treated as setup failures.
- The random baseline is very unstable, so mean, median, standard deviation, and failure count should be reported together.

## 15. Recommended Next Steps

Before adding more experiments, the next step is to use the current results in the supervision meeting and get feedback.

After that, the best follow-up depends on the supervisor's priorities:

- If the focus is depth pruning, run more seeds or a wider generation budget for the strongest Mistral settings.
- If the focus is unstructured sparsity, request a Datalab/container allocation with more CPU RAM and retry Mistral SparseGPT database generation.
- If the focus is pipeline coverage, run the optional TinyLlama GPTQ debug pipeline.
- If the focus is thesis writing, convert the current results into a methods/results section draft.

The most defensible hardware request is not more GPU VRAM first. It is more CPU/container RAM. A `32 GB` RAM retry would be informative, but `64 GB` or more is the better target for full Mistral-7B SparseGPT/FastOBC database generation.
