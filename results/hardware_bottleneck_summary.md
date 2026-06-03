# Hardware Bottleneck Summary

The experiments were run on TU Wien Datalab with changing GPU allocations. The experiment log contains completed runs on both `Tesla T4` and `NVIDIA A40` hardware. Runtime values should therefore not be compared directly across all runs, but perplexity values remain comparable because model, dataset, sparsity, dtype, attention implementation, seed, and output directory are logged per experiment.

The current hardware snapshot from `2026-06-03` documents an `NVIDIA A40` allocation with approximately `44.4 GiB` usable VRAM. At the time of the snapshot, `nvidia-smi` reported no running GPU process and `0 MiB / 46068 MiB` VRAM in use. This indicates that GPU memory was not the limiting resource for the inspected environment.

The main constraint is the container memory limit. Although `free -h` reports a much larger host memory pool, `/sys/fs/cgroup/memory.max` is `17179869184` bytes, which is exactly `16.00 GB`. This matches the `cpu_ram_limit_gb=16.00` values recorded throughout `results/experiment_log.csv` for both T4 and A40 allocations.

This explains the observed feasibility boundary:

- EvoPress depth pruning on `mistralai/Mistral-7B-v0.3` is feasible in this environment, including the main sparsity grid, convergence extension, and seed robustness runs.
- Full Mistral-7B SparseGPT/FastOBC database generation is likely not feasible under the current Datalab container limit because that pipeline stores and processes layer-wise activation and Hessian-related state and is sensitive to CPU RAM pressure.
- A reduced TinyLlama SparseGPT database test did complete successfully. The `q_proj` database generated `22` module directories and `154` level files, with a recorded peak CPU memory of `10.31 GB` and peak GPU memory of `0.89 GB`.
- The subsequent TinyLlama sparse search also completed successfully, with final WikiText2 perplexity `9.00` and peak CPU memory `6.80 GB`.

For larger unstructured sparse database generation on Mistral-7B, the next hardware requirement is not primarily more GPU VRAM. The more important requirement is a container or node allocation with substantially more available CPU RAM, ideally at least `64 GB` and preferably more if generating databases for all projection and MLP modules at full sequence length and calibration scale.
