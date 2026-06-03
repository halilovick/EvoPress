# Baseline Comparison

Numeric aggregates include completed runs with finite WikiText2 perplexity only. Failed attempts remain in `results/experiment_log.csv` and are counted separately below.

Dense Mistral-7B WikiText2 reference PPL: `5.35`.

| Sparsity | EvoPress PPL | Random mean PPL | Random std | Random median | Random completed / failed | Late-layer PPL | Late-layer failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12.5% | 6.75 | 12693.60 | 21957.56 | 21.50 | 3/3 | 61.16 | 1 |
| 25.0% | 14.70 | 5435.12 | 7334.10 | 2490.00 | 3/0 | 541.00 | 1 |
| 37.5% | 51.69 | 5456.67 | 4863.72 | 4476.00 | 3/0 | 1419.00 | 1 |
| 50.0% | 371.75 | 15516.00 | 9939.09 | 15516.00 | 2/1 | 4992.00 | 1 |

The random baseline has very high variance, especially at low and high sparsity. The late-layer baseline is consistently worse than EvoPress. Runtime comparisons should be treated separately because the experiment log contains both Tesla T4 and NVIDIA A40 runs.
