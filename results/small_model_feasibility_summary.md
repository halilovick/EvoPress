# Small-Model Sparse/Quant Feasibility Summary

This summary reports small-model database and search pipeline tests. Failed setup attempts are kept visible but excluded from numeric success aggregates.

## SparseGPT/FastOBC Pipeline

| Run ID | Method | Status | Model | PPL | Train PPL | Runtime (min) | GPU | Peak CPU GB | Peak GPU GB | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `sparse_db_tinyllama_qproj_s0.50` | sparse_db | failed | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | n/a | n/a | 0.12 | Tesla T4 | 2.15 | 0.00 | Failed attempt preserved; excluded from aggregates. |
| `sparse_db_tinyllama_qproj_s0.50_retry1` | sparse_db | completed | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | n/a | n/a | 1.68 | Tesla T4 | 10.31 | 0.89 | Generated 22 module dirs and 154 level files; database size 1233 MB. |
| `sparse_search_tinyllama_qproj_s0.50_g20_seed0` | sparse_search | failed | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | n/a | n/a | 0.08 | NVIDIA A40 | 1.56 | 0.00 | Failed attempt preserved; excluded from aggregates. |
| `sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1` | sparse_search | completed | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 9.00 | 9.83 | 6.18 | NVIDIA A40 | 6.80 | 2.68 | Completed 20-generation sparse search using the TinyLlama q-proj database. |

## Quantization Pipeline

No small-model quantization feasibility run is logged yet. This is optional and lower priority than consolidating the completed depth-pruning and sparse-pipeline evidence.

## Interpretation

The reduced TinyLlama SparseGPT database generation and sparse search completed end-to-end. This demonstrates that the unstructured sparse pipeline is operational on a smaller model, while the Mistral-7B full sparse database remains constrained by the current `16 GB` container RAM limit.
