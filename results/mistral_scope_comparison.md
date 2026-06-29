# Mistral Scope Comparison Summary

This summary consolidates the Mistral-7B evidence for the thesis question: whether joint search helps when combining structural depth pruning with quantization, and how the conclusion changes when the quantization scope is expanded from `q_proj` to all attention projections `q/k/v/o`.

## Main Comparison

| Scope | Method | Compression | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | LM-eval macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | Dense FP16 | 1.000x | 5.960 | 8.860 | 7.340 | 0.787 |
| depth | Depth-only 25% | 1.317x | 11.817 | 14.760 | 12.980 | 0.647 |
| q_proj | q_proj quant-only | 1.064x | 5.938 |  |  |  |
| q_proj | Depth + independent q_proj quant | 1.400x | 11.947 | 14.910 | 13.140 | 0.647 |
| q_proj | Joint G50 depth + q_proj quant | 1.400x | 11.243 | 14.393 | 12.460 | 0.642 |
| attention | Attention q/k/v/o quant-only | 1.177x | 6.185 |  |  |  |
| attention | Depth + independent attention quant | 1.547x | 13.107 | 16.090 | 14.237 | 0.596 |
| attention | Joint G50 depth + attention quant | 1.547x | 12.670 | 15.593 | 13.783 | 0.601 |

## Core Findings

- `q_proj` joint G50 is the best quality-compression point found so far: 1.400x compression, WikiText2 PPL 11.243, C4 PPL 14.393, FineWeb-Edu PPL 12.460, and LM-eval macro 0.642.
- `q_proj` joint reduces perplexity relative to the matched independent composition by 0.703 WikiText2 PPL, 0.517 C4 PPL, and 0.680 FineWeb-Edu PPL. Its LM-eval macro is slightly lower: -0.005.
- Attention-scope joint reaches higher compression (1.547x) but with lower quality: WikiText2 PPL 12.670 and LM-eval macro 0.601.
- Within the broader attention scope, joint still reduces perplexity relative to the matched independent composition by 0.437 WikiText2 PPL, 0.497 C4 PPL, 0.453 FineWeb-Edu PPL, and 0.004 LM-eval macro.
- Depth-only remains a strong baseline: 1.317x compression, WikiText2 PPL 11.817, and LM-eval macro 0.647. Attention-scope compression does not beat this quality baseline, but it targets stronger compression.

## Thesis Interpretation

The evidence supports a nuanced claim rather than a simple win. Joint search consistently helps relative to independently combining depth and quantization at the same scope, especially in perplexity. However, expanding the quantization scope from `q_proj` to full attention projections increases compression while reducing model quality. The strongest thesis framing is therefore a compression-quality tradeoff: EvoPress-style joint search can recover part of the quality lost by broader combined compression, but the chosen scope and target compression strongly determine whether the result is competitive with depth-only pruning.

## Generated Artifacts

- `results/mistral_scope_comparison.csv`
- `results/mistral_scope_comparison.md`

## Source Summaries

- `results/mistral_medium_aggregate.csv`
- `results/mistral_generalization_aggregate.csv`
- `results/mistral_attention_generalization_aggregate.csv`
- `results/mistral_lmeval_aggregate.csv`
- `results/mistral_attention_lmeval_aggregate.csv`
- structured `run_summary.json` files under `results/runs/`
