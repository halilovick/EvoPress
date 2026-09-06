# Interaction-Aware Mistral q_proj Summary

This summary compares the original standard Mistral-7B joint depth+`q_proj` quantization search against the new interaction-aware mutation operator.

Both settings use:

- model: `mistralai/Mistral-7B-v0.3`
- depth sparsity: 25%
- quantization scope: `q_proj`
- target active bitwidth: 3.0
- generations: 50
- offspring: 16
- seeds: 0, 1, 2
- calibration data: WikiText2
- sequence length: 1024
- calibration tokens: 8192
- active quantization budget: enabled
- estimated compression ratio: 1.400x

## Search-Final Paired Results

| Seed | Standard W2 PPL | Interaction-aware W2 PPL | Delta | Standard train PPL | Interaction-aware train PPL | Delta | Standard KL | Interaction-aware KL | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.469 | 10.898 | -0.570 | 11.781 | 11.242 | -0.539 | 0.664 | 0.610 | -0.054 |
| 1 | 10.922 | 11.117 | +0.195 | 10.398 | 11.008 | +0.609 | 0.559 | 0.591 | +0.032 |
| 2 | 11.336 | 11.242 | -0.094 | 10.102 | 10.117 | +0.016 | 0.615 | 0.625 | +0.010 |

## Aggregate

| Metric | Standard mean | Interaction-aware mean | Mean delta | Wins |
| --- | ---: | ---: | ---: | ---: |
| WikiText2 PPL | 11.242 | 11.086 | -0.156 | 2/3 |
| Train PPL | 10.760 | 10.789 | +0.029 | 1/3 |
| Final calibration KL | 0.613 | 0.609 | -0.004 | 1/3 |
| Runtime seconds | 1355.0 | 1271.3 | -83.7 | 3/3 |

## Cross-Dataset Replay

The three interaction-aware final candidates were replayed on WikiText2, C4, and FineWeb-Edu with the same evaluation setup used for the standard joint G50 candidates.

| Seed | Standard W2 | Interaction-aware W2 | Delta | Standard C4 | Interaction-aware C4 | Delta | Standard FineWeb-Edu | Interaction-aware FineWeb-Edu | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.47 | 10.90 | -0.57 | 14.44 | 14.19 | -0.25 | 12.42 | 12.35 | -0.07 |
| 1 | 10.92 | 11.12 | +0.20 | 14.19 | 13.94 | -0.25 | 12.26 | 12.02 | -0.24 |
| 2 | 11.34 | 11.24 | -0.10 | 14.55 | 14.27 | -0.28 | 12.70 | 12.30 | -0.40 |

| Dataset | Standard mean | Interaction-aware mean | Mean delta | Wins |
| --- | ---: | ---: | ---: | ---: |
| WikiText2 PPL | 11.243 | 11.087 | -0.157 | 2/3 |
| C4 PPL | 14.393 | 14.133 | -0.260 | 3/3 |
| FineWeb-Edu PPL | 12.460 | 12.223 | -0.237 | 3/3 |

## LM-Eval Replay

The three interaction-aware final candidates were also evaluated on the same LM-eval task set used for the standard q_proj comparison. Higher is better.

| Task / aggregate | Standard joint G50 | Interaction-aware joint G50 | Delta |
| --- | ---: | ---: | ---: |
| ARC-Easy acc_norm | 0.580 | 0.602 | +0.023 |
| PIQA acc_norm | 0.739 | 0.736 | -0.003 |
| Winogrande acc | 0.608 | 0.591 | -0.017 |
| Macro average | 0.642 | 0.643 | +0.001 |

Against the independently composed depth+q_proj baseline, interaction-aware is still lower on macro average:

| Comparison | Macro delta |
| --- | ---: |
| Interaction-aware joint - standard joint | +0.001 |
| Interaction-aware joint - independent composition | -0.004 |
| Interaction-aware joint - depth-only | -0.004 |

## Interpretation

The new interaction-aware mutation is a useful implementation extension and shows a modest positive signal on perplexity. It improves the three-seed mean on WikiText2, C4, and FineWeb-Edu at the same estimated compression ratio. The strongest cross-dataset signal is on C4 and FineWeb-Edu, where it wins all three seeds. The result is still not uniform across all metrics: train PPL is essentially tied, final calibration KL improves only slightly on average, and downstream LM-eval is task-dependent.

This should be presented as encouraging but still conservative evidence. The current result supports saying that interaction-aware mutation is a concrete algorithmic extension that improves perplexity against the previous standard joint search in this Mistral q_proj setting. It does not support saying that interaction-aware mutation clearly improves downstream task accuracy.

## Needed Follow-Up

For thesis writing, the next step is to explain why the perplexity gain does not transfer cleanly to every downstream task. A broader task set can be run later, but the current three-task LM-eval result is enough to report the limitation honestly.
