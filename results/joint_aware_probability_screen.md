# TinyLlama Joint-Aware Probability Screen

This matched screen compares the unchanged joint depth-plus-quantization search (`p=0`) with joint-aware mutation probability `0.25`. Both variants use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2.

## Result

| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |
| --- | ---: | ---: | ---: | ---: |
| Unchanged joint search (p=0) | 3 | 11.247 +/- 0.268 | 0.2607 +/- 0.0191 | 212.3 s |
| Joint-aware search (p=0.25) | 3 | 11.307 +/- 0.618 | 0.2569 +/- 0.0388 | 211.0 s |

The paired mean difference `p=0.25 - baseline` is +0.060 PPL with sample SD 0.607. The joint-aware variant wins 2 of 3 seeds, but its mean PPL and seed variance are both worse. Mean final calibration KL changes by -0.0038.

## Paired seeds

| Seed | Baseline PPL | p=0.25 PPL | PPL delta | Baseline KL | p=0.25 KL | KL delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.289 | 12.016 | +0.727 | 0.2805 | 0.3013 | +0.0208 |
| 1 | 10.961 | 10.875 | -0.086 | 0.2423 | 0.2401 | -0.0022 |
| 2 | 11.492 | 11.031 | -0.461 | 0.2593 | 0.2292 | -0.0300 |

## Mutation selection

| Variant | Mutation | Generated | Selected as parent | Selection rate |
| --- | --- | ---: | ---: | ---: |
| Unchanged joint search (p=0) | depth | 262 | 15 | 5.7% |
| Unchanged joint search (p=0) | quantization | 218 | 16 | 7.3% |
| Unchanged joint search (p=0) | joint_aware | 0 | 0 | n/a |
| Joint-aware search (p=0.25) | depth | 183 | 8 | 4.4% |
| Joint-aware search (p=0.25) | quantization | 180 | 18 | 10.0% |
| Joint-aware search (p=0.25) | joint_aware | 117 | 4 | 3.4% |

The `p=0.25` runs generated 117 joint-aware offspring. Only 4 became the selected parent, a 3.4% proposal-level selection rate and 4/30 accepted replacements. This provenance records the immediate winning mutation type; it does not reconstruct the full ancestry of later candidates.

## Decision

Do not promote `p=0.25` directly to a three-seed Mistral G50 experiment. The screen is mixed rather than consistently positive: it wins two seeds, loses seed 0 by 0.727 PPL, slightly worsens the mean, and increases variance. Combined with the negative Mistral `p=0.5` ablation, there is not enough evidence that probability tuning alone improves the operator.

Keep the unchanged G50 Mistral search as the primary method. The next algorithmic iteration should improve the coupled move itself or use adaptive scheduling, and should first pass this same inexpensive TinyLlama screen before Mistral promotion.

## Generated artifacts

- `results/joint_aware_probability_screen.csv`: paired seed-level metrics.
- `results/joint_aware_probability_screen_mutations.csv`: generated and selected mutation counts.
- `results/joint_aware_probability_screen_convergence.csv`: generation-wise metrics and selected mutation provenance.
- `results/joint_aware_probability_screen.png`: paired final PPL and mean convergence.
