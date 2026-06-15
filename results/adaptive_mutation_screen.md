# TinyLlama Adaptive Mutation Screen

This matched ablation compares the existing joint search, adaptive mutation strength, and a fixed strength-1 control. All runs use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2.

## Result

| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |
| --- | ---: | ---: | ---: | ---: |
| Default mutation (max 3) | 3 | 11.247 +/- 0.268 | 0.2607 +/- 0.0191 | 212.3 s |
| Adaptive mutation (patience 3, max 3) | 3 | 10.987 +/- 0.058 | 0.2351 +/- 0.0163 | 208.7 s |
| Fixed local mutation (strength 1) | 3 | 10.971 +/- 0.055 | 0.2347 +/- 0.0166 | 211.0 s |

Adaptive mode changes mean PPL by -0.260 versus the default and wins 2 of 3 seeds. Fixed strength 1 changes mean PPL by -0.276 and wins 2 of 3 seeds. The mean `adaptive - fixed` difference is +0.016 PPL.

## Paired seeds

| Seed | Default PPL | Adaptive PPL | Fixed-1 PPL | Adaptive - default | Fixed-1 - default | Adaptive - fixed-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.289 | 10.922 | 10.922 | -0.367 | -0.367 | +0.000 |
| 1 | 10.961 | 11.031 | 11.031 | +0.070 | +0.070 | +0.000 |
| 2 | 11.492 | 11.008 | 10.961 | -0.484 | -0.531 | +0.047 |

## Calibration objective

| Variant | Final KL mean +/- SD | Mean delta from default |
| --- | ---: | ---: |
| Default mutation (max 3) | 0.2607 +/- 0.0191 | - |
| Adaptive mutation (patience 3, max 3) | 0.2351 +/- 0.0163 | -0.0256 |
| Fixed local mutation (strength 1) | 0.2347 +/- 0.0166 | -0.0260 |

## Strength schedule

| Seed | Strength | Generations | Generated depth | Generated quant | Selected depth | Selected quant | Parent retained |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 20 | 87 | 73 | 4 | 7 | 9 |
| 1 | 1 | 20 | 80 | 80 | 9 | 5 | 6 |
| 2 | 1 | 14 | 62 | 50 | 2 | 5 | 7 |
| 2 | 2 | 3 | 12 | 12 | 0 | 0 | 3 |
| 2 | 3 | 3 | 16 | 8 | 0 | 0 | 3 |

Elevated strengths were active for 6 generation(s), all in seed 2. They produced 0 accepted replacement(s). Seeds 0 and 1 stayed at strength 1 for all generations. Therefore the improved aggregate result cannot be attributed to escalation; the final candidates were selected entirely under strength-1 behavior.

The fixed-strength control resolves the earlier attribution problem. Seeds 0 and 1 produced byte-identical final candidates under adaptive and fixed-strength modes. Seed 2 used elevated adaptive strengths for six generations but accepted no elevated-strength replacement; fixed strength 1 was slightly better by 0.047 PPL.

## Decision

The supported contribution is mutation locality, not adaptive escalation. A single depth swap per mutation improved mean PPL and substantially reduced variance relative to allowing up to three swaps. Promote fixed strength 1 to a matched Mistral ablation. Do not spend Mistral compute on the current adaptive schedule unless a future design makes elevated mutations demonstrably useful.

## Generated artifacts

- `results/adaptive_mutation_screen.csv`: three-way paired final metrics.
- `results/adaptive_mutation_strengths.csv`: per-seed strength usage and selected mutations.
- `results/adaptive_mutation_convergence.csv`: generation-wise quality, strength, and provenance.
- `results/adaptive_mutation_screen.png`: paired final PPL and convergence.
