# TinyLlama Mutation Schedule Screen

This matched ablation compares four joint-search mutation schedules: the default max-3 depth mutation, adaptive mutation, fixed strength-1 local mutation, and coarse-to-fine depth mutation. All runs use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2.

## Result

| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |
| --- | ---: | ---: | ---: | ---: |
| Default mutation (max 3) | 3 | 11.247 +/- 0.268 | 0.2607 +/- 0.0191 | 212.3 s |
| Adaptive mutation (patience 3, max 3) | 3 | 10.987 +/- 0.058 | 0.2351 +/- 0.0163 | 208.7 s |
| Fixed local mutation (strength 1) | 3 | 10.971 +/- 0.055 | 0.2347 +/- 0.0166 | 211.0 s |
| Coarse-to-fine mutation (3 -> 1) | 3 | 11.216 +/- 0.228 | 0.2624 +/- 0.0271 | 221.0 s |

Adaptive mode changes mean PPL by -0.260 versus the default and wins 2 of 3 seeds. Fixed strength 1 changes mean PPL by -0.276 and wins 2 of 3 seeds. Coarse-to-fine changes mean PPL by -0.031 and wins 2 of 3 seeds. The mean `coarse-to-fine - fixed-1` difference is +0.245 PPL.

## Paired seeds

| Seed | Default PPL | Adaptive PPL | Fixed-1 PPL | Coarse-to-fine PPL | Adaptive - default | Fixed-1 - default | Coarse - default | Coarse - fixed-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.289 | 10.922 | 10.922 | 10.961 | -0.367 | -0.367 | -0.328 | +0.039 |
| 1 | 10.961 | 11.031 | 11.031 | 11.398 | +0.070 | +0.070 | +0.438 | +0.367 |
| 2 | 11.492 | 11.008 | 10.961 | 11.289 | -0.484 | -0.531 | -0.203 | +0.328 |

## Calibration objective

| Variant | Final KL mean +/- SD | Mean delta from default |
| --- | ---: | ---: |
| Default mutation (max 3) | 0.2607 +/- 0.0191 | - |
| Adaptive mutation (patience 3, max 3) | 0.2351 +/- 0.0163 | -0.0256 |
| Fixed local mutation (strength 1) | 0.2347 +/- 0.0166 | -0.0260 |
| Coarse-to-fine mutation (3 -> 1) | 0.2624 +/- 0.0271 | +0.0017 |

## Strength schedule

| Variant seed | Strength | Generations | Generated depth | Generated quant | Selected depth | Selected quant | Parent retained |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive mutation (patience 3, max 3) seed 0 | 1 | 20 | 87 | 73 | 4 | 7 | 9 |
| Adaptive mutation (patience 3, max 3) seed 1 | 1 | 20 | 80 | 80 | 9 | 5 | 6 |
| Adaptive mutation (patience 3, max 3) seed 2 | 1 | 14 | 62 | 50 | 2 | 5 | 7 |
| Adaptive mutation (patience 3, max 3) seed 2 | 2 | 3 | 12 | 12 | 0 | 0 | 3 |
| Adaptive mutation (patience 3, max 3) seed 2 | 3 | 3 | 16 | 8 | 0 | 0 | 3 |
| Coarse-to-fine mutation (3 -> 1) seed 0 | 1 | 7 | 28 | 28 | 2 | 4 | 1 |
| Coarse-to-fine mutation (3 -> 1) seed 0 | 2 | 7 | 27 | 29 | 0 | 1 | 6 |
| Coarse-to-fine mutation (3 -> 1) seed 0 | 3 | 6 | 30 | 18 | 3 | 1 | 2 |
| Coarse-to-fine mutation (3 -> 1) seed 1 | 1 | 7 | 29 | 27 | 1 | 1 | 5 |
| Coarse-to-fine mutation (3 -> 1) seed 1 | 2 | 7 | 28 | 28 | 2 | 3 | 2 |
| Coarse-to-fine mutation (3 -> 1) seed 1 | 3 | 6 | 28 | 20 | 2 | 1 | 3 |
| Coarse-to-fine mutation (3 -> 1) seed 2 | 1 | 7 | 27 | 29 | 2 | 3 | 2 |
| Coarse-to-fine mutation (3 -> 1) seed 2 | 2 | 7 | 33 | 23 | 1 | 1 | 5 |
| Coarse-to-fine mutation (3 -> 1) seed 2 | 3 | 6 | 28 | 20 | 2 | 1 | 3 |

Elevated strengths were active for 6 generation(s), all in seed 2. They produced 0 accepted replacement(s). Seeds 0 and 1 stayed at strength 1 for all generations. Therefore the improved aggregate result cannot be attributed to escalation; the final candidates were selected entirely under strength-1 behavior.

The fixed-strength control resolves the earlier attribution problem. Seeds 0 and 1 produced byte-identical final candidates under adaptive and fixed-strength modes. Seed 2 used elevated adaptive strengths for six generations but accepted no elevated-strength replacement; fixed strength 1 was slightly better by 0.047 PPL.

The coarse-to-fine schedule is a negative screen. It is worse than fixed strength 1 in all three seeds and increases final KL relative to fixed strength 1. Starting with broad mutations and decaying to local search did not preserve the fixed-locality benefit on TinyLlama.

## Decision

The supported small-model contribution remains mutation locality, not adaptive escalation or coarse-to-fine scheduling. A single depth swap per mutation is the best TinyLlama variant tested here. Do not promote the current coarse-to-fine schedule to a three-seed Mistral run.

## Generated artifacts

- `results/adaptive_mutation_screen.csv`: paired final metrics.
- `results/adaptive_mutation_strengths.csv`: per-seed strength usage and selected mutations.
- `results/adaptive_mutation_convergence.csv`: generation-wise quality, strength, and provenance.
- `results/adaptive_mutation_screen.png`: paired final PPL and convergence.
