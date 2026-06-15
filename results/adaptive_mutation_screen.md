# TinyLlama Adaptive Mutation Screen

This matched screen compares the existing joint search with an adaptive mutation-strength variant. Both use TinyLlama, WikiText2, 12.5% depth sparsity, an active 3-bit `q_proj` budget, 20 generations, 8 offspring, and seeds 0-2. Adaptive mode starts at strength 1, increases after three retained-parent generations, and caps at strength 3.

## Result

| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD | Runtime mean |
| --- | ---: | ---: | ---: | ---: |
| Default mutation | 3 | 11.247 +/- 0.268 | 0.2607 +/- 0.0191 | 212.3 s |
| Adaptive mutation (patience 3, max 3) | 3 | 10.987 +/- 0.058 | 0.2351 +/- 0.0163 | 208.7 s |

The paired mean difference `adaptive - baseline` is -0.260 PPL with sample SD 0.292. Adaptive mode wins 2 of 3 seeds, reduces PPL variance, and changes mean final calibration KL by -0.0256.

## Paired seeds

| Seed | Baseline PPL | Adaptive PPL | PPL delta | Baseline KL | Adaptive KL | KL delta | Elevated generations | Elevated replacements |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.289 | 10.922 | -0.367 | 0.2805 | 0.2229 | -0.0576 | 0 | 0 |
| 1 | 10.961 | 11.031 | +0.070 | 0.2423 | 0.2537 | +0.0114 | 0 | 0 |
| 2 | 11.492 | 11.008 | -0.484 | 0.2593 | 0.2289 | -0.0304 | 6 | 0 |

## Strength schedule

| Seed | Strength | Generations | Generated depth | Generated quant | Selected depth | Selected quant | Parent retained |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 20 | 87 | 73 | 4 | 7 | 9 |
| 1 | 1 | 20 | 80 | 80 | 9 | 5 | 6 |
| 2 | 1 | 14 | 62 | 50 | 2 | 5 | 7 |
| 2 | 2 | 3 | 12 | 12 | 0 | 0 | 3 |
| 2 | 3 | 3 | 16 | 8 | 0 | 0 | 3 |

Elevated strengths were active for 6 generation(s), all in seed 2. They produced 0 accepted replacement(s). Seeds 0 and 1 stayed at strength 1 for all generations. Therefore the improved aggregate result cannot be attributed to escalation; the final candidates were selected entirely under strength-1 behavior.

The current comparison is also not a pure scheduling ablation. The existing baseline permits up to three depth swaps from the start, while adaptive mode begins with exactly one. The result may indicate that smaller local mutations are better, not that increasing strength after stagnation is beneficial.

## Decision

Do not promote adaptive scheduling directly to Mistral yet. The quality signal is stronger than the joint-aware probability screen, but attribution is unresolved. Run a fixed-strength-1 control with the same three seeds. If fixed strength 1 reproduces the gain, the contribution is mutation locality. If adaptive mode beats fixed strength 1, escalation has evidence and can advance.

## Generated artifacts

- `results/adaptive_mutation_screen.csv`: paired final metrics.
- `results/adaptive_mutation_strengths.csv`: per-seed strength usage and selected mutations.
- `results/adaptive_mutation_convergence.csv`: generation-wise quality, strength, and provenance.
- `results/adaptive_mutation_screen.png`: paired final PPL and convergence.
