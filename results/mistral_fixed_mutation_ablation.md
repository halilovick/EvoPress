# Mistral-7B Fixed Mutation-Strength Ablation

This matched experiment tests whether the fixed strength-1 depth mutation that improved the TinyLlama joint search transfers to Mistral-7B. Both variants use 25% depth sparsity, an active 3-bit `q_proj` budget, 50 generations, 16 offspring, 32 initial candidates, 8192 WikiText2 calibration tokens, sequence length 1024, and seeds 0-2. Command validation confirms that mutation strength is the only search parameter changed.

## Result

| Variant | Seeds | WikiText2 PPL mean +/- SD | Final KL mean +/- SD |
| --- | ---: | ---: | ---: |
| Default mutation (max 3) | 3 | 11.242 +/- 0.285 | 0.6128 +/- 0.0525 |
| Fixed local mutation (strength 1) | 3 | 11.211 +/- 0.263 | 0.6133 +/- 0.0591 |

The paired mean difference `fixed-1 - max-3` is -0.031 PPL with sample SD 0.164. Fixed strength 1 wins 1 of 3 seeds. Mean final calibration KL changes by +0.0005, which is effectively neutral.

## Paired seeds

| Seed | Max-3 PPL | Fixed-1 PPL | PPL delta | Max-3 KL | Fixed-1 KL | KL delta | Attention Jaccard | MLP Jaccard | Equal quant bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.469 | 11.508 | +0.039 | 0.6641 | 0.6763 | +0.0122 | 0.600 | 0.600 | 21/32 |
| 1 | 10.922 | 11.008 | +0.086 | 0.5591 | 0.5591 | +0.0000 | 0.600 | 0.778 | 24/32 |
| 2 | 11.336 | 11.117 | -0.219 | 0.6152 | 0.6045 | -0.0107 | 0.231 | 0.455 | 19/32 |

## Interpretation

The TinyLlama locality result does not transfer as a clear Mistral improvement. Fixed strength 1 slightly improves the three-seed mean and variance, but it loses seeds 0 and 1 and gains mainly through seed 2. The effect size is small relative to seed variation, while the optimized masks and quantization profiles remain materially different.

This is useful scale-dependent evidence. Small local mutations appear helpful in TinyLlama's smaller search space, but Mistral's larger depth configuration may benefit from occasional multi-swap moves. Neither fixed strength 1 nor unrestricted max-3 is uniformly superior from the current three seeds.

## Runtime caveat

The max-3 runs used Tesla V100-SXM2-32GB, while fixed-1 used NVIDIA A40. Therefore the shorter fixed-1 wall-clock runtime must not be attributed to mutation strength.

## Decision

Keep the unchanged max-3 G50 search as the primary Mistral method because fixed strength 1 does not consistently improve it. Do not run more seeds for this binary comparison yet. The next useful algorithmic experiment is a scheduled locality operator that starts with broader mutations and decays toward strength 1, with stagnation-based expansion only if provenance shows expanded moves being selected. Screen that design on TinyLlama before another three-seed Mistral run.

## Generated artifacts

- `results/mistral_fixed_mutation_ablation.csv`: paired final metrics and candidate overlap.
- `results/mistral_fixed_mutation_convergence.csv`: generation-wise search and evaluation metrics.
- `results/mistral_fixed_mutation_ablation.png`: paired final PPL and mean convergence.
