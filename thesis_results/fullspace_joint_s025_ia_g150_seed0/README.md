# Full-space 25% joint search: interaction-aware mutation

## Configuration

- Model: Mistral-7B-v0.3
- Generations: 150
- Offspring per generation: 128
- Seed: 0
- Persistent population: 1
- Initial candidates: 1
- Initial single-candidate evaluation: skipped
- Mutation: interaction_aware
- Exact-budget ablation explicitly enabled
- Crossover: disabled
- Structural pruning: 8/32 attention and 8/32 MLP
- Exact storage target: 26,982,023,168 bits

## Final results

- WikiText-2 PPL: 9.5234375
- C4 PPL: 13.25
- Train PPL: 10.6015625
- Final calibration KL: 0.521484375
- Best search fitness: 0.5185546875
- Active average bitwidth: 4.083333333333333
- Searched average bitwidth: 4.008413461538462
- Exact budget valid: True
- Compression difference: 0 bits

## Computational cost

- Candidate evaluations: 22350
- Search evaluation tokens: 176947200
- Offspring attempts: 19200
- Reported runtime: 130814.58824614104 seconds
- Resumed from generation: 0

## Comparison with standard G150

Standard joint search, seed 0:
- WikiText-2 PPL: 8.9453125
- C4 PPL: 12.640625

Interaction-aware mutation has higher final perplexity on both
evaluation datasets in this matched seed-0 comparison.

Relative to its earlier G20 endpoint, interaction-aware search
improved over the longer search horizon, but it did not match
the standard G150 result.

This is a descriptive single-seed finding, not evidence of
a general performance difference.

## Provenance

Raw experiment directory:
/home/jovyan/evopress_extension_results/fullspace_joint_s025_ia_g150/fullspace_joint_s025_ia_g150_seed0_attempt1

The original launcher is preserved as launcher.sh.
The generation checkpoint remains outside Git.
