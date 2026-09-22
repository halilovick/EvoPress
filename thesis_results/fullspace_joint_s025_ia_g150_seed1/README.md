# Full-space 25% joint search: interaction-aware mutation

## Configuration

- Model: Mistral-7B-v0.3
- Generations: 150
- Offspring per generation: 128
- Seed: 1
- Persistent population: 1
- Initial candidates: 1
- Initial candidate evaluation: skipped
- Mutation: interaction_aware
- Exact-budget ablation: enabled
- Crossover: disabled
- Structural pruning: 8/32 attention and 8/32 MLP
- Exact storage target: 26,982,023,168 bits

## Final results

- WikiText-2 PPL: 10.0234375
- C4 PPL: 13.3046875
- Train PPL: 10.5625
- Final calibration KL: 0.51123046875
- Best search fitness: 0.51318359375
- Exact budget valid: True
- Compression difference: 0 bits

## Computational cost

- Candidate evaluations: 22350
- Search evaluation tokens: 176947200
- Offspring attempts: 19200
- Reported runtime: 131619.45755694096 seconds
- Resumed from generation: 0

## Matched standard baseline

Standard joint search, seed 1:

- WikiText-2 PPL: 8.828125
- C4 PPL: 12.328125

Interaction-aware mutation has higher final perplexity on both
datasets in this matched seed-1 comparison.

This is a descriptive single-seed comparison.
The three-seed analysis will be performed after seed 2 completes.

## Provenance

Raw experiment directory:
/home/jovyan/evopress_extension_results/fullspace_joint_s025_ia_g150/fullspace_joint_s025_ia_g150_seed1_attempt1

The original launcher is preserved as launcher.sh.
The search checkpoint remains outside Git.
