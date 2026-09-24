# Full-space 25% joint search: interaction-aware mutation

## Configuration

- Model: Mistral-7B-v0.3
- Generations: 150
- Offspring per generation: 128
- Seed: 2
- Persistent population: 1
- Mutation: interaction_aware
- Crossover: disabled
- Structural pruning: 8/32 attention and 8/32 MLP
- Exact storage target: 26,982,023,168 bits

## Final results

- WikiText-2 PPL: 11.0703125
- C4 PPL: 13.75
- Train PPL: 10.8359375
- Final calibration KL: 0.54345703125
- Best search fitness: 0.53759765625
- Active average bitwidth: 4.083333333333333
- Searched average bitwidth: 4.099759615384615
- Exact budget valid: True
- Compression difference: 0 bits

## Computational cost

- Candidate evaluations: 22350
- Search evaluation tokens: 176947200
- Offspring attempts: 19201
- Runtime: 132331.566843129 seconds
- Resumed from generation: 0

## Matched standard baseline

Standard joint search, seed 2:

- WikiText-2 PPL: 9.109375
- C4 PPL: 12.5234375

Interaction-aware mutation has higher final perplexity on both
datasets in this matched seed-2 comparison.

## Provenance

Raw experiment directory:
/home/jovyan/evopress_extension_results/fullspace_joint_s025_ia_g150/fullspace_joint_s025_ia_g150_seed2_attempt1

The original launcher is preserved as launcher.sh.
The search checkpoint remains outside Git.
