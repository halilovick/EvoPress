# Full-space 25% joint search: local-exchange crossover

## Configuration

- Model: Mistral-7B-v0.3
- Generations: 150
- Offspring per generation: 128
- Seed: 0
- Persistent population: 4
- Initial candidates: 4
- Mutation: standard
- Crossover: local_exchange
- Crossover probability: 0.5
- Parent selection: uniform
- Structural pruning: 8/32 attention and 8/32 MLP
- Exact storage target: 26,982,023,168 bits

## Final results

- WikiText-2 PPL: 9.5078125
- C4 PPL: 12.546875
- Train PPL: 9.8046875
- Final calibration KL: 0.44580078125
- Best search fitness: 0.450439453125
- Exact budget valid: True
- Compression difference: 0 bits

## Computational cost

- Candidate evaluations: 22804
- Search evaluation tokens: 235937792
- Runtime: 165663.038678054 seconds
- Total offspring attempts: 32868

## Crossover diagnostics

- Attempted: 16422
- Accepted: 2777 (16.91%)
- Duplicates: 12477 (75.98%)
- No legal exchange: 1168 (7.11%)
- Infeasible: 0
- Repaired proposals: 0
- Accepted children changing exactly two genes: confirmed by distance statistics
- Quantization exchanges: 2625 (94.53% of accepted crossover)
- Attention exchanges: 42
- MLP exchanges: 110
- Whole-block exchanges: 0

## Comparison with population-4 control

Control, seed 0:
- WikiText-2 PPL: 9.015625
- C4 PPL: 12.46875
- Candidate evaluations: 22,804
- Search evaluation tokens: 235,937,792

Local exchange has higher perplexity on both evaluation datasets
in this single-seed comparison, despite a lower best search fitness.

The operator maintains exact feasibility and requires no repair,
but produces many duplicate proposals. Most accepted crossover
children modify quantization rather than structural decisions.

This is a descriptive single-seed result, not evidence of a
general performance difference.

## Provenance

Raw experiment directory:
/home/jovyan/evopress_extension_results/fullspace_joint_s025_localxover_pop4_g150/fullspace_joint_s025_localxover_pop4_g150_seed0_attempt1

The generation checkpoint remains in the raw experiment directory.
It is not included in Git.

The saved launcher and run_summary.json contain configuration
and execution provenance.
