# Full-space 25% joint search — population 4, G150, seed 0

Model: Mistral-7B-v0.3
Search: G150, O128, standard mutation
Persistent population: 4
Initial candidates: 4
Crossover: disabled
Structural pruning: 8/32 attention and 8/32 MLP
Exact storage target: 26,982,023,168 bits

## Final results

WikiText-2 PPL: 9.015625
C4 PPL: 12.46875
Train PPL: 9.9453125
Best search fitness: 0.455810546875

Exact budget valid: true
Candidate evaluations: 22,804
Search evaluation tokens: 235,937,792
Reported runtime: 167,930.6496836621 seconds
Resumed from generation: 18

## Interpretation

Compared with population-1 standard joint search (seed 0),
population 4 has slightly worse WikiText-2 PPL but slightly
better C4 PPL. Search evaluation tokens increase by approximately
33%. This is a single-seed result, not evidence of a consistent
quality improvement.

Runtime accounting across the interrupted/resumed execution
should be verified before comparative wall-clock analysis.
