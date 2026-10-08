# EvoPress Thesis – Population and Crossover Screening
Date: 2026-09-06

## Protocol

Model: Mistral-7B-v0.3
Depth sparsity: 25%
Quantization target: 3-bit q_proj search space
Calibration: WikiText2
Calibration tokens: 8192
Sequence length: 1024
Generations: 20
Offspring: 16
Initial candidates: 32
Selection survivors: 8 / 2 / 1
Selection tokens: 512 / 2048 / 8192
Fitness: KL divergence
Active quantization budget enabled
Joint mutation mode: standard

These are cheap screening experiments, not the full paper-matched G150 protocol.

## Population-size ablation

Mutation, population 1:

Seed 0: 13.3828125
Seed 1: 12.28125
Seed 2: 12.15625

Mean ± population std:
12.607 ± 0.675

Mutation, population 4:

Seed 0: 12.4453125
Seed 1: 11.8515625
Seed 2: 12.3046875

Mean ± population std:
12.201 ± 0.253

Interpretation:
Maintaining four persistent candidates improved WikiText2 PPL by about
0.41 compared with the single-parent mutation baseline and reduced
between-seed variance substantially.

## Original component crossover

Population size: 4
Crossover probability: 0.25
Crossover type: component

Historical three-seed WikiText2:
Seed 0: 12.6640625
Seed 1: 13.8828125
Seed 2: 12.0625

Mean ± population std:
12.870 ± 0.927

This is worse than the population-only control.

New seed-0 rerun reproduced the historical seed-0 final configuration
and result, supporting reuse of the historical 3-seed results.

New seed-0 diagnostics:
Crossover attempts: 107
Accepted: 22
Acceptance rate: 20.6%
Duplicates: 85
Duplicate rate: 79.4%
Infeasible: 0
Repaired proposals: 10
Repair gene changes: 10

Interpretation:
Component crossover mostly fails to introduce new candidates because
the small persistent population causes many crossover proposals to
duplicate existing candidates.

## Layer-bundle crossover V2

Population size: 4
Crossover probability: 0.25
Parent selection: uniform

Seed 0:
Final calibration KL: 0.8623046875
WikiText2 PPL: 14.1015625
Train PPL: 14.15625

Diagnostics:
Crossover attempts: 87
Accepted: 71
Acceptance rate: 81.6%
Duplicates: 16
Duplicate rate: 18.4%
Infeasible: 0
Repaired proposals: 66
Repair gene changes: 107

Layer-bundle crossover became the selected best parent in 5/20 generations.

Interpretation:
Layer-bundle crossover successfully solved the duplicate-proposal problem,
but the resulting crossover children were more disruptive and required
substantial feasibility repair. About 76% of proposals required repair.

The operator therefore improved crossover proposal efficiency but worsened
optimization quality. Final KL and WikiText2 PPL were both worse than the
population-only control and even worse than the population-1 baseline.

This suggests that novelty alone is insufficient. Future crossover should
preserve feasibility and locality by construction rather than recombining
freely and repairing afterward.

## Current main conclusion

Population diversity itself appears useful.

The current evidence suggests:
1. population 4 > population 1;
2. original component crossover adds no benefit beyond population size;
3. layer-bundle crossover fixes duplicates but is too disruptive;
4. the next crossover design should target feasibility-preserving,
   locality-preserving recombination.

## Important files

Master ledger:
/home/jovyan/thesis_crossover_population_ledger_20260906.csv

Complete screening archive:
/home/jovyan/crossover_v2_screening_complete_20260906.tar.gz

Checksum:
/home/jovyan/crossover_v2_screening_complete_20260906.tar.gz.sha256

Code commit for Crossover V2:
ae43aef686b42560183644177fcef9d4b783c457
