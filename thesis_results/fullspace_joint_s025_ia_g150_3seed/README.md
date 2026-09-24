# Interaction-aware mutation: full-space G150 three-seed summary

## Setup

- Model: Mistral-7B-v0.3
- Seeds: 0, 1, 2
- Structural pruning: 8/32 attention + 8/32 MLP
- Generations: 150
- Offspring: 128
- Population: 1
- Crossover: disabled
- Exact target: 26,982,023,168 bits
- Candidate evaluations per seed: 22,350
- Search evaluation tokens per seed: 176,947,200
- Dispersion convention: population SD

## Results

Interaction-aware:
- WikiText-2: 10.205729 +/- 0.644530
- C4: 13.434896 +/- 0.223928

Matched standard joint:
- WikiText-2: 8.960938 +/- 0.115350
- C4: 12.497396 +/- 0.128900

Matched IA minus standard:
- WikiText-2: +1.244792 +/- 0.565614
- C4: +0.937500 +/- 0.253475

Per-seed differences all have the same direction: interaction-aware
mutation produced higher WikiText-2 and C4 perplexity than standard
joint mutation for seeds 0, 1, and 2.

This supports a descriptive robustness conclusion for the tested
configuration. No statistical-significance claim is made.
