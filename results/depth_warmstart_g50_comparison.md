# Depth Warm-Start G50 Experiment

This report tests whether depth warm-starting provides only an early convergence benefit or remains useful after 50 generations, and whether it combines with interaction-aware mutation. Lower WikiText2 perplexity (PPL) and KL fitness are better.

## Matched Experimental Design

All four conditions use `mistralai/Mistral-7B-v0.3`, the same q-projection reconstruction database, 25% separate attention/MLP depth sparsity, an exact 3-bit active budget with size grouping, 50 stage-two generations, 16 offspring, WikiText2 calibration, and the `[512, 2048, 8192]` token / `[8, 2, 1]` survivor schedule.

The standard-initialization conditions evaluate 32 initial candidates. Depth-warm conditions import the seed-matched G20 depth-only result and evaluate one exact combined initial candidate. This difference is intentional and is included in the candidate/token accounting.

## Final Quality

| Condition | Seeds | WikiText2 PPL mean ± SD | Final KL mean ± SD | Stage 2 min | Total min incl. depth stage 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard initialization + standard mutation | 3 | 11.242 ± 0.285 | 0.6128 ± 0.0528 | 22.58 | 22.58 |
| Depth→Joint warm start + standard mutation | 3 | 11.323 ± 0.364 | 0.6121 ± 0.0081 | 19.90 | 29.48 |
| Standard initialization + interaction-aware mutation | 3 | 11.086 ± 0.174 | 0.6090 ± 0.0169 | 21.19 | 21.19 |
| Depth→Joint warm start + interaction-aware mutation | 3 | 11.172 ± 0.196 | 0.6118 ± 0.0177 | 20.69 | 30.27 |

## Per-Seed Final PPL

| Seed | Standard init + standard | Depth warm + standard | Standard init + interaction | Depth warm + interaction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 11.469 | 10.945 | 10.898 | 10.984 |
| 1 | 10.922 | 11.672 | 11.117 | 11.375 |
| 2 | 11.336 | 11.352 | 11.242 | 11.156 |

## Paired Seed Deltas

Delta is `method PPL − baseline PPL`; negative values favor the method named first.

| Question | Mean delta ± SD | Seed wins |
| --- | ---: | ---: |
| Warm versus standard initialization under standard mutation | +0.081 ± 0.639 | 1/3 |
| Warm versus standard initialization under interaction-aware mutation | +0.086 ± 0.172 | 1/3 |
| Interaction-aware versus standard mutation under standard initialization | -0.156 ± 0.387 | 2/3 |
| Interaction-aware versus standard mutation under depth warm-starting | -0.151 ± 0.172 | 2/3 |

## Early Versus Late Convergence

The periodic PPL checkpoints are incomplete for standard mutation; inspect the convergence CSV.

The periodic PPL checkpoints are incomplete for interaction-aware mutation; inspect the convergence CSV.

The four generated convergence views show the same trajectories against generation, cumulative stage-two candidate evaluations, cumulative stage-two fitness-token exposures, and cumulative stage-two runtime.

### Mean Checkpoints

| Condition | Generation | Best KL fitness | WikiText2 PPL |
| --- | ---: | ---: | ---: |
| Standard initialization + standard mutation | 5 | 1.0264 |  |
| Standard initialization + standard mutation | 10 | 0.8646 |  |
| Standard initialization + standard mutation | 20 | 0.7371 |  |
| Standard initialization + standard mutation | 30 | 0.6833 |  |
| Standard initialization + standard mutation | 40 | 0.6646 |  |
| Standard initialization + standard mutation | 50 | 0.6128 | 11.242 |
| Depth→Joint warm start + standard mutation | 5 | 0.6740 |  |
| Depth→Joint warm start + standard mutation | 10 | 0.6712 |  |
| Depth→Joint warm start + standard mutation | 20 | 0.6421 |  |
| Depth→Joint warm start + standard mutation | 30 | 0.6299 |  |
| Depth→Joint warm start + standard mutation | 40 | 0.6227 |  |
| Depth→Joint warm start + standard mutation | 50 | 0.6151 | 11.323 |
| Standard initialization + interaction-aware mutation | 5 | 0.9067 |  |
| Standard initialization + interaction-aware mutation | 10 | 0.8195 |  |
| Standard initialization + interaction-aware mutation | 20 | 0.7173 |  |
| Standard initialization + interaction-aware mutation | 30 | 0.6351 |  |
| Standard initialization + interaction-aware mutation | 40 | 0.6211 |  |
| Standard initialization + interaction-aware mutation | 50 | 0.6089 | 11.086 |
| Depth→Joint warm start + interaction-aware mutation | 5 | 0.6484 |  |
| Depth→Joint warm start + interaction-aware mutation | 10 | 0.6283 |  |
| Depth→Joint warm start + interaction-aware mutation | 20 | 0.6247 |  |
| Depth→Joint warm start + interaction-aware mutation | 30 | 0.6239 |  |
| Depth→Joint warm start + interaction-aware mutation | 40 | 0.6143 |  |
| Depth→Joint warm start + interaction-aware mutation | 50 | 0.6120 | 11.172 |

## Search Cost

`Evaluated tokens` below counts calibration-token exposures in fitness evaluations at every selection stage, including parent re-evaluation for final-stage elitism. It is not the number of unique dataset tokens and does not include periodic final-PPL evaluation tokens.

| Condition | Initial candidates | Stage 2 candidate evals | Stage 2 evaluated tokens | Stage 1 candidate evals | Stage 1 evaluated tokens | Total candidate evals | Total evaluated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard initialization + standard mutation | 32 | 1,382 | 2,473,984 | 0 | 0 | 1,382 | 2,473,984 |
| Depth→Joint warm start + standard mutation | 1 | 1,351 | 2,458,112 | 572 | 999,424 | 1,923 | 3,457,536 |
| Standard initialization + interaction-aware mutation | 32 | 1,382 | 2,473,984 | 0 | 0 | 1,382 | 2,473,984 |
| Depth→Joint warm start + interaction-aware mutation | 1 | 1,351 | 2,458,112 | 572 | 999,424 | 1,923 | 3,457,536 |

## Validation

- All 12 final candidates contain exactly eight dropped attention and eight dropped MLP components.
- Every final candidate satisfies the exact active 3-bit q-projection sum.
- Warm runs use the expected seed-matched depth-only stage-one artifact.
- Stage-one depth-component hashes match the imported summaries.
- Warm initial-parent hashes match `loaded depth + uniform 3-bit q_proj` exactly.
- No run uses the deprecated `--joint_aware_mutation` path.
- Standard and interaction-aware mutation conditions were verified from saved commands.

## Interpretation

Under standard mutation, the final paired warm-start delta is +0.081 ± 0.639 PPL with 1/3 seed wins.
Under interaction-aware mutation, the final paired warm-start delta is +0.086 ± 0.172 PPL with 1/3 seed wins.
Use the generation-20 versus generation-50 deltas and the cost-normalized curves together: generation alone does not account for the 31-candidate initialization difference, while total pipeline cost additionally includes the depth-only search.

## Limitations

- Three seeds support descriptive paired comparisons, not strong significance claims.
- Runtime comes from restarted TU Wien DataLab sessions and is an approximate wall-clock measure.
- Evaluated-token exposure is a search-cost proxy; model execution cost also depends on caching and batch behavior.
- The conclusion applies to Mistral-7B, WikiText2, 25% depth sparsity, and q_proj-only active 3-bit quantization.

## Generated Artifacts

- `results/depth_warmstart_g50_runs.csv`
- `results/depth_warmstart_g50_summary.csv`
- `results/depth_warmstart_g50_paired_deltas.csv`
- `results/depth_warmstart_g50_convergence.csv`
- `results/depth_warmstart_g50_comparison.md`
- `results/depth_warmstart_g50_convergence_generation.png`
- `results/depth_warmstart_g50_convergence_candidate_evaluations.png`
- `results/depth_warmstart_g50_convergence_evaluated_tokens.png`
- `results/depth_warmstart_g50_convergence_stage2_runtime.png`
