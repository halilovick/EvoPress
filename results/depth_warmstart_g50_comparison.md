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

Under standard mutation, the warm-start delta is -1.081 PPL after 20 completed generations and +0.081 after generation 50; the early warm-start advantage reverses by generation 50.

Under interaction-aware mutation, the warm-start delta is -0.753 PPL after 20 completed generations and +0.086 after generation 50; the early warm-start advantage reverses by generation 50.

The four generated convergence views show the same trajectories against completed generations, cumulative stage-two candidate evaluations, cumulative stage-two fitness-token exposures, and cumulative stage-two runtime.

The source log evaluates the parent before each generation's mutation but writes it on the row numbered for that generation. Accordingly, logged row 1 is the initialized parent (zero completed generations), logged rows 6, 11, ..., 46 are the parents after 5, 10, ..., 45 completed generations, and the final point uses the post-generation-50 metrics from `run_summary.json`.

### Mean Checkpoints

| Condition | Completed generations | Best KL fitness | WikiText2 PPL |
| --- | ---: | ---: | ---: |
| Standard initialization + standard mutation | 0 | 1.2744 | 23.333 |
| Standard initialization + standard mutation | 5 | 0.9813 | 15.828 |
| Standard initialization + standard mutation | 10 | 0.8366 | 13.711 |
| Standard initialization + standard mutation | 15 | 0.7557 | 12.661 |
| Standard initialization + standard mutation | 20 | 0.7352 | 12.607 |
| Standard initialization + standard mutation | 25 | 0.7065 | 12.289 |
| Standard initialization + standard mutation | 30 | 0.6826 | 11.938 |
| Standard initialization + standard mutation | 35 | 0.6753 | 11.878 |
| Standard initialization + standard mutation | 40 | 0.6564 | 11.656 |
| Standard initialization + standard mutation | 45 | 0.6209 | 11.292 |
| Standard initialization + standard mutation | 50 | 0.6128 | 11.242 |
| Depth→Joint warm start + standard mutation | 0 | 0.7759 | 11.956 |
| Depth→Joint warm start + standard mutation | 5 | 0.6738 | 11.948 |
| Depth→Joint warm start + standard mutation | 10 | 0.6621 | 11.807 |
| Depth→Joint warm start + standard mutation | 15 | 0.6619 | 11.799 |
| Depth→Joint warm start + standard mutation | 20 | 0.6418 | 11.526 |
| Depth→Joint warm start + standard mutation | 25 | 0.6388 | 11.521 |
| Depth→Joint warm start + standard mutation | 30 | 0.6291 | 11.411 |
| Depth→Joint warm start + standard mutation | 35 | 0.6226 | 11.362 |
| Depth→Joint warm start + standard mutation | 40 | 0.6230 | 11.362 |
| Depth→Joint warm start + standard mutation | 45 | 0.6146 | 11.271 |
| Depth→Joint warm start + standard mutation | 50 | 0.6121 | 11.323 |
| Standard initialization + interaction-aware mutation | 0 | 1.2744 | 23.349 |
| Standard initialization + interaction-aware mutation | 5 | 0.8846 | 14.617 |
| Standard initialization + interaction-aware mutation | 10 | 0.8029 | 13.258 |
| Standard initialization + interaction-aware mutation | 15 | 0.7459 | 12.659 |
| Standard initialization + interaction-aware mutation | 20 | 0.7036 | 12.133 |
| Standard initialization + interaction-aware mutation | 25 | 0.6510 | 11.659 |
| Standard initialization + interaction-aware mutation | 30 | 0.6351 | 11.510 |
| Standard initialization + interaction-aware mutation | 35 | 0.6209 | 11.250 |
| Standard initialization + interaction-aware mutation | 40 | 0.6209 | 11.250 |
| Standard initialization + interaction-aware mutation | 45 | 0.6201 | 11.281 |
| Standard initialization + interaction-aware mutation | 50 | 0.6090 | 11.086 |
| Depth→Joint warm start + interaction-aware mutation | 0 | 0.7759 | 11.956 |
| Depth→Joint warm start + interaction-aware mutation | 5 | 0.6366 | 11.469 |
| Depth→Joint warm start + interaction-aware mutation | 10 | 0.6283 | 11.352 |
| Depth→Joint warm start + interaction-aware mutation | 15 | 0.6245 | 11.380 |
| Depth→Joint warm start + interaction-aware mutation | 20 | 0.6247 | 11.380 |
| Depth→Joint warm start + interaction-aware mutation | 25 | 0.6239 | 11.339 |
| Depth→Joint warm start + interaction-aware mutation | 30 | 0.6237 | 11.339 |
| Depth→Joint warm start + interaction-aware mutation | 35 | 0.6144 | 11.216 |
| Depth→Joint warm start + interaction-aware mutation | 40 | 0.6144 | 11.216 |
| Depth→Joint warm start + interaction-aware mutation | 45 | 0.6136 | 11.224 |
| Depth→Joint warm start + interaction-aware mutation | 50 | 0.6118 | 11.172 |

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
Use the delta after 20 completed generations versus the final generation-50 delta and the cost-normalized curves together: generation alone does not account for the 31-candidate initialization difference, while total pipeline cost additionally includes the depth-only search.

## Limitations

- Three seeds support descriptive paired comparisons, not strong significance claims.
- Runtime comes from restarted TU Wien DataLab sessions and is an approximate wall-clock measure.
- Intermediate runtime timestamps are written after each generation while the logged parent fitness and periodic PPL describe the state entering that generation; runtime-normalized intermediate points can therefore carry an offset of up to one generation. The final runtime point uses the finalized run summary.
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
