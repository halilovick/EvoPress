# Component Crossover Experiment Comparison and Handoff

Audit date: 2026-08-10

Experiment date: 2026-08-02

Implementation commit: `cdeeedd` (`Add component crossover to joint search`)

Result-artifact commit: `4adf5b1` (`Add component crossover experiment results`)

This handoff is artifact-first. Reported values were read from the committed
`run_summary.json`, `generation_log.csv`, `generation_metrics.csv`, and
`runtime.txt` files. Lower WikiText2 perplexity (PPL) and lower KL divergence are
better.

## 1. Executive conclusion

The minimal component-crossover extension is implemented, tested, and feasible:

- every run completed with exit code 0;
- the persistent population stayed at four unique candidates in every generation;
- all final candidates retained exactly eight attention drops and eight MLP drops;
- the active q-projection quantization average remained exactly 3.0 bits;
- no crossover proposal was rejected as infeasible;
- sequential initialization was not active.

The G20/O16 comparison does **not** show a consistent improvement over the matched
legacy mutation-only search. Crossover improved both final PPL and KL for seed 0,
degraded both for seed 1, and produced a small PPL improvement but a small KL
degradation for seed 2. Across three seeds, its mean PPL and KL were worse and its
mean runtime was about 14% higher.

The strongest operational finding is the 79.1% crossover duplicate rate. The
operator was feasible, but most recombinations reproduced a candidate already in
the population or current offspring pool.

These results support the thesis-safe conclusion that component crossover was
successfully integrated and tested, but this three-seed pilot provides no evidence
that it improves this search setting.

## 2. Implemented extension

For parents `A = (depth_A, quant_A)` and `B = (depth_B, quant_B)`, the operator
deep-copies one of:

- `(depth_A, quant_B)`; or
- `(depth_B, quant_A)`.

The selected depth component is preserved exactly. If active quantization budgeting
is enabled, the imported quantization component is repaired against that depth mask.
A child changed by repair is classified as `component crossover + repair`.

The extension adds:

- `--population_size` (legacy default: `1`);
- `--crossover_probability` (legacy default: `0.0`);
- `--crossover_type component`;
- uniform parent selection;
- distinct crossover parents;
- final-stage elitism over all persistent parents;
- population, crossover, duplicate, repair, and feasibility diagnostics.

With population size four, a configured survivor schedule `[8, 2, 1]` has the
effective schedule `[8, 2, 4]`: intermediate stages remain unchanged, while the
final stage retains exactly four unique candidates. The legacy population-one path
preserves `[8, 2, 1]` and does not consume additional parent-selection randomness.

Sequential modes reject the population/crossover extension in this implementation.
See [CROSSOVER_IMPLEMENTATION.md](CROSSOVER_IMPLEMENTATION.md) for the concise
implementation description and test commands.

## 3. Verification evidence

Local verification completed before the DataLab runs:

- 25 focused crossover and launcher tests passed;
- 32 existing standard, interaction-aware, and sequential tests passed;
- Python compilation passed;
- Ruff passed;
- shell syntax validation passed;
- `git diff --check` passed.

Each published run contains the standard nine lightweight artifacts:

- `command.sh`;
- `final_candidate.json`;
- `generation_log.csv`;
- `generation_metrics.csv`;
- `joint_config.json`;
- `joint_drop_config.txt`;
- `joint_quant_config.txt`;
- `run_summary.json`;
- `runtime.txt`.

The full G20 runs used source revision
`cdeeedddc095fa001522b71e167d84eb7344e423`.

## 4. Shared experiment configuration

| Field | Pilot | Full comparison |
|---|---:|---:|
| Model | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-v0.3` |
| Quantization scope | q_proj | q_proj |
| Depth sparsity | 0.25 | 0.25 |
| Attention / MLP drops | 8 / 8 | 8 / 8 |
| Target active bit-width | 3.0 | 3.0 |
| Active quantization budget | enabled | enabled |
| Group rule | `size` | `size` |
| Fitness | KL | KL |
| Mutation operator | standard | standard |
| Generations / offspring | 5 / 8 | 20 / 16 |
| Initial candidates | 16 | 32 |
| Configured survivors | `[4, 2, 1]` | `[8, 2, 1]` |
| Effective crossover survivors | `[4, 2, 4]` | `[8, 2, 4]` |
| Population size | 4 | 4 |
| Crossover probability | 0.25 | 0.25 |
| Crossover type | `component` | `component` |
| Seeds | 0 | 0, 1, 2 |

The matched controls use the same model, compression targets, datasets, fitness,
mutation operator, generation count, offspring count, initial-candidate count, and
token schedule. They retain the legacy single parent and have crossover disabled.

## 5. Small G5/O8 pilot

| Metric | Component crossover | Mutation-only control | Crossover minus control |
|---|---:|---:|---:|
| Final WikiText2 PPL | 19.390625 | 18.078125 | +1.312500 |
| Final calibration KL | 1.003906 | 0.965332 | +0.038574 |
| Runtime (seconds) | 431 | 383 | +48 |

Pilot crossover diagnostics:

- attempted: 8;
- accepted: 3 (37.5%);
- duplicates: 5 (62.5%);
- infeasible: 0;
- accepted with repair: 1 of 3 (33.3%);
- repair-changed quantization genes: 1;
- final and per-generation population diversity: 4/4.

Artifacts:

- [crossover pilot](results/runs/pilot_component_crossover_mistral_qproj_s025_g5_o8_p4_c025_seed0/run_summary.json)
- [mutation-only pilot control](results/runs/debug_standard_mistral_qproj_s025_g5_o8_seed0/run_summary.json)

The pilot confirmed end-to-end operation but did not outperform its matched control.

## 6. G20/O16 three-seed comparison

### 6.1 Per-seed crossover behavior

| Seed | Attempted | Accepted | Acceptance | Duplicates | Duplicate rate | Infeasible | Accepted with repair | Repair frequency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 107 | 22 | 20.6% | 85 | 79.4% | 0 | 10 | 45.5% |
| 1 | 93 | 25 | 26.9% | 68 | 73.1% | 0 | 13 | 52.0% |
| 2 | 101 | 16 | 15.8% | 85 | 84.2% | 0 | 2 | 12.5% |
| **Total** | **301** | **63** | **20.9%** | **238** | **79.1%** | **0** | **25** | **39.7%** |

Repair changed 25 quantization genes in total. In these runs, each accepted repaired
child changed one quantization gene. Every generation retained four unique persistent
candidates.

### 6.2 Per-seed quality and runtime

| Seed | Crossover PPL | Control PPL | PPL delta | Crossover KL | Control KL | KL delta | Crossover runtime | Control runtime | Runtime delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.664062 | 13.382812 | -0.718750 | 0.772461 | 0.793945 | -0.021484 | 753 s | 609 s | +144 s |
| 1 | 13.882812 | 12.281250 | +1.601562 | 0.842773 | 0.699219 | +0.143555 | 659 s | 611 s | +48 s |
| 2 | 12.062500 | 12.156250 | -0.093750 | 0.722168 | 0.712891 | +0.009277 | 671 s | 607 s | +64 s |

Negative quality deltas favor crossover. The seed-2 PPL improvement is small and is
not accompanied by a KL improvement.

### 6.3 Aggregate comparison

Values are mean ± sample standard deviation across seeds 0–2.

| Metric | Component crossover | Mutation-only control | Mean paired delta |
|---|---:|---:|---:|
| Final WikiText2 PPL | 12.869792 ± 0.927431 | 12.606771 ± 0.674972 | +0.263021 |
| Final calibration KL | 0.779134 ± 0.060579 | 0.735352 ± 0.051202 | +0.043783 |
| Runtime (seconds) | 694.33 ± 51.16 | 609.00 ± 2.00 | +85.33 |

The runtime increase is approximately 14.0% relative to the control mean.

Artifacts:

| Seed | Component crossover | Mutation-only control |
|---:|---|---|
| 0 | [summary](results/runs/thesis_component_crossover_mistral_s0.25_qproj3.0_g20_o16_p4_c025_seed0/run_summary.json) | [summary](results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed0/run_summary.json) |
| 1 | [summary](results/runs/thesis_component_crossover_mistral_s0.25_qproj3.0_g20_o16_p4_c025_seed1/run_summary.json) | [summary](results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed1/run_summary.json) |
| 2 | [summary](results/runs/thesis_component_crossover_mistral_s0.25_qproj3.0_g20_o16_p4_c025_seed2/run_summary.json) | [summary](results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed2/run_summary.json) |

## 7. Interpretation

### What worked

- Component recombination, deep copying, repair, validation, uniqueness checks, and
  population elitism worked end to end.
- Active-budget repair resolved every feasible recombination that required it.
- No crossover proposal was infeasible because of depth counts, the active budget,
  or missing reconstruction files.
- Duplicate rejection preserved a fully unique population in every generation.
- The disabled default remains the legacy single-parent search path.

### What did not improve

- Final PPL and KL varied substantially by seed.
- The three-seed means were worse than the matched legacy controls.
- Runtime increased because a persistent population and final-stage elitism require
  more candidate evaluations.
- Most crossover attempts were duplicates, limiting the number of genuinely new
  recombinations entering selection.

### Important causal limitation

This comparison changes both persistent population size (`1` to `4`) and crossover
probability (`0.0` to `0.25`). It therefore evaluates the complete crossover
extension against the legacy algorithm; it does not isolate the causal contribution
of crossover from the population change. A population-four, crossover-zero control
would be required for that narrower question, but it was outside this minimal pilot.

## 8. Scope and limitations

The evidence is limited to:

- one model (`Mistral-7B-v0.3`);
- q-projection quantization only;
- 25% attention and MLP depth sparsity;
- an active 3-bit target;
- population size four;
- crossover probability 0.25;
- standard mutation;
- three full-run seeds;
- WikiText2 PPL and calibration KL.

No sequential mode, interaction-aware mutation combination, crossover probability
sweep, population-size study, tournament selection, gene-level crossover, adaptive
crossover, or diversity-preservation algorithm was evaluated.

## 9. Recommended thesis wording

> A minimal component-crossover extension was evaluated by retaining four persistent
> candidates and recombining the depth mask of one parent with the quantization
> assignment of another. Across three matched seeds, the method maintained all depth
> and active quantization constraints and preserved a unique population, but 79.1% of
> crossover attempts were duplicates. Mean PPL and KL were slightly worse than the
> legacy mutation-only control, with mixed per-seed outcomes. These results do not
> establish a benefit from crossover in this setting, although they demonstrate that
> the mechanism can be integrated without violating compression constraints.

## 10. Handoff status

- Implementation and tests: complete.
- Seed-0 G5/O8 pilot: complete and validated.
- G20/O16 seeds 0–2: complete and validated.
- Canonical lightweight artifacts: committed under `results/runs`.
- Raw DataLab `run.log` and `memory_samples.csv`: intentionally not committed.
- Central `results/experiment_log.csv` rows: DataLab-local and not part of the
  published artifact commit.
- Scientific conclusion: no improvement claim; high duplicate rate is the primary
  finding.
