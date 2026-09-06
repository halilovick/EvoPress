# Sequential Experiment Status

Audit date: 2026-07-30

Repository revision inspected: `4ab6f4b` on `main`

This is a read-only, artifact-first audit. A directory name or Markdown claim was
not accepted as evidence that a run completed. The completion decision used the
saved `command.sh`, `runtime.txt`, finalized `run_summary.json`,
`final_candidate.json`, and complete `generation_log.csv` in each run directory.

## 1. Executive Answer

All four sequential modes are implemented, have focused synthetic CPU/unit
coverage, have a completed seed-0 Mistral-7B G2/O4 DataLab pilot, and have completed
three-seed Mistral-7B G20/O16 DataLab runs:

- `depth_to_quant_frozen`: three full G20 runs, seeds 0–2, plus one G2 pilot;
- `depth_to_joint_warm`: three full G20 standard-mutation runs, three full G50
  standard-mutation runs, three full G50 interaction-aware runs, plus one G2
  standard-mutation pilot;
- `quant_to_depth_frozen`: three full G20 runs, seeds 0–2, plus one G2 pilot;
- `quant_to_joint_warm`: three full G20 standard-mutation runs, seeds 0–2, plus
  one G2 standard-mutation pilot.

Thus the repository contains 22 successful explicit sequential invocations:
18 non-pilot runs and four small DataLab pilots. All 22 use
`mistralai/Mistral-7B-v0.3`, the q-projection reconstruction database, 25%
separate attention/MLP sparsity, an exact active 3-bit budget, and
`group_rule=size`.

The ordinary q-projection controls also exist:

- standard initialization + standard mutation: G20 and G50, seeds 0–2;
- standard initialization + interaction-aware mutation: G50, seeds 0–2;
- small G5/O8 seed-0 DataLab runs for both mutation operators.

The only one of the requested eight conceptual matrix cells with **no saved
DataLab run** is:

> Quantization → Joint warm start + interaction-aware mutation.

That combination is implemented and exercised with a synthetic fake-model unit
test, but no `command.sh` using both
`--sequential_mode quant_to_joint_warm` and
`--joint_mutation_mode interaction_aware` exists under `results/runs` or local
`outputs/experiments`.

Interaction-aware mutation with either frozen mode is rejected by the CLI and is
therefore not a missing experimental cell. The implemented frozen baselines
deliberately permit only the unfrozen component to mutate.

No incomplete or `_retryN` sequential directory was found. The local
`outputs/experiments` directory is empty, so this audit could inspect the
lightweight tracked copies under `results/runs`, but not DataLab-only `run.log`
files or the full local reconstruction database.

## 2. Implementation Status

### 2.1 Mode-by-mode status

| Sequential mode | Implemented behavior | Test evidence | DataLab evidence | Classification |
|---|---|---|---|---|
| `depth_to_quant_frozen` | Imports depth, initializes quantization, then dispatches quantization-only offspring and validates the frozen depth after initialization, mutation, selection, and before saving | Synthetic candidate loading, frozen mutation, exact budget, CLI conflict, launcher forwarding, and summary-metadata tests | Seed-0 G2 pilot; G20/O16 seeds 0–2 | **Full DataLab run completed** |
| `depth_to_joint_warm` | Imports depth exactly for the initial parent; after initialization uses the selected standard or interaction-aware joint operator | Synthetic standard representation test and interaction-aware warm-mutation test; launcher and metadata tests | Seed-0 G2 standard pilot; G20 standard seeds 0–2; G50 standard seeds 0–2; G50 interaction-aware seeds 0–2 | **Full DataLab run completed** |
| `quant_to_depth_frozen` | Imports quantization, generates exact feasible depth masks, and permits only contribution-vector-compatible depth swaps without quantization repair | Exact subset feasibility, legal/illegal swap, no-feasible-mask, no-legal-neighborhood, frozen quantization, CLI, launcher, and metadata tests | Seed-0 G2 pilot; G20/O16 seeds 0–2 | **Full DataLab run completed** |
| `quant_to_joint_warm` | Imports quantization unchanged under strict initialization, selects among exact feasible depth masks, then uses the selected joint operator | Synthetic strict initialization followed by interaction-aware mutation; CLI, launcher, and metadata tests | Seed-0 G2 standard pilot; G20 standard seeds 0–2; no interaction-aware DataLab run | **Full DataLab run completed for standard mutation; CPU/toy smoke-tested only for interaction-aware mutation** |

Implementation evidence:

- CLI choices: `evo_joint_search.py`, lines 818–849.
- Sequential loading and initialization: `evo_joint_search.py`, lines 917–1247.
- Frozen versus ordinary mutation dispatch: `evo_joint_search.py`, lines
  1362–1477.
- Survivor and final invariant checks: `evo_joint_search.py`, lines 1533–1559 and
  1712–1723.
- Candidate adapters and stable hashes: `src/sequential_search.py`, lines 88–95
  and 250–444.
- Exact active-budget arithmetic: `src/sequential_search.py`, lines 473–535.
- Exact feasible depth-state search: `src/sequential_search.py`, lines 614–913.
- Fixed-quant legal swaps and mutation: `src/sequential_search.py`, lines
  916–1051.
- CLI constraints and frozen validation: `src/sequential_search.py`, lines
  1161–1250.
- Wrapper forwarding all four modes: `scripts/run_sequential_search.sh`, lines
  17–35, 77–120.

### 2.2 Meaning of “toy tested”

The test suite uses small synthetic module names, a fake model, temporary
reconstruction files, and two- or four-layer candidates. It does not run an
end-to-end CPU language model. In particular:

- all four mode semantics are exercised in
  `tests/test_sequential_search.py`, lines 365–507;
- exact frozen-quant feasibility is tested in the same file, lines 265–362;
- all four modes are forwarded through the real shell wrapper in
  `tests/test_run_sequential_search.py`, lines 26–61;
- toy summary metadata for all four modes is written and reloaded in
  `tests/test_sequential_search.py`, lines 567–632.

`SEQUENTIAL_SEARCH_IMPLEMENTATION.md`, lines 445–496, records the implementation
verification as 44 focused tests passing and describes the synthetic/fake-model
scope. This audit did not rerun tests because it was read-only.

### 2.3 Git history

| Commit | Evidence |
|---|---|
| `92003c0` | Added all four sequential modes, wrapper, implementation documentation, and focused tests |
| `7cf2505` | Added the four seed-0 Mistral G2 pilot artifact sets |
| `69dad29` | Added the twelve Mistral G20 sequential artifact sets |
| `5412e1d` | Added the four-condition G50 depth-warm workflow and validators |
| `30cb9c8` | Added the first G50 aggregate report and plots |
| `8398d20` | Added the six new G50 depth-warm artifact sets |
| `4ab6f4b` | Corrected convergence time-state reporting and regenerated G50 deliverables |

The `git_commit` stored inside a run summary identifies the source revision used
by that run, not the later commit that added its lightweight artifacts:

- G2 pilots: `92003c0b82e2a02655ee42ae9871753075ce7600`;
- G20 sequential runs: `7cf25059719cc278c99d2c0772a68647d071231c`;
- G50 depth-warm runs: `5412e1daa89dc9786220cd6199c03bc67001480d`.

## 3. Actual Completed Experiments

### 3.1 Validation performed

Every run in Section 3.4 passed all applicable checks:

1. `runtime.txt` contains `exit_code=0`.
2. `run_summary.json` contains both `timestamp_end` and
   `launcher_finalized_at`.
3. `command.sh`, `run_summary.json`, `final_candidate.json`,
   `generation_log.csv`, and `runtime.txt` exist.
4. Generation numbers are exactly `1..generations`, with no missing or extra
   rows.
5. The exact command agrees with the summary mode, mutation operator, seed,
   generation count, offspring count, and active-budget setting.
6. The stage-one component hash was independently recomputed from the locally
   saved stage-one `final_candidate.json` and matches
   `stage1_candidate_hash`.
7. The first generation's `mutation_summary.parent_before_generation_hash`
   matches `initial_parent_hash`.
8. Every final candidate has eight dropped attention components and eight
   dropped MLP components.
9. Every final candidate has 24 active q-projection assignments whose integer
   bit-width sum is exactly 72, i.e. an exact active average of 3.
10. For `depth_to_quant_frozen`, the final depth masks equal the imported
    stage-one masks byte-for-byte after normalization, and every generation
    records zero depth changes and only `sequential_quantization` offspring.
11. For `quant_to_depth_frozen`, the final module-name-to-bit-width map equals
    the imported stage-one map, and every generation records zero quantization
    changes and only `fixed_quant_depth` offspring.
12. For every depth-warm run, the saved initial-parent hash was independently
    reconstructed from the imported depth masks plus one 32-gene uniform
    3-bit q-projection group.

For strict quantization-first warm runs, no standalone initial candidate is
saved. The artifacts prove that the stage-one quantization hash is correct, the
policy was `strict`, the repair changed zero genes, the exact-budget/depth
validators passed, and generation 1 begins from the recorded initial-parent
hash. They do not independently expose the selected initial depth mask and
quantization map as a separate JSON object. Therefore exact initial-component
identity for those warm runs is supported by the runtime validation path and
metadata, but cannot be re-compared from a saved initial-candidate artifact.

### 3.2 Shared configuration

All explicit sequential runs share:

| Field | Value |
|---|---|
| Model | `mistralai/Mistral-7B-v0.3` |
| Quantization database | `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit` |
| Quantization scope | 32 `model.layers.*.self_attn.q_proj` modules; stored levels 2, 3, and 4 |
| Depth mode | Separate attention and MLP masks; not whole-block |
| Depth sparsity | `0.25`, giving 8 attention and 8 MLP drops |
| Target active bit-width | `3.0` |
| Active budget | Enabled |
| Grouping rule | `size`; one equal-size q-projection group |
| Fitness | KL divergence; lower is better |
| Calibration dataset | WikiText2 |
| Quant initialization policy | `strict` |
| Maximum initialization/offspring attempts | 100,000 / 10,000 |

Configuration IDs used below:

| ID | Generations / offspring | Requested initial / actually evaluated | Calibration tokens | Initial tokens | Selection tokens | Survivors | Evaluation |
|---|---:|---:|---:|---:|---|---|---|
| P2-D | 2 / 4 | 8 / 1 | 2,048 | 512 | `[512, 2048]` | `[2, 1]` | WikiText2 every 2; requested 8,192 tokens |
| P2-Q | 2 / 4 | 8 / 8 | 2,048 | 512 | `[512, 2048]` | `[2, 1]` | WikiText2 every 2; requested 8,192 tokens |
| G20-D | 20 / 16 | 32 / 1 | 8,192 | 512 | `[512, 2048, 8192]` | `[8, 2, 1]` | WikiText2 every 5; requested 524,288 tokens |
| G20-Q | 20 / 16 | 32 / 32 | 8,192 | 512 | `[512, 2048, 8192]` | `[8, 2, 1]` | WikiText2 every 5; requested 524,288 tokens |
| G50-D | 50 / 16 | 32 / 1 | 8,192 | 512 | `[512, 2048, 8192]` | `[8, 2, 1]` | WikiText2 every 5; requested 524,288 tokens |

`D` denotes depth-first initialization, which constructs one exact initial
combined candidate. `Q` denotes strict quantization-first initialization, which
generates and evaluates the requested number of exact feasible depth masks.

### 3.3 Stage-one sources

| Seed | Depth-only source and component hash | Quant-only source and component hash |
|---:|---|---|
| 0 | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0` — `454b86987800d97eba43ad3d810527ff7143b7cabdea85e154ab2d26a1831402` | `results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0` — `9c9f5cb40255d4698ecd8bcc569f41e6083b8072cebca156030f6aa8851da70d` |
| 1 | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1` — `6314b9647d6ba62e35046b6b9555495ae4a8b41829eb9b8bd11da3c487396508` | `results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed1` — `614ce6680799eac434377a3133b278b654d29c4e785b1377cee19072d42e205a` |
| 2 | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2` — `d44e607d9348c77c8013375be1fd7e8e43861bac15c7393a9e295affacefc367` | `results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed2` — `bd325265dec9ae8f8887a21a092f9764f1d06190bcbd11a69b939ef1789f37f7` |

The depth source summaries use `search_type=depth_only`; the quantization source
summaries use `search_type=quant_only`, `group_rule=size`, target bit-width 3.0,
and the same 32-module q-projection scope.

### 3.4 Verified sequential runs

`Checks=PASS` means all applicable checks in Section 3.1 passed.

| Configuration | Run directory | Mode / mutation | Seed | Stage one | Final PPL | Best fitness / final KL | Runtime s | Checks | Source Git |
|---|---|---|---:|---|---:|---:|---:|---|---|
| P2-D | `results/runs/pilot_seq_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g2_o4_seed0` | Depth→Quant frozen / standard | 0 | depth-only seed 0 | 11.9921875 | 0.708984375 / 0.708984375 | 159 | PASS; frozen depth exact | `92003c0` |
| P2-D | `results/runs/pilot_seq_depth_to_joint_warm_mistral_s0.25_qproj3.0_g2_o4_seed0` | Depth→Joint warm / standard | 0 | depth-only seed 0 | 11.96875 | 0.7080078125 / 0.7080078125 | 160 | PASS; initial depth hash reconstructed | `92003c0` |
| P2-Q | `results/runs/pilot_seq_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g2_o4_seed0` | Quant→Depth frozen / standard | 0 | quant-only seed 0 | 109.4375 | 2.99609375 / 2.99609375 | 148 | PASS; frozen quant exact; legal swaps 326→326 | `92003c0` |
| P2-Q | `results/runs/pilot_seq_quant_to_joint_warm_mistral_s0.25_qproj3.0_g2_o4_seed0` | Quant→Joint warm / standard | 0 | quant-only seed 0 | 132.5 | 3.21484375 / 3.21484375 | 145 | PASS; strict import metadata | `92003c0` |
| G20-D | `results/runs/thesis_sequential_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g20_o16_seed0` | Depth→Quant frozen / standard | 0 | depth-only seed 0 | 11.921875 | 0.697265625 / 0.69775390625 | 542 | PASS; frozen depth exact | `7cf2505` |
| G20-D | `results/runs/thesis_sequential_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g20_o16_seed1` | Depth→Quant frozen / standard | 1 | depth-only seed 1 | 12.5234375 | 0.703125 / 0.703125 | 554 | PASS; frozen depth exact | `7cf2505` |
| G20-D | `results/runs/thesis_sequential_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g20_o16_seed2` | Depth→Quant frozen / standard | 2 | depth-only seed 2 | 11.3515625 | 0.61669921875 / 0.61669921875 | 564 | PASS; frozen depth exact | `7cf2505` |
| G20-D | `results/runs/thesis_sequential_depth_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed0` | Depth→Joint warm / standard | 0 | depth-only seed 0 | 11.5078125 | 0.671875 / 0.671875 | 546 | PASS; initial depth hash reconstructed | `7cf2505` |
| G20-D | `results/runs/thesis_sequential_depth_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed1` | Depth→Joint warm / standard | 1 | depth-only seed 1 | 11.7578125 | 0.64599609375 / 0.646484375 | 529 | PASS; initial depth hash reconstructed | `7cf2505` |
| G20-D | `results/runs/thesis_sequential_depth_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed2` | Depth→Joint warm / standard | 2 | depth-only seed 2 | 11.3125 | 0.607421875 / 0.60791015625 | 542 | PASS; initial depth hash reconstructed | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g20_o16_seed0` | Quant→Depth frozen / standard | 0 | quant-only seed 0 | 11.03125 | 0.6171875 / 0.61767578125 | 516 | PASS; frozen quant exact; legal swaps 352→352 | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g20_o16_seed1` | Quant→Depth frozen / standard | 1 | quant-only seed 1 | 16.453125 | 1.0126953125 / 1.013671875 | 519 | PASS; frozen quant exact; legal swaps 326→326 | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g20_o16_seed2` | Quant→Depth frozen / standard | 2 | quant-only seed 2 | 14.5234375 | 0.90869140625 / 0.908203125 | 520 | PASS; frozen quant exact; legal swaps 326→326 | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed0` | Quant→Joint warm / standard | 0 | quant-only seed 0 | 13.046875 | 0.7763671875 / 0.77587890625 | 546 | PASS; strict import metadata | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed1` | Quant→Joint warm / standard | 1 | quant-only seed 1 | 12.1796875 | 0.701171875 / 0.701171875 | 535 | PASS; strict import metadata | `7cf2505` |
| G20-Q | `results/runs/thesis_sequential_quant_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed2` | Quant→Joint warm / standard | 2 | quant-only seed 2 | 12.4921875 | 0.75048828125 / 0.75048828125 | 540 | PASS; strict import metadata | `7cf2505` |
| G50-D | `results/runs/thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed0` | Depth→Joint warm / standard | 0 | depth-only seed 0 | 10.9453125 | 0.60986328125 / 0.60986328125 | 1192 | PASS; initial depth hash reconstructed | `5412e1d` |
| G50-D | `results/runs/thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed1` | Depth→Joint warm / standard | 1 | depth-only seed 1 | 11.671875 | 0.62109375 / 0.62109375 | 1190 | PASS; initial depth hash reconstructed | `5412e1d` |
| G50-D | `results/runs/thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed2` | Depth→Joint warm / standard | 2 | depth-only seed 2 | 11.3515625 | 0.60546875 / 0.60498046875 | 1200 | PASS; initial depth hash reconstructed | `5412e1d` |
| G50-D | `results/runs/thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed0` | Depth→Joint warm / interaction-aware | 0 | depth-only seed 0 | 10.984375 | 0.626953125 / 0.626953125 | 1244 | PASS; initial depth hash reconstructed | `5412e1d` |
| G50-D | `results/runs/thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed1` | Depth→Joint warm / interaction-aware | 1 | depth-only seed 1 | 11.375 | 0.6162109375 / 0.6162109375 | 1233 | PASS; initial depth hash reconstructed | `5412e1d` |
| G50-D | `results/runs/thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed2` | Depth→Joint warm / interaction-aware | 2 | depth-only seed 2 | 11.15625 | 0.59228515625 / 0.591796875 | 1247 | PASS; initial depth hash reconstructed | `5412e1d` |

### 3.5 Verified ordinary joint controls

These are not sequential runs, but they occupy the first two rows of the
requested matrix and are the controls used by the sequential reports.

| Run directory | Mutation | Seed | G / O / initial | Selection tokens / survivors | Final PPL | Best fitness / final KL | Runtime s | Status |
|---|---|---:|---|---|---:|---:|---:|---|
| `results/runs/debug_standard_mistral_qproj_s025_g5_o8_seed0` | standard | 0 | 5 / 8 / 16 | `[128,512,1024]` / `[4,2,1]` | 18.078125 | 0.96533203125 / 0.96533203125 | 383 | Completed small DataLab run |
| `results/runs/debug_interaction_aware_mistral_qproj_s025_g5_o8_seed0` | interaction-aware | 0 | 5 / 8 / 16 | `[128,512,1024]` / `[4,2,1]` | 18.796875 | 0.841796875 / 0.841796875 | 382 | Completed small DataLab run |
| `results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed0` | standard | 0 | 20 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 13.3828125 | 0.7939453125 / 0.7939453125 | 609 | Full DataLab run completed |
| `results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed1` | standard | 1 | 20 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 12.28125 | 0.69921875 / 0.69921875 | 611 | Full DataLab run completed |
| `results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed2` | standard | 2 | 20 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 12.15625 | 0.71240234375 / 0.712890625 | 607 | Full DataLab run completed |
| `results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0` | standard | 0 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 11.46875 | 0.66455078125 / 0.6640625 | 1348 | Full DataLab run completed |
| `results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed1` | standard | 1 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 10.921875 | 0.55908203125 / 0.55908203125 | 1361 | Full DataLab run completed |
| `results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed2` | standard | 2 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 11.3359375 | 0.61474609375 / 0.615234375 | 1356 | Full DataLab run completed |
| `results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed0` | interaction-aware | 0 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 10.8984375 | 0.61083984375 / 0.6103515625 | 1294 | Full DataLab run completed |
| `results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed1` | interaction-aware | 1 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 11.1171875 | 0.59130859375 / 0.59130859375 | 1265 | Full DataLab run completed |
| `results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed2` | interaction-aware | 2 | 50 / 16 / 32 | `[512,2048,8192]` / `[8,2,1]` | 11.2421875 | 0.625 / 0.625 | 1255 | Full DataLab run completed |

The baseline summaries predate the sequential final-validation fields and do not
store `active_budget_valid` or `depth_counts_valid`. This audit independently
read their final candidates; all listed controls have 8/8 depth counts and the
same exact 24-active-module, 72-bit q-projection sum.

## 4. Incomplete or Interrupted Runs

No incomplete sequential run was found locally:

- all 22 directories containing a `command.sh` with `--sequential_mode` have
  the five required completion artifacts;
- all 22 runtime files contain `exit_code=0`;
- all 22 generation logs reach their configured generation;
- no sequential, `depthwarm`, `quantwarm`, `d2q`, or `q2d` `_retryN`
  directory exists in the working tree or current Git tree;
- no incomplete base directory shadows a successful retry.

The G50 launcher contains generic retry discovery and preservation logic in
`scripts/run_depth_warmstart_g50_grid.sh`, lines 140–178, but it did not need to
leave a tracked retry directory for the completed G50 matrix.

Result directories that could not be inspected:

- `outputs/experiments` exists locally but has no child run directories;
- the full DataLab output directories referenced by `run_summary.json` are not
  present in this checkout;
- consequently, per-run `run.log` files and any DataLab-only files not included
  in the ten-file lightweight sync could not be inspected;
- the q-projection `.pth` reconstruction database referenced by the commands is
  not present locally under `outputs/experiments`.

These absences do not invalidate the completion decision because the saved
lightweight copies contain the exact command, exit code, finalized summary,
final candidate, generation log, configuration files, and runtime. They do
limit independent inspection of stdout and reconstruction file contents.

## 5. Reports and What They Cover

| Report | Type | Actual scope | Audit assessment |
|---|---|---|---|
| `SEQUENTIAL_SEARCH_IMPLEMENTATION.md` | Implementation documentation and test handoff | Defines all four modes, candidate loading, exact budget behavior, example commands, metadata, tests, and limitations | It is not an executed-results report. Its “all four” wording refers to implemented/tested modes, not four completed DataLab experiment families. Lines 485–496 explicitly describe synthetic checks and test results. |
| `DEPTH_WARMSTART_G50_EXPERIMENT.md` | Experiment plan, launcher documentation, and reproducibility handoff | Only standard versus depth-warm initialization under standard and interaction-aware mutation at G50 | It never claims to cover all four sequential directions. Lines 294–296 correctly exclude frozen and quantization-first modes. Lines 470–472 still say the final report cannot be produced until six runs complete; that sentence is now stale because the runs and report were later committed. |
| `results/depth_warmstart_g50_comparison.md` | Executed-results report | Four G50 conditions: standard/standard, depth-warm/standard, standard/interaction-aware, depth-warm/interaction-aware; seeds 0–2 | Artifact-consistent. It is intentionally a depth-warm report, not a complete four-mode sequential report. Its validation claims at lines 109–116 agree with the saved artifacts. |
| `results/sequential_search_comparison.md` | Executed-results report generated from structured artifacts | All four sequential modes with standard mutation at G20, seeds 0–2, plus composition and ordinary joint references | Its 12 sequential rows agree with the saved run summaries, stage-one hashes, and invariants. It correctly states at line 9 that all 12 sequential summaries were verified. The report, its CSVs/plot, and `scripts/summarize_sequential_search.py` are currently untracked local files, so they are valid local evidence but not yet part of `HEAD`. |

The existing reports are complementary:

- the broad sequential report answers direction/frozen-versus-warm questions at
  G20 under standard mutation;
- the G50 report asks whether **depth** warm-starting persists and interacts
  with interaction-aware mutation.

Neither report contains a quantization-warm + interaction-aware DataLab result.

## 6. Full Sequential Matrix

“Toy tested” means the small fake-model/synthetic unit coverage described in
Section 2.2, not a saved end-to-end CPU LLM run. Small GPU pilots are shown
separately from full G20/G50 runs.

| Experiment | Implemented | Unit tested | Toy tested | Full runs | Seeds | Configuration | Evidence | Status |
|---|---|---|---|---:|---|---|---|---|
| Standard joint initialization + standard mutation | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G5/O8 pilot | `results/runs/debug_standard_mistral_qproj_s025_g5_o8_seed0` | Full DataLab run completed (small pilot configuration) |
| Standard joint initialization + standard mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G20/O16 | `results/runs/thesis_medium_joint_mistral_s0.25_qproj3.0_g20_o16_seed*` | Full DataLab runs completed |
| Standard joint initialization + standard mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G50/O16 | `results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed*` | Full DataLab runs completed |
| Standard joint initialization + interaction-aware mutation | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G5/O8 pilot | `results/runs/debug_interaction_aware_mistral_qproj_s025_g5_o8_seed0` | Full DataLab run completed (small pilot configuration) |
| Standard joint initialization + interaction-aware mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G50/O16 | `results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed*` | Full DataLab runs completed |
| Depth → Quantization frozen | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G2/O4 pilot | `results/runs/pilot_seq_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g2_o4_seed0` | Full DataLab run completed (small pilot configuration) |
| Depth → Quantization frozen | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G20/O16, standard dispatch | `results/runs/thesis_sequential_depth_to_quant_frozen_mistral_s0.25_qproj3.0_g20_o16_seed*` | Full DataLab runs completed |
| Depth → Joint warm + standard mutation | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G2/O4 pilot | `results/runs/pilot_seq_depth_to_joint_warm_mistral_s0.25_qproj3.0_g2_o4_seed0` | Full DataLab run completed (small pilot configuration) |
| Depth → Joint warm + standard mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G20/O16 | `results/runs/thesis_sequential_depth_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed*` | Full DataLab runs completed |
| Depth → Joint warm + standard mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G50/O16 | `results/runs/thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed*` | Full DataLab runs completed |
| Depth → Joint warm + interaction-aware mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G50/O16 | `results/runs/thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed*` | Full DataLab runs completed |
| Quantization → Depth frozen | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G2/O4 pilot | `results/runs/pilot_seq_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g2_o4_seed0` | Full DataLab run completed (small pilot configuration) |
| Quantization → Depth frozen | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G20/O16, standard dispatch | `results/runs/thesis_sequential_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g20_o16_seed*` | Full DataLab runs completed |
| Quantization → Joint warm + standard mutation | Yes | Yes | Yes | 0 | 0 | Mistral q_proj G2/O4 pilot | `results/runs/pilot_seq_quant_to_joint_warm_mistral_s0.25_qproj3.0_g2_o4_seed0` | Full DataLab run completed (small pilot configuration) |
| Quantization → Joint warm + standard mutation | Yes | Yes | Yes | 3 | 0–2 | Mistral q_proj G20/O16 | `results/runs/thesis_sequential_quant_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed*` | Full DataLab runs completed |
| Quantization → Joint warm + interaction-aware mutation | Yes | Yes | Yes | 0 | None | No command/artifact found | Synthetic interaction-aware warm test at `tests/test_sequential_search.py`, lines 473–507; generic launcher supports the combination | **CPU/toy smoke-tested; no DataLab evidence found** |

Frozen modes do not have an interaction-aware sub-row because
`validate_sequential_cli` rejects it
(`src/sequential_search.py`, lines 1193–1201), and the shell launcher repeats
that restriction (`scripts/run_joint_search_tiny.sh`, lines 258–264).

## 7. Missing Experiments

### 7.1 Missing conceptual matrix cell

Exactly one of the eight requested combinations remains unexecuted:

- `quant_to_joint_warm` + `joint_mutation_mode=interaction_aware`, for every
  seed and every practical generation count.

There is no direct command, environment-variable alias, renamed directory, or
retry artifact for that combination. Searches covered the full mode names,
`depthwarm`, `quantwarm`, `d2q`, `q2d`, `seq`, direct
`evo_joint_search.py` invocations, environment forwarding, and `_retryN`.

### 7.2 Configuration-level gaps

The completed experiment families do not form one uniform generation-count grid:

- standard interaction-aware and depth-warm interaction-aware have complete G50
  evidence but no matched G20 three-seed runs;
- the four original sequential variants have complete G20 evidence;
- only depth-warm standard and depth-warm interaction-aware were extended to
  G50;
- no frozen or quantization-first G50 extension was planned by the G50
  depth-warm experiment.

These are configuration gaps, not evidence that the existing runs are
incomplete. The immediate missing conceptual cell can be tested at G20 and
paired directly against the existing quantization-warm + standard G20 runs.

## 8. Recommended Next Runs

Only one new experiment cell is required to fill the conceptual matrix:
quantization → joint warm + interaction-aware mutation. The command below uses
the existing generic sequential launcher, the three compatible quant-only
stage-one runs already used by the standard quantization-warm comparison, and
the exact G20 configuration of those standard runs. The only intended treatment
change is `JOINT_MUTATION_MODE=interaction_aware`.

Run this on DataLab from `/home/jovyan/evopress`, not locally:

```bash
cd /home/jovyan/evopress

for SEED in 0 1 2; do
  STAGE1_RUN="results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed${SEED}"
  RUN_ID="thesis_sequential_quant_to_joint_warm_interactionaware_mistral_s0.25_qproj3.0_g20_o16_seed${SEED}"
  OUT="outputs/experiments/${RUN_ID}"

  env \
    MODEL="mistralai/Mistral-7B-v0.3" \
    QUANT_WEIGHTS_PATH="outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit" \
    DROP_SPARSITY="0.25" \
    TARGET_BITWIDTH="3.0" \
    CALIB_DATA="wikitext2" \
    CALIB_TOKENS="8192" \
    SEQUENCE_LENGTH="1024" \
    EVAL_TOKENS="524288" \
    EVAL_DATASETS="wikitext2" \
    EVAL_EVERY="5" \
    GENERATIONS="20" \
    OFFSPRING="16" \
    INITIALLY_GENERATED="32" \
    INITIAL_TOKENS="512" \
    TOKENS_PER_SELECTION="512 2048 8192" \
    SURVIVORS_PER_SELECTION="8 2 1" \
    FITNESS_FN="kl" \
    GROUP_RULE="size" \
    ACTIVE_QUANT_BUDGET="1" \
    JOINT_MUTATION_MODE="interaction_aware" \
    JOINT_AWARE_MUTATION="0" \
    ADAPTIVE_MUTATION="0" \
    COARSE_TO_FINE_MUTATION="0" \
    MAX_DROP_MUTATIONS="3" \
    STEP_SIZE="1" \
    ATTN_IMPLEMENTATION="sdpa" \
    DTYPE="float16" \
    USE_FAST_TOKENIZER="1" \
    SEED="$SEED" \
    RUN_ID="$RUN_ID" \
    OUTPUT_DIR="$OUT" \
    MAX_INITIALIZATION_ATTEMPTS="100000" \
    MAX_OFFSPRING_ATTEMPTS="10000" \
    scripts/run_sequential_search.sh \
      --mode quant_to_joint_warm \
      --stage1-run-dir "$STAGE1_RUN" \
      --output-dir "$OUT" \
      --policy strict
done
```

Command provenance:

- environment names and Python argument construction:
  `scripts/run_joint_search_tiny.sh`, lines 8–57 and 118–190;
- active-budget, interaction-aware, and sequential CLI validation:
  `scripts/run_joint_search_tiny.sh`, lines 202–276;
- wrapper options and forwarding:
  `scripts/run_sequential_search.sh`, lines 17–35 and 77–120;
- the G20 values come from the existing
  `results/runs/thesis_sequential_quant_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed*/command.sh`
  files.

Prerequisites:

- the local checkout cannot verify the database because
  `outputs/experiments` is empty; DataLab must still contain 32 q-projection
  module directories with 2/3/4-bit reconstruction files;
- the three required quant-only stage-one lightweight directories and
  candidates are present and have the correct model, grouping, target, and
  module scope;
- the proposed output names do not currently exist in this checkout, and the
  launcher will refuse to overwrite a non-empty DataLab output directory;
- `scripts/summarize_sequential_search.py` currently expects only the original
  four standard-mutation G20 sequential rows, so a new interaction-aware
  quant-warm comparison would require a reporting extension after the runs.

No frozen-mode interaction-aware command is recommended because it is an
invalid scientific baseline and is rejected before model loading.

## 9. Evidence Appendix

### 9.1 Artifact fields used

For each explicit sequential run, the audit read:

- `command.sh`: exact Python invocation, model, database, budget, mode,
  mutation operator, seed, schedules, stage-one source, and output path;
- `runtime.txt`: `runtime_seconds`, `runtime_minutes`, and `exit_code`;
- `run_summary.json`:
  - top level: `sequential_mode`, `sequential_direction`,
    `sequential_variant`, `frozen_component`, `stage1_run_dir`,
    `stage1_candidate_path`, `stage1_candidate_hash`,
    `stage1_search_type`, `initial_parent_hash`,
    `initial_feasible_candidate_count`,
    `initial_component_changed_by_repair`,
    `initial_repair_changed_gene_count`,
    `fixed_quant_legal_swap_count`, `frozen_depth_unchanged`,
    `frozen_quant_unchanged`, `active_budget_valid`,
    `depth_counts_valid`, `git_commit`, `timestamp_end`, and
    `launcher_finalized_at`;
  - `search_config`: `seed`, `generations`, `offspring`,
    `initial_candidates`, `initial_candidates_evaluated`,
    `initial_tokens`, `selection_tokens`, and `selection_survivors`;
  - `compression_config`: `quant_weights_path`,
    `target_depth_sparsity`, `target_average_bitwidth`,
    `active_quant_budget`, `group_rule`, `joint_mutation_mode`, and
    `drop_entire_block`;
  - `final_metrics`: `wikitext2_ppl`, `best_search_fitness`,
    `final_calibration_kl`, and `runtime_seconds`;
- `generation_log.csv`: generation completeness, first-parent hash, recorded
  sequential mode, generated mutation categories, and depth/quantization
  change counts;
- `final_candidate.json`: final attention/MLP masks and the exact
  module-name-to-bit-width map.

Representative artifact:

`results/runs/thesis_sequential_quant_to_depth_frozen_mistral_s0.25_qproj3.0_g20_o16_seed0/run_summary.json`
contains:

- `sequential_mode="quant_to_depth_frozen"`;
- `stage1_search_type="quant_only"`;
- `search_config.generations=20`;
- `search_config.offspring=16`;
- `search_config.initial_candidates_evaluated=32`;
- `compression_config.active_quant_budget=true`;
- `compression_config.group_rule="size"`;
- `frozen_quant_unchanged=true`;
- `active_budget_valid=true`;
- `depth_counts_valid=true`;
- `fixed_quant_legal_swap_count.initial_parent=352`;
- `fixed_quant_legal_swap_count.final_parent=352`.

### 9.2 Independent invariant reconstruction

The stage-one component hash format is defined by
`stable_json_hash` in `src/sequential_search.py`, lines 88–95:
sorted-key, compact JSON followed by SHA-256.

For depth-first runs, the recomputed object was:

```text
{
    "attn": normalized stage-one attention_mask booleans,
    "mlp": normalized stage-one mlp_mask booleans
}
```

For quantization-first runs, it was the sorted module-name-to-integer-bit-width
mapping. All 22 stored hashes match.

The active q-projection budget was independently reconstructed from each final
candidate:

```text
active q_proj modules = modules whose layer's attention mask is not dropped
active module count   = 24
active bit sum        = 72
target bit sum        = 24 × 3 = 72
```

This agrees with the exact integer/Fraction validation implemented in
`src/sequential_search.py`, lines 473–535.

### 9.3 Aliases, direct invocations, and retries

The repository-wide search included:

- all `command.sh` files containing `--sequential_mode`;
- exact mode strings;
- `seq`, `sequential`, `depthwarm`, `quantwarm`, `d2q`, and `q2d`;
- `SEQUENTIAL_MODE` and `JOINT_MUTATION_MODE` forwarding;
- direct `evo_joint_search.py` calls;
- `_retry1` through generic `_retryN` naming.

Exactly 22 commands contain an explicit sequential mode. There are no sequential
retry directories. The six G50 depth-warm commands use abbreviated
`depthwarm` directory names but correctly contain
`--sequential_mode depth_to_joint_warm`.

### 9.4 Aggregate-report cross-check

The local untracked aggregate files contain:

- 27 run rows in `results/sequential_search_runs.csv`;
- 9 method rows in `results/sequential_search_summary.csv`;
- 60 paired rows in `results/sequential_search_paired_deltas.csv`.

All 12 sequential rows in `sequential_search_runs.csv` match their named
`run_summary.json` for run ID and final WikiText2 PPL, and all record true
active-budget and depth-count validation. This supports the report, but the raw
run artifacts remain the source of truth.

### 9.5 Audit limitations

- Full `outputs/experiments/**` directories and `run.log` files were not
  available locally.
- No separate initial-candidate JSON is saved for quantization-first warm runs;
  the exact imported initial quantization component is therefore supported by
  the code's strict runtime checks, stage-one hash, zero-repair metadata, and
  first-parent hash rather than a second saved candidate object.
- Test execution was not repeated during this read-only audit. Unit-test status
  is based on test source plus the recorded verification in
  `SEQUENTIAL_SEARCH_IMPLEMENTATION.md`.
- No GPU experiment was launched.
