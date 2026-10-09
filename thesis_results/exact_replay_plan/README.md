# Exact-budget replays: RQ1 baselines and RQ3 interaction

Status: **run 2026-10-08/09, 97/97 completed.** Results:
`results/exact_replays/replay_20261008/` (analysis and findings in its `analysis/`
folder). This directory holds the plan; `pair_analysis.*` contains only exact
bit-width arithmetic on completed final candidates (no model evaluation).
Measured cost: mean 359 s per evaluation, about 10 hours in total (the estimate below
was too low). The replays ran in a rebuilt environment with the September package
versions and reproduced all 8 re-evaluated finals exactly.

## Why

* **RQ1.** In the full-space protocol, joint search was compared at the exact
  budget only with uniform 3-bit quantization (E1) and quantization-only search
  (E2). Heuristic structural baselines and independently optimized compositions
  exist only in the early depth-only and screening protocols.
* **RQ3.** The interaction contrast was measured only in the screening protocol
  (active-average budget, q_proj scope). Crossed candidates need a budget repair,
  which confounds the contrast.

## What the plan contains (`replay_plan.md`, 97 jobs)

| Tier | Jobs | Content |
| --- | ---: | --- |
| 0 | 8 | Re-evaluation of completed finals (E3 seeds 0–2, 12.5% seeds 0–2, depth-warm G150, E1) in the replay harness: reproduction check and the J(D_A, Q_A) terms |
| A | 16 | The two pairs that are repair-free under the shared-gene exchange (E3 seed 1 × depth-warm G150; 12.5% seeds 0 × 1), plus the plain crossing (D_A, Q_B) of the same pairs with production repair, 3 repair seeds |
| B | 30 | Remaining seed pairs of E3 and of the 12.5% runs, shared-gene exchange, repair restricted to the mask's exclusive genes, 3 repair seeds |
| C | 30 | RQ1 at the exact budget: heuristic masks (late layers, late layers keeping the last layer, 3 random, block-influence score) with near-uniform precision at 25% and 12.5%; independent composition of three 16-bit depth-only masks with the three E2 profiles (shifted to the budget) and with near-uniform precision; joint masks with near-uniform precision and with the E2 profile (allocation attribution) |
| D | 13 | 16-bit depth-only evaluation of joint, depth-only and heuristic masks (measured replacement for the paper reference value; not at the budget) |

### Key facts from `pair_analysis.md`

* Of the 48 same-sparsity pairs of completed full-space finals, **none is
  repair-free under plain crossing** (D_A, Q_B): the donor's bit-widths on
  projections it had removed are stale and change the active level sums.
* Under the **shared-gene exchange** (donor bit-widths only on projections active
  under both masks, the mask owner's own bit-widths elsewhere), two pairs are
  repair-free in both directions: E3 seed 1 × depth-warm G150 seed 0 (same
  attention removals, two MLP removals differ) and 12.5% seeds 0 × 1.
* Under the shared-gene exchange the two crossed deficits of a pair are always
  exact negatives, and the contrast is exactly zero for any metric that is
  additive over active projections (tested), so a non-zero contrast on a
  repair-free pair cannot be produced by repair.

## Constructions

* **Repair scopes.** `all` = production repair of the searches
  (`repair_quant_state_to_budget`, group-wise); `exclusive` = the same subset-sum
  repair restricted to projections active under the evaluated mask but removed
  under the bit-width donor; `none` = candidate must already be feasible.
* **Near-uniform precision.** Every active projection of a size group receives
  floor(B_k / a_k) bits and B_k mod a_k projections (evenly spaced in layer
  order) one more, e.g. at 25%: 4 bits, with 4/4/6 projections at 5 bits.
* **Shift.** An E2 profile keeps its relative allocation; the storage freed by
  removal is added as evenly as possible to its active projections, and any
  clipping residual is repaired.
* **Block-influence score.** 1 − mean cosine similarity between the residual
  stream before and after each sublayer on 131,072 FineWeb-Edu calibration
  tokens (dense model); the lowest-scoring sublayers of each type are removed.
  Adapted from ShortGPT's layer-level score to sublayers.
* **Independent depth-only masks.** `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed{0,1,2}`
  (16-bit depth-only EvoPress search, WikiText-2, 8,192 calibration tokens, G20/O16,
  572 s for seed 0). These come from a cheaper protocol than the joint runs; this
  must be stated with every composition result. 12.5% masks can be produced
  with stage `depth125` of the launcher (same protocol).

## How to run (DataLab, one A40)

```bash
# 1. CPU tests (the runner tests need torch + transformers)
python -m pytest tests/test_exact_replay_pure.py tests/test_exact_replay_runner.py \
    tests/test_exact_budget_depth_warm.py -q
# 2. Optional: 12.5% depth-only masks (~30 min), then plan, replay, summary
STAGES="depth125 plan replay summary" bash scripts/run_exact_replays.sh
#    or without 12.5% compositions:
STAGES="plan replay summary" bash scripts/run_exact_replays.sh
# 3. Optional sequential baseline (G150, ~30-40 GPU-hours per seed): print commands
STAGES=frozen bash scripts/run_exact_replays.sh
```

`--calibration_kl` (on by default in the launcher) caches dense logits for
524,288 calibration tokens on disk (about 34 GB) and adds the search objective
on the full calibration set as a metric. The runner validates the realized
storage cost of every candidate against 26,982,023,168 bits and records every
repaired gene. Jobs are resumable.

Runtime estimate (not measured): the E1 evaluation run took 237 s including
model loading; with the model loaded once, about 3–5 minutes per job, i.e.
roughly 5–8 GPU-hours for all 97 jobs.

## Verification status

* `tests/test_exact_replay_pure.py` (18 tests, standard library only): **run and
  passing** in the authoring environment.
* `tests/test_exact_replay_runner.py`, `tests/test_exact_budget_depth_warm.py`
  (incl. the `test_frozen_*` tests) and `tests/test_compression_budget.py`: run on
  DataLab on 2026-10-08 in the rebuilt environment, 55 passed.
