# Zero-shot LM-eval of the independent compositions (planned 2026-10-09)

The six independent compositions of the replay batches:
- jobs `ind_DO12s{0,1,2}_QE2s{i}_shift` (batch `replay_depth125`);
- jobs `ind_DO25s{0,1,2}_QE2s{i}_shift` (batch `replay_20261008`).

Each one is a 16-bit depth-only mask of the screening protocol combined with the E2 profile of the same seed, shifted to the budget and repaired with the production repair (seed 0). The jobs are copied verbatim from `thesis_results/exact_replay_plan_125/replay_plan.json` by `scripts/plan_exact_lmeval.py --set independent`.

Settings are identical to the 11-model LM-eval (`results/exact_lmeval/fullspace_lmeval/settings.json`). The runner needs `--allow_repair`.

`scripts/summarize_exact_lmeval.py --set independent` checks that:
- the bit-width hash of every candidate equals the replay record;
- the settings equal those of the 11-model run.

Commands: `thesis_results/GPU_RUNBOOK_2026-10-09b_independent_lmeval.md`.
