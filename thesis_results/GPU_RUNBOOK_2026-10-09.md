# DataLab runbook: full-space LM-eval and 12.5% independent baseline

Prepared 2026-10-09. Two GPU experiments, run one after the other on the A40
(never both at the same time). Nothing here changes completed results: every
output goes to a new directory, and the 97-job replay plan is not rewritten.

| Experiment | Output | Expected time |
| --- | --- | --- |
| A. Zero-shot LM-eval, 11 full-space models | `results/exact_lmeval/fullspace_lmeval/` | ~5 min setup + ~8 min checks + 35–50 min |
| B. 12.5% depth-only masks + 9 replays (6 at T, 3 in 16-bit) | `results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed{0,1,2}/`, `results/exact_replays/replay_depth125/` | ~30 min searches + ~70 min replays |

## 0. Session setup (every new terminal)

```bash
cd ~/evopress-replay
source ~/venvs/evopress-sep2026/bin/activate
export PYTHONNOUSERSITE=1
source ~/replay_env.sh          # sets QUANT_DB, logits cache in /tmp, unbuffered output
cd ~/evopress-replay
echo "$QUANT_DB"; ls "$QUANT_DB" | wc -l        # expect 224
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv   # expect A40, ~0 MiB used
```

Only once, to get the new code:

```bash
git status --short             # expect no modified tracked files
git pull --ff-only origin agent/evopress-crossover-v3
git log --oneline -1
```

## A. Full-space zero-shot LM-eval

### A1. Install lm_eval 0.4.13 without changing any installed package (once)

```bash
python -c "import lm_eval" 2>/dev/null && echo "lm_eval already installed" || echo "lm_eval missing"
pip list --format=freeze > ~/venv_before_lmeval.txt
pip install "lm_eval==0.4.13" -c ~/venv_before_lmeval.txt
pip list --format=freeze > ~/venv_after_lmeval.txt
comm -23 <(sort ~/venv_before_lmeval.txt) <(sort ~/venv_after_lmeval.txt)   # must print NOTHING
python -c "import torch, transformers, datasets; from importlib.metadata import version as v; print(torch.__version__, transformers.__version__, datasets.__version__, v('lm_eval'))"
# expect: 2.8.0+cu128 4.56.2 5.0.1 0.4.13
```

If `pip install` reports a conflict, stop and send the message; do not
relax the constraint file.

### A2. Tests and plan check (CPU, ~1 min)

```bash
python -m pytest tests/test_exact_lmeval_pure.py tests/test_exact_lmeval_runner.py -q
# expect: 14 passed
python scripts/plan_exact_lmeval.py --output_dir /tmp/lmeval_plan_check
diff /tmp/lmeval_plan_check/lmeval_plan.json thesis_results/exact_lmeval_plan/lmeval_plan.json && echo "plan identical"
```

### A3. Smoke test (~5 min; 2% of each task, separate directory, not reported)

```bash
python evo_exact_lmeval.py --plan thesis_results/exact_lmeval_plan/lmeval_plan.json \
  --quant_db "$QUANT_DB" --output_dir /tmp/smoke_fullspace_lmeval \
  --jobs dense own_E3s0 --limit 0.02 2>&1 | tee /tmp/lmeval_smoke.log | grep -E "^\[|Jobs selected|Error|error"
# expect two lines "[completed] dense {...}" and "[completed] own_E3s0 {...}"
```

### A4. Validate all 11 candidates without evaluation (~2 min)

```bash
python evo_exact_lmeval.py --plan thesis_results/exact_lmeval_plan/lmeval_plan.json \
  --quant_db "$QUANT_DB" --output_dir /tmp/smoke_lmeval_dryrun --dry_run 2>&1 | grep -E "^\[|Jobs selected"
# expect 11 lines "[validated] ..."
```

### A5. Full run (detached; resumable: rerun the same command after an interruption)

```bash
mkdir -p results/exact_lmeval/fullspace_lmeval
nohup python evo_exact_lmeval.py --plan thesis_results/exact_lmeval_plan/lmeval_plan.json \
  --quant_db "$QUANT_DB" --output_dir results/exact_lmeval/fullspace_lmeval \
  > results/exact_lmeval/fullspace_lmeval/run.log 2>&1 &
echo "PID $!"
```

### A6. Progress

```bash
grep -E "^\[|Jobs selected" results/exact_lmeval/fullspace_lmeval/run.log          # one line per finished model
ls results/exact_lmeval/fullspace_lmeval/jobs/*/result.json 2>/dev/null | wc -l   # finished models (of 11)
tail -n 3 results/exact_lmeval/fullspace_lmeval/run.log                           # current task progress
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

### A7. Verify and summarize (CPU)

```bash
python scripts/summarize_exact_lmeval.py --results_dir results/exact_lmeval/fullspace_lmeval
# every check must show [x]; then send me analysis/verification.md and analysis/lmeval_by_model.md
pip list --format=freeze > results/exact_lmeval/fullspace_lmeval/environment_freeze.txt
```

### A8. Commit and push

```bash
git add thesis_results/exact_lmeval_plan results/exact_lmeval/fullspace_lmeval
git commit -m "Full-space zero-shot LM-eval of E0, E1, E2, J12, E3 (11 models)"
git push origin agent/evopress-crossover-v3
```

The smoke and dry-run outputs are written to `/tmp` and are not committed.

## B. 12.5% independent baseline

### B1. Preflight (nothing may exist yet)

```bash
ls -d results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed* 2>/dev/null      # expect nothing
ls -d outputs/experiments/thesis_medium_depth_mistral_s0.125_g20_o16_seed* 2>/dev/null # expect nothing
ls -d thesis_results/exact_replay_plan_125 results/exact_replays/replay_depth125 2>/dev/null  # expect nothing
df -h /tmp | tail -1          # need >= 40 GB free for the dense calibration logits
```

### B2. Three depth-only searches (~30 min; same launcher and settings as the 25% masks)

```bash
RUN_DENSE=0 METHODS=depth DEPTH_SPARSITY=0.125 SEEDS="0 1 2" \
  EXPERIMENT_LOG=results/experiment_log_depth125.csv \
  nohup bash scripts/run_mistral_medium_grid.sh > /tmp/depth125.log 2>&1 &
echo "PID $!"
```

Progress:

```bash
grep -E "=== Starting|synced|failed|Error" /tmp/depth125.log
tail -n 2 /tmp/depth125.log
```

Check the masks when the log ends with `Mistral medium grid complete.`:

```bash
for s in 0 1 2; do python -c "
import json; c=json.load(open('results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed$s/final_candidate.json'))
print('seed $s attn', [i for i,v in enumerate(c['attention_mask']) if v], 'mlp', [i for i,v in enumerate(c['mlp_mask']) if v])"; done
diff <(sed 's/0\.125/X/g; s/s0\.125/X/g' results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed0/command.sh) \
     <(sed 's/0\.25/X/g; s/s0\.25/X/g' results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/command.sh)
# expect 4 attention and 4 MLP layers per seed; the diff may only show the working directory (cd line)
cp /tmp/depth125.log results/exact_replays/depth125_search.log
```

### B3. Extended plan in a NEW directory, and proof that the 97 completed jobs are unchanged

```bash
python scripts/plan_exact_replays.py --output_dir thesis_results/exact_replay_plan_125 --include_depth_only_125
# expect: pairs=48 owner_repair_free=2 jobs=106
python scripts/compare_replay_plans.py thesis_results/exact_replay_plan/replay_plan.json \
  thesis_results/exact_replay_plan_125/replay_plan.json
# expect: added (9) ...; removed (0); changed (0); "OK: all jobs of the old plan are unchanged"
```

### B4. Validate the 9 new candidates without evaluation (~2 min)

```bash
JOBS125="fp16_DO12s0 fp16_DO12s1 fp16_DO12s2 ind_DO12s0_QE2s0_shift ind_DO12s0_nu ind_DO12s1_QE2s1_shift ind_DO12s1_nu ind_DO12s2_QE2s2_shift ind_DO12s2_nu"
python evo_exact_replay.py --plan thesis_results/exact_replay_plan_125/replay_plan.json --quant_db "$QUANT_DB" \
  --output_dir /tmp/replay_depth125_dryrun --jobs $JOBS125 --dry_run 2>&1 | grep -E "^\[|Jobs selected"
# expect 9 lines "[validated] ..."
```

### B5. Replays (detached; ~70 min; resumable with the same command)

```bash
JOBS125="fp16_DO12s0 fp16_DO12s1 fp16_DO12s2 ind_DO12s0_QE2s0_shift ind_DO12s0_nu ind_DO12s1_QE2s1_shift ind_DO12s1_nu ind_DO12s2_QE2s2_shift ind_DO12s2_nu"
mkdir -p results/exact_replays/replay_depth125
nohup python evo_exact_replay.py --plan thesis_results/exact_replay_plan_125/replay_plan.json \
  --quant_db "$QUANT_DB" --output_dir results/exact_replays/replay_depth125 \
  --jobs $JOBS125 --calibration_kl > results/exact_replays/replay_depth125/run.log 2>&1 &
echo "PID $!"
```

Progress:

```bash
grep -E "^\[|Jobs selected" results/exact_replays/replay_depth125/run.log
ls results/exact_replays/replay_depth125/jobs/*/result.json 2>/dev/null | wc -l    # of 9
tail -n 2 results/exact_replays/replay_depth125/run.log
```

### B6. Verify

```bash
JOBS125="fp16_DO12s0 fp16_DO12s1 fp16_DO12s2 ind_DO12s0_QE2s0_shift ind_DO12s0_nu ind_DO12s1_QE2s1_shift ind_DO12s1_nu ind_DO12s2_QE2s2_shift ind_DO12s2_nu"
python scripts/verify_replay_jobs.py --results_dir results/exact_replays/replay_depth125 \
  --jobs $JOBS125 --expect_removed 4 4
# expect "ALL CHECKS PASSED"; send me the printed table
pip list --format=freeze > results/exact_replays/replay_depth125/environment_freeze.txt
```

### B7. Commit and push

```bash
git add results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed0 \
        results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed1 \
        results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed2 \
        results/experiment_log_depth125.csv results/exact_replays/depth125_search.log \
        thesis_results/exact_replay_plan_125 results/exact_replays/replay_depth125
git commit -m "12.5% depth-only masks (S protocol) and independent-composition replays at T"
git push origin agent/evopress-crossover-v3
```

## If something goes wrong

* A job prints `[failed] ... <error>`: the run continues with the next job;
  rerunning the same command retries only unfinished jobs. Send me the error line.
* The process disappeared (DataLab restart): rerun the same `nohup` command; finished
  jobs are skipped.
* Do not use `scripts/run_exact_replays.sh` for B: its default plan and output
  directories point at the completed batch (the plan stage now refuses to overwrite).
