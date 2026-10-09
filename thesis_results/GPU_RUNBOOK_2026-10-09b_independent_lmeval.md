# DataLab runbook C: zero-shot LM-eval of the six independent compositions

Prepared 2026-10-09. Evaluation only; about 30 minutes on the A40.
- Output goes to the new directory `results/exact_lmeval/independent_lmeval/`.
- Nothing completed is modified.
- Settings are the same as the 11-model LM-eval.

## 0. Setup

```bash
cd ~/evopress-replay
source ~/venvs/evopress-sep2026/bin/activate
export PYTHONNOUSERSITE=1
source ~/replay_env.sh
cd ~/evopress-replay
git status --short                      # expect only untracked dry-run folders, no modified files
git fetch origin
git checkout agent/evopress-crossover-v3
git merge --ff-only origin/agent/evopress-crossover-v3
git log --oneline -1                    # expect the commit named in the chat message
nvidia-smi --query-gpu=memory.used --format=csv     # expect close to 0 MiB (nothing else running)
```

## C1. Tests and plan check (CPU, ~1 min)

```bash
python -m pytest tests/test_exact_lmeval_pure.py tests/test_exact_lmeval_runner.py -q    # expect 18 passed
python scripts/plan_exact_lmeval.py --set independent --output_dir /tmp/lmeval_plan_ind_check
diff /tmp/lmeval_plan_ind_check/lmeval_plan.json thesis_results/exact_lmeval_plan_independent/lmeval_plan.json && echo "plan identical"
```

## C2. Dry run: build and repair the six candidates, compare with the replay records (~2 min)

```bash
python evo_exact_lmeval.py --plan thesis_results/exact_lmeval_plan_independent/lmeval_plan.json \
  --quant_db "$QUANT_DB" --output_dir /tmp/smoke_lmeval_ind_dryrun --dry_run --allow_repair 2>&1 | grep -E "^\[|Jobs selected"
# expect 6 lines "[validated] ..."
python scripts/summarize_exact_lmeval.py --set independent --dry_run --results_dir /tmp/smoke_lmeval_ind_dryrun
# expect 4 lines starting with "- [x]", in particular "bit-widths identical to the replay batches: all"
```

If any line shows `- [ ]`, stop and send me the output.

## C3. Full run (detached; resumable: rerun the same command after an interruption)

```bash
mkdir -p results/exact_lmeval/independent_lmeval
nohup python evo_exact_lmeval.py --plan thesis_results/exact_lmeval_plan_independent/lmeval_plan.json \
  --quant_db "$QUANT_DB" --output_dir results/exact_lmeval/independent_lmeval --allow_repair \
  > results/exact_lmeval/independent_lmeval/run.log 2>&1 &
echo "PID $!"
```

## C4. Progress

```bash
grep -E "^\[|Jobs selected" results/exact_lmeval/independent_lmeval/run.log
ls results/exact_lmeval/independent_lmeval/jobs/*/result.json 2>/dev/null | wc -l    # of 6
tail -n 2 results/exact_lmeval/independent_lmeval/run.log
pgrep -af evo_exact_lmeval || echo "process finished"
```

The run is finished when 6 lines start with `[completed]`; the last one is `ind_DO25s2_QE2s2_shift`.

## C5. Verify (send me the full output)

```bash
python scripts/summarize_exact_lmeval.py --set independent --results_dir results/exact_lmeval/independent_lmeval
pip list --format=freeze > results/exact_lmeval/independent_lmeval/environment_freeze.txt
```

Every check line must start with `- [x]`.

## C6. Commit and push

```bash
git add thesis_results/exact_lmeval_plan_independent results/exact_lmeval/independent_lmeval
git commit -m "Zero-shot LM-eval of the six independent compositions"
git pull --rebase origin agent/evopress-crossover-v3
git push origin agent/evopress-crossover-v3
git log --oneline -2
```
