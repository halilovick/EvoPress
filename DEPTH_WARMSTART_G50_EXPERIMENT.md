# Depth Warm-Start G50 Experiment

## 1. Purpose

This experiment asks two focused questions:

1. Does a seed-matched depth-only solution improve joint-search convergence
   only early, or does the advantage remain after 50 generations?
2. Does the same depth warm start combine well with the current
   interaction-aware joint mutation?

It does not introduce a new search algorithm. The stage-two search remains the
existing one-parent EvoPress loop with the existing selection schedule and
final-stage elitism. There is no crossover, population search, frozen
component, or quantization-first condition in this experiment.

## 2. Matched Conditions

| Condition key | Initialization | Mutation | Stage-two generations | Seeds |
| --- | --- | --- | ---: | --- |
| `standard_standard` | Normal random joint initialization (32 candidates) | `standard` | 50 | 0, 1, 2 |
| `depthwarm_standard` | Seed-matched depth-only result plus uniform feasible 3-bit q_proj state (1 candidate) | `standard` | 50 | 0, 1, 2 |
| `standard_interaction` | Normal random joint initialization (32 candidates) | `interaction_aware` | 50 | 0, 1, 2 |
| `depthwarm_interaction` | Seed-matched depth-only result plus uniform feasible 3-bit q_proj state (1 candidate) | `interaction_aware` | 50 | 0, 1, 2 |

The two standard-initialization conditions already exist for all three seeds.
The launcher validates and reuses them:

```text
results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed{0,1,2}
results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed{0,1,2}
```

The six new stage-two run identifiers are:

```text
results/runs/thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed{0,1,2}
results/runs/thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed{0,1,2}
```

A completed `_retryN` directory is accepted when an interrupted base directory
already exists. The original directory is never overwritten.

## 3. Reused Stage-One Runs

The warm conditions reuse exactly the compatible depth-only artifacts used by
the earlier sequential comparison:

```text
results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0
results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1
results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2
```

The stage-one and stage-two seeds are paired. The expected imported
depth-component hashes are:

| Seed | SHA-256 |
| ---: | --- |
| 0 | `454b86987800d97eba43ad3d810527ff7143b7cabdea85e154ab2d26a1831402` |
| 1 | `6314b9647d6ba62e35046b6b9555495ae4a8b41829eb9b8bd11da3c487396508` |
| 2 | `d44e607d9348c77c8013375be1fd7e8e43861bac15c7393a9e295affacefc367` |

For the q_proj database, the uniform 3-bit state is already feasible under
each imported depth mask. Therefore the expected initial parent hashes are:

| Seed | SHA-256 |
| ---: | --- |
| 0 | `d317f58ebca1b047b5be28028314aceae4264823132a18e538debc4c1382bc9b` |
| 1 | `1cf7040a7805247454680910f0d0e8d69f16144cb40389971e40db01e5b3c09d` |
| 2 | `d11f872bfc15c61877d7bb226ac6fb1948dcdffc1b2432d60613c3abb60b28a8` |

The comparison generator recomputes these hashes from the stage-one files. It
does not merely trust this table.

## 4. Fixed Experimental Configuration

All conditions use:

| Setting | Value |
| --- | --- |
| Model | `mistralai/Mistral-7B-v0.3` |
| Quantization database | `outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit` |
| Quantization scope | one `self_attn.q_proj` per decoder layer (32 modules) |
| Available files | 2-, 3-, and 4-bit reconstruction per module (96 `.pth` files) |
| Depth mode | separate attention and MLP masks |
| Depth sparsity | 0.25: exactly 8 attention and 8 MLP drops |
| Quantization constraint | exact 3-bit active average, `group_rule=size` |
| Calibration data | WikiText2 train split |
| Calibration tokens / sequence length | 8,192 / 1,024 |
| Fitness | KL divergence; lower is better |
| Generations / offspring | 50 / 16 |
| Selection tokens | 512, 2,048, 8,192 |
| Selection survivors | 8, 2, 1 |
| Initial tokens | 512 |
| Periodic evaluation | WikiText2 every 5 generations |
| Requested evaluation tokens / sequence length | 524,288 / 1,024 |
| Attention implementation / dtype | SDPA / float16 |
| Standard mutation parameters | maximum 3 depth switches; quantization step size 1 |

The existing DataLab artifacts report 333,824 actually loaded WikiText2
evaluation tokens because that dataset split is shorter than the requested
cap. This is validated and recorded separately from search fitness-token
exposure.

## 5. Launcher Architecture

`scripts/run_depth_warmstart_g50_grid.sh` is a thin matrix orchestrator over
the existing launchers:

```text
standard initialization
    -> scripts/run_joint_search_tiny.sh

depth warm start
    -> scripts/run_sequential_search.sh
    -> scripts/run_joint_search_tiny.sh

both paths
    -> evo_joint_search.py
```

For each condition and seed, the matrix launcher:

1. searches `results/runs` and `outputs/experiments` for a valid base or
   `_retry1` through `_retry20` run;
2. validates `run_summary.json`, `generation_log.csv`,
   `final_candidate.json`, `runtime.txt`, and successful exit status;
3. skips a valid completed run;
4. chooses a new retry identifier rather than overwriting an incomplete run;
5. launches the existing joint search with the fixed settings above;
6. validates the completed output;
7. copies only the ten lightweight reporting/configuration artifacts into
   `results/runs`;
8. generates the comparison deliverables after all 12 matrix cells are valid.

The nested launcher writes the exact Python invocation to each run's
`command.sh`. The master launcher explicitly sets the deprecated
`JOINT_AWARE_MUTATION=0`; interaction-aware cells use only
`--joint_mutation_mode interaction_aware`.

## 6. Validation

Before launching GPU work, the matrix launcher validates:

- the fixed G50/O16 and selection schedules;
- CUDA runtime dependencies and at least 30,000 MiB of visible GPU memory;
- the q_proj database directory, 32 module directories, and 96 `.pth` files;
- each seed-matched stage-one run before a warm condition;
- the successful structured output of any run selected for reuse.

The comparison generator then validates:

- model, seed, calibration, evaluation, fitness, and selection settings;
- 25% separate attention/MLP depth sparsity;
- exact active 3-bit size-group budget;
- 32 unique q_proj module assignments covering layers 0 through 31;
- exactly eight attention and eight MLP drops in every final candidate;
- the selected standard versus interaction-aware mode from `command.sh`;
- absence of deprecated `--joint_aware_mutation`;
- stage-one `depth_only` provenance and seed pairing;
- imported component and initial-parent hashes;
- one exact initial candidate for warm runs;
- zero changed initialization genes, valid active budget, and valid depth
  counts in warm metadata.

Any mismatch stops report generation instead of producing a mixed comparison.

## 7. Cost Accounting

One generation evaluates:

```text
16 candidates × 512 tokens
+ 8 candidates × 2,048 tokens
+ (2 survivors + 1 parent for elitism) × 8,192 tokens
= 27 candidate evaluations
= 49,152 fitness-token exposures
```

The resulting per-run search costs are:

| Initialization | Stage-two candidate evaluations | Stage-two evaluated tokens | Stage-one candidate evaluations | Stage-one evaluated tokens | Total candidate evaluations | Total evaluated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 1,382 | 2,473,984 | 0 | 0 | 1,382 | 2,473,984 |
| Depth warm | 1,351 | 2,458,112 | 572 | 999,424 | 1,923 | 3,457,536 |

The stage-two difference is the initialization: 32 candidates for standard
initialization versus one imported combined candidate for the depth warm
start. For warm runs, total pipeline cost adds the seed-matched G20
depth-only search.

“Evaluated tokens” means calibration-token exposures in fitness calls,
including final-stage parent re-evaluation. It is not a unique-data-token
count and excludes periodic/final PPL evaluation. Runtime is reported
separately as:

- stage-two wall-clock runtime from the joint run;
- depth stage-one wall-clock runtime;
- their sum as total pipeline runtime.

## 8. Generated Comparison Artifacts

After all runs validate, `scripts/summarize_depth_warmstart_g50.py` writes:

```text
results/depth_warmstart_g50_runs.csv
results/depth_warmstart_g50_summary.csv
results/depth_warmstart_g50_paired_deltas.csv
results/depth_warmstart_g50_convergence.csv
results/depth_warmstart_g50_comparison.md
results/depth_warmstart_g50_convergence_generation.png
results/depth_warmstart_g50_convergence_candidate_evaluations.png
results/depth_warmstart_g50_convergence_evaluated_tokens.png
results/depth_warmstart_g50_convergence_stage2_runtime.png
```

The per-run CSV contains final WikiText2 PPL, final search fitness, runtime,
candidate evaluations, and evaluated tokens. The summary CSV contains
three-seed mean and sample SD. The paired CSV contains seed-matched PPL and
cost deltas for:

1. warm versus standard initialization under standard mutation;
2. warm versus standard initialization under interaction-aware mutation;
3. interaction-aware versus standard mutation under standard initialization;
4. interaction-aware versus standard mutation under warm initialization.

The four plots show both KL search fitness and periodic WikiText2 PPL against:

- completed stage-two generations;
- cumulative stage-two candidate evaluations;
- cumulative stage-two evaluated tokens;
- cumulative stage-two runtime.

The long-form convergence CSV also contains total cumulative candidate,
token, and runtime columns including the depth stage-one run. Because the
source log evaluates a parent before mutating it, logged row 1 represents zero
completed generations, rows 6, 11, ..., 46 represent 5, 10, ..., 45 completed
generations, and the final point comes from the finalized post-generation-50
summary. Intermediate runtime timestamps are recorded after each generation,
so runtime-normalized intermediate states can carry an offset of up to one
generation; the final runtime point is exact.

## 9. TU Wien DataLab Commands

The repository evidence identifies the DataLab checkout as
`/home/jovyan/evopress`, branch `main`, remote `origin`, and the already active
Conda environment as `base`. Run the matrix sequentially on one GPU.

### 9.1 Pull and preflight

```bash
cd /home/jovyan/evopress
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main

source /opt/conda/etc/profile.d/conda.sh
conda activate base
python --version
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
nvidia-smi

QDB="outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit"
test -d "$QDB"
test "$(find "$QDB" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 32
test "$(find "$QDB" -mindepth 2 -maxdepth 2 -type f -name '*.pth' | wc -l)" -eq 96

for SEED in 0 1 2; do
  STAGE1="results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed${SEED}"
  python scripts/validate_run_outputs.py "$STAGE1"
done

python -m unittest discover \
  -s tests \
  -p 'test_*depth_warmstart_g50*.py' \
  -v
```

If `/opt/conda/etc/profile.d/conda.sh` is absent, the current Jupyter shell may
already have `(base)` active. Verify with `conda info --envs`; use the path
marked `*` and do not create a different environment for these matched runs.

### 9.2 Inspect the restart-safe plan

```bash
mkdir -p outputs/experiments

bash scripts/run_depth_warmstart_g50_grid.sh --dry-run \
  | tee outputs/experiments/depth_warmstart_g50_grid_dry_run.log
```

The expected plan skips six valid existing standard-initialization runs and
prints six new depth-warm commands. Stop if it proposes a frozen or
quantization-first mode, uses the deprecated `--joint_aware_mutation`, or
cannot find a seed-matched depth stage-one run.

### 9.3 Launch the six required new GPU runs

```bash
nohup bash scripts/run_depth_warmstart_g50_grid.sh \
  > outputs/experiments/depth_warmstart_g50_grid_launcher.log 2>&1 &

GRID_PID=$!
echo "GRID_PID=$GRID_PID"
```

Monitor the grid and the currently active run:

```bash
tail -F outputs/experiments/depth_warmstart_g50_grid_launcher.log
```

In a second terminal:

```bash
ps -fp "$GRID_PID"
nvidia-smi
find outputs/experiments -maxdepth 2 -path '*depthwarm*g50*seed*/generation_log.csv' -print
```

To inspect all completed generation counts without guessing the active run:

```bash
for LOG in outputs/experiments/thesis_depthwarm_*g50_o16_seed*/generation_log.csv; do
  test -f "$LOG" || continue
  printf '%s: ' "$LOG"
  tail -n 1 "$LOG" | cut -d, -f1
done
```

After the launcher exits:

```bash
wait "$GRID_PID"
GRID_EXIT=$?
echo "exit_code=$GRID_EXIT"
test "$GRID_EXIT" -eq 0
```

If the notebook kernel or server restarts, the old shell PID will no longer be
available. Check for a still-running process, then rerun the same launcher:

```bash
pgrep -af 'run_depth_warmstart_g50_grid|evo_joint_search.py' || true

nohup bash scripts/run_depth_warmstart_g50_grid.sh \
  > outputs/experiments/depth_warmstart_g50_grid_resume.log 2>&1 &

GRID_PID=$!
echo "GRID_PID=$GRID_PID"
tail -F outputs/experiments/depth_warmstart_g50_grid_resume.log
```

The launcher skips valid completed runs and preserves an incomplete directory
by selecting `_retryN`.

### 9.4 Validate and regenerate deliverables

```bash
bash scripts/run_depth_warmstart_g50_grid.sh --summarize-only

python - <<'PY'
import csv
from pathlib import Path

expected = {
    "depth_warmstart_g50_runs.csv": 12,
    "depth_warmstart_g50_summary.csv": 4,
    "depth_warmstart_g50_paired_deltas.csv": 12,
    "depth_warmstart_g50_convergence.csv": 600,
}
for name, expected_rows in expected.items():
    path = Path("results") / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_rows, (path, len(rows), expected_rows)
    print("PASS", path, len(rows))

for suffix in (
    "comparison.md",
    "convergence_generation.png",
    "convergence_candidate_evaluations.png",
    "convergence_evaluated_tokens.png",
    "convergence_stage2_runtime.png",
):
    path = Path("results") / f"depth_warmstart_g50_{suffix}"
    assert path.is_file() and path.stat().st_size > 0, path
    print("PASS", path)
PY

python - <<'PY' > /tmp/depth_warmstart_g50_run_dirs.txt
import csv
from pathlib import Path

with Path("results/depth_warmstart_g50_runs.csv").open(
    newline="", encoding="utf-8"
) as handle:
    for row in csv.DictReader(handle):
        if row["initialization"] == "depth_warm":
            print(Path(row["source_summary"]).parent)
PY

while IFS= read -r RUN; do
  python scripts/validate_run_outputs.py "$RUN"
done < /tmp/depth_warmstart_g50_run_dirs.txt

column -s, -t < results/depth_warmstart_g50_summary.csv | less -S
column -s, -t < results/depth_warmstart_g50_paired_deltas.csv | less -S
sed -n '1,240p' results/depth_warmstart_g50_comparison.md
```

### 9.5 Commit only the new lightweight results

Review first:

```bash
git status --short
git diff --check
git diff --stat
```

Add exactly the six warm run directories selected by the validated comparison
CSV, including any retry identifiers, and the nine generated comparison
artifacts:

```bash
while IFS= read -r RUN; do
  git add "$RUN"
done < /tmp/depth_warmstart_g50_run_dirs.txt

git add \
  results/depth_warmstart_g50_runs.csv \
  results/depth_warmstart_g50_summary.csv \
  results/depth_warmstart_g50_paired_deltas.csv \
  results/depth_warmstart_g50_convergence.csv \
  results/depth_warmstart_g50_comparison.md \
  results/depth_warmstart_g50_convergence_generation.png \
  results/depth_warmstart_g50_convergence_candidate_evaluations.png \
  results/depth_warmstart_g50_convergence_evaluated_tokens.png \
  results/depth_warmstart_g50_convergence_stage2_runtime.png

git diff --cached --check
git diff --cached --stat
git status --short
git commit -m "Add depth warm-start G50 experiment results"
git push origin main
```

Do not add `outputs/experiments` model caches, quantization weights, raw logs,
or unrelated dirty files.

## 10. Local Lightweight Verification

No Mistral experiment should be run locally. The implementation-level checks
are:

```bash
python -m unittest discover \
  -s tests \
  -p 'test_*depth_warmstart_g50*.py' \
  -v

bash -n scripts/run_depth_warmstart_g50_grid.sh
python -m py_compile scripts/summarize_depth_warmstart_g50.py
git diff --check
```

The summary generator cannot produce the final twelve-run numerical report
until the six new DataLab runs have completed. It is designed to fail clearly
if any matrix cell or scientific invariant is missing.

## 11. Interpretation Guardrails

- The primary paired comparisons are warm versus standard initialization
  within the same mutation operator.
- Paired deltas after 20 completed generations and after generation 50
  distinguish an early-only effect from a persistent effect.
- Candidate- and token-normalized plots account for the 31-candidate
  stage-two initialization difference.
- Total pipeline cost includes the depth stage-one search only for warm runs.
- Three seeds justify descriptive paired summaries, not strong significance
  claims.
- The conclusions are specific to Mistral-7B, WikiText2, 25% separate
  attention/MLP sparsity, and q_proj-only exact active 3-bit quantization.
