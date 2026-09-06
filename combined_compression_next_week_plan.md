# Combined Compression Experimental Plan

This plan focuses on the next thesis direction: combining compression methods instead of evaluating each method in isolation. The goal is to produce enough concrete, reproducible results for the next supervision meeting to show whether combinations such as depth pruning plus sparsity or depth pruning plus quantization are promising.

The plan is intentionally scoped to a smaller model first. The current Mistral-7B depth-pruning results are strong, but full Mistral-7B SparseGPT database generation is constrained by the `16 GB` Datalab container RAM limit. Combined-method experiments should therefore start on TinyLlama, where the sparse pipeline already completed successfully.

## 0. Meeting Goal

For the next meeting, produce a combined-compression story:

1. Establish single-method TinyLlama baselines under the same data and evaluation settings.
2. Evaluate at least one two-method combination end-to-end.
3. Preferably evaluate two combinations:
   - depth pruning + unstructured sparsity
   - depth pruning + quantization
4. If time allows, run one three-method prototype:
   - depth pruning + sparsity + quantization
5. Compare each combination against the corresponding single-method results.
6. Report whether the combined method improves compression at acceptable perplexity, or whether the combination compounds quality loss too strongly.

The minimum successful meeting material should include:

- a dense TinyLlama reference;
- TinyLlama depth-only results;
- TinyLlama sparse-only result using the already generated SparseGPT database;
- TinyLlama depth + sparse combined evaluation;
- a compact table comparing dense, single-method, and combined-method PPL;
- preserved logs and configs for every failed or completed run.

## 1. Current Starting Point

Already completed:

| Item | Status | Key Result |
| --- | --- | --- |
| Mistral-7B depth pruning | Completed | EvoPress beats random and late-layer baselines. |
| Mistral-7B 37.5% convergence | Completed | WikiText2 PPL improved from `51.69` at generation 10 to `26.00` at generation 20. |
| Mistral-7B seed robustness | Completed | Mean WikiText2 PPL `46.60`, std `5.86` at 37.5%. |
| TinyLlama SparseGPT DB | Completed | `q_proj` database: 22 module dirs, 154 level files, 1233 MB. |
| TinyLlama sparse search | Completed | Final WikiText2 PPL `9.00`, train PPL `9.83`. |
| Hardware documentation | Completed | GPU allocation varies; CPU/container RAM remains `16 GB`. |

Important existing artifacts:

```text
results/experiment_log.csv
results/runs/sparse_db_tinyllama_qproj_s0.50_retry1/
results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/
results/small_model_feasibility_summary.md
results/hardware_bottleneck_summary.md
```

Important Datalab-only artifact to keep:

```text
outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db/
```

Do not delete this sparse database. It is required for depth + sparse combined evaluations.

## 2. Main Experimental Question

The central question is:

> Does combining structural depth pruning with a second compression method produce a better compression-quality tradeoff than either method alone?

This should be tested in a controlled way:

- same model;
- same evaluation dataset;
- same sequence length;
- same dtype and attention implementation;
- same logging format;
- same hardware fields recorded per run.

For the first combined experiments, use:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CALIB_DATA=wikitext2
SEQUENCE_LENGTH=1024
CALIB_TOKENS=4096
EVAL_DATASETS=wikitext2
EVAL_TOKENS=4096
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
```

Note: WikiText2 evaluation in the repository may use the full WikiText2 test split for `wikitext2` regardless of `EVAL_TOKENS`. Record the exact command and script behavior in the run log.

## 3. Experiment Tracking Rules

Every combined run must use the same tracking discipline as the earlier experiments:

```text
outputs/experiments/<run_id>/
  command.sh
  run.log
  runtime.txt
  evaluation_metrics.csv or generation_metrics.csv
  layer_drop_config.txt              # when depth pruning is involved
  sparse_configuration.txt           # when sparse search is involved
  quant_configuration.txt            # when quantization is involved
  combined_config_summary.md
  memory_samples.csv                 # if launcher supports memory sampling
```

Every run should append one row to `results/experiment_log.csv`. Use method names that make combinations obvious:

```text
dense_tiny
depth_evo_tiny
sparse_db_tiny
sparse_search_tiny
combined_depth_sparse_eval
combined_depth_sparse_search
quant_db_tiny
quant_search_tiny
combined_depth_quant_search
combined_depth_sparse_quant_eval
```

Use failed rows honestly. Missing dependencies, missing databases, non-finite PPL, and incompatible loader behavior should all be logged.

Before every run on Datalab:

```bash
python scripts/check_runtime_dependencies.py --require-cuda
```

## 4. Required Code Preparation

### Task 4.1: Confirm Combined Loading Support in `eval_ppl.py`

**Objective**

Allow evaluation of models with more than one compression component loaded at once.

**Current issue**

`eval_ppl.py` currently chooses only one of:

```text
drop_layer_config
sparse_weights_path
quant_weights_path
```

because it uses an `if / elif / elif` chain. That prevents direct evaluation of depth + sparse or depth + quant in one command.

**Required minimal change**

Patch `eval_ppl.py` so loading becomes sequential:

1. load sparse weights if `--sparse_weights_path` is provided;
2. load quant weights if `--quant_weights_path` is provided;
3. apply depth drop config if `--drop_layer_config` is provided.

For combinations, apply structural dropping last. This makes dropped modules ignore any loaded compressed weights inside them, which is expected.

**Model**

Not applicable.

**Parameters**

Not applicable.

**Expected output files**

- patched `eval_ppl.py`
- a small test or dry-run fixture if practical

**Metric to record**

Whether `eval_ppl.py` can accept both:

```text
--drop_layer_config <path>
--sparse_weights_path <path>
--sparse_config_path <path>
```

without ignoring one of them.

**Success/failure criteria**

- Success: `eval_ppl.py` can evaluate depth-only, sparse-only, quant-only, depth+sparse, and depth+quant through the same loader path.
- Failure: combined arguments silently ignore one compression method.

**Small Codex prompt**

> Inspect `eval_ppl.py` and `src/model_utils.py`, then adapt minimally. Change compressed-model loading so sparse weights, quant weights, and depth drop configs can be applied sequentially instead of being mutually exclusive. Apply depth dropping last. Add or update a small test if feasible. Do not run model experiments.

### Task 4.2: Add a Combined Evaluation Launcher

**Objective**

Create a reusable launcher that evaluates an already selected combination, for example depth config + sparse config.

**Method/script to run**

Create:

```text
scripts/run_combined_eval_tiny.sh
```

It should wrap `eval_ppl.py`, save `command.sh`, `run.log`, `runtime.txt`, parse WikiText2 PPL into `evaluation_metrics.csv`, and append to `results/experiment_log.csv`.

The launcher should support:

```text
DROP_LAYER_CONFIG=<path>
SPARSE_WEIGHTS_PATH=<path>
SPARSE_CONFIG_PATH=<path>
SPARSE_DEFAULT_LEVEL=<int>
QUANT_WEIGHTS_PATH=<path>
QUANT_CONFIG_PATH=<path>
QUANT_DEFAULT_LEVEL=<int>
METHOD=<method_name>
RUN_ID=<run_id>
```

**Expected output files**

- `scripts/run_combined_eval_tiny.sh`
- `outputs/experiments/<run_id>/command.sh`
- `outputs/experiments/<run_id>/run.log`
- `outputs/experiments/<run_id>/runtime.txt`
- `outputs/experiments/<run_id>/evaluation_metrics.csv`
- one row in `results/experiment_log.csv`

**Metric to record**

- WikiText2 PPL
- runtime
- GPU name and VRAM
- CPU/container RAM limit
- exact configs used

**Success/failure criteria**

- Success: the launcher can evaluate dense, depth-only, sparse-only, and depth+sparse by changing environment variables.
- Failure: any compression component is silently ignored.

**Small Codex prompt**

> Inspect `scripts/run_dense_eval.sh`, `scripts/run_prune_search_tiny_debug.sh`, `scripts/parse_eval_ppl_log.py`, and `eval_ppl.py`, then adapt minimally. Create `scripts/run_combined_eval_tiny.sh` that can evaluate TinyLlama with optional depth, sparse, and quant configs applied together. Save command, log, runtime, parsed metrics, memory/hardware fields, and one experiment-log row. Do not run experiments automatically.

## 5. Experiment Group A: TinyLlama Single-Method Baselines

These baselines are needed so the combined runs have context.

### Task 5.1: Dense TinyLlama Reference

**Objective**

Record dense TinyLlama WikiText2 PPL under the same evaluation setup used for combined runs.

**Method/script to run**

Use `scripts/run_combined_eval_tiny.sh` with no compression configs.

**Model**

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Parameters**

```text
SEQUENCE_LENGTH=1024
EVAL_DATASETS=wikitext2
EVAL_TOKENS=4096
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
METHOD=dense_tiny
RUN_ID=dense_tinyllama_seq1024_seed0
```

**Expected output files**

```text
outputs/experiments/dense_tinyllama_seq1024_seed0/
results/runs/dense_tinyllama_seq1024_seed0/
```

**Metric to record**

- WikiText2 PPL
- runtime
- hardware fields

**Success/failure criteria**

- Success: finite WikiText2 PPL.
- Failure: dense reference is missing, making combination results hard to interpret.

**Small Codex prompt**

> Inspect the combined TinyLlama evaluator and run a dense TinyLlama WikiText2 reference with sequence length `1024`, `float16`, and `sdpa`. Save logs, parsed PPL, runtime, and an experiment-log row. Do not apply any compression configs.

### Task 5.2: TinyLlama Depth-Only Runs

**Objective**

Create depth-only baselines on the same model used for combined experiments.

**Method/script to run**

Use a parameterized TinyLlama version of the depth-pruning launcher. Existing script:

```text
scripts/run_drop_search_tiny.sh
```

should be adapted minimally or replaced with a logged launcher similar to `scripts/run_drop_search.sh`.

Run at least:

```text
SPARSITY=0.125
SPARSITY=0.25
```

Recommended first settings:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CALIB_DATA=wikitext2
SEQUENCE_LENGTH=1024
CALIB_TOKENS=4096
GENERATIONS=10
OFFSPRING=8
INITIALLY_GENERATED=16
INITIAL_TOKENS=512
SURVIVORS_PER_SELECTION="2 1"
TOKENS_PER_SELECTION="512 2048"
FITNESS_FN=kl
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
SEED=0
```

If runtime is too high, reduce to:

```text
GENERATIONS=5
OFFSPRING=4
```

but label these as quick/debug runs.

**Expected output files**

```text
outputs/experiments/depth_tinyllama_s0.125_seed0/
outputs/experiments/depth_tinyllama_s0.25_seed0/
```

Each should contain:

```text
command.sh
run.log
runtime.txt
layer_drop_config.txt
generation_metrics.csv
```

**Metric to record**

- final WikiText2 PPL
- train PPL
- dropped attention and MLP counts
- generation-wise convergence

**Success/failure criteria**

- Success: at least one finite TinyLlama depth-only result and saved drop config.
- Strong success: both `12.5%` and `25%` finish.
- Failure: no saved depth config, preventing combined evaluation.

**Small Codex prompt**

> Inspect `scripts/run_drop_search.sh`, `scripts/run_drop_search_tiny.sh`, and `evo_drop_search.py`, then adapt minimally. Create a logged TinyLlama depth-search launcher for sparsities `0.125` and `0.25` using WikiText2, sequence length `1024`, `4096` calibration tokens, `10` generations, and `8` offspring. Save configs, metrics, runtime, logs, and experiment-log rows.

### Task 5.3: TinyLlama Sparse-Only Reference

**Objective**

Reuse the completed TinyLlama sparse search as the sparse-only reference.

**Method/script to run**

Already completed:

```text
sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1
```

It used:

```text
SPARSE_WEIGHTS_PATH=outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db
GENERATIONS=20
OFFSPRING=8
```

**Metric to record**

- WikiText2 PPL: `9.00`
- train PPL: `9.83`
- sparse config: `results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt`

**Success/failure criteria**

- Success: treat this as sparse-only reference.
- Failure: do not proceed to depth+sparse if the Datalab sparse database directory has been deleted.

**Small Codex prompt**

> Inspect the completed TinyLlama sparse-search artifacts and confirm that the sparse database still exists on Datalab. Treat `sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1` as the sparse-only reference for combined experiments.

## 6. Experiment Group B: Depth + Sparse Combination

This is the most important combined-method experiment because the sparse pipeline already works on TinyLlama.

### Task 6.1: Evaluate Depth Config + Uniform Sparse Level 0

**Objective**

Test whether depth pruning can be combined with uniform `50%` q-proj sparsity.

**Method/script to run**

Use `scripts/run_combined_eval_tiny.sh` after Task 4.1 and Task 4.2 are complete.

Run:

```text
METHOD=combined_depth_sparse_eval
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DROP_LAYER_CONFIG=outputs/experiments/depth_tinyllama_s0.125_seed0/layer_drop_config.txt
SPARSE_WEIGHTS_PATH=outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db
SPARSE_DEFAULT_LEVEL=0
RUN_ID=combined_tiny_depth0125_sparse50_uniform_seed0
```

If `12.5%` depth succeeds, also run:

```text
DROP_LAYER_CONFIG=outputs/experiments/depth_tinyllama_s0.25_seed0/layer_drop_config.txt
RUN_ID=combined_tiny_depth025_sparse50_uniform_seed0
```

**Expected output files**

```text
outputs/experiments/combined_tiny_depth0125_sparse50_uniform_seed0/
outputs/experiments/combined_tiny_depth025_sparse50_uniform_seed0/
```

**Metric to record**

- WikiText2 PPL
- runtime
- exact depth config
- sparse default level
- hardware fields

**Success/failure criteria**

- Success: finite PPL for at least one depth+sparse combination.
- Failure: combined loader applies only one method or gives non-finite PPL without preserving logs.

**Small Codex prompt**

> Inspect the TinyLlama depth config and sparse database, then run combined evaluation using depth pruning plus uniform sparse default level `0`. Save command, log, runtime, parsed PPL, combined config summary, and an experiment-log row. Do not modify the sparse database.

### Task 6.2: Evaluate Depth Config + Searched Sparse Configuration

**Objective**

Test whether the sparse search configuration remains useful after depth pruning is applied.

**Method/script to run**

Use the sparse config from:

```text
results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt
```

On Datalab, use the matching sparse database path:

```text
outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db
```

Run:

```text
METHOD=combined_depth_sparse_eval
DROP_LAYER_CONFIG=outputs/experiments/depth_tinyllama_s0.125_seed0/layer_drop_config.txt
SPARSE_WEIGHTS_PATH=outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db
SPARSE_CONFIG_PATH=results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt
RUN_ID=combined_tiny_depth0125_sparse_searchcfg_seed0
```

Repeat for `25%` depth if the `12.5%` result is finite.

**Expected output files**

```text
outputs/experiments/combined_tiny_depth0125_sparse_searchcfg_seed0/
outputs/experiments/combined_tiny_depth025_sparse_searchcfg_seed0/
```

**Metric to record**

- WikiText2 PPL
- comparison against depth-only PPL
- comparison against sparse-only PPL `9.00`
- whether searched sparse config transfers to depth-pruned model

**Success/failure criteria**

- Success: finite PPL and clear comparison with both single-method baselines.
- Failure: searched sparse config is incompatible or worsens PPL catastrophically; preserve logs and report as negative result.

**Small Codex prompt**

> Run TinyLlama combined evaluation using the selected depth config plus the previously searched sparse configuration. Compare against depth-only and sparse-only references. Save the result and a short combined-config summary.

### Task 6.3: Optional Joint Depth + Sparse Search

**Objective**

Move beyond sequential evaluation and search over depth and sparse allocation jointly.

**Method/script to run**

Only attempt this after Tasks 6.1 and 6.2 produce interpretable results.

There is currently no confirmed `evo_joint_search.py` equivalent for depth + sparse. Implementing this may be more work than is needed before the meeting. A minimal version could combine:

- depth mutation from `evo_drop_search.py`;
- sparse level mutation from `evo_prune_search.py`;
- one candidate state containing both:

```text
drop_state
sparse_level_state
```

**Recommendation**

Do not make this the first combined experiment. Treat it as optional. Sequential depth+sparse evaluation is enough to show the combined-method direction.

**Success/failure criteria**

- Success: joint search completes at small scale and saves final depth and sparse configs.
- Failure: implementation complexity prevents producing meeting-ready results.

**Small Codex prompt**

> Inspect `evo_drop_search.py` and `evo_prune_search.py`, then prototype a minimal TinyLlama joint depth+sparse search only if sequential combined evaluation already works. Keep generations and offspring small, save both configs, and preserve all logs.

## 7. Experiment Group C: Depth + Quantization Combination

This is the second priority. The repo already contains `evo_joint_search.py`, a prototype joint EvoPress search for depth pruning + quantization.

### Task 7.1: Generate TinyLlama GPTQ Database

**Objective**

Generate the quantization database required for quant-only and depth+quant experiments.

**Method/script to run**

Start from:

```text
scripts/run_gptq_tiny_debug.sh
```

Recommended parameters:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CALIB_DATA=wikitext2
SEQUENCE_LENGTH=1024
CALIB_TOKENS=4096
BITS_LIST="2 3 4"
BITS_TO_LOAD=3
GROUP_SIZE=128
QUANTIZABLE_MODULES='.*layers.*q_proj$'
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
```

**Expected output files**

```text
outputs/experiments/quant_db_tinyllama_qproj_bits234/
```

or, if adapting existing script conventions:

```text
outputs/quant_db_tiny/
```

**Metric to record**

- generated quantized layer files
- database size
- runtime
- peak CPU/GPU memory

**Success/failure criteria**

- Success: q-proj quant database with bitwidth files `2.pth`, `3.pth`, and `4.pth`.
- Failure: quant database missing or incompatible with `evo_quant_search.py` / `evo_joint_search.py`.

**Small Codex prompt**

> Inspect `scripts/run_gptq_tiny_debug.sh`, `quant.py`, and the existing sparse DB launcher. Create a logged TinyLlama GPTQ database launcher for q-proj modules with bits `2 3 4`, group size `128`, WikiText2 `4096` calibration tokens, sequence length `1024`, `float16`, and `sdpa`. Save logs, runtime, database summary, memory samples, and an experiment-log row.

### Task 7.2: Quant-Only TinyLlama Search

**Objective**

Create a quant-only reference before evaluating depth+quant.

**Method/script to run**

Use:

```text
evo_quant_search.py
```

or adapt:

```text
scripts/run_quant_search_tiny_interesting.sh
```

Recommended:

```text
TARGET_BITWIDTH=3.0
GENERATIONS=20
OFFSPRING=8
CALIB_TOKENS=4096
SEQUENCE_LENGTH=1024
FITNESS_FN=kl
```

**Expected output files**

```text
outputs/experiments/quant_search_tinyllama_qproj_3bit_g20_seed0/
```

**Metric to record**

- final WikiText2 PPL
- train PPL
- average bitwidth
- selected quant config

**Success/failure criteria**

- Success: finite quant-only PPL and saved quant config.
- Failure: do not run depth+quant without a quant-only reference unless the meeting goal changes.

**Small Codex prompt**

> Inspect `evo_quant_search.py`, `scripts/run_quant_search_tiny_interesting.sh`, and the generated TinyLlama quant database. Create a logged quant-only TinyLlama search at target bitwidth `3.0`, `20` generations, `8` offspring, WikiText2 settings from the combined plan, and save final config, PPL, runtime, and experiment-log row.

### Task 7.3: Joint Depth + Quant Search

**Objective**

Run the repository's existing prototype joint search for depth pruning + quantization.

**Method/script to run**

Start from:

```text
scripts/run_joint_search_tiny.sh
evo_joint_search.py
```

Recommended first run:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DROP_SPARSITY=0.125
TARGET_BITWIDTH=3.5
GENERATIONS=10
OFFSPRING=8
INITIALLY_GENERATED=16
CALIB_TOKENS=4096
SEQUENCE_LENGTH=1024
FITNESS_FN=kl
```

If stable, run:

```text
DROP_SPARSITY=0.25
TARGET_BITWIDTH=3.5
```

**Expected output files**

```text
outputs/experiments/joint_tiny_depth0125_quant35_seed0/
outputs/experiments/joint_tiny_depth025_quant35_seed0/
```

Each should include:

```text
command.sh
run.log
runtime.txt
generation_metrics.csv
joint_drop_config.txt
joint_quant_config.txt
joint_config.json
```

**Metric to record**

- final WikiText2 PPL
- train PPL
- average bitwidth
- dropped module count
- runtime
- memory peak

**Success/failure criteria**

- Success: finite PPL and saved joint drop/quant configs.
- Partial success: one lower-compression setting completes.
- Failure: joint search has implementation issues; preserve logs and fall back to sequential depth+quant evaluation.

**Small Codex prompt**

> Inspect `evo_joint_search.py` and `scripts/run_joint_search_tiny.sh`, then adapt minimally into a logged launcher for TinyLlama depth+quant joint search. Use q-proj quant database, drop sparsity `0.125`, target bitwidth `3.5`, WikiText2 sequence length `1024`, `4096` calibration tokens, `10` generations, and `8` offspring. Save generation metrics, final joint configs, runtime, logs, memory samples, and an experiment-log row.

## 8. Experiment Group D: Three-Method Prototype

This is optional and should only be attempted if depth+sparse and depth+quant are already working.

### Task 8.1: Sequential Depth + Sparse + Quant Evaluation

**Objective**

Evaluate one model with all three compression components applied.

**Method/script to run**

Use patched `eval_ppl.py` and `scripts/run_combined_eval_tiny.sh`.

Recommended first combination:

```text
DROP_LAYER_CONFIG=<best finite TinyLlama depth config, probably 0.125>
SPARSE_WEIGHTS_PATH=outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db
SPARSE_CONFIG_PATH=results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt
QUANT_WEIGHTS_PATH=<TinyLlama quant database path>
QUANT_CONFIG_PATH=<TinyLlama quant search config, if available>
RUN_ID=combined_tiny_depth0125_sparse50_quant3_seed0
METHOD=combined_depth_sparse_quant_eval
```

**Important caveat**

Sparse and quant databases may both target `q_proj`. Loading both sparse and quant replacements for the exact same weights is not necessarily meaningful unless the order is explicitly defined. For a first three-method prototype, avoid overlapping modules if possible:

- sparse on `q_proj`;
- quant on a different projection type, such as `k_proj` or `v_proj`;
- depth pruning applied last.

If both sparse and quant target the same module, document the loading order and interpret the result cautiously.

**Success/failure criteria**

- Success: finite PPL with clear loading order and non-overlapping module scopes.
- Failure: overlapping replacement databases make the result hard to interpret.

**Small Codex prompt**

> Inspect the available sparse and quant database scopes. If they overlap on q-proj, either choose non-overlapping quant modules or explicitly document loading order. Run one sequential TinyLlama depth+sparse+quant evaluation only after two-method combinations are stable.

## 9. Result Tables and Plots for Combined Experiments

### Task 9.1: Build Combined Results Report

**Objective**

Generate meeting-ready tables and plots for combined methods.

**Method/script to run**

Create or extend:

```text
scripts/build_combined_compression_report.py
```

Generate:

```text
results/combined_compression_summary.csv
results/combined_compression_table.md
results/combined_compression_ppl.png
results/combined_compression_tradeoff.png
results/combined_compression_meeting_notes.md
```

Minimum table columns:

```text
run_id
method
model
depth_sparsity
sparse_scope
sparse_target
quant_scope
target_bitwidth
wikitext2_ppl
train_ppl
runtime_minutes
gpu_name
cpu_ram_limit_gb
status
notes
```

Plot 1:

- x-axis: method label
- y-axis: WikiText2 PPL
- bars: dense, depth-only, sparse-only, depth+sparse, quant-only, depth+quant

Plot 2:

- x-axis: estimated compression setting
- y-axis: WikiText2 PPL
- annotate each point with method label

Use matplotlib only. Do not use seaborn.

**Success/failure criteria**

- Success: one script regenerates all combined-result tables and plots.
- Failure: results require manual spreadsheet edits.

**Small Codex prompt**

> Inspect `results/experiment_log.csv`, combined run artifacts, and previous report builder. Create a matplotlib-only combined-compression report builder that generates a summary CSV, markdown table, PPL plot, and meeting notes for dense, single-method, and combined TinyLlama runs. Exclude failed runs from numeric aggregates but list them explicitly.

## 10. Recommended Execution Order

### Day 1: Prepare Combined Evaluation

1. Patch `eval_ppl.py` for sequential compression loading.
2. Create `scripts/run_combined_eval_tiny.sh`.
3. Run dense TinyLlama reference.
4. Confirm the TinyLlama sparse database still exists on Datalab.

Exit condition:

- dense TinyLlama reference logged;
- combined evaluator can dry-run depth+sparse.

### Day 2: TinyLlama Depth-Only Baselines

1. Create logged TinyLlama depth-search launcher.
2. Run `12.5%` depth-only.
3. Run `25.0%` depth-only if the first run is stable.
4. Sync lightweight artifacts.

Exit condition:

- at least one finite depth-only TinyLlama config available.

### Day 3: Depth + Sparse Evaluation

1. Evaluate `12.5%` depth + uniform sparse level `0`.
2. Evaluate `12.5%` depth + searched sparse config.
3. If both are finite, repeat for `25%` depth.

Exit condition:

- at least one finite depth+sparse result.

### Day 4: Quant Database

1. Generate TinyLlama q-proj quant database.
2. Record database size, runtime, memory peak.
3. If database generation fails, preserve logs and skip joint depth+quant.

Exit condition:

- quant database ready or failure documented.

### Day 5: Quant-Only and Depth + Quant

1. Run quant-only TinyLlama search.
2. Run joint depth+quant search at `12.5%` depth and `3.5` target bitwidth.
3. If stable, run `25%` depth + quant.

Exit condition:

- at least one finite depth+quant result or a preserved failure log explaining why not.

### Day 6: Optional Three-Method Prototype

1. Only attempt if the two-method experiments are stable.
2. Prefer non-overlapping sparse and quant module scopes.
3. Run one depth+sparse+quant evaluation.

Exit condition:

- one finite three-method result or a documented reason not to run it.

### Day 7: Combined Report

1. Generate combined summary CSV.
2. Generate comparison table.
3. Generate plots.
4. Write combined-compression meeting notes.

Exit condition:

- supervisor-ready combined-method report.

## 11. Minimum Viable Result Set

If time is short, prioritize this exact set:

1. Dense TinyLlama reference.
2. TinyLlama depth-only `12.5%`.
3. Existing TinyLlama sparse-only search result.
4. TinyLlama depth `12.5%` + uniform sparse level `0`.
5. TinyLlama depth `12.5%` + searched sparse config.
6. Combined comparison table and short interpretation.

This is enough to answer:

> Does combining depth pruning with unstructured q-proj sparsity look promising on a smaller model?

## 12. Interpretation Rules

Use these rules when explaining results:

- If combined PPL is close to the worse single-method PPL, the methods may be compatible.
- If combined PPL is much worse than both single-method PPLs, the errors compound and the combination may need joint search instead of sequential composition.
- If searched sparse config performs worse after depth pruning than uniform sparse level `0`, the sparse allocation does not transfer well to a depth-pruned model.
- If depth+quant joint search beats sequential depth+quant evaluation, joint search is likely necessary for combinations.
- If TinyLlama combined experiments work but Mistral remains blocked, the thesis can argue that the method direction is valid but scaling requires higher-RAM hardware.

## 13. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Datalab restart removes dependencies | Run `python scripts/check_runtime_dependencies.py --require-cuda` before experiments. |
| Sparse database deleted from `outputs/` | Regenerate TinyLlama sparse DB or restore from Datalab storage if available. |
| Combined loader ignores one method | Patch and test `eval_ppl.py` before running experiments. |
| Depth+sparse PPL collapses | Treat as a useful negative result; compare uniform sparse vs searched sparse. |
| Quant database fails | Skip depth+quant and focus meeting on depth+sparse. |
| Three-method result is hard to interpret | Use non-overlapping sparse and quant module scopes, or postpone. |

## 14. Ready-to-Copy Codex Prompts

### Prompt: Prepare Combined Evaluator

> Inspect `eval_ppl.py`, `src/model_utils.py`, `scripts/run_dense_eval.sh`, and `scripts/parse_eval_ppl_log.py`. Patch `eval_ppl.py` so depth, sparse, and quant compression loaders can be applied sequentially, with depth applied last. Then create `scripts/run_combined_eval_tiny.sh` to evaluate TinyLlama with optional `DROP_LAYER_CONFIG`, `SPARSE_WEIGHTS_PATH`, `SPARSE_CONFIG_PATH`, `QUANT_WEIGHTS_PATH`, and `QUANT_CONFIG_PATH`. Save command, log, runtime, parsed PPL, hardware fields, and one experiment-log row. Do not run model experiments automatically.

### Prompt: Run TinyLlama Depth Baselines

> Inspect `scripts/run_drop_search.sh`, `scripts/run_drop_search_tiny.sh`, and `evo_drop_search.py`. Create a logged TinyLlama depth-pruning launcher for `0.125` and `0.25` sparsity using WikiText2, sequence length `1024`, `4096` calibration tokens, `10` generations, `8` offspring, `float16`, and `sdpa`. Save configs, generation metrics, runtime, logs, and experiment-log rows.

### Prompt: Run Depth + Sparse Evaluation

> Inspect the completed TinyLlama sparse database and sparse-search config. Use the combined evaluator to run TinyLlama depth+sparse evaluations for the `0.125` depth config with both uniform sparse level `0` and the searched sparse config. Compare against dense, depth-only, and sparse-only references. Preserve all logs and append experiment-log rows.

### Prompt: Prepare TinyLlama Quant Database

> Inspect `scripts/run_gptq_tiny_debug.sh`, `quant.py`, and the sparse DB launcher. Create a logged TinyLlama GPTQ database launcher for q-proj modules with bits `2 3 4`, group size `128`, WikiText2 `4096` calibration tokens, sequence length `1024`, `float16`, and `sdpa`. Save logs, runtime, database summary, memory samples, and an experiment-log row.

### Prompt: Run Joint Depth + Quant

> Inspect `evo_joint_search.py` and `scripts/run_joint_search_tiny.sh`, then adapt minimally into a logged launcher for TinyLlama depth+quant joint search. Use the TinyLlama q-proj quant database, drop sparsity `0.125`, target bitwidth `3.5`, WikiText2 sequence length `1024`, `4096` calibration tokens, `10` generations, and `8` offspring. Save generation metrics, final joint configs, runtime, logs, memory samples, and an experiment-log row.

### Prompt: Build Combined Report

> Inspect `results/experiment_log.csv`, combined experiment artifacts, and `scripts/build_experiment_report.py`. Create a matplotlib-only combined-compression report builder that generates `results/combined_compression_summary.csv`, `results/combined_compression_table.md`, `results/combined_compression_ppl.png`, and `results/combined_compression_meeting_notes.md`. Include dense, depth-only, sparse-only, quant-only, and combined TinyLlama runs. Exclude failed runs from aggregates but list them explicitly.
