# EvoPress Next 7 Days Experimental Plan

This plan is intentionally scoped to one week of thesis-progress experiments. It prioritizes a coherent, reproducible experimental story under the current hardware limits. It does not attempt a full reproduction of every EvoPress paper result.

## 0. Goals for the next meeting

The goal is to show a defensible experimental narrative:

1. EvoPress depth pruning works end-to-end on `mistralai/Mistral-7B-v0.3`.
2. The compression-versus-quality tradeoff is visible across several depth-pruning sparsity levels.
3. EvoPress improves over cheap depth-pruning baselines under the same removal budget.
4. The search behavior is visible generation by generation.
5. At least one important setting has been repeated with multiple random seeds.
6. The failure of full Mistral-7B sparse database generation is documented as a hardware limitation rather than left as an unexplained crash.
7. If time permits, the sparse and quantization database pipelines are demonstrated on a smaller compatible model.

The meeting material should contain:

- depth-pruning results at `12.5%`, `25.0%`, `37.5%`, and `50.0%`
- a dense reference and simple random and late-layer baselines
- convergence evidence for the `37.5%` EvoPress run
- at least one three-seed robustness test
- a hardware snapshot and sparse-database failure note
- an optional smaller-model sparse and quantization pipeline feasibility result
- plots, compact tables, and a thesis-progress summary

The already observed reduced WikiText2 depth-pruning run is the starting point: for `mistralai/Mistral-7B-v0.3`, WikiText2 perplexity improved from approximately `994` to `85` over `5` generations at `37.5%` depth sparsity. Treat this as a promising preliminary observation. Archive its exact command and log if available; do not present the approximate values as final results.

## 1. Current known constraints

| Resource                           | Current environment                                                                      | Practical implication                                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| GPU                                | NVIDIA A40, approximately `45 GB` VRAM                                                   | Mistral-7B depth pruning is feasible in `float16`.                                                                        |
| CPU/container RAM                  | `16 GB` limit                                                                            | CPU memory is the main bottleneck. Monitor it during database generation.                                                 |
| Depth pruning                      | Already completed successfully on reduced WikiText2 settings                             | Make this the primary experimental path for the week.                                                                     |
| Full Mistral-7B SparseGPT database | Previous `scripts/run_sparse_gpt.sh` / `prune.py` attempts crashed                       | Do not immediately retry the full database. Preserve failure evidence and validate the pipeline on a smaller model first. |
| Disk usage                         | The repository README warns that sparse and quantized databases can require `100-200 GB` | Check free disk space before any database generation attempt.                                                             |

The likely bottleneck is CPU/container RAM, not GPU VRAM. `prune.py` and `quant.py` generate and save multiple replacement tensors per layer. Their CPU offload options can reduce GPU pressure while increasing CPU RAM pressure. Depth pruning is lighter because it patches module forwards in memory and does not precompute a full per-layer sparse or quantized weight database.

Important interpretation note: `evo_drop_search.py` calculates its removal budget from `int(sparsity * number_of_layers)` and may distribute removals across attention and MLP submodules. For fair baselines, match the exact final number of dropped attention and MLP modules produced by EvoPress at each setting instead of assuming that the displayed sparsity percentage directly equals a percentage of all submodules.

## 2. Experiment tracking setup

### Task 2.1: Create experiment logging structure

**Objective**

Create a single source of truth for completed, failed, and skipped runs.

**Method/script to run**

Create:

- `results/experiment_log.csv`
- `scripts/append_experiment_log.py`
- `outputs/experiments/`

The helper should append one CSV row per experiment, preserve the column order below, create the file with a header if it does not exist, and reject rows missing required identifiers such as `run_id`, `method`, `model`, `status`, and `output_dir`.

**Model**

Not applicable.

**Parameters**

Use exactly these CSV columns:

```text
date,run_id,method,model,sparsity_or_bits,generations,offspring,calibration_data,sequence_length,calibration_tokens,fitness_fn,attention_impl,dtype,seed,wikitext2_ppl,train_ppl,runtime_minutes,gpu_name,gpu_vram_gb,cpu_ram_limit_gb,status,notes,output_dir
```

Use these status values consistently:

```text
planned,running,completed,failed,skipped
```

Every run directory should contain:

```text
outputs/experiments/<run_id>/
  command.sh
  run.log
  runtime.txt
  layer_drop_config.txt        # when applicable
  generation_metrics.csv       # when applicable
  notes.md                     # optional manual notes
```

**Expected output files**

- `results/experiment_log.csv`
- `scripts/append_experiment_log.py`
- `outputs/experiments/`

**Metric to record**

All columns in `results/experiment_log.csv`. For failures, record `runtime_minutes`, the last successful step, and the error summary in `notes`.

**Success/failure criteria**

- Success: one test row can be appended and read back without corrupting the header or column order.
- Failure: rows require manual CSV editing, columns drift between runs, or failed experiments are omitted.

**Small Codex prompt**

> Inspect the existing scripts and adapt minimally. Create `results/experiment_log.csv`, `scripts/append_experiment_log.py`, and the output-directory convention described in `experiments_next_7_days.md`. Do not run model experiments. Validate the append helper with one clearly labeled temporary test row and remove that test row afterward.

### Task 2.2: Add a depth-search log parser

**Objective**

Turn stdout from `evo_drop_search.py` into generation-wise metrics that can be plotted without manual transcription.

**Method/script to run**

Create `scripts/parse_depth_search_log.py`. The current depth search prints:

```text
Generation <n>/<total>
Train fitness <value>
Parent: attn: [...] mlp: [...]
wikitext2: <ppl>
full train ppl: <ppl>
```

Parse these lines into one row per generation. Also capture the final evaluation separately or mark it with `phase=final`.

**Model**

Not applicable.

**Parameters**

Input:

```text
--log outputs/experiments/<run_id>/run.log
--output outputs/experiments/<run_id>/generation_metrics.csv
```

Recommended CSV columns:

```text
run_id,phase,generation,train_fitness,wikitext2_ppl,train_ppl,parent_attn_mask,parent_mlp_mask
```

**Expected output files**

- `scripts/parse_depth_search_log.py`
- `outputs/experiments/<run_id>/generation_metrics.csv`

**Metric to record**

Parser coverage: number of parsed generations versus expected generations, plus final WikiText2 PPL and final train PPL.

**Success/failure criteria**

- Success: the parser extracts all generations from the known five-generation Mistral log and preserves the parent masks.
- Failure: the parser silently drops generations or confuses per-generation evaluation with final evaluation.

**Small Codex prompt**

> Inspect `evo_drop_search.py` stdout formatting and adapt minimally. Create a parser that converts one depth-search `run.log` into `generation_metrics.csv`, including generation, train fitness, WikiText2 PPL, train PPL, and parent attention/MLP masks. Add a small parser test using a fixture log. Do not run a model experiment.

**Usage**

`python scripts/parse_depth_search_log.py \
  --log outputs/experiments/<run_id>/run.log \
  --output outputs/experiments/<run_id>/generation_metrics.csv`

## 3. Main experiment group A: Mistral-7B depth pruning curve

### Task 3.1: Run the four-point EvoPress depth-pruning grid

**Objective**

Produce the main Mistral-7B compression-versus-perplexity curve.

**Method/script to run**

Use `scripts/run_drop_search.sh` as the starting launcher and adapt it minimally so values can be passed by environment variable or CLI argument. For each sparsity, save the exact command before execution and tee stdout/stderr to `run.log`.

Run:

```bash
MODEL="mistralai/Mistral-7B-v0.3"
CALIB_DATA="wikitext2"
SEQUENCE_LENGTH=2048
CALIB_TOKENS=8192
GENERATIONS=10
OFFSPRING=8
INITIALLY_GENERATED=16
INITIAL_TOKENS=512
TOKENS_PER_SELECTION="512 2048"
FITNESS_FN="kl"
ATTN_IMPLEMENTATION="sdpa"
DTYPE="float16"
SEED=1

for SPARSITY in 0.125 0.25 0.375 0.50; do
  # Create a unique RUN_ID and OUTPUT_DIR.
  # Run evo_drop_search.py with the settings above.
  # Save command.sh, run.log, runtime.txt, layer_drop_config.txt,
  # generation_metrics.csv, and one experiment_log.csv row.
done
```

The adapted launcher should invoke the current entry point with:

```bash
python evo_drop_search.py \
  --model_name_or_path "$MODEL" \
  --sparsity "$SPARSITY" \
  --calibration_data "$CALIB_DATA" \
  --calibration_tokens "$CALIB_TOKENS" \
  --calibration_sequence_length "$SEQUENCE_LENGTH" \
  --eval_every 1 \
  --eval_datasets wikitext2 \
  --eval_sequence_length "$SEQUENCE_LENGTH" \
  --population_size 1 \
  --generations "$GENERATIONS" \
  --offspring "$OFFSPRING" \
  --initially_generated "$INITIALLY_GENERATED" \
  --initial_tokens "$INITIAL_TOKENS" \
  --survivors_per_selection 2 1 \
  --tokens_per_selection 512 2048 \
  --fitness_fn "$FITNESS_FN" \
  --use_fast_tokenizer \
  --drop_config_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION"
```

Do not add `--drop_entire_block` unless the thesis comparison is intentionally changed to whole-block pruning. Keep this week's primary runs aligned with the already successful reduced setup.

**Model**

`mistralai/Mistral-7B-v0.3`

**Parameters**

| Parameter                | Value                            |
| ------------------------ | -------------------------------- |
| Sparsity grid            | `0.125`, `0.25`, `0.375`, `0.50` |
| Calibration data         | `wikitext2`                      |
| Sequence length          | `2048`                           |
| Calibration tokens       | `8192`                           |
| Generations              | `10`                             |
| Offspring                | `8`                              |
| Initially generated      | `16`                             |
| Initial tokens           | `512`                            |
| Survivors per selection  | `2 1`                            |
| Tokens per selection     | `512 2048`                       |
| Fitness                  | `kl`                             |
| Attention implementation | `sdpa`                           |
| Dtype                    | `float16`                        |
| Main-curve seed          | `1`                              |

**Expected output files**

For each sparsity:

- `outputs/experiments/depth_mistral7b_s<value>_seed1/command.sh`
- `outputs/experiments/depth_mistral7b_s<value>_seed1/run.log`
- `outputs/experiments/depth_mistral7b_s<value>_seed1/runtime.txt`
- `outputs/experiments/depth_mistral7b_s<value>_seed1/layer_drop_config.txt`
- `outputs/experiments/depth_mistral7b_s<value>_seed1/generation_metrics.csv`
- one row in `results/experiment_log.csv`

**Metric to record**

- final WikiText2 PPL
- final train PPL
- generation-wise WikiText2 PPL
- generation-wise train PPL
- generation-wise KL train fitness
- runtime in minutes
- exact count of dropped attention and MLP modules from the final config

**Success/failure criteria**

- Success: all four runs finish, each has a final config, metrics parse cleanly, and WikiText2 PPL is finite.
- Partial success: at least three sparsity levels finish and any failed level has a preserved log and experiment-log row.
- Failure: runs are restarted without preserving logs, or results cannot be mapped to exact parameters and configs.

**Small Codex prompt**

> Inspect `scripts/run_drop_search.sh` and `evo_drop_search.py`, then adapt minimally. Add a parameterized launcher for the Mistral-7B depth-pruning grid at sparsities `0.125 0.25 0.375 0.50` with the exact reduced WikiText2 settings in `experiments_next_7_days.md`. Save commands, tee logs, record runtime, parse generation metrics, append experiment-log rows, and run the grid sequentially. Do not change the search algorithm.

**usage**

`bash scripts/run_drop_search_grid.sh`

`bash scripts/run_drop_search_grid.sh --continue-on-failure`

## 4. Experiment group B: convergence extension for the strongest existing setting

### Task 4.1: Extend the `37.5%` run

**Objective**

Check whether the search was still improving after generation `5` and whether a larger search budget lowers WikiText2 PPL further.

**Method/script to run**

Repeat the `37.5%` depth-pruning run from Task 3.1 as a fresh, clearly labeled run with:

```text
SPARSITY=0.375
GENERATIONS=20
OFFSPRING=8
SEED=1
```

If runtime is acceptable after the `20`-generation run, schedule a `30`-generation run rather than silently changing the completed run's metadata. The current search does not expose checkpoint resume behavior, so "continue" should mean a fresh longer run unless checkpoint support is inspected and added explicitly.

Use the same dataset, sequence length, calibration tokens, fitness, dtype, and attention implementation as Task 3.1.

**Model**

`mistralai/Mistral-7B-v0.3`

**Parameters**

| Parameter      | Value                                   |
| -------------- | --------------------------------------- |
| Sparsity       | `0.375`                                 |
| Generations    | `20`; optionally `30` if runtime allows |
| Offspring      | `8`                                     |
| Other settings | exactly the same as Task 3.1            |

**Expected output files**

- `outputs/experiments/depth_mistral7b_s0.375_g20_seed1/command.sh`
- `outputs/experiments/depth_mistral7b_s0.375_g20_seed1/run.log`
- `outputs/experiments/depth_mistral7b_s0.375_g20_seed1/runtime.txt`
- `outputs/experiments/depth_mistral7b_s0.375_g20_seed1/layer_drop_config.txt`
- `outputs/experiments/depth_mistral7b_s0.375_g20_seed1/generation_metrics.csv`
- optionally `best_config_generation_<n>.txt` snapshots if minimal instrumentation is added
- one row in `results/experiment_log.csv`

**Metric to record**

- generation-wise WikiText2 PPL
- generation-wise train PPL
- generation-wise train fitness
- best candidate attention/MLP masks per generation
- final config
- runtime
- best generation and PPL improvement after generation `5`

**Success/failure criteria**

- Success: at least `20` generations complete and the convergence plot clearly shows whether PPL stabilizes or continues improving.
- Failure: only the final metric is retained, making convergence analysis impossible.

**Small Codex prompt**

> Inspect the current depth-search launcher and `evo_drop_search.py`, then adapt minimally. Run a fresh `mistralai/Mistral-7B-v0.3` depth-pruning convergence experiment at `0.375` sparsity for `20` generations with the same reduced WikiText2 settings as the main grid. Preserve generation-wise metrics and parent masks. Add per-generation config snapshots only if the existing log is insufficient.

## 5. Experiment group C: simple depth-pruning baselines

### Task 5.1: Evaluate the dense reference

**Objective**

Measure an exact dense WikiText2 reference under the same evaluation setup. Do not infer the dense baseline from the preliminary `994 -> 85` observation.

**Method/script to run**

Use `eval_ppl.py` without a compression config:

```bash
python eval_ppl.py \
  --model_name_or_path "mistralai/Mistral-7B-v0.3" \
  --eval_datasets wikitext2 \
  --sequence_length 2048 \
  --dtype float16 \
  --attn_implementation sdpa \
  --use_fast_tokenizer \
  --seed 1
```

Tee the output and add a row with `method=dense`.

**Model**

`mistralai/Mistral-7B-v0.3`

**Parameters**

Use WikiText2 evaluation, sequence length `2048`, `float16`, `sdpa`, seed `1`.

**Expected output files**

- `outputs/experiments/dense_mistral7b_seed1/command.sh`
- `outputs/experiments/dense_mistral7b_seed1/run.log`
- `outputs/experiments/dense_mistral7b_seed1/runtime.txt`
- one row in `results/experiment_log.csv`

**Metric to record**

Dense WikiText2 PPL and runtime.

**Success/failure criteria**

- Success: one finite dense WikiText2 PPL is recorded with the exact command.
- Failure: the dense baseline is left as an approximate or inferred value.

**Small Codex prompt**

> Inspect `eval_ppl.py` and adapt minimally. Run and log one dense Mistral-7B WikiText2 perplexity evaluation using sequence length `2048`, `float16`, `sdpa`, and seed `1`. Append the result to `results/experiment_log.csv`.

### Task 5.2: Implement and run cheap depth-pruning baselines

**Objective**

Compare EvoPress against simple removal strategies with matched dropped-module counts.

**Method/script to run**

Create `scripts/evaluate_depth_baselines.py` or an equivalent minimal wrapper. Reuse the existing model utilities used by `evo_drop_search.py`:

- `get_layers`
- `get_attn_layer_name`
- `get_mlp_layer_name`
- `dummy_initialize`
- `make_dummy_forward`
- `restore_forward`

Use each EvoPress final `layer_drop_config.txt` to count dropped attention and MLP modules. Evaluate:

1. `random`: randomly choose the same count of attention and MLP removals as the matching EvoPress config. Run seeds `1`, `2`, and `3`.
2. `late_layer`: remove the matching number of attention and MLP modules starting from the deepest layer indices.
3. `early_layer` optional negative control: remove modules starting from the earliest layer indices. If removing layer `0` collapses evaluation or creates non-finite PPL, preserve the failure and rerun with layer `0` protected. Document both facts.

If later experiments intentionally use `--drop_entire_block`, baseline configs must also drop entire blocks. For the current setup, match attention and MLP counts independently.

The existing `drop_scoring.py` is useful reference code for applying dropped forwards and evaluating PPL, but the requested random and fixed-position heuristics should be kept simple.

**Model**

`mistralai/Mistral-7B-v0.3`

**Parameters**

| Parameter                | Value                                                                           |
| ------------------------ | ------------------------------------------------------------------------------- |
| Sparsity labels          | `0.125`, `0.25`, `0.375`, `0.50`                                                |
| Random seeds             | `1`, `2`, `3`                                                                   |
| Dataset                  | WikiText2 for evaluation and the same WikiText2 calibration split for train PPL |
| Sequence length          | `2048`                                                                          |
| Calibration tokens       | `8192`                                                                          |
| Attention implementation | `sdpa`                                                                          |
| Dtype                    | `float16`                                                                       |
| Removal counts           | derive from matching EvoPress final config                                      |

**Expected output files**

- `scripts/evaluate_depth_baselines.py`
- `results/depth_baseline_runs.csv`
- `outputs/experiments/baseline_<method>_mistral7b_s<value>_seed<seed>/layer_drop_config.txt`
- `outputs/experiments/baseline_<method>_mistral7b_s<value>_seed<seed>/run.log`
- one row per baseline run in `results/experiment_log.csv`

Required baseline table columns:

```text
sparsity,method,seed,wikitext2_ppl,train_ppl,runtime_minutes,notes,output_dir
```

**Metric to record**

- WikiText2 PPL
- train PPL
- runtime
- dropped attention count
- dropped MLP count
- seed
- whether layer `0` was protected

**Success/failure criteria**

- Success: random and late-layer results exist for all four sparsity labels, random has three seeds per level, and each baseline matches the EvoPress attention/MLP removal counts.
- Partial success: early-layer results are omitted or documented as infeasible; this does not block the main comparison.
- Failure: baselines use a different removal budget or evaluation setup from EvoPress.

**Small Codex prompt**

> Inspect `evo_drop_search.py`, `drop_scoring.py`, `eval_ppl.py`, and `src/model_utils.py`, then adapt minimally. Implement a cheap baseline evaluator for random, late-layer, and optional early-layer attention/MLP dropping. For each Mistral depth-grid config, derive and match the exact attention and MLP removal counts, evaluate WikiText2 and train PPL, save configs and logs, append experiment-log rows, and run random seeds `1 2 3`.

## 6. Experiment group D: repeatability / seed robustness

### Task 6.1: Run three seeds at `37.5%`

**Objective**

Measure how sensitive the main depth-pruning result is to the evolutionary-search seed.

**Method/script to run**

Run:

```text
MODEL=mistralai/Mistral-7B-v0.3
SPARSITY=0.375
GENERATIONS=10
OFFSPRING=8
SEEDS=1 2 3
```

Reuse the Task 3.1 seed-`1` run if it has exactly the same parameters. Add seed-`2` and seed-`3` runs. Do not rerun seed `1` unnecessarily.

Create `scripts/summarize_seed_robustness.py` to parse final configs into dropped module sets represented as:

```text
(layer_index, module_type)
```

where `module_type` is `attn` or `mlp`. Report pairwise Jaccard overlap:

```text
intersection_size / union_size
```

**Model**

`mistralai/Mistral-7B-v0.3`

**Parameters**

| Parameter      | Value                        |
| -------------- | ---------------------------- |
| Sparsity       | `0.375`                      |
| Generations    | `10`                         |
| Offspring      | `8`                          |
| Seeds          | `1`, `2`, `3`                |
| Other settings | exactly the same as Task 3.1 |

**Expected output files**

- `outputs/experiments/depth_mistral7b_s0.375_seed1/`
- `outputs/experiments/depth_mistral7b_s0.375_seed2/`
- `outputs/experiments/depth_mistral7b_s0.375_seed3/`
- `scripts/summarize_seed_robustness.py`
- `results/seed_robustness_table.md`
- one row per run in `results/experiment_log.csv`

**Metric to record**

- final WikiText2 PPL by seed
- final train PPL by seed
- runtime by seed
- dropped module set by seed
- mean and standard deviation of final WikiText2 PPL
- best and worst seed
- pairwise Jaccard overlap between selected dropped-module sets

**Success/failure criteria**

- Success: three completed seeds with finite PPL values and a generated robustness summary.
- Failure: only PPL is recorded and the selected configs are lost, preventing overlap analysis.

**Small Codex prompt**

> Inspect the parameterized depth launcher and existing completed runs, then adapt minimally. Reuse the matching seed-`1` Mistral `0.375` run if available, run seeds `2` and `3`, and create a small summarizer that reports PPL mean, standard deviation, best/worst seed, runtime, dropped configs, and pairwise Jaccard overlap of `(layer_index, module_type)` removals.

## 7. Experiment group E: small-model sparse database feasibility test

### Task 7.1: Generate a minimal sparse database on TinyLlama

**Objective**

Verify that the SparseGPT/FastOBC database pipeline works end-to-end on a smaller compatible model before spending more time on Mistral-7B.

**Method/script to run**

Do not immediately retry full Mistral-7B sparse generation. Start from `scripts/run_sparse_gpt.sh`, inspect `prune.py`, and create a minimal TinyLlama launcher.

Use:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CALIB_DATA=wikitext2
SEQUENCE_LENGTH=1024
CALIB_TOKENS=4096
SPARSITY=0.50
NUM_LEVELS=3
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
```

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` is selected because this repository already contains TinyLlama debug launchers for depth and quantization. If model compatibility differs in the actual container, use `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` after inspecting the model utility assumptions.

Stage the sparse test:

1. Smoke test only `q_proj` modules using the existing `'.*layers.*q_proj$'` pattern.
2. If successful and RAM remains below the limit, expand to the projection regex supported by the existing scripts.
3. Use `--low_cpu_mem_usage`, `--cpu_offload_modules`, and `--cpu_offload_activations`, but monitor RSS because CPU offload can increase RAM pressure.
4. Use only `NUM_LEVELS=3` for the first useful feasibility test.

Suggested launcher shape:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --master_port 29511 prune.py \
  --model_name_or_path "$MODEL" \
  --prunable_modules '.*layers.*q_proj$' \
  --pre_block_modules model.embed_tokens \
  --block_modules model.layers \
  --calibration_data "$CALIB_DATA" \
  --calibration_tokens "$CALIB_TOKENS" \
  --calibration_sequence_length "$SEQUENCE_LENGTH" \
  --sparsity "$SPARSITY" \
  --num_levels "$NUM_LEVELS" \
  --low_cpu_mem_usage \
  --cpu_offload_modules \
  --cpu_offload_activations \
  --verbose \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --dtype "$DTYPE" \
  --save_dir "$SAVE_DIR"
```

Allow `prune.py` to derive `weights_diff` unless inspection shows that an explicit smaller value is necessary.

**Model**

Primary: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Fallback: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

**Parameters**

| Parameter                | Value                                                   |
| ------------------------ | ------------------------------------------------------- |
| Sparsity                 | `0.50`                                                  |
| Calibration data         | `wikitext2`                                             |
| Calibration tokens       | `4096` initially; increase to `8192` only after success |
| Sequence length          | `1024` initially; use `2048` only after success         |
| Levels                   | `3`                                                     |
| Module scope             | `q_proj` smoke test first                               |
| Dtype                    | `float16`                                               |
| Attention implementation | `sdpa`                                                  |

**Expected output files**

- `scripts/run_sparse_gpt_tiny_debug.sh`
- `outputs/experiments/sparse_db_tinyllama_qproj_s0.50/run.log`
- `outputs/experiments/sparse_db_tinyllama_qproj_s0.50/runtime.txt`
- sparse database directory containing `metadata.pth`
- sparse database layer directories containing `<level>.pth` files
- one row in `results/experiment_log.csv`

**Metric to record**

- pipeline completion status
- runtime
- maximum observed CPU RAM/RSS if available
- maximum observed GPU VRAM if available
- generated database size on disk
- last successful layer/module if failed
- full error text in `run.log`

**Success/failure criteria**

- Success: `metadata.pth` and expected `q_proj` level files are generated for the selected model.
- Partial success: generation advances through some layers but fails; preserve the last successful module and memory evidence.
- Failure: the process is retried with larger settings before the first failure is documented.

**Small Codex prompt**

> Inspect `scripts/run_sparse_gpt.sh`, `prune.py`, and the TinyLlama scripts, then adapt minimally. Create and run a TinyLlama SparseGPT feasibility launcher at `50%` sparsity with WikiText2, `4096` calibration tokens, sequence length `1024`, `3` levels, `float16`, and `sdpa`. Start with `q_proj` only, monitor memory, preserve the full log, and append a completed or failed experiment-log row. Do not retry full Mistral-7B generation.

### Task 7.2: Run a minimal sparse search if database generation succeeds

**Objective**

Demonstrate the full sparse pipeline: database generation followed by non-uniform sparse allocation search and WikiText2 evaluation.

**Method/script to run**

Only run this task if Task 7.1 completes. Start from `scripts/run_prune_search.sh`, inspect the generated database layout, and adapt the launcher for TinyLlama and the reduced dataset.

Use a small search:

```text
GENERATIONS=20
OFFSPRING=8
CALIB_DATA=wikitext2
CALIB_TOKENS=4096
SEQUENCE_LENGTH=1024
EVAL_DATASETS=wikitext2
FITNESS_FN=kl
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
```

If runtime is low and the path is stable, increase to `50` generations as a second clearly labeled run.

**Model**

The exact TinyLlama model that completed Task 7.1.

**Parameters**

| Parameter                | Value                 |
| ------------------------ | --------------------- |
| Sparse database          | output of Task 7.1    |
| Generations              | `20`; optionally `50` |
| Offspring                | `8`                   |
| Calibration data         | `wikitext2`           |
| Calibration tokens       | `4096`                |
| Sequence length          | `1024`                |
| Fitness                  | `kl`                  |
| Dtype                    | `float16`             |
| Attention implementation | `sdpa`                |

**Expected output files**

- `scripts/run_prune_search_tiny_debug.sh`
- `outputs/experiments/sparse_search_tinyllama_s0.50/run.log`
- `outputs/experiments/sparse_search_tinyllama_s0.50/runtime.txt`
- final sparse configuration text written into or copied from the sparse database directory
- one row in `results/experiment_log.csv`

**Metric to record**

- final WikiText2 PPL
- generation-wise WikiText2 PPL where available
- train fitness
- runtime
- selected sparse configuration

**Success/failure criteria**

- Success: search completes, writes a final sparse config, and reports finite WikiText2 PPL.
- Failure: search is attempted without a complete database or its config cannot be mapped back to generated levels.

**Small Codex prompt**

> Inspect `scripts/run_prune_search.sh`, `evo_prune_search.py`, and the completed TinyLlama sparse database, then adapt minimally. Run a reduced `20`-generation sparse search with `8` offspring and WikiText2 settings from `experiments_next_7_days.md`. Save the selected sparse config, logs, runtime, WikiText2 PPL, and an experiment-log row.

## 8. Experiment group F: optional small-model quantization pipeline test

### Task 8.1: Run TinyLlama GPTQ database generation

**Objective**

Show that the quantization pipeline structure is understood and, if resources permit, executable on a smaller model.

**Method/script to run**

Use the existing `scripts/run_gptq_tiny_debug.sh` as the starting point. It already targets TinyLlama `q_proj` modules and candidate bits `2 3 4`. Run this only after the primary depth-pruning results and Task 7 sparse feasibility work are preserved.

Start with the existing smoke-test scale:

```text
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
SEQUENCE_LENGTH=128
CALIB_DATA=wikitext2
CALIB_TOKENS=512
BITS_LIST=2 3 4
BITS_TO_LOAD=3
GROUP_SIZE=128
MODULE_SCOPE=q_proj
DTYPE=float16
ATTN_IMPLEMENTATION=sdpa
```

If successful, schedule a second clearly labeled run with `CALIB_TOKENS=4096` and `SEQUENCE_LENGTH=1024`.

**Model**

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Parameters**

Candidate bits `2`, `3`, `4`; calibration bitwidth `3`; group size `128`; `q_proj` only for the first pass; `float16`; `sdpa`.

**Expected output files**

- existing or minimally adapted `scripts/run_gptq_tiny_debug.sh`
- `outputs/experiments/quant_db_tinyllama_qproj/run.log`
- `outputs/experiments/quant_db_tinyllama_qproj/runtime.txt`
- quant database under `outputs/quant_db_tiny/TinyLlama-1.1B-Chat-v1.0/3bit/`
- one row in `results/experiment_log.csv`

**Metric to record**

- completion status
- runtime
- peak CPU RSS if available
- peak GPU VRAM if available
- database size on disk
- last successful layer/module if failed

**Success/failure criteria**

- Success: the expected bitwidth files exist for the selected `q_proj` layers.
- Partial success: a failure is preserved with the last successful module and full log.
- Failure: the optional quant run delays completion of the primary depth-pruning report.

**Small Codex prompt**

> Inspect `scripts/run_gptq_tiny_debug.sh`, `quant.py`, and the generated output layout, then adapt minimally. Run the TinyLlama `q_proj` GPTQ smoke test with bits `2 3 4`, group size `128`, `float16`, and `sdpa`. Preserve logs, runtime, memory evidence, output size, and an experiment-log row.

### Task 8.2: Run a minimal quantization search

**Objective**

Complete the optional smaller-model quantization path by selecting a mixed-bit configuration and evaluating WikiText2 PPL.

**Method/script to run**

Only run this after Task 8.1 succeeds. Start from the existing `scripts/run_quant_search_tiny_interesting.sh`, which already points at the TinyLlama database and uses a mixed target bitwidth.

Use:

```text
TARGET_BITWIDTH=3.5
GENERATIONS=20
OFFSPRING=16
GROUP_RULE=none
STEP_SIZE=1
FITNESS_FN=kl
```

Keep the run small. The purpose is pipeline feasibility, not a paper-scale quantization result.

**Model**

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Parameters**

Use the output of Task 8.1, target average bitwidth `3.5`, bits available `2 3 4`, group size inherited from the database, `20` generations, `16` offspring, WikiText2, `float16`, and `sdpa`.

**Expected output files**

- minimally adapted `scripts/run_quant_search_tiny_interesting.sh`
- `outputs/experiments/quant_search_tinyllama_b3.5/run.log`
- `outputs/experiments/quant_search_tinyllama_b3.5/runtime.txt`
- final quantization configuration text
- one row in `results/experiment_log.csv`

**Metric to record**

- final WikiText2 PPL
- train fitness
- runtime
- selected per-layer bitwidth config

**Success/failure criteria**

- Success: search completes, writes a selected config, and reports finite WikiText2 PPL.
- Failure: quant search is run before the database is complete or the selected config is not preserved.

**Small Codex prompt**

> Inspect `scripts/run_quant_search_tiny_interesting.sh`, `evo_quant_search.py`, and the completed TinyLlama GPTQ database, then adapt minimally. Run a small mixed-bit search around target bitwidth `3.5`, preserve the selected config, log WikiText2 PPL and runtime, and append an experiment-log row.

## 9. Hardware bottleneck documentation

### Task 9.1: Capture hardware snapshot and write bottleneck note

**Objective**

Document why full Mistral-7B sparse database generation is deferred and make the limitation reproducible.

**Method/script to run**

Create `scripts/capture_hardware_snapshot.sh` or run the equivalent commands manually. Save the complete output in `results/hardware_snapshot.txt`.

Required commands:

```bash
nvidia-smi
free -h
df -h .
cat /sys/fs/cgroup/memory.max
python - <<'PY'
from pathlib import Path

raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
if raw == "max":
    print("cgroup memory.max: unlimited")
else:
    print(f"cgroup memory.max bytes: {raw}")
    print(f"cgroup memory.max GiB: {int(raw) / 1024**3:.2f}")
PY
python - <<'PY'
import torch

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"cuda total memory GiB: {props.total_memory / 1024**3:.2f}")
PY
```

Write `results/hardware_bottleneck_summary.md` after inspecting the snapshot and the preserved Mistral sparse-generation failure log. State evidence and inference separately.

**Model**

Not applicable.

**Parameters**

Record the capture date, container identity if available, and whether the commands were run inside the same Jupyter/container environment used for experiments.

**Expected output files**

- `scripts/capture_hardware_snapshot.sh` if scripted
- `results/hardware_snapshot.txt`
- `results/hardware_bottleneck_summary.md`
- preserved failing Mistral sparse-generation log if available

**Metric to record**

- GPU name
- total and used VRAM
- system-visible RAM
- cgroup `memory.max`
- disk free space
- PyTorch CUDA availability
- sparse-generation failure signature

**Success/failure criteria**

- Success: the snapshot demonstrates approximately `45 GB` GPU VRAM and the `16 GB` cgroup/container memory limit, and the markdown note carefully explains the likely CPU RAM bottleneck.
- Failure: the note claims an out-of-memory root cause without preserving either memory-limit evidence or a failure log.

**Small Codex prompt**

> Inspect the existing failed sparse-generation logs if present, then adapt minimally. Create a hardware snapshot script that records `nvidia-smi`, `free -h`, `df -h .`, cgroup `memory.max`, its GiB conversion, and a PyTorch CUDA check. Run it in the experiment container and write `results/hardware_bottleneck_summary.md`, separating direct evidence from the CPU-RAM-pressure inference.

## 10. Result plots and tables

### Task 10.1: Build reproducible report artifacts

**Objective**

Generate meeting-ready CSV files, plots, and tables directly from logged experiment artifacts.

**Method/script to run**

Create `scripts/build_experiment_report.py`. Use Python standard-library CSV handling and `matplotlib` only. Do not use seaborn.

Inputs:

- `results/experiment_log.csv`
- `results/depth_baseline_runs.csv`
- depth-search `generation_metrics.csv` files
- final layer-drop configs
- sparse/quant feasibility logs and experiment rows

Generate:

1. `results/depth_pruning_curve.csv`
2. `results/depth_pruning_curve.png`
   - x-axis: depth sparsity label
   - y-axis: WikiText2 PPL
   - lines: EvoPress, random baseline mean, late-layer baseline
   - include random-baseline standard-deviation error bars
   - optionally include dense reference as a horizontal dashed line
3. `results/convergence_37_5.csv`
4. `results/convergence_37_5.png`
   - x-axis: generation
   - y-axis: PPL
   - lines: WikiText2 PPL and train PPL
5. `results/baseline_comparison_table.md`
6. `results/seed_robustness_table.md`
7. `results/small_model_feasibility_summary.md`

Do not silently discard failed runs. Show them in the feasibility summary and keep them in `experiment_log.csv`, but exclude non-completed runs from numeric aggregates.

**Model**

Not applicable.

**Parameters**

Use deterministic plot ordering:

```text
0.125,0.25,0.375,0.50
```

Use the `20`-generation `37.5%` run for the convergence plot when available; otherwise use the completed `10`-generation run and state that limitation.

**Expected output files**

- `scripts/build_experiment_report.py`
- `results/depth_pruning_curve.csv`
- `results/depth_pruning_curve.png`
- `results/convergence_37_5.csv`
- `results/convergence_37_5.png`
- `results/baseline_comparison_table.md`
- `results/seed_robustness_table.md`
- `results/small_model_feasibility_summary.md`

**Metric to record**

- number of included completed runs
- number of excluded failed/skipped runs
- random baseline mean and standard deviation
- seed robustness mean and standard deviation

**Success/failure criteria**

- Success: rerunning one script regenerates all plots and tables from tracked artifacts.
- Failure: results require manual spreadsheet edits or seaborn is introduced.

**Small Codex prompt**

> Inspect `results/experiment_log.csv`, baseline CSVs, generation CSVs, and saved configs, then adapt minimally. Create one matplotlib-only report builder that generates the depth curve, `37.5%` convergence plot, baseline table, seed robustness table, and small-model feasibility summary required by `experiments_next_7_days.md`. Exclude failed runs from aggregates but report them explicitly.

## 11. Final meeting summary document

### Task 11.1: Write thesis-progress summary

**Objective**

Create a concise, evidence-based summary suitable for a master-thesis supervision meeting.

**Method/script to run**

Create `results/meeting_summary.md` after the report artifacts are generated. Pull exact numbers from tracked CSV files and link to the generated plots and tables.

Required sections:

1. Scope and experimental question
2. Environment and hardware constraints
3. Successfully completed experiments
4. Main Mistral-7B depth-pruning curve
5. Dense and simple-baseline comparison
6. `37.5%` convergence behavior
7. Three-seed repeatability result
8. Small-model SparseGPT feasibility result
9. Optional small-model GPTQ feasibility result
10. Limitations
11. Next hardware requirement estimate
12. Next experimental steps

For the next hardware estimate, distinguish measured evidence from recommendation. The repository README warns that sparse and quant databases can consume `100-200 GB` of disk. For RAM, report the observed peak or failure boundary first. If only the current `16 GB` limit and an OOM failure are known, recommend a measured retry on a container with at least `32 GB`, with `64 GB` RAM preferred for full Mistral-7B database generation, and revise that estimate after collecting peak-RSS evidence.

**Model**

Not applicable.

**Parameters**

Use thesis-progress style: factual, compact, and explicit about incomplete experiments. Do not overstate causal claims.

**Expected output files**

- `results/meeting_summary.md`

**Metric to record**

Completeness: every summary claim should be traceable to a CSV row, plot, table, hardware snapshot, or preserved log.

**Success/failure criteria**

- Success: the document communicates what worked, what failed, why the failure is likely hardware-related, and what experiment should happen next.
- Failure: approximate preliminary values are mixed with final measured values without labels.

**Small Codex prompt**

> Inspect all generated CSV files, plots, tables, hardware notes, and failed-run logs, then adapt minimally. Create `results/meeting_summary.md` in a formal thesis-progress style. Use exact tracked metrics, distinguish measured evidence from inference, include limitations, and provide an evidence-based next hardware estimate.

## 12. Execution order

Use the following order. Depth-pruning results are the required deliverable. Sparse and quantization work must not displace them.

### Day 1: Tracking, hardware, and verified reference run

- Create `results/experiment_log.csv` and `scripts/append_experiment_log.py`.
- Create and validate `scripts/parse_depth_search_log.py`.
- Capture `results/hardware_snapshot.txt`.
- Write the first version of `results/hardware_bottleneck_summary.md`.
- Archive any existing `37.5%`, five-generation Mistral log.
- Rerun or verify the existing reduced `37.5%` depth-pruning setup.
- Run the exact dense WikiText2 reference evaluation.

**Day 1 exit condition:** logging is reliable, hardware constraints are documented, and at least one Mistral depth run has a preserved config and parsed generation metrics.

### Day 2: Lower-sparsity Mistral depth runs

- Run Mistral depth pruning at `12.5%`.
- Run Mistral depth pruning at `25.0%`.
- Parse logs and append experiment rows immediately after each run.

**Day 2 exit condition:** both lower-sparsity points are reproducible from saved commands and logs.

### Day 3: Higher-sparsity runs and convergence extension

- Run or finalize the `37.5%`, ten-generation grid point.
- Run the `50.0%` grid point.
- Start the fresh `37.5%`, twenty-generation convergence run.
- Schedule `30` generations only if the `20`-generation runtime is acceptable.

**Day 3 exit condition:** the four-point EvoPress curve is available or every missing point has a documented failure log.

### Day 4: Baselines

- Implement `scripts/evaluate_depth_baselines.py`.
- Run three random seeds per sparsity.
- Run late-layer baselines per sparsity.
- Run the optional early-layer negative control if time remains.
- Generate the first baseline table.

**Day 4 exit condition:** EvoPress can be compared fairly against random and late-layer dropping with matched attention/MLP removal counts.

### Day 5: Seed robustness

- Reuse the matching `37.5%`, seed-`1` grid run.
- Run `37.5%` depth pruning with seeds `2` and `3`.
- Generate PPL mean, standard deviation, best/worst seed, and config-overlap summary.

**Day 5 exit condition:** one main EvoPress setting has a three-seed robustness result.

### Day 6: Smaller-model database pipelines

- Run TinyLlama SparseGPT `q_proj` feasibility generation.
- If successful, run the reduced sparse search.
- If the required depth results are complete and time remains, run TinyLlama GPTQ generation.
- If GPTQ generation succeeds, run the reduced mixed-bit quantization search.

**Day 6 exit condition:** sparse feasibility is documented as completed or failed with evidence. Quantization is optional.

### Day 7: Plots, tables, and meeting summary

- Generate all plots and tables with `scripts/build_experiment_report.py`.
- Review `results/experiment_log.csv` for missing metadata.
- Finalize `results/hardware_bottleneck_summary.md`.
- Create `results/meeting_summary.md`.
- Verify that every plot and claim can be traced to saved artifacts.

**Day 7 exit condition:** the meeting summary is ready and the underlying artifacts are reproducible.

## 13. Include per-task Codex prompts

The short prompts embedded in each task are sufficient for incremental work. The ready-to-copy prompts below group the work into practical Codex sessions. Every prompt asks Codex to inspect the current repository first and adapt minimally because exact script interfaces may change as the week progresses.

### Prompt 13.1: Logging CSV and parser

```text
Inspect the existing scripts and adapt minimally. Read experiments_next_7_days.md, evo_drop_search.py, and scripts/run_drop_search.sh. Create results/experiment_log.csv with the exact requested columns, scripts/append_experiment_log.py, and scripts/parse_depth_search_log.py. The parser must extract generation, train fitness, WikiText2 PPL, train PPL, and parent attention/MLP masks from stdout. Add focused tests or fixture-based validation. Do not run model experiments.
```

### Prompt 13.2: Hardware snapshot

```text
Inspect the existing repo and any preserved SparseGPT failure logs, then adapt minimally. Create and run a hardware snapshot script that writes results/hardware_snapshot.txt with nvidia-smi, free -h, df -h ., /sys/fs/cgroup/memory.max, a GiB conversion, and a PyTorch CUDA check. Create results/hardware_bottleneck_summary.md. Separate direct evidence from the inference that SparseGPT database generation fails because of CPU/container RAM pressure.
```

### Prompt 13.3: Mistral depth-pruning grid

```text
Inspect scripts/run_drop_search.sh, evo_drop_search.py, and the logging helpers, then adapt minimally. Parameterize the launcher and run sequential Mistral-7B depth-pruning experiments at sparsities 0.125, 0.25, 0.375, and 0.50 using WikiText2, sequence length 2048, calibration tokens 8192, generations 10, offspring 8, initially_generated 16, initial_tokens 512, survivors_per_selection 2 1, tokens_per_selection 512 2048, fitness_fn kl, sdpa, float16, and seed 1. Save command.sh, run.log, runtime.txt, final config, parsed generation metrics, and one experiment-log row per run. Do not change the search algorithm.
```

### Prompt 13.4: Convergence extension

```text
Inspect the current parameterized depth launcher and evo_drop_search.py, then adapt minimally. Run a fresh Mistral-7B depth-pruning experiment at sparsity 0.375 for 20 generations with offspring 8 and otherwise the exact same reduced WikiText2 settings as the main grid. Preserve generation-wise WikiText2 PPL, train PPL, train fitness, parent masks, final config, runtime, log, and experiment-log row. Only add minimal per-generation config snapshots if stdout is insufficient.
```

### Prompt 13.5: Dense, random, and late-layer baselines

```text
Inspect eval_ppl.py, evo_drop_search.py, drop_scoring.py, and src/model_utils.py, then adapt minimally. First run and log one exact dense Mistral-7B WikiText2 PPL evaluation. Then implement scripts/evaluate_depth_baselines.py for random, late-layer, and optional early-layer dropping. For each EvoPress grid config, derive and match its exact attention and MLP removal counts. Run random seeds 1, 2, and 3, run one late-layer baseline per sparsity, save configs and logs, write results/depth_baseline_runs.csv, and append experiment-log rows.
```

### Prompt 13.6: Seed robustness

```text
Inspect the parameterized depth launcher and completed grid runs, then adapt minimally. Reuse the matching Mistral-7B sparsity-0.375 seed-1 run if available, run seeds 2 and 3 with generations 10 and offspring 8, and create scripts/summarize_seed_robustness.py. Generate results/seed_robustness_table.md with final WikiText2 PPL, train PPL, runtime, mean, standard deviation, best/worst seed, final dropped configs, and pairwise Jaccard overlap over (layer_index, module_type) dropped-module sets.
```

### Prompt 13.7: Small-model SparseGPT feasibility

```text
Inspect scripts/run_sparse_gpt.sh, prune.py, scripts/run_prune_search.sh, evo_prune_search.py, and the existing TinyLlama launchers, then adapt minimally. Do not retry full Mistral-7B sparse generation. Create and run a TinyLlama q_proj-only SparseGPT feasibility launcher at sparsity 0.50 with WikiText2, 4096 calibration tokens, sequence length 1024, 3 levels, float16, and sdpa. Monitor memory, preserve the log, runtime, database size, last successful module on failure, and an experiment-log row. If generation succeeds, run a reduced sparse search for 20 generations with 8 offspring and evaluate WikiText2 PPL.
```

### Prompt 13.8: Optional small-model GPTQ feasibility

```text
Inspect scripts/run_gptq_tiny_debug.sh, quant.py, scripts/run_quant_search_tiny_interesting.sh, and evo_quant_search.py, then adapt minimally. Only proceed after required depth results are preserved. Run the TinyLlama q_proj GPTQ smoke test with bits 2 3 4, calibration bitwidth 3, group size 128, float16, and sdpa. Preserve logs, runtime, memory evidence, output size, and an experiment-log row. If generation succeeds, run a small mixed-bit search at target bitwidth 3.5 for 20 generations and preserve the selected config and WikiText2 PPL.
```

### Prompt 13.9: Plots and tables

```text
Inspect results/experiment_log.csv, results/depth_baseline_runs.csv, all generation_metrics.csv files, saved configs, and feasibility logs, then adapt minimally. Create scripts/build_experiment_report.py using matplotlib only, without seaborn. Generate results/depth_pruning_curve.csv and .png, results/convergence_37_5.csv and .png, results/baseline_comparison_table.md, results/seed_robustness_table.md, and results/small_model_feasibility_summary.md. Exclude failed runs from numeric aggregates but list them explicitly.
```

### Prompt 13.10: Meeting summary

```text
Inspect all generated CSV files, plots, tables, hardware notes, and failed-run logs, then adapt minimally. Create results/meeting_summary.md in a formal master-thesis progress style. Include completed experiments, the Mistral depth-pruning curve, dense and baseline comparison, convergence, three-seed repeatability, sparse and optional quant feasibility, hardware bottleneck evidence, limitations, and an evidence-based next hardware requirement estimate. Use exact tracked values and label preliminary observations clearly.
```
