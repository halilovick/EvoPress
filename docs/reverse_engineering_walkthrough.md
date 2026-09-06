# EvoPress Reverse-Engineering Walkthrough

## Repository map

### Top-level execution files
- `evo_drop_search.py`: evolutionary depth-pruning search.
- `evo_prune_search.py`: evolutionary search over per-layer sparse weight levels.
- `evo_quant_search.py`: evolutionary search over per-layer quantization levels.
- `prune.py`: SparseGPT/FastOBC database builder for unstructured sparsity.
- `quant.py`: GPTQ/FastOBQ database builder for quantization.
- `owl_prune.py`: OWL baseline database builder for non-uniform sparsity.
- `drop_scoring.py`: scoring-based depth-pruning baselines.
- `brute_force_drop.py`: brute-force depth-pruning baseline.
- `compute_layer_errors.py`: estimates per-layer normalized reconstruction errors for DP search.
- `dp_search.py`: dynamic-programming solver over `errors.pth`.
- `eval_ppl.py`: perplexity evaluation for dense, dropped, sparse, or quantized models.
- `lmeval.py`: LM Eval Harness integration with compressed-model loading hooks.

### Driver scripts
- `scripts/run_drop_search.sh`
- `scripts/run_sparse_gpt.sh`
- `scripts/run_prune_search.sh`
- `scripts/run_gptq.sh`
- `scripts/run_quant_search.sh`
- `scripts/run_owl_prune.sh`
- `scripts/run_lmeval.sh`

### Core support modules
- `src/model_utils.py`: model-family adapters, layer selection, dummy forward patching, compressed weight loading.
- `src/data_utils.py`: dataset loading and token chunking.
- `src/metrics.py`: PPL, KL, sparse-KL scoring.
- `src/pruner.py` + `src/fast_obc.py`: sparse artifact generation.
- `src/quantizer.py` + `src/fast_obq.py` + `src/quant_utils.py`: quantized artifact generation.
- `src/error_estimator.py`: layer-wise proxy error computation for DP.
- `src/owl_pruner.py`: OWL baseline implementation.
- `src/common_utils.py`, `src/dist_utils.py`, `src/linalg_utils.py`: utilities used everywhere.

### Shipped artifact examples
- `drop_configs/.../layer_drop_config.txt`: saved depth-pruning configs.
- `pruning_configs/*.txt`: saved sparse-search configs.
- `quantization_configs/*.txt`: saved quant-search configs.

## Real execution paths

### 1. Depth pruning
`scripts/run_drop_search.sh` -> `evo_drop_search.py`
-> `src.data_utils.get_data`
-> `src.model_utils.get_layers`, `get_attn_layer_name`, `get_mlp_layer_name`, `dummy_initialize`, `make_dummy_forward`, `restore_forward`
-> `src.metrics.compute_kl_div` or `compute_perplexity`
-> writes `layer_drop_config.txt` during/after search
-> optional `final_model.pth`
-> evaluation via `eval_ppl.py --drop_layer_config ...` or `lmeval.py --drop_layer_config ...`

Key idea: depth pruning never writes per-layer replacement weights. It mutates module forwards in-memory so dropped attention/MLP blocks become identity/zero behavior.

### 2. Unstructured sparsity
`scripts/run_sparse_gpt.sh` -> `prune.py`
-> `src.pruner.FastOBCPruner`
-> `src.fast_obc.FastOBC`
-> writes sparse weight database:
`<save_dir>/<layer_name>/<level>.pth` plus `metadata.pth`
-> `scripts/run_prune_search.sh` -> `evo_prune_search.py`
-> repeatedly swaps sparse weights into the live model
-> writes final config text into the same sparse database directory
-> evaluation via `eval_ppl.py --sparse_weights_path ... --sparse_config_path ...` or `lmeval.py ...`

Key idea: search state is an integer offset per layer. The offset chooses which precomputed sparse tensor file to load.

### 3. Quantization
`scripts/run_gptq.sh` -> `quant.py`
-> `src.quantizer.Quantizer`
-> `src.fast_obq.FastOBQ`
-> `src.quant_utils.QLinear`
-> writes quantized/dequantized weight database:
`<save_dir>/<model_name>/<calibration_bitwidth>bit/<layer_name>/<bit>.pth`
-> `scripts/run_quant_search.sh` -> `evo_quant_search.py`
-> groups layers, mutates per-layer bitwidth assignments, swaps chosen tensors into the model
-> writes final config text into the quant database directory
-> evaluation via `eval_ppl.py --quant_weights_path ... --quant_config_path ...` or `lmeval.py ...`

Key idea: the search uses already-dequantized tensors on disk, not integer-packed runtime kernels. GPTQ is used to produce the candidates; search/evaluation just reloads float weights.

## Pipeline diagram

```text
Dense HF model
  |
  +-- depth path ----------------------------------------------+
  |  run_drop_search.sh                                         |
  |    -> evo_drop_search.py                                    |
  |    -> patch attn/mlp/block forwards                         |
  |    -> score candidates on calibration minibatches          |
  |    -> save layer_drop_config.txt                            |
  |    -> eval_ppl.py / lmeval.py loads config and patches fwd |
  |
  +-- sparsity path ------------------------------------------------------+
  |  run_sparse_gpt.sh                                                     |
  |    -> prune.py                                                         |
  |    -> FastOBCPruner/FastOBC                                            |
  |    -> save per-layer sparse tensors at multiple levels + metadata.pth  |
  |  run_prune_search.sh                                                   |
  |    -> evo_prune_search.py                                              |
  |    -> mutate integer level vector, load tensors from disk              |
  |    -> save final sparse config txt                                     |
  |    -> eval_ppl.py / lmeval.py reload chosen tensors                    |
  |
  +-- quant path ------------------------------------------------------------------+
     run_gptq.sh                                                                     |
       -> quant.py                                                                   |
       -> Quantizer/FastOBQ/QLinear                                                  |
       -> save per-layer dequantized tensors at candidate bitwidths                  |
     run_quant_search.sh                                                             |
       -> evo_quant_search.py                                                        |
       -> mutate/group bitwidth assignments under average-bit budget                 |
       -> save final quant config txt                                                |
       -> eval_ppl.py / lmeval.py reload chosen tensors                             |
```

## Glossary
- `level`:
  Sparse path: integer offset from the average sparsity level generated by `prune.py`.
  Quant path: actual bitwidth filename, usually `2`, `3`, `4`, etc.
- `weights_diff`: fixed change in number of kept weights between adjacent sparse levels.
- `num_levels`: how many sparse levels above/below the average level to materialize.
- `target_bitwidth`: global average bit budget used by quant search.
- `calibration_bitwidth`: bitwidth whose `QLinear` modules stay in the model while later layers are processed during database generation.
- `group_rule`: quant-search mutation constraint: same-size layers, same suffix name, or no grouping.
- `sparse_kl`: KL computed only on top-K teacher logits to reduce memory.
- `removed_state`: depth-pruning chromosome with boolean `attn` and `mlp` masks.
- `legal_to_drop_path`: optional mask constraining which blocks may be removed.
- `drop_two_consecutive`: search over pairs of layers, represented as one decision and expanded back later.

## Critical files to fully understand
- `README.md`
- `scripts/run_drop_search.sh`
- `scripts/run_sparse_gpt.sh`
- `scripts/run_prune_search.sh`
- `scripts/run_gptq.sh`
- `scripts/run_quant_search.sh`
- `evo_drop_search.py`
- `evo_prune_search.py`
- `evo_quant_search.py`
- `prune.py`
- `quant.py`
- `src/model_utils.py`
- `src/data_utils.py`
- `src/metrics.py`
- `src/pruner.py`
- `src/fast_obc.py`
- `src/quantizer.py`
- `src/fast_obq.py`
- `src/quant_utils.py`

## Details you can safely ignore for the meeting
- `src/losses.py`, `src/optim_utils.py`, `src/prompter.py`: no call sites in this repo snapshot.
- `drop_scoring.py`, `brute_force_drop.py`, `owl_prune.py`, `src/owl_pruner.py`: useful as baselines, not necessary for the main EvoPress mechanism.
- `compute_layer_errors.py` and `dp_search.py`: alternative solver path, not part of the main evolutionary flows.
- `scripts/run_owl_prune.sh`: baseline-only.
- Most shipped `drop_configs/`, `pruning_configs/`, `quantization_configs/` files: examples of output format, not logic.

## File-by-file report

### `README.md`
1. Purpose of file
   Explains the intended workflow, repo layout, datasets, distributed usage, and evaluation interface.
2. Entry points / main functions / classes
   None.
3. Important arguments and defaults
   Documents key CLI flags informally: calibration data choices, 8M-token recommendation, 8k/4k sequence defaults.
4. Inputs consumed
   None.
5. Outputs written
   None.
6. Call graph to other files/functions
   Human-level map to `scripts/`, `evo_*`, `prune.py`, `quant.py`, `lmeval.py`, `eval_ppl.py`.
7. Which concept from the paper this implements
   High-level method decomposition: depth pruning, non-uniform sparsity, non-uniform quantization.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It describes two-stage sparse/quant flows explicitly: database generation first, search second.
9. What would likely need to change to support joint search over multiple compression methods
   README would need a fourth workflow describing a mixed database format and a mixed search state.

### `scripts/run_drop_search.sh`
1. Purpose of file
   Example launcher for evolutionary depth pruning.
2. Entry points / main functions / classes
   Shell entrypoint invoking `python evo_drop_search.py`.
3. Important arguments and defaults
   `SPARSITY=0.375`, `GENERATIONS=k*(n-k)/1.5`, `population_size=1`, `offspring=32`, `fitness_fn=kl`.
4. Inputs consumed
   HF model id, calibration dataset selection, output directory for config.
5. Outputs written
   Whatever `evo_drop_search.py` writes into `--drop_config_dir`.
6. Call graph to other files/functions
   Script -> `evo_drop_search.py:main`.
7. Which concept from the paper this implements
   Evolutionary search over layer/block dropping.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Assumes Mistral has 32 blocks in the generation heuristic.
9. What would likely need to change to support joint search over multiple compression methods
   The launcher would need extra inputs for sparse/quant databases and a mixed budget.

### `scripts/run_sparse_gpt.sh`
1. Purpose of file
   Example launcher for sparse database generation.
2. Entry points / main functions / classes
   Shell entrypoint invoking `torchrun ... prune.py`.
3. Important arguments and defaults
   `SPARSITY=0.7`, `NUM_LEVELS=8`, regex over q/k/v/o/gate/up/down projections, CPU offload on.
4. Inputs consumed
   Dense model, calibration data, pruning hyperparameters.
5. Outputs written
   Sparse weight database under `SAVE_DIR`.
6. Call graph to other files/functions
   Script -> `prune.py:main`.
7. Which concept from the paper this implements
   Preparation stage for EvoPress sparse search.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Uses one fixed average sparsity and generates nearby discrete levels around it.
9. What would likely need to change to support joint search over multiple compression methods
   Could remain as-is if joint search reuses sparse artifacts; otherwise it would need to emit metadata compatible with a joint state space.

### `scripts/run_prune_search.sh`
1. Purpose of file
   Example launcher for evolutionary search over sparse database levels.
2. Entry points / main functions / classes
   Shell entrypoint invoking `python evo_prune_search.py`.
3. Important arguments and defaults
   `generations=400`, `offspring=64`, multi-stage selection budgets, `fitness_fn=kl`.
4. Inputs consumed
   Sparse database path from `prune.py`.
5. Outputs written
   Final sparse config text inside the sparse database directory.
6. Call graph to other files/functions
   Script -> `evo_prune_search.py:main`.
7. Which concept from the paper this implements
   Evolutionary allocation of non-uniform unstructured sparsity.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Comment says “run gptq.sh first”; that is likely stale and should mean `run_sparse_gpt.sh`.
9. What would likely need to change to support joint search over multiple compression methods
   This would become one branch of a mixed search launcher, or its logic would be subsumed by a new joint script.

### `scripts/run_gptq.sh`
1. Purpose of file
   Example launcher for quantization database generation.
2. Entry points / main functions / classes
   Shell entrypoint invoking `torchrun ... quant.py`.
3. Important arguments and defaults
   Candidate bits `2 3 4 5 6`, `BITS_TO_LOAD=3`, `GROUP_SIZE=128`, `perchannel` enabled.
4. Inputs consumed
   Dense model, calibration data, bitwidth candidate list.
5. Outputs written
   Quantized/dequantized layer database under `SAVE_DIR/<model>/<calibration_bitwidth>bit`.
6. Call graph to other files/functions
   Script -> `quant.py:main`.
7. Which concept from the paper this implements
   Preparation stage for EvoPress quantization search.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Search later works on dequantized tensors written here, not low-bit runtime kernels.
9. What would likely need to change to support joint search over multiple compression methods
   Likely only metadata, unless the joint path needs interaction-specific artifacts.

### `scripts/run_quant_search.sh`
1. Purpose of file
   Example launcher for evolutionary bitwidth allocation.
2. Entry points / main functions / classes
   Shell entrypoint invoking `python evo_quant_search.py`.
3. Important arguments and defaults
   `target_bitwidth=3`, `generations=150`, `offspring=128`, staged selection, `fitness_fn=kl`.
4. Inputs consumed
   Quant database path from `quant.py`.
5. Outputs written
   Final quant configuration text in the quant database directory.
6. Call graph to other files/functions
   Script -> `evo_quant_search.py:main`.
7. Which concept from the paper this implements
   Evolutionary non-uniform quantization.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Assumes the provided path already points at the per-model/per-calibration-bitwidth subdirectory.
9. What would likely need to change to support joint search over multiple compression methods
   Needs extra arguments for sparse path and/or drop search constraints plus combined budget handling.

### `scripts/run_owl_prune.sh`
1. Purpose of file
   Launcher for the OWL sparsity baseline.
2. Entry points / main functions / classes
   Shell entrypoint invoking `owl_prune.py`.
3. Important arguments and defaults
   Sweeps `owl_lambda` and `owl_m`.
4. Inputs consumed
   Dense model and calibration data.
5. Outputs written
   OWL sparse database plus `metadata.pth`.
6. Call graph to other files/functions
   Script -> `owl_prune.py:main`.
7. Which concept from the paper this implements
   Baseline comparison for non-uniform sparsity allocation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Uses predefined model-family-specific `PRE_BLOCK_MODULES`.
9. What would likely need to change to support joint search over multiple compression methods
   Probably nothing unless OWL is also extended into a mixed baseline.

### `scripts/run_lmeval.sh`
1. Purpose of file
   Example evaluation command for LM Eval Harness.
2. Entry points / main functions / classes
   Shell entrypoint invoking `lmeval.py`.
3. Important arguments and defaults
   Defaults to quantized evaluation with uniform `DEFAULT_LEVEL=4`.
4. Inputs consumed
   Model alias, compressed weights path, LM Eval tasks.
5. Outputs written
   Whatever `lmeval.py` writes when `--output_path` is used.
6. Call graph to other files/functions
   Script -> `lmeval.py:cli_evaluate`.
7. Which concept from the paper this implements
   Final task evaluation of compressed models.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It only shows one compression mode at a time.
9. What would likely need to change to support joint search over multiple compression methods
   Evaluation interface must allow combined drop+sparse+quant loading instead of a single mutually exclusive option.

### `evo_drop_search.py`
1. Purpose of file
   Runs evolutionary search over which attention and/or MLP sub-blocks are dropped.
2. Entry points / main functions / classes
   `main`, `parse_args`, `selection`, `load_states`, `get_layer_drop_config`.
3. Important arguments and defaults
   `--sparsity` is fraction of blocks removed. `--population_size` default `1`. `--max_mutations=3`. `--drop_entire_block` ties `mlp` to `attn`. `--drop_two_consecutive` halves the decision space by pairing layers. `--legal_to_drop_path` constrains deletions.
4. Inputs consumed
   Dense HF model, tokenizer, calibration/eval datasets, optional legal-mask file.
5. Outputs written
   `layer_drop_config.txt` in `drop_config_dir`; optionally `final_model.pth` and a second config in `save_dir`.
6. Call graph to other files/functions
   `main` -> `get_data` -> `compute target logits` -> `get_layers`/`dummy_initialize` -> `selection` -> `load_states` -> `make_dummy_forward`/`restore_forward` -> `compute_kl_div` or `compute_perplexity`.
7. Which concept from the paper this implements
   Dynamic depth pruning via evolutionary search.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Search chromosome is two boolean arrays (`attn`, `mlp`), not a more abstract block object. Dropping is implemented by monkey-patching forwards, not re-building the model. `drop_two_consecutive` and `legal_to_drop_path` cannot be combined. It supports only model families whose attention/MLP names are known in `src/model_utils.py`.
9. What would likely need to change to support joint search over multiple compression methods
   The chromosome must expand from `{attn mask, mlp mask}` to a joint state carrying depth decisions plus sparse/quant levels. Fitness evaluation would need to both patch forwards and swap per-layer weights before scoring. Saving would need a composite config format rather than only `layer_drop_config.txt`.

### `drop_scoring.py`
1. Purpose of file
   Implements non-evolutionary depth-pruning baselines based on per-layer scores or single-drop evaluations.
2. Entry points / main functions / classes
   `main`, `get_embeddings`, several score functions.
3. Important arguments and defaults
   `--scoring_method` chooses among KL, PPL, cosine similarity, L2, normalized L2, norm ratio, or windowed cosine.
4. Inputs consumed
   Dense model, calibration data, eval datasets.
5. Outputs written
   None; results are printed.
6. Call graph to other files/functions
   `main` -> `get_data` -> `get_layers` -> `make_dummy_forward`/`restore_forward` -> metric functions.
7. Which concept from the paper this implements
   Baseline scoring methods for layer dropping.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It ranks whole blocks via one-shot scores, then evaluates prefixes of that order. It does not save configs.
9. What would likely need to change to support joint search over multiple compression methods
   Probably not worth extending; it is a baseline, not the main framework.

### `brute_force_drop.py`
1. Purpose of file
   Exhaustive baseline over all block-drop combinations of fixed cardinality.
2. Entry points / main functions / classes
   `main`, `load_states`, `compute_fitness`, `get_layer_drop_config`.
3. Important arguments and defaults
   Requires `--drop_entire_block`. Can optionally use `--drop_two_consecutive`.
4. Inputs consumed
   Dense model, calibration data.
5. Outputs written
   None in current implementation.
6. Call graph to other files/functions
   `main` -> `get_data` -> `get_layers`/`dummy_initialize` -> enumerate bitstrings -> `load_states` -> `compute_perplexity`.
7. Which concept from the paper this implements
   Exhaustive small-scale baseline for depth pruning.
8. Any assumptions, shortcuts, or implementation-specific decisions
   `get_layer_drop_config` is buggy here: it initializes `["none" * num_blocks]` instead of `["none"] * num_blocks`, but this helper is not actually used later. `get_data(..., streaming=...)` is also stale because `get_data` does not accept `streaming`.
9. What would likely need to change to support joint search over multiple compression methods
   Exhaustive joint search would be combinatorially intractable almost immediately.

### `evo_prune_search.py`
1. Purpose of file
   Searches over per-layer sparse database levels under a fixed global average sparsity encoded implicitly by the mutation rule.
2. Entry points / main functions / classes
   `main`, `parse_args`, `selection`, `load_layers`.
3. Important arguments and defaults
   `--sparse_weights_path` points to the precomputed database. `--max_level` bounds absolute level index. `--max_total_deviation` bounds L1 distance from uniform level-0 allocation. Multi-stage selection is controlled by `--survivors_per_selection` and `--tokens_per_selection`.
4. Inputs consumed
   Dense model, tokenizer, calibration/eval datasets, directory tree `<layer>/<level>.pth`.
5. Outputs written
   Configuration text `configuration_name` inside `sparse_weights_path`.
6. Call graph to other files/functions
   `main` -> `get_data` -> optional teacher logits -> enumerate subdirectories in sparse DB -> evolutionary loop -> `selection` -> `load_layers` -> `compute_kl_div`/`compute_perplexity`.
7. Which concept from the paper this implements
   Evolutionary non-uniform unstructured sparsity allocation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   The initial parent is always all zeros, meaning “uniform average sparsity from the database builder.” Mutation preserves the sum of levels by decrementing one layer and incrementing another, so global sparsity stays fixed. Search relies on presence/absence of neighboring level files to define feasible mutations.
9. What would likely need to change to support joint search over multiple compression methods
   The level vector would need to coexist with quant levels and drop decisions. Current mutation preserves only one budget dimension; a joint search would need either multiple coupled budgets or a unified cost function. `load_layers` would also need to coordinate with depth drops and quant weights.

### `prune.py`
1. Purpose of file
   Builds the sparse weight database used by `evo_prune_search.py`.
2. Entry points / main functions / classes
   `main`, `parse_args`.
3. Important arguments and defaults
   `--sparsity`, `--weights_diff`, `--num_levels`, `--rel_damp`, `--block_size`, `--prunable_modules`, `--pre_block_modules`, `--block_modules`.
4. Inputs consumed
   Dense model and calibration data.
5. Outputs written
   `metadata.pth` and per-layer sparse tensors under `<save_dir>/<layer_name>/<level>.pth`.
6. Call graph to other files/functions
   `main` -> distributed init -> `get_data` -> `FastOBCPruner.prune`.
7. Which concept from the paper this implements
   Sparse artifact preparation for dynamic non-uniform allocation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   If `weights_diff` is omitted, it is inferred from hidden size and target sparsity, which is an implementation heuristic rather than a paper-level necessity.
9. What would likely need to change to support joint search over multiple compression methods
   Mostly metadata: a joint search still needs these artifacts, but the save directory should record the base sparse budget so a joint searcher can combine it with quant/drop costs consistently.

### `owl_prune.py`
1. Purpose of file
   Builds OWL baseline sparse artifacts.
2. Entry points / main functions / classes
   `main`, `parse_args`.
3. Important arguments and defaults
   `--owl_lambda`, `--owl_m`, `--sparsity`, `--rel_damp`.
4. Inputs consumed
   Dense model and calibration data.
5. Outputs written
   `metadata.pth` and per-layer sparse tensors indexed by OWL candidate distribution id.
6. Call graph to other files/functions
   `main` -> distributed init -> `get_data` -> `OWLFastOBCPruner.prune`.
7. Which concept from the paper this implements
   OWL baseline for non-uniform sparsity.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Saves multiple candidate sparsity distributions per layer, not an evolutionary search output.
9. What would likely need to change to support joint search over multiple compression methods
   Not central unless you want a non-evolutionary joint baseline.

### `evo_quant_search.py`
1. Purpose of file
   Searches over per-layer quantization bitwidth assignments under a target average bit budget.
2. Entry points / main functions / classes
   `main`, `parse_args`, `selection`, `load_layers`.
3. Important arguments and defaults
   `--target_bitwidth` may be integer or fractional. Fractional targets require `--initially_generated` and `--initial_tokens`. `--group_rule` controls mutation scope. `--step_size=1` assumes adjacent candidate filenames differ by 1. `--fitness_fn` can be `ppl`, `kl`, or `sparse_kl`.
4. Inputs consumed
   Dense model, tokenizer, calibration/eval datasets, quant database tree `<layer>/<bit>.pth`.
5. Outputs written
   `evo-<fitness>-configuration-<target_bitwidth>.txt` inside `quant_weights_path`.
6. Call graph to other files/functions
   `main` -> `get_data` -> optional teacher logits/top-K teacher logits -> `layer_order_fn` -> `group_layers` -> evolutionary loop -> `selection` -> `load_layers` -> metrics.
7. Which concept from the paper this implements
   Evolutionary non-uniform quantization.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Search works on dequantized tensors, not packed low-bit modules. Integer targets start from a uniform assignment; fractional targets start from ceil-bit candidates and randomly decrement until the average budget is met. With `group_rule != none`, mutations are budget-neutral within a group. With `group_rule == none`, the code tracks actual parameter counts to preserve total bits approximately. The TODO about unavailable integer levels is real: if you ask for a target bitwidth absent from the database, initialization can fail conceptually.
9. What would likely need to change to support joint search over multiple compression methods
   This is the nearest starting point for joint search because it already reasons about weighted budgets. The state would need extra sparse and/or drop components, and the budget accounting would need to convert all compression choices into a common cost, likely total stored bits or effective FLOPs/latency. `group_rule` may also need method-aware grouping, e.g. only allow certain joint mutations within matching layer types.

### `quant.py`
1. Purpose of file
   Builds the quantized candidate database for later search.
2. Entry points / main functions / classes
   `main`, `parse_args`.
3. Important arguments and defaults
   `--bitwidth_options`, `--calibration_bitwidth`, `--group_size`, `--perchannel`, `--sym`, `--act_order`, `--rel_damp`, `--block_size`.
4. Inputs consumed
   Dense model and calibration data.
5. Outputs written
   Per-layer dequantized tensors under `<save_dir>/<model>/<calibration_bitwidth>bit/<layer>/<bit>.pth`.
6. Call graph to other files/functions
   `main` -> distributed init -> `get_data` -> reorder bitwidth list so calibration bit is last -> `Quantizer.quantize`.
7. Which concept from the paper this implements
   GPTQ-style candidate generation for dynamic bitwidth allocation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   `post_block_modules` is parsed but never used by the instantiated `src.quantizer.Quantizer`; in this repo snapshot it is effectively dead CLI surface. The saved artifacts are dequantized weights recovered from `QLinear.get_weight()`, which simplifies later search but loses the speed/storage properties of true low-bit inference during search/eval.
9. What would likely need to change to support joint search over multiple compression methods
   Again, mostly metadata and directory conventions, unless joint search needs cross-method candidate generation.

### `compute_layer_errors.py`
1. Purpose of file
   Computes per-layer normalized reconstruction error for every saved compression option.
2. Entry points / main functions / classes
   `main`, `parse_args`.
3. Important arguments and defaults
   `--compressed_weights_path`, `--group_by_numel`, plus model/block regex arguments.
4. Inputs consumed
   Dense model, calibration data, compressed weight directory.
5. Outputs written
   `errors.pth` in the compressed weights directory.
6. Call graph to other files/functions
   `main` -> distributed init -> `get_data` -> `ErrorEstimator.estimate` -> `torch.save(errors)`.
7. Which concept from the paper this implements
   Alternative DP-based search pipeline, not the main evolutionary method.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It uses normalized quadratic proxy error, not direct end-to-end KL/PPL.
9. What would likely need to change to support joint search over multiple compression methods
   Would need a joint error model across methods, which is much harder because drop, sparsity, and quantization interactions are not additive in a simple per-layer proxy.

### `dp_search.py`
1. Purpose of file
   Solves a knapsack-like allocation problem over `errors.pth`.
2. Entry points / main functions / classes
   `DPSolver`, `main`, `parse_args`.
3. Important arguments and defaults
   `--target_cost`, `--is_sparsity`, `--configuration_name`.
4. Inputs consumed
   `errors.pth` and the saved weight filenames in each layer directory.
5. Outputs written
   A config text file in the compressed weights directory.
6. Call graph to other files/functions
   `main` -> load `errors.pth` -> derive costs from filenames -> `DPSolver.solve` -> save config.
7. Which concept from the paper this implements
   Dynamic-programming alternative/ablation solver.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It treats costs as additive across layers and reads costs from filenames. For sparsity it flips sign because sparse “levels” are saved as negative/positive offsets around the base level.
9. What would likely need to change to support joint search over multiple compression methods
   A simple additive DP is unlikely to capture interactions between methods unless you collapse all methods into one per-layer option table, which explodes the option count.

### `lmeval.py`
1. Purpose of file
   Adapts LM Eval Harness to load one compression mode before evaluation.
2. Entry points / main functions / classes
   `parse_eval_args`, `load_compressed_weights`, `cli_evaluate`.
3. Important arguments and defaults
   Compression modes are mutually exclusive: `--drop_layer_config` or `--sparse_weights_path` or `--quant_weights_path`.
4. Inputs consumed
   Normal LM Eval args plus compressed configs/weights.
5. Outputs written
   Optional JSON/JSONL evaluation results and optional W&B logs.
6. Call graph to other files/functions
   `cli_evaluate` monkey-patches `AutoModelForCausalLM.from_pretrained` -> loads compression -> calls `lm_eval.evaluator.simple_evaluate`.
7. Which concept from the paper this implements
   Benchmark evaluation of the final compressed model.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It hard-codes mutual exclusivity of compression methods. It monkey-patches global `from_pretrained`, which is pragmatic but brittle.
9. What would likely need to change to support joint search over multiple compression methods
   Remove the exclusivity assertion and apply depth dropping plus sparse loading plus quant loading in a deterministic order.

### `eval_ppl.py`
1. Purpose of file
   Perplexity-only evaluation utility for one compression mode.
2. Entry points / main functions / classes
   `parse_args`, `load_compressed_weights`, `main`.
3. Important arguments and defaults
   Same mutually exclusive compression choices as `lmeval.py`; `--memory_efficient` switches to layer-by-layer evaluation.
4. Inputs consumed
   Dense model, evaluation datasets, optional compression config/weights.
5. Outputs written
   None besides logs/W&B.
6. Call graph to other files/functions
   `main` -> load model -> apply chosen compression (`drop_layers_from_config` or `load_compressed_weights`) -> `get_data` -> `compute_perplexity` or `compute_perplexity_layer_per_layer`.
7. Which concept from the paper this implements
   Perplexity evaluation of final compressed models.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Same one-method-only assumption as `lmeval.py`.
9. What would likely need to change to support joint search over multiple compression methods
   Same as `lmeval.py`: allow combined application of all selected compression artifacts.

### `src/common_utils.py`
1. Purpose of file
   Small generic utilities.
2. Entry points / main functions / classes
   `fix_seed`, `to`, `maybe_first_element`.
3. Important arguments and defaults
   `fix_seed` sets Python, NumPy, and Torch seeds and flips deterministic cuDNN.
4. Inputs consumed
   Arbitrary nested tensor structures for `to`.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used across search, preparation, metrics, and blockwise processing.
7. Which concept from the paper this implements
   Not paper-specific; infrastructure.
8. Any assumptions, shortcuts, or implementation-specific decisions
   `to` recursively maps tensors in tuples/lists/dicts/dataclasses, which is what makes the blockwise input-caching code convenient.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing essential.

### `src/data_utils.py`
1. Purpose of file
   Loads and chunks calibration/evaluation datasets.
2. Entry points / main functions / classes
   `get_data`, `get_wikitext2`, `get_fineweb_edu`, `get_c4`, `collect_samples_with_join`.
3. Important arguments and defaults
   `fineweb_edu` is handled by token count; `wikitext2` and `c4` are handled by sample count derived from `num_tokens // sequence_length`.
4. Inputs consumed
   HF datasets or a local `.pt` file of tokenized samples.
5. Outputs written
   None.
6. Call graph to other files/functions
   Called by every search/build/eval entrypoint.
7. Which concept from the paper this implements
   Calibration and evaluation data preparation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   The code explicitly warns that `collect_samples_with_join` biases toward shorter examples. `fineweb_edu` uses half the dataset for train and half for eval. C4 revision is pinned; FineWeb is not pinned in code.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing structural.

### `src/dist_utils.py`
1. Purpose of file
   Minimal wrappers around torch distributed utilities.
2. Entry points / main functions / classes
   `is_main`, `get_rank`, `get_world_size`, `gather_into_tensor`, `print_on_main`.
3. Important arguments and defaults
   Defaults to single-process behavior if distributed is unavailable/uninitialized.
4. Inputs consumed
   Distributed process group state.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used by `prune.py`, `quant.py`, `owl_prune.py`, and the blockwise backend modules.
7. Which concept from the paper this implements
   Multi-GPU execution support.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Very thin wrapper; no fancy sharding logic.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing unless joint search itself becomes distributed.

### `src/fast_obc.py`
1. Purpose of file
   Implements the per-layer Optimal Brain Compression style sparse reconstruction used by `prune.py`.
2. Entry points / main functions / classes
   `FastOBC` with `update`, `pruning_pre_step`, `step`, `prune`.
3. Important arguments and defaults
   `rel_damp` regularizes Hessian; `block_size` controls column-chunk processing.
4. Inputs consumed
   Layer input activations and target sparsity list.
5. Outputs written
   Returns sparse weight tensors; caller saves them.
6. Call graph to other files/functions
   Called by `src.pruner.FastOBCPruner`.
7. Which concept from the paper this implements
   One-shot sparse candidate generation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Computes a full Hessian approximation over input channels, broadcasts only final sparse weights, and zeroes columns with zero Hessian diagonal. Score thresholding is done per layer option list, not globally across layers.
9. What would likely need to change to support joint search over multiple compression methods
   Likely nothing if reused as a database builder; joint search would consume its outputs.

### `src/linalg_utils.py`
1. Purpose of file
   Stable symmetric matrix inversion helper.
2. Entry points / main functions / classes
   `inv_sym`.
3. Important arguments and defaults
   None.
4. Inputs consumed
   Symmetric positive-definite matrix.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used by `fast_obc.py` and `fast_obq.py`.
7. Which concept from the paper this implements
   Numerical support for Hessian-based compression.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Assumes Cholesky succeeds after damping.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing.

### `src/losses.py`
1. Purpose of file
   Defines feature-matching loss.
2. Entry points / main functions / classes
   `square_head_loss`.
3. Important arguments and defaults
   `eps=1e-6`.
4. Inputs consumed
   Teacher/student feature dicts.
5. Outputs written
   None.
6. Call graph to other files/functions
   No call sites in this repo snapshot.
7. Which concept from the paper this implements
   None in the executed paths.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Likely leftover from experiments.
9. What would likely need to change to support joint search over multiple compression methods
   Not relevant unless a future gradient-based fine-tuning stage is added.

### `src/pruner.py`
1. Purpose of file
   Orchestrates blockwise sparse database generation across the model.
2. Entry points / main functions / classes
   `FastOBCPruner.prune`.
3. Important arguments and defaults
   Receives regexes, module path names, save dir, offload options, and FastOBC hyperparameters.
4. Inputs consumed
   Cached block inputs, dense model, pruning target settings.
5. Outputs written
   Saves sparse tensors per layer/level.
6. Call graph to other files/functions
   `prune` -> `InputCollector` -> forward hooks -> `FastOBC.update` -> `_prune_group` -> `FastOBC.prune`.
7. Which concept from the paper this implements
   Layerwise sparse candidate generation at scale.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It processes blocks sequentially and feeds each block’s compressed outputs into the next block, so later-layer candidates are calibrated on already-compressed earlier blocks.
9. What would likely need to change to support joint search over multiple compression methods
   If joint artifact generation is desired, this sequential compressed-forward pattern is the piece you would reuse.

### `src/quant_utils.py`
1. Purpose of file
   Low-level quantization helpers and `QLinear`.
2. Entry points / main functions / classes
   `Quantizer`, `QLinear`, pack/unpack helpers.
3. Important arguments and defaults
   `Quantizer.configure` controls bits, per-channel, symmetric/asymmetric behavior.
4. Inputs consumed
   Float weights or activations for parameter fitting/quantization.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used by `src.fast_obq` and `src.quantizer`.
7. Which concept from the paper this implements
   Quantization primitive for GPTQ-style candidate generation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   `QLinear.forward` reconstructs float weights on the fly. In this repo, the important use is `get_weight()`, because saved search artifacts are dequantized float tensors.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing fundamental unless joint inference must preserve packed low-bit kernels.

### `src/fast_obq.py`
1. Purpose of file
   Implements GPTQ/FastOBQ quantization over a layer for several candidate bitwidths.
2. Entry points / main functions / classes
   `FastOBQ` with `update`, `quantization_pre_step`, `step`, `quantize`.
3. Important arguments and defaults
   `bitwidth_options`, `perchannel`, `group_size`, `sym`, `rel_damp`, `block_size`, `act_order`.
4. Inputs consumed
   Layer input activations and weight matrix.
5. Outputs written
   Returns quantized weights/scales/zeros; caller saves artifacts.
6. Call graph to other files/functions
   Called by `src.quantizer.Quantizer`.
7. Which concept from the paper this implements
   Candidate generation for dynamic quantization.
8. Any assumptions, shortcuts, or implementation-specific decisions
   If `act_order` is set, columns are permuted by Hessian diagonal magnitude. Groupwise quantization statistics are recomputed every `group_size` columns.
9. What would likely need to change to support joint search over multiple compression methods
   Probably nothing as a standalone artifact generator.

### `src/quantizer.py`
1. Purpose of file
   Orchestrates blockwise quantization database generation across the model.
2. Entry points / main functions / classes
   `Quantizer.quantize`.
3. Important arguments and defaults
   Receives regexes/module paths, save dir, offload flags, and a dict of `FastOBQ` kwargs.
4. Inputs consumed
   Dense model, cached block inputs, bitwidth options.
5. Outputs written
   Per-layer candidate tensors `<layer>/<bit>.pth`.
6. Call graph to other files/functions
   `quantize` -> `InputCollector` -> forward hooks -> `FastOBQ.update` -> `_quant_group` -> `FastOBQ.quantize` -> `QLinear.get_weight`.
7. Which concept from the paper this implements
   Multi-bit candidate generation for quantization search.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Like `src.pruner.py`, it pushes already-quantized outputs forward when moving to later layers. It also replaces each processed layer with `QLinear` at the calibration bitwidth so downstream Hessian estimation sees compressed upstream activations.
9. What would likely need to change to support joint search over multiple compression methods
   If joint search remains database-based, this probably stays unchanged.

### `src/error_estimator.py`
1. Purpose of file
   Measures normalized proxy reconstruction error of saved compressed tensors.
2. Entry points / main functions / classes
   `LayerErrorEstimator`, `ErrorEstimator.estimate`.
3. Important arguments and defaults
   `group_by_numel` optionally groups layers of equal size for DP.
4. Inputs consumed
   Dense model, cached inputs, compressed weight files.
5. Outputs written
   None directly; caller saves `errors.pth`.
6. Call graph to other files/functions
   `estimate` -> `InputCollector` -> hooks -> `LayerErrorEstimator.update/pre_step/estimate`.
7. Which concept from the paper this implements
   Proxy scoring for the DP baseline.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Uses normalized quadratic error relative to Hessian, not end-to-end eval.
9. What would likely need to change to support joint search over multiple compression methods
   A joint proxy would require modeling cross-method interactions, which this file does not.

### `src/owl_pruner.py`
1. Purpose of file
   OWL baseline backend.
2. Entry points / main functions / classes
   `OWLUtil`, `OWLFastOBCPruner.prune`.
3. Important arguments and defaults
   `owl_m`, `owl_lambda`.
4. Inputs consumed
   Dense model and calibration activations.
5. Outputs written
   Per-layer sparse tensors for several OWL-derived sparsity distributions.
6. Call graph to other files/functions
   `prune` -> collect Hessian diagonal proxy -> derive per-block sparsity distributions -> `FastOBC.prune`.
7. Which concept from the paper this implements
   OWL baseline.
8. Any assumptions, shortcuts, or implementation-specific decisions
   It first estimates outlier ratios, then converts them into candidate sparsity distributions before pruning.
9. What would likely need to change to support joint search over multiple compression methods
   Not central.

### `src/prompter.py`
1. Purpose of file
   Prompt-template helper.
2. Entry points / main functions / classes
   `Prompter`, `ZeroPrompter`.
3. Important arguments and defaults
   Alpaca template by default.
4. Inputs consumed
   Instruction/input/label strings.
5. Outputs written
   None.
6. Call graph to other files/functions
   No call sites in this repo snapshot.
7. Which concept from the paper this implements
   None in executed compression paths.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Leftover utility.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing.

### `src/metrics.py`
1. Purpose of file
   Defines fitness and evaluation metrics.
2. Entry points / main functions / classes
   `compute_perplexity`, `compute_kl_div`, `compute_sparse_kl_div`, `compute_perplexity_layer_per_layer`.
3. Important arguments and defaults
   Batch size defaults to `1`. `compute_sparse_kl_div` expects top-K teacher logits and indices.
4. Inputs consumed
   Live model, tokenized samples, optional teacher logits.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used by all search/eval entrypoints.
7. Which concept from the paper this implements
   Fitness functions and final perplexity evaluation.
8. Any assumptions, shortcuts, or implementation-specific decisions
   `compute_kl_div` chunk-slices the sequence dimension in steps of 1024 to reduce memory. The running average weights by element counts, not by sample count.
9. What would likely need to change to support joint search over multiple compression methods
   Probably nothing; the joint search would still call one of these metrics.

### `src/model_utils.py`
1. Purpose of file
   Central adapter layer between repo logic and Hugging Face model internals.
2. Entry points / main functions / classes
   Model-family getters, dummy/restore helpers, `drop_layers`, `drop_layers_from_config`, `InputCollector`, `select_layers`, `load_sparse_weights`, `layer_order_fn`, `group_layers`.
3. Important arguments and defaults
   `group_layers(..., group_rule)` accepts `none`, `name`, `size`. `drop_layers` understands config tokens `none`, `mlp`, `attn`, `attn+mlp`.
4. Inputs consumed
   Live model, module names, drop config files, sparse config files.
5. Outputs written
   None.
6. Call graph to other files/functions
   Used almost everywhere: drop search, evaluation, sparse builder, quant builder.
7. Which concept from the paper this implements
   Compression-application plumbing.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Model support is narrow. `get_attn_layer_name` and `get_mlp_layer_name` only support Llama/Mistral for depth pruning. `drop_layers` uses zero/identity module replacement, whereas search uses forward monkey-patching for temporary evaluation.
9. What would likely need to change to support joint search over multiple compression methods
   This is one of the main extension points. A joint search needs a single “apply mixed compression state” function here that can: drop layers/subblocks, load sparse tensors, and load quant tensors in a defined order.

### `src/optim_utils.py`
1. Purpose of file
   Optimizer-masking helpers.
2. Entry points / main functions / classes
   `wrap_optimizer`, `unwrap_optimizer`.
3. Important arguments and defaults
   None.
4. Inputs consumed
   Optimizer and mask tensors.
5. Outputs written
   None.
6. Call graph to other files/functions
   No call sites in this repo snapshot.
7. Which concept from the paper this implements
   None in executed paths.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Leftover experimental helper.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing.

### `src/__init__.py`
1. Purpose of file
   Empty package marker.
2. Entry points / main functions / classes
   None.
3. Important arguments and defaults
   None.
4. Inputs consumed
   None.
5. Outputs written
   None.
6. Call graph to other files/functions
   Enables `from src import ...`.
7. Which concept from the paper this implements
   None.
8. Any assumptions, shortcuts, or implementation-specific decisions
   Empty.
9. What would likely need to change to support joint search over multiple compression methods
   Nothing.

## Most important implementation conclusions

1. The repo is organized around two distinct phases for sparsity and quantization:
   database generation first, search second.
2. The evolutionary search code never recomputes compression candidates.
   It only mutates a lightweight state and reloads already-saved tensors or dummy forwards.
3. Depth pruning is architecturally different from the other two methods.
   It has no weight database; its state is structural and is applied by altering module behavior.
4. Sparse search preserves a fixed global sparsity budget by exchanging `+1` and `-1` level moves between layers.
5. Quant search preserves a target average bit budget, either exactly within groups or approximately by parameter-count accounting when `group_rule=none`.
6. Evaluation currently assumes one compression method at a time.
   That exclusivity is the first obvious blocker for a joint-search extension.

## What likely changes for joint search

### Minimal viable extension
- Introduce a joint chromosome:
  for each layer or block, hold `{drop_state, sparse_level, quant_level}` or a method-activation subset.
- Add one state-application function in `src/model_utils.py`:
  apply drops first, then load sparse/quant tensors for surviving modules.
- Define one common budget:
  total stored bits is the cleanest unifying metric in the current implementation.
- Update `lmeval.py` and `eval_ppl.py` to remove the one-method-only assertion and load mixed states.

### Hard parts
- Depth pruning acts at block/subblock granularity, while sparse/quant search act on linear submodules.
- A dropped block makes downstream sparse/quant choices inside that block irrelevant.
- Existing sparse levels are centered around one base sparsity; existing quant levels are actual absolute bitwidths.
- The current mutation operators preserve one budget dimension at a time; joint search needs coordinated mutations.
- The current saved config formats are method-specific text files. A joint search needs a composite config schema.

### Cleanest design direction
- Keep the preparation stage separate:
  continue generating sparse and quant candidate databases independently.
- Build a new `evo_joint_search.py`:
  load both databases plus optional depth constraints.
- Put the mixed-state application logic in `src/model_utils.py`, not inside the new search file.
- Add a new evaluation loader shared by `eval_ppl.py` and `lmeval.py` so the same mixed config is used everywhere.

## Uncertainties
- `post_block_modules` is parsed in `quant.py` but not used by `src.quantizer.Quantizer`. This looks stale rather than intentional, but I cannot prove the original intent without the paper appendix or prior commits.
- Some baseline files (`brute_force_drop.py`) contain stale code paths that do not match current helper signatures. They appear non-critical because they are not on the main EvoPress execution path.
- The repo stores dequantized tensors for search/eval. I can state that from the code; I cannot tell whether the paper’s reported runtime numbers used a separate packed-kernel inference path outside this repo.
