# Interaction-Aware Joint Mutation

This note documents the new optional interaction-aware mutation mode for EvoPress joint depth-pruning plus quantization search.

## Code Path

The current joint search is implemented in:

- `evo_joint_search.py`

The launcher most useful for smoke tests is:

- `scripts/run_joint_search_tiny.sh`

Despite its name, the launcher is parameterized and can run TinyLlama or Mistral by overriding environment variables.

## Current Candidate Representation

One joint-search candidate is a Python dictionary:

```python
{
    "drop": {
        "attn": [...],
        "mlp": [...],
    },
    "quant": [[...], ...],
}
```

- `drop["attn"]` and `drop["mlp"]` are boolean masks. `True` means the corresponding attention or MLP sub-block is skipped.
- `quant` stores the bitwidth assignment for the quantized projection modules grouped by `group_layers(...)`.
- A candidate is evaluated after applying both states to the model.

## Old Mutation Behavior

The default behavior is unchanged and remains selected by:

```bash
--joint_mutation_mode standard
```

In this mode, each offspring is copied from the current parent and then usually receives one of two mutation types:

- `depth`: mutate the depth mask with `mutate_drop_state(...)`
- `quantization`: mutate the quantization assignment with `mutate_quant_state(...)`

If `--active_quant_budget` is enabled and a depth mutation changes which quantized modules are active, the quantization state is repaired with `repair_active_quant_budget(...)`.

The important point is that even in the old mode, selection evaluates the final combined depth-pruned and quantized candidate. However, the proposed mutation itself usually changes only one side at a time.

## New Interaction-Aware Mutation

The new mode is selected by either CLI spelling:

```bash
--joint_mutation_mode interaction_aware
```

or:

```bash
--joint-mutation-mode interaction_aware
```

The new mutation is implemented in:

- `mutate_interaction_aware_candidate(...)` in `evo_joint_search.py`

The operator does the following:

1. Copy the current joint candidate.
2. Mutate the depth mask using the same budget-preserving swap logic as the standard depth mutation.
3. Identify which transformer layer indices were touched by the depth-mask change.
4. Repair the active quantization budget using `repair_active_quant_budget(...)`.
5. Try a quantization exchange involving one of the touched layers.
6. If no touched-layer exchange is possible, fall back to any valid active quantization exchange.
7. Return the new candidate plus mutation metadata.

This mode requires:

```bash
--active_quant_budget --group_rule size
```

The reason is that the new operator is explicitly about active-module interaction: dropped modules should not consume the active quantization budget.

## Why This Is A Real Implementation Extension

The old default search already evaluated combined compressed models, but the proposal step usually changed either pruning or quantization independently.

The new operator changes both compression components in one coordinated mutation:

- the depth mask changes first,
- the active quantization budget is repaired for the new active set,
- the quantization assignment is then adjusted, preferably on a layer touched by the depth change.

This makes the mutation operator interaction-aware rather than only the fitness evaluation being interaction-aware.

## Logging

The run summary now records:

- `compression_config.joint_mutation_mode`

Each `generation_log.csv` row stores the mode and mutation diagnostics inside the JSON `mutation_summary` field:

- `joint_mutation_mode`
- `generated_offspring_by_type`
- `depth_mask_entries_changed_by_type`
- `quant_assignments_changed_by_type`
- `depth_mask_entries_changed_total`
- `quant_assignments_changed_total`
- `interaction_aware_details`

For interaction-aware runs, `interaction_aware_details` includes:

- `budget_repair_quant_changes`
- `preferred_quant_exchanges_used`
- `fallback_quant_exchanges_used`

These fields allow comparison against the standard joint search without parsing raw stdout.

## Assumptions

- Active quantization budgeting uses `group_rule=size`.
- The target average bitwidth must be representable within each active size group.
- Projection modules are mapped back to transformer layer indices through names containing `.layers.<index>.`.
- If a touched-layer quantization exchange is not possible, the operator falls back to a valid active exchange elsewhere.

## Risks And Limitations

- This is a conservative operator, not proof that interaction-aware mutation is better.
- It can still propose candidates that selection rejects.
- If the quantization scope has few active modules or limited available bitwidths, the quantization exchange may be constrained.
- The current implementation optimizes the search proposal mechanism, not the low-level inference kernel or real latency.
- Results must be compared against matched standard runs with the same model, seed, generations, offspring, calibration data, and quantization database.

## Smoke Test Commands

Do not report results until these commands actually run on Datalab.

### A. TinyLlama Quick Smoke Test

Purpose: verify that the new mode runs end-to-end and writes structured logs.

```bash
RUN_ID=debug_interaction_aware_tiny_g2_o4_seed0 \
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_tinyllama_qproj_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit \
DROP_SPARSITY=0.125 \
TARGET_BITWIDTH=3.0 \
CALIB_TOKENS=1024 \
SEQUENCE_LENGTH=512 \
EVAL_TOKENS=1024 \
GENERATIONS=2 \
OFFSPRING=4 \
INITIALLY_GENERATED=8 \
INITIAL_TOKENS=128 \
SURVIVORS_PER_SELECTION="2 1" \
TOKENS_PER_SELECTION="128 512" \
GROUP_RULE=size \
ACTIVE_QUANT_BUDGET=1 \
JOINT_MUTATION_MODE=interaction_aware \
SEED=0 \
bash scripts/run_joint_search_tiny.sh
```

Expected output files:

- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/command.sh`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/run.log`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/generation_log.csv`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/generation_metrics.csv`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/run_summary.json`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/final_candidate.json`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/joint_drop_config.txt`
- `outputs/experiments/debug_interaction_aware_tiny_g2_o4_seed0/joint_quant_config.txt`
- one row in `results/experiment_log.csv`

Compare against a matched standard run by changing only:

```bash
RUN_ID=debug_standard_tiny_g2_o4_seed0
JOINT_MUTATION_MODE=standard
```

Metrics to compare:

- final WikiText2 PPL
- final train PPL
- final calibration KL
- `estimated_compression_ratio`
- `mutation_summary.generated_offspring_by_type`
- `mutation_summary.quant_assignments_changed_total`

### B. Mistral `q_proj` Small Run

Purpose: test the new mode on the main thesis model with a reduced budget.

```bash
RUN_ID=debug_interaction_aware_mistral_qproj_s0.25_g5_o8_seed0 \
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
DROP_SPARSITY=0.25 \
TARGET_BITWIDTH=3.0 \
CALIB_TOKENS=1024 \
SEQUENCE_LENGTH=512 \
EVAL_TOKENS=4096 \
GENERATIONS=5 \
OFFSPRING=8 \
INITIALLY_GENERATED=16 \
INITIAL_TOKENS=128 \
SURVIVORS_PER_SELECTION="4 2 1" \
TOKENS_PER_SELECTION="128 512 1024" \
GROUP_RULE=size \
ACTIVE_QUANT_BUDGET=1 \
JOINT_MUTATION_MODE=interaction_aware \
SEED=0 \
bash scripts/run_joint_search_tiny.sh
```

Compare against a matched standard run by changing only:

```bash
RUN_ID=debug_standard_mistral_qproj_s0.25_g5_o8_seed0
JOINT_MUTATION_MODE=standard
```

Metrics to compare:

- final WikiText2 PPL
- final train PPL
- final calibration KL
- final dropped attention and MLP modules
- final bitwidth histogram
- generation-wise `best_search_fitness`
- generation-wise accepted parent replacement rate

### C. Optional Mistral Attention Small Run

Purpose: test whether the operator also works with the broader attention quantization database.

```bash
RUN_ID=debug_interaction_aware_mistral_attention_s0.25_g3_o4_seed0 \
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit \
DROP_SPARSITY=0.25 \
TARGET_BITWIDTH=3.0 \
CALIB_TOKENS=1024 \
SEQUENCE_LENGTH=512 \
EVAL_TOKENS=4096 \
GENERATIONS=3 \
OFFSPRING=4 \
INITIALLY_GENERATED=8 \
INITIAL_TOKENS=128 \
SURVIVORS_PER_SELECTION="2 1" \
TOKENS_PER_SELECTION="128 512" \
GROUP_RULE=size \
ACTIVE_QUANT_BUDGET=1 \
JOINT_MUTATION_MODE=interaction_aware \
SEED=0 \
bash scripts/run_joint_search_tiny.sh
```

Compare against a matched standard run by changing only:

```bash
RUN_ID=debug_standard_mistral_attention_s0.25_g3_o4_seed0
JOINT_MUTATION_MODE=standard
```

Metrics to compare:

- final WikiText2 PPL
- final train PPL
- final calibration KL
- estimated compression ratio
- active average bitwidth
- whether interaction-aware offspring are ever selected as parent

