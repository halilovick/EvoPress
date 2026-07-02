# Replay-Based Joint Compression Attribution Framework

## Purpose

This addition provides a post-hoc attribution tool for EvoPress joint depth-pruning + quantization runs. It does not run a new evolutionary search. Instead, it recombines an already discovered depth mask with an already discovered quantization assignment, replays the combined compressed model, and evaluates the result.

The main thesis question this helps answer is:

> Did the joint search improve the depth mask, the quantization assignment, or only the final combination?

## Files Added

- `evo_joint_attribution.py`
  - main replay and attribution tool
  - supports one recombination or a full depth-source x quant-source matrix
  - writes candidate, metrics, compression metrics, and summary artifacts

- `scripts/aggregate_attribution.py`
  - aggregates one or more attribution matrix CSV files
  - intended for multi-seed summaries

- `results/attribution_framework_report.md`
  - this report

## Existing Code Reused

The tool reuses existing EvoPress implementation paths rather than duplicating model surgery:

- `evo_joint_search.py`
  - `apply_joint_state`
  - `get_layer_drop_config`
  - `repair_active_quant_budget`
  - `candidate_bits`
  - `quantizable_weights`

- `src.model_utils`
  - model layer discovery
  - attention/MLP module naming
  - quantization group construction

- `src.run_reporting`
  - final candidate formatting
  - depth-mask details
  - compression metric computation

- `src.data_utils` and `src.metrics`
  - dataset loading
  - perplexity evaluation

## What the Tool Does

For each recombination:

1. Load a depth candidate.
2. Load a quantization candidate, or create a uniform quantization assignment such as `uniform:3`.
3. Validate that the depth mask fits the model layer count.
4. Validate that the quantization assignment matches the modules present in the quantization database.
5. Optionally check the active quantization budget.
6. Optionally repair the active bitwidth budget using the same repair helper as joint search.
7. Replay the combined candidate with `apply_joint_state`.
8. Evaluate perplexity on one or more datasets.
9. Write structured artifacts.

## Supported Inputs

Depth sources:

- `final_candidate.json` from a depth or joint run
- `joint_config.json` from a joint run
- `layer_drop_config.txt`

Quantization sources:

- `final_candidate.json` from a quant or joint run
- `joint_config.json` from a joint run
- `quant_configuration.txt`
- `uniform:<bits>`, for example `uniform:3`

## Output Files

Single recombination mode writes:

- `combined_candidate.json`
- `combined_drop_config.txt`
- `combined_quant_config.txt`
- `final_candidate.json`
- `metrics.json`
- `compression_metrics.json`
- `attribution_summary.json`
- `attribution_summary.csv`
- `summary.md`

Batch matrix mode writes:

- `attribution_matrix.csv`
- `attribution_matrix.json`
- `attribution_matrix.md`
- one subdirectory under `combinations/` per depth-source x quant-source pair

The aggregation helper writes:

- aggregate CSV
- aggregate Markdown summary

## Active Quantization Budget Handling

When `--active_quant_budget` is enabled, the tool computes the parameter-weighted average bitwidth over active searched weights only. Dropped modules do not count toward this active average.

If the active average bitwidth does not match `--target_bitwidth`, the default behavior is to fail the recombination. If `--repair_active_budget` is also set, the tool calls the existing EvoPress `repair_active_quant_budget` helper and records:

- whether the budget was valid before repair
- whether a repair was applied
- how many bitwidth assignments changed
- active average bitwidth before and after repair

This is important because a depth mask from one run and a quantization assignment from another run can otherwise produce an unfair or invalid active quantization budget.

## Why This Is Useful for the Thesis

The previous independent-vs-joint comparison gives final results, but it does not fully isolate why one result is better. This framework tests controlled recombinations:

- independent depth mask + independent quant assignment
- joint depth mask + independent quant assignment
- independent depth mask + joint quant assignment
- joint depth mask + joint quant assignment
- joint depth mask + uniform 3-bit quantization

If a joint depth mask performs well with several quantization assignments, that supports the interpretation that joint search discovered a quantization-compatible pruning mask. If only the exact joint pair performs well, the benefit may come from a stronger interaction between both parts.

## Datalab Commands

Run these from the repository root on Datalab after pulling the implementation.

### 0. Environment Check

```bash
git pull
pip install -r requirements.txt

python - <<'PY'
import datasets
import torch
import transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
PY
```

### 1. q_proj Smoke Replay

This checks one recombination with a small WikiText2 evaluation budget.

```bash
python evo_joint_attribution.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --quant_db outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
  --depth_source results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json \
  --quant_source results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0/final_candidate.json \
  --depth_label independent_depth_seed0 \
  --quant_label independent_quant_seed0 \
  --output_dir results/attribution/smoke_mistral_qproj_seed0 \
  --eval_datasets wikitext2 \
  --eval_tokens 4096 \
  --sequence_length 512 \
  --target_bitwidth 3.0 \
  --active_quant_budget \
  --repair_active_budget \
  --dtype float16 \
  --attn_implementation sdpa \
  --use_fast_tokenizer
```

Expected main outputs:

- `results/attribution/smoke_mistral_qproj_seed0/attribution_summary.csv`
- `results/attribution/smoke_mistral_qproj_seed0/summary.md`
- `results/attribution/smoke_mistral_qproj_seed0/final_candidate.json`

Metric to compare:

- WikiText2 PPL
- estimated compression ratio
- active average bitwidth
- whether repair was needed

### 2. q_proj Seed-0 Attribution Matrix

This builds a controlled recombination matrix using existing Mistral q_proj runs.

```bash
python evo_joint_attribution.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --quant_db outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
  --depth_sources \
    independent=results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json \
    standard_joint=results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0/final_candidate.json \
    interaction_aware=results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed0/final_candidate.json \
  --quant_sources \
    independent=results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0/final_candidate.json \
    standard_joint=results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0/final_candidate.json \
    interaction_aware=results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed0/final_candidate.json \
    uniform3=uniform:3 \
  --output_dir results/attribution/mistral_qproj_seed0 \
  --eval_datasets wikitext2 c4 fineweb_edu \
  --eval_tokens 131072 \
  --sequence_length 1024 \
  --target_bitwidth 3.0 \
  --active_quant_budget \
  --repair_active_budget \
  --dtype float16 \
  --attn_implementation sdpa \
  --use_fast_tokenizer
```

Expected main outputs:

- `results/attribution/mistral_qproj_seed0/attribution_matrix.csv`
- `results/attribution/mistral_qproj_seed0/attribution_matrix.md`
- `results/attribution/mistral_qproj_seed0/combinations/*/summary.md`

Metric to compare:

- Row-wise WikiText2, C4, and FineWeb-Edu PPL
- which depth source performs best across quant sources
- which quant source performs best across depth sources
- whether interaction-aware depth or quant decisions generalize across recombinations

### 3. q_proj Three-Seed Attribution Matrix

Run the same matrix for seeds 0, 1, and 2.

```bash
for seed in 0 1 2; do
  python evo_joint_attribution.py \
    --base_model mistralai/Mistral-7B-v0.3 \
    --quant_db outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
    --depth_sources \
      independent=results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}/final_candidate.json \
      standard_joint=results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/final_candidate.json \
      interaction_aware=results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/final_candidate.json \
    --quant_sources \
      independent=results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed${seed}/final_candidate.json \
      standard_joint=results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/final_candidate.json \
      interaction_aware=results/runs/thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/final_candidate.json \
      uniform3=uniform:3 \
    --output_dir results/attribution/mistral_qproj_seed${seed} \
    --eval_datasets wikitext2 c4 fineweb_edu \
    --eval_tokens 131072 \
    --sequence_length 1024 \
    --target_bitwidth 3.0 \
    --active_quant_budget \
    --repair_active_budget \
    --dtype float16 \
    --attn_implementation sdpa \
    --use_fast_tokenizer
done
```

Then aggregate:

```bash
python scripts/aggregate_attribution.py \
  --inputs \
    results/attribution/mistral_qproj_seed0/attribution_matrix.csv \
    results/attribution/mistral_qproj_seed1/attribution_matrix.csv \
    results/attribution/mistral_qproj_seed2/attribution_matrix.csv \
  --output results/attribution/mistral_qproj_aggregate.csv
```

Expected main outputs:

- `results/attribution/mistral_qproj_aggregate.csv`
- `results/attribution/mistral_qproj_aggregate.md`

### 4. Optional Attention-Scope Attribution Matrix

Run this only if the attention quantization database exists and enough time is available.

```bash
python evo_joint_attribution.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --quant_db outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit \
  --depth_sources \
    independent=results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json \
    attention_joint=results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/final_candidate.json \
  --quant_sources \
    attention_independent=results/runs/thesis_attention_quant_mistral_attention3.0_g20_o16_seed0/final_candidate.json \
    attention_joint=results/runs/thesis_attention_g50_joint_mistral_s0.25_attention3.0_g50_o16_seed0/final_candidate.json \
    uniform3=uniform:3 \
  --output_dir results/attribution/mistral_attention_seed0 \
  --eval_datasets wikitext2 c4 fineweb_edu \
  --eval_tokens 131072 \
  --sequence_length 1024 \
  --target_bitwidth 3.0 \
  --active_quant_budget \
  --repair_active_budget \
  --dtype float16 \
  --attn_implementation sdpa \
  --use_fast_tokenizer
```

## Sync Commands After Datalab Runs

```bash
git add evo_joint_attribution.py scripts/aggregate_attribution.py results/attribution_framework_report.md results/attribution
git commit -m "Add joint compression attribution framework results"
git push
```

If `results/attribution` is too large, commit only CSV/JSON/MD summary files first and leave full per-combination artifacts untracked.

## Suggested Presentation Slide Text

### Slide: Attribution Framework

- Goal: identify whether joint search improves the depth mask, the quantization assignment, or only their exact pairing.
- Method: replay controlled recombinations of existing candidates.
- Example matrix:
  - independent depth + independent quant
  - joint depth + independent quant
  - independent depth + joint quant
  - joint depth + joint quant
  - joint depth + uniform 3-bit
- Evaluation uses the final replayed compressed model, not search fitness alone.

### Slide: How I Will Interpret It

- If joint depth works well with several quant assignments, joint search likely found a quantization-compatible depth mask.
- If joint quant works well with several depth masks, the quant assignment is likely robust.
- If only the exact joint pair works well, the benefit is pair-specific.
- If uniform 3-bit beats searched bit allocation, the bit-search operator needs improvement.

## Limitations

- This framework is post-hoc. It does not prove causality by itself.
- Recombined candidates may require active-budget repair, so repair counts must be reported.
- Runtime can be high because each combination is replayed and evaluated.
- Quantization scope must match the quantization database. A q_proj assignment cannot be replayed against an attention-scope database unless the required module names match.

## Current Status

The implementation is ready for Datalab runs. No attribution results are claimed in this document until the commands above are executed and committed.
