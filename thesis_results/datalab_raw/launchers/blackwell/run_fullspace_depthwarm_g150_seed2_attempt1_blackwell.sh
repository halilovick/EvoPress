#!/usr/bin/env bash
set -euo pipefail

# DW (depth-to-joint warm-start, s=0.25) seed 2, adapted for the Blackwell
# SLURM node. Identical search-defining flags to
# thesis_results/datalab_raw/launchers/run_fullspace_depthwarm_g150_seed2_attempt1.sh
# (the DataLab/A40 version); only repo path, env activation, and
# quant_weights_path/output_dir have changed.
#
# PREREQUISITE 1: run_prepare_quant_db_fullspace_blackwell_attempt1.sh must
# have completed first (224-module quant DB).
# PREREQUISITE 2: --stage1_run_dir points at results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2,
# which is already committed to the repo (results/runs/...), so a plain
# `git pull` on this node is enough -- no extra transfer needed for this one.
#
# Queue this AFTER the LX seed 2 run above finishes (or run it on a second,
# separate allocation if you have simultaneous GPU capacity). Expect to
# need checkpoint-resume across multiple 12h SLURM sessions, same pattern
# as the LX seed2 resume script.

cd ~/src/evopress

micromamba run -n evopress \
  python evo_joint_search.py \
  --model_name_or_path mistralai/Mistral-7B-v0.3 \
  --calibration_data fineweb_edu \
  --calibration_tokens 524288 \
  --calibration_sequence_length 8192 \
  --eval_datasets wikitext2 c4 \
  --eval_every 5 \
  --eval_tokens 524288 \
  --eval_sequence_length 8192 \
  --drop_sparsity 0.25 \
  --quant_weights_path "$HOME/evopress_quant_stage1_blackwell_fullspace_20261028/Mistral-7B-v0.3/3bit" \
  --target_bitwidth 3 \
  --group_rule size \
  --compression_budget_mode match_uniform_quantization_total \
  --allow_exact_budget_ablation \
  --quantization_group_size 128 \
  --budget_include_quantization_metadata \
  --budget_scale_bits 16 \
  --budget_zero_point_bits 16 \
  --budget_dense_dtype_bits 16 \
  --expected_dense_model_bits 115968376832 \
  --expected_target_cost_bits 26982023168 \
  --expected_bitwidths 2 3 4 5 6 \
  --expected_quantized_modules 224 \
  --joint_mutation_mode standard \
  --generations 150 \
  --offspring 128 \
  --initially_generated 1 \
  --initial_tokens 2048 \
  --skip_initial_single_candidate_evaluation \
  --survivors_per_selection 16 4 1 \
  --tokens_per_selection 2048 16384 131072 \
  --population_size 1 \
  --crossover_probability 0.0 \
  --crossover_type component \
  --crossover_parent_selection uniform \
  --dtype float16 \
  --attn_implementation flash_attention_2 \
  --seed 2 \
  --sequential_mode depth_to_joint_warm \
  --stage1_run_dir ~/src/evopress/results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2 \
  --max_initialization_attempts 100000 \
  --max_offspring_attempts 10000 \
  --output_dir "$HOME/evopress_extension_results/fullspace_depthwarm_g150/fullspace_depthwarm_g150_seed2_attempt1"
