#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/evopress-crossover-v3

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
  --quant_weights_path /home/jovyan/evopress_quant_stage1_1gpu_diskspill_attempt3_20260819/Mistral-7B-v0.3/3bit \
  --target_bitwidth 3 \
  --group_rule size \
  --compression_budget_mode match_uniform_quantization_total \
  --quantization_group_size 128 \
  --budget_include_quantization_metadata \
  --budget_scale_bits 16 \
  --budget_zero_point_bits 16 \
  --budget_dense_dtype_bits 16 \
  --expected_dense_model_bits 115968376832 \
  --expected_target_cost_bits 26982023168 \
  --expected_bitwidths 2 3 4 5 6 \
  --expected_quantized_modules 224 \
  --allow_exact_budget_ablation \
  --joint_mutation_mode interaction_aware \
  --generations 150 \
  --offspring 128 \
  --initially_generated 1 \
  --skip_initial_single_candidate_evaluation \
  --initial_tokens 2048 \
  --survivors_per_selection 16 4 1 \
  --tokens_per_selection 2048 16384 131072 \
  --population_size 1 \
  --crossover_probability 0.0 \
  --crossover_type component \
  --crossover_parent_selection uniform \
  --dtype float16 \
  --attn_implementation flash_attention_2 \
  --seed 0 \
  --sequential_mode none \
  --max_initialization_attempts 100000 \
  --max_offspring_attempts 10000 \
  --output_dir /home/jovyan/evopress_extension_results/fullspace_joint_s025_ia_g150/fullspace_joint_s025_ia_g150_seed0_attempt1
