#!/usr/bin/env bash
set -euo pipefail

# Resume template for LX seed 2 on the Blackwell node. Exact copy of
# run_fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1_blackwell.sh
# plus --resume_checkpoint, same --output_dir (not a new directory).
# Re-run this unchanged after every SLURM timeout until the job reaches
# Generation 150/150 -- the checkpoint is overwritten atomically at the
# end of each generation, so this always resumes from the latest one.

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
  --initially_generated 4 \
  --initial_tokens 2048 \
  --survivors_per_selection 16 4 1 \
  --tokens_per_selection 2048 16384 131072 \
  --population_size 4 \
  --crossover_probability 0.5 \
  --crossover_type local_exchange \
  --crossover_parent_selection uniform \
  --dtype float16 \
  --attn_implementation flash_attention_2 \
  --seed 2 \
  --sequential_mode none \
  --max_initialization_attempts 100000 \
  --max_offspring_attempts 10000 \
  --resume_checkpoint "$HOME/evopress_extension_results/fullspace_joint_s025_localxover_pop4_g150/fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1/search_checkpoint.pt" \
  --output_dir "$HOME/evopress_extension_results/fullspace_joint_s025_localxover_pop4_g150/fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1"
