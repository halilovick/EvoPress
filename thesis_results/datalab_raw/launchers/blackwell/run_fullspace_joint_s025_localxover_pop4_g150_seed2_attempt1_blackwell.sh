#!/usr/bin/env bash
set -euo pipefail

# LX (local-exchange crossover, population=4, s=0.25) seed 2, adapted for
# the new Blackwell SLURM node. Identical search-defining flags to
# thesis_results/datalab_raw/launchers/run_fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1.sh
# (the DataLab/A40 version of this exact run); only the repo path, env
# activation and quant_weights_path/output_dir have changed.
#
# PREREQUISITE: run_prepare_quant_db_fullspace_blackwell_attempt1.sh must
# have completed first, producing a 224-module quant DB at
# $HOME/evopress_quant_stage1_blackwell_fullspace_20261028/Mistral-7B-v0.3/3bit
#
# This run will almost certainly exceed the 12h SLURM wall-clock limit
# (the matching A40 seed0 run took 165,663s = ~46h; seed1 took 188,508s =
# ~52.4h). When the session ends, resubmit with
# run_fullspace_joint_s025_localxover_pop4_g150_seed2_resume_blackwell.sh
# (same output_dir, adds --resume_checkpoint) as many times as needed.

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
  --output_dir "$HOME/evopress_extension_results/fullspace_joint_s025_localxover_pop4_g150/fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1"
