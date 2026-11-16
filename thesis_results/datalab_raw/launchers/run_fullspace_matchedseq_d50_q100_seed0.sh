#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/evopress-crossover-v3

# Effort-matched structure-first pipeline under protocol F (seed 0):
#   stage 1: 16-bit depth-only search, FineWeb-Edu, 50 generations
#   stage 2: mask frozen, bit-width search at the exact budget, 100 generations
# 50 + 100 generations with the same offspring and selection schedule as the
# 150-generation joint search. Re-running this script skips a finished stage 1
# and resumes stage 2 from its last completed generation.
ROOT=/home/jovyan/evopress_extension_results/fullspace_matchedseq_d50_q100
S1=$ROOT/depth_only_f_s025_g50_seed0
OUT=$ROOT/frozen_q100_seed0
mkdir -p "$S1"

if [[ ! -f "$S1/final_candidate.json" ]]; then
  python evo_drop_search.py \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --sparsity 0.25 \
    --calibration_data fineweb_edu \
    --calibration_tokens 524288 \
    --calibration_sequence_length 8192 \
    --eval_datasets wikitext2 c4 \
    --eval_every 5 \
    --eval_tokens 524288 \
    --eval_sequence_length 8192 \
    --population_size 1 \
    --generations 50 \
    --offspring 128 \
    --initially_generated 1 \
    --initial_tokens 2048 \
    --survivors_per_selection 16 4 1 \
    --tokens_per_selection 2048 16384 131072 \
    --fitness_fn kl \
    --max_mutations 3 \
    --disk_teacher_logits \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    --seed 0 \
    --drop_config_dir "$S1"
else
  echo "Stage 1 already finished: $S1/final_candidate.json"
fi

RESUME=()
if [[ -f "$OUT/search_checkpoint.pt" ]]; then
  RESUME=(--resume_checkpoint "$OUT/search_checkpoint.pt")
  echo "Resuming stage 2 from $OUT/search_checkpoint.pt"
fi

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
  --generations 100 \
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
  --seed 0 \
  --sequential_mode depth_to_quant_frozen \
  --stage1_run_dir "$S1" \
  --max_initialization_attempts 100000 \
  --max_offspring_attempts 10000 \
  --output_dir "$OUT" ${RESUME[@]+"${RESUME[@]}"}
