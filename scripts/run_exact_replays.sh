#!/usr/bin/env bash
# Exact-budget evaluation-only replays (RQ1 baselines, RQ3 interaction) and the
# optional structure-first sequential baseline. Nothing here was run as part of
# the completed experiments; see thesis_results/exact_replay_plan/README.md.
#
# Stages (STAGES env var, default "plan replay summary"):
#   depth125  cheap 16-bit depth-only searches at 12.5% (3 seeds, ~10 min each),
#             same protocol as thesis_medium_depth_mistral_s0.25_g20_o16_seed*
#   plan      write the replay plan (CPU only)
#   replay    run evo_exact_replay.py (GPU; one process, jobs resumable)
#   summary   compute contrasts and baseline tables (CPU only)
#   frozen    print (or, with RUN_FROZEN=1, run) the G150 structure-first
#             sequential searches under the exact budget (about 30-40 GPU-hours each)
set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-plan replay summary}"
QUANT_DB="${QUANT_DB:-/home/jovyan/evopress_quant_stage1_1gpu_diskspill_attempt3_20260819/Mistral-7B-v0.3/3bit}"
PLAN_DIR="${PLAN_DIR:-thesis_results/exact_replay_plan}"
OUT="${OUT:-results/exact_replays/replay_$(date +%Y%m%d)}"
TIERS="${TIERS:-0 A B C D}"
CALIBRATION_KL="${CALIBRATION_KL:-1}"
INCLUDE_125="${INCLUDE_125:-0}"
RUN_FROZEN="${RUN_FROZEN:-0}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has_stage depth125; then
  for seed in 0 1 2; do
    "${PYTHON}" evo_drop_search.py --model_name_or_path mistralai/Mistral-7B-v0.3 --sparsity 0.125 \
      --calibration_data wikitext2 --calibration_tokens 8192 --calibration_sequence_length 1024 \
      --eval_every 5 --eval_datasets wikitext2 --eval_sequence_length 1024 --population_size 1 \
      --generations 20 --offspring 16 --initially_generated 32 --initial_tokens 512 \
      --survivors_per_selection 8 2 1 --tokens_per_selection 512 2048 8192 --fitness_fn kl \
      --use_fast_tokenizer --drop_config_dir "results/runs/thesis_medium_depth_mistral_s0.125_g20_o16_seed${seed}" \
      --seed "${seed}" --dtype float16 --attn_implementation sdpa
  done
  INCLUDE_125=1
fi

if has_stage plan; then
  extra=()
  [[ "${INCLUDE_125}" == "1" ]] && extra+=(--include_depth_only_125)
  "${PYTHON}" scripts/plan_exact_replays.py --output_dir "${PLAN_DIR}" "${extra[@]}"
fi

if has_stage replay; then
  extra=()
  [[ "${CALIBRATION_KL}" == "1" ]] && extra+=(--calibration_kl)
  # shellcheck disable=SC2086
  "${PYTHON}" evo_exact_replay.py --plan "${PLAN_DIR}/replay_plan.json" --quant_db "${QUANT_DB}" \
    --output_dir "${OUT}" --tiers ${TIERS} "${extra[@]}"
fi

if has_stage summary; then
  "${PYTHON}" scripts/summarize_exact_replays.py --results_dir "${OUT}"
fi

if has_stage frozen; then
  for seed in 0 1 2; do
    cmd=("${PYTHON}" evo_joint_search.py --model_name_or_path mistralai/Mistral-7B-v0.3
      --quant_weights_path "${QUANT_DB}" --target_bitwidth 3 --calibration_data fineweb_edu
      --calibration_tokens 524288 --calibration_sequence_length 8192 --eval_datasets wikitext2 c4
      --eval_tokens 524288 --eval_sequence_length 8192 --eval_every 5 --generations 150 --offspring 128
      --survivors_per_selection 16 4 1 --tokens_per_selection 2048 16384 131072 --initial_tokens 2048
      --fitness_fn kl --group_rule size --step_size 1
      --compression_budget_mode match_uniform_quantization_total --quantization_group_size 128
      --budget_scale_bits 16 --budget_zero_point_bits 16 --budget_dense_dtype_bits 16
      --expected_dense_model_bits 115968376832 --expected_target_cost_bits 26982023168
      --expected_bitwidths 2 3 4 5 6 --expected_quantized_modules 224 --dtype float16
      --attn_implementation flash_attention_2 --seed "${seed}"
      --output_dir "results/exact_sequential/depth_to_quant_frozen_s025_g150_seed${seed}"
      --budget_include_quantization_metadata --drop_sparsity 0.25 --max_drop_mutations 3
      --joint_mutation_mode standard --initially_generated 1 --population_size 1
      --crossover_probability 0.0 --skip_initial_single_candidate_evaluation
      --sequential_mode depth_to_quant_frozen --allow_exact_budget_ablation
      --stage1_run_dir "results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}")
    if [[ "${RUN_FROZEN}" == "1" ]]; then
      "${cmd[@]}"
    else
      printf '%q ' "${cmd[@]}"; echo
    fi
  done
fi
