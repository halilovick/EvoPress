#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python scripts/evaluate_depth_baselines.py --model_name_or_path mistralai/Mistral-7B-v0.3 --reference_config results/runs/depth_mistral7b_s0.125_seed1/layer_drop_config.txt --sparsity 0.125 --method late_layer --calibration_data wikitext2 --calibration_tokens 8192 --sequence_length 2048 --seed 1 --calibration_seed 1 --dtype float16 --attn_implementation sdpa --use_fast_tokenizer --run_id baseline_late_layer_mistral7b_s0.125_seed1_retry1 --output_dir outputs/experiments/baseline_late_layer_mistral7b_s0.125_seed1_retry1 
