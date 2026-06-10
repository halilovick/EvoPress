#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python evo_drop_search.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparsity 0.125 --calibration_data wikitext2 --calibration_tokens 1024 --calibration_sequence_length 512 --eval_every 1 --eval_datasets wikitext2 --eval_sequence_length 512 --population_size 1 --generations 2 --offspring 2 --initially_generated 2 --initial_tokens 128 --survivors_per_selection 2 1 --tokens_per_selection 128 512 --fitness_fn kl --use_fast_tokenizer --drop_config_dir outputs/experiments/debug_reporting_depth_tiny_seed0 --seed 0 --dtype float16 --attn_implementation sdpa 
