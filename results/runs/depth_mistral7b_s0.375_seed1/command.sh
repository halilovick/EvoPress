#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python evo_drop_search.py --model_name_or_path mistralai/Mistral-7B-v0.3 --sparsity 0.375 --calibration_data wikitext2 --calibration_tokens 8192 --calibration_sequence_length 2048 --eval_every 1 --eval_datasets wikitext2 --eval_sequence_length 2048 --population_size 1 --generations 10 --offspring 8 --initially_generated 16 --initial_tokens 512 --survivors_per_selection 2 1 --tokens_per_selection 512 2048 --fitness_fn kl --use_fast_tokenizer --drop_config_dir outputs/experiments/depth_mistral7b_s0.375_seed1 --seed 1 --dtype float16 --attn_implementation sdpa 
