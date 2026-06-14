#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python evo_drop_search.py --model_name_or_path mistralai/Mistral-7B-v0.3 --sparsity 0.25 --calibration_data wikitext2 --calibration_tokens 8192 --calibration_sequence_length 1024 --eval_every 5 --eval_datasets wikitext2 --eval_sequence_length 1024 --population_size 1 --generations 20 --offspring 16 --initially_generated 32 --initial_tokens 512 --survivors_per_selection 8 2 1 --tokens_per_selection 512 2048 8192 --fitness_fn kl --use_fast_tokenizer --drop_config_dir outputs/experiments/thesis_medium_depth_mistral_s0.25_g20_o16_seed0 --seed 0 --dtype float16 --attn_implementation sdpa 
