#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python evo_drop_search.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparsity 0.125 --calibration_data wikitext2 --calibration_tokens 4096 --calibration_sequence_length 1024 --eval_every 1 --eval_datasets wikitext2 --eval_sequence_length 1024 --population_size 1 --generations 10 --offspring 8 --initially_generated 16 --initial_tokens 512 --survivors_per_selection 2 1 --tokens_per_selection 512 2048 --fitness_fn kl --use_fast_tokenizer --drop_config_dir outputs/experiments/depth_tinyllama_s0.125_seed0 --seed 0 --dtype float16 --attn_implementation sdpa 
