#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python evo_prune_search.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparse_weights_path outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db --calibration_data wikitext2 --calibration_tokens 4096 --calibration_sequence_length 1024 --eval_every 1 --eval_datasets wikitext2 --eval_tokens 4096 --eval_sequence_length 1024 --generations 20 --offspring 8 --survivors_per_selection 2 1 --tokens_per_selection 512 2048 --fitness_fn kl --max_level 3 --max_total_deviation 99999 --dtype float16 --attn_implementation sdpa --seed 0 --configuration_name sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1_final_configuration.txt --use_fast_tokenizer 
