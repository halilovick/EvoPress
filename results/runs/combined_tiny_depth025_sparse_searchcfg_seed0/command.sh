#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eval_datasets wikitext2 --eval_tokens 4096 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 0 --use_fast_tokenizer --sparse_weights_path outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db --sparse_config_path results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/sparse_configuration.txt --sparse_default_level 0 --drop_layer_config outputs/experiments/depth_tinyllama_s0.25_seed0/layer_drop_config.txt 
