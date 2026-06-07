#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eval_datasets wikitext2 --eval_tokens 4096 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 0 --use_fast_tokenizer --quant_weights_path outputs/experiments/quant_db_tinyllama_alllinear_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit --quant_config_path outputs/experiments/quant_search_tinyllama_alllinear_3bit_g20_seed0/quant_configuration.txt --quant_default_level 0 --drop_layer_config outputs/experiments/depth_tinyllama_s0.125_seed0/layer_drop_config.txt 
