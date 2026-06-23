#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 c4 fineweb_edu --eval_tokens 131072 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 0 --use_fast_tokenizer --drop_layer_config results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/layer_drop_config.txt 
