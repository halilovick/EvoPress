#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 --eval_tokens 524288 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 1 --use_fast_tokenizer --quant_weights_path outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit --quant_default_level 3 --drop_layer_config results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1/layer_drop_config.txt 
