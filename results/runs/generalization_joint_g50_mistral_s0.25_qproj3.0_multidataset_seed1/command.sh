#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 c4 fineweb_edu --eval_tokens 131072 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 1 --use_fast_tokenizer --quant_weights_path outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit --quant_config_path results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed1/joint_quant_config.txt --quant_default_level 3 --drop_layer_config results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed1/joint_drop_config.txt 
