#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 --eval_tokens 4096 --sequence_length 512 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 0 --use_fast_tokenizer --quant_weights_path outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit --quant_config_path outputs/experiments/debug_reporting_quant_mistral_seed0/quant_configuration.txt --quant_default_level 0 --drop_layer_config outputs/experiments/debug_reporting_joint_mistral_seed0_retry1/joint_drop_config.txt 
