#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec /opt/micromamba/bin/python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 c4 --eval_tokens 524288 --sequence_length 8192 --dtype float16 --attn_implementation flash_attention_2 --seed 0 --quant_weights_path /home/jovyan/evopress_quant_stage1_1gpu_diskspill_attempt3_20260819/Mistral-7B-v0.3/3bit --quant_default_level 3
