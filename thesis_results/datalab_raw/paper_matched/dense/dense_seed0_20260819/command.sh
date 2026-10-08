#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec /opt/micromamba/bin/python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 c4 --eval_tokens 524288 --sequence_length 8192 --dtype float16 --attn_implementation flash_attention_2 --seed 0
