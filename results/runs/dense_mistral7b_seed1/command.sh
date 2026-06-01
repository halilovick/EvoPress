#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path mistralai/Mistral-7B-v0.3 --eval_datasets wikitext2 --sequence_length 2048 --dtype float16 --attn_implementation sdpa --use_fast_tokenizer --seed 1 
