#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python eval_ppl.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eval_datasets wikitext2 --eval_tokens 4096 --sequence_length 1024 --eval_batch_size 1 --dtype float16 --attn_implementation sdpa --seed 0 --use_fast_tokenizer 
