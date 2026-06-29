#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python lmeval.py --model hf --model_args pretrained=mistralai/Mistral-7B-v0.3\,low_cpu_mem_usage=True\,dtype=float16 --tasks arc_easy\,piqa\,winogrande --batch_size 4 --device cuda:0 --num_fewshot 0 --output_path outputs/experiments/lmeval_dense_mistral_tasks_seed0_retry4/lmeval_results.json 
