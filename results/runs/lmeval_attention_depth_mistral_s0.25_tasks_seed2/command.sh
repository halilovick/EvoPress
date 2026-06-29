#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python lmeval.py --model hf --model_args pretrained=mistralai/Mistral-7B-v0.3\,low_cpu_mem_usage=True\,dtype=float16 --tasks arc_easy\,piqa\,winogrande --batch_size 4 --device cuda:0 --num_fewshot 0 --output_path outputs/experiments/lmeval_attention_depth_mistral_s0.25_tasks_seed2/lmeval_results.json --drop_layer_config results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2/layer_drop_config.txt 
