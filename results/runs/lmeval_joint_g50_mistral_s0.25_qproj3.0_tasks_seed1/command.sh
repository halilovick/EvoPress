#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec python lmeval.py --model hf --model_args pretrained=mistralai/Mistral-7B-v0.3\,low_cpu_mem_usage=True\,dtype=float16 --tasks arc_easy\,piqa\,winogrande --batch_size 4 --device cuda:0 --num_fewshot 0 --output_path outputs/experiments/lmeval_joint_g50_mistral_s0.25_qproj3.0_tasks_seed1/lmeval_results.json --quant_weights_path outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit --quant_config_path results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed1/joint_quant_config.txt --quant_default_level 3 --drop_layer_config results/runs/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed1/joint_drop_config.txt 
