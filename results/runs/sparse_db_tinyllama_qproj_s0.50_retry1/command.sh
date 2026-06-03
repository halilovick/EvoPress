#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec torchrun --nnodes=1 --nproc-per-node=1 --master_port 29511 prune.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prunable_modules .\*layers.\*q_proj\$ --pre_block_modules model.embed_tokens --block_modules model.layers --calibration_data wikitext2 --calibration_tokens 4096 --calibration_sequence_length 1024 --sparsity 0.50 --num_levels 3 --rel_damp 1e-2 --block_size 128 --seed 0 --attn_implementation sdpa --dtype float16 --save_dir outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db --low_cpu_mem_usage --cpu_offload_modules --cpu_offload_activations --verbose 
