#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/evopress
exec torchrun --nnodes=1 --nproc-per-node=1 --master_port 29513 quant.py --model_name_or_path mistralai/Mistral-7B-v0.3 --quantizable_modules .\*layers.\*q_proj\$ --pre_block_modules model.embed_tokens --block_modules model.layers --post_block_modules model.norm lm_head --calibration_data wikitext2 --calibration_tokens 512 --calibration_sequence_length 128 --bitwidth_options 2 3 4 --calibration_bitwidth 3 --group_size 128 --rel_damp 1e-2 --block_size 128 --seed 0 --attn_implementation sdpa --dtype float16 --save_dir outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db --perchannel --low_cpu_mem_usage --verbose --drop_saved_file_cache 
