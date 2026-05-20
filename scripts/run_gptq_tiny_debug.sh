#!/bin/bash

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQUENCE_LENGTH=128
CALIB_DATA="wikitext2"

NUM_TOKENS=512
BITS_LIST="2 3 4"
BITS_TO_LOAD=3
GROUP_SIZE=128

SAVE_DIR="./outputs/quant_db_tiny"

mkdir -p "$SAVE_DIR"

torchrun --nnodes=1 --nproc-per-node=1 --master_port 29503 quant.py \
    --model_name_or_path "$MODEL" \
    --quantizable_modules '.*layers.*q_proj$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --post_block_modules model.norm lm_head \
    --calibration_data "$CALIB_DATA" \
    --calibration_tokens "$NUM_TOKENS" \
    --calibration_sequence_length "$SEQUENCE_LENGTH" \
    --bitwidth_options $BITS_LIST \
    --calibration_bitwidth "$BITS_TO_LOAD" \
    --group_size "$GROUP_SIZE" \
    --perchannel \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    --dtype float16 \
    --attn_implementation sdpa \
    --save_dir "$SAVE_DIR"
