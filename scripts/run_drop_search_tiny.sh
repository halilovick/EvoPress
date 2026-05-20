#!/bin/bash

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CALIB_DATA="wikitext2"

SEQUENCE_LENGTH=256
CALIB_TOKENS=2048
SPARSITY=0.125
GENERATIONS=3

CONFIG_PATH="./outputs/drop_tiny_test"

mkdir -p "$CONFIG_PATH"

python evo_drop_search.py \
    --model_name_or_path "$MODEL" \
    --sparsity "$SPARSITY" \
    --calibration_data "$CALIB_DATA" \
    --calibration_tokens "$CALIB_TOKENS" \
    --calibration_sequence_length "$SEQUENCE_LENGTH" \
    --eval_every 1 \
    --eval_datasets wikitext2 \
    --eval_tokens 8192 \
    --eval_sequence_length "$SEQUENCE_LENGTH" \
    --population_size 1 \
    --generations "$GENERATIONS" \
    --offspring 4 \
    --initially_generated 8 \
    --initial_tokens 256 \
    --survivors_per_selection 2 1 \
    --tokens_per_selection 256 1024 \
    --fitness_fn kl \
    --drop_config_dir "$CONFIG_PATH" \
    --dtype float16 \
    --attn_implementation sdpa
