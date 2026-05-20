#!/bin/bash

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQUENCE_LENGTH=256

# Use WikiText2 for small stable local tests
CALIB_DATA="wikitext2"

CALIB_TOKENS=4096
EVAL_TOKENS=8192

# Fractional target forces a mixed-bit configuration, e.g. 3-bit and 4-bit layers
BIT_LEVEL=3.5

# This is the database created by run_gptq_tiny_debug.sh
COMPR_PATH="./outputs/quant_db_tiny/TinyLlama-1.1B-Chat-v1.0/3bit"

python evo_quant_search.py \
    --model_name_or_path "$MODEL" \
    --quant_weights_path "$COMPR_PATH" \
    --target_bitwidth "$BIT_LEVEL" \
    --calibration_data "$CALIB_DATA" \
    --calibration_tokens "$CALIB_TOKENS" \
    --calibration_sequence_length "$SEQUENCE_LENGTH" \
    --eval_every 1 \
    --eval_datasets wikitext2 \
    --eval_tokens "$EVAL_TOKENS" \
    --eval_sequence_length "$SEQUENCE_LENGTH" \
    --generations 10 \
    --offspring 16 \
    --initially_generated 32 \
    --initial_tokens 512 \
    --survivors_per_selection 8 2 1 \
    --tokens_per_selection 256 1024 2048 \
    --fitness_fn kl \
    --group_rule none \
    --step_size 1 \
    --dtype float16 \
    --attn_implementation sdpa
