#!/bin/bash

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CALIB_DATA="wikitext2"

SEQUENCE_LENGTH=256
CALIB_TOKENS=4096
EVAL_TOKENS=8192

DROP_SPARSITY=0.125
TARGET_BITWIDTH=3.5

QUANT_WEIGHTS_PATH="./outputs/quant_db_tiny/TinyLlama-1.1B-Chat-v1.0/3bit"
OUTPUT_DIR="./outputs/joint_tiny_drop0125_quant35"

python evo_joint_search.py \
    --model_name_or_path "$MODEL" \
    --calibration_data "$CALIB_DATA" \
    --calibration_tokens "$CALIB_TOKENS" \
    --calibration_sequence_length "$SEQUENCE_LENGTH" \
    --eval_datasets wikitext2 \
    --eval_tokens "$EVAL_TOKENS" \
    --eval_sequence_length "$SEQUENCE_LENGTH" \
    --eval_every 1 \
    --drop_sparsity "$DROP_SPARSITY" \
    --quant_weights_path "$QUANT_WEIGHTS_PATH" \
    --target_bitwidth "$TARGET_BITWIDTH" \
    --group_rule none \
    --step_size 1 \
    --generations 5 \
    --offspring 8 \
    --initially_generated 16 \
    --initial_tokens 512 \
    --survivors_per_selection 4 2 1 \
    --tokens_per_selection 256 512 1024 \
    --fitness_fn kl \
    --dtype float16 \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR"
