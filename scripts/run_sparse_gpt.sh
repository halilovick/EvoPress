MODEL="mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH=128
SPARSITY=0.7

CALIB_DATA="wikitext2"
NUM_TOKENS=512
WEIGHTS_DIFF=200000
NUM_LEVELS=1

SAVE_DIR="./outputs/sparse_db_debug_qproj"

mkdir -p "$SAVE_DIR"

torchrun --nnodes=1 --nproc-per-node=1 --master_port 29501 prune.py \
    --model_name_or_path "$MODEL" \
    --prunable_modules '.*layers.*q_proj$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --calibration_data "$CALIB_DATA" \
    --calibration_tokens "$NUM_TOKENS" \
    --calibration_sequence_length "$SEQUENCE_LENGTH" \
    --sparsity "$SPARSITY" \
    --weights_diff "$WEIGHTS_DIFF" \
    --num_levels "$NUM_LEVELS" \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    --attn_implementation sdpa \
    --dtype float16 \
    --save_dir "$SAVE_DIR"