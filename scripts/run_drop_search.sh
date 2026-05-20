# export CUDA_VISIBLE_DEVICES=0

MODEL="mistralai/Mistral-7B-v0.3"
CALIB_DATA="wikitext2"

SEQUENCE_LENGTH=2048
CALIB_TOKENS=8192

SPARSITY=0.375

CONFIG_PATH="./outputs/drop_search_test"

GENERATIONS=5

echo "Running with SPARSITY=$SPARSITY and GENERATIONS=$GENERATIONS"

mkdir -p "$CONFIG_PATH"

python evo_drop_search.py  \
    --model_name_or_path $MODEL \
    --sparsity $SPARSITY \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $CALIB_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --eval_every 1 \
    --eval_datasets wikitext2 \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --population_size 1 \
    --generations $GENERATIONS \
    --offspring 8 \
    --initially_generated 16 \
    --initial_tokens 512 \
    --survivors_per_selection 2 1 \
    --tokens_per_selection 512 2048 \
    --fitness_fn kl \
    --use_fast_tokenizer \
    --drop_config_dir $CONFIG_PATH \
    --dtype float16 \
    --attn_implementation sdpa