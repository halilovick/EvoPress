#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
EVAL_TOKENS="${EVAL_TOKENS:-4096}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-0}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-1}"
DROP_LAYER_CONFIG="${DROP_LAYER_CONFIG:-}"
SPARSE_WEIGHTS_PATH="${SPARSE_WEIGHTS_PATH:-}"
SPARSE_CONFIG_PATH="${SPARSE_CONFIG_PATH:-}"
SPARSE_DEFAULT_LEVEL="${SPARSE_DEFAULT_LEVEL:-0}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-}"
QUANT_CONFIG_PATH="${QUANT_CONFIG_PATH:-}"
QUANT_DEFAULT_LEVEL="${QUANT_DEFAULT_LEVEL:-0}"
SPARSITY_OR_BITS="${SPARSITY_OR_BITS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PPL_SCRIPT="${EVAL_PPL_SCRIPT:-eval_ppl.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_RUNTIME_DEPENDENCIES="${CHECK_RUNTIME_DEPENDENCIES:-1}"

read -r -a EVAL_DATASETS_ARGS <<< "$EVAL_DATASETS"

COMPONENTS=()
if [[ -n "$DROP_LAYER_CONFIG" ]]; then
    COMPONENTS+=("depth")
fi
if [[ -n "$SPARSE_WEIGHTS_PATH" ]]; then
    COMPONENTS+=("sparse")
fi
if [[ -n "$QUANT_WEIGHTS_PATH" ]]; then
    COMPONENTS+=("quant")
fi

if ((${#COMPONENTS[@]} == 0)); then
    DEFAULT_METHOD="dense_tiny"
    DEFAULT_RUN_ID="dense_tinyllama_seq${SEQUENCE_LENGTH}_seed${SEED}"
else
    COMPONENT_LABEL="$(IFS=_; printf '%s' "${COMPONENTS[*]}")"
    DEFAULT_METHOD="combined_${COMPONENT_LABEL}_eval"
    DEFAULT_RUN_ID="${DEFAULT_METHOD}_tinyllama_seq${SEQUENCE_LENGTH}_seed${SEED}"
fi

METHOD="${METHOD:-$DEFAULT_METHOD}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"

usage() {
    cat <<'EOF'
Usage: scripts/run_combined_eval_tiny.sh [--dry-run]

Evaluate TinyLlama with zero or more compression components through eval_ppl.py.
Override parameters with environment variables.

Supported compression variables:
  DROP_LAYER_CONFIG=<path>
  SPARSE_WEIGHTS_PATH=<path>
  SPARSE_CONFIG_PATH=<path>
  SPARSE_DEFAULT_LEVEL=<int>
  QUANT_WEIGHTS_PATH=<path>
  QUANT_CONFIG_PATH=<path>
  QUANT_DEFAULT_LEVEL=<int>
  METHOD=<experiment_log method>
  RUN_ID=<run identifier>

Examples:
  scripts/run_combined_eval_tiny.sh --dry-run

  nohup bash scripts/run_combined_eval_tiny.sh \
    > outputs/combined_dense_tiny_launcher.log 2>&1 &

  METHOD=combined_depth_sparse_eval \
  RUN_ID=combined_depth0125_sparse_uniform_tinyllama_seed0 \
  DROP_LAYER_CONFIG=outputs/experiments/depth_tinyllama_s0.125_seed0/layer_drop_config.txt \
  SPARSE_WEIGHTS_PATH=outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db \
  SPARSE_DEFAULT_LEVEL=0 \
  bash scripts/run_combined_eval_tiny.sh
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

RUN_LOG="${OUTPUT_DIR}/run.log"
RUNTIME_FILE="${OUTPUT_DIR}/runtime.txt"
COMMAND_FILE="${OUTPUT_DIR}/command.sh"
METRICS_FILE="${OUTPUT_DIR}/evaluation_metrics.csv"
CONFIG_SUMMARY_FILE="${OUTPUT_DIR}/combined_config_summary.md"

COMMAND=(
    "$PYTHON_BIN" "$EVAL_PPL_SCRIPT"
    --model_name_or_path "$MODEL"
    --eval_datasets "${EVAL_DATASETS_ARGS[@]}"
    --eval_tokens "$EVAL_TOKENS"
    --sequence_length "$SEQUENCE_LENGTH"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
)

if [[ "$USE_FAST_TOKENIZER" == "1" ]]; then
    COMMAND+=(--use_fast_tokenizer)
fi
if [[ -n "$SPARSE_WEIGHTS_PATH" ]]; then
    COMMAND+=(--sparse_weights_path "$SPARSE_WEIGHTS_PATH")
    if [[ -n "$SPARSE_CONFIG_PATH" ]]; then
        COMMAND+=(--sparse_config_path "$SPARSE_CONFIG_PATH")
    fi
    COMMAND+=(--sparse_default_level "$SPARSE_DEFAULT_LEVEL")
fi
if [[ -n "$QUANT_WEIGHTS_PATH" ]]; then
    COMMAND+=(--quant_weights_path "$QUANT_WEIGHTS_PATH")
    if [[ -n "$QUANT_CONFIG_PATH" ]]; then
        COMMAND+=(--quant_config_path "$QUANT_CONFIG_PATH")
    fi
    COMMAND+=(--quant_default_level "$QUANT_DEFAULT_LEVEL")
fi
if [[ -n "$DROP_LAYER_CONFIG" ]]; then
    COMMAND+=(--drop_layer_config "$DROP_LAYER_CONFIG")
fi

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

check_runtime_dependencies() {
    if [[ "$CHECK_RUNTIME_DEPENDENCIES" == "1" ]]; then
        "$PYTHON_BIN" scripts/check_runtime_dependencies.py --require-cuda
    fi
}

write_command_file() {
    {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n'
        printf 'cd %q\n' "$REPO_ROOT"
        printf 'exec '
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$COMMAND_FILE"
    chmod +x "$COMMAND_FILE"
}

write_config_summary() {
    {
        printf '# Combined Evaluation Config\n\n'
        printf -- '- run_id: `%s`\n' "$RUN_ID"
        printf -- '- method: `%s`\n' "$METHOD"
        printf -- '- model: `%s`\n' "$MODEL"
        printf -- '- sequence_length: `%s`\n' "$SEQUENCE_LENGTH"
        printf -- '- eval_tokens: `%s`\n' "$EVAL_TOKENS"
        printf -- '- eval_datasets: `%s`\n' "$EVAL_DATASETS"
        printf -- '- dtype: `%s`\n' "$DTYPE"
        printf -- '- attention_impl: `%s`\n' "$ATTN_IMPLEMENTATION"
        printf -- '- seed: `%s`\n' "$SEED"
        printf -- '- drop_layer_config: `%s`\n' "${DROP_LAYER_CONFIG:-none}"
        printf -- '- sparse_weights_path: `%s`\n' "${SPARSE_WEIGHTS_PATH:-none}"
        printf -- '- sparse_config_path: `%s`\n' "${SPARSE_CONFIG_PATH:-none}"
        printf -- '- sparse_default_level: `%s`\n' "$SPARSE_DEFAULT_LEVEL"
        printf -- '- quant_weights_path: `%s`\n' "${QUANT_WEIGHTS_PATH:-none}"
        printf -- '- quant_config_path: `%s`\n' "${QUANT_CONFIG_PATH:-none}"
        printf -- '- quant_default_level: `%s`\n' "$QUANT_DEFAULT_LEVEL"
    } > "$CONFIG_SUMMARY_FILE"
}

get_gpu_name() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true
    fi
}

get_gpu_vram_gb() {
    local memory_mib
    if command -v nvidia-smi >/dev/null 2>&1; then
        memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
        if [[ -n "$memory_mib" ]]; then
            awk -v memory_mib="$memory_mib" 'BEGIN { printf "%.2f", memory_mib / 1024 }'
        fi
    fi
}

get_cpu_ram_limit_gb() {
    local memory_max
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        memory_max="$(cat /sys/fs/cgroup/memory.max)"
        if [[ "$memory_max" =~ ^[0-9]+$ ]]; then
            awk -v memory_max="$memory_max" 'BEGIN { printf "%.2f", memory_max / 1024 / 1024 / 1024 }'
        fi
    fi
}

extract_wikitext2_ppl() {
    "$PYTHON_BIN" - "$METRICS_FILE" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
matches = [row for row in rows if row["dataset"] == "wikitext2"]
if len(matches) != 1:
    raise SystemExit(1)
print(matches[0]["ppl"])
PY
}

config_notes() {
    printf 'drop_layer_config=%s; sparse_weights_path=%s; sparse_config_path=%s; sparse_default_level=%s; quant_weights_path=%s; quant_config_path=%s; quant_default_level=%s; eval_tokens=%s' \
        "${DROP_LAYER_CONFIG:-none}" \
        "${SPARSE_WEIGHTS_PATH:-none}" \
        "${SPARSE_CONFIG_PATH:-none}" \
        "$SPARSE_DEFAULT_LEVEL" \
        "${QUANT_WEIGHTS_PATH:-none}" \
        "${QUANT_CONFIG_PATH:-none}" \
        "$QUANT_DEFAULT_LEVEL" \
        "$EVAL_TOKENS"
}

append_experiment_row() {
    local status="$1"
    local notes="$2"
    local runtime_minutes="$3"
    local wikitext2_ppl="$4"
    local gpu_name="$5"
    local gpu_vram_gb="$6"
    local cpu_ram_limit_gb="$7"

    "$PYTHON_BIN" scripts/append_experiment_log.py \
        --log-file "$EXPERIMENT_LOG" \
        --run-id "$RUN_ID" \
        --method "$METHOD" \
        --model "$MODEL" \
        --sparsity-or-bits "$SPARSITY_OR_BITS" \
        --sequence-length "$SEQUENCE_LENGTH" \
        --attention-impl "$ATTN_IMPLEMENTATION" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        --wikitext2-ppl "$wikitext2_ppl" \
        --runtime-minutes "$runtime_minutes" \
        --gpu-name "$gpu_name" \
        --gpu-vram-gb "$gpu_vram_gb" \
        --cpu-ram-limit-gb "$cpu_ram_limit_gb" \
        --status "$status" \
        --notes "$notes" \
        --output-dir "$OUTPUT_DIR"
}

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run only. Would prepare command for %s:\n' "$RUN_ID"
    printf '  method=%s\n' "$METHOD"
    printf '  output_dir=%s\n' "$OUTPUT_DIR"
    printf '  command_file=%s\n' "$COMMAND_FILE"
    printf '  command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

check_runtime_dependencies || exit 2

if directory_has_files "$OUTPUT_DIR"; then
    printf 'Refusing to overwrite non-empty output directory: %s\n' "$OUTPUT_DIR" >&2
    printf 'Set RUN_ID or OUTPUT_DIR to a new location before rerunning.\n' >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
write_command_file
write_config_summary

GPU_NAME="$(get_gpu_name)"
GPU_VRAM_GB="$(get_gpu_vram_gb)"
CPU_RAM_LIMIT_GB="$(get_cpu_ram_limit_gb)"
START_TIME="$(date +%s)"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'method=%s\n' "$METHOD"
    printf 'output_dir=%s\n' "$OUTPUT_DIR"
    printf 'started_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'command_file=%s\n' "$COMMAND_FILE"
    printf 'combined_config_summary=%s\n' "$CONFIG_SUMMARY_FILE"
    printf '\n'
} | tee "$RUN_LOG"

"${COMMAND[@]}" 2>&1 | tee -a "$RUN_LOG"
RUN_EXIT_CODE="${PIPESTATUS[0]}"

END_TIME="$(date +%s)"
RUNTIME_SECONDS="$((END_TIME - START_TIME))"
RUNTIME_MINUTES="$(awk -v runtime_seconds="$RUNTIME_SECONDS" 'BEGIN { printf "%.2f", runtime_seconds / 60 }')"
{
    printf 'runtime_seconds=%s\n' "$RUNTIME_SECONDS"
    printf 'runtime_minutes=%s\n' "$RUNTIME_MINUTES"
    printf 'exit_code=%s\n' "$RUN_EXIT_CODE"
} > "$RUNTIME_FILE"

PARSER_EXIT_CODE=0
if [[ "$RUN_EXIT_CODE" == "0" ]]; then
    "$PYTHON_BIN" scripts/parse_eval_ppl_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" \
        --required-datasets "${EVAL_DATASETS_ARGS[@]}" || PARSER_EXIT_CODE="$?"
fi

WIKITEXT2_PPL=""
if [[ "$PARSER_EXIT_CODE" == "0" && -f "$METRICS_FILE" ]]; then
    WIKITEXT2_PPL="$(extract_wikitext2_ppl)"
fi

STATUS=completed
NOTES="last_successful_step=final_evaluation; $(config_notes)"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=combined_evaluation_process_started; command_exit_code=${RUN_EXIT_CODE}; $(config_notes)"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$PARSER_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=combined_evaluation_process_completed; parser_exit_code=${PARSER_EXIT_CODE}; $(config_notes)"
    FINAL_EXIT_CODE=1
fi

append_experiment_row \
    "$STATUS" \
    "$NOTES" \
    "$RUNTIME_MINUTES" \
    "$WIKITEXT2_PPL" \
    "$GPU_NAME" \
    "$GPU_VRAM_GB" \
    "$CPU_RAM_LIMIT_GB"

printf 'Experiment %s finished with status=%s.\n' "$RUN_ID" "$STATUS"
printf 'Artifacts: %s\n' "$OUTPUT_DIR"
exit "$FINAL_EXIT_CODE"
