#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
CALIB_DATA="${CALIB_DATA:-wikitext2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
CALIB_TOKENS="${CALIB_TOKENS:-4096}"
SPARSITY="${SPARSITY:-0.50}"
NUM_LEVELS="${NUM_LEVELS:-3}"
PRUNABLE_MODULES="${PRUNABLE_MODULES:-.*layers.*q_proj$}"
PRE_BLOCK_MODULES="${PRE_BLOCK_MODULES:-model.embed_tokens}"
BLOCK_MODULES="${BLOCK_MODULES:-model.layers}"
WEIGHTS_DIFF="${WEIGHTS_DIFF:-}"
REL_DAMP="${REL_DAMP:-1e-2}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29511}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PRUNE_SCRIPT="${PRUNE_SCRIPT:-prune.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RUN_ID="${RUN_ID:-sparse_db_tinyllama_qproj_s0.50}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_DIR}/sparse_db}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-1}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-1}"
VERBOSE="${VERBOSE:-1}"
MEMORY_POLL_INTERVAL_SECONDS="${MEMORY_POLL_INTERVAL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_RUNTIME_DEPENDENCIES="${CHECK_RUNTIME_DEPENDENCIES:-1}"
DROP_SAVED_FILE_CACHE="${DROP_SAVED_FILE_CACHE:-1}"

read -r -a PRE_BLOCK_MODULES_ARGS <<< "$PRE_BLOCK_MODULES"

usage() {
    cat <<'EOF'
Usage: scripts/run_sparse_gpt_tiny_debug.sh [--dry-run]

Generate a minimal TinyLlama SparseGPT/FastOBC database for Task 7.1.
Override parameters with environment variables such as MODEL, SPARSITY,
PRUNABLE_MODULES, CALIB_TOKENS, SEQUENCE_LENGTH, RUN_ID, OUTPUT_DIR, or SAVE_DIR.

Defaults:
  MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
  PRUNABLE_MODULES='.*layers.*q_proj$'
  CALIB_TOKENS=4096
  SEQUENCE_LENGTH=1024
  SPARSITY=0.50
  NUM_LEVELS=3

Examples:
  scripts/run_sparse_gpt_tiny_debug.sh --dry-run
  nohup bash scripts/run_sparse_gpt_tiny_debug.sh > outputs/sparse_tiny_launcher.log 2>&1 &
  RUN_ID=sparse_db_tinyllama_qproj_s0.50_retry1 scripts/run_sparse_gpt_tiny_debug.sh
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
MEMORY_SAMPLES_FILE="${OUTPUT_DIR}/memory_samples.csv"
DB_SUMMARY_FILE="${OUTPUT_DIR}/sparse_db_summary.txt"

COMMAND=(
    "$TORCHRUN_BIN"
    --nnodes=1
    --nproc-per-node=1
    --master_port "$MASTER_PORT"
    "$PRUNE_SCRIPT"
    --model_name_or_path "$MODEL"
    --prunable_modules "$PRUNABLE_MODULES"
    --pre_block_modules "${PRE_BLOCK_MODULES_ARGS[@]}"
    --block_modules "$BLOCK_MODULES"
    --calibration_data "$CALIB_DATA"
    --calibration_tokens "$CALIB_TOKENS"
    --calibration_sequence_length "$SEQUENCE_LENGTH"
    --sparsity "$SPARSITY"
    --num_levels "$NUM_LEVELS"
    --rel_damp "$REL_DAMP"
    --block_size "$BLOCK_SIZE"
    --seed "$SEED"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --dtype "$DTYPE"
    --save_dir "$SAVE_DIR"
)

if [[ -n "$WEIGHTS_DIFF" ]]; then
    COMMAND+=(--weights_diff "$WEIGHTS_DIFF")
fi
if [[ "$LOW_CPU_MEM_USAGE" == "1" ]]; then
    COMMAND+=(--low_cpu_mem_usage)
fi
if [[ "$CPU_OFFLOAD_MODULES" == "1" ]]; then
    COMMAND+=(--cpu_offload_modules)
fi
if [[ "$CPU_OFFLOAD_ACTIVATIONS" == "1" ]]; then
    COMMAND+=(--cpu_offload_activations)
fi
if [[ "$VERBOSE" == "1" ]]; then
    COMMAND+=(--verbose)
fi
if [[ "$DROP_SAVED_FILE_CACHE" == "1" ]]; then
    COMMAND+=(--drop_saved_file_cache)
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

get_gpu_used_gb() {
    local memory_mib
    if command -v nvidia-smi >/dev/null 2>&1; then
        memory_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
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

get_cpu_memory_current_gb() {
    local memory_current
    if [[ -r /sys/fs/cgroup/memory.current ]]; then
        memory_current="$(cat /sys/fs/cgroup/memory.current)"
        if [[ "$memory_current" =~ ^[0-9]+$ ]]; then
            awk -v memory_current="$memory_current" 'BEGIN { printf "%.2f", memory_current / 1024 / 1024 / 1024 }'
        fi
    fi
}

monitor_memory() {
    printf 'unix_time,cpu_memory_current_gb,gpu_memory_used_gb\n'
    while true; do
        printf '%s,%s,%s\n' "$(date +%s)" "$(get_cpu_memory_current_gb)" "$(get_gpu_used_gb)"
        sleep "$MEMORY_POLL_INTERVAL_SECONDS"
    done
}

max_csv_column() {
    local column="$1"
    local file="$2"
    awk -F, -v column="$column" '
        NR > 1 && $column != "" {
            value = $column + 0
            if (!seen || value > max_value) {
                max_value = value
                seen = 1
            }
        }
        END {
            if (seen) {
                printf "%.2f", max_value
            }
        }
    ' "$file"
}

database_size_mb() {
    if [[ -d "$SAVE_DIR" ]]; then
        du -sm "$SAVE_DIR" 2>/dev/null | awk '{ print $1 }'
    fi
}

count_module_dirs() {
    if [[ -d "$SAVE_DIR" ]]; then
        find "$SAVE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{ print $1 }'
    fi
}

count_level_files() {
    if [[ -d "$SAVE_DIR" ]]; then
        find "$SAVE_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.pth' | wc -l | awk '{ print $1 }'
    fi
}

last_progress_line() {
    if [[ -f "$RUN_LOG" ]]; then
        grep -E '^(Processing|Pruning) ' "$RUN_LOG" | tail -n 1 || true
    fi
}

write_db_summary() {
    local metadata_present="$1"
    local module_dirs="$2"
    local level_files="$3"
    local size_mb="$4"
    local max_cpu_memory_gb="$5"
    local max_gpu_memory_gb="$6"
    local last_progress="$7"

    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'model=%s\n' "$MODEL"
        printf 'prunable_modules=%s\n' "$PRUNABLE_MODULES"
        printf 'sparsity=%s\n' "$SPARSITY"
        printf 'num_levels=%s\n' "$NUM_LEVELS"
        printf 'drop_saved_file_cache=%s\n' "$DROP_SAVED_FILE_CACHE"
        printf 'calibration_data=%s\n' "$CALIB_DATA"
        printf 'calibration_tokens=%s\n' "$CALIB_TOKENS"
        printf 'sequence_length=%s\n' "$SEQUENCE_LENGTH"
        printf 'dtype=%s\n' "$DTYPE"
        printf 'attention_impl=%s\n' "$ATTN_IMPLEMENTATION"
        printf 'save_dir=%s\n' "$SAVE_DIR"
        printf 'metadata_present=%s\n' "$metadata_present"
        printf 'generated_module_dirs=%s\n' "$module_dirs"
        printf 'generated_level_files=%s\n' "$level_files"
        printf 'database_size_mb=%s\n' "$size_mb"
        printf 'max_cpu_memory_gb=%s\n' "$max_cpu_memory_gb"
        printf 'max_gpu_memory_gb=%s\n' "$max_gpu_memory_gb"
        printf 'last_progress=%s\n' "$last_progress"
    } > "$DB_SUMMARY_FILE"
}

append_experiment_row() {
    local status="$1"
    local notes="$2"
    local runtime_minutes="$3"
    local gpu_name="$4"
    local gpu_vram_gb="$5"
    local cpu_ram_limit_gb="$6"

    "$PYTHON_BIN" scripts/append_experiment_log.py \
        --log-file "$EXPERIMENT_LOG" \
        --run-id "$RUN_ID" \
        --method sparse_db \
        --model "$MODEL" \
        --sparsity-or-bits "$SPARSITY" \
        --calibration-data "$CALIB_DATA" \
        --sequence-length "$SEQUENCE_LENGTH" \
        --calibration-tokens "$CALIB_TOKENS" \
        --attention-impl "$ATTN_IMPLEMENTATION" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
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
    printf '  output_dir=%s\n' "$OUTPUT_DIR"
    printf '  save_dir=%s\n' "$SAVE_DIR"
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
mkdir -p "$SAVE_DIR"
write_command_file

GPU_NAME="$(get_gpu_name)"
GPU_VRAM_GB="$(get_gpu_vram_gb)"
CPU_RAM_LIMIT_GB="$(get_cpu_ram_limit_gb)"
START_TIME="$(date +%s)"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'output_dir=%s\n' "$OUTPUT_DIR"
    printf 'save_dir=%s\n' "$SAVE_DIR"
    printf 'started_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'command_file=%s\n' "$COMMAND_FILE"
    printf 'memory_samples_file=%s\n' "$MEMORY_SAMPLES_FILE"
    printf '\n'
} | tee "$RUN_LOG"

monitor_memory > "$MEMORY_SAMPLES_FILE" &
MONITOR_PID="$!"

"${COMMAND[@]}" 2>&1 | tee -a "$RUN_LOG"
RUN_EXIT_CODE="${PIPESTATUS[0]}"

kill "$MONITOR_PID" >/dev/null 2>&1 || true
wait "$MONITOR_PID" >/dev/null 2>&1 || true

END_TIME="$(date +%s)"
RUNTIME_SECONDS="$((END_TIME - START_TIME))"
RUNTIME_MINUTES="$(awk -v runtime_seconds="$RUNTIME_SECONDS" 'BEGIN { printf "%.2f", runtime_seconds / 60 }')"
{
    printf 'runtime_seconds=%s\n' "$RUNTIME_SECONDS"
    printf 'runtime_minutes=%s\n' "$RUNTIME_MINUTES"
    printf 'exit_code=%s\n' "$RUN_EXIT_CODE"
} > "$RUNTIME_FILE"

METADATA_PRESENT=0
if [[ -f "${SAVE_DIR}/metadata.pth" ]]; then
    METADATA_PRESENT=1
fi
MODULE_DIRS="$(count_module_dirs)"
LEVEL_FILES="$(count_level_files)"
DATABASE_SIZE_MB="$(database_size_mb)"
MAX_CPU_MEMORY_GB="$(max_csv_column 2 "$MEMORY_SAMPLES_FILE")"
MAX_GPU_MEMORY_GB="$(max_csv_column 3 "$MEMORY_SAMPLES_FILE")"
LAST_PROGRESS="$(last_progress_line)"

write_db_summary \
    "$METADATA_PRESENT" \
    "$MODULE_DIRS" \
    "$LEVEL_FILES" \
    "$DATABASE_SIZE_MB" \
    "$MAX_CPU_MEMORY_GB" \
    "$MAX_GPU_MEMORY_GB" \
    "$LAST_PROGRESS"

STATUS=completed
NOTES="last_successful_step=sparse_database_generated; prunable_modules=${PRUNABLE_MODULES}; num_levels=${NUM_LEVELS}; drop_saved_file_cache=${DROP_SAVED_FILE_CACHE}; metadata_present=${METADATA_PRESENT}; generated_module_dirs=${MODULE_DIRS}; generated_level_files=${LEVEL_FILES}; database_size_mb=${DATABASE_SIZE_MB}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=sparse_database_process_started; command_exit_code=${RUN_EXIT_CODE}; prunable_modules=${PRUNABLE_MODULES}; num_levels=${NUM_LEVELS}; drop_saved_file_cache=${DROP_SAVED_FILE_CACHE}; metadata_present=${METADATA_PRESENT}; generated_module_dirs=${MODULE_DIRS}; generated_level_files=${LEVEL_FILES}; database_size_mb=${DATABASE_SIZE_MB}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}; last_progress=${LAST_PROGRESS}"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$METADATA_PRESENT" != "1" ]]; then
    STATUS=failed
    NOTES="last_successful_step=sparse_database_process_completed; missing_file=${SAVE_DIR}/metadata.pth; generated_module_dirs=${MODULE_DIRS}; generated_level_files=${LEVEL_FILES}; database_size_mb=${DATABASE_SIZE_MB}"
    FINAL_EXIT_CODE=1
elif [[ "${MODULE_DIRS:-0}" == "0" || "${LEVEL_FILES:-0}" == "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=sparse_database_process_completed; no_sparse_level_files; metadata_present=${METADATA_PRESENT}; generated_module_dirs=${MODULE_DIRS}; generated_level_files=${LEVEL_FILES}; database_size_mb=${DATABASE_SIZE_MB}"
    FINAL_EXIT_CODE=1
fi

append_experiment_row \
    "$STATUS" \
    "$NOTES" \
    "$RUNTIME_MINUTES" \
    "$GPU_NAME" \
    "$GPU_VRAM_GB" \
    "$CPU_RAM_LIMIT_GB"

printf 'Experiment %s finished with status=%s.\n' "$RUN_ID" "$STATUS"
printf 'Artifacts: %s\n' "$OUTPUT_DIR"
printf 'Sparse database: %s\n' "$SAVE_DIR"
exit "$FINAL_EXIT_CODE"
