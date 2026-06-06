#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_tinyllama_qproj_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit}"
TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
CALIB_DATA="${CALIB_DATA:-wikitext2}"
CALIB_TOKENS="${CALIB_TOKENS:-4096}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
EVAL_TOKENS="${EVAL_TOKENS:-4096}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2}"
EVAL_EVERY="${EVAL_EVERY:-1}"
GENERATIONS="${GENERATIONS:-20}"
OFFSPRING="${OFFSPRING:-8}"
INITIALLY_GENERATED="${INITIALLY_GENERATED:-16}"
INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-2 1}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048}"
FITNESS_FN="${FITNESS_FN:-kl}"
GROUP_RULE="${GROUP_RULE:-none}"
STEP_SIZE="${STEP_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-0}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVO_QUANT_SEARCH_SCRIPT="${EVO_QUANT_SEARCH_SCRIPT:-evo_quant_search.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RUN_ID="${RUN_ID:-quant_search_tinyllama_qproj_3bit_g20_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"
CONFIGURATION_NAME="${CONFIGURATION_NAME:-${RUN_ID}_final_configuration.txt}"
DRY_RUN="${DRY_RUN:-0}"
MEMORY_POLL_INTERVAL_SECONDS="${MEMORY_POLL_INTERVAL_SECONDS:-5}"
CHECK_RUNTIME_DEPENDENCIES="${CHECK_RUNTIME_DEPENDENCIES:-1}"

read -r -a EVAL_DATASETS_ARGS <<< "$EVAL_DATASETS"
read -r -a SURVIVORS_PER_SELECTION_ARGS <<< "$SURVIVORS_PER_SELECTION"
read -r -a TOKENS_PER_SELECTION_ARGS <<< "$TOKENS_PER_SELECTION"

usage() {
    cat <<'EOF'
Usage: scripts/run_quant_search_tiny_interesting.sh [--dry-run]

Run a logged TinyLlama quant-only evolutionary search using the generated q-proj
GPTQ database. Override parameters with environment variables.

Defaults:
  QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_tinyllama_qproj_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit
  TARGET_BITWIDTH=3.0
  GENERATIONS=20
  OFFSPRING=8
  CALIB_TOKENS=4096
  SEQUENCE_LENGTH=1024

Examples:
  scripts/run_quant_search_tiny_interesting.sh --dry-run
  nohup bash scripts/run_quant_search_tiny_interesting.sh > outputs/quant_search_tiny_launcher.log 2>&1 &
  RUN_ID=quant_search_tinyllama_qproj_3bit_g20_seed0_retry1 scripts/run_quant_search_tiny_interesting.sh
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
METRICS_FILE="${OUTPUT_DIR}/generation_metrics.csv"
FINAL_CONFIG_FILE="${OUTPUT_DIR}/quant_configuration.txt"
MEMORY_SAMPLES_FILE="${OUTPUT_DIR}/memory_samples.csv"

COMMAND=(
    "$PYTHON_BIN" "$EVO_QUANT_SEARCH_SCRIPT"
    --model_name_or_path "$MODEL"
    --quant_weights_path "$QUANT_WEIGHTS_PATH"
    --target_bitwidth "$TARGET_BITWIDTH"
    --calibration_data "$CALIB_DATA"
    --calibration_tokens "$CALIB_TOKENS"
    --calibration_sequence_length "$SEQUENCE_LENGTH"
    --eval_every "$EVAL_EVERY"
    --eval_datasets "${EVAL_DATASETS_ARGS[@]}"
    --eval_tokens "$EVAL_TOKENS"
    --eval_sequence_length "$SEQUENCE_LENGTH"
    --generations "$GENERATIONS"
    --offspring "$OFFSPRING"
    --initially_generated "$INITIALLY_GENERATED"
    --initial_tokens "$INITIAL_TOKENS"
    --survivors_per_selection "${SURVIVORS_PER_SELECTION_ARGS[@]}"
    --tokens_per_selection "${TOKENS_PER_SELECTION_ARGS[@]}"
    --fitness_fn "$FITNESS_FN"
    --group_rule "$GROUP_RULE"
    --step_size "$STEP_SIZE"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --configuration_name "$CONFIGURATION_NAME"
)

if [[ "$USE_FAST_TOKENIZER" == "1" ]]; then
    COMMAND+=(--use_fast_tokenizer)
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

extract_final_metrics() {
    "$PYTHON_BIN" - "$METRICS_FILE" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
final_rows = [row for row in rows if row["phase"] == "final"]
if not final_rows:
    raise SystemExit(1)
print(final_rows[-1]["wikitext2_ppl"])
print(final_rows[-1]["train_ppl"])
print(final_rows[-1]["search_point"])
PY
}

average_config_bitwidth() {
    "$PYTHON_BIN" - "$FINAL_CONFIG_FILE" <<'PY'
import sys

levels = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        _, level = line.rsplit(":", 1)
        levels.append(int(level.strip()))
if not levels:
    raise SystemExit(1)
print(f"{sum(levels) / len(levels):.4f}")
PY
}

is_finite_number() {
    "$PYTHON_BIN" - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) else 1)
PY
}

append_experiment_row() {
    local status="$1"
    local notes="$2"
    local runtime_minutes="$3"
    local wikitext2_ppl="$4"
    local train_ppl="$5"
    local gpu_name="$6"
    local gpu_vram_gb="$7"
    local cpu_ram_limit_gb="$8"

    "$PYTHON_BIN" scripts/append_experiment_log.py \
        --log-file "$EXPERIMENT_LOG" \
        --run-id "$RUN_ID" \
        --method quant_search \
        --model "$MODEL" \
        --sparsity-or-bits "$TARGET_BITWIDTH" \
        --generations "$GENERATIONS" \
        --offspring "$OFFSPRING" \
        --calibration-data "$CALIB_DATA" \
        --sequence-length "$SEQUENCE_LENGTH" \
        --calibration-tokens "$CALIB_TOKENS" \
        --fitness-fn "$FITNESS_FN" \
        --attention-impl "$ATTN_IMPLEMENTATION" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        --wikitext2-ppl "$wikitext2_ppl" \
        --train-ppl "$train_ppl" \
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
    printf '  quant_weights_path=%s\n' "$QUANT_WEIGHTS_PATH"
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

if [[ ! -d "$QUANT_WEIGHTS_PATH" ]]; then
    printf 'Quant weights path does not exist: %s\n' "$QUANT_WEIGHTS_PATH" >&2
    exit 2
fi
if [[ -z "$(find "$QUANT_WEIGHTS_PATH" -mindepth 2 -maxdepth 2 -type f -name '*.pth' -print -quit)" ]]; then
    printf 'Quant weights path contains no candidate weight files: %s\n' "$QUANT_WEIGHTS_PATH" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
write_command_file

GPU_NAME="$(get_gpu_name)"
GPU_VRAM_GB="$(get_gpu_vram_gb)"
CPU_RAM_LIMIT_GB="$(get_cpu_ram_limit_gb)"
START_TIME="$(date +%s)"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'output_dir=%s\n' "$OUTPUT_DIR"
    printf 'quant_weights_path=%s\n' "$QUANT_WEIGHTS_PATH"
    printf 'target_bitwidth=%s\n' "$TARGET_BITWIDTH"
    printf 'started_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'command_file=%s\n' "$COMMAND_FILE"
    printf 'memory_samples_file=%s\n' "$MEMORY_SAMPLES_FILE"
    printf 'configuration_name=%s\n' "$CONFIGURATION_NAME"
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

PARSER_EXIT_CODE=0
if [[ "$RUN_EXIT_CODE" == "0" ]]; then
    "$PYTHON_BIN" scripts/parse_quant_search_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" || PARSER_EXIT_CODE="$?"
else
    "$PYTHON_BIN" scripts/parse_quant_search_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" \
        --allow-incomplete || PARSER_EXIT_CODE="$?"
fi

CONFIG_SOURCE="${QUANT_WEIGHTS_PATH}/${CONFIGURATION_NAME}"
if [[ -f "$CONFIG_SOURCE" ]]; then
    cp "$CONFIG_SOURCE" "$FINAL_CONFIG_FILE"
fi

FINAL_WIKITEXT2_PPL=""
FINAL_TRAIN_PPL=""
FINAL_SEARCH_POINT=""
if [[ "$PARSER_EXIT_CODE" == "0" && -f "$METRICS_FILE" ]]; then
    FINAL_METRICS="$(extract_final_metrics)" || true
    FINAL_WIKITEXT2_PPL="$(printf '%s\n' "$FINAL_METRICS" | sed -n '1p')"
    FINAL_TRAIN_PPL="$(printf '%s\n' "$FINAL_METRICS" | sed -n '2p')"
    FINAL_SEARCH_POINT="$(printf '%s\n' "$FINAL_METRICS" | sed -n '3p')"
fi

ACTUAL_AVERAGE_BITWIDTH=""
if [[ -f "$FINAL_CONFIG_FILE" ]]; then
    ACTUAL_AVERAGE_BITWIDTH="$(average_config_bitwidth)" || true
fi

MAX_CPU_MEMORY_GB="$(max_csv_column 2 "$MEMORY_SAMPLES_FILE")"
MAX_GPU_MEMORY_GB="$(max_csv_column 3 "$MEMORY_SAMPLES_FILE")"

STATUS=completed
NOTES="last_successful_step=final_evaluation; quant_weights_path=${QUANT_WEIGHTS_PATH}; configuration_name=${CONFIGURATION_NAME}; target_bitwidth=${TARGET_BITWIDTH}; actual_average_bitwidth=${ACTUAL_AVERAGE_BITWIDTH}; final_search_point=${FINAL_SEARCH_POINT}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=quant_search_process_started; command_exit_code=${RUN_EXIT_CODE}; parser_exit_code=${PARSER_EXIT_CODE}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$PARSER_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=quant_search_process_completed; parser_exit_code=${PARSER_EXIT_CODE}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE=1
elif [[ ! -f "$FINAL_CONFIG_FILE" ]]; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; missing_file=${FINAL_CONFIG_FILE}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE=1
elif ! is_finite_number "$FINAL_WIKITEXT2_PPL"; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; non_finite_wikitext2_ppl=${FINAL_WIKITEXT2_PPL}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE=1
elif ! is_finite_number "$FINAL_TRAIN_PPL"; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; non_finite_train_ppl=${FINAL_TRAIN_PPL}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE=1
elif ! is_finite_number "$ACTUAL_AVERAGE_BITWIDTH"; then
    STATUS=failed
    NOTES="last_successful_step=configuration_copied; invalid_average_bitwidth=${ACTUAL_AVERAGE_BITWIDTH}; quant_weights_path=${QUANT_WEIGHTS_PATH}; max_cpu_memory_gb=${MAX_CPU_MEMORY_GB}; max_gpu_memory_gb=${MAX_GPU_MEMORY_GB}"
    FINAL_EXIT_CODE=1
fi

append_experiment_row \
    "$STATUS" \
    "$NOTES" \
    "$RUNTIME_MINUTES" \
    "$FINAL_WIKITEXT2_PPL" \
    "$FINAL_TRAIN_PPL" \
    "$GPU_NAME" \
    "$GPU_VRAM_GB" \
    "$CPU_RAM_LIMIT_GB"

printf 'Experiment %s finished with status=%s.\n' "$RUN_ID" "$STATUS"
printf 'Artifacts: %s\n' "$OUTPUT_DIR"
exit "$FINAL_EXIT_CODE"
