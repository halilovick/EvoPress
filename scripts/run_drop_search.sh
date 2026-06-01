#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
CALIB_DATA="${CALIB_DATA:-wikitext2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
CALIB_TOKENS="${CALIB_TOKENS:-8192}"
SPARSITY="${SPARSITY:-0.375}"
GENERATIONS="${GENERATIONS:-10}"
OFFSPRING="${OFFSPRING:-8}"
INITIALLY_GENERATED="${INITIALLY_GENERATED:-16}"
INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-2 1}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048}"
FITNESS_FN="${FITNESS_FN:-kl}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVO_DROP_SEARCH_SCRIPT="${EVO_DROP_SEARCH_SCRIPT:-evo_drop_search.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RUN_ID="${RUN_ID:-depth_mistral7b_s${SPARSITY}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SURVIVORS_PER_SELECTION_ARGS <<< "$SURVIVORS_PER_SELECTION"
read -r -a TOKENS_PER_SELECTION_ARGS <<< "$TOKENS_PER_SELECTION"

usage() {
    cat <<'EOF'
Usage: scripts/run_drop_search.sh [--dry-run]

Run one EvoPress Mistral depth-pruning experiment. Override parameters with
environment variables such as SPARSITY, GENERATIONS, SEED, or OUTPUT_DIR.

Examples:
  SPARSITY=0.125 scripts/run_drop_search.sh
  GENERATIONS=20 SPARSITY=0.375 RUN_ID=depth_mistral7b_s0.375_g20_seed1 scripts/run_drop_search.sh
  SPARSITY=0.125 scripts/run_drop_search.sh --dry-run
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
CONFIG_FILE="${OUTPUT_DIR}/layer_drop_config.txt"

COMMAND=(
    "$PYTHON_BIN" "$EVO_DROP_SEARCH_SCRIPT"
    --model_name_or_path "$MODEL"
    --sparsity "$SPARSITY"
    --calibration_data "$CALIB_DATA"
    --calibration_tokens "$CALIB_TOKENS"
    --calibration_sequence_length "$SEQUENCE_LENGTH"
    --eval_every 1
    --eval_datasets wikitext2
    --eval_sequence_length "$SEQUENCE_LENGTH"
    --population_size 1
    --generations "$GENERATIONS"
    --offspring "$OFFSPRING"
    --initially_generated "$INITIALLY_GENERATED"
    --initial_tokens "$INITIAL_TOKENS"
    --survivors_per_selection "${SURVIVORS_PER_SELECTION_ARGS[@]}"
    --tokens_per_selection "${TOKENS_PER_SELECTION_ARGS[@]}"
    --fitness_fn "$FITNESS_FN"
    --use_fast_tokenizer
    --drop_config_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
)

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
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

get_cpu_ram_limit_gb() {
    local memory_max
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        memory_max="$(cat /sys/fs/cgroup/memory.max)"
        if [[ "$memory_max" =~ ^[0-9]+$ ]]; then
            awk -v memory_max="$memory_max" 'BEGIN { printf "%.2f", memory_max / 1024 / 1024 / 1024 }'
        fi
    fi
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
        --method depth_evo \
        --model "$MODEL" \
        --sparsity-or-bits "$SPARSITY" \
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
    printf '  command_file=%s\n' "$COMMAND_FILE"
    printf '  command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

if directory_has_files "$OUTPUT_DIR"; then
    printf 'Refusing to overwrite non-empty output directory: %s\n' "$OUTPUT_DIR" >&2
    printf 'Set RUN_ID or OUTPUT_DIR to a new location before rerunning.\n' >&2
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
    printf 'started_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'command_file=%s\n' "$COMMAND_FILE"
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
    "$PYTHON_BIN" scripts/parse_depth_search_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" || PARSER_EXIT_CODE="$?"
else
    "$PYTHON_BIN" scripts/parse_depth_search_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" \
        --allow-incomplete || PARSER_EXIT_CODE="$?"
fi

FINAL_WIKITEXT2_PPL=""
FINAL_TRAIN_PPL=""
if [[ "$PARSER_EXIT_CODE" == "0" && -f "$METRICS_FILE" ]]; then
    FINAL_METRICS="$(extract_final_metrics)"
    FINAL_WIKITEXT2_PPL="$(printf '%s\n' "$FINAL_METRICS" | sed -n '1p')"
    FINAL_TRAIN_PPL="$(printf '%s\n' "$FINAL_METRICS" | sed -n '2p')"
fi

DROPPED_ATTN_MODULES=""
DROPPED_MLP_MODULES=""
if [[ -f "$CONFIG_FILE" ]]; then
    DROPPED_ATTN_MODULES="$(awk '$0 == "attn" || $0 == "attn+mlp" { count++ } END { print count + 0 }' "$CONFIG_FILE")"
    DROPPED_MLP_MODULES="$(awk '$0 == "mlp" || $0 == "attn+mlp" { count++ } END { print count + 0 }' "$CONFIG_FILE")"
fi

STATUS=completed
NOTES="last_successful_step=final_evaluation; dropped_attn_modules=${DROPPED_ATTN_MODULES}; dropped_mlp_modules=${DROPPED_MLP_MODULES}"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=depth_search_process_started; command_exit_code=${RUN_EXIT_CODE}; parser_exit_code=${PARSER_EXIT_CODE}"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$PARSER_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=depth_search_process_completed; parser_exit_code=${PARSER_EXIT_CODE}"
    FINAL_EXIT_CODE=1
elif [[ ! -f "$CONFIG_FILE" ]]; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; missing_file=${CONFIG_FILE}"
    FINAL_EXIT_CODE=1
elif ! is_finite_number "$FINAL_WIKITEXT2_PPL"; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; non_finite_wikitext2_ppl=${FINAL_WIKITEXT2_PPL}"
    FINAL_EXIT_CODE=1
elif ! is_finite_number "$FINAL_TRAIN_PPL"; then
    STATUS=failed
    NOTES="last_successful_step=metrics_parsed; non_finite_train_ppl=${FINAL_TRAIN_PPL}"
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
