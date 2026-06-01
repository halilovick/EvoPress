#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
METHOD="${METHOD:-random}"
SPARSITY="${SPARSITY:-0.125}"
SEED="${SEED:-1}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1}"
CALIB_DATA="${CALIB_DATA:-wikitext2}"
CALIB_TOKENS="${CALIB_TOKENS:-8192}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
PROTECT_LAYER_ZERO="${PROTECT_LAYER_ZERO:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_EVAL_SCRIPT="${BASELINE_EVAL_SCRIPT:-scripts/evaluate_depth_baselines.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
BASELINE_RESULTS_LOG="${BASELINE_RESULTS_LOG:-results/depth_baseline_runs.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-results/runs/depth_mistral7b_s${SPARSITY}_seed1/layer_drop_config.txt}"
RUN_ID="${RUN_ID:-baseline_${METHOD}_mistral7b_s${SPARSITY}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/run_depth_baseline.sh [--dry-run]

Evaluate one cheap depth-pruning baseline matched to an EvoPress config.
Override parameters with environment variables such as METHOD, SPARSITY,
SEED, REFERENCE_CONFIG, or OUTPUT_DIR.

Examples:
  METHOD=random SPARSITY=0.125 SEED=1 scripts/run_depth_baseline.sh --dry-run
  METHOD=random SPARSITY=0.125 SEED=1 scripts/run_depth_baseline.sh
  METHOD=late_layer SPARSITY=0.25 SEED=1 scripts/run_depth_baseline.sh
  METHOD=early_layer SPARSITY=0.25 SEED=1 PROTECT_LAYER_ZERO=1 scripts/run_depth_baseline.sh
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

case "$METHOD" in
    random|late_layer|early_layer) ;;
    *)
        printf 'Unsupported baseline METHOD: %s\n' "$METHOD" >&2
        exit 2
        ;;
esac

RUN_LOG="${OUTPUT_DIR}/run.log"
RUNTIME_FILE="${OUTPUT_DIR}/runtime.txt"
COMMAND_FILE="${OUTPUT_DIR}/command.sh"
METRICS_FILE="${OUTPUT_DIR}/baseline_metrics.csv"

COMMAND=(
    "$PYTHON_BIN" "$BASELINE_EVAL_SCRIPT"
    --model_name_or_path "$MODEL"
    --reference_config "$REFERENCE_CONFIG"
    --sparsity "$SPARSITY"
    --method "$METHOD"
    --calibration_data "$CALIB_DATA"
    --calibration_tokens "$CALIB_TOKENS"
    --sequence_length "$SEQUENCE_LENGTH"
    --seed "$SEED"
    --calibration_seed "$CALIBRATION_SEED"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --use_fast_tokenizer
    --run_id "$RUN_ID"
    --output_dir "$OUTPUT_DIR"
)

if [[ "$PROTECT_LAYER_ZERO" == "1" ]]; then
    COMMAND+=(--protect_layer_zero)
fi

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

extract_metrics() {
    "$PYTHON_BIN" - "$METRICS_FILE" "$METHOD" "$SPARSITY" "$SEED" <<'PY'
import csv
import math
import sys

metrics_file, expected_method, expected_sparsity, expected_seed = sys.argv[1:]
with open(metrics_file, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(1)
row = rows[0]
if row["method"] != expected_method or row["sparsity"] != expected_sparsity or row["seed"] != expected_seed:
    raise SystemExit(1)
if not math.isfinite(float(row["wikitext2_ppl"])) or not math.isfinite(float(row["train_ppl"])):
    raise SystemExit(1)
print(row["wikitext2_ppl"])
print(row["train_ppl"])
print(row["dropped_attn_modules"])
print(row["dropped_mlp_modules"])
PY
}

append_result_rows() {
    local status="$1"
    local notes="$2"
    local runtime_minutes="$3"
    local wikitext2_ppl="$4"
    local train_ppl="$5"
    local gpu_name="$6"
    local gpu_vram_gb="$7"
    local cpu_ram_limit_gb="$8"

    "$PYTHON_BIN" scripts/append_depth_baseline_result.py \
        --log-file "$BASELINE_RESULTS_LOG" \
        --sparsity "$SPARSITY" \
        --method "$METHOD" \
        --seed "$SEED" \
        --wikitext2-ppl "$wikitext2_ppl" \
        --train-ppl "$train_ppl" \
        --runtime-minutes "$runtime_minutes" \
        --notes "status=${status}; ${notes}" \
        --output-dir "$OUTPUT_DIR"

    "$PYTHON_BIN" scripts/append_experiment_log.py \
        --log-file "$EXPERIMENT_LOG" \
        --run-id "$RUN_ID" \
        --method "depth_baseline_${METHOD}" \
        --model "$MODEL" \
        --sparsity-or-bits "$SPARSITY" \
        --calibration-data "$CALIB_DATA" \
        --sequence-length "$SEQUENCE_LENGTH" \
        --calibration-tokens "$CALIB_TOKENS" \
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

if [[ ! -f "$REFERENCE_CONFIG" ]]; then
    printf 'Reference EvoPress config does not exist: %s\n' "$REFERENCE_CONFIG" >&2
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
    printf 'reference_config=%s\n' "$REFERENCE_CONFIG"
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

WIKITEXT2_PPL=""
TRAIN_PPL=""
DROPPED_ATTN_MODULES=""
DROPPED_MLP_MODULES=""
METRICS_EXIT_CODE=0
if [[ "$RUN_EXIT_CODE" == "0" && -f "$METRICS_FILE" ]]; then
    METRICS="$(extract_metrics)" || METRICS_EXIT_CODE="$?"
    if [[ "$METRICS_EXIT_CODE" == "0" ]]; then
        WIKITEXT2_PPL="$(printf '%s\n' "$METRICS" | sed -n '1p')"
        TRAIN_PPL="$(printf '%s\n' "$METRICS" | sed -n '2p')"
        DROPPED_ATTN_MODULES="$(printf '%s\n' "$METRICS" | sed -n '3p')"
        DROPPED_MLP_MODULES="$(printf '%s\n' "$METRICS" | sed -n '4p')"
    fi
elif [[ "$RUN_EXIT_CODE" == "0" ]]; then
    METRICS_EXIT_CODE=1
fi

STATUS=completed
NOTES="last_successful_step=final_evaluation; reference_config=${REFERENCE_CONFIG}; calibration_seed=${CALIBRATION_SEED}; dropped_attn_modules=${DROPPED_ATTN_MODULES}; dropped_mlp_modules=${DROPPED_MLP_MODULES}; protect_layer_zero=${PROTECT_LAYER_ZERO}"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=baseline_evaluation_process_started; command_exit_code=${RUN_EXIT_CODE}; reference_config=${REFERENCE_CONFIG}; calibration_seed=${CALIBRATION_SEED}; protect_layer_zero=${PROTECT_LAYER_ZERO}"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$METRICS_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=baseline_evaluation_process_completed; metrics_exit_code=${METRICS_EXIT_CODE}; reference_config=${REFERENCE_CONFIG}; calibration_seed=${CALIBRATION_SEED}; protect_layer_zero=${PROTECT_LAYER_ZERO}"
    FINAL_EXIT_CODE=1
fi

append_result_rows \
    "$STATUS" \
    "$NOTES" \
    "$RUNTIME_MINUTES" \
    "$WIKITEXT2_PPL" \
    "$TRAIN_PPL" \
    "$GPU_NAME" \
    "$GPU_VRAM_GB" \
    "$CPU_RAM_LIMIT_GB"

printf 'Experiment %s finished with status=%s.\n' "$RUN_ID" "$STATUS"
printf 'Artifacts: %s\n' "$OUTPUT_DIR"
exit "$FINAL_EXIT_CODE"
