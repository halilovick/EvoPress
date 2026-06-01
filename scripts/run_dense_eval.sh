#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PPL_SCRIPT="${EVAL_PPL_SCRIPT:-eval_ppl.py}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RUN_ID="${RUN_ID:-dense_mistral7b_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_ROOT}/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/run_dense_eval.sh [--dry-run]

Evaluate the dense Mistral-7B WikiText2 perplexity reference. Override
parameters with environment variables such as MODEL, SEQUENCE_LENGTH, SEED,
or OUTPUT_DIR.

Examples:
  scripts/run_dense_eval.sh --dry-run
  scripts/run_dense_eval.sh
  RUN_ID=dense_mistral7b_seed1_retry1 scripts/run_dense_eval.sh
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

COMMAND=(
    "$PYTHON_BIN" "$EVAL_PPL_SCRIPT"
    --model_name_or_path "$MODEL"
    --eval_datasets wikitext2
    --sequence_length "$SEQUENCE_LENGTH"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --use_fast_tokenizer
    --seed "$SEED"
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
        --method dense \
        --model "$MODEL" \
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
    "$PYTHON_BIN" scripts/parse_eval_ppl_log.py \
        --log "$RUN_LOG" \
        --output "$METRICS_FILE" \
        --run-id "$RUN_ID" || PARSER_EXIT_CODE="$?"
fi

WIKITEXT2_PPL=""
if [[ "$PARSER_EXIT_CODE" == "0" && -f "$METRICS_FILE" ]]; then
    WIKITEXT2_PPL="$(extract_wikitext2_ppl)"
fi

STATUS=completed
NOTES="last_successful_step=final_evaluation"
FINAL_EXIT_CODE=0

if [[ "$RUN_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=dense_evaluation_process_started; command_exit_code=${RUN_EXIT_CODE}"
    FINAL_EXIT_CODE="$RUN_EXIT_CODE"
elif [[ "$PARSER_EXIT_CODE" != "0" ]]; then
    STATUS=failed
    NOTES="last_successful_step=dense_evaluation_process_completed; parser_exit_code=${PARSER_EXIT_CODE}"
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
