#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_tinyllama_qproj_bits234/quant_db/TinyLlama-1.1B-Chat-v1.0/3bit}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RESULTS_RUNS_ROOT="${RESULTS_RUNS_ROOT:-results/runs}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
RUN_PREFIX="${RUN_PREFIX:-screen_fixedstrength_tiny}"
SEEDS="${SEEDS:-0 1 2}"

DROP_SPARSITY="${DROP_SPARSITY:-0.125}"
TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
CALIB_TOKENS="${CALIB_TOKENS:-4096}"
EVAL_TOKENS="${EVAL_TOKENS:-4096}"
EVAL_EVERY="${EVAL_EVERY:-2}"
GENERATIONS="${GENERATIONS:-20}"
OFFSPRING="${OFFSPRING:-8}"
INITIALLY_GENERATED="${INITIALLY_GENERATED:-16}"
INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-2 1}"
FITNESS_FN="${FITNESS_FN:-kl}"
GROUP_RULE="${GROUP_RULE:-size}"
ACTIVE_QUANT_BUDGET="${ACTIVE_QUANT_BUDGET:-1}"
FIXED_MUTATION_STRENGTH="${FIXED_MUTATION_STRENGTH:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_RUNTIME_DEPENDENCIES="${CHECK_RUNTIME_DEPENDENCIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_RESULTS="${SYNC_RESULTS:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/run_fixed_mutation_strength_screen.sh [--dry-run] [--continue-on-failure]

Run a three-seed TinyLlama fixed-local-mutation control. By default, every
depth offspring performs exactly one swap and every quantization offspring
performs one exchange. Adaptive and joint-aware mutation are disabled.

This control isolates mutation locality from adaptive scheduling.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --continue-on-failure)
            CONTINUE_ON_FAILURE=1
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

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

runtime_succeeded() {
    local runtime_file="$1"
    [[ -f "$runtime_file" ]] && grep -qx 'exit_code=0' "$runtime_file"
}

run_is_complete() {
    local output_dir="$1"
    [[ -f "$output_dir/run_summary.json" ]] || return 1
    runtime_succeeded "$output_dir/runtime.txt" || return 1
    "$PYTHON_BIN" scripts/validate_run_outputs.py "$output_dir" >/dev/null 2>&1
}

resolve_run_id() {
    local base_run_id="$1"
    local candidate="$base_run_id"
    local retry=0
    local output_dir

    while true; do
        output_dir="${OUTPUTS_ROOT}/${candidate}"
        if run_is_complete "$output_dir"; then
            printf '%s' "$candidate"
            return
        fi
        if ! directory_has_files "$output_dir"; then
            printf '%s' "$candidate"
            return
        fi
        retry=$((retry + 1))
        candidate="${base_run_id}_retry${retry}"
    done
}

sync_lightweight_artifacts() {
    local run_id="$1"
    local source="${OUTPUTS_ROOT}/${run_id}"
    local destination="${RESULTS_RUNS_ROOT}/${run_id}"
    local filename
    local -a filenames=(
        command.sh
        final_candidate.json
        generation_log.csv
        generation_metrics.csv
        joint_config.json
        joint_drop_config.txt
        joint_quant_config.txt
        memory_samples.csv
        run_summary.json
        runtime.txt
    )

    if [[ "$SYNC_RESULTS" != "1" || "$DRY_RUN" == "1" ]]; then
        return
    fi

    mkdir -p "$destination"
    for filename in "${filenames[@]}"; do
        if [[ -f "$source/$filename" ]]; then
            cp "$source/$filename" "$destination/$filename"
        fi
    done
    printf 'Lightweight artifacts synced to %s\n' "$destination"
}

validate_configuration() {
    local -a tokens survivors
    local last_index
    read -r -a tokens <<< "$TOKENS_PER_SELECTION"
    read -r -a survivors <<< "$SURVIVORS_PER_SELECTION"

    if ((${#tokens[@]} != ${#survivors[@]})); then
        printf 'TOKENS_PER_SELECTION and SURVIVORS_PER_SELECTION must have equal lengths.\n' >&2
        return 2
    fi
    if ((${#survivors[@]} == 0)); then
        printf 'At least one selection stage is required.\n' >&2
        return 2
    fi
    last_index=$((${#survivors[@]} - 1))
    if [[ "${survivors[$last_index]}" != "1" ]]; then
        printf 'The final survivor count must be 1.\n' >&2
        return 2
    fi
    if [[ "$ACTIVE_QUANT_BUDGET" != "1" || "$GROUP_RULE" != "size" ]]; then
        printf 'The control requires ACTIVE_QUANT_BUDGET=1 and GROUP_RULE=size.\n' >&2
        return 2
    fi
    if ((FIXED_MUTATION_STRENGTH < 1)); then
        printf 'FIXED_MUTATION_STRENGTH must be at least 1.\n' >&2
        return 2
    fi
}

validate_database() {
    local module_dirs
    local weight_files

    if [[ ! -d "$QUANT_WEIGHTS_PATH" ]]; then
        printf 'Quantization database does not exist: %s\n' "$QUANT_WEIGHTS_PATH" >&2
        return 2
    fi
    module_dirs="$(
        find "$QUANT_WEIGHTS_PATH" -mindepth 1 -maxdepth 1 -type d |
            wc -l |
            tr -d ' '
    )"
    weight_files="$(
        find "$QUANT_WEIGHTS_PATH" -mindepth 2 -maxdepth 2 -type f -name '*.pth' |
            wc -l |
            tr -d ' '
    )"
    if [[ "$module_dirs" != "22" || "$weight_files" != "66" ]]; then
        printf 'Incomplete TinyLlama q_proj database: modules=%s/22 files=%s/66\n' \
            "$module_dirs" "$weight_files" >&2
        return 2
    fi
}

run_fixed_strength() {
    local seed="$1"
    local base_run_id
    local run_id
    local output_dir
    local exit_code
    local -a args=()

    base_run_id="${RUN_PREFIX}_max${FIXED_MUTATION_STRENGTH}_g${GENERATIONS}_o${OFFSPRING}_seed${seed}"
    run_id="$(resolve_run_id "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"

    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if run_is_complete "$output_dir"; then
        printf 'Skipping completed run: %s\n' "$run_id"
        sync_lightweight_artifacts "$run_id"
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry-run)
    fi

    printf '\n=== Starting fixed-strength TinyLlama control: seed=%s run_id=%s ===\n' \
        "$seed" "$run_id"

    env \
        MODEL="$MODEL" \
        QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
        DROP_SPARSITY="$DROP_SPARSITY" \
        TARGET_BITWIDTH="$TARGET_BITWIDTH" \
        CALIB_DATA=wikitext2 \
        SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
        CALIB_TOKENS="$CALIB_TOKENS" \
        EVAL_TOKENS="$EVAL_TOKENS" \
        EVAL_EVERY="$EVAL_EVERY" \
        GENERATIONS="$GENERATIONS" \
        OFFSPRING="$OFFSPRING" \
        INITIALLY_GENERATED="$INITIALLY_GENERATED" \
        INITIAL_TOKENS="$INITIAL_TOKENS" \
        TOKENS_PER_SELECTION="$TOKENS_PER_SELECTION" \
        SURVIVORS_PER_SELECTION="$SURVIVORS_PER_SELECTION" \
        FITNESS_FN="$FITNESS_FN" \
        GROUP_RULE="$GROUP_RULE" \
        ACTIVE_QUANT_BUDGET="$ACTIVE_QUANT_BUDGET" \
        JOINT_AWARE_MUTATION=0 \
        ADAPTIVE_MUTATION=0 \
        MAX_DROP_MUTATIONS="$FIXED_MUTATION_STRENGTH" \
        ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
        DTYPE="$DTYPE" \
        SEED="$seed" \
        PYTHON_BIN="$PYTHON_BIN" \
        CHECK_RUNTIME_DEPENDENCIES="$CHECK_RUNTIME_DEPENDENCIES" \
        EXPERIMENT_LOG="$EXPERIMENT_LOG" \
        OUTPUTS_ROOT="$OUTPUTS_ROOT" \
        RUN_ID="$run_id" \
        OUTPUT_DIR="$output_dir" \
        scripts/run_joint_search_tiny.sh "${args[@]}"
    exit_code="$?"

    if [[ "$exit_code" == "0" ]]; then
        sync_lightweight_artifacts "$run_id"
    fi
    return "$exit_code"
}

validate_configuration || exit 2
if [[ "$DRY_RUN" != "1" ]]; then
    validate_database || exit 2
fi

FAILED_RUNS=()
for seed in $SEEDS; do
    run_fixed_strength "$seed"
    exit_code="$?"
    if [[ "$exit_code" != "0" ]]; then
        FAILED_RUNS+=("fixed:seed${seed}")
        printf 'Fixed-strength run failed: seed=%s exit_code=%s\n' \
            "$seed" "$exit_code" >&2
        if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
            exit "$exit_code"
        fi
    fi
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Fixed-strength screen finished with failed runs: %s\n' \
        "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. No experiments were launched.\n'
else
    printf '\nFixed mutation-strength screen complete.\n'
    printf 'Lightweight artifacts are ready under %s.\n' "$RESULTS_RUNS_ROOT"
fi
