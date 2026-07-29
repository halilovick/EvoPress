#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RESULTS_RUNS_ROOT="${RESULTS_RUNS_ROOT:-results/runs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
SEEDS="${SEEDS:-0 1 2}"
CONDITIONS="${CONDITIONS:-standard_standard depthwarm_standard standard_interaction depthwarm_interaction}"

DROP_SPARSITY="${DROP_SPARSITY:-0.25}"
TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
CALIB_DATA="${CALIB_DATA:-wikitext2}"
CALIB_TOKENS="${CALIB_TOKENS:-8192}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2}"
EVAL_EVERY="${EVAL_EVERY:-5}"
GENERATIONS="${GENERATIONS:-50}"
OFFSPRING="${OFFSPRING:-16}"
INITIALLY_GENERATED="${INITIALLY_GENERATED:-32}"
INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048 8192}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-8 2 1}"
FITNESS_FN="${FITNESS_FN:-kl}"
GROUP_RULE="${GROUP_RULE:-size}"
ACTIVE_QUANT_BUDGET="${ACTIVE_QUANT_BUDGET:-1}"
MAX_DROP_MUTATIONS="${MAX_DROP_MUTATIONS:-3}"
STEP_SIZE="${STEP_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-1}"
MAX_INITIALIZATION_ATTEMPTS="${MAX_INITIALIZATION_ATTEMPTS:-100000}"
MAX_OFFSPRING_ATTEMPTS="${MAX_OFFSPRING_ATTEMPTS:-10000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_QUANT_MODULES="${EXPECTED_QUANT_MODULES:-32}"
EXPECTED_QUANT_WEIGHT_FILES="${EXPECTED_QUANT_WEIGHT_FILES:-96}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-30000}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
SYNC_RESULTS="${SYNC_RESULTS:-1}"
GENERATE_SUMMARY="${GENERATE_SUMMARY:-1}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run_depth_warmstart_g50_grid.sh \
    [--dry-run] [--continue-on-failure] [--summarize-only] [--no-summary]

Run the matched 4-condition x 3-seed Mistral q_proj G50 matrix:
  standard_standard:
      standard initialization + standard mutation
  depthwarm_standard:
      depth-to-joint warm start + standard mutation
  standard_interaction:
      standard initialization + interaction-aware mutation
  depthwarm_interaction:
      depth-to-joint warm start + interaction-aware mutation

The launcher reuses valid tracked or output artifacts, preserves incomplete
runs under retry identifiers, and runs conditions sequentially. It never uses
the legacy --joint_aware_mutation path.

Environment overrides include SEEDS, CONDITIONS, MODEL, QUANT_WEIGHTS_PATH,
OUTPUTS_ROOT, RESULTS_RUNS_ROOT, RESULTS_DIR, GENERATIONS, OFFSPRING,
CALIB_TOKENS, TOKENS_PER_SELECTION, SURVIVORS_PER_SELECTION, and EVAL_TOKENS.
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
        --summarize-only)
            SUMMARIZE_ONLY=1
            ;;
        --no-summary)
            GENERATE_SUMMARY=0
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
    local run_dir="$1"
    [[ -f "$run_dir/run_summary.json" ]] || return 1
    [[ -f "$run_dir/final_candidate.json" ]] || return 1
    runtime_succeeded "$run_dir/runtime.txt" || return 1
    "$PYTHON_BIN" scripts/validate_run_outputs.py "$run_dir" >/dev/null 2>&1
}

condition_run_is_complete() {
    local run_dir="$1"
    local condition="$2"
    local seed="$3"
    run_is_complete "$run_dir" || return 1
    "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from scripts.summarize_depth_warmstart_g50 import run_row_and_convergence

run_row_and_convergence(
    sys.argv[2],
    int(sys.argv[3]),
    Path(sys.argv[1]),
    Path(sys.argv[4]),
)
' "$run_dir" "$condition" "$seed" "$RESULTS_RUNS_ROOT" >/dev/null 2>&1
}

find_complete_run() {
    local base_run_id="$1"
    local condition="$2"
    local seed="$3"
    local retry
    local candidate
    local root

    for root in "$RESULTS_RUNS_ROOT" "$OUTPUTS_ROOT"; do
        if condition_run_is_complete \
            "$root/$base_run_id" "$condition" "$seed"; then
            printf '%s|%s' "$base_run_id" "$root/$base_run_id"
            return 0
        fi
        for retry in $(seq 1 20); do
            candidate="${base_run_id}_retry${retry}"
            if condition_run_is_complete \
                "$root/$candidate" "$condition" "$seed"; then
                printf '%s|%s' "$candidate" "$root/$candidate"
                return 0
            fi
        done
    done
    return 1
}

resolve_new_run_id() {
    local base_run_id="$1"
    local candidate="$base_run_id"
    local retry=0

    while directory_has_files "$OUTPUTS_ROOT/$candidate" ||
        directory_has_files "$RESULTS_RUNS_ROOT/$candidate"; do
        retry=$((retry + 1))
        candidate="${base_run_id}_retry${retry}"
    done
    printf '%s' "$candidate"
}

sync_lightweight_artifacts() {
    local run_id="$1"
    local source_dir="$2"
    local destination="$RESULTS_RUNS_ROOT/$run_id"
    local filename
    local -a filenames=(
        command.sh
        runtime.txt
        final_candidate.json
        generation_log.csv
        generation_metrics.csv
        memory_samples.csv
        run_summary.json
        joint_config.json
        joint_drop_config.txt
        joint_quant_config.txt
    )

    if [[ "$SYNC_RESULTS" != "1" || "$DRY_RUN" == "1" ]]; then
        return
    fi
    if [[ "$source_dir" == "$destination" ]]; then
        return
    fi

    mkdir -p "$destination"
    for filename in "${filenames[@]}"; do
        if [[ ! -f "$source_dir/$filename" ]]; then
            printf 'Missing completed-run artifact: %s\n' \
                "$source_dir/$filename" >&2
            return 2
        fi
        cp "$source_dir/$filename" "$destination/$filename"
    done
    printf 'Lightweight artifacts synced to %s\n' "$destination"
}

base_run_id_for_condition() {
    local condition="$1"
    local seed="$2"
    case "$condition" in
        standard_standard)
            printf 'thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed%s' "$seed"
            ;;
        depthwarm_standard)
            printf 'thesis_depthwarm_standard_joint_mistral_s0.25_qproj3.0_g50_o16_seed%s' "$seed"
            ;;
        standard_interaction)
            printf 'thesis_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed%s' "$seed"
            ;;
        depthwarm_interaction)
            printf 'thesis_depthwarm_interactionaware_joint_mistral_s0.25_qproj3.0_g50_o16_seed%s' "$seed"
            ;;
        *)
            printf 'Unsupported condition: %s\n' "$condition" >&2
            return 2
            ;;
    esac
}

validate_configuration() {
    local -a tokens survivors
    local condition
    local last_survivor_index
    read -r -a tokens <<< "$TOKENS_PER_SELECTION"
    read -r -a survivors <<< "$SURVIVORS_PER_SELECTION"

    if ((${#tokens[@]} != ${#survivors[@]})); then
        printf 'TOKENS_PER_SELECTION and SURVIVORS_PER_SELECTION must have equal lengths.\n' >&2
        return 2
    fi
    if ((${#survivors[@]} == 0)); then
        printf 'The selection schedule must be non-empty.\n' >&2
        return 2
    fi
    last_survivor_index=$((${#survivors[@]} - 1))
    if [[ "${survivors[$last_survivor_index]}" != "1" ]]; then
        printf 'The selection schedule must be non-empty and end with one survivor.\n' >&2
        return 2
    fi
    if [[ "$ACTIVE_QUANT_BUDGET" != "1" || "$GROUP_RULE" != "size" ]]; then
        printf 'This experiment requires ACTIVE_QUANT_BUDGET=1 and GROUP_RULE=size.\n' >&2
        return 2
    fi
    if [[ "$GENERATIONS" != "50" || "$OFFSPRING" != "16" ]]; then
        printf 'This matched experiment requires GENERATIONS=50 and OFFSPRING=16.\n' >&2
        return 2
    fi
    for condition in $CONDITIONS; do
        case "$condition" in
            standard_standard|depthwarm_standard|standard_interaction|depthwarm_interaction)
                ;;
            *)
                printf 'Unsupported condition in CONDITIONS: %s\n' "$condition" >&2
                return 2
                ;;
        esac
    done
}

validate_hardware_and_database() {
    local gpu_memory_mib
    local module_dirs
    local weight_files

    "$PYTHON_BIN" scripts/check_runtime_dependencies.py --require-cuda || return 2
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_memory_mib="$(
            nvidia-smi --query-gpu=memory.total \
                --format=csv,noheader,nounits 2>/dev/null |
                head -n 1 |
                tr -d ' '
        )"
        if [[ "$gpu_memory_mib" =~ ^[0-9]+$ ]] &&
            ((gpu_memory_mib < MIN_GPU_MEMORY_MIB)); then
            printf 'At least %s MiB GPU memory is required; detected %s MiB.\n' \
                "$MIN_GPU_MEMORY_MIB" "$gpu_memory_mib" >&2
            return 2
        fi
    fi
    if [[ ! -d "$QUANT_WEIGHTS_PATH" ]]; then
        printf 'Quantization database does not exist: %s\n' \
            "$QUANT_WEIGHTS_PATH" >&2
        return 2
    fi
    module_dirs="$(
        find "$QUANT_WEIGHTS_PATH" -mindepth 1 -maxdepth 1 -type d |
            wc -l |
            tr -d ' '
    )"
    weight_files="$(
        find "$QUANT_WEIGHTS_PATH" -mindepth 2 -maxdepth 2 \
            -type f -name '*.pth' |
            wc -l |
            tr -d ' '
    )"
    if [[ "$module_dirs" != "$EXPECTED_QUANT_MODULES" ||
        "$weight_files" != "$EXPECTED_QUANT_WEIGHT_FILES" ]]; then
        printf 'Incomplete q_proj database: modules=%s/%s files=%s/%s\n' \
            "$module_dirs" "$EXPECTED_QUANT_MODULES" \
            "$weight_files" "$EXPECTED_QUANT_WEIGHT_FILES" >&2
        return 2
    fi
}

validate_stage1_run() {
    local seed="$1"
    local run_dir="$RESULTS_RUNS_ROOT/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}"
    if ! run_is_complete "$run_dir"; then
        printf 'Missing or invalid seed-matched depth stage-one run: %s\n' \
            "$run_dir" >&2
        return 2
    fi
    printf '%s' "$run_dir"
}

run_condition() {
    local condition="$1"
    local seed="$2"
    local base_run_id
    local complete
    local complete_run_id
    local complete_dir
    local run_id
    local output_dir
    local mutation_mode=standard
    local sequential_mode=none
    local stage1_run_dir=""
    local exit_code
    local -a launcher_args=()

    base_run_id="$(base_run_id_for_condition "$condition" "$seed")" || return 2
    if complete="$(find_complete_run "$base_run_id" "$condition" "$seed")"; then
        complete_run_id="${complete%%|*}"
        complete_dir="${complete#*|}"
        printf 'Skipping valid completed run: condition=%s seed=%s run_id=%s\n' \
            "$condition" "$seed" "$complete_run_id"
        sync_lightweight_artifacts "$complete_run_id" "$complete_dir"
        return
    fi

    case "$condition" in
        depthwarm_standard)
            sequential_mode=depth_to_joint_warm
            stage1_run_dir="$(validate_stage1_run "$seed")" || return 2
            ;;
        standard_interaction)
            mutation_mode=interaction_aware
            ;;
        depthwarm_interaction)
            sequential_mode=depth_to_joint_warm
            mutation_mode=interaction_aware
            stage1_run_dir="$(validate_stage1_run "$seed")" || return 2
            ;;
    esac

    run_id="$(resolve_new_run_id "$base_run_id")"
    output_dir="$OUTPUTS_ROOT/$run_id"
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        launcher_args+=(--dry-run)
    fi

    printf '\n=== Starting condition=%s seed=%s run_id=%s ===\n' \
        "$condition" "$seed" "$run_id"

    if [[ "$sequential_mode" == "none" ]]; then
        env \
            MODEL="$MODEL" \
            QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
            DROP_SPARSITY="$DROP_SPARSITY" \
            TARGET_BITWIDTH="$TARGET_BITWIDTH" \
            CALIB_DATA="$CALIB_DATA" \
            CALIB_TOKENS="$CALIB_TOKENS" \
            SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
            EVAL_TOKENS="$EVAL_TOKENS" \
            EVAL_DATASETS="$EVAL_DATASETS" \
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
            JOINT_MUTATION_MODE="$mutation_mode" \
            JOINT_AWARE_MUTATION=0 \
            ADAPTIVE_MUTATION=0 \
            COARSE_TO_FINE_MUTATION=0 \
            MAX_DROP_MUTATIONS="$MAX_DROP_MUTATIONS" \
            STEP_SIZE="$STEP_SIZE" \
            ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
            DTYPE="$DTYPE" \
            USE_FAST_TOKENIZER="$USE_FAST_TOKENIZER" \
            SEED="$seed" \
            PYTHON_BIN="$PYTHON_BIN" \
            EXPERIMENT_LOG="$EXPERIMENT_LOG" \
            RUN_ID="$run_id" \
            OUTPUT_DIR="$output_dir" \
            scripts/run_joint_search_tiny.sh "${launcher_args[@]}"
    else
        env \
            MODEL="$MODEL" \
            QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
            DROP_SPARSITY="$DROP_SPARSITY" \
            TARGET_BITWIDTH="$TARGET_BITWIDTH" \
            CALIB_DATA="$CALIB_DATA" \
            CALIB_TOKENS="$CALIB_TOKENS" \
            SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
            EVAL_TOKENS="$EVAL_TOKENS" \
            EVAL_DATASETS="$EVAL_DATASETS" \
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
            JOINT_MUTATION_MODE="$mutation_mode" \
            JOINT_AWARE_MUTATION=0 \
            ADAPTIVE_MUTATION=0 \
            COARSE_TO_FINE_MUTATION=0 \
            MAX_DROP_MUTATIONS="$MAX_DROP_MUTATIONS" \
            STEP_SIZE="$STEP_SIZE" \
            ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
            DTYPE="$DTYPE" \
            USE_FAST_TOKENIZER="$USE_FAST_TOKENIZER" \
            SEED="$seed" \
            PYTHON_BIN="$PYTHON_BIN" \
            EXPERIMENT_LOG="$EXPERIMENT_LOG" \
            MAX_INITIALIZATION_ATTEMPTS="$MAX_INITIALIZATION_ATTEMPTS" \
            MAX_OFFSPRING_ATTEMPTS="$MAX_OFFSPRING_ATTEMPTS" \
            scripts/run_sequential_search.sh \
            --mode depth_to_joint_warm \
            --stage1-run-dir "$stage1_run_dir" \
            --output-dir "$output_dir" \
            --policy strict \
            "${launcher_args[@]}"
    fi
    exit_code="$?"
    if [[ "$exit_code" == "0" && "$DRY_RUN" != "1" ]]; then
        if ! condition_run_is_complete "$output_dir" "$condition" "$seed"; then
            printf 'Launcher exited successfully but output validation failed: %s\n' \
                "$output_dir" >&2
            return 1
        fi
        sync_lightweight_artifacts "$run_id" "$output_dir" || return
    fi
    return "$exit_code"
}

generate_summary() {
    "$PYTHON_BIN" scripts/summarize_depth_warmstart_g50.py \
        --runs-root "$RESULTS_RUNS_ROOT" \
        --output-dir "$RESULTS_DIR"
}

validate_configuration || exit 2

if [[ "$SUMMARIZE_ONLY" == "1" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s\n' '--summarize-only and --dry-run cannot be combined.' >&2
        exit 2
    fi
    generate_summary
    exit "$?"
fi

if [[ "$DRY_RUN" != "1" ]]; then
    validate_hardware_and_database || exit 2
fi

FAILED_RUNS=()
for seed in $SEEDS; do
    for condition in $CONDITIONS; do
        run_condition "$condition" "$seed"
        run_exit_code="$?"
        if [[ "$run_exit_code" != "0" ]]; then
            FAILED_RUNS+=("${condition}:seed${seed}")
            printf 'Run failed: condition=%s seed=%s exit_code=%s\n' \
                "$condition" "$seed" "$run_exit_code" >&2
            if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
                exit "$run_exit_code"
            fi
        fi
    done
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Matrix finished with failed runs: %s\n' "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. No experiments were launched.\n'
elif [[ "$GENERATE_SUMMARY" == "1" ]]; then
    generate_summary
else
    printf '\nG50 depth-warmstart matrix complete; summary generation disabled.\n'
fi
