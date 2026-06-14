#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
RESULTS_RUNS_ROOT="${RESULTS_RUNS_ROOT:-results/runs}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
RUN_PREFIX="${RUN_PREFIX:-thesis_medium}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
METHODS="${METHODS:-depth quant joint}"
SEEDS="${SEEDS:-0 1 2}"
RUN_DENSE="${RUN_DENSE:-1}"

DEPTH_SPARSITY="${DEPTH_SPARSITY:-0.25}"
TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
CALIB_TOKENS="${CALIB_TOKENS:-8192}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
EVAL_EVERY="${EVAL_EVERY:-5}"
GENERATIONS="${GENERATIONS:-20}"
OFFSPRING="${OFFSPRING:-16}"
INITIALLY_GENERATED="${INITIALLY_GENERATED:-32}"
INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048 8192}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-8 2 1}"
FITNESS_FN="${FITNESS_FN:-kl}"
GROUP_RULE="${GROUP_RULE:-size}"
ACTIVE_QUANT_BUDGET="${ACTIVE_QUANT_BUDGET:-1}"
JOINT_AWARE_MUTATION="${JOINT_AWARE_MUTATION:-0}"
JOINT_AWARE_PROBABILITY="${JOINT_AWARE_PROBABILITY:-0.5}"
MAX_DROP_MUTATIONS="${MAX_DROP_MUTATIONS:-3}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-30000}"
EXPECTED_QUANT_MODULES="${EXPECTED_QUANT_MODULES:-32}"
EXPECTED_QUANT_WEIGHT_FILES="${EXPECTED_QUANT_WEIGHT_FILES:-96}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_RESULTS="${SYNC_RESULTS:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_medium_grid.sh [--dry-run] [--continue-on-failure]

Run a thesis-scale Mistral comparison sequentially:
  - one dense WikiText2 reference
  - depth-only, quant-only, and joint searches for seeds 0, 1, and 2

The default search uses 20 generations, 16 offspring, 32 initial candidates,
three-stage selection, 25% depth sparsity, and a 3-bit q-projection budget.
Completed structured runs are validated and skipped automatically.
Interrupted or failed non-empty runs are preserved and retried under the next
available `_retryN` run identifier.
By default, completed lightweight artifacts are copied to results/runs.
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
    local method="$1"
    local output_dir="$2"

    if [[ "$method" == "dense" ]]; then
        [[ -f "$output_dir/evaluation_metrics.csv" ]] && runtime_succeeded "$output_dir/runtime.txt"
        return
    fi

    [[ -f "$output_dir/run_summary.json" ]] || return 1
    "$PYTHON_BIN" scripts/validate_run_outputs.py "$output_dir" >/dev/null 2>&1
}

sync_lightweight_artifacts() {
    local method="$1"
    local run_id="$2"
    local output_dir="${OUTPUTS_ROOT}/${run_id}"
    local destination="${RESULTS_RUNS_ROOT}/${run_id}"
    local filename
    local -a filenames=(
        command.sh
        runtime.txt
    )

    if [[ "$SYNC_RESULTS" != "1" || "$DRY_RUN" == "1" ]]; then
        return
    fi
    if [[ "$method" == "dense" ]]; then
        filenames+=(evaluation_metrics.csv)
    else
        filenames+=(
            final_candidate.json
            generation_log.csv
            generation_metrics.csv
            memory_samples.csv
            run_summary.json
        )
        case "$method" in
            depth)
                filenames+=(layer_drop_config.txt)
                ;;
            quant)
                filenames+=(quant_configuration.txt)
                ;;
            joint)
                filenames+=(
                    joint_config.json
                    joint_drop_config.txt
                    joint_quant_config.txt
                )
                ;;
        esac
    fi

    mkdir -p "$destination"
    for filename in "${filenames[@]}"; do
        if [[ -f "$output_dir/$filename" ]]; then
            cp "$output_dir/$filename" "$destination/$filename"
        fi
    done
    printf 'Lightweight artifacts synced to %s\n' "$destination"
}

resolve_run_id() {
    local method="$1"
    local base_run_id="$2"
    local candidate="$base_run_id"
    local retry=0
    local output_dir

    while true; do
        output_dir="${OUTPUTS_ROOT}/${candidate}"
        if run_is_complete "$method" "$output_dir"; then
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

validate_selection_schedule() {
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
    if ((survivors[0] > OFFSPRING)); then
        printf 'First-stage survivors cannot exceed OFFSPRING.\n' >&2
        return 2
    fi
}

validate_hardware_and_database() {
    local gpu_memory_mib
    local weight_files
    local module_dirs

    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_memory_mib="$(
            nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
                head -n 1 |
                tr -d ' '
        )"
        if [[ "$gpu_memory_mib" =~ ^[0-9]+$ ]] && ((gpu_memory_mib < MIN_GPU_MEMORY_MIB)); then
            printf 'This grid requires at least %s MiB GPU memory; detected %s MiB.\n' \
                "$MIN_GPU_MEMORY_MIB" "$gpu_memory_mib" >&2
            return 2
        fi
    fi

    if [[ "$METHODS" == *quant* || "$METHODS" == *joint* ]]; then
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
        if [[ "$module_dirs" != "$EXPECTED_QUANT_MODULES" ||
              "$weight_files" != "$EXPECTED_QUANT_WEIGHT_FILES" ]]; then
            printf 'Incomplete quantization database: modules=%s/%s files=%s/%s\n' \
                "$module_dirs" "$EXPECTED_QUANT_MODULES" \
                "$weight_files" "$EXPECTED_QUANT_WEIGHT_FILES" >&2
            return 2
        fi
    fi
}

run_dense() {
    local base_run_id="${RUN_PREFIX}_dense_mistral_seq${SEQUENCE_LENGTH}_seed0${RUN_SUFFIX}"
    local run_id
    local output_dir
    local exit_code
    local -a args=()

    run_id="$(resolve_run_id dense "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if run_is_complete dense "$output_dir"; then
        printf 'Skipping completed run: %s\n' "$run_id"
        sync_lightweight_artifacts dense "$run_id"
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry-run)
    fi

    env \
        MODEL="$MODEL" \
        SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
        ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
        DTYPE="$DTYPE" \
        SEED=0 \
        EXPERIMENT_LOG="$EXPERIMENT_LOG" \
        OUTPUTS_ROOT="$OUTPUTS_ROOT" \
        RUN_ID="$run_id" \
        OUTPUT_DIR="$output_dir" \
        scripts/run_dense_eval.sh "${args[@]}"
    exit_code="$?"
    if [[ "$exit_code" == "0" ]]; then
        sync_lightweight_artifacts dense "$run_id"
    fi
    return "$exit_code"
}

run_search() {
    local method="$1"
    local seed="$2"
    local base_run_id
    local run_id
    local output_dir
    local exit_code
    local -a args=()

    case "$method" in
        depth)
            base_run_id="${RUN_PREFIX}_depth_mistral_s${DEPTH_SPARSITY}_g${GENERATIONS}_o${OFFSPRING}_seed${seed}${RUN_SUFFIX}"
            ;;
        quant)
            base_run_id="${RUN_PREFIX}_quant_mistral_qproj${TARGET_BITWIDTH}_g${GENERATIONS}_o${OFFSPRING}_seed${seed}${RUN_SUFFIX}"
            ;;
        joint)
            base_run_id="${RUN_PREFIX}_joint_mistral_s${DEPTH_SPARSITY}_qproj${TARGET_BITWIDTH}_g${GENERATIONS}_o${OFFSPRING}_seed${seed}${RUN_SUFFIX}"
            ;;
        *)
            printf 'Unsupported method: %s\n' "$method" >&2
            return 2
            ;;
    esac

    run_id="$(resolve_run_id "$method" "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if run_is_complete "$method" "$output_dir"; then
        printf 'Skipping completed run: %s\n' "$run_id"
        sync_lightweight_artifacts "$method" "$run_id"
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry-run)
    fi

    printf '\n=== Starting %s search: seed=%s run_id=%s ===\n' "$method" "$seed" "$run_id"

    case "$method" in
        depth)
            env \
                MODEL="$MODEL" \
                SPARSITY="$DEPTH_SPARSITY" \
                CALIB_DATA=wikitext2 \
                SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
                CALIB_TOKENS="$CALIB_TOKENS" \
                EVAL_EVERY="$EVAL_EVERY" \
                GENERATIONS="$GENERATIONS" \
                OFFSPRING="$OFFSPRING" \
                INITIALLY_GENERATED="$INITIALLY_GENERATED" \
                INITIAL_TOKENS="$INITIAL_TOKENS" \
                TOKENS_PER_SELECTION="$TOKENS_PER_SELECTION" \
                SURVIVORS_PER_SELECTION="$SURVIVORS_PER_SELECTION" \
                FITNESS_FN="$FITNESS_FN" \
                ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
                DTYPE="$DTYPE" \
                SEED="$seed" \
                EXPERIMENT_LOG="$EXPERIMENT_LOG" \
                OUTPUTS_ROOT="$OUTPUTS_ROOT" \
                RUN_ID="$run_id" \
                OUTPUT_DIR="$output_dir" \
                scripts/run_drop_search.sh "${args[@]}"
            ;;
        quant)
            env \
                MODEL="$MODEL" \
                QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
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
                ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
                DTYPE="$DTYPE" \
                SEED="$seed" \
                EXPERIMENT_LOG="$EXPERIMENT_LOG" \
                OUTPUTS_ROOT="$OUTPUTS_ROOT" \
                RUN_ID="$run_id" \
                OUTPUT_DIR="$output_dir" \
                scripts/run_quant_search_tiny_interesting.sh "${args[@]}"
            ;;
        joint)
            env \
                MODEL="$MODEL" \
                QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
                DROP_SPARSITY="$DEPTH_SPARSITY" \
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
                JOINT_AWARE_MUTATION="$JOINT_AWARE_MUTATION" \
                JOINT_AWARE_PROBABILITY="$JOINT_AWARE_PROBABILITY" \
                MAX_DROP_MUTATIONS="$MAX_DROP_MUTATIONS" \
                ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
                DTYPE="$DTYPE" \
                SEED="$seed" \
                EXPERIMENT_LOG="$EXPERIMENT_LOG" \
                OUTPUTS_ROOT="$OUTPUTS_ROOT" \
                RUN_ID="$run_id" \
                OUTPUT_DIR="$output_dir" \
                scripts/run_joint_search_tiny.sh "${args[@]}"
            ;;
    esac
    exit_code="$?"
    if [[ "$exit_code" == "0" ]]; then
        sync_lightweight_artifacts "$method" "$run_id"
    fi
    return "$exit_code"
}

validate_selection_schedule || exit 2
if [[ "$DRY_RUN" != "1" ]]; then
    validate_hardware_and_database || exit 2
fi

FAILED_RUNS=()

if [[ "$RUN_DENSE" == "1" ]]; then
    run_dense
    run_exit_code="$?"
    if [[ "$run_exit_code" != "0" ]]; then
        FAILED_RUNS+=("dense")
        if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
            exit "$run_exit_code"
        fi
    fi
fi

for seed in $SEEDS; do
    for method in $METHODS; do
        run_search "$method" "$seed"
        run_exit_code="$?"
        if [[ "$run_exit_code" != "0" ]]; then
            FAILED_RUNS+=("${method}:seed${seed}")
            printf 'Run failed: method=%s seed=%s exit_code=%s\n' \
                "$method" "$seed" "$run_exit_code" >&2
            if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
                exit "$run_exit_code"
            fi
        fi
    done
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Grid finished with failed runs: %s\n' "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. No experiments were launched.\n'
else
    printf '\nMistral medium grid complete.\n'
fi
