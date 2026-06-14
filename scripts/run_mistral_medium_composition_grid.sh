#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-results/runs}"
RESULTS_RUNS_ROOT="${RESULTS_RUNS_ROOT:-results/runs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/experiments}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-results/experiment_log.csv}"
SOURCE_PREFIX="${SOURCE_PREFIX:-thesis_medium}"
RUN_PREFIX="${RUN_PREFIX:-thesis_medium}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-independent uniform}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_medium_composition_grid.sh [--dry-run] [--continue-on-failure]

Evaluate the missing matched Mistral medium-grid controls for seeds 0, 1, 2:
  independent: independently searched depth mask + searched q_proj profile
  uniform:     independently searched depth mask + uniform 3-bit q_proj

The launcher uses the tracked source configs under results/runs, evaluates the
full WikiText2 split at sequence length 1024, records experiment-log rows, and
copies lightweight completed artifacts into results/runs for the next sync.
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
    [[ -f "$output_dir/evaluation_metrics.csv" ]] &&
        runtime_succeeded "$output_dir/runtime.txt"
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

validate_inputs() {
    local seed
    local depth_config
    local quant_config
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
    if [[ "$module_dirs" != "32" || "$weight_files" != "96" ]]; then
        printf 'Incomplete q_proj quantization database: modules=%s/32 files=%s/96\n' \
            "$module_dirs" "$weight_files" >&2
        return 2
    fi

    for seed in $SEEDS; do
        depth_config="${SOURCE_RUNS_ROOT}/${SOURCE_PREFIX}_depth_mistral_s0.25_g20_o16_seed${seed}/layer_drop_config.txt"
        quant_config="${SOURCE_RUNS_ROOT}/${SOURCE_PREFIX}_quant_mistral_qproj3.0_g20_o16_seed${seed}/quant_configuration.txt"
        [[ -f "$depth_config" ]] || {
            printf 'Missing depth config: %s\n' "$depth_config" >&2
            return 2
        }
        [[ -f "$quant_config" ]] || {
            printf 'Missing quant config: %s\n' "$quant_config" >&2
            return 2
        }
    done
}

sync_lightweight_artifacts() {
    local run_id="$1"
    local output_dir="${OUTPUTS_ROOT}/${run_id}"
    local destination="${RESULTS_RUNS_ROOT}/${run_id}"
    local filename

    mkdir -p "$destination"
    for filename in \
        command.sh \
        runtime.txt \
        evaluation_metrics.csv \
        combined_config_summary.md; do
        cp "$output_dir/$filename" "$destination/$filename"
    done
}

run_evaluation() {
    local mode="$1"
    local seed="$2"
    local depth_config="${SOURCE_RUNS_ROOT}/${SOURCE_PREFIX}_depth_mistral_s0.25_g20_o16_seed${seed}/layer_drop_config.txt"
    local quant_config="${SOURCE_RUNS_ROOT}/${SOURCE_PREFIX}_quant_mistral_qproj3.0_g20_o16_seed${seed}/quant_configuration.txt"
    local base_run_id
    local method
    local config_path
    local sparsity_or_bits
    local run_id
    local output_dir
    local exit_code
    local -a args=()

    case "$mode" in
        independent)
            base_run_id="${RUN_PREFIX}_independent_depth_quant_mistral_s0.25_qproj3.0_seed${seed}"
            method="independent_depth_quant_eval"
            config_path="$quant_config"
            sparsity_or_bits="depth0.25+qproj3.0_independent"
            ;;
        uniform)
            base_run_id="${RUN_PREFIX}_depth_uniform_quant_mistral_s0.25_qproj3.0_seed${seed}"
            method="depth_uniform_quant_eval"
            config_path=""
            sparsity_or_bits="depth0.25+qproj3.0_uniform"
            ;;
        *)
            printf 'Unsupported composition mode: %s\n' "$mode" >&2
            return 2
            ;;
    esac

    run_id="$(resolve_run_id "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"
    if run_is_complete "$output_dir"; then
        printf 'Skipping completed composition: %s\n' "$run_id"
        sync_lightweight_artifacts "$run_id"
        return 0
    fi
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry-run)
    fi

    printf '\n=== Mistral composition: mode=%s seed=%s run_id=%s ===\n' \
        "$mode" "$seed" "$run_id"
    env \
        MODEL="$MODEL" \
        SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
        EVAL_TOKENS="$EVAL_TOKENS" \
        EVAL_DATASETS="wikitext2" \
        ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
        DTYPE="$DTYPE" \
        SEED="$seed" \
        DROP_LAYER_CONFIG="$depth_config" \
        QUANT_WEIGHTS_PATH="$QUANT_WEIGHTS_PATH" \
        QUANT_CONFIG_PATH="$config_path" \
        QUANT_DEFAULT_LEVEL=3 \
        SPARSITY_OR_BITS="$sparsity_or_bits" \
        METHOD="$method" \
        RUN_ID="$run_id" \
        OUTPUT_DIR="$output_dir" \
        OUTPUTS_ROOT="$OUTPUTS_ROOT" \
        EXPERIMENT_LOG="$EXPERIMENT_LOG" \
        PYTHON_BIN="$PYTHON_BIN" \
        scripts/run_combined_eval_tiny.sh "${args[@]}"
    exit_code="$?"

    if [[ "$DRY_RUN" != "1" && "$exit_code" == "0" ]]; then
        sync_lightweight_artifacts "$run_id"
    fi
    return "$exit_code"
}

validate_inputs || exit $?

FAILED_RUNS=()
for mode in $MODES; do
    for seed in $SEEDS; do
        if ! run_evaluation "$mode" "$seed"; then
            FAILED_RUNS+=("${mode}_seed${seed}")
            if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
                printf 'Composition grid stopped after failure: mode=%s seed=%s\n' \
                    "$mode" "$seed" >&2
                exit 1
            fi
        fi
    done
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Composition grid finished with failures: %s\n' "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nMistral composition dry run complete.\n'
else
    printf '\nMistral composition grid complete.\n'
    printf 'Lightweight artifacts are ready under %s.\n' "$RESULTS_RUNS_ROOT"
fi
