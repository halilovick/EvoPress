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
RUN_PREFIX="${RUN_PREFIX:-generalization}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-dense depth independent joint_g50}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2 c4 fineweb_edu}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_RESULTS="${SYNC_RESULTS:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_generalization_eval.sh [--dry-run] [--continue-on-failure]

Replay tracked Mistral configurations on multiple evaluation datasets:
  dense       dense FP16 reference, seed 0 only
  depth       thesis_medium depth-only masks, seeds 0-2
  independent independently searched depth mask + searched q_proj profile
  joint_g50   compute-matched joint G50 depth + q_proj profiles

Default datasets: wikitext2 c4 fineweb_edu
Default EVAL_TOKENS: 131072. Set EVAL_TOKENS=524288 for a heavier full-scale
check after the screening run is validated.
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

runtime_succeeded() {
    local runtime_file="$1"
    [[ -f "$runtime_file" ]] && grep -qx 'exit_code=0' "$runtime_file"
}

metrics_have_datasets() {
    local metrics_file="$1"
    shift
    [[ -f "$metrics_file" ]] || return 1
    "$PYTHON_BIN" - "$metrics_file" "$@" <<'PY'
import csv
import sys

metrics_file = sys.argv[1]
expected = set(sys.argv[2:])
with open(metrics_file, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
observed = {row["dataset"] for row in rows}
missing = expected - observed
raise SystemExit(1 if missing else 0)
PY
}

run_is_complete() {
    local run_dir="$1"
    read -r -a datasets <<< "$EVAL_DATASETS"
    runtime_succeeded "$run_dir/runtime.txt" &&
        metrics_have_datasets "$run_dir/evaluation_metrics.csv" "${datasets[@]}"
}

sync_lightweight_artifacts() {
    local run_id="$1"
    local output_dir="${OUTPUTS_ROOT}/${run_id}"
    local destination="${RESULTS_RUNS_ROOT}/${run_id}"
    local filename
    [[ "$SYNC_RESULTS" == "1" && "$DRY_RUN" != "1" ]] || return
    [[ -d "$output_dir" ]] || return

    mkdir -p "$destination"
    for filename in \
        command.sh \
        runtime.txt \
        evaluation_metrics.csv \
        combined_config_summary.md; do
        if [[ -f "$output_dir/$filename" ]]; then
            cp "$output_dir/$filename" "$destination/$filename"
        fi
    done
    printf 'Lightweight artifacts synced to %s\n' "$destination"
}

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

resolve_run_id() {
    local base_run_id="$1"
    local candidate="$base_run_id"
    local retry=0
    while true; do
        if run_is_complete "${OUTPUTS_ROOT}/${candidate}" ||
            run_is_complete "${RESULTS_RUNS_ROOT}/${candidate}"; then
            printf '%s' "$candidate"
            return
        fi
        if ! directory_has_files "${OUTPUTS_ROOT}/${candidate}"; then
            printf '%s' "$candidate"
            return
        fi
        retry=$((retry + 1))
        candidate="${base_run_id}_retry${retry}"
    done
}

validate_inputs() {
    local seed
    local path
    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    if [[ "$METHODS" == *independent* || "$METHODS" == *joint_g50* ]]; then
        [[ -d "$QUANT_WEIGHTS_PATH" ]] || {
            printf 'Missing Mistral q_proj quant database: %s\n' "$QUANT_WEIGHTS_PATH" >&2
            return 2
        }
    fi
    for seed in $SEEDS; do
        for path in \
            "${SOURCE_RUNS_ROOT}/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}/layer_drop_config.txt" \
            "${SOURCE_RUNS_ROOT}/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed${seed}/quant_configuration.txt" \
            "${SOURCE_RUNS_ROOT}/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/joint_drop_config.txt" \
            "${SOURCE_RUNS_ROOT}/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/joint_quant_config.txt"; do
            [[ -f "$path" ]] || {
                printf 'Missing source config: %s\n' "$path" >&2
                return 2
            }
        done
    done
}

run_eval() {
    local method="$1"
    local seed="$2"
    local base_run_id
    local run_id
    local output_dir
    local eval_method
    local sparsity_or_bits
    local drop_config=""
    local quant_config=""
    local quant_weights=""
    local quant_default_level=0
    local -a args=()

    case "$method" in
        dense)
            base_run_id="${RUN_PREFIX}_dense_mistral_multidataset_seq${SEQUENCE_LENGTH}_seed0"
            eval_method="generalization_dense_eval"
            sparsity_or_bits="dense"
            ;;
        depth)
            base_run_id="${RUN_PREFIX}_depth_mistral_s0.25_multidataset_seed${seed}"
            eval_method="generalization_depth_eval"
            sparsity_or_bits="depth0.25"
            drop_config="${SOURCE_RUNS_ROOT}/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}/layer_drop_config.txt"
            ;;
        independent)
            base_run_id="${RUN_PREFIX}_independent_depth_quant_mistral_s0.25_qproj3.0_multidataset_seed${seed}"
            eval_method="generalization_independent_depth_quant_eval"
            sparsity_or_bits="depth0.25+qproj3.0_independent"
            drop_config="${SOURCE_RUNS_ROOT}/thesis_medium_depth_mistral_s0.25_g20_o16_seed${seed}/layer_drop_config.txt"
            quant_config="${SOURCE_RUNS_ROOT}/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed${seed}/quant_configuration.txt"
            quant_weights="$QUANT_WEIGHTS_PATH"
            quant_default_level=3
            ;;
        joint_g50)
            base_run_id="${RUN_PREFIX}_joint_g50_mistral_s0.25_qproj3.0_multidataset_seed${seed}"
            eval_method="generalization_joint_g50_eval"
            sparsity_or_bits="depth0.25+qproj3.0_joint_g50"
            drop_config="${SOURCE_RUNS_ROOT}/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/joint_drop_config.txt"
            quant_config="${SOURCE_RUNS_ROOT}/thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed${seed}/joint_quant_config.txt"
            quant_weights="$QUANT_WEIGHTS_PATH"
            quant_default_level=3
            ;;
        *)
            printf 'Unsupported method: %s\n' "$method" >&2
            return 2
            ;;
    esac

    run_id="$(resolve_run_id "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"
    if run_is_complete "$output_dir" || run_is_complete "${RESULTS_RUNS_ROOT}/${run_id}"; then
        printf 'Skipping completed generalization eval: %s\n' "$run_id"
        sync_lightweight_artifacts "$run_id"
        return 0
    fi
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry-run)
    fi

    printf '\n=== Mistral generalization eval: method=%s seed=%s run_id=%s ===\n' \
        "$method" "$seed" "$run_id"
    env \
        MODEL="$MODEL" \
        SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
        EVAL_TOKENS="$EVAL_TOKENS" \
        EVAL_DATASETS="$EVAL_DATASETS" \
        ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
        DTYPE="$DTYPE" \
        SEED="$seed" \
        METHOD="$eval_method" \
        SPARSITY_OR_BITS="$sparsity_or_bits" \
        DROP_LAYER_CONFIG="$drop_config" \
        QUANT_WEIGHTS_PATH="$quant_weights" \
        QUANT_CONFIG_PATH="$quant_config" \
        QUANT_DEFAULT_LEVEL="$quant_default_level" \
        RUN_ID="$run_id" \
        OUTPUT_DIR="$output_dir" \
        OUTPUTS_ROOT="$OUTPUTS_ROOT" \
        EXPERIMENT_LOG="$EXPERIMENT_LOG" \
        PYTHON_BIN="$PYTHON_BIN" \
        scripts/run_combined_eval_tiny.sh "${args[@]}"
    local exit_code="$?"
    if [[ "$exit_code" == "0" ]]; then
        sync_lightweight_artifacts "$run_id"
    fi
    return "$exit_code"
}

validate_inputs || exit $?

FAILED_RUNS=()
for method in $METHODS; do
    if [[ "$method" == "dense" ]]; then
        run_eval dense 0
        exit_code="$?"
        if [[ "$exit_code" != "0" ]]; then
            FAILED_RUNS+=("dense")
            [[ "$CONTINUE_ON_FAILURE" == "1" ]] || exit "$exit_code"
        fi
        continue
    fi
    for seed in $SEEDS; do
        run_eval "$method" "$seed"
        exit_code="$?"
        if [[ "$exit_code" != "0" ]]; then
            FAILED_RUNS+=("${method}_seed${seed}")
            [[ "$CONTINUE_ON_FAILURE" == "1" ]] || exit "$exit_code"
        fi
    done
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Generalization eval finished with failures: %s\n' \
        "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nMistral generalization dry run complete.\n'
else
    printf '\nMistral generalization evaluation complete.\n'
    printf 'Lightweight artifacts are ready under %s.\n' "$RESULTS_RUNS_ROOT"
fi
