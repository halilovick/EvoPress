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
DEPTH_SOURCE_PREFIX="${DEPTH_SOURCE_PREFIX:-$SOURCE_PREFIX}"
QUANT_SOURCE_PREFIX="${QUANT_SOURCE_PREFIX:-$SOURCE_PREFIX}"
JOINT_SOURCE_PREFIX="${JOINT_SOURCE_PREFIX:-thesis_compute_matched}"
RUN_PREFIX="${RUN_PREFIX:-lmeval}"
QUANT_SCOPE_LABEL="${QUANT_SCOPE_LABEL:-qproj}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-dense depth independent joint_g50}"
DEPTH_SPARSITY="${DEPTH_SPARSITY:-0.25}"
TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
SOURCE_GENERATIONS="${SOURCE_GENERATIONS:-20}"
SOURCE_OFFSPRING="${SOURCE_OFFSPRING:-16}"
JOINT_GENERATIONS="${JOINT_GENERATIONS:-50}"
JOINT_OFFSPRING="${JOINT_OFFSPRING:-16}"
TASKS="${TASKS:-arc_easy,piqa,winogrande}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
LIMIT="${LIMIT:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=${MODEL},low_cpu_mem_usage=True,dtype=float16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LMEVAL_SCRIPT="${LMEVAL_SCRIPT:-lmeval.py}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_RESULTS="${SYNC_RESULTS:-1}"
CHECK_RUNTIME_DEPENDENCIES="${CHECK_RUNTIME_DEPENDENCIES:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_lmeval_comparison.sh [--dry-run] [--continue-on-failure]

Evaluate the tracked Mistral compression configurations with LM Evaluation
Harness tasks. The default task set is intentionally modest:
  arc_easy,piqa,winogrande

Set TASKS="arc_easy,arc_challenge,piqa,winogrande,hellaswag" for a broader run
after the first pass succeeds. Set LIMIT=0.05 only for smoke testing; limited
LM-eval results should not be reported as final metrics.

The default quantization scope is q_proj for backwards compatibility. Use
QUANT_SCOPE_LABEL=attention with the attention q/k/v/o database and source
prefixes to evaluate broader-scope runs.
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

summary_value() {
    local summary_file="$1"
    local key="$2"
    [[ -f "$summary_file" ]] || return 1
    awk -F'`' -v key="$key" '$0 ~ "^- " key ": `" { print $2; exit }' "$summary_file"
}

run_matches_current_config() {
    local run_dir="$1"
    local summary_file="${run_dir}/lmeval_config_summary.md"
    local expected_limit="${LIMIT:-none}"
    [[ -f "$summary_file" ]] || return 1
    [[ "$(summary_value "$summary_file" "tasks")" == "$TASKS" ]] || return 1
    [[ "$(summary_value "$summary_file" "limit")" == "$expected_limit" ]] || return 1
    [[ "$(summary_value "$summary_file" "num_fewshot")" == "$NUM_FEWSHOT" ]] || return 1
    return 0
}

run_is_complete() {
    local run_dir="$1"
    runtime_succeeded "$run_dir/runtime.txt" &&
        [[ -f "$run_dir/lmeval_results.json" ]] &&
        run_matches_current_config "$run_dir"
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

methods_include() {
    local wanted="$1"
    local method
    for method in $METHODS; do
        [[ "$method" == "$wanted" ]] && return 0
    done
    return 1
}

depth_config_path() {
    local seed="$1"
    printf '%s/%s_depth_mistral_s%s_g%s_o%s_seed%s/layer_drop_config.txt' \
        "$SOURCE_RUNS_ROOT" \
        "$DEPTH_SOURCE_PREFIX" \
        "$DEPTH_SPARSITY" \
        "$SOURCE_GENERATIONS" \
        "$SOURCE_OFFSPRING" \
        "$seed"
}

quant_config_path() {
    local seed="$1"
    printf '%s/%s_quant_mistral_%s%s_g%s_o%s_seed%s/quant_configuration.txt' \
        "$SOURCE_RUNS_ROOT" \
        "$QUANT_SOURCE_PREFIX" \
        "$QUANT_SCOPE_LABEL" \
        "$TARGET_BITWIDTH" \
        "$SOURCE_GENERATIONS" \
        "$SOURCE_OFFSPRING" \
        "$seed"
}

joint_drop_config_path() {
    local seed="$1"
    printf '%s/%s_joint_mistral_s%s_%s%s_g%s_o%s_seed%s/joint_drop_config.txt' \
        "$SOURCE_RUNS_ROOT" \
        "$JOINT_SOURCE_PREFIX" \
        "$DEPTH_SPARSITY" \
        "$QUANT_SCOPE_LABEL" \
        "$TARGET_BITWIDTH" \
        "$JOINT_GENERATIONS" \
        "$JOINT_OFFSPRING" \
        "$seed"
}

joint_quant_config_path() {
    local seed="$1"
    printf '%s/%s_joint_mistral_s%s_%s%s_g%s_o%s_seed%s/joint_quant_config.txt' \
        "$SOURCE_RUNS_ROOT" \
        "$JOINT_SOURCE_PREFIX" \
        "$DEPTH_SPARSITY" \
        "$QUANT_SCOPE_LABEL" \
        "$TARGET_BITWIDTH" \
        "$JOINT_GENERATIONS" \
        "$JOINT_OFFSPRING" \
        "$seed"
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
        lmeval_results.json \
        lmeval_config_summary.md; do
        if [[ -f "$output_dir/$filename" ]]; then
            cp "$output_dir/$filename" "$destination/$filename"
        fi
    done
    printf 'Lightweight artifacts synced to %s\n' "$destination"
}

validate_inputs() {
    local seed
    local path
    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    if methods_include independent || methods_include joint_g50; then
        [[ -d "$QUANT_WEIGHTS_PATH" ]] || {
            printf 'Missing Mistral %s quant database: %s\n' \
                "$QUANT_SCOPE_LABEL" "$QUANT_WEIGHTS_PATH" >&2
            return 2
        }
    fi
    for seed in $SEEDS; do
        if methods_include depth || methods_include independent; then
            path="$(depth_config_path "$seed")"
            [[ -f "$path" ]] || {
                printf 'Missing source config: %s\n' "$path" >&2
                return 2
            }
        fi
        if methods_include independent; then
            path="$(quant_config_path "$seed")"
            [[ -f "$path" ]] || {
                printf 'Missing source config: %s\n' "$path" >&2
                return 2
            }
        fi
        if methods_include joint_g50; then
            for path in \
                "$(joint_drop_config_path "$seed")" \
                "$(joint_quant_config_path "$seed")"; do
                [[ -f "$path" ]] || {
                    printf 'Missing source config: %s\n' "$path" >&2
                    return 2
                }
            done
        fi
    done
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

write_command_file() {
    local command_file="$1"
    shift
    {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n'
        printf 'cd %q\n' "$REPO_ROOT"
        printf 'exec '
        printf '%q ' "$@"
        printf '\n'
    } > "$command_file"
    chmod +x "$command_file"
}

write_config_summary() {
    local output_file="$1"
    local run_id="$2"
    local method="$3"
    local seed="$4"
    local drop_config="$5"
    local quant_config="$6"
    {
        printf '# Mistral LM-Eval Config\n\n'
        printf -- '- run_id: `%s`\n' "$run_id"
        printf -- '- method: `%s`\n' "$method"
        printf -- '- model: `%s`\n' "$MODEL"
        printf -- '- model_args: `%s`\n' "$MODEL_ARGS"
        printf -- '- tasks: `%s`\n' "$TASKS"
        printf -- '- batch_size: `%s`\n' "$BATCH_SIZE"
        printf -- '- max_batch_size: `%s`\n' "${MAX_BATCH_SIZE:-none}"
        printf -- '- limit: `%s`\n' "${LIMIT:-none}"
        printf -- '- num_fewshot: `%s`\n' "$NUM_FEWSHOT"
        printf -- '- device: `%s`\n' "$DEVICE"
        printf -- '- seed: `%s`\n' "$seed"
        printf -- '- drop_layer_config: `%s`\n' "${drop_config:-none}"
        printf -- '- quant_weights_path: `%s`\n' "${QUANT_WEIGHTS_PATH:-none}"
        printf -- '- quant_config_path: `%s`\n' "${quant_config:-none}"
    } > "$output_file"
}

config_notes() {
    local drop_config="$1"
    local quant_config="$2"
    printf 'tasks=%s; limit=%s; num_fewshot=%s; drop_layer_config=%s; quant_weights_path=%s; quant_config_path=%s' \
        "$TASKS" \
        "${LIMIT:-none}" \
        "$NUM_FEWSHOT" \
        "${drop_config:-none}" \
        "${QUANT_WEIGHTS_PATH:-none}" \
        "${quant_config:-none}"
}

append_experiment_row() {
    local run_id="$1"
    local method="$2"
    local seed="$3"
    local sparsity_or_bits="$4"
    local output_dir="$5"
    local status="$6"
    local notes="$7"
    local runtime_minutes="$8"
    local gpu_name="$9"
    local gpu_vram_gb="${10}"
    local cpu_ram_limit_gb="${11}"

    "$PYTHON_BIN" scripts/append_experiment_log.py \
        --log-file "$EXPERIMENT_LOG" \
        --run-id "$run_id" \
        --method "$method" \
        --model "$MODEL" \
        --sparsity-or-bits "$sparsity_or_bits" \
        --attention-impl "lm_eval_harness" \
        --dtype "float16" \
        --seed "$seed" \
        --runtime-minutes "$runtime_minutes" \
        --gpu-name "$gpu_name" \
        --gpu-vram-gb "$gpu_vram_gb" \
        --cpu-ram-limit-gb "$cpu_ram_limit_gb" \
        --status "$status" \
        --notes "$notes" \
        --output-dir "$output_dir"
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
    local command_file
    local run_log
    local runtime_file
    local results_file
    local summary_file
    local -a command=()

    case "$method" in
        dense)
            base_run_id="${RUN_PREFIX}_dense_mistral_tasks_seed0"
            eval_method="lmeval_dense"
            sparsity_or_bits="dense"
            ;;
        depth)
            base_run_id="${RUN_PREFIX}_depth_mistral_s${DEPTH_SPARSITY}_tasks_seed${seed}"
            eval_method="lmeval_depth"
            sparsity_or_bits="depth${DEPTH_SPARSITY}"
            drop_config="$(depth_config_path "$seed")"
            ;;
        independent)
            base_run_id="${RUN_PREFIX}_independent_depth_quant_mistral_s${DEPTH_SPARSITY}_${QUANT_SCOPE_LABEL}${TARGET_BITWIDTH}_tasks_seed${seed}"
            eval_method="lmeval_independent_depth_quant"
            sparsity_or_bits="depth${DEPTH_SPARSITY}+${QUANT_SCOPE_LABEL}${TARGET_BITWIDTH}_independent"
            drop_config="$(depth_config_path "$seed")"
            quant_config="$(quant_config_path "$seed")"
            ;;
        joint_g50)
            base_run_id="${RUN_PREFIX}_joint_g50_mistral_s${DEPTH_SPARSITY}_${QUANT_SCOPE_LABEL}${TARGET_BITWIDTH}_tasks_seed${seed}"
            eval_method="lmeval_joint_g50"
            sparsity_or_bits="depth${DEPTH_SPARSITY}+${QUANT_SCOPE_LABEL}${TARGET_BITWIDTH}_joint_g50"
            drop_config="$(joint_drop_config_path "$seed")"
            quant_config="$(joint_quant_config_path "$seed")"
            ;;
        *)
            printf 'Unsupported method: %s\n' "$method" >&2
            return 2
            ;;
    esac

    run_id="$(resolve_run_id "$base_run_id")"
    output_dir="${OUTPUTS_ROOT}/${run_id}"
    command_file="${output_dir}/command.sh"
    run_log="${output_dir}/run.log"
    runtime_file="${output_dir}/runtime.txt"
    results_file="${output_dir}/lmeval_results.json"
    summary_file="${output_dir}/lmeval_config_summary.md"

    if run_is_complete "$output_dir" || run_is_complete "${RESULTS_RUNS_ROOT}/${run_id}"; then
        printf 'Skipping completed LM-eval run: %s\n' "$run_id"
        sync_lightweight_artifacts "$run_id"
        return 0
    fi
    if [[ "$run_id" != "$base_run_id" ]]; then
        printf 'Preserving incomplete run; selected retry id: %s\n' "$run_id"
    fi

    command=(
        "$PYTHON_BIN" "$LMEVAL_SCRIPT"
        --model hf
        --model_args "$MODEL_ARGS"
        --tasks "$TASKS"
        --batch_size "$BATCH_SIZE"
        --device "$DEVICE"
        --num_fewshot "$NUM_FEWSHOT"
        --output_path "$results_file"
    )
    if [[ -n "$MAX_BATCH_SIZE" ]]; then
        command+=(--max_batch_size "$MAX_BATCH_SIZE")
    fi
    if [[ -n "$LIMIT" ]]; then
        command+=(--limit "$LIMIT")
    fi
    if [[ -n "$quant_config" ]]; then
        command+=(
            --quant_weights_path "$QUANT_WEIGHTS_PATH"
            --quant_config_path "$quant_config"
            --quant_default_level 3
        )
    fi
    if [[ -n "$drop_config" ]]; then
        command+=(--drop_layer_config "$drop_config")
    fi

    printf '\n=== Mistral LM-eval: method=%s seed=%s run_id=%s ===\n' \
        "$method" "$seed" "$run_id"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'Dry run only. Would prepare command for %s:\n' "$run_id"
        printf '  method=%s\n' "$eval_method"
        printf '  output_dir=%s\n' "$output_dir"
        printf '  command_file=%s\n' "$command_file"
        printf '  command='
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    if directory_has_files "$output_dir"; then
        printf 'Refusing to overwrite non-empty output directory: %s\n' "$output_dir" >&2
        return 2
    fi
    mkdir -p "$output_dir"
    write_command_file "$command_file" "${command[@]}"
    write_config_summary "$summary_file" "$run_id" "$eval_method" "$seed" "$drop_config" "$quant_config"

    local gpu_name
    local gpu_vram_gb
    local cpu_ram_limit_gb
    local start_time
    local end_time
    local runtime_seconds
    local runtime_minutes
    local run_exit_code
    local status
    local notes
    local final_exit_code
    gpu_name="$(get_gpu_name)"
    gpu_vram_gb="$(get_gpu_vram_gb)"
    cpu_ram_limit_gb="$(get_cpu_ram_limit_gb)"
    start_time="$(date +%s)"

    {
        printf 'run_id=%s\n' "$run_id"
        printf 'method=%s\n' "$eval_method"
        printf 'output_dir=%s\n' "$output_dir"
        printf 'started_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'command_file=%s\n' "$command_file"
        printf 'results_file=%s\n' "$results_file"
        printf '\n'
    } | tee "$run_log"

    "${command[@]}" 2>&1 | tee -a "$run_log"
    run_exit_code="${PIPESTATUS[0]}"

    end_time="$(date +%s)"
    runtime_seconds="$((end_time - start_time))"
    runtime_minutes="$(awk -v runtime_seconds="$runtime_seconds" 'BEGIN { printf "%.2f", runtime_seconds / 60 }')"
    {
        printf 'runtime_seconds=%s\n' "$runtime_seconds"
        printf 'runtime_minutes=%s\n' "$runtime_minutes"
        printf 'exit_code=%s\n' "$run_exit_code"
    } > "$runtime_file"

    status=completed
    notes="last_successful_step=lmeval_completed; $(config_notes "$drop_config" "$quant_config")"
    final_exit_code=0
    if [[ "$run_exit_code" != "0" ]]; then
        status=failed
        notes="last_successful_step=lmeval_process_started; command_exit_code=${run_exit_code}; $(config_notes "$drop_config" "$quant_config")"
        final_exit_code="$run_exit_code"
    elif [[ ! -f "$results_file" ]]; then
        status=failed
        notes="last_successful_step=lmeval_process_completed; missing_results_json=1; $(config_notes "$drop_config" "$quant_config")"
        final_exit_code=1
    fi

    append_experiment_row \
        "$run_id" \
        "$eval_method" \
        "$seed" \
        "$sparsity_or_bits" \
        "$output_dir" \
        "$status" \
        "$notes" \
        "$runtime_minutes" \
        "$gpu_name" \
        "$gpu_vram_gb" \
        "$cpu_ram_limit_gb"

    printf 'Experiment %s finished with status=%s.\n' "$run_id" "$status"
    printf 'Artifacts: %s\n' "$output_dir"
    if [[ "$final_exit_code" == "0" ]]; then
        sync_lightweight_artifacts "$run_id"
    fi
    return "$final_exit_code"
}

validate_inputs || exit $?

if [[ "$CHECK_RUNTIME_DEPENDENCIES" == "1" && "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/check_runtime_dependencies.py \
        --require-cuda \
        --packages datasets numpy torch transformers tqdm accelerate sentencepiece lm_eval || exit 2
fi

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
    printf 'LM-eval comparison finished with failures: %s\n' \
        "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nMistral LM-eval dry run complete.\n'
else
    printf '\nMistral LM-eval comparison complete.\n'
    printf 'Lightweight artifacts are ready under %s.\n' "$RESULTS_RUNS_ROOT"
fi
