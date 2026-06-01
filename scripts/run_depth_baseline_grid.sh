#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPARSITIES="${SPARSITIES:-0.125 0.25 0.375 0.50}"
RANDOM_SEEDS="${RANDOM_SEEDS:-1 2 3}"
RUN_EARLY_LAYER="${RUN_EARLY_LAYER:-0}"
PROTECT_LAYER_ZERO="${PROTECT_LAYER_ZERO:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/run_depth_baseline_grid.sh [--dry-run] [--continue-on-failure] [--skip-existing]

Run matched random and late-layer depth-pruning baselines sequentially.
Random baselines use seeds 1, 2, and 3 by default. Set RUN_EARLY_LAYER=1
to include the optional early-layer negative control.

Examples:
  scripts/run_depth_baseline_grid.sh --dry-run
  scripts/run_depth_baseline_grid.sh
  scripts/run_depth_baseline_grid.sh --continue-on-failure
  scripts/run_depth_baseline_grid.sh --skip-existing
  RUN_EARLY_LAYER=1 PROTECT_LAYER_ZERO=1 scripts/run_depth_baseline_grid.sh
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
        --skip-existing)
            SKIP_EXISTING=1
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

FAILED_RUNS=()

run_one() {
    local method="$1"
    local sparsity="$2"
    local seed="$3"
    local run_id="baseline_${method}_mistral7b_s${sparsity}_seed${seed}"
    local output_dir="${OUTPUTS_ROOT:-outputs/experiments}/${run_id}"
    local single_run_args=()
    if [[ "$DRY_RUN" == "1" ]]; then
        single_run_args+=(--dry-run)
    fi

    printf '\n=== Preparing depth baseline: method=%s sparsity=%s seed=%s ===\n' "$method" "$sparsity" "$seed"
    if [[ "$SKIP_EXISTING" == "1" && -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Skipping existing output directory: %s\n' "$output_dir"
        return
    fi
    METHOD="$method" \
        SPARSITY="$sparsity" \
        SEED="$seed" \
        PROTECT_LAYER_ZERO="$PROTECT_LAYER_ZERO" \
        RUN_ID="$run_id" \
        OUTPUT_DIR="$output_dir" \
        scripts/run_depth_baseline.sh "${single_run_args[@]}"
    local run_exit_code="$?"
    if [[ "$run_exit_code" != "0" ]]; then
        FAILED_RUNS+=("$run_id")
        printf 'Depth baseline failed: run_id=%s exit_code=%s\n' "$run_id" "$run_exit_code" >&2
        if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
            printf 'Stopping grid. Pass --continue-on-failure to attempt remaining baselines.\n' >&2
            exit "$run_exit_code"
        fi
    fi
}

for sparsity in $SPARSITIES; do
    for seed in $RANDOM_SEEDS; do
        run_one random "$sparsity" "$seed"
    done
done

for sparsity in $SPARSITIES; do
    run_one late_layer "$sparsity" 1
done

if [[ "$RUN_EARLY_LAYER" == "1" ]]; then
    for sparsity in $SPARSITIES; do
        run_one early_layer "$sparsity" 1
    done
fi

if ((${#FAILED_RUNS[@]})); then
    printf 'Baseline grid finished with failed runs: %s\n' "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. No experiments were launched and no CSV rows were appended.\n'
else
    printf '\nDepth-baseline grid complete.\n'
fi
