#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPARSITIES="${SPARSITIES:-0.125 0.25 0.375 0.50}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/run_drop_search_grid.sh [--dry-run] [--continue-on-failure]

Run the four-point Mistral-7B depth-pruning grid sequentially. Environment
variables accepted by scripts/run_drop_search.sh can be overridden here too.

Examples:
  scripts/run_drop_search_grid.sh
  scripts/run_drop_search_grid.sh --dry-run
  SPARSITIES="0.125 0.25" scripts/run_drop_search_grid.sh
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

FAILED_RUNS=()

for sparsity in $SPARSITIES; do
    printf '\n=== Preparing depth-pruning run: sparsity=%s ===\n' "$sparsity"
    run_id="depth_mistral7b_s${sparsity}_seed${SEED:-1}"
    output_dir="${OUTPUTS_ROOT:-outputs/experiments}/${run_id}"
    SINGLE_RUN_ARGS=()
    if [[ "$DRY_RUN" == "1" ]]; then
        SINGLE_RUN_ARGS+=(--dry-run)
    fi

    SPARSITY="$sparsity" RUN_ID="$run_id" OUTPUT_DIR="$output_dir" scripts/run_drop_search.sh "${SINGLE_RUN_ARGS[@]}"
    RUN_EXIT_CODE="$?"
    if [[ "$RUN_EXIT_CODE" != "0" ]]; then
        FAILED_RUNS+=("$sparsity")
        printf 'Depth-pruning run failed: sparsity=%s exit_code=%s\n' "$sparsity" "$RUN_EXIT_CODE" >&2
        if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
            printf 'Stopping grid. Pass --continue-on-failure to attempt remaining sparsities.\n' >&2
            exit "$RUN_EXIT_CODE"
        fi
    fi
done

if ((${#FAILED_RUNS[@]})); then
    printf 'Grid finished with failed sparsities: %s\n' "${FAILED_RUNS[*]}" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. No experiments were launched and no CSV rows were appended.\n'
else
    printf '\nDepth-pruning grid complete.\n'
fi
