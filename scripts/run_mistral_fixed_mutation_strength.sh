#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_PREFIX="${RUN_PREFIX:-thesis_fixedstrength}"
SEEDS="${SEEDS:-0 1 2}"
GENERATIONS="${GENERATIONS:-50}"
OFFSPRING="${OFFSPRING:-16}"
MAX_DROP_MUTATIONS="${MAX_DROP_MUTATIONS:-1}"
DRY_RUN=0
CONTINUE_ON_FAILURE=0

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_fixed_mutation_strength.sh [--dry-run] [--continue-on-failure]

Run the matched three-seed Mistral joint-search ablation with a fixed local
depth-mutation strength of one. The launcher reuses the thesis compute-matched
configuration: 50 generations, 16 offspring, 25% depth sparsity, and an active
3-bit q_proj quantization budget.
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

if [[ "$MAX_DROP_MUTATIONS" != "1" ]]; then
    printf 'This ablation requires MAX_DROP_MUTATIONS=1.\n' >&2
    exit 2
fi

args=()
if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
fi
if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
    args+=(--continue-on-failure)
fi

env \
    RUN_PREFIX="$RUN_PREFIX" \
    METHODS=joint \
    SEEDS="$SEEDS" \
    RUN_DENSE=0 \
    GENERATIONS="$GENERATIONS" \
    OFFSPRING="$OFFSPRING" \
    MAX_DROP_MUTATIONS=1 \
    JOINT_AWARE_MUTATION=0 \
    ADAPTIVE_MUTATION=0 \
    scripts/run_mistral_medium_grid.sh "${args[@]}"
