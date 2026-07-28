#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOINT_LAUNCHER="$REPO_ROOT/scripts/run_joint_search_tiny.sh"

MODE=""
STAGE1_RUN_DIR_ARG=""
STAGE1_CANDIDATE_ARG=""
OUTPUT_DIR_ARG=""
POLICY="${SEQUENTIAL_QUANT_INITIALIZATION_POLICY:-strict}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run_sequential_search.sh \
    --mode MODE \
    (--stage1-run-dir PATH | --stage1-candidate PATH) \
    --output-dir PATH \
    [--policy strict|repair] [--dry-run]

Modes:
  depth_to_quant_frozen
  depth_to_joint_warm
  quant_to_depth_frozen
  quant_to_joint_warm

All model, data, budget, mutation, selection, and seed settings use the same
environment variables as scripts/run_joint_search_tiny.sh. In particular,
set MODEL, QUANT_WEIGHTS_PATH, DROP_SPARSITY, TARGET_BITWIDTH, GROUP_RULE,
ACTIVE_QUANT_BUDGET, GENERATIONS, OFFSPRING, and the selection schedule as
needed. The delegated launcher saves command.sh and refuses a non-empty output
directory.
EOF
}

while (($#)); do
    case "$1" in
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --stage1-run-dir)
            STAGE1_RUN_DIR_ARG="${2:?--stage1-run-dir requires a path}"
            shift 2
            ;;
        --stage1-candidate)
            STAGE1_CANDIDATE_ARG="${2:?--stage1-candidate requires a path}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_ARG="${2:?--output-dir requires a path}"
            shift 2
            ;;
        --policy)
            POLICY="${2:?--policy requires strict or repair}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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
done

case "$MODE" in
    depth_to_quant_frozen|depth_to_joint_warm|quant_to_depth_frozen|quant_to_joint_warm)
        ;;
    *)
        printf 'A supported --mode is required.\n' >&2
        usage >&2
        exit 2
        ;;
esac

if [[ -z "$OUTPUT_DIR_ARG" ]]; then
    printf '%s\n' '--output-dir is required.' >&2
    exit 2
fi
if [[ -n "$STAGE1_RUN_DIR_ARG" && -n "$STAGE1_CANDIDATE_ARG" ]] ||
   [[ -z "$STAGE1_RUN_DIR_ARG" && -z "$STAGE1_CANDIDATE_ARG" ]]; then
    printf '%s\n' 'Specify exactly one of --stage1-run-dir or --stage1-candidate.' >&2
    exit 2
fi
if [[ -n "$STAGE1_RUN_DIR_ARG" && ! -d "$STAGE1_RUN_DIR_ARG" ]]; then
    printf 'Stage-one run directory does not exist: %s\n' "$STAGE1_RUN_DIR_ARG" >&2
    exit 2
fi
if [[ -n "$STAGE1_CANDIDATE_ARG" && ! -f "$STAGE1_CANDIDATE_ARG" ]]; then
    printf 'Stage-one candidate does not exist: %s\n' "$STAGE1_CANDIDATE_ARG" >&2
    exit 2
fi
if [[ "$POLICY" != "strict" && "$POLICY" != "repair" ]]; then
    printf 'Policy must be strict or repair: %s\n' "$POLICY" >&2
    exit 2
fi

LAUNCHER_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
    LAUNCHER_ARGS+=(--dry-run)
fi

SEQUENTIAL_MODE="$MODE" \
STAGE1_RUN_DIR="$STAGE1_RUN_DIR_ARG" \
STAGE1_CANDIDATE="$STAGE1_CANDIDATE_ARG" \
SEQUENTIAL_QUANT_INITIALIZATION_POLICY="$POLICY" \
OUTPUT_DIR="$OUTPUT_DIR_ARG" \
RUN_ID="${RUN_ID:-$(basename "$OUTPUT_DIR_ARG")}" \
"$JOINT_LAUNCHER" "${LAUNCHER_ARGS[@]}"
