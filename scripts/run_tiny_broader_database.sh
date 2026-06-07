#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHOD="${1:-}"
SCOPE="${2:-}"
shift "$(( $# >= 2 ? 2 : $# ))"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_tiny_broader_database.sh <quant|sparse> <qproj|attention|all-linear> [--dry-run]

This wrapper selects a projection scope and delegates to the existing logged
TinyLlama database launcher. Environment variables accepted by the delegated
launcher can still be used to override defaults.

Recommended first run:
  bash scripts/run_tiny_broader_database.sh quant attention --dry-run

  nohup bash scripts/run_tiny_broader_database.sh quant attention \
    > outputs/quant_db_tiny_attention_launcher.log 2>&1 &

Scopes:
  qproj       q_proj only, matching the completed feasibility experiments
  attention   q_proj, k_proj, v_proj, and o_proj
  all-linear  attention projections plus gate_proj, up_proj, and down_proj
EOF
}

if [[ -z "$METHOD" || -z "$SCOPE" ]]; then
    usage >&2
    exit 2
fi

case "$SCOPE" in
    qproj)
        MODULE_REGEX='.*layers.*q_proj$'
        SCOPE_LABEL="qproj"
        QUANT_ESTIMATED_MB=529
        SPARSE_ESTIMATED_MB=1233
        ;;
    attention)
        MODULE_REGEX='.*layers.*self_attn.*((q|k|v|o)_proj)$'
        SCOPE_LABEL="attention"
        QUANT_ESTIMATED_MB=1191
        SPARSE_ESTIMATED_MB=2775
        ;;
    all-linear)
        MODULE_REGEX='.*layers.*((q|k|v|o|gate|up|down)_proj)$'
        SCOPE_LABEL="alllinear"
        QUANT_ESTIMATED_MB=5555
        SPARSE_ESTIMATED_MB=12947
        ;;
    *)
        printf 'Unknown scope: %s\n' "$SCOPE" >&2
        usage >&2
        exit 2
        ;;
esac

free_disk_mb() {
    df -Pm . | awk 'NR == 2 { print $4 }'
}

check_disk_headroom() {
    local estimated_mb="$1"
    local required_mb=$((estimated_mb + estimated_mb / 2 + 2048))
    local available_mb
    available_mb="$(free_disk_mb)"

    printf 'Estimated database size: approximately %s MB\n' "$estimated_mb"
    printf 'Required free-disk safety threshold: %s MB\n' "$required_mb"
    printf 'Available free disk: %s MB\n' "$available_mb"

    if ((available_mb < required_mb)); then
        printf 'Insufficient disk headroom for this scope.\n' >&2
        exit 2
    fi
}

case "$METHOD" in
    quant)
        check_disk_headroom "$QUANT_ESTIMATED_MB"
        export QUANTIZABLE_MODULES="${QUANTIZABLE_MODULES:-$MODULE_REGEX}"
        export RUN_ID="${RUN_ID:-quant_db_tinyllama_${SCOPE_LABEL}_bits234}"
        export DROP_SAVED_FILE_CACHE="${DROP_SAVED_FILE_CACHE:-1}"
        exec bash scripts/run_gptq_tiny_debug.sh "$@"
        ;;
    sparse)
        check_disk_headroom "$SPARSE_ESTIMATED_MB"
        export PRUNABLE_MODULES="${PRUNABLE_MODULES:-$MODULE_REGEX}"
        export RUN_ID="${RUN_ID:-sparse_db_tinyllama_${SCOPE_LABEL}_s0.50}"
        export DROP_SAVED_FILE_CACHE="${DROP_SAVED_FILE_CACHE:-1}"
        exec bash scripts/run_sparse_gpt_tiny_debug.sh "$@"
        ;;
    *)
        printf 'Unknown method: %s\n' "$METHOD" >&2
        usage >&2
        exit 2
        ;;
esac
