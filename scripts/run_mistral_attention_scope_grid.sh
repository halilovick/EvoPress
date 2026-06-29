#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_attention_scope_grid.sh [--dry-run] [--continue-on-failure]

Run a conservative Mistral-7B attention-quantization smoke grid using a
precomputed q/k/v/o projection database. This is the first broader-scope test
after the completed q_proj-only thesis grid.

Default grid:
  METHODS='quant joint'
  SEEDS='0'
  GENERATIONS=10
  OFFSPRING=8
  INITIALLY_GENERATED=16
  DEPTH_SPARSITY=0.25
  TARGET_BITWIDTH=3.0

Recommended Datalab run after database generation succeeds:
  bash scripts/run_mistral_attention_scope_grid.sh --dry-run

  nohup bash scripts/run_mistral_attention_scope_grid.sh --continue-on-failure \
    > outputs/mistral_attention_scope_grid.log 2>&1 &

  tail -f outputs/mistral_attention_scope_grid.log

If the smoke grid is stable, rerun this wrapper with SEEDS='0 1 2',
GENERATIONS=50, OFFSPRING=16, and INITIALLY_GENERATED=32 for thesis-scale
evidence.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run|--continue-on-failure|-h|--help)
            ;;
        *)
            printf 'Unknown argument: %s\n' "$arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

export MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
export QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit}"
export RUN_PREFIX="${RUN_PREFIX:-attention_scope_smoke}"
export QUANT_SCOPE_LABEL="${QUANT_SCOPE_LABEL:-attention}"
export METHODS="${METHODS:-quant joint}"
export SEEDS="${SEEDS:-0}"
export RUN_DENSE="${RUN_DENSE:-0}"
export DEPTH_SPARSITY="${DEPTH_SPARSITY:-0.25}"
export TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
export CALIB_TOKENS="${CALIB_TOKENS:-8192}"
export EVAL_TOKENS="${EVAL_TOKENS:-131072}"
export EVAL_EVERY="${EVAL_EVERY:-5}"
export GENERATIONS="${GENERATIONS:-10}"
export OFFSPRING="${OFFSPRING:-8}"
export INITIALLY_GENERATED="${INITIALLY_GENERATED:-16}"
export INITIAL_TOKENS="${INITIAL_TOKENS:-512}"
export TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-512 2048 8192}"
export SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-4 2 1}"
export FITNESS_FN="${FITNESS_FN:-kl}"
export GROUP_RULE="${GROUP_RULE:-size}"
export ACTIVE_QUANT_BUDGET="${ACTIVE_QUANT_BUDGET:-1}"
export MAX_DROP_MUTATIONS="${MAX_DROP_MUTATIONS:-3}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export DTYPE="${DTYPE:-float16}"
export MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-30000}"
export EXPECTED_QUANT_MODULES="${EXPECTED_QUANT_MODULES:-128}"
export EXPECTED_QUANT_WEIGHT_FILES="${EXPECTED_QUANT_WEIGHT_FILES:-384}"

exec bash scripts/run_mistral_medium_grid.sh "$@"
