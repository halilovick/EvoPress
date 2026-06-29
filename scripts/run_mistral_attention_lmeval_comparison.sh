#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_attention_lmeval_comparison.sh [--dry-run] [--continue-on-failure]

Evaluate Mistral attention-scope compression configurations with LM Evaluation
Harness tasks. Defaults match the q_proj downstream comparison:
  arc_easy,piqa,winogrande

This wrapper uses:
  - depth masks from thesis_medium depth-only runs
  - attention quant configs from thesis_attention quant-only runs
  - attention joint configs from thesis_attention_g50 joint runs
  - q/k/v/o attention quant database

Recommended Datalab run:
  bash scripts/run_mistral_attention_lmeval_comparison.sh --dry-run

  nohup bash scripts/run_mistral_attention_lmeval_comparison.sh --continue-on-failure \
    > outputs/mistral_attention_lmeval.log 2>&1 &

  tail -f outputs/mistral_attention_lmeval.log
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

export QUANT_WEIGHTS_PATH="${QUANT_WEIGHTS_PATH:-outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit}"
export RUN_PREFIX="${RUN_PREFIX:-lmeval_attention}"
export METHODS="${METHODS:-dense depth independent joint_g50}"
export DEPTH_SOURCE_PREFIX="${DEPTH_SOURCE_PREFIX:-thesis_medium}"
export QUANT_SOURCE_PREFIX="${QUANT_SOURCE_PREFIX:-thesis_attention}"
export JOINT_SOURCE_PREFIX="${JOINT_SOURCE_PREFIX:-thesis_attention_g50}"
export QUANT_SCOPE_LABEL="${QUANT_SCOPE_LABEL:-attention}"
export DEPTH_SPARSITY="${DEPTH_SPARSITY:-0.25}"
export TARGET_BITWIDTH="${TARGET_BITWIDTH:-3.0}"
export SOURCE_GENERATIONS="${SOURCE_GENERATIONS:-20}"
export SOURCE_OFFSPRING="${SOURCE_OFFSPRING:-16}"
export JOINT_GENERATIONS="${JOINT_GENERATIONS:-50}"
export JOINT_OFFSPRING="${JOINT_OFFSPRING:-16}"
export TASKS="${TASKS:-arc_easy,piqa,winogrande}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

exec bash scripts/run_mistral_lmeval_comparison.sh "$@"
