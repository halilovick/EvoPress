#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: scripts/run_mistral_attention_quant_db.sh [--dry-run]

Generate a logged Mistral-7B GPTQ/FastOBQ database for all attention
projection modules: q_proj, k_proj, v_proj, and o_proj.

This is intentionally conservative by default because Datalab exposes only a
16 GB container RAM limit even on larger GPUs. The database can be regenerated
with larger CALIB_TOKENS or SEQUENCE_LENGTH after the conservative run succeeds.

Recommended Datalab run:
  bash scripts/run_mistral_attention_quant_db.sh --dry-run

  nohup bash scripts/run_mistral_attention_quant_db.sh \
    > outputs/quant_db_mistral_attention_launcher.log 2>&1 &

  tail -f outputs/quant_db_mistral_attention_launcher.log

Important defaults:
  MODEL=mistralai/Mistral-7B-v0.3
  RUN_ID=quant_db_mistral_attention_bits234
  QUANTIZABLE_MODULES='.*layers.*self_attn.*((q|k|v|o)_proj)$'
  CALIB_TOKENS=512
  SEQUENCE_LENGTH=128
  EXPECTED_MODULE_DIRS=128
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run|-h|--help)
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

free_disk_mb() {
    df -Pm . | awk 'NR == 2 { print $4 }'
}

check_disk_headroom() {
    local estimated_mb="${MISTRAL_ATTENTION_DB_ESTIMATED_MB:-13000}"
    local required_mb=$((estimated_mb + estimated_mb / 2 + 4096))
    local available_mb
    available_mb="$(free_disk_mb)"

    printf 'Estimated Mistral attention quant database size: approximately %s MB\n' "$estimated_mb"
    printf 'Required free-disk safety threshold: %s MB\n' "$required_mb"
    printf 'Available free disk: %s MB\n' "$available_mb"

    if ((available_mb < required_mb)); then
        printf 'Insufficient disk headroom for the Mistral attention quantization database.\n' >&2
        exit 2
    fi
}

check_gpu_headroom() {
    local min_gpu_memory_mib="${MIN_GPU_MEMORY_MIB:-30000}"
    local gpu_memory_mib

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi

    gpu_memory_mib="$(
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
            head -n 1 |
            tr -d ' '
    )"
    if [[ "$gpu_memory_mib" =~ ^[0-9]+$ ]] && ((gpu_memory_mib < min_gpu_memory_mib)); then
        printf 'Mistral attention database generation needs at least %s MiB GPU memory; detected %s MiB.\n' \
            "$min_gpu_memory_mib" "$gpu_memory_mib" >&2
        printf 'Use the V100 32GB or A40 Datalab profile for this run.\n' >&2
        exit 2
    fi
}

check_disk_headroom
check_gpu_headroom

export MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
export RUN_ID="${RUN_ID:-quant_db_mistral_attention_bits234}"
export QUANTIZABLE_MODULES="${QUANTIZABLE_MODULES:-.*layers.*self_attn.*((q|k|v|o)_proj)$}"
export CALIB_DATA="${CALIB_DATA:-wikitext2}"
export CALIB_TOKENS="${CALIB_TOKENS:-512}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-128}"
export BITS_LIST="${BITS_LIST:-2 3 4}"
export BITS_TO_LOAD="${BITS_TO_LOAD:-3}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export DTYPE="${DTYPE:-float16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
export CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-1}"
export CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-1}"
export DROP_SAVED_FILE_CACHE="${DROP_SAVED_FILE_CACHE:-1}"
export EXPECTED_MODULE_DIRS="${EXPECTED_MODULE_DIRS:-128}"

exec bash scripts/run_gptq_tiny_debug.sh "$@"
