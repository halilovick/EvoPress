#!/usr/bin/env bash
set -uo pipefail
# Deliberately NOT 'set -e': one cell's failure must not abort the sweep.

# Full crossover hyperparameter sweep, cheap S-protocol (q_proj-only debug
# quant DB, WikiText-2 calibration, G20). Grid: 3 crossover types x
# {0.1, 0.25, 0.5, 0.75, 1.0} crossover_probability, plus the p_c=0.0
# (no-crossover) reference -- 16 distinct (type, probability) settings x 3
# seeds = 48 cells. 4 already exist (p_c=0.25 for all 3 types at seed 0,
# from joint_g20_{component,layerbundle,localexchange}_xover_pop4_seed0;
# p_c=0.0 at seed 0, from joint_g20_population_pop4_seed0) and are skipped.
# Every other search-defining flag is copied verbatim from those existing
# command.sh files (verified by diff before this script was written).
#
# Idempotent: re-running this script after a crash skips any cell whose
# output_dir already has a run_summary.json, and continues with whatever
# is left -- no manual bookkeeping needed.
#
# Execution order is breadth-first by seed (see the main loop at the
# bottom): every (type, probability) combination gets its seed-0 cell
# before anything gets a seed-1 or seed-2 cell. This way a complete,
# presentable picture of the whole hyperparameter space exists as early
# as possible, with extra seeds (full mean+-SD statistics) layered on
# top opportunistically rather than risking full depth on a few
# combinations and zero data on the rest if time runs out.
#
# Watchdog: a single cell has been observed to wedge for hours (genuine
# I/O stall: near-zero CPU-time accumulation AND 0% GPU util for 4+ hours
# straight, confirmed against a healthy, low-contention node). Each cell
# runs under a watchdog that polls every POLL_INTERVAL_SECONDS and treats
# a cell as "alive" if EITHER its cumulative CPU time (from
# /proc/<pid>/stat) grows by at least MIN_CPU_SECONDS_PER_POLL OR its log
# file grows at all. A cell is only killed once BOTH signals stay flat for
# STALL_LIMIT_SECONDS straight.
#
# An earlier version used CPU time alone with a 30-minute window and a
# false-positive rate of 100%: for this workload, long GPU-bound stretches
# (e.g. "Computing KL Divergence") legitimately leave the host process
# mostly idle while the GPU does real work, so CPU ticks alone cannot
# distinguish a genuinely stalled job from a healthy one. Log growth
# doesn't share that blind spot -- the search script prints fresh content
# (generation number, eval results) as real progress happens, and every
# manual check this session has shown that output landing in the log file
# without meaningful delay, so it is a reliable second signal, not a
# redundant one. The combination only kills a cell when NEITHER a CPU nor
# an I/O/GPU-side explanation can be given for its silence -- exactly the
# profile of the original multi-hour CephFS-side stall this was built for.
#
# A stalled/killed cell has no run_summary.json, so it is retried
# automatically on the NEXT full run of this script, same idempotent-skip
# logic as a crash.
#
# Run with nohup, not tmux:
#   nohup bash run_crossover_sweep_g20.sh > ~/crossover_sweep_driver.log 2>&1 &
#   disown

cd /home/jovyan/evopress-crossover-v3

BASE_OUT="/home/jovyan/evopress-crossover-v3/outputs/experiments/sweep_crossover_g20"
LOG_DIR="$BASE_OUT/sweep_logs"
mkdir -p "$BASE_OUT" "$LOG_DIR"

QUANT_DB="/home/jovyan/evopress/outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit"

POLL_INTERVAL_SECONDS=60
STALL_LIMIT_SECONDS=2700          # 45 min of BOTH signals flat => stalled
MIN_CPU_SECONDS_PER_POLL=5         # CPU gain per poll that counts as "alive" on its own
CLK_TCK=$(getconf CLK_TCK)
stall_count=0

TYPES=(component layer_bundle local_exchange)
PROBS=(0.0 0.1 0.25 0.5 0.75 1.0)
SEEDS=(0 1 2)

type_label() {
  case "$1" in
    component) echo "component" ;;
    layer_bundle) echo "layerbundle" ;;
    local_exchange) echo "localexchange" ;;
  esac
}

prob_label() {
  case "$1" in
    0.0) echo "p000" ;;
    0.1) echo "p010" ;;
    0.25) echo "p025" ;;
    0.5) echo "p050" ;;
    0.75) echo "p075" ;;
    1.0) echo "p100" ;;
  esac
}

run_count=0
skip_count=0
fail_count=0

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_cpu_ticks() {
  # Cumulative utime+stime (clock ticks) for a PID, from /proc. Empty
  # output means the process is already gone.
  local pid="$1"
  local fields
  read -r -a fields < "/proc/$pid/stat" 2>/dev/null || { echo ""; return; }
  echo $(( ${fields[13]} + ${fields[14]} ))
}

run_cell() {
  local xtype="$1" prob="$2" seed="$3"
  local tlabel=$(type_label "$xtype")
  local plabel=$(prob_label "$prob")
  local name="joint_g20_${tlabel}_${plabel}_pop4_seed${seed}"
  local out_dir="$BASE_OUT/$name"
  local log_file="$LOG_DIR/${name}.log"

  if [[ -f "$out_dir/run_summary.json" ]]; then
    log "[SKIP] $name already has run_summary.json"
    skip_count=$((skip_count + 1))
    return 0
  fi

  log "[RUN ] $name -> $log_file"
  mkdir -p "$out_dir"

  python evo_joint_search.py \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --quant_weights_path "$QUANT_DB" \
    --drop_sparsity 0.25 \
    --target_bitwidth 3.0 \
    --calibration_data wikitext2 \
    --calibration_tokens 8192 \
    --calibration_sequence_length 1024 \
    --eval_every 5 \
    --eval_datasets wikitext2 \
    --eval_tokens 524288 \
    --eval_sequence_length 1024 \
    --generations 20 \
    --offspring 16 \
    --initially_generated 32 \
    --initial_tokens 512 \
    --survivors_per_selection 8 2 1 \
    --tokens_per_selection 512 2048 8192 \
    --population_size 4 \
    --crossover_probability "$prob" \
    --crossover_type "$xtype" \
    --crossover_parent_selection uniform \
    --fitness_fn kl \
    --group_rule size \
    --joint_mutation_mode standard \
    --step_size 1 \
    --max_drop_mutations 3 \
    --dtype float16 \
    --attn_implementation sdpa \
    --seed "$seed" \
    --output_dir "$out_dir" \
    --use_fast_tokenizer \
    --active_quant_budget \
    > "$log_file" 2>&1 &
  local pid=$!

  # Watchdog: a cell counts as "alive" this poll if EITHER its CPU time
  # grew by at least MIN_CPU_SECONDS_PER_POLL OR its log file grew at all.
  # Only kill once NEITHER signal has moved for STALL_LIMIT_SECONDS straight.
  local last_ticks
  last_ticks=$(get_cpu_ticks "$pid")
  local last_log_size
  last_log_size=$(stat -c %s "$log_file" 2>/dev/null || echo 0)
  local silent_for=0
  local min_ticks_per_poll=$(( CLK_TCK * MIN_CPU_SECONDS_PER_POLL ))

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_INTERVAL_SECONDS"
    kill -0 "$pid" 2>/dev/null || break
    local cur_ticks
    cur_ticks=$(get_cpu_ticks "$pid")
    [[ -z "$cur_ticks" ]] && break
    local cur_log_size
    cur_log_size=$(stat -c %s "$log_file" 2>/dev/null || echo "$last_log_size")
    local cpu_delta=$(( cur_ticks - last_ticks ))
    local log_grew=0
    [[ "$cur_log_size" != "$last_log_size" ]] && log_grew=1
    last_ticks=$cur_ticks
    last_log_size=$cur_log_size
    if (( cpu_delta < min_ticks_per_poll )) && (( log_grew == 0 )); then
      silent_for=$(( silent_for + POLL_INTERVAL_SECONDS ))
    else
      silent_for=0
    fi
    if (( silent_for >= STALL_LIMIT_SECONDS )); then
      log "[STALL] $name: no CPU progress (<${MIN_CPU_SECONDS_PER_POLL}s/poll) AND no log growth for ${silent_for}s -- killing PID $pid, will retry on next full sweep run"
      stall_count=$((stall_count + 1))
      kill -9 "$pid" 2>/dev/null
      local death_wait=0
      while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        death_wait=$(( death_wait + 10 ))
        if (( death_wait % 300 == 0 )); then
          log "[WARN ] $name: PID $pid still alive ${death_wait}s after SIGKILL (likely wedged in D state) -- still waiting, may need manual check"
        fi
      done
      break
    fi
  done

  wait "$pid" 2>/dev/null
  local status=$?
  if [[ -f "$out_dir/run_summary.json" ]]; then
    log "[DONE] $name"
    run_count=$((run_count + 1))
  else
    log "[FAIL] $name (exit $status) -- see $log_file"
    fail_count=$((fail_count + 1))
  fi
}

# Breadth-first ordering: seed is the OUTER loop, so every (type,
# probability) combination gets its first seed before anything gets a
# second or third. This guarantees that if time runs out, what exists is
# a complete single-seed picture of the whole hyperparameter space
# (directly answering "explore every combination, don't just try one and
# give up") rather than full 3-seed depth on a few combinations and zero
# data on the rest. Extra seeds (for proper mean+-SD statistics) accumulate
# on top of that as time allows, each pass strictly improving on the last.
for seed in "${SEEDS[@]}"; do
  for prob in "${PROBS[@]}"; do
    if [[ "$prob" == "0.0" ]]; then
      if [[ "$seed" == "0" ]]; then
        log "[SKIP] p_c=0.0 seed0 (existing: joint_g20_population_pop4_seed0)"
        skip_count=$((skip_count + 1))
        continue
      fi
      run_cell "component" "$prob" "$seed"
      continue
    fi
    for xtype in "${TYPES[@]}"; do
      if [[ "$prob" == "0.25" ]] && [[ "$seed" == "0" ]]; then
        log "[SKIP] p_c=0.25 $xtype seed0 (existing run)"
        skip_count=$((skip_count + 1))
        continue
      fi
      run_cell "$xtype" "$prob" "$seed"
    done
  done
done

echo ""
log "=== Sweep finished: $run_count run, $skip_count skipped, $fail_count failed, $stall_count killed by watchdog ==="
