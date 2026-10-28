#!/usr/bin/env bash
set -euo pipefail

# Regenerates the full 224-module (q/k/v/o/gate/up/down_proj x 32 layers) GPTQ
# quant database on the new Blackwell (RTX PRO 6000, 96GB) SLURM node.
#
# This is byte-identical in every quantization-determining flag to the
# original stage-1 run that produced
# /home/jovyan/evopress_quant_stage1_1gpu_diskspill_attempt3_20260819 on the
# A40 DataLab workspace (see
# thesis_results/datalab_raw/paper_matched/prepare_db/mistral7b_full_gptq_db_1gpu_diskspill_attempt3_20260819/command.sh).
# Only --save_dir, --activation_cache_dir and --attempt_id differ (new
# node, new attempt tag). seed is unchanged (0) because that is the fixed
# seed used to build the shared quant DB for every search run — it is NOT
# the per-search seed.
#
# The original run took 25,928s (~7.2h) on an A40 and fits comfortably in
# one 12h SLURM allocation; it is NOT resumable mid-run, so run it to
# completion in a single session.

cd ~/src/evopress

micromamba run -n evopress \
  torchrun --nnodes=1 --nproc-per-node=1 quant.py \
  --configured_torchrun_processes 8 \
  --model_name_or_path mistralai/Mistral-7B-v0.3 \
  --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
  --expected_module_count 224 \
  --expected_reconstruction_dtype float16 \
  --pre_block_modules model.embed_tokens \
  --block_modules model.layers \
  --post_block_modules model.norm lm_head \
  --calibration_data fineweb_edu \
  --calibration_tokens 8388608 \
  --calibration_sequence_length 8192 \
  --bitwidth_options 2 3 4 5 6 \
  --calibration_bitwidth 3 \
  --group_size 128 \
  --rel_damp 0.01 \
  --block_size 128 \
  --seed 0 \
  --low_cpu_mem_usage \
  --cpu_offload_activations \
  --drop_saved_file_cache \
  --verbose \
  --dtype float16 \
  --attn_implementation flash_attention_2 \
  --save_dir "$HOME/evopress_quant_stage1_blackwell_fullspace_20261028" \
  --perchannel \
  --attempt_id 7e3583e36ebe45c0af1df5e17bb2cf66 \
  --load_model_to_gpu \
  --activation_cache_dir /tmp/evopress_activation_spill_blackwell_fullspace_20261028
