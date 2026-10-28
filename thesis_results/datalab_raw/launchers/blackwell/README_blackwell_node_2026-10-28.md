# Blackwell SLURM node — setup, quant-DB finding, and next-experiment plan (2026-10-28)

## New environment

- TU Wien GPU Cluster Hub, SLURM-managed, 12h allocation limit.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB VRAM. 64GB CPU RAM, 2 CPU cores.
- Repo: `~/src/evopress`, branch `agent/evopress-crossover-v3` (matches the
  DataLab/A40 checkout; confirmed via `git log` against
  `origin/agent/evopress-crossover-v3` — HEAD `028cb82` as of this writing).
- Python env: `~/micromamba/envs/evopress` (PyTorch 2.8.0+cu128). All new
  launchers in this directory use `micromamba run -n evopress <cmd>` instead
  of a venv `source activate` step.

## Finding: the user's own test quant DB is not usable for the established protocol

The node already had a self-generated validation artifact:
`outputs/experiments/quant_db_mistral_attention_calib8k_blackwell/quant_db/Mistral-7B-v0.3/3bit`,
produced from a GPTQ pass over only the four **attention** projections
(`q_proj`,`k_proj`,`v_proj`,`o_proj`), 2/3/4-bit, 8,192 calibration tokens,
sequence length 1024. A 20-generation standard joint search against it
(`thesis_blackwell_attention_calib8k`, seed 0, 16 offspring) completed in
4,247s and reported 14.10 WikiText-2 PPL.

This is **not comparable** to any existing fullspace-extension or
paper-matched result and should not be folded into `fullspace_ledger.csv`:

- Protocol F quantizes all **224** modules (`q/k/v/o/gate/up/down_proj` x 32
  layers) — `evo_joint_search.py` hard-asserts this via
  `--expected_quantized_modules` against the module-directory count found at
  `--quant_weights_path` (see `evo_joint_search.py:2030-2040`); it would
  reject the attention-only DB outright if pointed at it with
  `--expected_quantized_modules 224`. The attention-only DB covers at most
  4x32=128 modules — the MLP projections (the majority of the model's
  parameters) are untouched, so the search has far less room to place bits.
- The original stage-1 DB (`.../evopress_quant_stage1_1gpu_diskspill_attempt3_20260819`)
  was built from **8,388,608** calibration tokens at sequence length 8192
  (1,024 sequences); the new DB used roughly 1/1024th of that calibration
  budget.
- Net effect: 14.10 PPL vs. ~9.0-9.4 PPL for every existing s=0.25 joint
  condition at G150 is consistent with a much weaker, non-equivalent setup,
  not a real data point for the thesis. Useful as an end-to-end pipeline
  smoke test on the new hardware; not useful as a result.

## Plan: regenerate the real quant DB, then run the highest-value missing experiment

1. **Regenerate the full 224-module quant DB** with
   `run_prepare_quant_db_fullspace_blackwell_attempt1.sh` — every
   quantization-determining flag copied verbatim from the original
   `command.sh` used on DataLab (confirmed via `diff`; only `--save_dir`,
   `--activation_cache_dir`, and `--attempt_id` changed, plus `--seed 0`
   which is unchanged — it's the fixed DB-construction seed, not a search
   seed). Original run: 25,928s (~7.2h) on an A40 — fits in one 12h SLURM
   session with margin; not resumable mid-run, so let it finish in one go.

2. **Next experiment: LX (local-exchange crossover) seed 2.** Chosen over
   DW seed 2 because:
   - LX is the actual crossover contribution (the thesis's core novelty);
     DW is a sequential-mode warm-start ablation, secondary to that.
   - LX currently has only seed 0 folded into the ledger (seed 1 was
     launched on the A40 on 2026-10-27; status as of this writing unconfirmed
     — ask before assuming it's done). Running seed 2 on the new GPU now,
     independent of the A40, is the fastest way to get LX to a complete
     3-seed set, matching P4's already-complete 3-seed status.
   - LX has no `--stage1_run_dir` dependency (`--sequential_mode none`), so
     once the quant DB is ready, nothing else is needed.
   - Launcher: `run_fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1_blackwell.sh`.
     Diffed against the DataLab original — only repo path, env activation,
     `--quant_weights_path`, and `--output_dir` differ; every search-defining
     flag is identical.
   - This run will not finish inside 12h (matching A40 seeds took 46-52h).
     Resubmit `run_fullspace_joint_s025_localxover_pop4_g150_seed2_resume_blackwell.sh`
     unchanged after each SLURM timeout (same `--output_dir`, adds
     `--resume_checkpoint`) until it reaches generation 150/150. The
     checkpoint is written atomically at the end of every generation, so at
     most one generation of work is ever lost to a timeout.

3. **Queued after that: DW seed 2**, same quant DB, no extra transfer needed
   (`--stage1_run_dir` points at `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2`,
   already committed to git). Launcher ready:
   `run_fullspace_depthwarm_g150_seed2_attempt1_blackwell.sh`. Run this on
   whichever GPU frees up first (A40 or Blackwell) — if a second concurrent
   allocation is available, it can run in parallel with LX seed 2 instead of
   queued after it.

## Not done here

No GPU command was executed — these are inspection, script preparation, and
a local syntax check (`bash -n` on every script) only. The user runs every
command on the cluster themselves, one experiment at a time.
