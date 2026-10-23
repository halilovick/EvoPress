# Protocol-F seed completion (2026-10-23)

Five new launcher scripts complete the three remaining protocol-F
(full-space, paper-matched) seed gaps identified in
`thesis_results/EXPERIMENT_MATRIX.md` §2 and §5 ("+seeds"):

| Script | Condition | Seed | Depends on |
| --- | --- | --- | --- |
| `run_fullspace_depthwarm_g150_seed1_attempt1.sh` | DW (depth→joint warm start) | 1 | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed1` (already present) |
| `run_fullspace_depthwarm_g150_seed2_attempt1.sh` | DW | 2 | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed2` (already present) |
| `run_fullspace_joint_s025_pop4_g150_seed2_attempt1.sh` | P4 (population 4) | 2 | none (no stage-1 prerequisite) |
| `run_fullspace_joint_s025_localxover_pop4_g150_seed1_attempt1.sh` | LX (local-exchange crossover) | 1 | none |
| `run_fullspace_joint_s025_localxover_pop4_g150_seed2_attempt1.sh` | LX | 2 | none |

Each was created by copying the matching seed-0 (or, for P4, seed-1)
launcher verbatim and changing only `--seed`, and for DW
`--stage1_run_dir`, and `--output_dir` — confirmed by `diff` against
the template (see commit). No other flag, budget setting, calibration
source, or model argument was touched, so these five runs stay
apples-to-apples with the existing DW/P4/LX seed-0/1 results and with
E3/J12/IA.

Not included in this batch (status unchanged, see
`EXPERIMENT_MATRIX.md` §5): SEQ (structure→quant frozen sequential
baseline, ~30–40 GPU-h/seed, no seed run yet, awaiting an explicit
go/no-go) and diversity-based parent selection (implemented, "not
planned").

## Local verification performed (no GPU)

- `bash -n` on all five scripts: pass.
- Every flag name used appears in `evo_joint_search.py`'s argument
  parser, with matching type/choices for the flags that differ
  between conditions (`--seed` int, `--sequential_mode` choice,
  `--crossover_type` choice, `--crossover_probability` float,
  `--population_size` int, `--output_dir` str).
- `diff` against each script's template shows only the intended
  seed/path fields changed.
- No local output directory under
  `thesis_results/datalab_raw/fullspace_extensions/` collides with
  any of the five new `--output_dir` paths.
- DW's two stage-1 prerequisite directories exist locally with a
  `final_candidate.json` each, and their `search_config.seed` fields
  are 1 and 2 respectively (distinct masks per seed, verified by
  hash).

A full dry run through `evo_joint_search.py` itself was not possible
here: this container has no `torch` and no GPU. The flag-level check
above is what Task 3 ("local tests, syntax checks, dry runs without
requiring the GPU") can do without one.

## Environment note (compatibility)

The original protocol-F runs (E0–E3, J12, IA, DW/P4/LX seed 0/1) and
the October replay batch both used torch 2.8.0 / transformers 4.56.2
/ flash-attn 2.8.3 / datasets 5.0.1 (`thesis_results/datalab_raw/notes/a40_python_environment_20260903.txt`;
`claude/project-state-2026-10-07.md`). DataLab's *default* environment
was replaced after that (torch 2.14 / transformers 5.17, as of
2026-09-24) — only the `~/venvs/evopress-sep2026` virtualenv still
has the matched versions. These five runs must use that venv, not
whatever is active by default, or their results will not be
comparable to the existing DW/P4/LX seed-0/1 and E3/J12/IA finals.
