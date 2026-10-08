# DataLab raw artifacts (imported 2026-10-08)

Lightweight run artifacts copied from the TU Wien DataLab home directory
(`/home/jovyan`) on 2026-10-08. Nothing here was re-run or edited; files are
byte-for-byte copies. The consolidated table of final metrics is
[`../fullspace_ledger.csv`](../fullspace_ledger.csv).

Source archive: `thesis_artifacts_20261008_1102.tar.gz`
(SHA-256 `2923105a06edae6d16ec8433b4659488dee68055f1139d0b1ea9cffe5e45adec`),
still present in the DataLab home directory.

## Layout

| Directory | DataLab source | Contents |
| --- | --- | --- |
| `paper_matched/` | `evopress/results/apples_to_apples/paper_matched/` | E0 dense, E1 uniform-3, E2 quant-only (seeds 0–2), E3 joint 25% (seeds 0–2), GPTQ database preparation; **all attempts**, including interrupted and resumed ones |
| `fullspace_extensions/` | `evopress_extension_results/*/` | Full-space joint extensions: 12.5% G20/G150, IA G20/G150, depth-warm G20/G150, population-4 G150, local-exchange G150 |
| `cheap_population_crossover/` | `evopress-crossover-v{2,3}/outputs/experiments/joint_g*` | Cheap q_proj population/crossover runs (G20/G50): pop1, pop4, component, layer-bundle, local-exchange, IA+pop4 |
| `launchers/` | `~/run_*.sh` | Launch scripts for the full-space extension runs |
| `notes/` | `~/` and `~/evopress_backups/` | Crossover/population notes and ledger, A40 Python environment, archive SHA-256 files |
| `inventory/` | collector output | `RUN_INDEX.tsv` (all 407 run summaries found), `MANIFEST.tsv` (every thesis text file with size/mtime), `ENVIRONMENT.txt`, `NOT_IMPORTED.tsv` |

## Not imported

`run.log`, `*.log`, `resource_samples.jsonl` and any file over 5 MB are listed in
`inventory/NOT_IMPORTED.tsv` and remain on DataLab. Binary files
(`search_checkpoint.pt`, quantization database, teacher-logit cache) were never
collected.

## Caveats for analysis

- **Resumed runs.** E2 seed 0 (12 attempts), E2 seed 1 (4), E2 seed 2 (2),
  E3 seed 1 (3) and population-4 G150 seed 0 (resumed at G18) were interrupted
  and resumed from checkpoints. The `runtime_seconds` of the final attempt
  covers only that attempt. Total wall-clock time must be summed over attempts
  before any runtime comparison.
- **Empty run.** `fullspace_joint_s00625_g20_seed0_attempt1` (6.25% sparsity)
  produced no output files on DataLab; its launcher is in `launchers/`.
- **E0/E1 precision.** Dense and uniform-3 perplexities are stored with two
  decimals in their summaries (4.83/7.71 and 5.50/8.55).
- **Active bit average.** For 12.5% joint runs the active searched average is
  exactly 97/28 = 3.4643 bits in every seed. The per-seed values 3.44–3.49 in
  `../fullspace_joint_s0125_g150_3seed/README.md` are a different quantity
  (searched average including inactive genes).
- **`/share` storage.** On 2026-10-08 the DataLab server had no `/share` mount.
  Backups that the research log places under `/share/kerim.halilovic/...` should
  not be assumed to exist; copies are in `~/evopress_backups/` and
  `~/thesis_backups/`.
