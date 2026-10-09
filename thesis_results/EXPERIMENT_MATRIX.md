# Experiment matrix (verified 2026-10-08)

Every entry below was checked against run artifacts in this repository:
`thesis_results/datalab_raw/` (run summaries, commands, generation logs),
`thesis_results/fullspace_ledger.csv`, and `results/runs/`. Status values:

* **done**: run summary and final candidate present;
* **partial**: fewer seeds than the other conditions of the same comparison;
* **planned**: implemented and tested where possible, **not run**;
* **proposed**: needs a decision (compute) before implementation or launch;
* **not run**: launched or implemented, no usable output.

The thesis research questions are RQ1–RQ3 (Chapter 1). The research log's
internal RQ1–RQ8 numbering is not used here.

## 1. Protocols

| Protocol | Model | Searched scope and levels | Budget rule | Calibration (search) | Schedule | Final evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| **F: full-space, paper-matched** | Mistral-7B-v0.3, FP16, FlashAttention-2 | all 224 projections; GPTQ levels 2–6 | exact group-wise budget, T = 26,982,023,168 bits (b_ref = 3, g = 128, 16-bit scale and zero point) | FineWeb-Edu (sample-10BT, first half, shuffled with seed 0), 524,288 tokens, sequences up to 8,192 | λ = 128; survivors 16/4/1 (final stage P); tokens 2,048/16,384/131,072; KL to dense model | WikiText-2 test and C4 validation perplexity, sequence length 8,192; final calibration KL on all 524,288 calibration tokens |
| **S: screening, q_proj** | Mistral-7B-v0.3, FP16, SDPA | 32 q_proj matrices; levels 2–4 | active-average (per size group, 3 bits) | WikiText-2 train, 8,192 tokens, length 1,024 | λ = 16; 32 initial candidates on 512 tokens; survivors 8/2/1; tokens 512/2,048/8,192; KL | WikiText-2 test perplexity, length 1,024 (multi-dataset PPL and LM-eval for selected runs) |
| **S-attn: screening, attention** | as S | 128 attention projections (q, k, v, o); levels 2–4 | active-average | as S | as S; standard joint G50 used survivors 4/2/1, IA G50 8/2/1 | as S |
| **Tiny: TinyLlama screens** | TinyLlama-1.1B-Chat-v1.0 | q_proj; levels 2–4 | active-average | WikiText-2, 4,096 tokens, length 1,024 | λ = 8; 16 initial candidates; survivors 2/1; tokens 512/2,048; s = 12.5% | WikiText-2 perplexity (4,096 tokens) |
| **Early: depth-only feasibility (June)** | Mistral-7B-v0.3 | sublayer masks only, 16-bit | parameter count | WikiText-2, 8,192 tokens, length 2,048 | G10/O8 | WikiText-2 perplexity |
| **R: exact-budget replays** | as F | as F | exact group-wise | — (no search) | evaluation only, rebuilt environment with the F package versions | as F, plus calibration KL |

Level databases:

* **F:** GPTQ, FineWeb-Edu 8,388,608 calibration tokens, length 8,192, group size
  128, per-channel, asymmetric, relative damping 0.01, block size 128, no
  activation ordering, block-wise propagation with 3-bit calibration context,
  levels 2–6, 224 modules (`paper_matched/prepare_db/.../resolved_config.json`).
* **S, S-attn:** reduced feasibility databases, levels 2–4, WikiText-2, 512
  calibration tokens, sequence length 128 (research log §8.2,
  `scripts/run_mistral_attention_quant_db.sh`).

Software and hardware (F): one NVIDIA A40, torch 2.8.0, transformers
4.56.2, flash-attn 2.8.3, datasets 5.0.1, lm_eval 0.4.13, Python 3.11
(`notes/a40_python_environment_20260903.txt`, `inventory/ENVIRONMENT.txt`).

## 2. Full-space experiments (protocol F)

Search effort per seed follows the generation cost of Chapter 4: P = 1: 149 evaluations and
1,179,648 tokens per generation; P = 4: 152 evaluations and 1,572,864 tokens.
Metrics: mean ± population SD over seeds (ledger). PPL = perplexity.

| ID | Method | s | Init | Mutation | P | Crossover | G | Seeds | Evals / tokens per seed | W2 PPL | C4 PPL | Calib. KL | Status | RQ |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| E0 | dense FP16 | 0 | – | – | – | – | – | – | – | 4.83 | 7.71 | – | done | ref |
| E1 | uniform 3-bit | 0 | – | – | – | – | – | – | – | 5.50 | 8.55 | – | done | RQ1 |
| E2 | quantization-only (EvoPress) | 0 | uniform | min(U,U) level switches | 1 | – | 150 | 0, 1, 2 | 22,350 / 176,947,200 | 5.229 ± 0.005 | 8.411 ± 0.007 | 0.073 | done | RQ1 |
| E3 | joint, standard | 0.25 | random | standard | 1 | – | 150 | 0, 1, 2 | 22,350 / 176,947,200 | 8.961 ± 0.115 | 12.497 ± 0.129 | 0.461 | done | RQ1, RQ2 |
| J12 | joint, standard | 0.125 | random | standard | 1 | – | 150 | 0, 1, 2 | 22,350 / 176,947,200 | 6.621 ± 0.202 | 9.646 ± 0.063 | 0.208 | done | RQ1 |
| J12-G20 | joint, standard | 0.125 | random | standard | 1 | – | 20 | 0 | 2,980 / 23,592,960 | 6.645 | 10.023 | 0.242 | done (1 seed) | RQ2 horizon |
| J6 | joint, standard | 0.0625 | random | standard | 1 | – | 20 | 0 | – | – | – | – | not run (no output) | – |
| IA | joint, interaction-aware | 0.25 | random | IA | 1 | – | 150 | 0, 1, 2 | 22,350 / 176,947,200 | 10.206 ± 0.645 | 13.435 ± 0.224 | 0.525 | done | RQ2 |
| IA-G20 | joint, interaction-aware | 0.25 | random | IA | 1 | – | 20 | 0 | 2,980 / 23,592,960 | 10.898 | 13.992 | 0.560 | done (1 seed) | RQ2 |
| DW | depth → joint, warm | 0.25 | depth-only mask (S-protocol, WikiText-2, G20) | standard | 1 | – | 150 | 0 | 22,350 / 176,947,200 + stage 1 | 9.234 | 12.664 | 0.472 | partial (1 seed) | RQ2 |
| DW-G20 | depth → joint, warm | 0.25 | as DW | standard | 1 | – | 20 | 0 | 2,980 / 23,592,960 + stage 1 | 9.883 | 13.430 | 0.511 | done (1 seed) | RQ2 |
| P4 | joint, population | 0.25 | 4 random, evaluated on 2,048 tokens | standard | 4 | – | 150 | 0, 1 | 22,804 / 235,937,792 | 9.016, 8.828 | 12.469, 12.469 | 0.464, 0.480 | partial (2 seeds) | RQ2 |
| LX | joint, local-exchange crossover | 0.25 | 4 random | standard | 4 | local exchange, p_c = 0.5 | 150 | 0 | 22,804 / 235,937,792 | 9.508 | 12.547 | 0.446 | partial (1 seed) | RQ2 |

Matched-seed E3 references: seed 0 8.945 / 12.641, seed 1 8.828 / 12.328,
seed 2 9.109 / 12.523 (W2 / C4).

## 3. Screening experiments (protocols S, S-attn, Tiny, Early)

All under the active-average budget unless stated; results are not
numerically comparable with protocol F.

| Family (`results/runs/` prefix or directory) | Content | G | Seeds | Status | RQ |
| --- | --- | ---: | --- | --- | --- |
| `thesis_medium_quant_*` | S quantization-only | 20 | 0–2 | done | RQ1 (screen) |
| `thesis_medium_depth_*` | 16-bit depth-only masks at 25% | 20 | 0–2 | done | RQ1 (screen); masks reused by DW and R |
| `thesis_medium_joint_*` | S joint, standard | 20 | 0–2 | done | RQ1/RQ2 |
| `thesis_compute_matched_joint_*` | S joint, standard | 50 | 0–2 | done | RQ1/RQ2 |
| `thesis_interactionaware_*` | S joint, IA | 20, 50 | 0–2 each | done | RQ2 |
| `thesis_jointaware_*` | S joint-aware mutation, p_ja = 0.5 | 50 | 0–2 (seed 2 retried) | done | RQ2 |
| `thesis_fixedstrength_*` | S fixed structural strength k_max = 1 | 50 | 0–2 | done | RQ2 |
| `thesis_sequential_*` | S sequential: depth→quant frozen, depth→joint warm, quant→depth frozen, quant→joint warm (standard and IA) | 20 | 0–2 each | done | RQ1 (independent/sequential), RQ2 |
| `thesis_depthwarm_{standard,interactionaware}_*` | S depth→joint warm | 50 | 0–2 each | done | RQ2 |
| `thesis_component_crossover_*` | S component crossover, P = 4, p_c = 0.25 | 20 | 0–2 | done | RQ2 |
| `datalab_raw/cheap_population_crossover/` | S population (P = 1 vs 4, G20 and G50), layer-bundle and local-exchange crossover (p_c = 0.25), IA + P = 4 | 20, 50 | pop4: 0–2; others: 0 | done | RQ2 |
| `thesis_attention_*` | S-attn quantization-only (G20), joint (G50), IA (G50) | 20, 50 | 0–2 each | done | RQ1/RQ2 (scope) |
| `results/attribution/mistral_qproj_seed{0,1,2}`, `mistral_attention_full_seed{0,1,2}` | replay matrix: depth source {independent, standard, IA} × bit source {independent, standard, IA, uniform 3}, active-average repair | – | 0–2 | done | RQ3 (screen) |
| `generalization_*` | multi-dataset perplexity of S / S-attn finals (length 1,024) | – | 0–2 | done | quality |
| `lmeval_*` | LM-eval 0-shot ARC-Easy, PIQA, WinoGrande on S / S-attn finals | – | 0–2 | done | quality |
| `screen_{adaptive,coarsetofine,fixedstrength,jointaware}_tiny_*` | Tiny: adaptive (patience 3, max 3), coarse-to-fine (3→1), fixed (max 1), joint-aware p = 0 vs 0.25 | 20 | 0–2 each | done | RQ2 (screen) |
| Early depth-only (research log §8.1) | EvoPress vs random vs late-layer masks, s = 12.5–50% | 10 | 1 (37.5%: 1–3) | done | RQ1 (heuristic, 16-bit only) |

## 4. Coverage of the research questions

| RQ | Covered at the exact budget (F) | Covered only in screening | Gaps |
| --- | --- | --- | --- |
| RQ1 | E1 uniform, E2 quantization-only, joint at 12.5% and 25%; replays: heuristic masks, independent compositions, 16-bit references | sequential searches (S) | sequential search at T → proposed only; independent masks not effort-matched |
| RQ2 | standard vs IA (3 seeds), depth-warm (1), population (2), local exchange (1), horizon G20 vs G150 | joint-aware, strength schedules, component and layer-bundle crossover, sequential warm starts | single-seed DW and LX; P4 has +33% tokens |
| RQ3 | replays: shared-gene exchanges (2 repair-free pairs, 5 repaired seed pairs), plain crossings, allocation attribution | attribution replay matrices (S, S-attn) | no control for the cost of repairing optimized genes |

## 5. Planned and proposed additions (not run)

| ID | What | Cost (estimate) | Status | Tooling |
| --- | --- | --- | --- | --- |
| R-0 | Re-evaluate 7 finals + E1 in the replay harness | in R batch | **done** (exact reproduction) | `evo_exact_replay.py` |
| R-A | Repair-free shared-gene exchange: E3 seed 1 × DW; J12 seeds 0 × 1; plus plain crossing with 3 repair seeds | in R batch | **done** | `scripts/plan_exact_replays.py` |
| R-B | Shared-gene exchange for the other E3 and J12 seed pairs, exclusive repair, 3 repair seeds | in R batch | **done** | same |
| R-C | RQ1 at T: heuristic masks (late, late-keep-last, random ×3, block influence) + near-uniform precision at 25% and 12.5%; independent composition (depth-only masks × E2 profiles); joint masks × near-uniform / × E2 | in R batch | **done** | same |
| R-D | 16-bit depth-only evaluation of joint, depth-only and heuristic masks | in R batch | **done** | same |
| R total | 97 evaluation jobs | measured: 359 s per job, ~10 h total | **done 2026-10-09**, results in `results/exact_replays/replay_20261008/` | `evo_exact_replay.py` |
| D125 | 16-bit depth-only masks at 12.5% (S-protocol, 3 seeds) for 12.5% compositions | ~30 min | proposed | stage `depth125` |
| SEQ | structure → quantization, frozen mask, at T, G150 | ~30–40 GPU-hours per seed | proposed (opt-in mode implemented) | stage `frozen` |
| LM | LM-eval on full-space finals (E1, E2, J12, E3) | a few GPU-hours | proposed | `lmeval.py` |
| +seeds | extra seeds for DW, P4, LX at G150 | ~35–45 GPU-hours each | proposed (supervisor decision) | existing launchers |

Not planned: diversity-based parent selection (implemented, never run); the
6.25% run; further screening.

## 6. Caveats that affect interpretation

1. **Exact-budget quantization children can exchange inactive genes.** In
   protocol F, the standard joint mutation's quantization child draws its level
   exchange from all projections of a size group (`drop_state=None`) and is then
   repaired. Measured mean changed genes per quantization child: 2.36 at 25%,
   2.20 at 12.5% (expected 2.375 / 2.219; 2.0 if active-only), see
   `offspring_gene_changes.md`. About 6% of quantization children at 25% change
   only inactive genes and are functionally identical to their parent.
   Interaction-aware mutation and quantization-only search are not affected.
2. **Resumed runs.** E2 seeds 0/1/2 (12/4/2 attempts), E3 seed 1 (3), P4 seed 0
   (resumed at G18): wall-clock time must be summed over attempts.
3. **Duplicate log rows.** The DW G150 generation log records every generation
   twice (rows differ only in cumulative runtime); deduplicate by generation.
4. **Unequal effort.** P4 and LX evaluate 33% more tokens than P = 1; DW adds the
   stage-1 depth-only search (S protocol) outside the counted effort.
5. **Different settings across protocols.** LX used p_c = 0.5 and 4 evaluated
   initial candidates in F, p_c = 0.25 and 32 initial candidates in S.
6. **Depth-only 16-bit reference (8.66 / 12.04 at 25%)** is a supplied paper value,
   not a measurement of this thesis; R-D would measure it for our masks.
7. **Screening databases** were built from 512 WikiText-2 tokens with levels 2–4;
   screening numbers are not comparable with F.
