# EvoPress apples-to-apples audit: quantization-only versus joint depth + quantization

Date: 2026-08-18

Status: implementation and CPU-side validation complete. The first Stage 1
database attempt ended with a DataLab server/container restart after writing a
`status=generating` manifest but before producing any module reconstruction.
The legacy retained-activation design required far more than the detected
16 GiB CPU-memory cgroup and is the leading explanation, but no surviving OOM
trace proves the exact termination mechanism or phase. No E0--E3 experiment was
launched. A lossless disk-streaming retry path is implemented but has not yet
been launched.

## 1. Paper configuration

The reproduction target is the quantization experiment in *Accurate Dynamic Model Compression via Evolutionary Search* ([paper](https://arxiv.org/abs/2410.14649), [upstream repository](https://github.com/IST-DASLab/EvoPress)). The paper reports the following Mistral-7B-v0.3 WikiText-2/C4 perplexities: dense `4.82/7.72`, uniform GPTQ 3-bit `5.54/8.57`, and EvoPress KL 3-bit `5.21/8.42`.

The upstream launchers, not this fork's currently modified `scripts/run_gptq.sh`, define the paper-like setup ([GPTQ launcher](https://github.com/IST-DASLab/EvoPress/blob/main/scripts/run_gptq.sh), [search launcher](https://github.com/IST-DASLab/EvoPress/blob/main/scripts/run_quant_search.sh)):

| Item | Paper/upstream setting |
| --- | --- |
| Model | `mistralai/Mistral-7B-v0.3` |
| Quantized modules | all `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` weights in all 32 blocks: 224 modules |
| GPTQ levels | 2, 3, 4, 5, and 6 bits |
| GPTQ group size | 128 |
| GPTQ details | per-output-channel, asymmetric default, no activation ordering, relative dampening 0.01, block size 128 |
| GPTQ database calibration | FineWeb-Edu, 8,388,608 tokens, sequence length 8,192; sequential calibration level 3; 8 processes |
| Search calibration | FineWeb-Edu, 524,288 tokens, sequence length 8,192 |
| Fitness | dense-teacher KL divergence |
| Search | 150 generations, 128 offspring |
| Selection | 16 survivors at 2,048 tokens; 4 at 16,384; 1 at 131,072 |
| Diagnostics/evaluation | upstream requests FineWeb-Edu, WikiText-2, and C4, 524,288 requested tokens, sequence length 8,192, every 5 generations |
| Grouping | modules are grouped by equal weight count; mutations exchange integral levels within one equal-size group |
| Tokenizer | slow tokenizer, because upstream does not opt into `--use_fast_tokenizer` |

The new `paper_matched` profile retains all search settings above. Its final evaluation includes WikiText-2 and C4, as requested here, but omits the additional FineWeb-Edu diagnostic to avoid extra nonessential evaluation compute. This does not alter search selection. The omission is a documented deviation from the upstream launch script.

## 2. Current implementation before the new exact-budget path

### Candidate representations and scopes

1. A quantization-only candidate in `evo_quant_search.py` is a nested Python list. Each outer entry corresponds to a `group_layers(...)` group and each inner integer is the selected database level for the aligned module name. With `group_rule=size`, Mistral's full projection scope produces three equal-weight groups.
2. A joint candidate in `evo_joint_search.py` is `{"drop": {"attn": [...], "mlp": [...]}, "quant": [[...], ...]}`. `True` means bypass the corresponding submodule; the quantization component uses the same nested representation as quantization-only search.
3. Search scope is inferred from subdirectories in `quant_weights_path`; the search script itself does not impose a projection regex. Therefore the database determines whether the search is q-projection-only, attention-only, or full-projection.
4. The currently checked-in debug `scripts/run_gptq.sh` selects only `q_proj`, levels 2--4, WikiText-2, 512 calibration tokens, and length 128. It is not the upstream paper launcher.
5. Recorded thesis runs used either 32 `q_proj` modules or 128 attention projection modules (`q/k/v/o`). None used the full 224-module `q/k/v/o/gate/up/down` scope. The new profile is separate and does not change those old experiments.
6. Depth masks independently drop the complete attention submodule and the complete MLP submodule of a block. With `--drop_entire_block`, the masks are tied; recorded Mistral runs and the new standard joint profile leave them independent.

### Legacy budget behavior

7. Quantization-only EvoPress computes `sum(weight_numel * assigned_level)` and preserves the target parameter-weighted average. Under `group_rule=size`, its exchange mutation preserves that cost exactly.
8. Legacy joint search fixes the number of dropped attention and MLP modules using `floor(drop_sparsity * number_of_blocks)`. Its optional `--active_quant_budget` separately forces the average assigned bits among active searched weights back to the nominal target in each equal-size group.
9. That legacy joint convention is not a total-model-size constraint. It combines a fixed depth count with an active average bit target, so `3-bit + depth` is more compressed than quantization-only 3-bit.
10. The legacy reporting function excluded parameters under dropped attention/MLP parents, but this reporting value did not constrain search feasibility.
11. Under `--active_quant_budget`, assignments belonging to dropped modules were ignored by the active-average calculation. Without that option they were counted. The new total-cost path always charges dropped module weights and their group metadata as zero.
12. No legacy search-budget formula included GPTQ scale or zero-point storage. Fixed FP16 parameters were also outside the evolutionary constraint.

### Quantization database representation

13. Each `level_database/module_name/N.pth` file is a dequantized floating-point reconstruction of that module's quantized weight, not a packed N-bit tensor. `FastOBQ` internally produces integer codes plus scale/zero tensors, and `QLinear.get_weight()` reconstructs a floating-point tensor before it is saved.
14. Candidate evaluation replaces the dense module's weight data with that reconstruction. Consequently current checkpoints/database files do not physically occupy the theoretical packed size reported by the budget function.
15. Joint evaluation still loads level weights for dropped modules, then bypasses their forward calls. Their tensors remain allocated. Zero cost for a dropped module describes the intended exported compressed architecture, not the present in-memory Python object.

### Initialization, mutation, and selection

16. Original quantization-only EvoPress starts an integral target such as 3-bit from the uniform parent with fitness `inf`; it does not run a separate initialization selection. Each offspring applies a biased 1--3 same-group level exchange. The current fork had added an optional single uniform evaluation; the paper profile now explicitly skips it.
17. Standard joint initialization samples fixed-cardinality attention and MLP masks and generates/repaints a quantization profile. Previous joint runs typically evaluated 32 initial candidates. The paper profile uses one random exact-budget parent and deliberately skips its separate initialization evaluation so total search compute matches quantization-only EvoPress exactly.
18. Standard joint mutation dispatches depth-only or quantization-only proposals. A depth mutation swaps currently kept/dropped locations without changing counts; a quantization mutation performs one active same-size level exchange. This operator is not identical to quantization-only EvoPress's biased 1--3 exchange operator, and that difference is inherent to the standard joint search space.
19. Interaction-aware mutation, joint-aware mutation, sequential/warm initialization, adaptive/coarse-to-fine mutation, persistent populations, and component crossover remain available for existing experiments. The first apples-to-apples joint profile rejects or disables them: standard mutation, population 1, crossover probability 0, sequential mode `none`.
20. Both searches use the paper's successive-halving stages and add the incumbent parent only at the final stage. The joint implementation can generalize the final survivor count for persistent populations, but the apples profile uses the original one-parent behavior.

### Defaults and representative prior runs

| Setting | Quant-only CLI default | Joint CLI default | Representative previous Mistral runs | New paper profile |
| --- | ---: | ---: | ---: | ---: |
| Generations/offspring | required | required | 20 or 50 / 16 | 150 / 128 |
| Initial candidates | optional for integral target | required | 32 | 1, unevaluated on both sides |
| Survivors | required | required | usually 8/2/1; attention quant-only used 4/2/1 | 16/4/1 |
| Selection tokens | required | required | 512/2,048/8,192 | 2,048/16,384/131,072 |
| Calibration tokens | 524,288 | 2,048 | 8,192 | 524,288 |
| Sequence length | min(model max, 8,192) | min(model max, 8,192) | 1,024 | 8,192 |
| Fitness | KL | KL | KL | KL |
| Evaluation datasets | FineWeb/WikiText-2/C4 | WikiText-2 | WikiText-2 | WikiText-2/C4 |
| Seed | 0 | 0 | 0, 1, 2 | 0, 1, 2 for stochastic searches |
| Group rule | size | none | size | size |
| Quant levels/scope | database-defined | database-defined | 2--4; q-only or attention-only | 2--6; all seven projections |
| Tokenizer | slow unless opted in | slow unless opted in | fast | slow, matching upstream |
| Attention implementation | optional | optional | SDPA | FlashAttention 2 |
| Budget | searched-weight average | legacy fixed depth + optional active average | active average 3.0 | exact common total cost |

The recorded q-projection database itself was a very small feasibility database: WikiText-2, 512 tokens, sequence length 128, levels 2--4, and one process. This is much farther from the paper than only the later search schedule would suggest.

## 3. Compression-cost definition

`src/compression_budget.py` now supplies one `candidate_compression_cost(...)` implementation for raw quantization-only candidates, raw joint candidates, and serialized candidate mappings.

Let `A_Q` be active searched linear weights, `A_F` active parameters outside the quantization scope, and `D` parameters under removed attention/MLP modules. For a searched linear weight `i`, let `n_i` be its number of elements and `b_i` its selected integral level. For a linear matrix with `out_i x in_i` weights and GPTQ group size `G=128`, the number of per-channel group metadata entries is `out_i * (in_i/G)`. The primary accounting convention is

```text
C(candidate) = sum(i in A_Q, n_i * b_i)
             + 16 * |A_F|
             + sum(i in A_Q, out_i * (in_i / 128) * (16 scale bits + 16 zero bits)).
```

Every parameter in `D` contributes zero, and metadata for a removed searched module also contributes zero. Dense reference cost is `16 * total_dense_parameters`. Biases or other parameters outside the searched weights remain FP16 unless their containing attention/MLP parent is removed.

This is an ideal packed-storage estimate. It includes FP16 GPTQ scales and zeros because those are the dtypes created by this repository at FP16, but excludes allocator/packing padding, file/container headers, module names, bit-profile/topology descriptors, and executable code. It is more complete than the paper's searched-weight average, while remaining exactly common to both methods. The report also retains `paper_weight_only_cost_bits`, which omits scale/zero metadata, for comparison with the paper's convention.

The exact-budget repair solves a bounded subset-sum over available database levels. For the apples profile it repairs each equal-weight group independently, preserving the original compatible-switch rule. It validates every initial candidate, every accepted offspring, and the final candidate; an unrepresentable target raises an error rather than silently rounding.

## 4. Target budget

The counts below are derived from the actual Mistral-7B-v0.3 architecture represented by this repository and are asserted by a zero-allocation meta-model test. Full runs additionally derive them from the live loaded model and abort if they differ from the configured references. A newly generated level-database manifest records and validates the same counts.

| Quantity | Exact value |
| --- | ---: |
| Dense parameters | 7,248,023,552 |
| Searched projection weights | 6,979,321,856 |
| Fixed FP16 parameters | 268,701,696 |
| Dense FP16 cost | 115,968,376,832 bits = 14,496,047,104 bytes = 13,824.5078125 MiB |
| Uniform/EvoPress 3-bit weight-only cost | 25,237,192,704 bits = 3,154,649,088 bytes = 3,008.5078125 MiB |
| GPTQ scale + zero metadata | 1,744,830,464 bits = 218,103,808 bytes = 208 MiB |
| **Primary common target** | **26,982,023,168 bits = 3,372,752,896 bytes = 3,216.5078125 MiB (3.1411209106 GiB)** |
| Dense / target compression ratio | **4.2979867043x** |
| Paper-style weight-only compression ratio | 4.5951377474x |

All quantization-only EvoPress candidates at a weighted 3-bit target have this same total cost because every searched module remains active and metadata count does not depend on its selected level.

The standard joint profile fixes 25% independent submodule dropping: 8 attention and 8 MLP submodules. This removes exactly 1,744,830,464 searched parameters, leaving 5,234,491,392 active searched weights. At the exact common target it has:

| Joint component | Exact bits |
| --- | ---: |
| Fixed FP16 parameters | 4,299,227,136 |
| Active quantized weight codes | 21,374,173,184 |
| Active GPTQ scale/zero metadata | 1,308,622,848 |
| Total | 26,982,023,168 |

The active assigned weight average is exactly `49/12 = 4.083333...` bits. Thus depth savings are genuinely reallocated to higher precision rather than added on top of a 3-bit model. Target and realized difference are 0 bits and 0% for every accepted candidate in this configured search space.

## 5. Search-space comparison

Both methods use the same 224-module database and the same levels 2--6. `group_rule=size` gives:

| Equal-weight group | Modules | Modules before/after 25% joint dropping | Quant-only level sum | Joint active level sum |
| --- | --- | ---: | ---: | ---: |
| 4,194,304 weights | `k_proj`, `v_proj` | 64 / 48 | 192 | 196 |
| 16,777,216 weights | `q_proj`, `o_proj` | 64 / 48 | 192 | 196 |
| 58,720,256 weights | `gate_proj`, `up_proj`, `down_proj` | 96 / 72 | 288 | 294 |

Quantization-only varies only the per-module levels while preserving each equal-size group's level sum. Joint search also varies which eight attention and which eight MLP submodules are bypassed. Exact repair adjusts only active assignments and keeps each group's total model-size contribution equal to its uniform-3-bit reference contribution. Assignments stored for inactive modules remain in the genotype for compatibility but consume no budget and do not affect inference.

## 6. Search-compute comparison

### `paper_matched`

Both E2 and E3 use 150 generations, 128 accepted offspring, and the 16/4/1 schedule. Each generation evaluates 128, 16, and 5 candidates at the three stages; the last number includes four second-stage survivors plus the incumbent.

```text
candidate evaluations per generation = 128 + 16 + 5 = 149
candidate-tokens per generation       = 128*2,048 + 16*16,384 + 5*131,072
                                      = 1,179,648
total search evaluations              = 150*149 = 22,350
total search candidate-tokens         = 150*1,179,648 = 176,947,200
```

Both start with a single unevaluated parent, so these totals are exactly equal. Offspring proposals rejected as duplicates/infeasible are separately logged as attempts but do not consume fitness-evaluation tokens.

Teacher-logit construction, periodic perplexity diagnostics, and final evaluation are outside the search candidate-token count. Their configurations are identical between E2 and E3. Wall-clock differences remain scientifically relevant because joint depth dispatch/repair and bypassed computation can change overhead and forward cost.

### `compute_matched`

The optional smaller profile gives both methods 20 generations, 16 offspring, 8/2/1 survivors, 512/2,048/8,192 selection tokens, and one initial evaluation at 512 tokens. Each has 541 search candidate evaluations and 983,552 search candidate-tokens. It uses SDPA, 131,072 calibration/evaluation tokens, and length 2,048. This profile is compute-matched to itself but is not a paper reproduction.

Each generation CSV records accepted offspring, proposal attempts, candidates/tokens at each stage, cumulative candidates/tokens, selected best search fitness/KL, cumulative wall time, and peak GPU memory. The final JSON records total values and launcher wall time.

## 7. Changes made

- `src/compression_budget.py`: shared cost, exact target derivation, database inspection, exact feasibility validation, and bounded repair.
- `evo_quant_search.py`: opt-in exact total budgeting, live-reference validation, database scope/level checks, paper-compatible unevaluated integer initialization, exact counters, and structured cost/runtime reporting.
- `evo_joint_search.py`: opt-in exact total budgeting and repair for every candidate, live-reference validation, paper-compute-matched single-parent initialization, and detailed evaluation/attempt/runtime counters.
- `src/run_reporting.py`: metadata-inclusive storage breakdown, expanded generation counters/KL logging, and atomic JSON replacement.
- `src/activation_cache.py`: exclusive, non-resumable, lossless per-sequence disk storage for complete block input args/kwargs, with ordered one-record-at-a-time reads, atomic per-record hidden-state replacement, block-boundary progress/byte counters, and an attempt-linked manifest.
- `src/memory_utils.py`: cgroup-v2 memory snapshots and explicit release of unused CPU allocator arenas.
- `src/model_utils.py` and `src/quantizer.py`: optional activation streaming that preserves the original two-pass sequential-GPTQ semantics while retaining only one calibration record in CPU memory at a time.
- `quant.py`: effective seeding, configured-shard emulation for explicit smaller process counts, ordered calibration-token digests, verified direct-to-GPU parameter and buffer placement, a 32 GiB GPU-capacity floor for disk mode, exclusive database creation, lifecycle/attempt provenance, device-byte inventories, cgroup/CUDA memory records, activation-cache statistics, and per-reconstruction shape/dtype/size/SHA-256 metadata.
- `configs/apples_to_apples/mistral7b_v03_paper_matched.json`: full paper-schedule profile, including the exact expected Mistral projection shapes and FP16 reconstruction dtype used by generation and postflight validation.
- `configs/apples_to_apples/mistral7b_v03_compute_matched.json`: smaller equal-compute profile.
- `requirements.txt`: explicit prebuilt FlashAttention 2.8.3 wheel for the audited DataLab Python 3.11/PyTorch 2.8/CUDA 12/CXX11-ABI environment; this avoids an unavailable local CUDA-toolkit build.
- `scripts/run_apples_to_apples.py`: non-overwriting database/evaluation/search launcher with an explicit disk-cache mode; path-separation, nonexistence, non-`tmpfs`/`ramfs`, write/fsync, capacity, and inode preflights; line-flushed child logging, attempt-linked cgroup/process-tree samples, and orphan-safe process-group cleanup; lifecycle status; strict search-summary checks; and full reconstruction/cache postflight validation before completion.
- `scripts/smoke_stage1_memory_path.py`: non-quantizing one-A40 preflight that loads the exact Mistral model directly on CUDA, verifies FP16 parameter/buffer residency and architecture counts, and round-trips one real 8,192-token first-block record through the disk cache without touching the production database.
- `scripts/aggregate_apples_to_apples.py`: strict equal-budget aggregation, mean/std table, and paired `joint - quant_only` seed differences.
- `tests/test_compression_budget.py`: exact cost and repair tests, including the Mistral architecture constants.
- `tests/test_apples_to_apples_workflow.py`: config, command, dry-run, one/eight-process calibration-prefix equivalence, manifest/file validation, memory-mode CLI, and aggregation tests.
- `tests/test_activation_cache.py`: lossless ordered Mistral-shaped payload tests, collision safety, hidden-state propagation, and a real two-block FastOBQ/QLinear disk-versus-memory bit-exact comparison.
- `tests/test_smoke_stage1_memory_path.py`: CPU-only validation of smoke-test path ownership/cleanup, model residency checks, exact Mistral architecture assertions, and captured-input validation.

All new behavior is opt-in. Legacy CLI budget modes and previous result files remain unchanged.

## 8. Validation

### DataLab Stage 1 failure audit

The failed one-process attempt is retained, not repaired or reused:

- launcher artifacts: `/home/jovyan/evopress/results/apples_to_apples/paper_matched/prepare_db/mistral7b_full_gptq_db`;
- partial database: `/home/jovyan/evopress_quant/Mistral-7B-v0.3/3bit`;
- database contents: only a `status=generating` manifest, zero module directories, and zero reconstructions;
- launcher contents: command/resolved configuration and an empty buffered log,
  with no runtime/status file because the process/container did not return
  control to that version of the launcher before the server restarted.

The manifest proves that 7,513 sequences and all 8,388,608 requested tokens
were loaded. Emulation of the configured eight-way floor split retained the
first 7,512 sequences (8,387,504 tokens) and dropped one 1,104-token tail. The
retained tensor payload implied by the observed Mistral block inputs is
estimated as:

```text
FP16 hidden states = 8,387,504 * 4,096 * 2 = 68,710,432,768 bytes = 63.9916 GiB
FP16 RoPE cos/sin  = 8,387,504 * 128 * 2 * 2 = 4,294,402,048 bytes = 3.9995 GiB
position/cache IDs = at most 8,387,504 * 8 * 2 = 134,200,064 bytes = 0.1250 GiB
estimated activation tensor payload                  73,139,034,880 bytes = 68.1160 GiB
FP16 dense parameters = 7,248,023,552 * 2            14,496,047,104 bytes = 13.5005 GiB
combined tensor payload                               87,635,081,984 bytes = 81.6165 GiB
```

The activation estimate assumes FP16 hidden/RoPE tensors and two int64 ID
tensors for the recorded Transformers 4.56 Mistral input structure; aliasing or
additional kwargs can change the realized serialized size. It also excludes
Python and serialization overhead. Even without that overhead, the legacy
algorithm could not complete while retaining this payload under the exact
17,179,869,184-byte cgroup limit. The partial artifacts are consistent with a
failure during block-input collection, but the empty log and reset cgroup event
counters mean neither the exact phase nor a kernel OOM action is forensically
proven.

The retry loads the model directly onto the assigned CUDA device and rejects
disk mode unless every model parameter and buffer is observed there. It also
requires at least 32 GiB total GPU memory; the audited A40 reports about
44.4 GiB. Complete `(args, kwargs)` records are serialized losslessly,
processed one at a time in original order, and updated only after all seven
projections of a block have been replaced by their calibration-level QLinear
modules. It does not batch or pad samples, so the calibration examples and the
original two-pass sequential-GPTQ semantics are unchanged. A clean two-block
FastOBQ test produces `torch.equal` reconstructions and final outputs for the
in-memory and disk paths.

The estimated activation-cache tensor payload is 73,139,034,880 bytes
(68.1160 GiB); the actual serialized scratch size is not known until generation
and is recorded by cache telemetry. The exact raw tensor payload of the five
FP16 reconstruction levels is
`6,979,321,856 * 5 * 2 = 69,793,218,560` bytes (65.0000 GiB), before `.pth`
container/filesystem overhead. Their combined estimated/raw payload is
133.1160 GiB. The launcher therefore requires at least 100 GiB free for each
output when they are on separate filesystems, or 200 GiB when they share one.
Across initial collection and two reads plus one rewrite for each of 32 blocks,
the cache causes roughly `68.116 * (1 + 3*32) = 6,607.3 GiB` (about 6.45 TiB)
of cumulative scratch I/O.

Cache creation is exclusive and deliberately non-resumable. Each record is
replaced atomically, but the manifest is checkpointed only after initial
collection and each complete block. A crash during a block can therefore leave
mixed-generation records and stale counters; the launcher never treats that as
a reusable checkpoint. Telemetry records the attempt ID, status, sample count,
current size, cumulative bytes read/written, and completed blocks. Only a child
exit code of zero followed by a `status=complete` database manifest, a passing
postflight, and launcher `status=completed` establishes Stage 1 success.

The production postflight reopens all 1,120 expected files and validates the
exact 224 module names and five levels; FP16 dtype; shapes `q_proj/o_proj =
[4096, 4096]`, `k_proj/v_proj = [1024, 4096]`, `gate_proj/up_proj = [14336,
4096]`, and `down_proj = [4096, 14336]`; nonempty/finite/contiguous CPU
tensors; per-file size and SHA-256; and the total database payload. It also
cross-checks the completed cache manifest, attempt ID, 7,512 records, and 32
completed blocks. Calibration-token digests bind dtype, shape, record
boundaries, and order. These hashes are recomputed against metadata created by
the same run; they detect corruption or mismatch but are not an externally
published known-good paper checksum.

Executed without a full model download or GPU experiment:

- `python -m pytest -q tests/test_compression_budget.py tests/test_apples_to_apples_workflow.py tests/test_run_reporting.py tests/test_eval_ppl_compression_loading.py --disable-warnings`: passed.
- Component-crossover, joint-aware, and sequential-search tests: passed.
- The activation-cache, real-model smoke helpers, I/O integrity, and launcher/workflow target passes **34 tests**.
- Complete isolated suite after memory-safe hardening: **184 passed**.
- `python -m py_compile` on every changed Python entry point: passed.
- `git diff --check`: passed.
- Paper and compute profile dry runs: passed; commands contain the intended full scope, exact target assertions, schedules, and slow-tokenizer behavior.
- The exact planned DataLab retry dry-run passed and contains one process,
  expected module count 224, no CPU module offload, direct-GPU loading, and the
  unique disk-cache path.

The first unisolated full-suite attempt reached 95 passes and failed one launcher dry-run test because existing repository result directories were detected as completed. The pre-spill implementation's latest isolated suite passed all 161 tests. Optional launcher dependency probes remain disabled locally because `datasets`, `accelerate`, and `sentencepiece` are not installed; the DataLab environment has already verified those dependencies and the FlashAttention CUDA kernel.

No multi-hour Mistral search, GPTQ generation, or perplexity evaluation was run during implementation.

## 9. Exact full-run commands

Run from the repository root. The paths below are the planned unique DataLab
retry targets. They intentionally do not reuse the failed attempt.

```bash
PAPER_CONFIG=configs/apples_to_apples/mistral7b_v03_paper_matched.json
DB_ROOT=/home/jovyan/evopress_quant_stage1_1gpu_diskspill_attempt2_20260818
QUANT_DB="$DB_ROOT/Mistral-7B-v0.3/3bit"
ACTIVATION_CACHE=/tmp/evopress_activation_spill_stage1_attempt2_20260818
```

After the DataLab filesystem and dependency checks, but before database
generation, exercise the actual one-A40 model-loading and disk-record path
without creating a quantization database:

```bash
python scripts/smoke_stage1_memory_path.py \
  --scratch-parent /tmp \
  --device-index 0 \
  --seed 0
```

Generate the one shared GPTQ level database:

```bash
python scripts/run_apples_to_apples.py prepare_db --config "$PAPER_CONFIG" --seed 0 --quant-db-root "$DB_ROOT" --run-id mistral7b_full_gptq_db_8gpu
```

The command above retains the upstream eight-process default. The Stage 1
DataLab preflight on 2026-08-18 exposed one NVIDIA A40, and the one-process
deviation was explicitly authorized. For that allocation, use:

```bash
python scripts/run_apples_to_apples.py prepare_db --config "$PAPER_CONFIG" --seed 0 --quant-db-root "$DB_ROOT" --run-id mistral7b_full_gptq_db --torchrun-processes 1
```

The command immediately above is no longer safe under DataLab's 16 GiB CPU
cgroup. The required one-A40 invocation is:

```bash
python scripts/run_apples_to_apples.py prepare_db \
  --config "$PAPER_CONFIG" \
  --seed 0 \
  --quant-db-root "$DB_ROOT" \
  --run-id mistral7b_full_gptq_db_1gpu_diskspill_attempt2_20260818 \
  --torchrun-processes 1 \
  --database-memory-mode disk_activation_cache \
  --activation-cache-dir "$ACTIVATION_CACHE"
```

Before launch, the launcher rejects any existing database, spill, or run leaf;
rejects nested output paths and memory-backed `tmpfs`/`ramfs`; performs a real
create/write/fsync/unlink permission probe; and requires at least 100 GiB free
on each separate filesystem (200 GiB if they share one), plus the configured
inode floor. Checks are performed against the nearest existing ancestor before
the exclusive leaves are created. The spill performs roughly 6.45 TiB of
cumulative read/write traffic across 32 blocks, so a local disk-backed `/tmp`
is preferred over the network-mounted home filesystem only after the preflight
confirms its filesystem type, writable/fsync behavior, free capacity, and
inodes; a `tmpfs` `/tmp` is rejected. The spill is execution scratch and is not
part of the scientific database. Actual cache and database payload sizes are
recorded rather than inferred from the preflight estimates.

This override does not modify the `paper_matched` profile. Database generation
first truncates to the same calibration-sequence prefix selected by the
configured eight-way floor-division rule, then processes that prefix on the
effective worker count. Thus one and eight processes consume the same examples
and are algebraically equivalent at the Hessian level; floating-point
accumulation and collective-reduction order can still produce small numerical
differences. Both configured and effective counts, loaded/used calibration
counts, and the override flag are recorded in the manifest and resolved run
configuration. E1--E3 all reuse this one database, so this infrastructure
deviation does not create an asymmetry between the compared methods.

E0 and E1:

```bash
python scripts/run_apples_to_apples.py dense --config "$PAPER_CONFIG" --seed 0 --run-id E0_dense
python scripts/run_apples_to_apples.py uniform3 --config "$PAPER_CONFIG" --seed 0 --quant-db "$QUANT_DB" --run-id E1_uniform3
```

E2, quantization-only EvoPress:

```bash
python scripts/run_apples_to_apples.py quant_only --config "$PAPER_CONFIG" --seed 0 --quant-db "$QUANT_DB" --run-id E2_quant_only_seed0
python scripts/run_apples_to_apples.py quant_only --config "$PAPER_CONFIG" --seed 1 --quant-db "$QUANT_DB" --run-id E2_quant_only_seed1
python scripts/run_apples_to_apples.py quant_only --config "$PAPER_CONFIG" --seed 2 --quant-db "$QUANT_DB" --run-id E2_quant_only_seed2
```

E3, clean standard joint search:

```bash
python scripts/run_apples_to_apples.py joint --config "$PAPER_CONFIG" --seed 0 --quant-db "$QUANT_DB" --run-id E3_joint_seed0
python scripts/run_apples_to_apples.py joint --config "$PAPER_CONFIG" --seed 1 --quant-db "$QUANT_DB" --run-id E3_joint_seed1
python scripts/run_apples_to_apples.py joint --config "$PAPER_CONFIG" --seed 2 --quant-db "$QUANT_DB" --run-id E3_joint_seed2
```

Aggregate after all runs finish:

```bash
python scripts/aggregate_apples_to_apples.py --input-root results/apples_to_apples/paper_matched --output-dir artifacts/apples_to_apples_paper_matched_aggregate
```

The launcher refuses to overwrite a nonempty database target, nonempty run directory, or existing aggregate output. Use a new run ID/output directory for every retry.

## 10. Expected result files

Default run directories are `results/apples_to_apples/<profile>/<method>/<run-id>/`.

Every run contains `resolved_config.json`, `command.sh`, `run.log`,
`resource_samples.jsonl`, `runtime.json`, and `launcher_status.json`;
completed experiment runs also contain `run_summary.json`. Search runs
additionally contain `generation_log.csv`, `final_candidate.json`, and the
text/JSON depth and/or quantization configurations. Dense and uniform runs also
receive an explicit `final_candidate.json`; uniform lists all 224 modules at
level 3. Database provenance and the 1,120-file reconstruction inventory
(shape, dtype, number of elements, contiguity, file size, and SHA-256), plus its
total byte payload, are stored at
`<quant-db>/quant_database_manifest.json`. Disk mode also retains
`<activation-cache>/activation_cache_manifest.json` plus the 7,512 scratch
records for explicit postmortem inspection; they are not a reusable checkpoint.
Cache counters are only block-boundary durable and may lag the files after an
abrupt failure. Launcher lifecycle JSON and periodic resource samples are
atomically/fsync persisted, while the child log is line-flushed and fsynced on
normal return, so its final tail is not guaranteed after container loss.

Aggregation produces:

- `comparison.csv` and `comparison.md`;
- `paired_seed_differences.csv` with joint-minus-quant-only WikiText-2 PPL, C4 PPL, KL, and runtime;
- `aggregation.json` with source paths and machine-readable rows.

The aggregator rejects duplicate method/seed summaries, differing compressed targets, and any nonzero realized-target difference.

## 11. Remaining caveats

1. This is now apples-to-apples for the explicitly defined theoretical packed-parameter budget and for search candidate evaluations/tokens. It is not yet an empirical comparison because E0--E3 have not been run.
2. The repository does not export a packed 2/3/4/5/6-bit checkpoint. Database files and evaluated module weights are floating-point reconstructions. Therefore exact equality has been proven for the estimator, not by comparing serialized checkpoint byte sizes. A production exporter would need its padding/header/profile format audited against this formula.
3. Depth pruning is implemented by forward bypass, not physical deletion. The cost assumes those weights and metadata are absent from an exported model; current runtime memory and `.pth` files do not shrink accordingly.
4. The cost includes FP16 scale and zero tensors but not architecture masks, module names, per-module bitwidth descriptors, or container overhead. Both methods need a bitwidth profile; the joint method additionally needs a small depth mask. These control-plane bytes are negligible relative to 3.37 GB but prevent a claim of literal byte-for-byte artifact equality.
5. The new joint experiment fixes the depth amount at 25% (8 attention plus 8 MLP submodules) and searches their locations. It does not evolve the number of removals. This choice should be stated in the thesis and ideally followed by a depth-ratio sensitivity analysis.
6. The joint operator devotes proposals to two component types and uses one quantization exchange, whereas original quant-only EvoPress spends every proposal on a biased 1--3 quantization exchange. Candidate evaluations/tokens are equal, but the mutation kernels necessarily differ.
7. FineWeb-Edu and the model ID are not pinned to immutable Hugging Face revisions in upstream EvoPress or this profile. The new database manifest records the resolved model commit when available, but exact future data replay would benefit from a pinned FineWeb snapshot/cache artifact.
8. WikiText-2 and C4 loaders ignore the nominal `eval_tokens` argument and evaluate the complete number of 8,192-token chunks constructed by the repository. Actual loaded token counts are recorded in search summaries. This is common to both methods but should not be described as exactly 524,288 evaluation tokens.
9. The current DataLab allocation has one A40 rather than the upstream eight-process database-generation setup. Stage 1 therefore uses the explicit `--torchrun-processes 1` deviation while emulating the configured eight-way calibration prefix. This preserves the shared examples and method-to-method fairness but is not bit-for-bit identical to an eight-GPU reduction order. The deviation must remain reported when comparing against the paper. FlashAttention 2 passed its kernel preflight; the new local spill path still requires a fresh DataLab capacity/write/inode check before launch.
10. The paper result is a reference, not an acceptance threshold. Tokenizer/library/model/data revisions and hardware kernels can shift perplexity. The uniform 3-bit E1 result is the first required sanity check before interpreting E2/E3.
11. Disk streaming changes only execution storage, not calibration examples or
GPTQ math, and the toy FastOBQ path is bit-exact. The real-Mistral one-record
smoke utility has passed CPU-side helper tests but has not yet been executed on
DataLab; that smoke and a successful 224-module postflight are still required
before claiming that Stage 1 itself is complete.
12. The scratch path sees approximately 6.4 TiB of cumulative I/O. Runtime can
therefore be materially longer than an unconstrained in-memory eight-GPU run;
database generation runtime is infrastructure provenance, not matched E2/E3
search compute. Both comparison methods reuse the exact same completed database.
13. The activation-cache manifest is checkpointed after collection and after
each full block, not after every record. Its counters may therefore be stale
after an abrupt mid-collection or mid-block failure. This is diagnostic only:
the cache is never resumable, and the validated quantization-database manifest
plus launcher `completed` state are the sole success criteria.
14. Reconstruction and calibration hashes are strong artifact-integrity and
provenance checks, but the expected values are generated by the same run rather
than compared with an independently published EvoPress checksum. They do not
establish agreement with the paper by themselves.

Required full GPU workload: eight experimental runs (one dense, one uniform, three quant-only, three joint) plus one shared, expensive GPTQ database-generation job: nine GPU jobs total. Upstream uses eight database-generation processes; the currently authorized Stage 1 invocation uses one physical process with the configured eight-way calibration prefix. Running the optional smaller three-seed search comparison adds six search jobs but reuses the same database.
