# EvoPress Thesis Implementation and Experiment Roadmap

## Purpose

This roadmap defines the next development and experiment steps for extending EvoPress toward joint compression search. The immediate priority is **not** to implement new search algorithms yet. The first priority is to make the current search framework produce complete, thesis-quality outputs:

- perplexity
- calibration KL divergence
- active parameters
- average bitwidth
- estimated model size
- effective compression ratio
- depth-pruning statistics
- quantization statistics
- runtime
- peak memory
- full candidate/configuration export

After this instrumentation is implemented and validated, stronger searches can be run and compared. Only after that should algorithmic extensions such as joint-aware mutation or adaptive mutation rate be implemented.

---

## Overall Work Plan

The work should proceed in four main phases:

1. **Instrumentation and reporting**
2. **Validation on small/debug configurations**
3. **Thorough independent, sequential, and joint search experiments**
4. **Implementation extension / new contribution**

The reason for this order is simple: before running expensive experiments, the search code must export all measurements needed for the thesis. Otherwise, expensive runs may produce only partial results and need to be repeated.

---

## Repository-Specific Clarifications

Use these definitions consistently during implementation:

- Run artifacts belong under `outputs/experiments/<run_id>/`, matching the
  existing launchers and `results/experiment_log.csv`.
- `active_parameters` is a theoretical inference count. Depth-pruned modules
  are bypassed during the forward pass but their tensors remain allocated in
  the current in-memory model.
- Bitwidth averages are parameter-weighted, not simple averages over module
  counts.
- `estimated_compression_ratio` is
  `dense_weight_memory_mb / estimated_weight_memory_mb`.
- The estimated model size is theoretical. Current GPTQ database files contain
  reconstructed floating-point tensors and are not the deployable packed model
  size.
- `generation_log.csv` is the new native structured search log.
  `generation_metrics.csv` remains supported as the existing stdout-derived
  compatibility artifact.
- Peak GPU allocation should come from PyTorch peak-memory counters. Peak CPU
  RSS should use an operating-system peak metric or the existing sampled
  cgroup monitor; a single `psutil.Process(...).memory_info().rss` reading is
  current RSS and must not be labeled as a peak.
- A generation's search fitness is only a fixed calibration KL value when the
  complete fixed calibration set was used. Mini-batch selection fitness must
  remain labeled `best_search_fitness`.
- The first implementation milestone may write unavailable expensive metrics
  as `null`. Final fixed-set KL and optional C4/FineWeb evaluation are separate
  validation steps and should not block the initial reporting implementation.

---

# Phase 1 — Instrumentation and Reporting

## Goal

Extend the EvoPress search scripts so that every search run produces a complete, structured output that can be directly used in the thesis.

The output should answer:

- What compression configuration was found?
- How many modules were dropped?
- Which modules were quantized to which bitwidth?
- What is the estimated compressed model size?
- What is the active parameter count?
- What is the average bitwidth?
- What is the resulting WikiText2 perplexity?
- What is the calibration KL divergence?
- How long did the run take?
- How much GPU memory was used?
- What were the generation-by-generation search dynamics?

---

## 1.1 Identify Current Output Files

Codex should first inspect the current repository and identify:

- which scripts run depth search
- which scripts run quantization search
- which scripts run joint depth + quantization search
- where the best candidate is stored
- where logs are written
- where perplexity evaluation is performed
- where KL fitness is computed
- where candidate masks / bitwidth profiles are saved

Expected files to inspect may include, depending on the current repo state:

```text
evo_drop_search.py
evo_quant_search.py
evo_prune_search.py
search.py
evolution.py
datautils.py
eval.py
modelutils.py
quant.py
prune.py
run_drop_search.sh
run_quant_search.sh
run_joint_search.sh
```

If file names differ, Codex should locate equivalent files by searching for:

```text
KL
kl_divergence
perplexity
ppl
offspring
mutation
candidate
mask
bitwidth
bits
sparsity
drop
```

---

## 1.2 Define a Unified Run Summary Format

Create a JSON summary file for every run.

Suggested output path:

```text
outputs/experiments/<run_id>/run_summary.json
```

Suggested schema:

```json
{
  "run_name": "string",
  "timestamp_start": "string",
  "timestamp_end": "string",
  "git_commit": "string",
  "model_name": "string",
  "dataset_calibration": "string",
  "dataset_eval": ["wikitext2", "c4"],
  "search_type": "depth_only | quant_only | sequential_depth_then_quant | sequential_quant_then_depth | joint_depth_quant",
  "search_config": {
    "generations": 0,
    "offspring": 0,
    "initial_candidates": 0,
    "selection_tokens": [],
    "selection_survivors": [],
    "fitness_fn": "kl | ppl",
    "sequence_length": 0,
    "calibration_tokens": 0,
    "seed": 0
  },
  "compression_config": {
    "target_depth_sparsity": 0.0,
    "target_average_bitwidth": 0.0,
    "bits_available": [],
    "group_size": 0
  },
  "final_metrics": {
    "calibration_kl": null,
    "wikitext2_ppl": null,
    "c4_ppl": null,
    "fineweb_ppl": null,
    "active_parameters": null,
    "total_parameters_dense": null,
    "active_parameter_ratio": null,
    "average_bitwidth_active": null,
    "average_bitwidth_total": null,
    "estimated_weight_memory_mb": null,
    "estimated_compression_ratio": null,
    "runtime_seconds": null,
    "peak_gpu_memory_mb": null,
    "peak_cpu_memory_mb": null
  },
  "depth_statistics": {
    "num_layers": null,
    "num_attention_modules": null,
    "num_mlp_modules": null,
    "dropped_attention_count": null,
    "dropped_mlp_count": null,
    "dropped_total_count": null,
    "dropped_modules": []
  },
  "quantization_statistics": {
    "quantized_module_count": null,
    "bitwidth_histogram": {},
    "average_bitwidth_by_projection_type": {},
    "bitwidth_by_module": {}
  },
  "artifacts": {
    "candidate_path": "string",
    "generation_log_path": "string",
    "config_path": "string",
    "stdout_log_path": "string"
  }
}
```

Important: the schema does not need to be perfect immediately, but all fields that can be computed should be filled. Unknown or unavailable fields should be set to `null`, not omitted.

---

## 1.3 Add Per-Generation Logging

Create a CSV or JSONL file that logs every generation.

Suggested path:

```text
outputs/experiments/<run_id>/generation_log.csv
```

Suggested columns:

```text
generation
best_fitness
best_calibration_kl
best_train_ppl
eval_tokens_used
num_offspring
num_survivors_stage_1
num_survivors_stage_2
num_survivors_stage_3
active_parameters
average_bitwidth_active
estimated_weight_memory_mb
dropped_attention_count
dropped_mlp_count
mutation_summary
accepted_parent_replacement
runtime_seconds_cumulative
peak_gpu_memory_mb
```

If the search does not currently expose all values during generation, add helper functions gradually. Do not block the whole implementation because one metric is initially missing.

---

## 1.4 Add Candidate Export

Every run should export the final candidate in a human-readable format.

Suggested path:

```text
outputs/<run_name>/final_candidate.json
```

For depth search:

```json
{
  "dropped_modules": [
    "model.layers.10.self_attn",
    "model.layers.10.mlp"
  ],
  "kept_modules": [],
  "attention_mask": [],
  "mlp_mask": []
}
```

For quantization:

```json
{
  "bitwidth_by_module": {
    "model.layers.0.self_attn.q_proj": 3,
    "model.layers.0.self_attn.k_proj": 4
  }
}
```

For joint search:

```json
{
  "dropped_modules": [],
  "bitwidth_by_module": {},
  "candidate_vector_raw": []
}
```

The raw candidate vector should always be saved, even if a nicer representation is also available.

---

# Phase 2 — Metric Helper Functions

## Goal

Implement reusable metric functions that work for depth-only, quant-only, and joint candidates.

---

## 2.1 Active Parameter Count

Add a helper function:

```python
def compute_active_parameters(model, candidate, module_metadata) -> dict:
    ...
```

It should return:

```python
{
    "total_parameters_dense": int,
    "active_parameters": int,
    "dropped_parameters": int,
    "active_parameter_ratio": float
}
```

Rules:

- Dense model parameters are counted once.
- Dropped modules contribute zero active parameters.
- Non-dropped quantized modules still count as active parameters.
- Embedding and LM head parameters should be handled consistently.
- If only transformer projection layers are searched, report both:
  - active searched parameters
  - active total model parameters

Suggested additional fields:

```json
{
  "searched_parameters_dense": 0,
  "searched_parameters_active": 0,
  "nonsearched_parameters": 0
}
```

This distinction is important because quantization may only apply to selected projection layers.

---

## 2.2 Average Bitwidth

Add a helper function:

```python
def compute_average_bitwidth(candidate, module_metadata, include_nonsearched=True) -> dict:
    ...
```

It should return:

```python
{
    "average_bitwidth_active": float,
    "average_bitwidth_searched": float,
    "average_bitwidth_total": float,
    "bitwidth_histogram": {
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "16": 0
    }
}
```

Definitions:

- `average_bitwidth_searched`: average over modules included in the quantization search.
- `average_bitwidth_active`: average over active, non-dropped searched modules.
- `average_bitwidth_total`: average over all model parameters, treating non-quantized parameters as 16-bit unless otherwise specified.

Dropped modules should not contribute to `average_bitwidth_active`, because they are not active. For total effective model size, dropped modules contribute zero memory.

---

## 2.3 Estimated Model Size

Add a helper function:

```python
def estimate_model_size(candidate, module_metadata, dense_dtype_bits=16) -> dict:
    ...
```

It should return:

```python
{
    "estimated_weight_memory_mb": float,
    "dense_weight_memory_mb": float,
    "estimated_compression_ratio": float,
    "searched_weight_memory_mb": float,
    "nonsearched_weight_memory_mb": float
}
```

Rules:

- Dropped modules contribute zero.
- Quantized searched modules contribute `num_params * bitwidth`.
- Non-searched modules contribute `num_params * dense_dtype_bits`.
- Optional: include groupwise quantization scale overhead if metadata is available.
- If the current database stores FP16 reconstructed tensors, still report the **theoretical compressed size**, not only the actual database size.

Add a note in the JSON if model size is theoretical:

```json
"model_size_note": "The estimated model size is theoretical and based on assigned bitwidths. It does not represent the on-disk size of the current FP16 reconstruction database."
```

This is important because the current sparse/quant database may store reconstructed FP16 tensors.

---

## 2.4 Depth Statistics

Add a helper function:

```python
def compute_depth_statistics(candidate, module_metadata) -> dict:
    ...
```

It should return:

```python
{
    "num_layers": int,
    "num_attention_modules": int,
    "num_mlp_modules": int,
    "dropped_attention_count": int,
    "dropped_mlp_count": int,
    "dropped_total_count": int,
    "dropped_attention_layers": [],
    "dropped_mlp_layers": [],
    "dropped_modules": []
}
```

This is needed for explaining the structural pruning profile.

---

## 2.5 Runtime and Memory Tracking

Add timing:

```python
import time
start = time.time()
...
runtime_seconds = time.time() - start
```

Add GPU memory tracking:

```python
import torch

torch.cuda.reset_peak_memory_stats()
...
peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
peak_gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
```

For CPU memory, use `psutil` if available:

```python
import resource

peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
```

Normalize `ru_maxrss` for the host operating system. When the launcher already
produces `memory_samples.csv`, the sampled cgroup maximum is the preferred
container-level value. If neither source is available, set the field to
`null`.

---

# Phase 3 — Evaluation Metrics

## Goal

Ensure each final candidate can be evaluated consistently.

---

## 3.1 WikiText2 PPL

Add or standardize a command-line option:

```bash
--eval-wikitext2
```

The run should write:

```json
"wikitext2_ppl": 0.0
```

into `run_summary.json`.

---

## 3.2 C4 PPL

Add or standardize:

```bash
--eval-c4
```

C4 is more expensive, so it can be optional for debug runs. For final runs, it should be enabled.

---

## 3.3 Calibration KL

Ensure that the best final candidate has a final KL evaluation on a fixed calibration set, not only the noisy mini-batch value used during selection.

Add:

```bash
--final-kl-tokens 65536
```

or similar.

The final summary should distinguish:

```json
"best_search_kl": 0.0,
"final_calibration_kl": 0.0
```

Reason: during multi-step selection, KL may be evaluated on randomly sampled tokens. The final KL should be computed in a reproducible way.

---

# Phase 4 — Validation Before Expensive Runs

## Goal

Verify that the instrumentation works before running expensive experiments.

Use TinyLlama or another small model first, then optionally a small Mistral debug run.

---

## 4.1 Debug Run Matrix

Run the following small experiments:

| Run | Model | Search Type | Purpose |
|---|---|---|---|
| D1 | TinyLlama | depth-only | Validate dropped module statistics |
| D2 | TinyLlama | quant-only | Validate bitwidth statistics |
| D3 | TinyLlama | joint depth+quant | Validate combined metrics |
| D4 | Mistral-7B | depth-only tiny config | Validate compatibility |
| D5 | Mistral-7B | joint tiny config | Validate memory/runtime/logging |

Suggested tiny config:

```bash
GENERATIONS=2
OFFSPRING=2
INITIALLY_GENERATED=2
SEQUENCE_LENGTH=512
CALIB_TOKENS=1024
TOKENS_PER_SELECTION="128 512"
```

Expected validation checks:

- run finishes
- `run_summary.json` exists
- `generation_log.csv` exists
- `final_candidate.json` exists
- active parameters are less than or equal to dense parameters
- dropped modules have zero contribution to estimated model size
- average bitwidth is within expected range
- runtime is non-null
- peak GPU memory is non-null
- PPL is written if evaluation is enabled
- final KL is written if enabled

---

## 4.2 Add a Validation Script

Create:

```text
scripts/validate_run_outputs.py
```

Usage:

```bash
python scripts/validate_run_outputs.py outputs/experiments/<run_id>
```

It should check:

- required files exist
- JSON is valid
- required keys exist
- numeric metrics are not NaN
- active parameters <= dense parameters
- estimated compression ratio > 1 for compressed candidates
- bitwidth histogram sums are consistent
- dropped module counts match final candidate

This script is important because final experiments may be expensive, and failed logging should be caught immediately.

---

# Phase 5 — Thorough Experiment Plan

## Goal

Run final-quality experiments after instrumentation is validated.

---

## 5.1 Experiment Types

Run the following comparison groups:

1. Dense baseline
2. Depth-only EvoPress
3. Quant-only EvoPress
4. Sequential depth then quant
5. Sequential quant then depth
6. Joint depth + quant search

The most important comparison is:

```text
joint depth+quant vs independent/sequential compression under matched budgets
```

---

## 5.2 Recommended Primary Model

Use:

```text
mistralai/Mistral-7B-v0.3
```

Reason:

- It is the model already used in the current work.
- It is included in the EvoPress paper.
- Your proof-of-concept already worked on it.
- It is thesis-relevant.

If full Mistral runs are blocked by CPU RAM or database generation issues, use TinyLlama or Llama-2/Llama-3 smaller feasible variants as secondary experiments, but keep Mistral as the target model whenever possible.

---

## 5.3 Recommended Final Metrics

Each final run should report:

```text
WikiText2 PPL
C4 PPL
final calibration KL
active parameters
active parameter ratio
average bitwidth over searched modules
average bitwidth over active modules
estimated theoretical model size
estimated compression ratio
number of dropped attention modules
number of dropped MLP modules
runtime
peak GPU memory
```

Optional but useful:

```text
Fineweb-Edu PPL
LM Eval Harness zero-shot average
generation where best candidate was found
total number of candidate evaluations
```

---

## 5.4 Suggested Search Configurations

### Medium Search

Use this first for final-style results.

```bash
GENERATIONS=20
OFFSPRING=16
INITIALLY_GENERATED=32
SEQUENCE_LENGTH=1024
CALIB_TOKENS=8192
TOKENS_PER_SELECTION="512 2048 8192"
SURVIVORS_PER_SELECTION="8 2 1"
FINAL_KL_TOKENS=32768
```

### Strong Search

Use this for final thesis results if resources allow.

```bash
GENERATIONS=50
OFFSPRING=32
INITIALLY_GENERATED=64
SEQUENCE_LENGTH=2048
CALIB_TOKENS=16384
TOKENS_PER_SELECTION="1024 4096 16384"
SURVIVORS_PER_SELECTION="16 4 1"
FINAL_KL_TOKENS=65536
```

### Very Strong Search

Use only if stable and resources allow.

```bash
GENERATIONS=100
OFFSPRING=64
INITIALLY_GENERATED=64
SEQUENCE_LENGTH=2048
CALIB_TOKENS=32768
TOKENS_PER_SELECTION="2048 8192 32768"
SURVIVORS_PER_SELECTION="16 4 1"
FINAL_KL_TOKENS=131072
```

---

## 5.5 Compression Targets

Suggested primary target:

```text
depth sparsity: 25% or 37.5%
average bitwidth: 3 bit
```

Suggested target matrix:

| Target ID | Depth Sparsity | Average Bitwidth | Purpose |
|---|---:|---:|---|
| T1 | 25.0% | 3.0 | moderate joint compression |
| T2 | 37.5% | 3.0 | stronger joint compression |
| T3 | 25.0% | 2.5 | stronger quant pressure |
| T4 | 37.5% | 2.5 | extreme combined compression |

Start with T1 and T2. Only run T3/T4 after the pipeline is stable.

---

## 5.6 Seeds

For final thesis results:

```text
minimum: 1 seed
good: 3 seeds
strong: 5 seeds
```

Use:

```text
seed 0
seed 1
seed 2
```

for the first multi-seed comparison.

Do not run many seeds until one complete run pipeline has been validated.

---

# Phase 6 — Results Tables to Produce

## 6.1 Main Comparison Table

Create a table like this:

| Method | Wiki2 PPL ↓ | C4 PPL ↓ | KL ↓ | Active Params ↓ | Avg Bits ↓ | Est. Size ↓ | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | | | | | | | |
| Depth-only | | | | | | | |
| Quant-only | | | | | | | |
| Depth→Quant | | | | | | | |
| Quant→Depth | | | | | | | |
| Joint | | | | | | | |

---

## 6.2 Compression Profile Table

| Method | Dropped Attn | Dropped MLP | Avg q | Avg k | Avg v | Avg o | Avg gate | Avg up | Avg down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

This table helps explain what the search actually learned.

---

## 6.3 Search Dynamics Table

| Method | Best Gen | Initial KL | Final KL | Initial PPL | Final PPL | Runtime |
|---|---:|---:|---:|---:|---:|---:|

---

# Phase 7 — After Strong Baselines: Implementation Contribution

Only start this phase after the instrumentation and strong baseline experiments work.

---

## 7.1 Candidate Extension A: Joint-Aware Mutation

Implement a mutation operator that explicitly trades depth and quantization budget.

Current likely behavior:

```text
depth mutation changes dropped modules
quant mutation changes bitwidths
```

New behavior:

```text
one mutation may drop a module and use the saved budget to increase bitwidth elsewhere
or restore a module and compensate by lowering bitwidth elsewhere
```

This is a stronger thesis contribution because the search space becomes genuinely joint.

Possible mutation types:

1. Quantization-only switch
2. Depth-only switch
3. Drop-and-upgrade mutation
4. Restore-and-downgrade mutation

Suggested interface:

```python
def joint_aware_mutation(candidate, module_metadata, budget, mutation_config):
    ...
```

The mutation must preserve the global compression budget.

---

## 7.2 Candidate Extension B: Adaptive Mutation Rate

Implement mutation scheduling:

```text
early generations: larger mutations
later generations: smaller mutations
stagnation: temporarily increase mutation size
```

Suggested strategy:

```python
if no_improvement_for >= patience:
    mutation_strength += 1
else:
    mutation_strength = max(1, mutation_strength - decay)
```

This is simpler than joint-aware mutation and easier to ablate.

---

## 7.3 Candidate Extension C: Adaptive Offspring Count

Increase offspring when search stagnates, decrease when search improves.

Suggested logic:

```python
if improvement_found:
    offspring = max(min_offspring, offspring // 2)
else:
    offspring = min(max_offspring, offspring * 2)
```

This connects well to EvoPress because the paper discusses the role of offspring count in convergence.

---

## 7.4 Recommended Extension Order

Recommended order:

1. Joint-aware mutation
2. Adaptive mutation rate
3. Adaptive offspring count

If time becomes limited, implement only:

```text
joint-aware mutation + old-vs-new ablation
```

---

# Phase 8 — Extension Ablation Experiments

After implementing one extension, compare:

| Method | Description |
|---|---|
| Joint baseline | current joint depth+quant search |
| Joint + new mutation | joint-aware mutation |
| Joint + adaptive mutation | if implemented |
| Joint + adaptive offspring | if implemented |

Use the same:

- model
- seed
- compression target
- generations
- offspring
- token schedule
- evaluation datasets

This is critical. The extension must be compared fairly.

---

# Phase 9 — Meeting Preparation Checklist

Before the next supervisor meeting, prepare:

## 9.1 Technical Explanation

Be ready to explain:

- how a candidate is represented
- how mutation works
- how compression budget is preserved
- how depth pruning and quantization interact
- how the model is materialized from a candidate
- how KL fitness is computed
- how multi-step selection works
- what exactly your implementation changed

## 9.2 Results

Bring:

- one main comparison table
- one compression profile table
- one generation curve if available
- one example final candidate
- one paragraph explaining limitations

## 9.3 Implementation Contribution

Even if the new extension is not fully evaluated yet, show:

- where it is implemented
- why it is different from the original EvoPress
- what hypothesis it tests
- how you will ablate it

---

# Immediate Next Codex Task

Start with this task:

```text
Inspect the EvoPress repository and identify where search candidates, mutation, fitness evaluation, and result saving are implemented. Then add a unified run-summary export system that writes run_summary.json, generation_log.csv, and final_candidate.json for depth-only, quant-only, and joint depth+quant searches. The summary must include PPL, calibration KL, active parameters, average bitwidth, estimated model size, dropped module counts, bitwidth histogram, runtime, and peak GPU memory where available. Unknown metrics should be written as null, not omitted.
```

---

# First Implementation Milestone

The first milestone is complete when this works:

```bash
python scripts/validate_run_outputs.py outputs/<debug_run_name>
```

and confirms:

```text
run_summary.json exists
generation_log.csv exists
final_candidate.json exists
required metrics are present
numeric fields are valid
candidate export is consistent
```

Only after this milestone should expensive searches be started.
