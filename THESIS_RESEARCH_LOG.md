# Master Thesis Research Log

Last updated: 2026-09-07

Experimental status: FROZEN pending supervisor feedback

Repository: EvoPress thesis fork, `agent/evopress-crossover-v3`.
Source revision inspected: `c5eddec55a7eec1cd3dbfa4e1314f857f1468dc5`.
Author/researcher: Kerim Halilović, TU Wien master-thesis project.

## 1. Document purpose and status

This is the living research record for the master thesis on evolutionary joint
compression of large language models. It preserves the experimental rationale,
protocols, results, unsuccessful directions, implementation decisions, and evidence
boundaries needed for supervisor presentations and thesis writing. It is a working
record, not a finished thesis chapter or a claim that all historical experiments
used one common protocol.

The experimental phase is frozen as of 2026-09-07. The full-space interaction-aware
and depth-warm G20 transfer experiments are completed according to the researcher's
DataLab results supplied on that date. Earlier plans to run those experiments are
historical. No V4 crossover or further algorithmic experiment is currently planned.
New algorithmic experiments require a subsequent supervisor-driven decision.

This file was assembled by inspecting Git history, source code, configs, tests,
saved commands, JSON summaries, result CSVs, analysis reports, and presentation
sources. No search, evaluation, database preparation, or regression test was run
to create this record. Existing code, results, and documentation were not edited.

### 1.1 Evidence labels and authority

| Label | Meaning | Permitted use |
| --- | --- | --- |
| **Measured — repository** | A result is present in an inspected local artifact or result table | Cite the linked artifact and its protocol; distinguish raw precision from rounded reports |
| **Measured — supplied DataLab** | Completed DataLab runs and terminal outputs supplied by the researcher during the experimental sessions | Treat the measurement as authoritative even when its run directory is absent locally; do not claim local artifact verification |
| **Paper result — supplied/audit** | A paper comparison value was supplied by the researcher or recorded in the repository reproduction audit | Attribute it to the paper; do not relabel it as a measurement made by this project |
| **Implementation fact** | Verified from code, configuration, tests, or Git objects | Describes implemented behavior, not necessarily every historical run |
| **Derived quantity** | Calculated from explicitly identified measurements or architecture/storage counts | Give the inputs and units; retain the uncertainty/rounding of those inputs |
| **Interpretation** | An explanation supported by the observed pattern | State its scope and plausible confounders |
| **Hypothesis** | A proposed mechanism or expected outcome not established by these experiments | Preserve it as a question, including when later evidence weakens it |

Results labeled **Measured — supplied DataLab** were transcribed from completed
DataLab runs and terminal outputs supplied by the researcher during the experimental
sessions. Exact results, hashes, and backup facts are preserved here as a durable
research record.

The researcher transferred the three archives listed in §17.2 to persistent DataLab
storage under `/share/kerim.halilovic/evopress/backups/` and verified each remote
SHA256 digest after transfer on 2026-09-07. Codex/local repository inspection did
not independently reopen the archive bytes. `/share` is described as persistent
storage, not provider-backed-up storage.

At inspection, `results/experiment_log.csv` contained 205 rows: 187 marked completed
and 18 failed, dated 2026-06-01 through 2026-07-03. There were 91 local
`results/runs/*/run_summary.json` files. These are inventory counts, not counts of
independent thesis experiments: the CSV includes retries, debug jobs, and replays,
and it does not cover all later experiments.

### 1.2 Data consistency notes

**Data consistency note — historical status is not freeze status.**
[The reproduction audit](artifacts/evopress_apples_to_apples_audit.md) opens with a
2026-08-18 status saying no E0–E3 run had launched. The supplied freeze record says
the reproduction is completed. These describe different dates; retain the audit
as the implementation/rationale record and use the supplied completed results for
the freeze status. Similarly, [SEQUENTIAL_EXPERIMENT_STATUS.md](SEQUENTIAL_EXPERIMENT_STATUS.md)
predates the July 31 quant-warm plus interaction-aware results, and
[CROSSOVER_IMPLEMENTATION.md](CROSSOVER_IMPLEMENTATION.md) describes only the first
component operator. Its “only component crossover” limitation is superseded by V2/V3.
[The small-model feasibility report](results/small_model_feasibility_summary.md)
also predates the subsequently logged TinyLlama GPTQ runs.

**Data consistency note — source-file SHA256 depends on line endings.**
The supplied depth-source artifact hash is
`29d4c3806c05003f4e8ba0d69dff22bbb79d3f00bfe58e3d17eb058ce99b4b42`.
The checked-out Windows file hashes to
`908e9f3afc638cf4509b03376ef1e73d4fed96a8832059868c3d6e250af69ab2`.
Converting only CRLF line endings to LF reproduces the supplied hash exactly.
This is a byte-representation difference, not a different depth mask. Preserve
both hashes and their conventions rather than silently substituting one.

**Data consistency note — “canonical depth-mask hash” needs a type convention.**
The supplied canonical hash
`fae5d211e01ab38b85aca37b8ced9569d09780a8a37948d9dd16fdab7c911bda`
is reproduced by compact, sorted JSON with keys `attn`/`mlp` and **integer 0/1**
arrays. The application's normalized **Boolean** arrays produce
`454b86987800d97eba43ad3d810527ff7143b7cabdea85e154ab2d26a1831402`.
The latter matches the historical warm run's `stage1_candidate_hash` and the
hashing convention used by `src/sequential_search.py`. Both represent the exact
same masks. They must not be treated as interchangeable checkpoint hashes.

**Data consistency note — results from different evaluation protocols.**
Historical dense WikiText2 references of 5.35 and 5.96 coexist with the supplied
paper-matched E0 value of approximately 4.83. These belong to different experimental
phases and evaluation configurations. They must remain separate rows, not be
averaged or interpreted as a failed E0 reproduction. Older reports also round
the same cheap G50 control to 11.243 rather than the raw-summary mean 11.2421875.
Use the raw-summary aggregate for that experiment and label rounded replay tables.

**Data consistency note — historical “matched” is not always exact matching.**
The early independent q-projection composition reports mean active precision about
2.972, versus exactly 3.0 for active-budget joint search. Its approximately 1.400×
storage comparison was not the later integer-exact common storage experiment.
The later replay attribution machinery can repair composed profiles to the active
budget. Neither historical convention should be relabeled as the full-space
26,982,023,168-bit protocol.

The inspected local screening means agree with the corresponding rounded values
supplied for this log. Full-space results and recent population/V2/V3 measurements
without local run directories remain explicitly marked as supplied DataLab evidence.

### 1.3 How to maintain this record

When adding evidence, record the date, source revision, exact command/config,
artifact location, completion status, seeds, metric definition, aggregation rule,
and whether the update changes an earlier interpretation. Add a clearly marked
consistency note for discrepancies. Do not overwrite an unsuccessful experiment,
silently replace a baseline, or promote a hypothesis to a measured finding.

## 2. Thesis objective

The main objective is to investigate evolutionary search for combined structural
depth pruning and weight quantization, with emphasis on comparison under a common
total-model storage budget. The main model is `mistralai/Mistral-7B-v0.3`.
TinyLlama was used for feasibility and inexpensive operator screening.

The conceptual research progression is: understand and reproduce EvoPress;
establish a paper-matched joint baseline; investigate search improvements; and
test whether improvements from cheap screening transfer to the real compression
space. Calendar history differs from that conceptual order: early depth and joint
prototypes began in May/June, while the strict full-projection reproduction protocol
was implemented in August. This distinction matters when reconstructing decisions.

The project contributes a joint candidate representation and feasibility machinery,
an exact common storage protocol, controlled studies of mutation, initialization,
population, and recombination, and an empirical account of which ideas transfer.
It does not establish joint compression as universally superior to quantization
alone. The completed exact-budget comparison provides evidence to the contrary
for the tested 25% structural-pruning setting.

Early scope included depth plus unstructured sparsity and possible three-method
compression. Hardware constraints and the eventual research evidence focused the
main thesis experiment on depth plus quantization. Those earlier feasibility
experiments remain part of the research history, not the final main protocol.

Sources: [initial prototype plan](plan1-20-05.md),
[combined-method plan](combined_compression_next_week_plan.md),
[instrumentation/next-steps plan](evopress_next_steps_codex_plan.md), Git ledger in §16.

## 3. Research questions

| ID | Question and motivation | Evidence/status at freeze |
| --- | --- | --- |
| RQ1 | Can the EvoPress quantization results be reproduced closely enough to trust the experimental platform? | E0/E1/E2 approximately match the paper; supplied completed DataLab results |
| RQ2 | At equal total storage, is 25% depth pruning plus quantization competitive with quantization alone? | E3 is substantially worse in perplexity than E2 at the same target; three seeds |
| RQ3 | Does joint optimization improve on independently composing depth and quantization solutions? | Cheap q-projection G20 initially loses; longer G50 and broader-scope studies recover quality; historical budget/compute caveats apply |
| RQ4 | Can coordinated structural/quantization mutation improve search? | Interaction-aware improves cheap mean PPL; full-space G20 transfer is mixed, not robust |
| RQ5 | Does informed initialization accelerate joint search? | Depth warm is strongest in G20 screening and improves all supplied full-space early downstream checkpoints; one full-space seed |
| RQ6 | Does a persistent population or crossover improve search? | Population helps the cheap G20 mean, but its G50 benefit disappears; crossover variants give negative/operator-analysis findings |
| RQ7 | Are gains caused by the depth mask, precision allocation, or their compatibility? | Replay attribution suggests a strong depth-mask contribution; repair and post-search replay limit causal interpretation |
| RQ8 | Do PPL gains extend to downstream task accuracy and wider search spaces? | Cheap LM-eval gains are task-dependent; full-space transfer separates warm initialization from interaction-aware mutation |

**Initial hypothesis:** better joint search could exploit interactions between
structural removal and precision allocation to improve the compression-quality
tradeoff. **Updated interpretation:** those interactions are worth studying, but
fixed structural removal can be costly relative to quantization alone; much of the
benefit from tested extensions concerns initialization or early convergence.

## 4. EvoPress / baseline methodology

### 4.1 Upstream method and thesis extension

[README.md](README.md) identifies the upstream EvoPress implementation and paper
(arXiv identifier `2410.14649`; README records ICML 2025 acceptance). Upstream
contains separate depth, unstructured-pruning, and quantization searches.
`evo_joint_search.py` is the thesis extension, not an upstream joint experiment
reported in the paper. Bibliographic title/version should be finalized from the
paper before thesis submission; this log uses the repository and supplied paper
comparison values rather than claiming a new external-paper audit.

The paper-style scaffold is a mutation-based `(1+λ)` evolutionary strategy:
retain one parent, generate λ offspring, screen them with progressively more
calibration tokens, add the incumbent at the last expensive selection stage,
and retain the best candidate. It does not require crossover. Temporary survivors
within a generation are distinct from a persistent population across generations.

The implementation adds practical details: compatible level exchanges, available
reconstruction-file checks, duplicate rejection, biased local switch counts, and
dynamic weight loading. Quantization-only and joint standard mutation are not
identical operators: the joint search must also propose fixed-cardinality structural
swaps. The exact-budget experiment matches data, selection effort, and storage,
while preserving the intended operator for each search space.

### 4.2 Joint representation and mutation

```python
candidate = {
    "drop": {"attn": [bool, ...], "mlp": [bool, ...]},
    "quant": [[bitwidth, ...], ...],  # aligned with grouped module names
}
```

`True` means remove/bypass that attention or MLP contribution. Independent masks
do not imply that the sets of affected decoder layers are disjoint: attention and
MLP can both be removed at the same index. With `drop_entire_block=false`, either
mask can change independently while preserving its own number of removals.

The standard joint operator chooses a structural proposal or a quantization
proposal with probability 0.5 each. Structural proposals exchange kept/dropped
locations. Quantization proposals exchange levels between compatible genes.
Depth mutation has configured maximum strength 3; small swap counts are favored.
One quantization exchange is used on the standard joint branch. Budget repair can
make a structural proposal change quantization assignments as well; operator labels
describe proposal origin rather than a guarantee that only one component changes.

Grouping by size means equal weight-element count, not simply the same projection
name. Database directories determine the available search scope and levels.
Inactive genes remain represented in the candidate but do not contribute to its
active storage cost. Current standard exact-budget quantization mutation preserves
the existing baseline dispatch, followed by exact repair; it should not be silently
rewritten to use a different active-gene proposal distribution.

### 4.3 Fitness, evaluation, and logging

KL fitness compares the compressed model's predictions with the dense teacher.
Teacher logits are reused; candidate fitness is evaluated on the current stage's
calibration minibatch. Final calibration KL and the final selection fitness are
different measurements and need not be equal. Lower KL and PPL are better;
higher downstream task accuracy is better.

Perplexity is the exponential of token-weighted next-token cross-entropy. Sequence
length, tokenizer, dataset split, and actual loaded token count matter. “Train PPL”
is calibration-data PPL, not evidence of training/fine-tuning the compressed model.
The search selects configurations from reconstructed weights; it does not perform
gradient-based fine-tuning of every candidate.

**Generation alignment:** current downstream diagnostics run before offspring
selection at displayed generations G1, G6, G11, G16, etc. Thus downstream G1
measures the initialized parent. In contrast, current `best_search_fitness` and
`best_calibration_kl` record the selected fitness after that generation's selection;
`parent_search_fitness_before_generation` explicitly records the previous fitness.
Do not join a G1 PPL and G1 selected KL as if both measured the same search state.
Some older reports predate logging corrections; inspect the run revision and fields.

Sources: [algorithm audit](ALGORITHM_AUDIT.md), [walkthrough](docs/reverse_engineering_walkthrough.md),
[joint search](evo_joint_search.py), [metrics](src/metrics.py), [reporting](src/run_reporting.py).

### 4.4 Storage estimate versus current execution footprint

GPTQ database files contain dequantized floating-point reconstructions selected
during evaluation, not packed 2–6-bit deployed operators. Bypassed modules can
remain allocated in the evaluation model. The exact budget describes theoretical
packed parameter and metadata storage of the compressed architecture; it is not
the disk size of `.pth` files, observed peak VRAM, runtime, or deployment latency.
Packing padding, file headers, topology descriptors, and executable code are outside
this accounting convention. All methods in the exact comparison use the same rule.

## 5. Reproduction methodology

### 5.1 Paper-matched protocol

Primary configuration: [mistral7b_v03_paper_matched.json](configs/apples_to_apples/mistral7b_v03_paper_matched.json).
Workflow: [run_apples_to_apples.py](scripts/run_apples_to_apples.py).
Design rationale: [reproduction audit](artifacts/evopress_apples_to_apples_audit.md).

| Field | Paper-matched setting | Evidence |
| --- | --- | --- |
| Model | `mistralai/Mistral-7B-v0.3`, 32 decoder layers | Config / supplied DataLab |
| Quantization scope | All 224 projections: q, k, v, o, gate, up, down in every layer | Config |
| Levels | 2, 3, 4, 5, 6 | Config |
| Reconstruction dtype | FP16 | Config |
| GPTQ group size | 128 | Config |
| GPTQ details | Per-channel; asymmetric; no activation ordering; damping 0.01; block size 128; sequential calibration level 3 | Config |
| DB calibration dataset | FineWeb-Edu prefix | Config / supplied DataLab |
| DB requested tokens | 8,388,608 | Config |
| DB exact retained tokens | 8,387,504 | Supplied DataLab; manifest not present locally |
| DB retained sequences | 7,512, maximum sequence length 8,192 | Supplied DataLab |
| Configured DB workers | 8 logical shards | Config; physical worker count for completed DB must be recovered from its manifest |
| Joint/quant search calibration | FineWeb-Edu, 524,288 requested tokens, maximum sequence length 8,192 | Config |
| Search calibration samples | 529 | Supplied DataLab |
| Fitness | KL divergence to dense teacher | Config |
| Generations / offspring | 150 / 128 | Config |
| Survivors | 16, 4, 1 | Config |
| Selection-token requests | 2,048; 16,384; 131,072 | Config |
| Initial-token setting | 2,048 | Config; initial single-candidate evaluation is skipped |
| Initial candidates / population | 1 / 1 | Config |
| Crossover | Disabled, probability 0 | Config |
| Joint mutation | Standard; maximum depth strength 3; quantization step size 1 | Config |
| Grouping | `group_rule=size` | Config |
| Depth sparsity | 0.25; eight attention and eight MLP removals | Config / supplied DataLab |
| Whole-block removal | False; independent masks | Config |
| Tokenizer | Slow/default, `use_fast_tokenizer=false` | Config / upstream audit |
| Attention implementation | FlashAttention 2 | Config |
| Evaluation | WikiText2 and C4; 524,288 requested tokens, sequence length 8,192; every five generations and final evaluation | Config |
| Budget mode | `match_uniform_quantization_total` | Config |
| Target | 26,982,023,168 bits, including metadata and fixed dense parameters | Config / supplied DataLab |

The profile omits the upstream launcher's additional FineWeb-Edu downstream
diagnostic; it retains FineWeb-Edu search calibration and KL selection. This is a
documented evaluation-scope deviation, not a change to the selection objective.

FineWeb examples can be shorter than 8,192 tokens; sequence length is a cap, not
a claim that every sample has that length. The loader uses a shuffled prefix with
fixed dataset-shuffle seed. `configured_calibration_partition()` preserves the
prefix retained by the configured logical shard count even when fewer physical
workers are used. The requested-versus-retained DB difference is 1,104 tokens
**[derived]**. The actual ordered-token digest and completed database manifest
still need to be indexed from the external reproduction artifacts.

### 5.2 Exact common storage constraint

For active searched matrices `Q`, active parameters outside that scope `F`, matrix
sizes `n_i = out_i * in_i`, and chosen integer bitwidths `b_i`:

```text
C = Σ(i in Q) n_i b_i
    + 16 |F|
    + Σ(i in Q) out_i (in_i / 128) (16 scale bits + 16 zero-point bits).
```

Dropped attention/MLP weights and their GPTQ metadata cost zero. The target is
derived from a no-depth, uniformly 3-bit reference over the full searched scope.
It is not recomputed downward when a joint candidate removes modules.

| Quantity | Exact value | Status |
| --- | ---: | --- |
| Dense parameters | 7,248,023,552 | Config / architecture assertion |
| Searched projection parameters before dropping | 6,979,321,856 | Config / architecture assertion |
| Fixed dense parameters | 268,701,696 | Config / architecture assertion |
| Dense FP16 storage | 115,968,376,832 bits | Config / derived |
| Dense FP16 storage | 14,496,047,104 bytes = 13.5004959106 GiB | Derived |
| Uniform-3 weight-only target, including fixed dense parameters | 25,237,192,704 bits | Config / derived |
| Uniform-3 GPTQ metadata | 1,744,830,464 bits | Derived / tested |
| Common metadata-inclusive target | **26,982,023,168 bits** | Config / supplied exact-budget results |
| Common target in bytes | **3,372,752,896** | Derived |
| Common target in MiB / GiB | **3,216.5078125 MiB / 3.1411209106 GiB** | Derived |
| Dense/target ratio | **4.2979867043×** | Derived |
| Storage reduction | **76.7332923810%** | Derived |

With eight attention and eight MLP removals:

| Joint quantity | Exact value |
| --- | ---: |
| Structurally removed parameters | 1,744,830,464 |
| Active total parameters | 5,503,193,088 |
| Active searched parameters | 5,234,491,392 |
| Active parameter ratio | 0.7592681023341133 |
| Fixed dense storage | 4,299,227,136 bits |
| Active quantized weight codes | 21,374,173,184 bits |
| Active GPTQ metadata | 1,308,622,848 bits |
| Total | 26,982,023,168 bits |
| Active searched-weight average | 49/12 = 4.083333333333333 bits |

The removed-parameter count and the uniform-reference metadata-bit count happen
to have the same numerical value here; they have different units and meanings.

`repair_quant_state_to_budget()` reinvests removed storage into precision, using
available reconstruction levels and preserving each equal-size group's reference
cost. It does not use the legacy active-average repair convention.

| Size group | Projection types | Dense / active module counts | Reference level sum | Required joint active level sum |
| --- | --- | ---: | ---: | ---: |
| 4,194,304 weights | k, v | 64 / 48 | 192 | 196 |
| 16,777,216 weights | q, o | 64 / 48 | 192 | 196 |
| 58,720,256 weights | gate, up, down | 96 / 72 | 288 | 294 |

The repeated Mistral shapes make these counts independent of which eight layer
indices each mask removes. Exact initial feasibility is representable with the
complete levels-2–6 database. The implementation validates the realized integer
cost and fails rather than accepting an approximately feasible initializer.

### 5.3 Search-compute accounting

The final selection pool is four surviving offspring plus the persistent parent.
Therefore one full-protocol generation evaluates 128 + 16 + 5 = 149 candidates.

```text
Scheduled candidate-token evaluations per generation
  = 128 × 2,048 + 16 × 16,384 + 5 × 131,072
  = 1,179,648.
```

| Joint-stage horizon | Initial evaluations | Search candidate evaluations | Scheduled search evaluation tokens |
| --- | ---: | ---: | ---: |
| G20 | 0 | 2,980 | 23,592,960 |
| G150 | 0 | 22,350 | 176,947,200 |

These count candidate evaluations at each selection stage, not unique architectures
or distinct training tokens. They exclude database preparation, dense teacher
construction, periodic/final downstream evaluations, and external initialization
searches. They are scheduled compute proxies, not a guarantee of identical FLOPs,
wall-clock time, or actual valid next-token predictions for every candidate.

### 5.4 Engineering required to make the reproduction feasible

The original retained-activation/database path exceeded the practical envelope of
the observed 16 GiB CPU-memory cgroup. An early Stage-1 attempt ended in a container
restart before reconstructions were produced. No surviving OOM trace establishes
the precise failure mechanism; memory pressure was the leading explanation.

The implementation history adds an explicit one-GPU database option, preservation
of the configured calibration prefix, disk-backed activations, direct GPU model
loading, manifest/preflight checks, disk-backed dense teacher logits, resumable
search checkpoints, and eviction of teacher/quantization file pages after use.
The tagged reproduction revision includes the final quant-weight page-eviction fix.
These address feasibility and restartability without deliberately reducing the
paper calibration/search budget.

Sources: [calibration partition helper](src/calibration_utils.py),
[teacher cache](src/teacher_logits_cache.py), [checkpoint implementation](src/search_checkpoint.py),
[cost implementation](src/compression_budget.py), [workflow tests](tests/test_apples_to_apples_workflow.py).

## 6. Reproduction results

### 6.1 E0–E3 overview

These are **measured — supplied DataLab** unless the column explicitly says paper.
The completed full-space run directories are not present in this checkout.

| Experiment | Configuration | WikiText2 PPL | C4 PPL | Paper comparison / evidence |
| --- | --- | ---: | ---: | --- |
| E0 | Dense FP16 | ≈4.83 | ≈7.71 | Paper ≈4.82 / 7.72 |
| E1 | Uniform 3-bit full-scope GPTQ | ≈5.50 | ≈8.55 | Paper ≈5.54 / 8.57 |
| E2 | Quantization-only EvoPress, G150/O128, three seeds | ≈5.229 ± 0.005 | ≈8.411 ± 0.007 | Paper ≈5.21 / 8.42 |
| E3 | Joint exact-budget, 25% independent pruning, G150/O128, three seeds | **8.961 ± 0.115** | **12.497 ± 0.129** | Thesis joint experiment; no corresponding supplied paper E3 value |

E3 dispersion is explicitly **population standard deviation**. The supplied E2
record gives three seeds and a ± value but does not specify its dispersion
convention or individual seed values; do not silently label it sample or population
standard deviation. Exact E0/E1 values and repetition counts are also not supplied.

**Interpretation:** quantization reproduction essentially matches the paper at the
precision supplied. This establishes a credible platform for the joint comparison;
it is not a claim of bit-for-bit reproduction or statistical equivalence.

### 6.2 Joint E3 per seed

| Seed | WikiText2 PPL | C4 PPL | Supplied KL | Exact target satisfied |
| --- | ---: | ---: | ---: | --- |
| 0 | 8.9453 | 12.6406 | ≈0.4670 | Yes |
| 1 | 8.8281 | 12.3281 | ≈0.46094 | Yes |
| 2 | 9.109375 | 12.5234 | ≈0.45605 | Yes |
| Mean ± population SD | 8.961 ± 0.115 | 12.497 ± 0.129 | No new aggregate asserted | All three |

The seed values retain the researcher's supplied precision. The exact KL field
name should be recovered from archived E3 summaries before labeling those three
values as final full-calibration KL versus final selection fitness in a publication.
Every seed satisfied 26,982,023,168 bits according to the supplied record.

### 6.3 Provenance anchor

The annotated tag `apples-to-apples-reproduction` resolves locally to
`6d3f04608a74182ba5b974f26f62153553a0fc24`, “Evict quant weight pages after search loads.”
The commit is dated 2026-08-25; the annotated tag was created on 2026-09-05.
This source revision is the reproduction anchor, distinct from the later
interaction-aware and warm-start implementation revisions.

## 7. Joint compression interpretation

### 7.1 Equal storage: quantization-only versus joint

At the common 26,982,023,168-bit target, E2 quantization-only is substantially
better in perplexity than E3 joint pruning plus quantization. The approximate
mean differences are +3.732 WikiText2 and +4.086 C4 PPL for joint **[derived from
rounded supplied means]**. This result is not explained by giving quant-only a
larger modeled storage budget: the target is explicitly common.

**Interpretation:** forcing 25% structural removal imposes a substantial quality
cost in this setting, even though the surviving searched weights can use higher
precision. This does not prove that every structural sparsity, model, or deployment
objective favors quantization-only. Latency or device-specific benefits were not
established by these storage/PPL measurements.

### 7.2 Relative to depth-only FP16

The supplied paper depth-only reference at 25% independent pruning is approximately
8.66 WikiText2 / 12.04 C4 PPL **[paper result — supplied]**. The corresponding
FP16 storage, derived from the architecture's active parameter count, is
10.2504959106 GiB. Exact-budget joint storage is 3.1411209106 GiB.

| Comparison | Value | Classification |
| --- | ---: | --- |
| Depth-only FP16 / exact-budget joint storage | 3.2633242089× | Derived from architecture counts and common target |
| Joint E3 mean minus paper depth-only W2 | ≈+0.30 | Derived cross-source comparison |
| Joint E3 mean minus paper depth-only C4 | ≈+0.46 | Derived cross-source comparison |

Thus joint is about 3.26× smaller than the depth-only FP16 architecture while
adding a comparatively modest PPL penalty relative to that supplied paper reference.
This is a useful tradeoff statement alongside the negative equal-storage E2/E3
result. It is not a controlled decomposition using an identical depth mask:
the paper depth-only solution and E3 masks need not be the same.

**Interpretation, not causal proof:** the pattern suggests that much of the overall
quality loss originates in structural pruning, with a smaller additional penalty
from quantizing the surviving network. Cheap replay attribution provides related
evidence, but cannot by itself establish that decomposition for every full-space run.

## 8. Extension methodology

### 8.1 Why cheap screening was used

A full paper-matched G150 run can consume roughly 30+ hours according to the
researcher's operational context. Testing every operator directly at that scale
would be expensive. The project therefore screened ideas in a much smaller
q-projection search space, with shorter calibration/evaluation sequences and fewer
offspring, before promoting selected ideas to full-space G20 transfer tests.

This is a staged experimental design, not a claim that cheap PPL values are
comparable to full-space PPL values. The scope, database calibration, total storage
constraint, tokenizer, attention implementation, and selection budget all differ.

#### Early feasibility and composition history

**Measured — repository:** the June Mistral depth baseline used WikiText2,
8,192 calibration tokens, sequence length 2,048, G10/O8, KL, FP16/SDPA, and seed 1.
Its dense reference was 5.35 PPL. This is an early depth experiment, not E0/E3.

| Structural sparsity | EvoPress W2 PPL | Random median PPL | Late-layer PPL |
| --- | ---: | ---: | ---: |
| 12.5% | 6.75 | 21.50 | 61.16 |
| 25% | 14.70 | 2,490.00 | 541.00 |
| 37.5% | 51.69 | 4,476.00 | 1,419.00 |
| 50% | 371.75 | 15,516.00 | 4,992.00 |

Evolutionary selection substantially outperformed those simple baselines in that
protocol. Random means were dominated by unstable runs; one 50% random run had
non-finite PPL. Setup failures/retries must be distinguished from numerical collapse.
Extending the 37.5% run to G20 improved final W2 from 51.69 to 26.00. G10 seeds
1/2/3 gave 51.69, 40.19, 47.91: mean 46.60, sample SD 5.86. Mask Jaccard overlaps
were 0.371–0.412. Different T4/A40 allocations prevent a clean runtime comparison.

TinyLlama SparseGPT/FastOBC feasibility then completed on q projections: 22 module
directories, seven levels each (154 files), reported DB size 1,233 MB, runtime
1.68 min; subsequent G20/O8 sparse search reached W2 9.00 in 6.18 min. This showed
that the pipeline worked on a smaller model; it did not resolve full Mistral
unstructured-sparsity feasibility under the 16 GiB memory limit.

Later TinyLlama logs contain a dense W2 reference of 8.97 and successful GPTQ/joint
work. Selected broader all-linear results are preserved below as historical
single-seed feasibility/attribution evidence, not the final thesis protocol.

| TinyLlama all-linear condition | W2 PPL | Historical lesson |
| --- | ---: | --- |
| Uniform 3-bit | 13.62 | Broader quantization was materially harder than q-only |
| Quantization search G20 | 14.58 | Search fitness did not guarantee better final PPL |
| Independent depth + searched quantization | 23.81 | Naive composition compounded degradation |
| Active-budget joint G10 | 21.47 | Joint search recovered some quality |
| Joint mask + uniform 3-bit replay | 21.25 | The learned precision profile was not clearly the source of the gain |
| Independent mask + joint precision replay | 22.14 | Motivated separating mask and precision effects |

Sources: [depth curve](results/depth_pruning_curve.csv), [seed robustness](results/seed_robustness_table.md),
[early meeting report](results/meeting_summary.md), [experiment CSV](results/experiment_log.csv),
[small-model feasibility](results/small_model_feasibility_summary.md).

### 8.2 Standard screening baseline

| Field | Main Mistral cheap-screen setting |
| --- | --- |
| Scope | q_proj only, 32 genes |
| Available levels | Mainly 2/3/4 in the historical debug DB |
| DB provenance | WikiText2, 512 calibration tokens, sequence length 128; reduced feasibility database |
| Depth | 25%, independent attention/MLP masks, eight removals each |
| Precision convention | Active searched average exactly 3 bits within size groups |
| Search calibration | WikiText2, 8,192 tokens, sequence length 1,024 |
| Search | Usually G20 or G50, O16 |
| Initialization | 32 random candidates for the ordinary baseline; explicit depth-first warm/frozen initialization uses one |
| Initial evaluation | 512 requested tokens per initial candidate |
| Selection | Survivors 8/2/1; token requests 512/2,048/8,192 |
| Fitness / grouping | KL / size |
| Mutation | Standard unless explicitly ablated; max depth strength 3 |
| Execution | FP16, SDPA, fast tokenizer |
| Main downstream metric | WikiText2; saved typical final loaded count 333,824 tokens despite 524,288 requested |
| Seeds for main multi-seed summaries | 0, 1, 2 |

The historical G20 joint baseline reached 12.607 ± 0.675 W2, initially worse than
independent composition at about 11.947. However, independent composition used two
source optimizations and roughly 22.73 search minutes, versus about 10.15 minutes
for joint G20. This motivated the longer G50 control, whose roughly 22.58 minutes
and 11.242 ± 0.285 PPL changed the comparison. “Compute-matched” in those old run
names refers to that approximate historical comparison, not the later exact
selection-counter protocol or a universal FLOP match.

For the cheap single-parent schedule, a generation evaluates 16 + 8 + 3 = 27
candidates and requests 49,152 candidate-tokens. Ordinary G20 uses 572 evaluations
and 999,424 scheduled tokens, including 32 initial candidates. A depth-first G20
stage uses 541 evaluations and 983,552 tokens, including one initial candidate.
The external depth-source optimization is additional. At G50 the corresponding
stage-2 counts are 1,382 / 2,473,984 and 1,351 / 2,458,112.

Sources: [medium comparison](results/mistral_medium_comparison.md),
[sequential summary CSV](results/sequential_search_summary.csv),
[G50 warm summary CSV](results/depth_warmstart_g50_summary.csv).

### 8.3 Interaction-aware mutation

The rationale was to account for the fact that changing the depth mask changes
which quantized modules are active and which layers may need precision. The
operator first proposes a count-preserving depth mutation, repairs the applicable
budget, then tries a quantization exchange involving a decoder-layer index touched
by the structural mutation. It falls back to an ordinary active exchange when the
preferred exchange is unavailable.

“Touched” means decoder-layer index; it does not establish that the exact same
attention/MLP subcomponent receives the compensating precision. The operator is
also distinct from the earlier `joint_aware_mutation`, which exchanges attention
locations and targets the restored attention layer with configured probability.

Cheap q-projection results were initially encouraging:
G20 standard 12.607 ± 0.675 versus IA 12.133 ± 0.508;
G50 standard 11.242 ± 0.285 versus IA 11.086 ± 0.174.
This motivated the exact-budget transfer implementation and one full-space G20 run.
The transfer outcome is recorded in §9.1 rather than inferred from screening.

#### Earlier mutation controls and stopping decisions

| Study | Main result | Interpretation / decision |
| --- | --- | --- |
| Mistral joint-aware G50, p=0.5 | 11.547 ± 0.617 W2 versus 11.242 ± 0.285 | Worse mean; one win in three seeds; keep standard baseline |
| TinyLlama joint-aware p=0.25 | Seed deltas +0.7265625, −0.0859375, −0.4609375 W2 | No consistent improvement; do not infer a general benefit from coupling alone |
| TinyLlama adaptive schedule | 10.987 ± 0.058 versus default 11.247 ± 0.268 | Promising aggregate, but escalation was not responsible for accepted improvements |
| TinyLlama fixed strength 1 | 10.971 ± 0.055 | Slightly better than adaptive; supports locality in that small space |
| TinyLlama coarse-to-fine 3→1 | 11.216 ± 0.228 | Worse than fixed strength 1 in all three seeds; not promoted |
| Mistral fixed strength 1 G50 | 11.211 ± 0.263 versus 11.242 ± 0.285 | Tiny mean difference; wins only seed 2; locality finding did not clearly transfer |

Adaptive TinyLlama strengths above one occurred for six generations in seed 2
and produced zero accepted parent replacements. Seeds 0/1 ended with identical
adaptive and fixed-strength candidates. That observation motivated the fixed-strength
control and prevented attributing an improvement to a mechanism that did not
produce selected improvements. Different GPU types confound the Mistral fixed-strength
runtime comparison.

Sources: [IA design](docs/interaction_aware_joint_mutation.md),
[IA results](results/interaction_aware_mistral_qproj_summary.md),
[joint-aware probability screen](results/joint_aware_probability_screen.csv),
[adaptive/locality study](results/adaptive_mutation_screen.md),
[Mistral fixed-strength control](results/mistral_fixed_mutation_ablation.md).

### 8.4 Sequential / warm-start methods

The motivation was to ask whether solving one component first supplies a better
joint starting point, and whether freezing that solution helps or restricts later
search. Sequential invocation imports a completed stage-1 artifact; it does not
execute the source optimization inside the joint run.

| Mode | Imported component | Initialization | Components mutable during stage 2 |
| --- | --- | --- | --- |
| `depth_to_quant_frozen` | Depth | Fresh quantization; one combined initializer | Quantization only |
| `depth_to_joint_warm` | Depth | Fresh quantization; one combined initializer | Both |
| `quant_to_depth_frozen` | Quantization | Feasible depth masks compatible with fixed precision profile | Depth only, through contribution-compatible swaps |
| `quant_to_joint_warm` | Quantization | Strict compatible masks by default; optional repair policy | Both after initialization |

Quantization-first strict feasibility is more constrained: frozen bit profiles
limit which depth masks and swaps preserve the active budget. The optional
`sequential_quant_initialization_policy=repair` applies to quant-to-joint warm,
not to the new exact depth-warm conversion.

Depth-to-joint warm gave the strongest tested multi-seed G20 mean, 11.526 ± 0.223.
By G50, depth warm plus standard was 11.323 ± 0.364 versus standard 11.242 ± 0.285.
The early advantage did not persist as a clear longer-run quality advantage.
This is why the final full-space question concerned early optimization, not an
expected G150 improvement.

Sources: [sequential implementation](SEQUENTIAL_SEARCH_IMPLEMENTATION.md),
[historical status](SEQUENTIAL_EXPERIMENT_STATUS.md),
[comparison](results/sequential_search_comparison.md),
[G50 study](DEPTH_WARMSTART_G50_EXPERIMENT.md).

### 8.5 Persistent population

The population-only control was necessary to distinguish the effect of retaining
multiple parents from the effect of recombination. With population four, mutation
can start from different retained candidates; with crossover disabled there is
still no recombination. Final-stage elitism includes all persistent parents.
Configured survivors `[8,2,1]` become effective survivors `[8,2,4]` for population
four, while intermediate stages remain unchanged.

Consequently equal G/O does not imply exactly equal evaluation compute between
population one and four: the last stage evaluates two surviving offspring plus
four parents instead of two plus one. With 32 initial candidates, the configured
cheap population-four G20 schedule would use 632 evaluations / 1,490,944 scheduled
tokens; G50 would use 1,532 / 3,702,784 **[derived from the implementation schedule,
not recovered counters for the externally supplied population-only runs]**.
Restore those run summaries before making strict compute-efficiency claims.

| Horizon | Population 1 W2 | Population 4 W2 | Evidence |
| --- | ---: | ---: | --- |
| G20 | 12.607 ± 0.675 | 12.201 ± 0.253 | Pop1 local; pop4 supplied DataLab |
| G50 | 11.2421875 ± 0.2852365 | 11.2942708333 ± 0.4045436923 | Sample SD; pop4 supplied DataLab |

| G50 seed | Pop1 | Pop4 | Pop4 − pop1 |
| --- | ---: | ---: | ---: |
| 0 | 11.46875 | 11.6015625 | +0.1328125 |
| 1 | 10.921875 | 10.8359375 | −0.0859375 |
| 2 | 11.3359375 | 11.4453125 | +0.109375 |

Paired mean delta is +0.0520833 PPL and relative mean change is approximately
+0.463% **[derived / supplied]**. Population four improves the G20 descriptive
mean, but has no clear G50 final-quality benefit. **Interpretation:** retaining
multiple parents primarily appears to affect early progress; it is not an
established gain in attainable final quality or quality per unit of total compute.

### 8.6 Crossover

Crossover was studied as both an optimization idea and an operator-diagnostics
question. A useful operator needs to produce feasible, novel children whose
changes are small enough to retain useful parent structure. Successful proposal
generation alone does not establish better solution quality.

| Condition | Operator | Population | Crossover probability |
| --- | --- | ---: | ---: |
| A | Standard mutation only | 1 | 0 |
| B | Standard mutation only | 4 | 0 |
| C | Component crossover plus mutation | 4 | 0.25 |
| D | Layer-bundle crossover plus mutation | 4 | 0.25 |

The later local-exchange study retained the population/crossover framework and
replaced the recombination operator. The first component study is dated August 2
and has local per-seed artifacts; V2 layer-bundle and V3 local-exchange should not
be mistaken for reruns of that identical operator. Version hashes are in §16.

#### Component crossover: duplicate-heavy

Component crossover combines one parent's whole depth component with another's
whole quantization component, repairing active precision if needed. Its three-seed
G20 W2 mean was 12.870 ± 0.927 **[local artifacts, sample SD]**.

| Seed | W2 PPL | Attempts | Accepted unique offspring | Duplicates |
| --- | ---: | ---: | ---: | ---: |
| 0 | 12.6640625 | 107 | 22 | 85 |
| 1 | 13.8828125 | 93 | 25 | 68 |
| 2 | 12.0625 | 101 | 16 | 85 |
| Total | — | 301 | 63 | 238 |

The 238/301 duplicate rate is about 79.1%. These counts are aggregated over three
seeds, not one run. There were zero infeasible proposals and 25 accepted children
requiring repair, with 25 changed quantization genes in the original component
study. **Interpretation:** as parents become similar, whole-component recombination
offers too few distinct combinations. The population-only control is essential
before crediting any favorable seed to recombination.

#### Layer-bundle V2: novelty with disruption and repair

V2 recombines bundles associated with decoder layers rather than only two whole
components. The supplied seed-0 result is W2 ≈14.1016 and KL ≈0.8623, with 87
attempts, 71 accepted offspring, 16 duplicates, 66 accepted-with-repair children,
and 107 repair changes. Duplicate pressure was reduced, but 66/71 accepted children
needed repair **[92.96%, derived]**. **Interpretation:** novelty alone did not help;
large changes and repair could disrupt useful parent combinations. The quality
result does not support promoting this operator unchanged.

#### Local-exchange V3: feasible, local, and still not a quality win

V3 uses parent A as base and parent B as donor, applying exactly one donor-guided
legal exchange. An attention or MLP exchange preserves removal counts. A quantization
exchange moves to adjacent **available database levels** in opposite directions
with equal integer cost. Available-level adjacency is not necessarily numeric ±1
for a non-contiguous database. Active-budget group constraints and, when requested,
exact total-cost constraints are checked. No repair helper is called.

Supplied seed-0 quality: W2 **13.4609375**, train PPL ≈13.9375, final KL
**0.81787109375**. This is a negative quality result despite mechanically clean
proposals and observable influence on the selected trajectory.

| V3 diagnostic | Supplied value |
| --- | ---: |
| Crossover attempts | 105 |
| Accepted unique crossover offspring | 52 |
| Duplicates | 47 |
| No legal exchange | 6 |
| Infeasible | 0 |
| Repaired proposals / repair changes | 0 / 0 |
| Accepted mutation offspring | 268 |
| Crossover share among 320 accepted offspring | 52/320 = 16.25% |
| Parent distance mean / min / max | ≈6.038 / 2 / 22 |
| Child distance from base | Exactly 2 |
| Child distance from donor, mean | ≈7 |
| Donor-distance reduction, mean | ≈1.904 |
| Accepted attention / MLP / quant exchanges | 8 / 19 / 25 |
| Generations selecting a crossover-produced parent | G6, G9, G11, G13, G15, G18, G20 |

Counts reconcile: 52 + 47 + 6 = 105 attempts; 52 + 268 = 320 accepted offspring;
8 + 19 + 25 = 52 accepted crossover subtypes **[derived checks]**.
In the implementation, parent distances include attempted parent pairs, whereas
child distances and subtype statistics count accepted unique offspring. Therefore
the reported parent mean and accepted-child mean must not be subtracted directly
to reconstruct the donor-distance-reduction mean.

“Accepted offspring” means admitted to the offspring pool after feasibility and
duplicate checks, not necessarily selected as the next parent. The listed selected
generations establish that crossover did influence this trajectory. Its failure
cannot be explained solely by the operator never being used.

**Decision:** preserve the negative results and stop crossover development at V3.
Component crossover was duplicate-heavy; layer bundles were more disruptive and
repair-dependent; local exchange fixed those mechanical problems without improving
quality. The evidence points more toward a population/early-search effect than a
benefit from the tested recombination operators. No V4 was pursued.

Sources: [component handoff](CROSSOVER_EXPERIMENT_HANDOFF.md),
[component implementation](CROSSOVER_IMPLEMENTATION.md),
[local-exchange tests](tests/test_local_exchange_crossover.py), supplied V2/V3 DataLab results.

### 8.7 Combined ablations

Combining individually promising ideas tested whether their benefits were additive.
Quantization-to-joint warm plus IA reached 11.846 ± 0.532 at cheap G20, versus
quant-warm plus standard 12.573 ± 0.439. Depth warm plus IA at G50 reached
11.172 ± 0.196, versus standard-initialized IA 11.086 ± 0.174. Population four
plus IA gave seed-0 G20 W2 **13.328125**, KL **0.80078125** **[supplied DataLab]**.
These combinations do not support an additive-benefit assumption.

The final exact-budget experiments deliberately separate IA mutation and depth-warm
initialization. Combining them is rejected by the CLI. That safeguard preserves
an interpretable transfer experiment with one intended algorithmic difference.

### 8.8 Scope expansion, generalization, and replay attribution

The June/July program also expanded q-only quantization to all 128 attention
projections and replayed final candidates across WikiText2, C4, FineWeb-Edu, and
three LM-eval tasks. These remain **cheap/legacy-protocol** measurements.

| Historical condition | Approx. storage ratio | W2 PPL | C4 PPL | FineWeb-Edu PPL | LM-eval macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense reference | 1.000× | 5.960 | 8.860 | 7.340 | 0.787 |
| Depth-only 25% | 1.317× | 11.817 | 14.760 | 12.980 | 0.647 |
| Independent depth + q quant | ≈1.400× | 11.947 | 14.910 | 13.140 | 0.647 |
| Standard joint q G50 | 1.400× | 11.243 | 14.393 | 12.460 | 0.642 |
| IA joint q G50 | 1.400× | 11.087 | 14.133 | 12.223 | 0.643 |
| Independent depth + attention quant | ≈1.547× | 13.107 | 16.090 | 14.237 | 0.596 |
| Standard joint attention G50 | 1.547× | 12.670 | 15.593 | 13.783 | 0.601 |

These values are the rounded local scope/generalization report values. The
quantization scope alone cannot explain every row difference because composition
and evaluation variants must also be respected. q-only quantization alone produced
only about 1.064× whole-model compression; attention quantization alone about 1.177×.

The LM-eval task set was ARC-Easy (`acc_norm`), PIQA (`acc_norm`), and Winogrande
(`acc`). IA versus standard q G50 improved ARC-Easy approximately 0.580→0.602,
slightly reduced PIQA 0.739→0.736, and reduced Winogrande 0.608→0.591. The macro
change 0.642→0.643 is effectively a small, task-dependent descriptive difference;
it is not a demonstrated broad downstream improvement.

Replay attribution combines depth masks from independent, standard-joint, and
IA-joint searches with independent, standard, IA, and uniform-3 precision profiles.
It performs evaluation and budget handling, not new evolutionary optimization.
For example, q-only replay means using the IA depth mask were W2 10.95 with
uniform-3 versus 11.09 with its IA precision profile. The standard joint mask gave
11.21 with uniform-3 versus 11.24 with its own profile. Full-attention replay gave
12.07 for IA depth + uniform-3 versus 12.42 for IA depth + IA precision.

**Interpretation:** the optimized depth mask often carried much of the replay
benefit, and a precision profile was not universally useful with another mask.
This helped motivate depth-informed initialization. Repair, candidate selection,
and replay evaluation differences prevent treating these matrices as a complete
causal decomposition or as full-224 exact-budget experiments.

Sources: [scope comparison](results/mistral_scope_comparison.md),
[generalization](results/mistral_generalization_eval.md), [LM-eval](results/mistral_lmeval_comparison.md),
[IA downstream results](results/interaction_aware_mistral_qproj_summary.md),
[attribution framework](results/attribution_framework_report.md),
[q attribution aggregate](results/attribution/mistral_qproj_aggregate.md),
[attention attribution aggregate](results/attribution/mistral_attention_full_aggregate.md).

## 9. Full-space transfer validation

The final transfer studies use the real full 224-projection database, levels 2–6,
25% independent attention/MLP pruning, the common metadata-inclusive target,
FineWeb-Edu stage-2 calibration, G20/O128, seed 0, population one, and crossover zero.
They test early optimization under the full protocol, not the eventual G150 optimum.
The standard comparison is the supplied seed-0 early trajectory of the reproduced
standard search. The local checkout does not contain those full-space run directories.

### 9.1 Interaction-aware

**Reason for promotion:** cheap q-projection G20/G50 mean PPL and cross-dataset
results made IA a plausible operator to test in the actual compression space.

**Implementation fact:** commit
`395f983548ed93235434ca1e6bc38fefe836d62b` extends IA to the exact total-model budget
behind `--allow_exact_budget_ablation`. After the structural proposal it performs
the same group-preserving exact repair as baseline, then the touched-layer-aware
precision exchange. Repair must precede the exchange: the exchange preserves the
repaired cost, so the main loop's later generic exact repair is a no-op rather than
erasing the coordinated exchange. Legacy active-average IA behavior remains available.

#### Final full-space IA G20 seed-0 result

All values in this table are **measured — supplied DataLab**.

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | 10.8984375 |
| C4 PPL | 13.9921875 |
| Train/calibration PPL | 11.0703125 |
| Final calibration KL | 0.56005859375 |
| Best search fitness | 0.56591796875 |
| Exact-budget validity | True |
| Realized cost | 26,982,023,168 bits |
| Runtime | ≈18,336.27 s, or ≈5.0934 h [hours derived] |

#### Matched downstream checkpoints: IA versus standard

These are displayed-generation diagnostics on the parent before that generation's
selection. Deltas are IA minus standard **[derived]**; negative PPL deltas favor IA.

| Checkpoint | Standard W2 | IA W2 | W2 delta | Standard C4 | IA C4 | C4 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 45.4375 | 45.4375 | 0 | 47.15625 | 47.15625 | 0 |
| G6 | 13.25 | 12.890625 | −0.359375 | 17.28125 | 17.796875 | +0.515625 |
| G11 | 10.484375 | 11.875 | +1.390625 | 14.46875 | 14.3515625 | −0.1171875 |
| G16 | 10.3828125 | 10.9453125 | +0.5625 | 13.9375 | 13.9921875 | +0.0546875 |

#### Matched selected search KL

These are the supplied selected-search fitness checkpoints, distinct from the
downstream diagnostic state and from final full-calibration KL.

| Generation | Standard KL | IA KL | IA − standard [derived] |
| --- | ---: | ---: | ---: |
| G1 | 1.2158203125 | 1.2119140625 | −0.00390625 |
| G5 | 0.751953125 | 0.77734375 | +0.025390625 |
| G10 | 0.5693359375 | 0.5751953125 | +0.005859375 |
| G15 | 0.55224609375 | 0.54296875 | −0.00927734375 |
| G20 | 0.568359375 | 0.56591796875 | −0.00244140625 |

**Interpretation:** the cheap-space improvement did not robustly transfer.
IA has a small G20 search-KL advantage, mixed earlier KL, and mixed downstream
behavior. It improves W2 at G6 but loses W2 at G11/G16; C4 changes do not consistently
follow the W2 changes. This supports “mixed/non-robust transfer,” not “IA universally
hurts.” One seed cannot establish a distribution-wide comparison.

**Decision:** retain IA as an implemented method and informative negative transfer
finding. Do not extend it to a new G150 or multi-seed full-space campaign at freeze.

### 9.2 Depth warm-start

**Reason for promotion:** depth-to-joint warm was the strongest cheap G20 mean,
attribution suggested that masks matter, and the disappearing G50 advantage made
early convergence the appropriate question. The source was selected before seeing
the full-space transfer outcome, according to the supplied provenance statement.
It is the same seed-0 source historically used for the cheap depth-warm run.

#### Stage-1 source, locally verified

Source directory:
[thesis_medium_depth_mistral_s0.25_g20_o16_seed0](results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/).
Key artifacts: [summary](results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/run_summary.json),
[candidate](results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/final_candidate.json),
[command](results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0/command.sh).

| Source field | Value | Evidence |
| --- | --- | --- |
| Model | Mistral-7B-v0.3 | Local summary |
| Search type | Depth-only | Local candidate |
| Seed | 0 | Local summary |
| Generations / offspring | 20 / 16 | Local summary |
| Calibration | WikiText2, 8,192 tokens, sequence length 1,024 | Local summary/command |
| Fitness | KL | Local summary |
| Initial candidates / tokens | 32 / 512 | Local summary |
| Survivors / selection tokens | 8/2/1 and 512/2,048/8,192 | Local summary |
| Depth constraint | Eight attention and eight MLP removals; whole-block false | Local candidate/summary |
| WikiText2 PPL | 11.8046875 | Local summary |
| Train PPL | 12.15625 | Local summary |
| Final calibration KL | 0.68505859375 | Local summary |
| Best search fitness | 0.685546875 | Local summary |
| Launcher runtime | 572.0 s | Local summary; supplied value agrees |
| Search-process runtime | 561.6483190208673 s | Separate local-summary field |
| Source revision | `2b4e4dc75c91a23982f05347a079927e9c32fd4c` | Local summary |
| Final W2 loaded tokens | 333,824 | Local summary |

The exact masks are preserved below. Values are integer 0/1 for portable reporting;
1 means removed. Indices are zero-based.

```json
{
  "attn": [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0],
  "mlp":  [0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0]
}
```

Removed attention indices: `3, 9, 14, 16, 20, 21, 28, 30`.
Removed MLP indices: `3, 8, 12, 13, 15, 21, 25, 27`.
Each mask has exactly eight removals. The hash conventions and exact source-file
hashes are recorded in §1.2 and §17.3.

#### Conversion into the exact-budget joint initializer

Commit `c5eddec55a7eec1cd3dbfa4e1314f857f1468dc5`,
“Add exact-budget depth warm-start ablation,” implements the following transition:

1. Import the depth-only candidate; copy and validate its masks.
2. Initialize fresh full-scope quantization genes at reference level 3.
3. Reinvest structural storage savings using the existing group-preserving exact
   repair, including scale/zero metadata and dense non-quantized parameters.
4. Verify the imported mask is unchanged and initial cost equals the target.
5. Skip the initial single-candidate evaluation, as in standard exact-budget search.
6. Run G20 of the existing standard joint mutation/selection mechanism, population
   one and crossover zero. Both depth and quantization remain mutable.

The source's quantization assignments are not imported. The mask is preserved
through initialization, not frozen for stage 2. Its WikiText2 source calibration
does not change stage-2 FineWeb-Edu calibration; it is disclosed as part of the
source-optimization provenance.

#### Final full-space depth-warm G20 seed-0 result

All values below are **measured — supplied DataLab**.

| Metric | Value |
| --- | ---: |
| WikiText2 PPL | **9.8828125** |
| C4 PPL | **13.4296875** |
| Train/calibration PPL | 10.5 |
| Final calibration KL | **0.51123046875** |
| Best search fitness | **0.48193359375** |
| Exact-budget validity | True |
| Target cost | 26,982,023,168 bits |
| Actual cost | 26,982,023,168 bits |
| Difference | **0 bits** |
| Active average bitwidth | 4.083333333333333 |
| Searched average bitwidth | 3.9188701923076925 |
| Stage-2 runtime | 16,833.7591 s, approximately 4.6760 h [hours derived] |
| Search candidate evaluations | 2,980 |
| Scheduled search evaluation tokens | 23,592,960 |
| Initial evaluations | 0 |

The active and searched averages have different denominators: inactive assignments
can remain in the stored searched genotype, while only active weights contribute
to the exact cost. Neither average is the whole-model effective bitwidth including
fixed dense parameters and metadata.

#### Matched downstream checkpoints: warm versus standard

These are the supplied matched diagnostic checkpoints; delta is warm minus standard.

| Checkpoint | Standard W2 | Warm W2 | W2 delta | Standard C4 | Warm C4 | C4 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 45.4375 | 11.578125 | −33.859375 | 47.15625 | 17.65625 | −29.5 |
| G6 | 13.25 | 9.4140625 | −3.8359375 | 17.28125 | 14.5234375 | −2.7578125 |
| G11 | 10.484375 | 9.140625 | −1.34375 | 14.46875 | 13.78125 | −0.6875 |
| G16 | 10.3828125 | 9.0703125 | −1.3125 | 13.9375 | 13.40625 | −0.53125 |

Warm initialization is better on both datasets at every supplied matched downstream
checkpoint. The very large G1 difference measures the initializer itself, before
the first joint generation's offspring selection. Later differences show that the
advantage is not confined to the initial diagnostic.

#### Matched selected search KL

The table retains the precision supplied for the warm comparison. Small last-digit
differences from the exact standard values in §9.1 are rounding, not different runs.

| Generation | Standard KL | Warm KL | Supplied delta |
| --- | ---: | ---: | ---: |
| G1 | 1.215820312 | 0.640136719 | −0.575683594 |
| G5 | 0.751953125 | 0.565429688 | −0.186523438 |
| G10 | 0.569335938 | 0.547851562 | −0.021484375 |
| G15 | 0.552246094 | 0.513671875 | −0.038574219 |
| G20 | 0.568359375 | 0.481933594 | −0.086425781 |

**Interpretation:** depth-informed initialization transfers successfully in this
single-seed full-space early-search experiment. It supplies substantially better
initial quality and maintains an advantage at the supplied matched checkpoints,
including lower selected search KL at G20.

**Endpoint/metric boundary:** the supplied record does not include a separately
identified standard final G20 W2/C4 endpoint pair. Do not compare the warm final
G20 endpoint against standard G16 and label it a matched final-G20 win. The standard
G150 endpoint is a different horizon. Warm final W2 9.8828125 is also worse than
its G16 diagnostic 9.0703125; downstream PPL need not improve monotonically when
selection optimizes noisy/minibatch KL. Final calibration KL 0.51123046875 must
not be conflated with best search fitness 0.48193359375.

#### Compute and fairness

The appropriate claim is **matched subsequent joint-search budget**. The source
depth search is separate optimization with G20/O16 and approximately 572 seconds
of launcher runtime. It is additional work even if its artifact is reused.

| Effort category | Depth source | Joint stage | Arithmetic pipeline total |
| --- | ---: | ---: | ---: |
| Runtime seconds | 572.0 | 16,833.7591 | ≈17,405.7591 |
| Candidate evaluations | 572, reconstructed from source schedule | 2,980, supplied measured counter | 3,552, reconstructed/summed |
| Scheduled candidate-token evaluations | 999,424, reconstructed | 23,592,960, supplied counter | 24,592,384, reconstructed/summed |

The numerical coincidence of 572 seconds and 572 source evaluations should not be
mistaken for a shared unit. Runtime fields cover different recorded processes,
and candidate-token totals across two protocols are bookkeeping, not equivalent
FLOPs. The approximately 9.53-minute source cost is about 3.40% of the recorded
joint-stage runtime **[derived descriptive ratio]**. It is still nonzero, additional
optimization; report it rather than calling the experiment equal total compute.

### 9.3 What transferred and what did not

| Idea | Screening signal | Full-space evidence | Freeze decision |
| --- | --- | --- | --- |
| IA mutation | Lower cheap G20/G50 mean PPL | Mixed seed-0 G20 checkpoints; small final selected-KL difference | Report non-robust transfer; stop further full runs |
| Depth-informed warm initialization | Strongest cheap G20 mean; advantage disappears by cheap G50 | Better W2/C4 at all supplied matched early checkpoints and lower G20 search KL | Report successful early transfer with single-seed and source-compute caveats |
| Persistent population | Better cheap G20 mean; no clear G50 gain | No supplied full-space population transfer | Preserve screening conclusion only |
| Crossover V1/V2/V3 | Mechanical improvements did not yield quality gains | No supplied full-space crossover transfer | Preserve negative/operator-analysis results; no V4 |
| Fixed/adaptive locality | Small-model signal; weak Mistral fixed-strength transfer | No full-space transfer result | Preserve scope-dependent finding |

The broader methodological finding is that simplified screening can overestimate
an operator's benefit. Promotion to the real search setting was necessary to
distinguish a transferable initializer from a non-robust mutation improvement.

## 10. Consolidated experiment tables

### 10.1 Cheap G20 screening

Lower W2 PPL is better. Rows with ± summarize three seeds unless stated otherwise.
Local multi-seed summaries use sample SD. The supplied population-four G20 ± value
is retained as reported; its dispersion convention should be checked in the archive.

| Method | W2 PPL | KL where supplied | Evidence / boundary |
| --- | ---: | ---: | --- |
| Standard joint | 12.607 ± 0.675 | — | Local sequential/control summaries |
| Interaction-aware | 12.133 ± 0.508 | — | Local sequential summary |
| **Depth → joint warm + standard** | **11.526 ± 0.223** | — | Local sequential summary; extra source search |
| Depth → quantization frozen | 11.932 ± 0.586 | — | Local sequential summary; extra source search |
| Quantization → depth frozen | 14.003 ± 2.748 | — | Local sequential summary; constrained depth moves |
| Quantization → joint warm + standard | 12.573 ± 0.439 | — | Local sequential summary |
| Quantization → joint warm + IA | 11.846 ± 0.532 | — | Local sequential summary |
| Population-four standard mutation | 12.201 ± 0.253 | — | Supplied DataLab; population changes final-stage effort |
| Component crossover | 12.870 ± 0.927 | — | Local three-seed component results |
| Local-exchange crossover | 13.4609375 | 0.81787109375 | Supplied DataLab; seed 0 only |
| IA + population four | 13.328125 | 0.80078125 | Supplied DataLab; seed 0 only |

Depth-informed initialization is the strongest **multi-seed G20 mean** in this
table. The table is a screening matrix, not a rank ordering under equal end-to-end
compute: source searches, initialization counts, and population costs differ.

### 10.2 Cheap G50 screening

Values are mean ± sample SD across three seeds, with rounding for presentation.

| Method | W2 PPL | Change from standard mean, approximately | Evidence |
| --- | ---: | ---: | --- |
| Standard | 11.242 ± 0.285 | — | Local |
| Interaction-aware | 11.086 ± 0.174 | −0.156 | Local |
| Depth warm + standard | 11.323 ± 0.364 | +0.081 | Local |
| Depth warm + IA | 11.172 ± 0.196 | −0.070 | Local |
| Fixed depth strength 1 | 11.211 ± 0.263 | −0.031 | Local |
| Population four | 11.294 ± 0.405 | +0.052 | Supplied DataLab |
| Earlier joint-aware p=0.5 | ≈11.547 ± 0.617 | +0.305 | Local |

The strong G20 warm/population differences largely disappear by G50. This is
consistent with an early-convergence effect. It does not establish identical
limiting search distributions or prove that G50 is the final attainable optimum.
No statistical-significance claim is made from n=3.

### 10.3 Full-space final endpoints and comparison availability

| Method / horizon | Seeds | W2 PPL | C4 PPL | Final calibration KL | Selected search fitness | Exact target |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E2 quant-only G150 | Three | ≈5.229 ± 0.005 | ≈8.411 ± 0.007 | Not supplied | Not supplied | Common target |
| E3 standard joint G150 | 0/1/2 | 8.961 ± 0.115 | 12.497 ± 0.129 | Per-seed KL supplied; precise field needs archival confirmation | Not separately supplied | All valid |
| Standard joint final G20 | 0 | Not supplied as a final endpoint | Not supplied as a final endpoint | Not supplied | 0.568359375 at G20 | Standard exact-budget protocol |
| IA joint G20 | 0 | 10.8984375 | 13.9921875 | 0.56005859375 | 0.56591796875 | Valid |
| Depth-warm standard joint G20 | 0 | 9.8828125 | 13.4296875 | 0.51123046875 | 0.48193359375 | Valid |

Every exact target in this table means 26,982,023,168 bits. G150 and G20 rows are
different horizons. The E3 ± values use population SD; E2's convention is unspecified.
The IA and warm endpoints are single-seed results with no cross-seed uncertainty
estimate. Use the matched trajectories in §9 for early standard-versus-ablation claims.

## 11. Main findings

### 11.1 Strongly supported within the stated evidence

1. **The quantization reproduction is close to the paper.** E0/E1/E2 align at the
   supplied precision, and the common-budget methodology is explicit and tested.
2. **Quantization-only is much better than 25% joint structural pruning plus
   quantization at the same modeled storage in this experiment.** E2/E3 differ
   substantially across the supplied three-seed summaries.
3. **The main budget distinction is real.** Legacy active average and common total
   storage are different constraints; exact joint search reinvests structural and
   metadata savings, reaching active precision 4.0833 rather than retaining 3.0.
4. **Operator mechanics can be diagnosed separately from quality.** Crossover
   duplicate rates, repair dependence, and selected-child provenance distinguish
   component, layer-bundle, and local-exchange behavior.

“Strongly supported” describes the clarity of the observed comparison, not a
formal significance test or a claim of generality beyond the studied setting.

### 11.2 Supported but limited

1. **Depth-informed initialization accelerates early full-space joint search in
   the supplied seed-0 transfer.** Both downstream metrics improve at all supplied
   matched checkpoints, and G20 selected KL is lower. Source compute is additional.
2. **Population and depth warm primarily appear to affect early search.** Their
   cheap G20 improvements do not yield a clear G50 final-quality benefit.
3. **IA has a useful cheap-space signal but non-robust full-space transfer.** The
   direction varies by metric/checkpoint in the full-space seed-0 experiment.
4. **Masks often explain much of the cheap replay improvement.** Precision-profile
   compatibility and repair prevent an unrestricted causal conclusion.
5. **PPL improvements need not imply downstream accuracy gains.** The three-task
   LM-eval macro is nearly tied for standard versus IA q-projection joint search.

### 11.3 Negative findings worth preserving

Component crossover produced excessive duplicates. Layer-bundle crossover reduced
duplicates but relied heavily on repair and had poor seed-0 quality. Local exchange
achieved feasible, local, repair-free children and influenced selection without
improving quality. IA plus population did not combine additively. Earlier joint-aware
mutation, adaptive escalation, and coarse-to-fine scheduling did not justify
promotion as general improvements. Fixed locality did not clearly transfer from
TinyLlama to Mistral. Exact-budget joint E3 did not outperform quantization-only E2.

These are useful boundaries on the thesis contribution, not experiments to omit
because they weaken an initial hypothesis.

## 12. Claims safe for the thesis

The following formulations are supported when accompanied by their protocol and
evidence qualifiers:

- “We reproduced the EvoPress quantization results closely and established a
  joint depth-plus-quantization comparison under an explicitly shared, metadata-inclusive
  total storage constraint.”
- “At the tested 26,982,023,168-bit budget, three-seed quantization-only results
  substantially outperform the joint configuration with 25% independent structural pruning.”
- “Joint compression is approximately 3.26× smaller than the corresponding
  depth-pruned FP16 architecture, with an approximately 0.30/0.46 W2/C4 penalty
  relative to the supplied paper depth-only reference; these are cross-source,
  derived comparisons rather than an identical-mask causal ablation.”
- “A depth-informed warm start improves all supplied matched early W2/C4 checkpoints
  and G20 selected KL in a seed-0 full-224-projection exact-budget transfer, when
  the subsequent joint-search budget is held fixed.”
- “That warm-start experiment demonstrates improved early optimization from an
  informed initializer, with additional source-search compute and without evidence
  of improved G150 final quality.”
- “A promising q-projection mutation improvement did not robustly transfer to the
  full exact-budget setting; operator validation in the target compression space
  is necessary.”
- “Crossover feasibility, novelty, and locality can be improved without obtaining
  a corresponding quality gain.”
- “Persistent populations improve the cheap G20 descriptive mean but show no clear
  G50 quality advantage in the available seeds.”

Use “observed,” “in this setting,” and the seed/horizon qualifier where they carry
real evidentiary meaning. Do not turn these sentences into universal claims.

## 13. Claims we must NOT make

| Unsupported claim | Why it is unsupported / correct boundary |
| --- | --- |
| “The improvements are statistically significant.” | Three screening seeds and single-seed transfers do not establish that claim; no such test is reported |
| “Depth warm improves G150 final quality.” | Only the full-space G20 warm transfer is supplied; cheap G50 loses its early advantage |
| “Depth warm uses equal total compute.” | It imports a separately optimized depth source; match is stage-2 search effort |
| “Warm has a measured final-G20 W2/C4 win over standard.” | The standard final-G20 downstream endpoint pair is not supplied; use the matched checkpoint table |
| “IA is universally worse.” | Full-space KL and downstream differences are mixed; cheap means often improve |
| “All promising cheap improvements transfer.” | IA and locality controls contradict that generalization |
| “Cheap and full-space PPL can be directly ranked.” | Scope, storage convention, data, context, tokenizer, and budgets differ |
| “Population four is compute-identical to population one at the same G/O.” | More persistent parents enter final-stage selection |
| “Crossover had no opportunity to matter.” | V3 crossover children were selected at seven supplied generations |
| “An accepted crossover proposal is a selected next parent.” | Acceptance into the pool and final selection are different events |
| “3-bit plus 25% pruning always means 3-bit active precision.” | Under the common exact budget the active average is 49/12 |
| “25% structural sparsity means 25% of total model parameters removed.” | Embeddings, head, norms, and module sizes change the parameter fraction; active ratio is ≈0.7593 |
| “The compressed model occupies 3.14 GiB of VRAM during current search.” | The budget is modeled packed storage; execution retains floating reconstructions and other buffers |
| “Most loss is causally proven to come from depth.” | Paper-reference and replay comparisons are not an identical-mask full-space causal decomposition |
| “Better KL/PPL guarantees better downstream accuracy.” | Existing task-level evidence is mixed |
| “The cluster provider backs up `/share`.” | Only persistence and the researcher's archive locations are supplied |
| “All current results are locally archived and hash-verified.” | Full-space and recent extension archives are reported externally, not verified here |

## 14. Methodological limitations

### 14.1 Statistical and selection limits

Main screening summaries typically have n=3; full-space extension transfers have
n=1. Standard deviations describe observed variability and are not confidence
intervals. Multiple ideas were screened, so choosing the best screen can introduce
selection optimism. The depth source was selected in advance of its full-space
result, but that does not remove all broader method-selection effects.

G20 and G50 do not establish asymptotic convergence. A disappearing G50 advantage
is consistent with earlier convergence, but could also reflect noise or different
search trajectories. Do not estimate a universal convergence speedup from four
downstream checkpoints without a specified target-quality/time definition.

### 14.2 Protocol and metric limits

Legacy q-only, attention-only, full-space, TinyLlama, and early Mistral depth runs
must remain separate experimental families. The old q-only DB used much less
calibration data than the full database. Requested tokens, retained input tokens,
next-token predictions, and scheduled candidate-token evaluations are different
quantities. The final dataset load may be smaller than a CLI request.

Search KL uses stage minibatches, while final calibration KL and PPL are separate
evaluations. Values can fluctuate across generations even with elitism because
fitness is reevaluated on different samples. Matching generation labels alone is
insufficient unless the logged state convention is also matched.

Three LM-eval tasks cannot establish broad language-model capability retention.
The full-space IA/warm study supplies PPL/KL, not a corresponding full-space
downstream task battery. This is an evidence boundary, not an instruction to run
more experiments during the freeze.

### 14.3 Compute, hardware, and storage limits

The source optimizer is extra compute for warm-start methods. Population changes
final-stage evaluation effort. Historical “compute-matched G50” comparisons were
approximately matched by recorded search time; exact-budget E2/E3 use explicit
scheduled evaluation matching. Hardware allocations varied across T4, A40, and
V100 records, so historical runtime differences are not pure algorithm effects.

The 2026-06-03 A40 snapshot documents roughly 44.42 GiB visible GPU memory and a
17,179,869,184-byte CPU-memory cgroup limit, i.e. 16 GiB. Some old reports label
binary quantities “GB”; retain the exact bytes when reconstructing resources.
The full completed reproduction's per-run device, memory, and process configuration
must be recovered from its artifacts rather than assumed from this old snapshot.

Packed storage accounting does not establish deployable low-bit kernels, latency,
throughput, KV-cache savings, energy savings, or end-to-end serving memory. Those
would require separate implementation/evaluation and are not claimed here.

### 14.4 Artifact and source limits

Local lightweight artifacts cover much of the cheap study, but not the completed
full-space runs or recent population/V2/V3 runs. The supplied archive hashes were
not recomputed against archive bytes during local inspection. E2 seed-level results,
exact E0/E1 metrics, full-space standard final-G20 endpoints, and some run IDs/manifests
are not supplied.

The stage-1 importer allows a missing summary and checks model metadata only when
available. It does not require identical stage-1/stage-2 calibration. The selected
source for this experiment does have a locally verified summary, masks, and command.

## 15. Important implementation decisions

### 15.1 Preserve historical behavior behind opt-ins

The standard exact-budget reproduction still validates without
`--allow_exact_budget_ablation`, uses `sequential_mode=none`, standard joint mutation,
population one, and crossover zero. Existing experiment configs were not repurposed
for the late ablations. The current flag explicitly permits two separate studies:
exact-budget IA mutation or exact-budget depth-warm initialization with standard
mutation. They cannot be combined under that safeguard.

Exact depth warm requires one initial candidate, population one, crossover zero,
no `active_quant_budget`, standard mutation, and no adaptive/coarse-to-fine mutation.
Existing exact requirements for size grouping, integral reference bitwidth, and
metadata group size remain. Other exact-budget sequential directions/frozen modes
remain rejected. No new mutation operator was introduced for warm initialization.

### 15.2 Use one storage definition and fail on infeasibility

The exact paths reuse `candidate_compression_cost()`,
`repair_quant_state_to_budget()`, and `validate_exact_budget()`.
Repair preserves equal-size reference group costs, including active metadata.
It does not silently reinterpret `repair_active_quant_budget()` as a total-budget
solver. Imported masks are preserved during warm initialization; failed exact
initialization does not fall back to a different source mask or looser target.

Local-exchange crossover has a separate “validate, do not repair” path so its
one-exchange locality guarantee survives the surrounding offspring loop. Other
operators retain their established repair behavior.

### 15.3 Record identity and compute honestly

Warm summaries identify `exact_budget_ablation=depth_to_joint_warm`, initial exact
cost/difference, skipped initial evaluation, source-summary path and contents,
and `stage1_compute_included_in_search_totals=false`. Existing sequential metadata
retains source candidate path/hash, initial repair details, and initial-parent hash.
Missing source information is not synthesized.

New checkpoint identity fields are conditional on this ablation: ablation label,
opt-in flag, imported Boolean-mask hash, and skipped-initial-evaluation policy.
Replacing a source mask at the same path fails identity validation on resume.
Old checkpoint identities are preserved. The checkpoint schema was not redesigned;
existing parent/population, initial state, counters, and RNG restoration remain.

`scripts/summarize_sequential_search.py::search_effort()` now prefers
`candidate_evaluations_search_total` and `evaluation_tokens_search_total` when
available, then actual initial evaluation counters, with reconstruction fallback
for older artifacts. This prevents counting an initial candidate that was created
but deliberately not evaluated. Raw historical counters and reconstructed estimates
must remain distinguishable in the research record.

### 15.4 Verification evidence and its limits

The repository contains deterministic synthetic CPU tests for the exact warm path,
IA, budgets, sequential imports, checkpoints, reporting, and crossover. They cover
the Mistral-shaped 224-module target without downloading real weights. Tests exercise
production initialization, dispatch, selection counters, source immutability,
hash-based resume rejection, and preservation of the standard paper command.

The following is **researcher-observed validation from the experimental sessions**,
not a repository-stored CI report.

For the exact-budget interaction-aware implementation, OLD JUPYTER/Linux validation
passed 66 targeted/regression tests and 28 additional regression tests, for a total
of **94 passed**. `py_compile` passed with no output.

For the exact-budget depth-warm implementation, **20 new targeted tests passed**
during the implementation phase. After commit/push, the requested targeted and
broader regression suites were run on OLD JUPYTER/Linux and all tests passed.
`py_compile` and diff checks passed. A numerical broader-suite test count is not
available in the supplied validation evidence and is not inferred here.

No tests were rerun for this document, and synthetic coverage is not a substitute
for the supplied actual DataLab transfer measurement.

Relevant test files:

- [exact warm](tests/test_exact_budget_depth_warm.py)
- [exact IA](tests/test_exact_budget_interaction_aware.py)
- [compression cost/repair](tests/test_compression_budget.py)
- [sequential search](tests/test_sequential_search.py)
- [checkpoints](tests/test_search_checkpoint.py)
- [run reporting](tests/test_run_reporting.py)
- [paper workflow](tests/test_apples_to_apples_workflow.py)
- [component crossover](tests/test_component_crossover.py)
- [local exchange](tests/test_local_exchange_crossover.py)

### 15.5 Reconstructing commands after the freeze

For a standard E3 command, use the tagged source and paper config, validate the
full DB manifest, and preserve the recorded runtime options. The core full-space
warm difference at the later implementation revision is:

```text
--compression_budget_mode match_uniform_quantization_total
--allow_exact_budget_ablation
--sequential_mode depth_to_joint_warm
--stage1_run_dir <verified depth-only source directory>
--joint_mutation_mode standard
--generations 20
--offspring 128
--seed 0
--initially_generated 1
--population_size 1
--crossover_probability 0
--skip_initial_single_candidate_evaluation
```

Keep the full database, target, GPTQ metadata settings, calibration, and survivor
schedule from §5. For IA, use its implementation revision and opt-in with
`joint_mutation_mode=interaction_aware`, leaving sequential mode `none`.
These are reconstruction notes, not launch authorization or verbatim copies of
the absent completed full-space `command.sh` files. The cheap sequential shell
launcher does not forward every exact-budget option; use the archived direct
command when reproducing those runs.

## 16. Git / reproducibility ledger

Dates below are local Git author dates; they are not asserted to be run completion
dates. Git order and author dates can differ slightly around merges.

| Date / period | Revision | Change and research role |
| --- | --- | --- |
| 2024-10-17 | `ba83cb2` | Upstream initial commit; not the start of the thesis |
| 2025-05 | `7a69a21` | Upstream README acceptance information |
| 2026-05-20 | `8a93201` | Thesis debug scripts and joint-search prototype |
| 2026-05-20 | `a157a0f` | Early notes/prototype planning |
| 2026-06-01 | `b42144b`, `f29c489` | Mistral depth grid and longer convergence result |
| 2026-06-02 | `05237f6`, `36664cb` | Dropped-attention output contract and overlapping-drop baseline fixes |
| 2026-06-02 | `1246ab2`, `1e833aa` | Simple depth baselines and seed robustness |
| 2026-06-03 | `47d7a24`, `54f6e13`, `d966342` | TinyLlama sparse feasibility/search and hardware snapshot |
| 2026-06-06/07 | `2bd62fe`, `c37792e`, `53446fb` | TinyLlama joint results, active-budget enforcement, all-linear feasibility |
| 2026-06-10 | `a2d1982`, `10c3433`, `28b727e` | Structured reporting, actual evaluation tokens, effective-bitwidth correction |
| 2026-06-14 | `2b4e4dc`, `5fea7c2`, `b5179fc` | Thesis-scale Mistral grid, results, composition controls |
| 2026-06-14/15 | `2301889`, `b1cff56`, `2b1739f` | Longer approximate compute control and early joint-aware study |
| 2026-06-15 | `aa445b1`, `1004b96`, `a2839f8` | Adaptive/locality controls and Mistral fixed-strength results |
| 2026-06-23 | `ed34cac`, `f7cfab8` | Generalization and downstream LM-eval tooling |
| 2026-06-29 | `c84d60b`, `e1cfb70`, `ee47560` | Attention-scope extension and scope comparison |
| 2026-07-01 | `343d63d` | Interaction-aware joint mutation |
| 2026-07-01/02 | `5c9e519`, `10a7534`, `ed6d38c`, `61d543c` | IA q-projection seeds, generalization, downstream results |
| 2026-07-02 | `95d1c8d`, `666ab05` | Replay attribution framework and q-projection matrices |
| 2026-07-03/04 | `035c0b4`, `eeb4ea3` | Full attention attribution matrices |
| 2026-07-28 | `92003c0` | Four sequential initialization modes |
| 2026-07-29 | `7cf2505`, `69dad29` | Sequential pilots and full Mistral runs |
| 2026-07-30 | `5412e1d`, `30cb9c8`, `8398d20`, `4ab6f4b` | G50 warm workflow/results/artifacts and convergence-report correction |
| 2026-07-31 | `abfa812`, `4c3e6b2`, `eb22d11` | Sequential comparison, quant-warm IA, matched IA G20 control |
| 2026-08-02 | `cdeeedd`, `4adf5b1` | Component crossover implementation and results |
| 2026-08-18 | `b9c8190` | Exact-budget apples-to-apples comparison workflow |
| 2026-08-18 | `b0cba15`, `6e53bc9` | Explicit one-GPU GPTQ and DataLab prebuilt FlashAttention support |
| 2026-08-19 | `ea8e022`, `26284c0` | Memory-safe Stage-1 generation and unknown Ceph inode handling |
| 2026-08-20 | `bf83cbc` | Disk-backed dense teacher logits |
| 2026-08-21 | `24ac4d1` | Resumable search checkpoints |
| 2026-08-25 | `e3e9e79`, `6d3f046` | Evict teacher-logit and quant-weight file pages after reads |
| 2026-08-28 | `ae43aef` | V2 layer-bundle crossover |
| 2026-09-06 | `bfdba59` | Preserve thesis notes/analysis; parent of the V3 implementation |
| 2026-09-06 | `5a388ea` | Feasibility-preserving local-exchange crossover |
| 2026-09-06 | `395f983` | Explicit exact-budget IA ablation |
| 2026-09-07 | `c5eddec` | Exact-budget depth-warm ablation; inspected freeze revision |

### 16.1 Exact revisions supplied and verified locally

| Role | Full commit hash |
| --- | --- |
| Completed reproduction | `6d3f04608a74182ba5b974f26f62153553a0fc24` |
| Crossover V2 | `ae43aef686b42560183644177fcef9d4b783c457` |
| Later V3 branch parent | `bfdba5908e2d2b8aaf62bda6a0d11eb8daa69b7b` |
| Local-exchange V3 | `5a388ea41d9bc1fec74285fed0f81abc4b7ded97` |
| Exact IA support | `395f983548ed93235434ca1e6bc38fefe836d62b` |
| Exact depth-warm support | `c5eddec55a7eec1cd3dbfa4e1314f857f1468dc5` |

Annotated tag: `apples-to-apples-reproduction` → reproduction commit above.
The branch name alone is not a reproducibility anchor; preserve full commit IDs,
config/command artifacts, and the database/source identities together.

## 17. Experiment artifact / backup ledger

### 17.1 Locally inspected artifact families

| Artifact family | Local evidence | What it supports / limitation |
| --- | --- | --- |
| Early experiment history | [experiment_log.csv](results/experiment_log.csv), [detailed report](results/detailed_experiment_report.md), [full early meeting report](results/supervisor_meeting_full_experiment_report.md) | Early runs and failures; CSV coverage ends July 3, not the freeze date |
| Depth baselines | [depth curve](results/depth_pruning_curve.csv), [baseline table](results/baseline_comparison_table.md), [seed table](results/seed_robustness_table.md) | Early depth quality/convergence/robustness |
| Hardware | [snapshot](results/hardware_snapshot_2026-06-03_a40.txt), [bottleneck report](results/hardware_bottleneck_summary.md) | Historical allocation and CPU-memory constraint; not every later run's hardware |
| Mistral q-projection controls | [medium comparison](results/mistral_medium_comparison.md) and `results/runs/thesis_medium_*`, `thesis_compute_matched_*` | Cheap standard and composition controls |
| Mutation controls | [adaptive report](results/adaptive_mutation_screen.md), [fixed-strength report](results/mistral_fixed_mutation_ablation.md), [probability report](results/joint_aware_probability_screen.md) | Negative/locality-control rationale and measurements |
| Sequential comparison | [summary CSV](results/sequential_search_summary.csv), [runs CSV](results/sequential_search_runs.csv), [paired deltas](results/sequential_search_paired_deltas.csv) | G20 means, source/stage-2 costs, and sequential direction effects |
| G50 warm study | [summary](results/depth_warmstart_g50_summary.csv), [runs](results/depth_warmstart_g50_runs.csv), [convergence](results/depth_warmstart_g50_convergence.csv), [report](results/depth_warmstart_g50_comparison.md) | Four G50 conditions, sample SD, compute and trajectories |
| IA cheap results | [paired summary](results/interaction_aware_mistral_qproj_summary.csv), [interpretation](results/interaction_aware_mistral_qproj_summary.md) | q-projection PPL and downstream limitations |
| Generalization and LM-eval | `results/mistral_*generalization*`, `results/mistral_*lmeval*` | Legacy-protocol downstream replays; do not label as full-space transfer |
| Attribution | [framework](results/attribution_framework_report.md), `results/attribution/mistral_qproj_seed*/`, `results/attribution/mistral_attention_full_seed*/` | Recombined candidates, budget handling, replay matrices |
| Original component crossover | [handoff](CROSSOVER_EXPERIMENT_HANDOFF.md), `results/runs/thesis_component_crossover_mistral_s0.25_qproj3.0_g20_o16_p4_c025_seed*/` | Three-seed component metrics and diagnostics |
| Exact-budget protocol | [config](configs/apples_to_apples/mistral7b_v03_paper_matched.json), [audit](artifacts/evopress_apples_to_apples_audit.md), source/tests | Protocol and accounting; no completed E0–E3 directories locally |
| Recent population/V2/V3 extensions | Supplied metrics and external archive in §17.2 | No matching population-only/V2/V3 completed summary directories found locally |
| Full-space IA and warm | Supplied metrics and external archives in §17.2 | Completed DataLab measurements; not locally reverified from raw run artifacts |

Neither `results/apples_to_apples` nor `outputs/apples_to_apples` existed in the
inspected checkout. Local `outputs` contained earlier test-support/dependency/cache
directories, not the full reconstruction database. A configured path is not evidence
that an artifact is locally available.

### 17.2 External persistent archives verified by the researcher on 2026-09-07

Persistent directory: `/share/kerim.halilovic/evopress/backups/`.
The researcher transferred the three archives below to this persistent DataLab
storage and verified each remote SHA256 digest after transfer on 2026-09-07.
This is researcher-observed verification; Codex/local repository inspection did
not independently reopen the archive bytes. Archive sizes and member inventories
were not independently inspected here. `/share` is persistent storage, not
provider-backed-up storage.

| Archive | Researcher-verified remote SHA256 | Reported purpose |
| --- | --- | --- |
| `fullspace_ia_g20_seed0_20260907.tar.gz` | `4e8433c43559d7ff5e6110483b989b246dee791696ea1eba1c8e0030a6967f04` | Full-space IA G20 seed 0 |
| `recent_extensions_20260907.tar.gz` | `dc5f902703cca3ff711b226ca6fab865f6078aa39c195ba59f7ad92fdeb56e12` | Recent population/crossover extensions |
| `fullspace_depthwarm_g20_seed0_20260907.tar.gz` | `b4ed69b3827e29b6570036d0e2ef495f30863bc9a964e1d56650ae1ccdb302d6` | Full-space depth-warm G20 seed 0 |

Before relying on a restored archive, verify the bytes against the supplied digest,
inspect the member list, and identify each run's summary, candidate, commands,
generation metrics, source revision, and runtime. Archive names alone do not prove
that a full quantization database or teacher cache is included. The completed
reproduction/database backup inventory is not specified in the supplied record.

Historical commands use `/home/jovyan/evopress` and
`outputs/experiments/...` inside DataLab. Those are recorded execution paths,
not a guarantee of persistent survival across container restarts.
The old [combined-method plan](combined_compression_next_week_plan.md) explicitly
preserves `outputs/experiments/sparse_db_tinyllama_qproj_s0.50_retry1/sparse_db/`
as a DataLab-only dependency. Its current external availability was not checked.

### 17.3 Depth-source identity ledger

| Identity | Value / convention |
| --- | --- |
| Local source directory | `results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0` |
| Artifact | `final_candidate.json` |
| Supplied/Linux LF artifact SHA256 | `29d4c3806c05003f4e8ba0d69dff22bbb79d3f00bfe58e3d17eb058ce99b4b42` |
| Windows checked-out CRLF artifact SHA256 | `908e9f3afc638cf4509b03376ef1e73d4fed96a8832059868c3d6e250af69ab2` |
| Supplied integer-mask canonical SHA256 | `fae5d211e01ab38b85aca37b8ced9569d09780a8a37948d9dd16fdab7c911bda` |
| Application Boolean-mask SHA256 | `454b86987800d97eba43ad3d810527ff7143b7cabdea85e154ab2d26a1831402` |

Both canonical forms serialize `{"attn": ..., "mlp": ...}` with sorted keys,
compact separators `(',', ':')`, UTF-8, and no added newline. They differ in JSON
values `0/1` versus `false/true`. Compare a checkpoint with the Boolean form used
by the importer, and a portable source ledger with the explicitly declared form.

### 17.4 Minimum artifact set for thesis reconstruction

For each main result, index `command.sh`, finalized `run_summary.json`,
`final_candidate.json`, generation CSVs, runtime/resource files, source revision,
and relevant checkpoint provenance. For joint candidates retain both masks and
module-name-to-bitwidth assignments, not just a screenshot of PPL or a raw vector
without its module ordering.

For the full DB, retain its manifest, exact module/level inventory, calibration
prefix and token digest, tokenizer/model identifiers, reconstruction dtype/shapes,
GPTQ options, process configuration, and the actual storage location. The expected
inventory is 224 × 5 = 1,120 reconstruction files **[derived expectation, not a
local count of the completed database]**.

For plots, preserve the input CSV and an explanation of seed aggregation,
generation alignment, metric direction, and compute axis. Do not smooth or average
away failed runs, mixed metrics, or unmatched horizons without disclosure.

## 18. Supervisor-meeting-ready story

The next presentation should communicate a completed experimental argument rather
than an open-ended list of new operators to try.

1. **Establish credibility:** show the exact protocol and E0/E1/E2 agreement with
   the paper, anchored to the reproduction tag.
2. **State the hard baseline result:** at equal modeled storage, quantization-only
   outperforms the tested 25% joint configuration. Explain why that does not erase
   the depth-only-versus-joint storage tradeoff.
3. **Explain why screening existed:** the q-only protocol made controlled operator
   studies affordable but was deliberately far from the full compression problem.
4. **Show early versus longer search:** depth warm and population improve cheap
   G20 results; their advantage largely disappears by G50.
5. **Preserve the negative crossover story:** duplicate-heavy component recombination,
   repair-heavy layer bundles, and clean local exchange without a quality win.
6. **Show the two full-space transfers together:** IA is mixed; depth warm gives
   a strong early checkpoint advantage in seed 0 under matched stage-2 effort.
7. **End with evidence boundaries and freeze:** stage-1 compute is additional,
   transfers have one seed, G150 warm quality is unknown, and the next work is
   presentation/thesis consolidation unless the supervisor requests more experiments.

Suggested central sentence:

> The thesis now has a validated reproduction, an exact-budget joint comparison,
> and a controlled study showing that informed depth initialization transfers to
> early full-space search more convincingly than the tested mutation and crossover
> improvements, while quantization-only remains stronger at the tested common storage.

Existing decks are historical inputs:
[prototype/composition deck](evopress_supervisor_meeting_presentation-2.tex),
[July attribution deck](evopress_supervisor_meeting_presentation-3.tex),
[algorithm/sequential/crossover deck](SUPERVISOR_PROGRESS_PRESENTATION.tex), and
[speaker notes](SUPERVISOR_PROGRESS_PRESENTATION_NOTES.md).
Their earlier “most promising” and “next experiment” statements need dating when
reused. They were not changed while creating this log.

## 19. Thesis writing map

| Likely chapter | Research-log sections | Material to develop / boundary |
| --- | --- | --- |
| Introduction | §§2–3, 11 | Motivation, combined compression, questions, contributions; acknowledge the negative equal-storage result |
| Background | §§4–5 | LLM structure, GPTQ reconstruction, depth bypass, evolutionary selection, storage versus execution footprint |
| Related Work | §4.1 and repository paper references | Verify final bibliography and situate EvoPress/GPTQ/depth pruning; this log is not a completed literature review |
| Methodology | §§4, 5.2–5.3, 8.3–8.8, 15 | Candidate representation, operators, feasibility, initialization, population, attribution, accounting |
| Experimental Setup | §§5, 8.1–8.2, 14, 16–17 | Separate protocol families, data/token details, seeds, hardware, revisions, evaluation timing |
| Reproduction | §§5–7 | E0–E3, paper comparison, exact common target, engineering feasibility and documented deviations |
| Extensions | §§8–9 | Rationale, screening, operator diagnostics, progression to full-space transfer |
| Results | §§6, 9–10 | Exact tables, matched trajectories, negative findings; preserve sample/population SD distinctions |
| Discussion | §§7, 9.3, 11–13 | Structural penalty, early convergence, transfer failures, mechanistic interpretations and safe claims |
| Limitations | §§13–14 | n=3/n=1, compute fairness, protocol shifts, deployment/accounting, missing artifact details |
| Conclusion | §§11–12, 18 | What was established, what failed, and what remains unproven without promising a universal improvement |

Potential figures to prepare from existing data: E0–E3 common-storage comparison;
cheap G20/G50 paired means; population per-seed deltas; crossover proposal-outcome
counts; IA/warm/standard matched full-space checkpoint trajectories; a storage
accounting diagram; and a source-stage-versus-joint-stage compute table. Plotting
must preserve missing endpoints and the distinction between pre-generation PPL
and post-selection KL. No figure artifacts were generated for this file.

## 20. Open work after experimental freeze

### 20.1 Presentation and writing

Prepare the next supervisor progress presentation from the frozen evidence, then
write the methodology, setup, reproduction, extension, results, and discussion
chapters. Incorporate negative results and the compute caveats as part of the main
argument. Obtain supervisor feedback on emphasis and chapter structure before
expanding the experimental program.

### 20.2 Artifact consolidation and unresolved details

| Open item | Why it matters | Allowed next step under the freeze |
| --- | --- | --- |
| Full-space E0/E1/E2/E3 raw summaries and commands | Exact endpoint precision, E2 seed values/dispersion, KL field definitions, runtime/hardware | Retrieve/index existing artifacts; do not rerun experiments merely to fill the log |
| Standard final-G20 downstream endpoint, if already evaluated | Enables a true final-G20 comparison; current evidence is checkpoint-based | Search existing logs/archive; otherwise mark unavailable |
| Full-space IA/warm generation CSVs and source commands | Verify plotted checkpoint alignment, run IDs, and complete trajectories | Restore/read the supplied archives |
| Recent population/V2/V3 run summaries | Confirm individual seeds, actual compute, dispersion and parent-selection settings | Restore/read `recent_extensions_20260907.tar.gz` |
| Full database manifest and persistent location | Reconstruct retained prefix, ordered-token digest, effective workers, exact dependency | Index the completed DB artifacts and backup inventory |
| External archive verification | Supplied hashes are not local byte verification | Check archive SHA256 and member inventory when storage is available |
| Formal approved proposal, supervisor details, thesis template | Presentation files are not a formal approval record | Obtain existing administrative material; none was found among inspected tracked files |
| Paper bibliography/table references | Final title/version and exact source locations should be precise | Verify the paper and references during writing |
| Final hardware/software environment records | Avoid inferring completed-run hardware from the June snapshot | Recover existing manifests, environment exports, and resource summaries |
| Historical implementation-verification report | Researcher-observed outcomes are recorded in §15.4; a repository-stored CI report remains unavailable | Index existing test/compilation logs when available |
| Figure inputs and statistical notation | Avoid mixed protocols or sample/population SD errors | Build figures/tables from already recorded data with explicit labels |

Downstream evaluation discussion can use the existing three-task legacy results.
Any new full-space task evaluation or new search campaign requires an explicit
post-freeze decision; it is not implied by the presence of an open question here.

### 20.3 Freeze decision and update entry

The experiment set is sufficient to support a thesis story centered on reproducible
joint search, exact-budget comparison, controlled extension studies, and transfer
validation. The freeze does not mean every hypothesis was resolved or every method
won. It means the evidence boundaries are explicit enough to write and discuss.

Initial log entry, 2026-09-07: consolidated local history through commit `c5eddec`,
the supplied completed DataLab reproduction and final transfers, the external
archive ledger, and the consistency notes. Next updates should append the date,
new evidence source, and any changed interpretation while preserving this frozen
experimental record.
