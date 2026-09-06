# EvoPress Experimental Evidence and Findings for Supervisor Meeting

**Report date:** June 7, 2026  
**Project:** Master thesis experiments with EvoPress  
**Models studied:** `mistralai/Mistral-7B-v0.3` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Primary evaluation dataset:** WikiText2  
**Primary metric:** perplexity (PPL), where lower is better

## 1. Purpose of this report

This document consolidates the experimental evidence collected so far into one
supervisor-facing report. It covers:

- Mistral-7B depth-pruning experiments;
- convergence and seed robustness;
- random and late-layer pruning baselines;
- the hardware and memory constraints observed on TU Wien Datalab;
- TinyLlama SparseGPT and GPTQ feasibility experiments;
- sequential depth+sparsity experiments;
- independent and jointly optimized depth+quantization experiments;
- replay checks and component ablations;
- supported findings, limitations, and recommended next experiments;
- a suggested structure for the supervisor presentation.

The goal is not to claim a complete reproduction of every EvoPress result.
Instead, the current work establishes a coherent experimental story:

1. EvoPress depth search clearly outperforms simple depth-pruning heuristics.
2. Additional generations materially improve difficult high-sparsity searches.
3. Sparse and quantization database pipelines work end to end on a 1.1B model,
   including all 154 transformer projection matrices for GPTQ.
4. Two compression methods can be combined and evaluated reproducibly.
5. Joint depth+quantization search preserves almost the same average PPL as
   depth-only search while adding `q_proj` quantization.
6. The current three-seed evidence does not yet demonstrate consistent
   optimization synergy between depth pruning and quantization.
7. A broader all-linear joint experiment discovered a depth mask that is more
   compatible with quantization, but its nonuniform bit allocation was slightly
   worse than uniform 3-bit quantization.

## 2. Executive summary

### 2.1 Main Mistral-7B result

The dense Mistral-7B reference achieved **5.35 WikiText2 PPL**. Ten-generation
EvoPress depth searches produced:

| Requested depth sparsity | Final WikiText2 PPL | Final train PPL | Dropped attention | Dropped MLP |
|---:|---:|---:|---:|---:|
| 12.5% | 6.75 | 6.27 | 4 | 4 |
| 25.0% | 14.70 | 13.30 | 8 | 8 |
| 37.5% | 51.69 | 48.20 | 12 | 12 |
| 50.0% | 371.75 | 337.00 | 16 | 16 |

This is the main compression-versus-quality curve. It shows that:

- 12.5% depth pruning retains relatively good language-model quality;
- degradation becomes substantial at 25%;
- 37.5% and 50% are much harder operating points;
- the search budget becomes increasingly important at higher sparsity.

At 37.5% sparsity, extending the search from 10 to 20 generations improved
WikiText2 PPL from **51.69 to 26.00**, a **49.7% reduction in PPL** relative to
the 10-generation result. This is strong convergence evidence that the
10-generation result had not saturated.

### 2.2 Baseline result

EvoPress substantially outperformed random module dropping and the late-layer
heuristic at every tested Mistral-7B sparsity:

| Sparsity | EvoPress | Random median | Late-layer |
|---:|---:|---:|---:|
| 12.5% | **6.75** | 21.50 | 61.16 |
| 25.0% | **14.70** | 2490.00 | 541.00 |
| 37.5% | **51.69** | 4476.00 | 1419.00 |
| 50.0% | **371.75** | 15516.00 | 4992.00 |

The random distributions are highly unstable and include catastrophic
configurations. Medians are therefore more representative than means.

### 2.3 Small-model pipeline result

TinyLlama allowed the full sparse and quantization workflow to run under the
16 GB CPU RAM limit:

- dense TinyLlama: **8.97 PPL**;
- 50% average SparseGPT search over attention `q_proj`: **9.00 PPL**;
- 3-bit-average GPTQ search over attention `q_proj`: **9.00 PPL**.

These are not full-model 50% sparsity or full-model 3-bit results. Both methods
were deliberately restricted to `q_proj` to establish an end-to-end feasible
pipeline with controlled memory usage.

The quantization scope was subsequently expanded:

| Quantization scope | Candidate modules | Database result | WikiText2 PPL |
|---|---:|---|---:|
| `q_proj` only | 22 | 66 files, 529 MB | 9.00 searched |
| All attention projections | 88 | 264 files, 1.19 GB | database feasibility |
| All transformer projections | 154 | 462 files, 5.55 GB | 13.62 uniform, 14.58 searched |

The all-linear scope includes `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` in all 22 transformer blocks. It still
excludes embeddings, normalization layers, and `lm_head`.

### 2.4 Combined-method result

For requested 12.5% TinyLlama depth pruning and 3-bit-average `q_proj`
quantization, three paired seeds gave:

| Seed | Depth-only PPL | Joint depth+quant PPL | Difference, joint - depth |
|---:|---:|---:|---:|
| 0 | 12.02 | **11.03** | -0.99 |
| 1 | **12.28** | 12.55 | +0.27 |
| 2 | **11.03** | 11.72 | +0.69 |
| **Mean** | **11.78** | **11.77** | **-0.01** |

The means are virtually identical. The correct interpretation is:

- joint search successfully combines depth removal and quantization;
- the added `q_proj` quantization has only a small quality cost;
- joint search is not consistently better than depth-only search;
- one seed improved, while two became worse;
- three seeds are insufficient to claim statistically significant synergy.

The no-quantization replay of each joint depth mask produced a mean PPL of
**11.62**. Adding the joint quantization configuration increased PPL by
**0.11, 0.22, and 0.12** for seeds 0, 1, and 2. The quantization penalty was
small and consistent, averaging **0.15 PPL**.

### 2.5 Broader all-linear combined result

The all-linear database made the combined experiment substantially more
meaningful because almost 969 million transformer projection weights became
quantization candidates.

| Configuration | WikiText2 PPL |
|---|---:|
| Dense | 8.97 |
| Depth-only seed 0 | 12.02 |
| Uniform all-linear 3-bit | 13.62 |
| Searched all-linear 3-bit | 14.58 |
| Depth + uniform all-linear 3-bit | 21.77 |
| Depth + independently searched all-linear 3-bit | 23.81 |
| Joint active-budget depth + all-linear 3-bit | 21.47 |
| Joint all-linear depth mask without quantization | 12.52 |
| Joint all-linear depth mask + uniform 3-bit | **21.25** |
| Independent depth mask + joint quant allocation | 22.14 |

The joint result exactly replayed at **21.47 PPL**. It improved over the
independently searched combination by 2.34 PPL and over uniform composition by
0.30 PPL. The completed cross-ablations show that the joint depth mask paired
with uniform 3-bit quantization is better still at **21.25 PPL**. Conversely,
the joint nonuniform allocation is worse than uniform quantization on both
tested masks. The supported result is therefore a mask-compatibility
interaction: joint search discovered a depth mask that tolerates broad
quantization better, but did not discover a superior bit allocation.

## 3. Research questions addressed

The experiments collected so far address the following questions.

### RQ1: Does evolutionary depth search find better module-removal patterns than simple heuristics?

**Answer:** Yes. On Mistral-7B, EvoPress was better than the random and
late-layer baselines at all four tested sparsity levels.

### RQ2: Does increasing the evolutionary search budget improve the result?

**Answer:** Yes, at the difficult 37.5% Mistral-7B setting. Increasing the
budget from 10 to 20 generations reduced final PPL from 51.69 to 26.00.

### RQ3: Are the selected depth configurations stable across seeds?

**Answer:** Only partially. Final PPL varies moderately, and selected module
sets have low overlap. Different masks can therefore produce comparable
quality.

### RQ4: Can sparse and quantization database generation run in the available environment?

**Answer:** Yes for TinyLlama. Sparse feasibility was demonstrated for
`q_proj`, and GPTQ database generation was expanded to all 154 transformer
projection matrices. Full Mistral-7B sparse database generation was not
feasible under the current 16 GB CPU/container RAM limit.

### RQ5: Can depth pruning be combined with weight sparsity?

**Answer:** Yes. Depth+sparsity replay runs completed, although the searched
sparse configuration did not consistently outperform uniform sparsity after
depth pruning.

### RQ6: Can depth pruning and quantization be optimized jointly?

**Answer:** Yes. A joint evolutionary search was implemented and completed for
three seeds, with exact replay verification for seed 0.

### RQ7: Does joint optimization currently outperform independently selected methods?

**Answer:** Not consistently. Seed 0 showed a gain, but the ablations indicate
that the gain came from the selected depth mask rather than from a superior
quantization allocation. Seeds 1 and 2 did not improve over depth-only search.

### RQ8: Does broader all-linear joint optimization improve over composition?

**Answer:** At seed 0, yes. The active-budget joint result achieved 21.47 PPL,
compared with 23.81 for independent searched composition and 21.77 for uniform
composition. Cross-ablations show that the advantage came from the jointly
discovered depth mask: combining that mask with uniform 3-bit quantization
improved PPL further to 21.25, while the joint nonuniform allocation slightly
degraded both tested masks.

## 4. Experimental infrastructure and reproducibility

### 4.1 Run tracking

Experiments are tracked in:

```text
results/experiment_log.csv
```

The log contains the model, method, compression target, search parameters,
dataset settings, dtype, seed, final metrics, runtime, hardware, status, notes,
and output directory.

At the time of this report, the log contains:

- **59 total recorded attempts**;
- **48 completed attempts**;
- **11 failed attempts**.

The failed count includes dependency and launcher failures that were fixed and
successfully retried. It must not be interpreted as an 11-run algorithmic
failure rate.

### 4.2 Standard run artifacts

Successful run directories generally contain:

```text
command.sh
run.log
runtime.txt
layer_drop_config.txt          # depth methods
generation_metrics.csv         # evolutionary searches
memory_samples.csv             # selected database/search jobs
```

Lightweight artifacts required for analysis are copied into:

```text
results/runs/<run_id>/
```

This is necessary because experiments run remotely on Datalab while analysis
and report preparation take place in the local repository.

### 4.3 Log parsing

The depth-search parser extracts:

- phase;
- generation number;
- train fitness;
- WikiText2 PPL;
- train PPL;
- parent attention mask;
- parent MLP mask.

This avoids manual metric transcription and allows convergence plots to be
regenerated from logs.

### 4.4 Dependency preflight

Several initial retries failed after the Datalab server restarted without the
Python requirements being installed. Launchers were subsequently given a
dependency/preflight check. This is an engineering lesson rather than an
experimental result:

- environment validation should occur before a long background job starts;
- missing dependencies should fail before a run is added as a scientific
  attempt;
- software-environment identity should ideally be recorded with every run.

## 5. Hardware environment and bottleneck evidence

### 5.1 Variable GPU allocation

TU Wien Datalab assigns different hardware depending on availability. The
recorded experiments used:

- **NVIDIA Tesla T4**, approximately 15 GB logged VRAM;
- **NVIDIA A40**, approximately 45 GB logged VRAM.

This does not invalidate PPL comparisons because the model, configuration,
dtype, and evaluation setup determine the numerical experiment. It does mean
that runtime comparisons should only be made with the GPU type explicitly
reported.

### 5.2 CPU RAM limit

The container cgroup limit is approximately:

```text
16 GB CPU RAM
```

The A40 snapshot showed roughly 45 GB VRAM available, while the container still
had only 16 GB host RAM. The observed bottleneck is therefore CPU/container
memory rather than GPU VRAM.

### 5.3 Consequence for compression pipelines

Depth pruning is feasible because it evaluates structural masks without
precomputing and retaining a large multi-level database for every weight
matrix.

SparseGPT/FastOBC and GPTQ database workflows are more memory intensive because
they create and process multiple compressed alternatives. The attempted full
Mistral-7B sparse database generation failed under the 16 GB host-memory limit.

The practical response was:

1. stop retrying full Mistral-7B database generation on the same hardware;
2. validate the pipeline on TinyLlama;
3. restrict the first feasibility study to `q_proj`;
4. collect memory and artifact evidence;
5. use the smaller database in sparse and quant evolutionary searches.

For a future full Mistral-7B database study, **64 GB host RAM** is a defensible
minimum request. A 32 GB node could be used for a monitored diagnostic retry,
but it would still provide limited safety margin.

## 6. Evaluation conventions and interpretation

### 6.1 Perplexity

Perplexity is the primary quality metric:

- lower PPL is better;
- dense-model PPL is the reference;
- a small PPL increase means better quality retention;
- extremely large or infinite PPL indicates severe model degradation.

Perplexity is not linear in perceived quality. For example, moving from 5 to 10
and moving from 100 to 105 are not equivalent changes.

### 6.2 Train PPL versus WikiText2 PPL

The search records:

- a train/calibration PPL used to monitor the search;
- WikiText2 evaluation PPL used as the principal reported outcome.

The final conclusions are based on WikiText2 PPL. Train PPL is useful for
checking whether the search objective and evaluation trend move together.

### 6.3 Requested versus realized depth sparsity

Mistral-7B has 32 transformer layers. Requested sparsities map exactly to:

- 4 of 32 attention and 4 of 32 MLP modules at 12.5%;
- 8 of 32 at 25%;
- 12 of 32 at 37.5%;
- 16 of 32 at 50%.

TinyLlama has 22 transformer layers. The implementation uses integer module
counts, so:

- requested 12.5% removes 2 attention and 2 MLP modules;
- realized per-type removal is `2 / 22 = 9.09%`;
- requested 25% removes 5 attention and 5 MLP modules;
- realized per-type removal is `5 / 22 = 22.73%`.

Results should retain the requested labels for run identification while also
reporting the exact removed-module count.

### 6.4 Scope of weight compression

The sparse experiments and the first quantization experiments modify attention
`q_proj` matrices. The later quantization study covers all seven transformer
projection types:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Therefore:

- “50% sparse” still means a 50% average target over selected `q_proj` weights;
- the first “3-bit quantized” result means a 3-bit average over `q_proj`;
- the all-linear 3-bit result covers all 154 transformer projection matrices;
- embeddings, normalization layers, and `lm_head` remain unquantized;
- database files contain dequantized FP16 reconstructions, not packed 3-bit
  deployment tensors;
- the work establishes quality effects and search behavior, not a measured
  whole-model storage or runtime compression ratio.

This distinction must be stated in the presentation.

## 7. Phase I: Mistral-7B depth-pruning experiments

### 7.1 Dense reference

| Run ID | Model | Sequence length | Dtype | WikiText2 PPL |
|---|---|---:|---|---:|
| `dense_mistral7b_seed1` | Mistral-7B-v0.3 | 2048 | float16 | **5.35** |

This establishes that the model and WikiText2 evaluation pipeline were loaded
correctly. It also provides the baseline required to interpret compression
damage.

### 7.2 Main four-point depth curve

Common search parameters:

| Parameter | Value |
|---|---|
| Model | `mistralai/Mistral-7B-v0.3` |
| Calibration data | WikiText2 |
| Sequence length | 2048 |
| Calibration tokens | 8192 |
| Generations | 10 |
| Offspring | 8 |
| Initially generated | 16 |
| Initial tokens | 512 |
| Tokens per selection | 512, 2048 |
| Fitness | KL divergence |
| Attention implementation | SDPA |
| Dtype | float16 |
| Seed | 1 |

Results:

| Run ID | Sparsity | WikiText2 PPL | Train PPL | PPL increase over dense | Runtime, min | GPU |
|---|---:|---:|---:|---:|---:|---|
| `depth_mistral7b_s0.125_seed1` | 12.5% | **6.75** | 6.27 | +1.40 | 64.22 | T4 |
| `depth_mistral7b_s0.25_seed1` | 25.0% | **14.70** | 13.30 | +9.35 | 56.30 | T4 |
| `depth_mistral7b_s0.375_seed1` | 37.5% | **51.69** | 48.20 | +46.34 | 46.08 | T4 |
| `depth_mistral7b_s0.50_seed1` | 50.0% | **371.75** | 337.00 | +366.40 | 37.00 | T4 |

Interpretation:

- 12.5% is the strongest quality-preserving point in the tested curve.
- 25% may still be useful when compression is prioritized over quality.
- 37.5% is difficult but can be improved with a larger search budget.
- 50% causes severe degradation even after evolutionary optimization.
- Runtime falls as more modules are dropped because later evaluation passes
  execute a structurally smaller model.

### 7.3 Convergence extension at 37.5%

The 37.5% setting was repeated with 20 generations:

| Generations | WikiText2 PPL | Train PPL | Runtime, min | GPU |
|---:|---:|---:|---:|---|
| 10 | 51.69 | 48.20 | 46.08 | T4 |
| 20 | **26.00** | **24.50** | 87.92 | T4 |

Generation-wise evidence:

| Generation | WikiText2 PPL | Train PPL |
|---:|---:|---:|
| 1 | 277.25 | 267.00 |
| 2 | 208.50 | 193.00 |
| 3 | 167.50 | 155.00 |
| 4 | 137.75 | 125.00 |
| 5 | 80.69 | 74.60 |
| 6 | 68.50 | 63.30 |
| 7 | 69.31 | 64.60 |
| 8 | 59.28 | 55.00 |
| 9 | 62.34 | 57.70 |
| 10 | 51.69 | 48.20 |
| 11 | 51.69 | 48.20 |
| 12 | 42.78 | 43.00 |
| 13 | 42.78 | 43.00 |
| 14 | 37.16 | 36.40 |
| 15 | 33.78 | 32.10 |
| 16 | 26.41 | 25.40 |
| 17 | 26.41 | 25.40 |
| 18 | 26.77 | 25.40 |
| 19 | 26.00 | 24.50 |
| 20 | 26.00 | 24.50 |

Important observations:

- generation 5 was not converged;
- generation 10 was also not converged;
- plateaus occurred at generations 10-11, 12-13, and 16-17;
- later mutations still found substantial improvements;
- the final 20-generation PPL is 67.8% lower than the generation-5 PPL;
- the final 20-generation PPL is 49.7% lower than the generation-10 PPL.

This supports using larger generation budgets at aggressive sparsity.

### 7.4 Random and late-layer baselines

The baseline methods remove the same numbers of attention and MLP modules as
the corresponding EvoPress configuration.

Random baseline raw results:

| Sparsity | Seed 1 | Seed 2 | Seed 3 | Completed | Scientific failures |
|---:|---:|---:|---:|---:|---:|
| 12.5% | 21.50 | 11.29 | 38048.00 | 3 | 0 |
| 25.0% | 31.36 | 2490.00 | 13784.00 | 3 | 0 |
| 37.5% | 10736.00 | 4476.00 | 1158.00 | 3 | 0 |
| 50.0% | 8488.00 | 22544.00 | non-finite | 2 | 1 |

Aggregate comparison:

| Sparsity | EvoPress | Random mean | Random median | Random sample SD | Late-layer |
|---:|---:|---:|---:|---:|---:|
| 12.5% | **6.75** | 12693.60 | 21.50 | 21957.56 | 61.16 |
| 25.0% | **14.70** | 5435.12 | 2490.00 | 7334.10 | 541.00 |
| 37.5% | **51.69** | 5456.67 | 4476.00 | 4863.72 | 1419.00 |
| 50.0% | **371.75** | 15516.00 | 15516.00 | 9939.09 | 4992.00 |

Interpretation:

- random module removal has very high variance;
- even at 12.5%, a random mask can range from 11.29 to 38048 PPL;
- late-layer removal is deterministic but consistently much worse than
  EvoPress;
- module location matters strongly;
- the evolutionary search finds nontrivial structures that simple heuristics
  miss.

The 50% random non-finite result is scientifically meaningful and should be
retained as model collapse, not silently discarded. Means should nevertheless
be presented with medians and raw values because the distributions are
extremely skewed.

### 7.5 Seed robustness at 37.5%

| Seed | WikiText2 PPL | Train PPL | GPU |
|---:|---:|---:|---|
| 1 | 51.69 | 48.20 | T4 |
| 2 | **40.19** | 28.50 | A40 |
| 3 | 47.91 | 41.90 | A40 |
| **Mean** | **46.60** | | |
| **Sample SD** | **5.86** | | |

Pairwise Jaccard overlap of the selected dropped-module sets:

| Seed pair | Jaccard overlap |
|---|---:|
| 1 versus 2 | 0.371 |
| 1 versus 3 | 0.371 |
| 2 versus 3 | 0.412 |

Interpretation:

- final quality varies across seeds but remains in the same broad range;
- the selected masks are not identical;
- multiple structurally different solutions can have comparable PPL;
- reporting a single seed is insufficient for strong robustness claims;
- GPU differences affect runtime interpretation, but there is no evidence that
  they explain the mask differences.

## 8. Phase II: TinyLlama single-method feasibility

### 8.1 Why TinyLlama was selected

TinyLlama was chosen to:

- fit comfortably within the 16 GB CPU RAM constraint;
- test complete sparse and quantization workflows;
- support faster multi-seed and ablation experiments;
- provide a practical platform for combined-method research.

### 8.2 Dense and depth-only references

| Run ID | Method | Requested depth sparsity | Dropped attn/MLP | WikiText2 PPL | Train PPL | GPU |
|---|---|---:|---:|---:|---:|---|
| `dense_tinyllama_seq1024_seed0` | Dense | 0% | 0 / 0 | **8.97** | - | T4 |
| `depth_tinyllama_s0.125_seed0` | Depth EvoPress | 12.5% | 2 / 2 | 12.02 | 13.00 | T4 |
| `depth_tinyllama_s0.125_seed1` | Depth EvoPress | 12.5% | 2 / 2 | 12.28 | 13.20 | A40 |
| `depth_tinyllama_s0.125_seed2` | Depth EvoPress | 12.5% | 2 / 2 | **11.03** | 12.50 | A40 |
| `depth_tinyllama_s0.25_seed0` | Depth EvoPress | 25.0% | 5 / 5 | 42.59 | 45.20 | T4 |

For the three requested-12.5% depth seeds:

| Statistic | WikiText2 PPL |
|---|---:|
| Mean | 11.78 |
| Median | 12.02 |
| Sample SD | 0.66 |
| Best | 11.03 |
| Worst | 12.28 |

The move from requested 12.5% to requested 25% causes a large quality loss,
consistent with the Mistral-7B depth curve.

### 8.3 SparseGPT database feasibility

Database run:

```text
sparse_db_tinyllama_qproj_s0.50_retry1
```

Configuration and evidence:

| Property | Value |
|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Compressed matrix type | attention `q_proj` only |
| Average sparsity target | 50% |
| Levels | -3 through +3 around the target |
| Transformer layers | 22 |
| Database files | 154 `.pth` files |
| Database size | approximately 1.23 GB |
| Runtime | 1.68 minutes |
| Peak sampled CPU RAM | 10.31 GB |
| Peak sampled GPU memory | 0.89 GB |
| GPU | Tesla T4 |
| Status | Completed |

The database was then consumed by a 20-generation sparse search:

```text
sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1
```

| Metric | Result |
|---|---:|
| Final WikiText2 PPL | **9.00** |
| Final train PPL | 9.83 |
| Runtime | 6.18 minutes |
| Peak sampled CPU RAM | 6.80 GB |
| Peak sampled GPU memory | 2.68 GB |
| GPU | NVIDIA A40 |

The selected per-layer sparse levels ranged from -2 to +3 and summed to zero,
preserving the requested average 50% `q_proj` sparsity target.

The 9.00 PPL result is close to the 8.97 dense reference. This demonstrates
that:

- the sparse database is internally usable;
- the evolutionary search can select per-layer sparse alternatives;
- the selected `q_proj` matrices tolerate substantial sparsity;
- the end-to-end sparse pipeline is feasible on the smaller model.

### 8.4 GPTQ database feasibility

Database run:

```text
quant_db_tinyllama_qproj_bits234
```

Configuration and evidence:

| Property | Value |
|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Quantized matrix type | attention `q_proj` only |
| Candidate bit widths | 2, 3, 4 |
| Calibration bit width | 3 |
| Group size | 128 |
| Sequence length | 1024 |
| Calibration tokens | 4096 |
| Transformer layers | 22 |
| Database files | 66 `.pth` files |
| Missing expected files | 0 |
| Database size | approximately 529 MB |
| Runtime | 1.43 minutes |
| Peak sampled CPU RAM | 9.96 GB |
| Peak sampled GPU memory | 1.06 GB |
| GPU | NVIDIA A40 |
| Status | Completed |

The database was consumed by:

```text
quant_search_tinyllama_qproj_3bit_g20_seed0
```

Final allocation:

```text
[2, 2, 2, 2, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4]
```

Allocation summary:

| Bit width | Number of layers |
|---:|---:|
| 2-bit | 4 |
| 3-bit | 14 |
| 4-bit | 4 |
| Average | exactly 3.0 bits |

Results:

| Metric | Value |
|---|---:|
| WikiText2 PPL | **9.00** |
| Train PPL | 9.87 |
| Runtime | 6.25 minutes |
| Peak sampled CPU RAM | 6.47 GB |
| Peak sampled GPU memory | 2.81 GB |

The searched allocation preserves the exact bit budget while assigning higher
or lower precision to different layers. Its PPL is only 0.03 above dense.

### 8.5 Broader all-linear GPTQ feasibility

Linux page-cache eviction was added after each saved candidate tensor to reduce
cgroup RAM pressure. This allowed the GPTQ scope to expand successfully.

| Scope | Module directories | Candidate files | Database size | Peak CPU RAM | Peak GPU memory | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| All attention projections | 88 | 264 | 1.19 GB | 8.48 GB | 1.10 GB | 3.42 min |
| All transformer projections | 154 | 462 | 5.55 GB | 7.43 GB | 1.77 GB | 6.73 min |

The all-linear database completed with zero missing files under the 16 GB CPU
RAM limit. This demonstrates that the earlier `q_proj` restriction was a
conservative feasibility choice rather than a fundamental TinyLlama limit.

Quality results:

| All-linear quantization method | WikiText2 PPL | Difference from dense |
|---|---:|---:|
| Uniform 3-bit | **13.62** | +4.65 |
| 20-generation searched 3-bit allocation | 14.58 | +5.61 |

The searched allocation was 0.96 PPL worse than uniform 3-bit quantization.
The KL-calibration fitness therefore did not produce a better WikiText2 result
for this seed and search budget. Uniform quantization is the stronger
quant-only baseline for the current all-linear study.

## 9. Phase III-A: sequential depth+sparsity

### 9.1 Experimental design

The sparse configuration was first optimized on the dense TinyLlama model.
That configuration, or a uniform 50% alternative, was then applied together
with a previously selected depth mask.

This tests method compatibility but is not a joint optimization of the depth
and sparse decisions.

### 9.2 Results

| Requested depth | Depth-only | + uniform 50% `q_proj` sparse | + searched 50% `q_proj` sparse |
|---:|---:|---:|---:|
| 12.5% | 12.02 | 11.97 | **11.92** |
| 25.0% | **42.59** | 43.19 | 43.44 |

Differences relative to depth-only:

| Requested depth | Uniform difference | Searched difference |
|---:|---:|---:|
| 12.5% | -0.05 (-0.42%) | -0.10 (-0.83%) |
| 25.0% | +0.60 (+1.41%) | +0.85 (+2.00%) |

Interpretation:

- both combined models evaluated successfully;
- adding 50% `q_proj` sparsity caused little additional damage;
- at the lower depth target, small improvements are within likely run
  variation and should not be treated as proven regularization effects;
- the searched sparse allocation was not consistently better than uniform
  sparsity after applying a depth mask;
- optimizing a sparse configuration on the dense model does not guarantee that
  it is optimal after structural modules are removed.

This result motivated moving from sequential composition to joint search.

## 10. Phase III-B: depth+quantization

### 10.1 Independent combination

The independently selected depth mask and independently selected 3-bit-average
quant configuration were combined:

```text
combined_tiny_depth0125_quant3_independent_seed0
```

Result:

| Method | WikiText2 PPL |
|---|---:|
| Dense | 8.97 |
| Depth-only seed 0 | 12.02 |
| Independent depth + independent quant | 11.97 |

The independent combination worked and introduced no significant additional
quality loss relative to depth-only seed 0.

### 10.2 Joint-search implementation

The joint search simultaneously evolves:

- a depth mask with 2 dropped attention and 2 dropped MLP modules;
- one 2-, 3-, or 4-bit `q_proj` choice per transformer layer;
- a quantization allocation constrained to an average of 3.0 bits.

Core settings:

| Parameter | Value |
|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Requested depth sparsity | 12.5% |
| Realized dropped modules | 2 attention, 2 MLP |
| Quantization scope | attention `q_proj` |
| Quantization candidates | 2, 3, 4 bits |
| Average bit target | 3.0 |
| Generations | 10 |
| Offspring | 8 |
| Calibration data | WikiText2 |
| Calibration tokens | 4096 |
| Sequence length | 1024 |
| Fitness | KL divergence |
| Dtype | float16 |

The search implementation was corrected to reload the final selected parent
before final evaluation. Without this, the displayed final configuration and
the evaluated in-memory model could diverge.

### 10.3 Seed 0 joint result

Run:

```text
joint_tiny_depth0125_quant3_g10_seed0
```

Final depth configuration:

```text
['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
 'none', 'none', 'none', 'mlp', 'mlp', 'attn', 'none', 'none',
 'none', 'none', 'none', 'none', 'attn', 'none']
```

Final quantization allocation:

```text
[2, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 4, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3]
```

Result:

| Metric | Value |
|---|---:|
| Dropped attention modules | 2 |
| Dropped MLP modules | 2 |
| Average quant bit width | 3.0 |
| WikiText2 PPL | **11.03** |
| Train PPL | 12.09 |

The saved final configuration was replayed independently:

```text
replay_joint_tiny_depth0125_quant3_seed0
```

Replay WikiText2 PPL was also **11.03**, providing an exact reproducibility
check for the saved configuration and evaluation path.

### 10.4 Seed 0 attribution ablations

The seed 0 result initially appeared to improve over the independently combined
result, 11.03 versus 11.97. Four ablations were used to identify the source:

| Depth mask | Quant allocation | WikiText2 PPL |
|---|---|---:|
| Joint | None | **10.92** |
| Joint | Joint | 11.03 |
| Joint | Uniform 3-bit | 11.05 |
| Joint | Independent searched quant | **11.01** |
| Independent | Joint quant | 12.02 |
| Independent | Independent quant | 11.97 |

Interpretation:

- the joint depth mask alone achieved 10.92 PPL;
- adding the joint quant allocation increased PPL by only 0.11;
- uniform 3-bit and the independently searched quant allocation performed
  nearly identically on the joint depth mask;
- applying the joint quant allocation to the independent depth mask did not
  improve it;
- the seed 0 improvement came primarily from the depth mask discovered during
  joint search, not from a special interaction with the quant allocation.

This is an important negative result. It prevents overstating the evidence as
quantization-depth synergy.

### 10.5 Three-seed joint comparison

Full results:

| Seed | Depth-only | Joint mask without quant | Full joint depth+quant |
|---:|---:|---:|---:|
| 0 | 12.02 | **10.92** | 11.03 |
| 1 | **12.28** | 12.33 | 12.55 |
| 2 | **11.03** | 11.60 | 11.72 |
| **Mean** | 11.78 | **11.62** | 11.77 |
| **Median** | 12.02 | **11.60** | 11.72 |
| **Sample SD** | 0.66 | 0.71 | 0.76 |

Paired full-joint differences:

| Seed | Full joint - depth-only |
|---:|---:|
| 0 | -0.99 |
| 1 | +0.27 |
| 2 | +0.69 |
| **Mean** | **-0.01** |
| **Median** | **+0.27** |
| **Sample SD** | **0.87** |

Joint-mask-only differences:

| Seed | Joint mask without quant - depth-only |
|---:|---:|
| 0 | -1.10 |
| 1 | +0.05 |
| 2 | +0.57 |
| **Mean** | **-0.16** |

Quantization penalty on each joint mask:

| Seed | Full joint - joint mask without quant |
|---:|---:|
| 0 | +0.11 |
| 1 | +0.22 |
| 2 | +0.12 |
| **Mean** | **+0.15** |
| **Sample SD** | **0.06** |

The strongest current conclusion is:

> At the tested TinyLlama operating point, joint search preserved essentially
> the same average WikiText2 PPL as depth-only search while adding 3-bit-average
> `q_proj` quantization. The quantization penalty was small and consistent, but
> the joint search did not produce a consistent PPL advantage across seeds.

### 10.6 Configuration overlap

Zero-based dropped-module indices:

| Seed | Search | Dropped attention | Dropped MLP |
|---:|---|---|---|
| 0 | Depth-only | 12, 19 | 9, 11 |
| 0 | Joint | 13, 20 | 11, 12 |
| 1 | Depth-only | 17, 20 | 11, 14 |
| 1 | Joint | 12, 16 | 11, 14 |
| 2 | Depth-only | 4, 12 | 4, 11 |
| 2 | Joint | 12, 20 | 11, 17 |

Same-seed depth-only versus joint overlap:

| Seed | Attention Jaccard | MLP Jaccard |
|---:|---:|---:|
| 0 | 0.000 | 0.333 |
| 1 | 0.000 | 1.000 |
| 2 | 0.333 | 0.333 |

This reinforces the earlier Mistral observation that:

- masks are seed-sensitive;
- several different module subsets can produce similar PPL;
- low mask overlap does not necessarily imply low result quality;
- search stability should be evaluated in both metric space and
  configuration space.

### 10.7 Joint depth+quantization over all transformer projections

The quantization database was expanded from 22 `q_proj` matrices to all 154
transformer projection matrices. The joint search used:

| Parameter | Value |
|---|---|
| Requested depth sparsity | 12.5% |
| Realized dropped modules | 2 attention, 2 MLP |
| Quantized scope | all attention and MLP projections |
| Active quantization target | 3.0 bits |
| Grouping rule | matrix size |
| Active-budget enforcement | enabled |
| Generations | 10 |
| Offspring | 8 |
| Fitness | KL divergence |

Active-budget enforcement excludes projections inside dropped attention or MLP
modules from the bit-average calculation. This prevents removed modules from
absorbing the quantization budget.

Results:

| Configuration | WikiText2 PPL | Difference from dense |
|---|---:|---:|
| Dense | 8.97 | - |
| Depth-only seed 0 | 12.02 | +3.05 |
| Uniform all-linear 3-bit | 13.62 | +4.65 |
| Searched all-linear 3-bit | 14.58 | +5.61 |
| Depth + uniform all-linear 3-bit | 21.77 | +12.80 |
| Depth + independent searched allocation | 23.81 | +14.84 |
| Joint active-budget search | 21.47 | +12.50 |
| Joint depth mask without quantization | 12.52 | +3.55 |
| **Joint depth mask + uniform 3-bit** | **21.25** | **+12.28** |
| Independent depth mask + joint allocation | 22.14 | +13.17 |

Comparisons:

- joint versus independent searched composition: **-2.34 PPL**;
- joint versus uniform composition: **-0.30 PPL**;
- joint versus its own unquantized mask: **+8.95 PPL**;
- joint mask + uniform quant versus original mask + uniform quant:
  **-0.52 PPL**;
- joint allocation versus uniform quant on the joint mask: **+0.22 PPL**;
- joint allocation versus uniform quant on the independent mask:
  **+0.37 PPL**;
- searched quant-only versus uniform quant-only: **+0.96 PPL**.

The joint result improved from 30.81 PPL at generation 1 to 21.47 at generation
9, then remained unchanged at generation 10. Exact replay also produced 21.47
PPL.

The final mask dropped:

```text
MLP layer 11
attention layer 13
attention and MLP layer 19
```

All indices are zero-based. The final quant allocation was almost uniform:

```text
layer 1 v_proj: 4-bit
layer 3 k_proj: 2-bit
layer 13 k_proj: 2-bit
all other active projections: 3-bit
```

Layer 13 attention is dropped, so its 2-bit `k_proj` does not affect the active
budget or inference. The two active nonuniform assignments balance each other,
giving an exact active average of 3.0 bits.

Configuration accounting:

| Quantity | Value |
|---|---:|
| Full quantizable transformer projection weights | 968,884,224 |
| Active projection weights after depth removal | 880,803,840 |
| Active fraction | 90.91% |
| Active average bit width | 3.0000 |
| Full-database average, including dropped modules | 2.99946 |
| Peak sampled CPU RAM during joint search | 10.84 GB |
| Peak sampled GPU memory | 2.74 GB |

Interpretation:

- joint optimization is better than the independently searched composition;
- the best broad combined result is the joint depth mask with uniform
  quantization, at 21.25 PPL;
- the joint depth mask alone is 0.50 PPL worse than the original depth mask;
- most of the final quality loss comes from broad all-linear quantization;
- under uniform quantization, the joint mask is 0.52 PPL better than the
  original mask;
- the quantization penalty is 9.75 PPL for the original mask and 8.73 PPL for
  the joint mask, a 1.02 PPL reduction;
- the joint nonuniform allocation is worse than uniform 3-bit quantization on
  both depth masks;
- the broad result is therefore attributable to quantization-compatible depth
  mask selection, not to nonuniform bit allocation.

This is evidence of an interaction at seed 0: the joint mask is worse without
quantization but better after uniform all-linear quantization. It should be
described as promising single-seed evidence and repeated before making a
general robustness claim.

## 11. Consolidated TinyLlama evidence table

| Run or condition | Depth scope | Weight scope | WikiText2 PPL | Main interpretation |
|---|---|---|---:|---|
| Dense | none | none | **8.97** | Reference |
| Sparse search | none | 50% avg `q_proj` sparsity | 9.00 | Near-dense sparse result |
| Quant search | none | 3-bit avg `q_proj` | 9.00 | Near-dense quant result |
| Depth seed 0 | 2 attn + 2 MLP | none | 12.02 | Depth reference |
| Depth seed 1 | 2 attn + 2 MLP | none | 12.28 | Depth reference |
| Depth seed 2 | 2 attn + 2 MLP | none | **11.03** | Best depth seed |
| Depth seed 0 + uniform sparse | 2 + 2 | 50% uniform `q_proj` | 11.97 | Compatible |
| Depth seed 0 + searched sparse | 2 + 2 | 50% searched `q_proj` | 11.92 | Compatible |
| Independent depth+quant | 2 + 2 | 3-bit avg `q_proj` | 11.97 | Independent composition |
| Joint seed 0 | 2 + 2 | 3-bit avg `q_proj` | **11.03** | Best joint seed |
| Joint seed 1 | 2 + 2 | 3-bit avg `q_proj` | 12.55 | Worse than depth seed 1 |
| Joint seed 2 | 2 + 2 | 3-bit avg `q_proj` | 11.72 | Worse than depth seed 2 |
| Uniform quant-only | none | 3-bit all transformer projections | 13.62 | Strongest broad quant baseline |
| Searched quant-only | none | 3-bit avg all transformer projections | 14.58 | Worse than uniform |
| Depth + uniform broad quant | 2 + 2 | uniform 3-bit all projections | 21.77 | Strong combined baseline |
| Depth + independent searched broad quant | 2 + 2 | searched 3-bit all projections | 23.81 | Independent composition |
| Joint active-budget broad quant | 2 + 2 | active 3-bit avg all projections | 21.47 | Joint search output |
| Joint broad mask without quant | 2 + 2 | none | 12.52 | Quantization causes most added loss |
| Joint broad mask + uniform quant | 2 + 2 | uniform 3-bit all projections | **21.25** | Best broad combined result |
| Independent mask + joint quant | 2 + 2 | joint 3-bit allocation | 22.14 | Joint allocation is worse than uniform |

## 12. Engineering failures and what was learned

### 12.1 Dependency failures after server restart

Several jobs initially failed with errors such as:

```text
ModuleNotFoundError: No module named 'datasets'
```

These were environment failures. Requirements were installed and the jobs were
rerun under unique retry IDs. The failed rows were preserved for provenance.

### 12.2 Baseline implementation compatibility

Early baseline runs exposed interface problems when replacing Mistral attention
or MLP modules:

```text
ValueError: too many values to unpack
AttributeError: 'tuple' object has no attribute 'dtype'
```

The replacement modules had to preserve the output shape and tuple conventions
expected by the installed Transformers version. Once corrected, the baseline
grid completed.

### 12.3 Non-finite random baseline

One 50% random baseline produced infinite train and WikiText2 perplexity. This
is not an infrastructure error. It is evidence that the randomly selected
structure caused model collapse.

### 12.4 Joint final-state replay

The joint-search code required a correction so the final selected parent was
loaded before final evaluation. Exact seed 0 replay, 11.03 PPL in both cases,
provides evidence that the corrected pipeline now evaluates the saved result.

### 12.5 Tokenizer warning

One evaluation printed:

```text
Token indices sequence length is longer than the specified maximum sequence
length for this model (341468 > 2048)
```

The data loader tokenizes the full WikiText2 text and then slices it into
fixed-length evaluation samples before model inference. The full 341,468-token
sequence is not sent through the model in one forward pass. The completed PPL
result is therefore usable, although the warning should be removed or
suppressed in a future cleanup to avoid confusion.

## 13. Findings supported by the current evidence

### Finding 1: EvoPress depth search is materially better than simple structural heuristics

Supported by four Mistral-7B sparsity levels and multiple random seeds.

### Finding 2: Aggressive depth pruning requires a larger search budget

Supported by the 37.5% Mistral run, where 20 generations reduced PPL from 51.69
to 26.00.

### Finding 3: Depth mask selection is seed-sensitive and non-unique

Supported by Mistral and TinyLlama mask overlap analyses.

### Finding 4: The 16 GB CPU RAM limit, rather than A40 VRAM, blocks the large database workflow

Supported by hardware snapshots, the Mistral sparse-database failure, and
successful smaller-model database runs.

### Finding 5: TinyLlama supports end-to-end sparse and quant search in this environment

Supported by complete database creation, search, final evaluation, and saved
artifacts.

### Finding 6: Depth pruning can be composed with `q_proj` sparsity or quantization

Supported by completed sequential and joint evaluations.

### Finding 7: At the tested point, quantization adds little PPL damage after depth pruning

Supported by the consistent joint quant penalties of 0.11, 0.22, and 0.12 PPL.

### Finding 8: Current evidence does not establish joint-search synergy

Supported by:

- only one of three joint seeds beating its depth-only counterpart;
- nearly identical depth-only and joint mean PPL;
- seed 0 ablations attributing the gain to the depth mask;
- similar performance from joint, independent, and uniform quant allocations
  on the seed 0 joint mask.

### Finding 9: All transformer projections can be included under the current memory limit

The 154-module GPTQ database completed with 7.43 GB peak sampled CPU RAM and
zero missing candidate files.

### Finding 10: Joint search finds a depth mask that is more compatible with broad quantization

The all-linear joint mask with uniform quantization achieved 21.25 PPL,
compared with 21.77 for the original depth mask with uniform quantization. The
joint mask is 0.50 PPL worse without quantization but 0.52 PPL better after
quantization. The nonuniform joint allocation slightly worsens both masks.

## 14. Claims that should not be made yet

The presentation should not claim:

- full-model 50% sparsity for TinyLlama;
- full-model 3-bit quantization, because embeddings and `lm_head` are excluded;
- a measured whole-model size, latency, throughput, or energy reduction;
- statistically significant superiority of joint search;
- that the all-linear joint quant allocation is better than uniform 3-bit
  allocation;
- that three seeds characterize the full search distribution;
- direct runtime speedups across T4 and A40 without hardware normalization;
- that Mistral-7B sparse/quant search is impossible in general;
- that 16 GB host RAM is sufficient for full Mistral-7B database generation.

## 15. Threats to validity and limitations

### 15.1 Weight-compression scope

Sparse experiments still target `q_proj` only. Quantization now covers all
transformer projections, but excludes embeddings, normalization layers, and
`lm_head`. The database stores dequantized FP16 reconstructions, so it does not
measure the disk size or runtime of a packed 3-bit deployment.

### 15.2 Single evaluation dataset

Most conclusions rely on WikiText2 PPL. Downstream tasks, instruction-following
quality, and broader language benchmarks have not yet been evaluated.

### 15.3 Small seed count

Three seeds expose variability but do not provide high statistical power.

### 15.4 Hardware variation

T4 and A40 runs can be compared for quality when all numerical settings match,
but runtime comparisons are confounded.

### 15.5 Requested sparsity rounding

TinyLlama’s 22 layers cause requested depth percentages to round down to
integer module counts.

### 15.6 Search-budget equality

The Mistral main curve uses 10 generations, while the convergence extension
uses 20. Results from different budgets answer different questions and should
not be mixed in one curve without labeling.

### 15.7 Full compression accounting

No final report yet includes:

- total parameter count after structural removal;
- serialized model size;
- effective bits per original model parameter;
- inference latency or throughput;
- peak inference memory.

These measurements are necessary before making systems-level compression
claims.

## 16. Recommended thesis interpretation

The current work supports a thesis direction centered on **multi-method
compression under a shared evolutionary search framework**, with the following
nuanced result:

> Structural pruning is the dominant source of quality variation at the tested
> TinyLlama operating point. `q_proj` sparsity and quantization can be layered
> on top with little additional perplexity damage. Joint search is technically
> feasible and can discover alternative depth masks, but the current
> three-seed narrow-scope study does not show consistent optimization synergy.
> At the broader all-linear scope, joint search discovers a depth mask that is
> more robust to quantization. Its nonuniform bit allocation does not improve
> over uniform 3-bit quantization.

This is still a valuable thesis result. A thesis does not require every
combined method to outperform every single method. It can contribute:

- a correct implementation of joint search;
- controlled comparisons against independent composition;
- attribution through cross-ablations;
- repeatability evidence;
- identification of the dominant decision variable;
- a clear description of when joint optimization is or is not beneficial;
- practical memory limits for reproducing the pipeline.

## 17. Recommended next experiments

### Priority 1: Measure effective compression

For each key TinyLlama model, calculate:

- original parameter count;
- parameters skipped by depth removal;
- number of transformer projection parameters remaining after depth removal;
- average bit width of active quantized projection matrices;
- serialized checkpoint/database-backed representation size;
- a clearly defined effective compression ratio.

Key configurations:

- dense;
- best depth-only;
- sparse-only;
- quant-only;
- independent depth+quant;
- joint depth+quant.

This converts the PPL comparison into a true compression-quality comparison.

### Priority 2: Add inference measurements

On one fixed GPU, preferably the A40, measure:

- peak GPU memory;
- tokens per second;
- latency per generated token;
- warm-up and measurement protocol;
- batch size and sequence length.

Structural depth removal may improve latency, while database replay of
dequantized weights does not provide real kernel-level quantization speedups.
This distinction must be measured rather than assumed.

### Priority 3: Repeat the all-linear mask-compatibility result

The attribution is complete for seed 0. Repeat the broad joint search and the
uniform-quantization mask comparison for seeds 1 and 2. The key paired
comparison is:

```text
original depth mask + uniform all-linear 3-bit
versus
joint-search depth mask + uniform all-linear 3-bit
```

This tests whether the observed 0.52 PPL mask advantage is reproducible.

### Priority 4: Compare joint and independent search with matched budgets

Use:

- identical depth target;
- identical quant target;
- identical generation and offspring budgets;
- at least five seeds if runtime permits;
- paired statistical reporting.

The independent approach should account for the total search compute used by
its separate depth and quant searches.

### Priority 5: Test a stronger combined target

The all-linear 3-bit target is already substantially more challenging. After
the attribution ablations, useful alternatives include:

- requested 12.5% depth plus all-linear 4-bit quantization as a higher-quality
  operating point;
- requested 12.5% depth plus all-linear 2.5-bit average as a stronger
  compression point;
- requested 25% depth plus all-linear 3-bit quantization only if severe quality
  degradation is acceptable.

Start with one diagnostic seed, then repeat only a promising and stable setup.

### Priority 6: Add another evaluation signal

At minimum, add one of:

- another perplexity dataset;
- a small zero-shot benchmark subset;
- task accuracy through the repository’s supported evaluation tools.

This tests whether WikiText2 conclusions generalize.

## 18. Suggested supervisor presentation

### Slide 1: Thesis direction

**Title:** Combining structural and weight-level compression with EvoPress

State the question:

> Can evolutionary search select complementary depth-pruning and
> weight-compression decisions under a fixed quality budget?

### Slide 2: Environment and practical constraint

Show:

- A40 or T4 depending on Datalab availability;
- 16 GB CPU/container RAM;
- full Mistral sparse database failure;
- TinyLlama as the controlled feasibility platform.

### Slide 3: Mistral depth curve

Plot:

- x-axis: requested depth sparsity;
- y-axis: WikiText2 PPL;
- dense reference line.

Main message:

> Quality degrades nonlinearly as structural sparsity increases.

### Slide 4: Why evolutionary search matters

Show EvoPress, random median, and late-layer PPL.

Main message:

> The location of removed modules matters more than the removal count alone.

### Slide 5: Convergence evidence

Show the 20-generation 37.5% curve.

Main message:

> Ten generations were insufficient at aggressive sparsity.

### Slide 6: Small-model sparse and quant feasibility

Show:

| Model state | PPL |
|---|---:|
| Dense | 8.97 |
| 50% avg sparse `q_proj` | 9.00 |
| 3-bit avg quant `q_proj` | 9.00 |
| Uniform 3-bit all transformer projections | 13.62 |
| Searched 3-bit all transformer projections | 14.58 |

Clearly distinguish the early `q_proj` scope from the later all-linear scope.

### Slide 7: Combined-method design

Diagram:

```text
Independent:
depth search -> depth mask
quant search -> bit allocation
combine -> evaluate

Joint:
one population member = depth mask + bit allocation
mutate/select both under shared fitness
```

### Slide 8: Three-seed joint result

Show the paired depth-only versus full-joint table.

Main message:

> Joint search retained the same mean PPL while adding `q_proj` quantization,
> but did not consistently beat depth-only search.

### Slide 9: Ablation and attribution

Show seed 0:

| Configuration | PPL |
|---|---:|
| Joint mask, no quant | 10.92 |
| Joint mask, independent quant | 11.01 |
| Joint mask, joint quant | 11.03 |
| Joint mask, uniform quant | 11.05 |

Main message:

> The seed 0 gain was caused by its depth mask, not a special quant allocation.

### Slide 10: Broad all-linear combined result

Show:

| Configuration | PPL |
|---|---:|
| Depth-only | 12.02 |
| Uniform all-linear quant-only | 13.62 |
| Searched all-linear quant-only | 14.58 |
| Depth + uniform quant | 21.77 |
| Depth + independent searched quant | 23.81 |
| Joint active-budget search | 21.47 |
| Joint mask + uniform quant | **21.25** |

Main message:

> Joint all-linear search beats independent composition, but its advantage over
> uniform composition comes from a more quantization-compatible depth mask,
> not from its nonuniform bit allocation.

### Slide 11: Scientific conclusion and limitations

State:

- broad all-linear compression is feasible under 16 GB CPU RAM;
- active-budget joint search is reproducible;
- the best broad combined result is joint mask + uniform quantization;
- packed 3-bit deployment is not yet implemented;
- evidence is WikiText2-only and the broad result has one seed;
- latency and whole-model compression are not yet measured.

### Slide 12: Next step and supervisor decision

Ask which direction should be prioritized:

1. repeat the broad mask-compatibility result across seeds;
2. measure effective size and inference speed;
3. repeat the broad joint result across seeds;
4. scale combined experiments to a larger model on a higher-RAM node.

## 19. Suggested spoken summary

The following can be used as a concise meeting narrative:

> I first established a reliable Mistral-7B depth-pruning pipeline. EvoPress
> strongly outperformed random and late-layer removal, and a 20-generation
> convergence run showed that aggressive sparsity needs a larger search budget.
> Full Mistral sparse-database generation was blocked by the 16 GB container RAM
> limit, so I moved the combined-method study to TinyLlama.
>
> On TinyLlama, I completed both a SparseGPT database and a GPTQ database for
> attention q-projections. Sparse-only and quant-only searches both retained
> near-dense perplexity. I then evaluated sequential depth+sparsity,
> independently combined depth+quantization, and a true joint evolutionary
> depth+quantization search.
>
> Across three seeds, joint depth+quantization had almost exactly the same mean
> perplexity as depth-only search while adding a 3-bit-average q-projection
> configuration. However, the joint result was better in only one seed. The
> ablations showed that this seed's gain came from a better depth mask rather
> than from the quantization allocation. Therefore, the supported conclusion is
> that the methods are compatible and the additional quantization cost is
> small, but I do not yet have evidence of consistent joint-search synergy.
>
> I then expanded GPTQ from q-projections to all 154 transformer projections.
> The database completed under the 16 GB CPU RAM limit. Uniform all-linear
> 3-bit quantization achieved 13.62 PPL, while the searched quant-only
> allocation achieved 14.58. Combining the original depth mask with uniform
> quantization gave 21.77 PPL and independent searched composition gave 23.81.
> A joint active-budget search achieved 21.47 PPL and replayed exactly. This is
> better than both combined controls. The cross-ablations then showed that the
> joint depth mask with uniform 3-bit quantization improves further to 21.25,
> while the joint nonuniform bit allocation slightly worsens either mask. The
> broad-scope contribution is therefore a quantization-compatible depth mask,
> not a superior mixed-bit allocation.
>
> The next important step is to repeat this mask-compatibility result across
> seeds and measure effective whole-model compression and latency.

## 20. Questions to discuss with the supervisor

1. Is the thesis contribution best framed around achieving maximum compression,
   or around analyzing when joint optimization helps?
2. Should the next experiments broaden compression scope on TinyLlama, or move
   immediately to a larger model with more host RAM?
3. Is WikiText2 sufficient for the next milestone, or should one downstream
   benchmark be added now?
4. Which systems metric is most important: serialized size, peak memory,
   latency, or throughput?
5. Should future joint-search comparisons match generation count or total
   compute budget?
6. Is a negative result about joint synergy acceptable if it is supported by
   controlled ablations and repeatability evidence?

## 21. Artifact index

### Main aggregate files

```text
results/experiment_log.csv
results/depth_pruning_curve.csv
results/depth_pruning_curve.png
results/convergence_37_5.csv
results/convergence_37_5.png
results/baseline_comparison_table.md
results/seed_robustness_table.md
results/small_model_feasibility_summary.md
results/hardware_bottleneck_summary.md
results/hardware_snapshot.txt
```

### Mistral depth runs

```text
results/runs/dense_mistral7b_seed1/
results/runs/depth_mistral7b_s0.125_seed1/
results/runs/depth_mistral7b_s0.25_seed1/
results/runs/depth_mistral7b_s0.375_seed1/
results/runs/depth_mistral7b_s0.50_seed1/
results/runs/depth_mistral7b_s0.375_g20_seed1_retry1/
results/runs/depth_mistral7b_s0.375_seed2/
results/runs/depth_mistral7b_s0.375_seed3/
```

### TinyLlama database and search runs

```text
results/runs/dense_tinyllama_seq1024_seed0/
results/runs/sparse_db_tinyllama_qproj_s0.50_retry1/
results/runs/sparse_search_tinyllama_qproj_s0.50_g20_seed0_retry1/
results/runs/quant_db_tinyllama_qproj_bits234/
results/runs/quant_search_tinyllama_qproj_3bit_g20_seed0/
```

### TinyLlama depth and combined runs

```text
results/runs/depth_tinyllama_s0.125_seed0/
results/runs/depth_tinyllama_s0.125_seed1/
results/runs/depth_tinyllama_s0.125_seed2/
results/runs/depth_tinyllama_s0.25_seed0/
results/runs/combined_tiny_depth0125_sparse50_uniform_seed0/
results/runs/combined_tiny_depth0125_sparse_searchcfg_seed0/
results/runs/combined_tiny_depth025_sparse50_uniform_seed0/
results/runs/combined_tiny_depth025_sparse_searchcfg_seed0/
results/runs/combined_tiny_depth0125_quant3_independent_seed0/
results/runs/joint_tiny_depth0125_quant3_g10_seed0/
results/runs/joint_tiny_depth0125_quant3_g10_seed1/
results/runs/joint_tiny_depth0125_quant3_g10_seed2/
results/runs/replay_joint_tiny_depth0125_quant3_seed0/
results/runs/combined_tiny_depth0125_alllinear_quant3_independent_seed0/
results/runs/joint_tiny_depth0125_alllinear_quant3_active_g10_seed0/
results/runs/replay_joint_tiny_depth0125_alllinear_quant3_seed0/
```

### Joint-search ablations

```text
results/runs/ablation_jointdrop_noquant_seed0/
results/runs/ablation_jointdrop_noquant_seed1/
results/runs/ablation_jointdrop_noquant_seed2/
results/runs/ablation_jointdrop_independentquant_seed0/
results/runs/ablation_independentdrop_jointquant_seed0/
results/runs/ablation_jointdrop_uniformquant3_seed0/
results/runs/ablation_jointdrop_alllinear_noquant_seed0/
results/runs/ablation_jointdrop_alllinear_uniformquant3_seed0/
results/runs/ablation_independentdrop_alllinear_jointquant_seed0/
```

## 22. Final conclusion

The experiments now form a coherent progression:

1. validate depth pruning on Mistral-7B;
2. show that evolutionary selection beats simple baselines;
3. establish convergence and seed variability;
4. document why full large-model database generation is hardware-limited;
5. validate sparse and quant pipelines on TinyLlama;
6. combine depth pruning with sparsity and quantization;
7. implement true joint depth+quant search;
8. replay and ablate the joint result;
9. repeat the joint search across seeds;
10. expand quantization to all 154 transformer projections;
11. enforce the quantization budget over active projections;
12. compare broad joint search with independent and uniform composition;
13. attribute the broad result with both mask/quant cross-ablations;
14. separate method compatibility from unsupported synergy claims.

The most defensible result for the meeting is not that joint search is already
universally superior. It is that a reproducible joint compression pipeline now
exists at both narrow and broad quantization scopes. Broad joint search
outperformed independent composition while maintaining an exact active 3-bit
budget. Cross-ablations show that its depth mask, paired with uniform
quantization, is the best broad combined configuration. The learned nonuniform
bit allocation is not beneficial. This provides a precise, testable thesis
finding: joint search can discover structural masks that tolerate broad
quantization better, even when mixed-bit allocation offers no advantage.
