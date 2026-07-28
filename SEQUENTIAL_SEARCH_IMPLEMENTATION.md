# Sequential Search Implementation

## 1. Purpose

Sequential initialization tests whether a candidate optimized by one compression
method is a better starting point for a second method or for joint refinement.
The implementation extends `evo_joint_search.py` without changing its
single-parent EvoPress selection loop, fitness functions, final-stage elitism,
or default behavior.

The stage-two candidate remains:

```python
{
    "drop": {
        "attn": [bool, ...],
        "mlp": [bool, ...],
    },
    "quant": [[int, ...], ...],
}
```

No crossover, alternating optimization, population-based search, or new fitness
function is introduced.

## 2. Modes

### `depth_to_quant_frozen`

- Imports a `depth_only` structured candidate.
- Initializes quantization with the normal joint-search initializer.
- Applies one initial active-budget repair when active budgeting is enabled.
- Evaluates exactly that one combined initial candidate.
- Freezes the imported depth masks.
- Generates only quantization offspring.
- Never invokes depth, interaction-aware, or old joint-aware mutation.

The imported depth hash is checked after initialization, after every proposed
offspring, after every selection stage, and before final serialization.

### `depth_to_joint_warm`

- Imports a `depth_only` candidate.
- Initializes and, when requested, repairs quantization.
- Evaluates exactly that loaded-depth combined candidate.
- Verifies that the first parent contains the imported masks.
- Uses the existing standard, interaction-aware, or old joint-aware mutation
  policy after generation 1 begins.

The depth state is a warm start, not a constraint after initial selection.

### `quant_to_depth_frozen`

- Imports a `quant_only` candidate by module name.
- Uses an exact bounded search to generate the requested number of feasible
  depth masks.
- Selects the best combined initial candidate with the normal initial
  selection.
- Freezes the imported quantization profile.
- Generates only contribution-compatible depth swaps.
- Never invokes quantization mutation or quantization repair.

Active budgeting and `group_rule=size` are mandatory.

### `quant_to_joint_warm`

- Imports a `quant_only` candidate by module name.
- With the default `strict` policy, generates exact feasible depth masks while
  retaining the imported quantization profile in every initial candidate.
- Selects the best combined candidate with normal initial selection.
- Verifies that the selected initial parent still contains the imported
  profile.
- Allows the existing joint mutation and repair behavior once generation 1
  begins.

The optional `repair` initialization policy is allowed only in this mode. It
generates ordinary depth masks and repairs a deep copy of the imported profile.
The selected initial parent's changed module names are recorded in
`run_summary.json`.

## 3. Frozen and Warm Semantics

“Frozen” is enforced by mutation dispatch: the forbidden operator is never
called. The code does not mutate both components and overwrite one afterward.
The frozen component is compared structurally with a deep-copied imported
component throughout the run. An invariant failure raises an error.

“Warm” means exact only through initial candidate generation and initial
selection. Normal mutation is permitted after the generation loop begins.

The four state transitions are:

| Mode | Imported component | Initial candidates | Stage-two mutation |
|---|---|---|---|
| `depth_to_quant_frozen` | depth | one loaded-depth candidate | quantization only |
| `depth_to_joint_warm` | depth | one loaded-depth candidate | existing joint policy |
| `quant_to_depth_frozen` | quantization | exact feasible depth candidates | fixed-quant depth only |
| `quant_to_joint_warm` | quantization | exact feasible depth candidates by default | existing joint policy |

## 4. Candidate Loading

The CLI accepts exactly one of:

```text
--stage1_run_dir PATH
--stage1_candidate PATH
```

For a run directory, `final_candidate.json` is preferred locally. The
`run_summary.json` artifact entry is consulted when necessary, with a local
basename fallback for synced result directories whose summary contains a
remote output path.

Only current structured artifacts are imported:

- depth: `candidate_type=depth_only`, `attention_mask`, `mlp_mask`, and
  `candidate_vector_raw`;
- quantization: `candidate_type=quant_only` and `bitwidth_by_module`.

Quantization is always reconstructed from `bitwidth_by_module`, never from raw
list position. Imports validate:

- source search type;
- model identity when a summary is present;
- decoder-layer count;
- depth drop counts;
- whole-block compatibility;
- target grouping rule when recorded;
- target bit-width when recorded;
- exact missing and extra quantization module sets;
- duplicate JSON keys;
- integer bit-width values;
- availability of every imported `<bit>.pth` reconstruction.

The imported component is deep-copied and hashed as canonical sorted JSON with
SHA-256. Legacy text configurations are not imported because they do not
reliably contain all validation metadata.

## 5. Strict Active-Budget Behavior

For group \(g\), the exact active constraint is:

\[
\sum_{i \in \mathrm{active}(g)} b_i
=
|\mathrm{active}(g)| B_{\mathrm{target}}.
\]

Validation uses integer bit sums and `Fraction(str(target_bitwidth))`; it does
not use a floating-point compression-ratio tolerance. Empty active groups are
ignored, matching the existing repair behavior.

The strict quantization-first modes require active budgeting and size grouping.
The reporting-only estimated model compression ratio remains unrelated to
feasibility.

## 6. Contribution-Compatible Depth Swaps

For every decoder-layer attention or MLP component, the implementation computes
two vectors over quantization groups:

```text
level_sums(component)[g]
    = sum(bit-widths of searched modules in component and group g)

module_counts(component)[g]
    = number of searched modules in component and group g
```

A frozen-quantization swap may remove a kept component and restore a dropped
component only when both vectors are equal. Equality of the level-sum vector is
the requested contribution-vector condition. Equality of module counts is also
required because otherwise the target active sum would change even when the
raw bit sum did not.

Whole-block mode combines the attention and MLP vectors at each decoder layer
and swaps one shared mask position.

Every proposed swap is followed by exact checks for:

- unchanged attention and MLP drop counts;
- an unchanged quantization object;
- exact active group sums.

If no legal pair exists, the operator returns a controlled no-mutation result.
The offspring loop retries up to `--max_offspring_attempts` and then raises a
diagnostic containing requested/generated counts, no-ops, duplicates, the
sequential mode, frozen component, and legal-swap count.

## 7. Exact Feasible Initial Depth Masks

The strict quantization-first initializer does not use random rejection.

Let \(B=p/q\) be the exact rational target. For each structural component and
group it forms a scaled deviation:

\[
\Delta_{c,g}=q\sum_{i\in c,g}b_i-p|c,g|.
\]

The full profile deviation is:

\[
\Delta_{\mathrm{full},g}
=q\sum_{i\in g}b_i-p|g|.
\]

A dropped subset is feasible exactly when:

```text
number of dropped attention components = requested drop count
number of dropped MLP components       = requested drop count
sum of removed deviations              = full-profile deviation
```

Whole-block mode instead selects exactly one shared subset of the requested
size using combined attention-plus-MLP deviations.

The solver is bounded memoized backtracking. It prunes impossible remaining
drop counts and memoizes feasibility states. It first samples reproducible
random paths through that exact feasibility oracle to avoid lexicographically
clustered masks, then uses deterministic exact enumeration as a fallback. It
stops after `initially_generated` unique candidates. The number of explored
states is bounded by `--max_initialization_attempts`. Exhaustion and
infeasibility produce different clear errors. Diagnostics include the requested
drop count, target, required removed-deviation vector, and unique attainable
component deviations.

## 8. Implementation-Faithful Pseudocode

### 8.1 Depth → Quantization, Frozen

```text
load imported_depth from depth_only final_candidate.json
validate model, layer count, drop count, and whole-block setting

quant ← normal_joint_quant_initialization(target_bitwidth)
if active_quant_budget:
    quant ← repair_active_quant_budget(quant, imported_depth)

initial_candidates ← [{drop: imported_depth, quant: quant}]
parent ← initial_selection(initial_candidates)
assert parent.drop == imported_depth

for each generation:
    offspring ← bounded_unique_generation:
        child ← deepcopy(parent)
        child.quant ← existing_quant_mutation(child.quant, active genes)
        assert child.drop == imported_depth
        validate depth counts and active budget

    parent ← existing_multi_step_selection_with_final_elitism(offspring, parent)
    assert parent.drop == imported_depth

assert final parent.drop == imported_depth
save candidate, summary, hashes, and invariant results
```

### 8.2 Depth → Joint, Warm

```text
load and validate imported_depth
quant ← normal initialization plus optional initial active-budget repair

initial_candidates ← [{drop: imported_depth, quant: quant}]
parent ← initial_selection(initial_candidates)
assert parent.drop == imported_depth

for each generation:
    use existing selected joint mutation policy
    validate sequential candidate feasibility
    parent ← unchanged multi-step selection with final elitism

save final joint candidate and sequential provenance
```

### 8.3 Quantization → Depth, Frozen

```text
load imported_quant by exact module-name mapping
validate scope, grouping, files, target, model, and layer count

depth_states ← exact_bounded_feasible_subset_search(
    quant=imported_quant,
    exact attention/MLP drop counts,
    exact active per-group target sums)

initial_candidates ← [
    {drop: state, quant: deepcopy(imported_quant)}
    for state in depth_states
]
parent ← initial_selection(initial_candidates)
assert parent.quant == imported_quant

for each generation:
    offspring ← bounded_unique_generation:
        child ← deepcopy(parent)
        legal_pairs ← equal contribution-and-count vector swaps
        if no legal pair:
            record controlled no-op and retry
        child.drop ← apply random legal pair(s)
        assert child.quant == imported_quant
        validate depth counts and exact active budget

    parent ← unchanged multi-step selection with final elitism
    assert parent.quant == imported_quant

assert final parent.quant == imported_quant
save candidate, legal-neighborhood summary, and invariant results
```

### 8.4 Quantization → Joint, Warm

```text
load and validate imported_quant by module name

if policy == strict:
    depth_states ← exact bounded feasible subset search
    initial_candidates ← [
        {drop: state, quant: deepcopy(imported_quant)}
        for state in depth_states
    ]
else:  # repair, allowed only in this mode
    initial_candidates ← bounded unique random depth masks
    for candidate:
        candidate.quant ← repair(deepcopy(imported_quant), candidate.drop)

parent ← initial_selection(initial_candidates)
if policy == strict:
    assert parent.quant == imported_quant
record repaired gene names when policy == repair

for each generation:
    use existing selected joint mutation and repair policy
    validate candidate feasibility
    parent ← unchanged multi-step selection with final elitism

save final joint candidate and sequential provenance
```

## 9. CLI and Examples

New Python options:

```text
--sequential_mode {
  none,
  depth_to_quant_frozen,
  depth_to_joint_warm,
  quant_to_depth_frozen,
  quant_to_joint_warm
}
--stage1_run_dir PATH
--stage1_candidate PATH
--sequential_quant_initialization_policy strict|repair
--max_initialization_attempts N
--max_offspring_attempts N
```

The wrapper delegates to the existing logged launcher. Existing experiment
settings are supplied through the same environment variables.

### Depth → Quantization, Frozen

```bash
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
GROUP_RULE=size ACTIVE_QUANT_BUDGET=1 \
scripts/run_sequential_search.sh \
  --mode depth_to_quant_frozen \
  --stage1-run-dir results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0 \
  --output-dir outputs/experiments/sequential_depth_to_quant_seed0
```

### Depth → Joint, Warm

```bash
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
GROUP_RULE=size ACTIVE_QUANT_BUDGET=1 \
JOINT_MUTATION_MODE=interaction_aware \
scripts/run_sequential_search.sh \
  --mode depth_to_joint_warm \
  --stage1-run-dir results/runs/thesis_medium_depth_mistral_s0.25_g20_o16_seed0 \
  --output-dir outputs/experiments/sequential_depth_to_joint_seed0
```

### Quantization → Depth, Frozen

```bash
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_qproj_debug_bits234/quant_db/Mistral-7B-v0.3/3bit \
GROUP_RULE=size ACTIVE_QUANT_BUDGET=1 \
scripts/run_sequential_search.sh \
  --mode quant_to_depth_frozen \
  --stage1-run-dir results/runs/thesis_medium_quant_mistral_qproj3.0_g20_o16_seed0 \
  --output-dir outputs/experiments/sequential_quant_to_depth_seed0
```

### Quantization → Joint, Warm

```bash
MODEL=mistralai/Mistral-7B-v0.3 \
QUANT_WEIGHTS_PATH=outputs/experiments/quant_db_mistral_attention_bits234/quant_db/Mistral-7B-v0.3/3bit \
GROUP_RULE=size ACTIVE_QUANT_BUDGET=1 \
JOINT_MUTATION_MODE=interaction_aware \
scripts/run_sequential_search.sh \
  --mode quant_to_joint_warm \
  --stage1-run-dir results/runs/thesis_attention_quant_mistral_attention3.0_g20_o16_seed0 \
  --output-dir outputs/experiments/sequential_attention_quant_to_joint_seed0
```

The stage-one quantization scope must exactly match the selected stage-two
database. A q-projection-only stage-one candidate cannot initialize a
full-attention stage-two database.

## 10. Outputs and Metadata

The normal joint artifacts remain unchanged:

- `command.sh`;
- `run.log`;
- `generation_log.csv`;
- `joint_drop_config.txt`;
- `joint_quant_config.txt`;
- `joint_config.json`;
- `final_candidate.json`;
- `run_summary.json`.

The summary adds:

- `sequential_mode`, `sequential_direction`, and `sequential_variant`;
- stage-one run/candidate path, component hash, and search type;
- `frozen_component`;
- quant initialization policy and repaired gene identities;
- initial feasible candidate count;
- initial and final fixed-quant legal-swap counts;
- final depth-count, active-budget, frozen-depth, and frozen-quant checks.

Generation `mutation_summary` contains the sequential mode, parent-before and
parent-after SHA-256 hashes, accepted mutation categories, retry diagnostics,
and fixed-quant legal-neighborhood counts. This explicitly distinguishes
generation-start and generation-end parent state.

## 11. Tests

The focused CPU tests cover:

- structured depth and quant candidate adapters;
- module-name mapping and compatibility failures;
- deep-copy behavior;
- exact feasible-mask generation;
- legal and illegal contribution-compatible swaps;
- infeasible masks and empty mutation neighborhoods;
- all four sequential mode semantics;
- frozen invariant and exact active-budget checks;
- CLI conflict validation;
- wrapper command forwarding;
- structured sequential summary fields;
- existing standard, old joint-aware, and interaction-aware mutation tests.

See `tests/test_sequential_search.py`,
`tests/test_run_sequential_search.py`, and `tests/test_run_reporting.py`.

Verification executed during implementation:

```text
python -m unittest \
  tests.test_sequential_search \
  tests.test_joint_aware_mutation \
  tests.test_run_sequential_search \
  tests.test_run_joint_search_tiny \
  tests.test_run_reporting -q
→ 44 tests passed

ruff check <modified Python files>
→ passed

python -m py_compile <modified Python files>
bash -n scripts/run_joint_search_tiny.sh scripts/run_sequential_search.sh
git diff --check
→ passed
```

The four CPU toy modes generate and reload their sequential summary metadata in
`test_toy_summary_metadata_for_all_four_modes`. A separate CPU stress check
used the saved 32-layer Mistral full-attention quantization profile and produced
32 requested exact feasible candidates with 32 distinct attention masks and 32
distinct MLP masks under the default state bound.

The full practical test discovery run completed 119 tests: 112 passed and 7
unrelated launcher tests failed because optional runtime packages (`datasets`,
`accelerate`, and `sentencepiece`) are absent in this environment or because
pre-existing completed result directories caused dry-run matrices to skip
commands expected by those tests. All sequential, joint mutation, joint
launcher, and reporting regression tests passed.

## 12. Limitations

`quant_to_depth_frozen` may have a smaller depth-search neighborhood than
ordinary depth search because only swaps preserving the active quantization
contribution vector are legal. This is an intentional consequence of freezing
quantization while maintaining an exact active budget.

Additional limitations:

- legacy raw text candidates are rejected rather than imported ambiguously;
- source metadata that is absent from a direct standalone candidate cannot be
  validated, although layer counts, module scope, assignments, and files still
  are;
- exact backtracking is deliberately bounded and may require increasing
  `--max_initialization_attempts` for unusually large or diverse group spaces;
- requesting more distinct feasible initial masks than exist fails explicitly;
- no full Mistral/TinyLlama run is performed by the test suite.
