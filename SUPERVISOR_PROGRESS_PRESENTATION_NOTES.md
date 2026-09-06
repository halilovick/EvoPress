# Supervisor Progress Presentation — Speaker Notes

These notes follow the 22 PDF pages in `SUPERVISOR_PROGRESS_PRESENTATION.tex`: 16 main slides and 6 backup slides. Numerical results are descriptive summaries over seeds 0–2 unless stated otherwise. Lower perplexity (PPL) and lower KL divergence are better.

## Slide 1 — Joint LLM Compression with EvoPress: Algorithm Extensions and Experimental Results

### Main purpose

Open the meeting with a precise scope: this is a progress report about the search algorithm, the extensions tested since the previous meeting, and the decision about what deserves further work.

### What to say

“Since the previous meeting, I concentrated on reconstructing the exact search algorithm and testing the two main open directions: sequential initialization and crossover. I will first separate the EvoPress paper algorithm from the behavior of the official code and from my joint depth-plus-quantization implementation. I will then show the interaction-aware, sequential, and crossover results, and finish with a proposal to stabilize the methodology and prioritize the thesis text.”

### Important technical details

- The deck is about Mistral-7B joint depth pruning plus `q_proj` quantization unless a slide explicitly says otherwise.
- “Current joint algorithm” can mean the established single-parent baseline or one of the optional extensions. State the configuration when it matters.
- The results are not a claim that every extension generalizes to every compression scope or evaluation metric.

### Likely supervisor questions

1. What is the one-sentence thesis contribution at this point?
2. Are these new experiments or a consolidation of existing work?

### Suggested answers

1. “The contribution is an EvoPress-style joint depth-plus-quantization search with explicit active-budget handling, a coordinated interaction-aware mutation operator, and a controlled study of initialization and recombination choices.”
2. “This presentation consolidates experiments already completed and recorded in the repository. No new experiments were run to prepare it.”

### Transition

“To make the progress easy to judge, I will start from the exact tasks we agreed on at the last meeting.”

## Slide 2 — What was agreed at the last meeting

### Main purpose

Show that the work directly responds to the previous meeting rather than adding disconnected experiments.

### What to say

“The main request was to stop treating the implementation as a black box. I needed to explain the exact source-level algorithm, compare it with EvoPress Algorithm 1, and then evaluate two meaningful extensions: sequential initialization and crossover. I completed the audit, wrote implementation-faithful pseudocode, implemented four sequential modes, ran the matched comparisons, and completed a minimal population-based crossover pilot. The meeting decision I want today is whether this is enough algorithmic development for the thesis.”

### Important technical details

- `ALGORITHM_AUDIT.md` distinguishes paper pseudocode, official upstream code, current joint code, and analysis tools.
- Sequential initialization is not one experiment: it has two directions and frozen versus warm semantics.
- Crossover required a population change because crossover is undefined with one persistent parent.
- The existing replay attribution framework is still relevant, but it is post-search analysis rather than a search operator.

### Likely supervisor questions

1. Which item took the most implementation effort?
2. Which item produced the clearest scientific finding?
3. Is the audit still accurate after crossover was added?

### Suggested answers

1. “Quantization-first sequential search required the most constrained logic because a fixed quantization profile sharply restricts legal depth swaps under an exact active budget.”
2. “The clearest finding is that sequential initialization is direction- and operator-dependent. Interaction-aware mutation also remains the most promising operator extension.”
3. “The audit describes the baseline single-parent path. Crossover is documented separately as an optional extension; the deck explicitly separates those configurations.”

### Transition

“Before discussing my extensions, I need to establish the EvoPress search scaffold that they extend.”

## Slide 3 — EvoPress Algorithm 1: a mutation-only (1+λ) search

### Main purpose

Give the supervisor a compact mental model of EvoPress Algorithm 1 and clarify why it is not an ordinary population genetic algorithm.

### What to say

“Algorithm 1 is best understood as a `(1+λ)` evolutionary strategy. It samples initial candidates, selects one persistent parent, creates λ mutated copies of that same parent, and progressively screens them using more calibration tokens and fewer survivors. The current parent is added only at the final expensive selection stage, which implements elitism. The best final survivor becomes the parent of the next generation. Mutation is a compatible level switch: compression is increased for one unit and decreased for another, so the global constraint is preserved. There is no crossover in the paper algorithm.”

### Important technical details

- `(1+λ)` means one incumbent parent plus λ offspring per generation. Only one candidate persists as a parent.
- This differs from a conventional GA that maintains a population of several persistent parents and may combine two parents.
- “Level” is an abstract compression setting. In quantization it can be a bit-width; in pruning it can be a sparsity/depth choice.
- A compatible level switch preserves the constraint by balancing a down move with an up move between compatible units.
- Multi-step selection reduces noise and compute: many children receive a cheap evaluation, fewer receive a more expensive evaluation.
- Final-stage elitism protects the incumbent from being lost merely because it was not in the offspring set.

### Likely supervisor questions

1. Why call it `(1+λ)` if the parent joins only the last selection stage?
2. Is multi-step selection equivalent to successively halving?
3. Does a level switch always modify exactly two genes?
4. Why does EvoPress avoid crossover?

### Suggested answers

1. “Because the effective final competition is the current parent plus selected offspring. The parent is delayed to the last stage to avoid spending early-stage compute on an already known incumbent.”
2. “It is conceptually similar—progressively fewer candidates receive more evaluation—but the survivor counts and token budgets are explicitly configured rather than derived from a fixed halving rule.”
3. “At the conceptual level, a switch balances an increase and a decrease. The practical implementation can apply multiple switches to one child.”
4. “The method is designed as a local budget-preserving search around one incumbent. With one persistent parent, there is no second independently retained genotype to recombine.”

### Transition

“That pseudocode is intentionally clean. The official source adds details that materially affect how candidate neighborhoods and feasibility work.”

## Slide 4 — What the official implementation actually adds

### Main purpose

Prevent an oversimplified paper-versus-code comparison and establish source-level details used in the thesis implementation.

### What to say

“The official implementation realizes Algorithm 1 with several practical choices. The number of switches is sampled as the minimum of two uniform integers, so small local changes are more likely. Duplicate candidates are rejected. Mutation endpoints must satisfy compatibility constraints such as grouping, tensor size, and available reconstructed levels. Candidate budgets are normally preserved constructively by paired moves. In my joint code, a separate repair step becomes necessary only when a structural change changes which quantization genes are active. The parent is inserted and evaluated again at the final stage. Also, the official depth code exposes a population parameter, although the paper/default case is the single-parent search.”

### Important technical details

- If `U1,U2 ~ Uniform{1,…,M}`, `min(U1,U2)` is biased toward smaller numbers. This increases locality without forbidding larger moves.
- Duplicate rejection is genotype equality in the relevant candidate representation.
- File compatibility matters because a move is only executable if the target reconstructed level exists, e.g. a requested `2.pth`, `3.pth`, or `4.pth`.
- Do not attribute active-budget repair to paper Algorithm 1. It is required by the joint active-set semantics.
- The official upstream repository has separate depth, sparsity, and quantization searches; the current `evo_joint_search.py` is the thesis implementation.

### Likely supervisor questions

1. Is the biased switch count described in the paper?
2. Does official EvoPress already support a population?
3. Does the source cache fitness values?
4. Which details are algorithmic versus engineering?

### Suggested answers

1. “The paper permits practical multi-switch behavior, but the exact `min` sampling rule is a source-level implementation choice.”
2. “The official depth search exposes a population-size option. At size one it is the paper-style `(1+λ)` method. My established joint baseline remained size one until the explicit crossover extension.”
3. “Teacher logits and model weight state are cached for efficiency, but candidate fitness is evaluated afresh on the stage minibatch.”
4. “Switch sampling, compatibility, duplicate rejection, and elitism affect the search. Dynamic reconstructed-weight loading and reporting are primarily engineering.”

### Transition

“With that distinction in place, the next step is to show what had to change when the search candidate became joint.”

## Slide 5 — My joint candidate: depth masks plus grouped bit-widths

### Main purpose

Make the joint genotype and its two separate feasibility constraints visually clear.

### What to say

“A candidate is no longer one homogeneous vector. It contains two Boolean structural masks—one for attention and one for MLP—and grouped per-module quantization bit-width assignments. In the main Mistral experiment, the quantization scope contains one `q_proj` gene per decoder layer, with 2-, 3-, or 4-bit reconstructed weights. We enforce an exact number of attention drops and an exact number of MLP drops. Separately, within each compatible quantization group, the active genes must average exactly the target bit-width. Quantization assignments inside dropped components remain stored, but they do not count toward the active budget and their weights are unused in the forward pass.”

### Important technical details

- `drop["attn"][l] = True` means attention in layer `l` is bypassed; similarly for MLP.
- In the 25% Mistral runs, there are 32 decoder layers and exactly 8 dropped attention modules plus 8 dropped MLP modules.
- `quant` is grouped according to repository grouping rules. The studied active-budget runs use `group_rule=size`.
- The exact active constraint for group `g` is `sum_{i in A_g(D)} b_i = |A_g(D)| B_target`.
- With `q_proj` scope, an attention drop deactivates that layer’s `q_proj`. An MLP-only drop does not deactivate an attention projection.
- Two genotypes can have the same active phenotype but differ in inactive stored genes. This matters for duplicate identity.

### Likely supervisor questions

1. Why keep inactive quantization genes?
2. Why have separate attention and MLP drop counts?
3. Is the active target parameter-weighted or a simple average?
4. Does `q_proj` quantization represent much of the full model?

### Suggested answers

1. “It keeps a stable full module mapping and allows a later depth mutation to reactivate a module with a defined assignment. The drawback is that inactive-only differences can create genotype-level diversity without phenotype diversity.”
2. “The implementation treats attention and MLP as independently droppable structural components. Separate counts enforce the requested sparsity for each type.”
3. “Groups are formed by equal size in these runs, so the exact per-group level sum is also the correct parameter-weighted active average within the group.”
4. “No. `q_proj` is deliberately a restricted quantization scope. Claims must not be generalized to all linear layers without evidence.”

### Transition

“This composite representation still uses the EvoPress search scaffold; the baseline mutation and feasibility path is shown next.”

## Slide 6 — Baseline joint search: implementation-faithful view

### Main purpose

Explain the established joint baseline precisely enough to compare all later extensions against it.

### What to say

“The baseline creates joint initial candidates, repairs their active budgets if necessary, and retains one parent. Every child is a deep copy of that parent. Standard mutation chooses either a depth branch or a quantization branch with equal probability. A depth mutation swaps kept and dropped positions while preserving the structural counts; because it may change the active quantized modules, the quantization state is repaired afterward. A quantization mutation performs a compatible level exchange among active genes. No-ops and duplicates are rejected. The resulting children pass through the same progressive selection schedule, with the parent reintroduced only for final elitism.”

### Important technical details

- Deep copying prevents in-place mutation from changing the incumbent.
- Standard mutation does not intentionally modify both components. However, a “depth” proposal can indirectly change quant genes through repair.
- Quant mutation preserves the bit sum by decrementing one active compatible gene and incrementing another.
- All candidates within one selection stage are evaluated on the same sampled minibatch, so their within-stage ranking is meaningful.
- With `population_size=1` and `crossover_probability=0`, the new optional crossover code preserves the legacy baseline behavior, including random-number consumption relevant to reproducibility.

### Likely supervisor questions

1. Is a repaired depth mutation still depth-only?
2. Why reject duplicates instead of evaluating them?
3. Can repair fail?
4. Is the parent’s old fitness reused in the final stage?

### Suggested answers

1. “Conceptually it is initiated by a depth move, but if repair changes quant genes it is not a pure single-component genotype change. Diagnostics count those changes separately.”
2. “A duplicate provides no new search information and would consume evaluation budget. Rejection forces the offspring set to contain unique genotypes.”
3. “Yes, if the exact target cannot be represented with the available active reconstruction levels. Current validation either rejects or raises a clear failure depending on the path.”
4. “No. The parent is evaluated on the same newly sampled final-stage minibatch as the surviving children.”

### Transition

“The scaffold is therefore recognizable as EvoPress, but the representation and feasibility logic are materially more complex.”

## Slide 7 — Main differences from EvoPress Algorithm 1

### Main purpose

Summarize the thesis-specific algorithmic changes without losing the continuity with EvoPress.

### What to say

“The left column is the paper abstraction; the right column is the joint implementation. The key generalization is from one compression-level vector to a product space of structural masks and bit-width assignments. Mutation must navigate both spaces, and feasibility is now the conjunction of exact structural counts and exact active quantization targets. Inactive genes are explicitly represented. Directly budget-preserving moves are still used where possible, but a depth change can alter the active set, so repair becomes part of the proposal. Interaction-aware mutation, sequential initialization, and the optional crossover experiment are my algorithmic extensions. Replay attribution is different: it analyzes saved candidates after search and does not change the search process.”

### Important technical details

- The baseline topology still follows `(1+λ)`: all children come from one current parent and final selection retains one.
- “Current implementation” on this slide includes optional capabilities, not one configuration that enables all of them simultaneously.
- Sequential modes and crossover are intentionally incompatible in the first crossover implementation.
- The official experimental multimodal branch is not the same algorithm: it uses a different representation and sequential depth then quant selection within a generation.

### Likely supervisor questions

1. Which changes are required by joint compression, and which are optional research ideas?
2. Is repair a bias in the search?
3. Does attribution count as an algorithmic contribution?

### Suggested answers

1. “The composite representation, separate constraints, active-set logic, and repair are required for this joint formulation. Interaction-aware mutation, sequential initialization, and crossover are optional policies tested on top.”
2. “Yes, repair defines a randomized feasibility map and can modify genes unrelated to the structural move. It is necessary for exact feasibility but is not neutral.”
3. “It is a methodological analysis contribution. It replays recombined saved components to explain outcomes, but it is not a search operator.”

### Transition

“The first operator designed specifically for the joint representation was interaction-aware mutation.”

## Slide 8 — Interaction-aware mutation coordinates both components

### Main purpose

Explain exactly why the operator is interaction-aware, how it differs from standard mutation, and what evidence supports it.

### What to say

“Standard mutation normally proposes a change on one side of the candidate. Interaction-aware mutation makes a coordinated proposal every time. It first performs the existing count-preserving depth mutation, identifies the decoder layers whose attention or MLP mask changed, repairs the active quantization budget, and then tries a quantization exchange involving one of those touched layers. If no such exchange exists, it falls back to any valid active exchange. This targets the coupling introduced by the active set. In the matched G50 `q_proj` study, it reduced mean WikiText2 PPL from 11.242 to 11.086 and won two of three seeds. The evidence is encouraging but modest, and downstream macro accuracy was essentially tied.”

### Important technical details

- The operator is selected by `joint_mutation_mode=interaction_aware` and requires active budgeting with size grouping.
- “Touched” is decoder-layer scope. It does not require the quant gene to belong to the same subcomponent that was structurally changed.
- If an MLP mask changes in a `q_proj`-only database, an attention `q_proj` in that decoder layer can be preferred even though the MLP does not contain the gene.
- Repair happens before the explicit quant exchange. Diagnostics separate repair changes, preferred exchanges, and fallback exchanges.
- Matched G50 summary: PPL 11.086 vs 11.242; train PPL 10.789 vs 10.760; final KL 0.609 vs 0.613; C4 and FineWeb-Edu replay improved, but the three-task LM-eval macro differed by only +0.001.

### Likely supervisor questions

1. Why not mutate the quant gene inside the exact changed component?
2. Can the explicit quant exchange undo repair?
3. Is the PPL improvement statistically significant?
4. Why call it interaction-aware if it sometimes falls back to any active pair?

### Suggested answers

1. “The current implementation maps interactions at decoder-layer granularity because the database and module naming make that robust. A stricter component-level definition is possible future work, but it was not tested here.”
2. “The exchange preserves the active group sum by decrementing one endpoint and incrementing another, so it should retain the repaired budget even though it changes assignments.”
3. “No significance claim is made with three seeds. It is a descriptive paired improvement with wins on two seeds.”
4. “The primary attempt is touched-layer-biased. The fallback guarantees a feasible joint proposal when the preferred neighborhood is empty; logs reveal how often fallback is used.”

### Transition

“After defining a joint-aware neighborhood, I investigated whether the starting point of that neighborhood also matters.”

## Slide 9 — Sequential initialization: four stage-two variants

### Main purpose

Define the two directions and the crucial distinction between frozen and warm semantics.

### What to say

“Sequential initialization asks whether a solution optimized for one compression component is a better input to stage two. I implemented both directions. In depth-to-quant frozen, the depth masks never change and stage two searches only quantization. In depth-to-joint warm, the imported depth defines the initial parent but both components can later change. Quantization-first has the symmetric warm mode, but the frozen mode is harder: the imported quantization profile cannot be repaired, so a depth swap is legal only if removed and restored components contribute exactly the same bit sum and module count in every group. Frozen means an invariant for the entire run; warm means an exact initialization only.”

### Important technical details

- Frozen behavior is enforced by mutation dispatch, not by mutating both components and overwriting the frozen one.
- The imported component is deep-copied, hashed, and checked after initialization, selection, and final serialization.
- Quantization-first strict initialization finds exact feasible depth masks with bounded memoized search, not random rejection.
- Contribution-compatible swaps compare vectors of group-level bit sums and module counts. Both must match because the active target depends on sum and active cardinality.
- Warm modes use the existing standard or interaction-aware joint mutation after initialization.
- Sequential and crossover modes are rejected together in the current crossover pilot to avoid combining two mechanisms.

### Likely supervisor questions

1. Why is quantization-first frozen more constrained than depth-first frozen?
2. Why require equal module counts as well as equal bit sums?
3. Does warm initialization guarantee the imported component remains useful?
4. How are infeasible strict initial states handled?

### Suggested answers

1. “With frozen depth, quantization exchanges can be chosen to preserve the active budget. With frozen quantization, changing depth changes which fixed bit assignments count, so only a subset of structural swaps is exactly budget-neutral.”
2. “The target sum is `number of active genes × target bits`. Equal raw bit contribution without equal count could still change the required target.”
3. “No. Warm means only that the first parent contains the imported component. Selection is free to move away from it.”
4. “The bounded exact solver either produces the requested number of unique feasible masks or raises a diagnostic that distinguishes infeasibility from search-bound exhaustion.”

### Transition

“With these four definitions, the experiment becomes a question of direction, freezing, and stage-two mutation operator.”

## Slide 10 — Sequential results: direction and operator both matter

### Main purpose

Provide the central visual summary of the sequential study while keeping G20 and G50 budgets explicit.

### What to say

“The left panel compares the five G20 sequential conditions. The bars show three-seed means and sample standard deviations; black dots are individual seeds. The strongest G20 sequential result is depth warm plus standard mutation at 11.526 PPL. Quantization-to-depth frozen is clearly the most unstable. The right panel adds the matched standard-initialization controls and the separately labeled G50 references. The important message is not a universal ranking. The benefit depends on which component supplies the warm start and which mutation operator performs the refinement.”

### Important technical details

- Shared G20 configuration: Mistral-7B-v0.3, 25% attention and MLP drops, `q_proj`, active 3-bit target, size grouping, 20 generations, 16 offspring, seeds 0–2.
- Standard-initialization G20 evaluates 32 initial candidates. Depth-first modes evaluate one imported-depth combined initial candidate. Quantization-first strict modes generate 32 exact feasible depth masks.
- G50 bars are references with 50 stage-two generations and must not be treated as equal-budget entries in the G20 matrix.
- The plot uses PPL only; KL and compute accounting remain in the reports/backups.

### Likely supervisor questions

1. Why does depth frozen look competitive if quantization search adds little?
2. Is the lowest G20 bar also the cheapest method?
3. Why include G50 on the same figure?

### Suggested answers

1. “It begins from a strong seed-matched depth-only result. Its near tie with depth plus uniform or independently searched quantization suggests that stage-two `q_proj` optimization adds little once depth is fixed.”
2. “Not necessarily. End-to-end cost includes the stage-one search that produced the warm start. The backup cost slide separates stage-two and total pipeline cost.”
3. “It shows whether early advantages persist with a larger stage-two budget, but the labels explicitly separate G20 and G50. They are context, not pooled comparisons.”

### Transition

“The figure is easiest to interpret through a small set of matched paired comparisons.”

## Slide 11 — What sequential initialization tells us

### Main purpose

Convert the broad result plot into defensible technical conclusions and matched comparisons.

### What to say

“The best G20 sequential condition is depth warm plus standard mutation. Freezing depth and optimizing `q_proj` changes little relative to target-matched composition. Quantization-first frozen search has a high standard deviation, consistent with its constrained reachable neighborhood. For the quantization warm start, changing only the stage-two operator from standard to interaction-aware improves PPL by 0.727 on average and wins all three paired seeds. Holding interaction-aware mutation fixed, quant warm versus standard initialization is a smaller 0.286 improvement with two wins. Holding standard initialization fixed, interaction-aware versus standard mutation improves by 0.474 with two wins. These are descriptive three-seed results.”

### Important technical details

- Paired deltas are `method PPL − baseline PPL`; negative values favor the method named first.
- Quant warm interaction-aware minus quant warm standard: `-0.727 ± 0.396`, wins 3/3.
- Quant warm interaction-aware minus standard-init interaction-aware: `-0.286 ± 0.918`, wins 2/3.
- Standard-init interaction-aware minus standard-init standard: `-0.474 ± 1.128`, wins 2/3.
- The large SD of some paired deltas means seed interaction is substantial.
- The depth-warm standard result is not proof that depth-first initialization always helps; it is the strongest observed G20 condition in this scope.

### Likely supervisor questions

1. Which comparison best isolates initialization?
2. Which comparison best isolates mutation?
3. Can we say interaction-aware mutation is better?
4. Why is quantization-to-depth frozen so variable despite many legal swaps?

### Suggested answers

1. “Quant-warm interaction-aware versus standard-init interaction-aware at the same G20 schedule changes initialization while holding the operator fixed.”
2. “Quant-warm interaction-aware versus quant-warm standard holds the warm start and stage-two budget fixed while changing the operator.”
3. “We can say it shows a positive descriptive signal in these matched `q_proj` settings. Three seeds and mixed downstream metrics do not support a universal claim.”
4. “Its legal swaps are still a constrained subset defined by identical contribution vectors. Connectivity and the quality of reachable masks can differ strongly by seed even if the raw legal-pair count is nonzero.”

### Transition

“A natural next question is whether the strong depth warm-start result reflects a better final basin or only faster convergence.”

## Slide 12 — Warm start versus a longer search

### Main purpose

Show that warm starts primarily improve early search quality and convergence speed, not the final G50 result.

### What to say

“This figure extends the depth warm-start comparison to 50 stage-two generations under both standard and interaction-aware mutation. Warm-start conditions begin with much better candidates and remain ahead around generation 20. As the standard initialization runs receive more generations, they catch up. Under matched interaction-aware G50, depth warm minus standard initialization is plus 0.086 PPL and wins only one of three seeds. Under standard mutation the corresponding final delta is plus 0.081. Therefore the safe conclusion is that the warm start changes early search quality and convergence speed; it does not clearly move the final G50 solution to a better basin.”

### Important technical details

- Interaction-aware matched G50: depth warm 11.172 ± 0.196 vs standard initialization 11.086 ± 0.174; paired delta `+0.086 ± 0.172`, wins 1/3.
- Standard mutation matched G50: paired warm-start delta `+0.081 ± 0.639`, wins 1/3.
- At 20 completed generations, the warm-start advantage is still large in the convergence report, but this is an intermediate checkpoint, not a separate finalized G20 run with identical initialization accounting.
- The depth-warm stage-two runs initialize from one candidate, while standard initialization evaluates 32 candidates. Candidate-evaluation and token-exposure plots exist to account for this difference.
- Final PPL can fluctuate slightly relative to periodic checkpoints because PPL is an external evaluation, while search selection optimizes sampled KL fitness.

### Likely supervisor questions

1. Why can the final PPL be slightly worse even if search KL decreases?
2. Does this mean warm starts are useless?
3. Is generation count a fair compute axis?
4. Could more than 50 generations reverse the ordering again?

### Suggested answers

1. “Selection optimizes KL on sampled calibration minibatches, while final PPL is measured separately. They are correlated but not identical objectives.”
2. “No. Warm starts are valuable when the stage-two budget is limited or when good early candidates matter. They simply do not show a clear final G50 advantage here.”
3. “It is useful but incomplete because initialization sizes differ. The repository also reports candidate evaluations, token exposure, and runtime.”
4. “It is possible, but the present evidence only supports the observed convergence pattern through G50. Additional long runs would need a specific thesis question to justify them.”

### Transition

“After studying initialization, the remaining requested operator question was whether recombining whole depth and quantization components could be useful.”

## Slide 13 — Why investigate crossover?

### Main purpose

Explain why crossover is conceptually absent from one-parent EvoPress and why the joint representation nevertheless motivates a minimal test.

### What to say

“In an ordinary genetic algorithm, several candidates persist, two parents can be selected, and their genetic material can be recombined. EvoPress instead uses one persistent incumbent and local mutation. With one parent, crossover either combines the candidate with itself or requires a transient child that has not been selected as a persistent parent, so it is not the same mechanism. The joint genotype gives a natural coarse factorization: depth and quantization. The question was whether one good candidate might contain a useful depth mask while another contains a useful quantization assignment. Testing that question requires first retaining a small persistent population.”

### Important technical details

- Crossover is not simply “another mutation” because it imports an intact component selected under another parent context.
- A persistent population changes the algorithm even before any crossover is applied: parent sampling and final-stage elitism now involve multiple incumbents.
- The pilot intentionally avoids one-point, uniform gene-level, blockwise, adaptive, or diversity-preserving crossover variants.
- Parent selection is uniform; there is no tournament or fitness-proportional study.

### Likely supervisor questions

1. Why not crossover two offspring from the same generation?
2. Why use whole components instead of one-point crossover?
3. Does a population necessarily improve exploration?

### Suggested answers

1. “The intended test is recombination between candidates that survived prior selection and therefore represent persistent alternatives. Transient offspring have not passed that criterion.”
2. “The representation has a meaningful semantic split into depth and quantization. Whole-component crossover is the smallest interpretable pilot and avoids a broad crossover design study.”
3. “No. It can preserve several basins, but it also increases evaluation cost and may retain candidates whose components are highly similar. That is why the population change is part of the empirical extension.”

### Transition

“The implemented operator therefore recombines exactly those two semantic components and nothing more.”

## Slide 14 — Minimal component crossover

### Main purpose

Define the crossover operator, repair semantics, and population selection behavior precisely.

### What to say

“Given parents A and B, the child is either the depth mask of A with the quantization assignment of B, or the reverse. The child is a deep copy, so neither parent can be modified. The selected depth component is preserved exactly. Because the imported quantization profile was optimized under a different active set, it may violate the active bit target. The existing repair is therefore applied, followed by depth-count, budget, reconstruction-file, feasibility, and uniqueness checks. If repair changes the quantization assignment, the child is classified as component crossover plus repair. The pilot retains four persistent parents, samples crossover with probability 0.25, and uses unchanged standard mutation otherwise.”

### Important technical details

- Crossover parents are distinct and uniformly selected from the persistent population.
- Mutation offspring choose one uniformly selected parent and use the existing operator without semantic changes.
- Intermediate survivor counts are unchanged. With configured `[8,2,1]` and population size 4, the effective final survivor schedule is `[8,2,4]`.
- All persistent parents are added for final-stage elitism. The next population must contain exactly four unique feasible candidates or the run fails clearly.
- The best population member is the reported best candidate; the population is not collapsed to one during search.
- Crossover combined with any sequential mode is rejected in this implementation.

### Likely supervisor questions

1. Is a repaired child still crossover?
2. Why preserve depth rather than preserve both imported components exactly?
3. Why is the final effective survivor count four?
4. Could two distinct parents produce the same child?

### Suggested answers

1. “It originates from component recombination, but it is explicitly labeled ‘component crossover + repair’ because the final quantization component is no longer unchanged.”
2. “Both components are initially imported exactly, but they may be jointly infeasible. The design choice is to preserve the selected structural mask and repair only quantization because the active budget is defined conditional on depth.”
3. “A persistent population of four must survive to the next generation. Keeping the configured final count of one would immediately collapse the population and defeat crossover.”
4. “Yes. If parents share one component or the recombined pair already exists, distinct parent identities can still yield a duplicate genotype.”

### Transition

“The implementation worked as intended; the main empirical issue was how often recombination actually created a new candidate.”

## Slide 15 — Crossover result: feasible, but little useful novelty

### Main purpose

Present the matched three-seed quality, runtime, and diagnostic outcome without claiming that crossover itself is harmful.

### What to say

“Across seeds 0–2, the population-based crossover extension achieved 12.870 mean PPL versus 12.607 for the mutation-only control, and 0.779 mean KL versus 0.735. It was better on PPL for seeds 0 and 2 but much worse for seed 1, so the mean was worse. Runtime increased by about 14 percent. The more informative mechanism result is that 238 of 301 crossover attempts—79.1 percent—were duplicates. Only 20.9 percent were accepted. None was infeasible, and 25 accepted children required repair. The population still remained four out of four unique candidates every generation. This means the feasibility machinery worked, but the retained parents did not expose many novel cross-component combinations.”

### Important technical details

- Per-seed PPL crossover minus control: seed 0 `-0.71875`, seed 1 `+1.60156`, seed 2 `-0.09375`.
- Per-seed KL deltas: `-0.02148`, `+0.14355`, `+0.00928`.
- Aggregate acceptance: 63/301. Duplicate rate: 238/301. Infeasible: 0/301.
- Repair frequency among accepted crossover children: 25/63 = 39.7%.
- “Duplicate” means matching an existing persistent candidate or already accepted offspring under the implementation’s candidate identity.
- The comparison changes population size from 1 to 4 and crossover probability from 0 to 0.25. A population-4/crossover-0 control was not run.
- Therefore it is incorrect to claim that crossover alone caused the quality or runtime difference.

### Likely supervisor questions

1. Why is the duplicate rate so high?
2. Why were there zero infeasible proposals?
3. Can we conclude crossover is harmful?
4. What experiment would isolate crossover causally?
5. Is two out of three PPL wins a positive result?

### Suggested answers

1. “The population has only four candidates and each child chooses one of two whole-component combinations. If parents share depth or quantization components, or if those combinations already exist, recombination quickly repeats known genotypes. The operator has very low combinatorial granularity.”
2. “Depth components already satisfy exact drop counts, and the existing active-budget repair successfully restored quantization feasibility whenever needed. Reconstruction files were available for the tested 2/3/4-bit moves.”
3. “No. The experiment tests population size four plus crossover probability 0.25 as one extension. It shows no benefit for the full extension in this setting, not that crossover itself is generally harmful.”
4. “Add a matched population-size-four, crossover-probability-zero control while holding final survivor behavior and parent sampling fixed.”
5. “Not by itself. Seed 1’s degradation dominates the mean, KL is mixed, and there are only three seeds. The safe conclusion is inconsistent quality with high duplicate pressure.”

### Transition

“Taken together, these experiments now give a coherent algorithmic story and also indicate where further operator development has diminishing returns.”

## Slide 16 — Current conclusions and next steps

### Main purpose

Recommend a deliberate shift from open-ended operator development to thesis consolidation and narrowly justified remaining evaluation.

### What to say

“The core EvoPress scaffold remains a useful basis for joint compression. The strongest operator contribution is interaction-aware mutation, which explicitly coordinates structural and quantization changes and shows a modest positive perplexity signal. Sequential initialization teaches us that good starting points can accelerate moderate-budget search, but the result depends on direction and operator and largely disappears by G50 for depth warm starts. Component crossover is technically feasible but added little novelty because most recombinations were duplicates. My recommendation is to freeze the methodology, convert the audit and implementation descriptions into the thesis methods chapter, finalize the key tables and figures, and run only evaluations that close a specific evidence gap.”

### Important technical details

- “Most promising” does not mean statistically established or uniformly superior on downstream tasks.
- Sequential cost must be reported both for stage two and for the complete stage-one-plus-stage-two pipeline.
- The crossover result is a negative/neutral pilot in one scope, not a general statement about genetic algorithms.
- Potential remaining evaluation should be chosen based on thesis claims, e.g. whether broader attention scope or downstream validation is necessary, not because another operator is available.
- The replay attribution framework remains useful in the thesis as a post-hoc explanation of component contributions.

### Likely supervisor questions

1. What should be written first?
2. Is one more crossover control necessary?
3. Which result belongs in the abstract-level contribution?
4. What experiment would add the most value if time permits?

### Suggested answers

1. “The methodology chapter: EvoPress scaffold, joint representation, active-budget constraint and repair, standard and interaction-aware mutation, then sequential and crossover variants.”
2. “Only if the thesis needs a causal claim about crossover itself. For the current pilot conclusion—that the tested population-based extension did not help—it is not necessary.”
3. “The joint search formulation with active-budget handling and the interaction-aware proposal mechanism are the clearest algorithmic contributions. Sequential results support the analysis of search behavior.”
4. “A carefully scoped evaluation that tests the intended final claim—probably broader quantization scope or a selected downstream check—would add more value than another operator sweep.”

### Transition

“The remaining slides are backup material for implementation, feasibility, cost, and result-detail questions.”

## Slide 17 — Backup: full current search pseudocode

### Main purpose

Provide an implementation-level overview that includes both the legacy single-parent reduction and the optional population/crossover path.

### What to say

“This is the complete control flow at a higher level. After configuration validation, the search creates feasible initial candidates and retains the requested persistent population. For each child, the code either samples two distinct parents for component crossover or one parent for an existing mutation. It repairs and validates as needed and rejects infeasible, no-op, or duplicate candidates. Intermediate selection stages keep their configured counts. At the final stage all persistent parents are added and exactly `population_size` unique survivors are retained. With size one and crossover disabled, the path reduces to the established joint `(1+λ)` behavior.”

### Important technical details

- The real implementation also records per-generation mutation and crossover diagnostics, effective selection schedule, and best population fitness.
- Sequential modes use their own initialization/mutation dispatch and reject the crossover configuration.
- The initial population is selected by the existing initial-selection mechanism; crossover does not invent a new fitness function.
- Failure to retain the requested number of unique feasible candidates is explicit rather than silently shrinking the population.

### Likely supervisor questions

1. Does every offspring come from the best parent?
2. What defines the reported final result for a population?
3. Are intermediate survivors allowed to reproduce immediately?

### Suggested answers

1. “In the population extension, parents are uniformly sampled from all persistent population members. In the legacy path there is only one parent.”
2. “The population persists through search, and the best population member is reported as the best candidate at the end.”
3. “No. Reproduction is from the persistent population at the generation start. Intermediate survivors are part of selection, not new persistent parents until the generation finishes.”

### Transition

“The central feasibility operation referenced in this pseudocode is active-budget repair.”

## Slide 18 — Backup: active-budget repair

### Main purpose

Explain mathematically why structural mutation and component recombination can violate the quantization constraint and what the repair guarantees.

### What to say

“The active cost in group g is the sum of bit-widths over modules active under depth state D. Exact feasibility requires this sum to equal the active count times the target bit-width. If a depth mutation drops or restores a quantized module, the active set changes. If crossover imports a quantization profile from a different depth mask, the bit sum and active count may also become incompatible. Repair preserves the chosen depth mask, then moves active quantization genes up or down through available reconstruction levels until the exact target sum is restored. If that cannot be done, the candidate must not enter selection.”

### Important technical details

- Exact representability is checked using an integer target sum; sequential strict modes use rational arithmetic to avoid floating-point ambiguity.
- Repair operates independently per compatible quantization group.
- It chooses eligible genes randomly; it is not guaranteed to minimize the number of changes or preserve locality.
- Empty active groups are ignored, matching implementation behavior.
- For component crossover, repair changes are counted and the child type is relabeled accordingly.
- For quantization-first frozen mode, repair is forbidden because it would violate the frozen imported profile; legal structural moves must be contribution-compatible instead.

### Likely supervisor questions

1. Does repair preserve the original quantization component?
2. Is repair deterministic?
3. Could repair change a gene far from the touched layer?
4. Why not repair depth instead?

### Suggested answers

1. “It preserves it as the initial recombination but may change active genes to restore feasibility. That is why repaired crossover is labeled separately.”
2. “No. It samples among eligible adjacent level changes, subject to the exact target and available files.”
3. “Yes. The current helper can alter any eligible active gene in the deficient or excess group; this is a known locality limitation.”
4. “The design treats the selected depth mask as the structural component to preserve and quantization as the conditional budget to adapt. Repairing depth would no longer preserve the chosen component crossover or structural mutation.”

### Transition

“With that constraint in mind, the full sequential table shows how differently the four directions behave.”

## Slide 19 — Backup: full sequential quality table

### Main purpose

Provide all major sequential conditions, matched controls, and separately labeled G50 references in one place.

### What to say

“This table is the numeric version of the sequential plot. The first two rows are target-matched composition baselines. The next two are standard-initialization G20 controls. The five G20 sequential rows show the four modes, with both standard and interaction-aware operators for quant warm. The final three rows are G50 references. The most important guardrail is the generation column: the G50 rows should not be directly ranked as equal-budget competitors with G20.”

### Important technical details

- Depth warm + standard G20 is 11.526 ± 0.223.
- Quant warm + interaction-aware G20 is 11.846 ± 0.532.
- Standard-init interaction-aware G20 is 12.133 ± 0.508.
- Quant-to-depth frozen has the highest variability at 14.003 ± 2.748.
- Interaction-aware joint G50 is 11.086 ± 0.174, while matched depth-warm interaction-aware G50 is 11.172 ± 0.196.
- Composition baselines are not evolutionary stage-two searches, so their generation entry is not comparable in the same way.

### Likely supervisor questions

1. Which row is the fairest baseline for each sequential method?
2. Why is standard-init standard G20 worse than some composition baselines?
3. Should the G50 interaction-aware result replace the G20 control?

### Suggested answers

1. “It depends on the question. Use target-matched composition for end-result composition, standard-init G20 for a matched stage-two search schedule, and same-operator controls to isolate initialization.”
2. “With only 20 generations, a random joint initialization may not catch up to a strong pre-optimized depth solution. That is exactly the convergence-speed question sequential initialization tests.”
3. “No. It answers a longer-budget question. The G20 interaction-aware control is the correct matched baseline for quant-warm interaction-aware G20.”

### Transition

“Quality rankings alone are insufficient because a sequential pipeline includes the cost of creating its stage-one candidate.”

## Slide 20 — Backup: sequential cost accounting

### Main purpose

Clarify the difference between stage-two search cost and end-to-end pipeline cost.

### What to say

“All four rows use the same G20 stage-two generation, offspring, and selection schedule, so each stage-two search evaluates 572 candidates. A quant warm start, however, first requires a quantization-only search. Adding that stage doubles the nominal candidate-evaluation count to 1,144 and increases end-to-end wall time. Stage-two cost is appropriate when asking how initialization changes the same refinement procedure. End-to-end cost is appropriate when comparing complete methods. Both should be reported.”

### Important technical details

- Candidate evaluations count fitness evaluations across initialization and progressive selection, including final-stage parent evaluation.
- Token exposure multiplies evaluations by the token budget of their selection stage; it excludes final PPL evaluation.
- Standard-init G20 stage-two evals: 572 and 999,424 token exposures.
- Quant-warm end-to-end evals: 1,144 and 1,998,848 token exposures after adding stage one.
- Depth-first warm modes differ because stage two evaluates one initial candidate rather than 32; their full accounting is in the report.
- Runtime comparisons are approximate because DataLab sessions/hardware conditions vary.

### Likely supervisor questions

1. Which cost number should go in the thesis main table?
2. Is wall-clock runtime or token exposure the better comparison?
3. Were stage-one results reused?

### Suggested answers

1. “Report stage-two and end-to-end separately. The main methods table should include end-to-end cost, while the operator analysis can use matched stage-two accounting.”
2. “Token exposure and candidate evaluations are more reproducible search-cost measures. Runtime is still useful operationally but should be labeled approximate.”
3. “Yes. The experiments reused already computed stage-one artifacts, but end-to-end accounting adds their recorded cost to represent the complete pipeline.”

### Transition

“The crossover study also benefits from diagnostics beyond final PPL and KL.”

## Slide 21 — Backup: crossover diagnostics by seed

### Main purpose

Expose the seed-level mechanism and quality outcomes behind the aggregate crossover result.

### What to say

“Seed 0 is the positive crossover case, seed 1 is the clear negative case, and seed 2 has a small PPL gain but slightly worse KL. The operational pattern is more stable: every seed has a high duplicate rate, acceptance ranges from about 16 to 27 percent, and no proposal is infeasible. Repair frequency varies substantially by seed. This supports a mechanism conclusion—whole-component recombination often reproduces known candidates—but not a universal quality conclusion.”

### Important technical details

- Seed 0: 22/107 accepted, 85 duplicates, 10 repaired, PPL 12.664 vs 13.383.
- Seed 1: 25/93 accepted, 68 duplicates, 13 repaired, PPL 13.883 vs 12.281.
- Seed 2: 16/101 accepted, 85 duplicates, 2 repaired, PPL 12.062 vs 12.156.
- All persistent populations remained exactly 4/4 unique after selection.
- Repair changed one quantization gene per accepted repaired child in these runs.
- Duplicate rejection increased the number of proposal attempts needed to fill each offspring pool.

### Likely supervisor questions

1. Why did seed 1 accept more crossover children but perform worse?
2. Could duplicate rejection itself harm crossover?
3. Would gene-level crossover reduce duplicates?
4. Why not run a crossover probability sweep?

### Suggested answers

1. “Acceptance only measures novelty and feasibility, not quality. Seed 1’s accepted recombinations were not selected into a better final basin.”
2. “It prevents wasted evaluations of identical genotypes and preserves population uniqueness. The high rejection rate reveals limited operator novelty rather than causing the underlying duplicates.”
3. “Probably it would enlarge the combinatorial neighborhood, but it would introduce a different, much broader operator study and more feasibility/repair complexity.”
4. “The task was a minimal thesis pilot. Given no clear benefit and high duplicate pressure, a sweep has lower priority than consolidating the thesis.”

### Transition

“The final backup slide summarizes the boundaries that should accompany every claim from this work.”

## Slide 22 — Backup: evidence boundaries and limitations

### Main purpose

Make the safe claim boundary explicit and prepare concise answers to overgeneralization questions.

### What to say

“The main sequential and crossover evidence is limited to three seeds, Mistral-7B, WikiText2, 25 percent separate attention and MLP sparsity, and an active 3-bit `q_proj` scope. G20 and G50 are different budgets. Compression values are theoretical weight accounting, not packed checkpoint size or measured latency. Runtime spans restarted DataLab sessions. Therefore I report descriptive means, sample standard deviations, paired deltas, and wins, without significance or universal claims. I also distinguish search interventions from analysis: sequential initialization and crossover change candidate generation, while replay attribution only recombines and evaluates saved candidates after search.”

### Important technical details

- The interaction-aware study has some C4, FineWeb-Edu, and three-task LM-eval replay evidence, but the sequential and crossover conclusions are principally WikiText2 `q_proj` results.
- “Theoretical compression” counts low-bit weights and zero-bit dropped weights; search constraints do not directly optimize deployment latency.
- No result supports saying sequential initialization always helps.
- No result isolates crossover from population size.
- Three seeds are useful for paired descriptive trends but too few for robust significance or tail-risk claims.
- Result artifacts and run summaries preserve commands, seeds, compression constraints, final metrics, and diagnostics for reproducibility.

### Likely supervisor questions

1. What is the strongest claim that is safe?
2. What is the most important missing generalization test?
3. Why use three seeds at all if significance is impossible?
4. Is the thesis weakened by a negative crossover result?

### Suggested answers

1. “The joint EvoPress scaffold was extended with exact active-budget handling and a coordinated mutation operator; in the tested Mistral `q_proj` setting, interaction-aware mutation shows a modest perplexity signal, sequential warm starts mainly improve early search, and the tested population-crossover extension adds little novelty.”
2. “A broader quantization scope or carefully selected downstream evaluation would test whether the observed `q_proj` behavior generalizes, but it should be chosen according to the final thesis claim.”
3. “Three paired seeds reveal large seed dependence and prevent presenting a single favorable run. They support descriptive direction and variance estimates even though they do not justify significance claims.”
4. “No. It documents a motivated extension, verifies feasibility, and explains why it was not beneficial. A well-bounded negative result strengthens the methodological story and supports stopping operator expansion.”

### Transition

“These limitations define the scope of the thesis conclusions and support the recommendation to prioritize a precise written methodology over additional open-ended search variants.”

## Repository evidence used

- `README.md` and the repository-local EvoPress paper reference (`arXiv:2410.14649`)
- `ALGORITHM_AUDIT.md`
- `SEQUENTIAL_SEARCH_IMPLEMENTATION.md`
- `SEQUENTIAL_EXPERIMENT_STATUS.md`
- `results/sequential_search_comparison.md`
- `results/sequential_search_summary.csv`
- `results/sequential_search_paired_deltas.csv`
- `results/depth_warmstart_g50_comparison.md`
- `docs/interaction_aware_joint_mutation.md`
- `results/interaction_aware_mistral_qproj_summary.md`
- `results/interaction_aware_mistral_qproj_summary.csv`
- `results/attribution_framework_report.md`
- `evo_joint_attribution.py`
- `CROSSOVER_IMPLEMENTATION.md`
- `CROSSOVER_EXPERIMENT_HANDOFF.md`
- committed `results/runs/**/run_summary.json` and `generation_log.csv` artifacts for the reported comparisons
- `evopress_supervisor_meeting_presentation-3.tex` as the latest visual/style reference
