# Stage 3 Tuning Space

This file defines the current Stage 3 tuning space for the current Stage 2 merged model.
It is the strategy document for Stage 3.

Companion files:

- [STAGE3_WORKFLOW.md](/G:/RelLLM-2/Rel-LLM/STAGE3_WORKFLOW.md)
- [STAGE3_PROGRAM.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_PROGRAM.md)
- [STAGE3_TUNING_SPACE_ZH.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_TUNING_SPACE_ZH.md)
- [STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md)
- [STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md)
- [experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)
- [baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)
- [pipeline_config.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/pipeline_config.json)

## 1. Goal and Scope

Stage 3 is a finetune-only research program.

It is not allowed to:

- enter pretrain
- use `--pretrain`
- switch to a new training stage
- optimize one representative task in isolation
- commit or push exploratory changes unless there is a real global gain

Stage 3 is allowed to:

- modify the finetune path
- add small finetune-time auxiliary objectives
- adjust basis construction and injection
- adjust sampling and token consumption
- make engineering changes that materially improve the representative-task bundle

The optimization target is:

- non-regressive behavior across:
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- plus at least one credible representative-task improvement

Anything else is a failed candidate.

Repository-result confirmation rule:

- a subset-screened gain is not yet a repository-facing result
- if a candidate is going to be kept as a real Stage 3 outcome or used to justify a
  commit / push, it must first survive a full-test confirmation
- do not treat a test-subset win as a publishable or repo-level result by itself
- for architecture families that survive subset screening, hyperparameter retune may proceed
  per representative task rather than forcing one shared best setting across all three tasks
- even when retune is done per task, the final Stage 3 outcome must still be judged at the
  representative-task bundle level
- routine task tries and fixed-hyperparameter subset screening should stay on one GPU per task by
  default
- if a screening run is otherwise justified but blocked by single-GPU memory, Stage 3 may promote
  that task attempt to a multi-GPU launch on the currently idle GPUs needed to make the run
  viable
- if a representative-task family is bottlenecked by one much slower task, Stage 3 may switch the
  screening shape itself to serial batch-equivalent DDP so the active task can absorb the GPU
  budget that parallel screening would otherwise leave idle at the tail
- once subset evidence is strong enough to justify long validation, Stage 3 should stop manually
  balancing GPU counts across representative tasks
- instead, per-task Optuna and later full confirmation should run serially, with the active task
  consuming all currently idle GPUs the local launcher can safely claim
- do not treat idle GPUs as a reason to silently increase optimization scale: if DDP is used for
  screening throughput, preserve approximate global-batch equivalence rather than multiplying the
  effective batch by world size
- practical throughput comparisons should use per-GPU items/sec rather than raw steps/sec:
  - train primary metric:
    `per_gpu_items/sec = visible step/sec * per_rank_batch_size`
  - train secondary aggregate metric:
    `global_items/sec = visible step/sec * per_rank_batch_size * world_size`
  - eval/test primary metric, when the visible eval batch is per-rank and known:
    `per_gpu_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size`
  - eval/test secondary aggregate metric, when the visible eval batch is per-rank and known:
    `global_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size * world_size`
  - raw step/sec can be recorded, but it should not be the primary comparison when GPU count or
    per-rank batch differs across runs

Complexity rule for autocomplete multiclass heads:

- Stage 3 may use an `O(L*C)` exhaustive candidate head as research evidence
- Stage 3 may not keep an `O(L*C)` multiclass decision interface as the final promoted path for
  large-class tasks
- acceptable long-term directions should behave like `O(L + C)` or another non-multiplicative
  scaling in input length `L` and class count `C`

Stage 4 compatibility rule for autocomplete multiclass heads:

- the final preferred interface should not require source databases to expose true supervised
  benchmark multiclass tasks in order to learn any useful decision mechanism
- prefer shared candidate-scoring or ranking forms that could later be pretrained from proxy
  candidate-selection tasks such as categorical-value discrimination, candidate retrieval, or
  value-ranking episodes across non-target databases
- treat target-database-only fixed-class heads as weaker long-term endpoints even when they are
  locally strong in Stage 3 finetune

## 2. Architecture View

The paper-level architecture should be optimized as a four-layer system:

1. representation fidelity
2. database-conditioned semantic basis construction and injection
3. constraint-driven conservative alignment
4. token compression and LLM consumption

The core dataflow is:

`graph encoder -> basis-aligned latent space -> conservative transfer -> compressed graph tokens -> frozen LLM`

Stage 3 should prioritize the paper's innovation-bearing parts first.

Core layers:

1. basis construction / injection
2. constraint-aware loss
3. constraint-aware sampling
4. token compression / consumption

Secondary engineering knobs are still allowed, but they should not drive the search space
before the core layers are better understood:

- prompt structure
- output formatting
- prediction head details

## 3. Global Guidance

### 3.1 Working philosophy

The main question is not whether the backbone can become generically stronger.

The main question is whether the graph representation can be transferred into a
database-conditioned semantic coordinate system while preserving the structural and temporal
information that downstream reasoning actually needs.

Every candidate should therefore answer one of these questions:

- does it make the basis a better database-specific semantic coordinate system
- does it make the alignment mapping more conservative with respect to key invariants
- does it expose the right graph evidence to the LLM more reliably
- does it make the compressed graph tokens easier for the frozen LLM to use

If a candidate cannot answer one of these questions clearly, it is probably too far from the
paper's main contribution and should be deprioritized.

### 3.2 Search-space discipline

Stage 3 should prefer:

- small structural changes
- one causal idea per bundle
- paper-backed or ablation-backed candidates
- machine-readable specs and reports
- interpretation after every failed bundle

Stage 3 should avoid:

- broad untargeted hyperparameter sweeps
- mixing several orthogonal ideas in one bundle
- prompt wording search without a structural hypothesis
- changing the representative-task protocol mid-comparison
- any candidate that cannot be explained in terms of the paper's architecture

Hyperparameter-search exception:

- once a structural candidate family is already justified by fixed-hyperparameter screening and
  replay evidence, a targeted per-task retune is allowed
- that retune should still use the Stage 3 subset-selection protocol rather than escalating every
  trial to full test
- long retune or full confirmation may use DDP / multi-GPU, but those phases should run serially
  across representative tasks rather than by manual cross-task GPU splitting
- screening remains single-GPU by default, but single-GPU OOM is an allowed exception path for
  escalating one representative-task attempt to multi-GPU
- screening may also use serial batch-equivalent multi-GPU when one representative task dominates
  wall clock and would otherwise strand most of the bundle GPU budget after the faster tasks end

## 4. Layer 1: Basis Construction / Injection

### 4.1 Core claim

The basis should act as a database-conditioned semantic coordinate system rather than a weak
auxiliary prompt decoration.

### 4.2 Failure modes we currently suspect

- the basis is informative but too coarse
- token-side and graph-side basis usage are imbalanced
- graph residual injection may be over-aggressive
- the basis does not adequately represent schema-route or statistics-derived semantics
- the injection path may leak noise faster than it transfers useful coordinate structure

### 4.3 Allowed knobs

- basis atom inventory design
- statistics signature construction
- token residual vs graph residual weighting
- basis temperature and sharpness
- basis mixing, gating, or residual scheduling
- basis projection normalization or calibration
- route-aware or join-aware basis features if they are derived from existing database metadata

### 4.4 Disallowed or low-priority knobs

- generic prompt wording edits without basis-interface consequences
- unrelated backbone expansion with no basis-transfer story
- task-specific basis hacks that break the unified architecture story

### 4.5 Low-cost experiments

- token-only vs graph-only vs mixed basis injection variants
- calibrated residual/gating changes
- basis statistics enrichment without changing the outer training loop
- per-task-neutral basis normalization changes

### 4.6 Code-change experiments

- cleaner basis construction from schema atoms plus statistics signatures
- join-path-aware basis channels
- basis gating conditioned on confidence, sparsity, or route type
- dynamic weighting between token and graph residual paths

### 4.7 Promotion criteria

- at least one representative task improves
- no representative task regresses
- interpretation is consistent with the basis-transfer hypothesis

### 4.8 Retirement criteria

- repeated evidence that a basis variant only helps one task family
- repeated evidence that graph residual strength is the main regression source without any
  compensating bundle-wide gain

## 5. Layer 2: Constraint-Aware Loss

### 5.1 Core claim

Alignment should behave more like a constrained coordinate transformation than a loose semantic
projection.

### 5.2 Failure modes we currently suspect

- current basis BCE terms are too indirect
- alignment can preserve naming cues while losing structural usability
- graph-level auxiliary pressure may be too global and overconstrain some tasks
- regression tasks may be especially sensitive to scale drift and structure loss

### 5.3 Allowed knobs

- explicit local reconstruction losses
- graph-level or route-level conservative penalties
- role, direction, or cardinality-aware auxiliary objectives
- temporal-availability-aware penalties
- scale-preserving penalties for regression-sensitive tasks
- weighting and scheduling among alignment losses

### 5.4 Disallowed or low-priority knobs

- adding opaque auxiliary losses with no invariant story
- importing pretrain objectives into Stage 3
- large multitask loss packages that cannot be interpreted bundle-by-bundle

### 5.5 Low-cost experiments

- token-only reconstruction
- graph-only reconstruction
- local vs global conservative loss decomposition
- alignment weighting changes with fixed backbone and fixed prompt interface

### 5.6 Code-change experiments

- route-aware consistency loss
- foreign-key direction preservation objectives
- bridge-table sensitivity penalties
- temporal mask consistency penalties
- regression-scale stabilization losses

### 5.7 Promotion criteria

- gains are not limited to one task type
- failure-mode diagnosis becomes more coherent, especially for `user-ltv`
- the added loss improves alignment without visibly destabilizing the finetune path

### 5.8 Retirement criteria

- the same loss family repeatedly improves `user-churn` while hurting `user-ltv` and
  `item-incoterms`
- the new loss causes unstable optimization or inconsistent checkpoint selection

## 6. Layer 3: Constraint-Aware Sampling

### 6.1 Core claim

Sampling is not just an efficiency detail.
It determines which structural evidence and invariants are visible to the alignment and token
compression pipeline.

### 6.2 Failure modes we currently suspect

- random neighbor sampling may hide key structural evidence
- useful bridge or route evidence may be undersampled
- temporal leakage constraints may be weakly respected by the sampling view
- uniform budget reduction destroys signal instead of removing noise

### 6.3 Allowed knobs

- deterministic or semi-deterministic representative sampling
- route-aware neighbor prioritization
- bridge-table-aware inclusion policies
- temporal-validity-aware filtering
- schema-role-aware budget allocation
- task-neutral sampling rules that expose more meaningful evidence

### 6.4 Disallowed or low-priority knobs

- arbitrary randomization changes without an evidence-visibility hypothesis
- task-specific sampling hacks that undermine the unified task interface
- sampling changes that cannot be explained in terms of invariants or semantic evidence

### 6.5 Low-cost experiments

- reduce randomness while keeping budget fixed
- compare uniform vs route-aware allocation
- preserve bridge evidence before expanding raw neighborhood size

### 6.6 Code-change experiments

- constraint-aware candidate scoring before neighborhood truncation
- temporal-safe route selection
- role- or relation-type quota systems
- deterministic schema-aware subgraph serialization

### 6.7 Promotion criteria

- lower run-to-run instability
- better bundle-wide consistency without simply increasing token volume
- improved performance that is explainable by evidence exposure

### 6.8 Retirement criteria

- the sampling variant merely shifts wins across tasks without improving bundle stability
- the variant only behaves like a hidden prompt-length increase

## 7. Layer 4: Token Compression / Consumption

### 7.1 Core claim

Even good aligned representations can fail if compression destroys the information that the
frozen LLM needs to consume.

### 7.2 Failure modes we currently suspect

- compression may discard local evidence too aggressively
- graph tokens may blur heterogeneous semantic roles
- the frozen LLM may not be reading the injected tokens in a stable way
- output instability may partly come from poor graph-token packaging rather than bad
  alignment

### 7.3 Allowed knobs

- number of graph tokens
- pooling/compression strategy
- separation of local vs global graph tokens
- position and formatting of graph-token insertion
- consumption-friendly serialization choices

### 7.4 Disallowed or low-priority knobs

- pure wording search without a token-consumption hypothesis
- broad prompt engineering disconnected from graph-token use
- changing the frozen-LLM assumption

### 7.5 Low-cost experiments

- vary token count within a narrow range
- separate one global token from several local evidence tokens
- modest prompt-structure changes around the graph-token slot

### 7.6 Code-change experiments

- hierarchical compression
- route-specific token slots
- dual-stream token packaging for local vs global evidence
- consumption-aware gating before prompt injection

### 7.7 Promotion criteria

- bundle-wide improvements are explainable by improved LLM access to graph evidence
- gains do not rely on excessive token inflation

### 7.8 Retirement criteria

- larger or more complex token packaging gives no representative-task improvement
- changes only shift behavior through superficial prompt effects

## 8. Secondary Engineering Knobs

These are allowed because the end goal is strong performance, not only architectural purity.
However, they are subordinate to the four core layers.

### 8.1 Prompt structure

Allowed:

- one shared prompt for all tasks if it remains competitive
- one prompt per task type if the structure clearly differs between classification,
  multiclass, and regression
- controlled output formatting for regression stability

Not allowed:

- free-form prompt wording search as the main Stage 3 activity

### 8.2 Prediction head / output interface

Allowed:

- cleaner candidate-set interfaces
- more stable scalar-output constraints
- minor head changes that improve compatibility with the frozen LLM interface

Not allowed:

- large head redesigns before the alignment pipeline is better understood

## 9. Current Priority Order

Current search priority:

1. basis construction / injection
2. constraint-aware loss
3. constraint-aware sampling
4. token compression / consumption
5. secondary engineering knobs

This order can change only after the log shows repeated saturation of a higher-priority layer.

## 10. Layer Execution Menus

This section turns the four layers into an actionable search menu.

Each layer is divided into:

- priority attempts
- deferred attempts
- prohibited or currently low-value attempts

### 10.1 Layer 1 menu: basis construction / injection

Priority attempts:

- refine token-vs-graph basis injection balance before changing the outer prompt interface
- enrich basis semantics with route-aware or join-aware features derived from existing schema
  metadata
- test calibrated basis gating instead of constant residual strength
- test basis normalization or confidence-aware scaling to reduce noisy graph-side injection

Deferred attempts:

- large basis inventory redesign that requires major preprocessing changes
- task-type-specific basis systems for classification vs regression
- heavy encoder-side changes whose only purpose is to compensate for a weak basis path

Prohibited or currently low-value attempts:

- prompt wording edits presented as basis improvements
- basis changes that only optimize one representative task family
- generic feature stuffing without a database-conditioned coordinate-system story

### 10.2 Layer 2 menu: constraint-aware loss

Priority attempts:

- local token-level reconstruction or consistency losses
- route-aware conservative penalties that preserve evidence-path readability
- direction or role preservation objectives tied to foreign-key semantics
- regression-stability penalties that protect scale-sensitive tasks without changing the task
  head

Deferred attempts:

- broad global graph reconstruction losses that compress every task through one bottleneck
- large multiterm auxiliary packages introduced all at once
- multitask weighting schedules that add many degrees of freedom before the loss family is
  validated

Prohibited or currently low-value attempts:

- pretrain-style losses
- opaque alignment penalties with no identifiable invariant
- losses that cannot be tied back to the paper's conservative-alignment claim

### 10.3 Layer 3 menu: constraint-aware sampling

Priority attempts:

- reduce unnecessary randomness while keeping token budget fixed
- route-aware neighbor prioritization
- bridge-table-aware retention rules
- temporal-validity-safe filtering that is consistent with prediction-time availability

Deferred attempts:

- large raw neighborhood expansion as the main fix
- sophisticated learned samplers before deterministic heuristics are tested
- dataset-specific sampling rules with weak cross-database transfer stories

Prohibited or currently low-value attempts:

- uniform budget pruning as a primary direction unless newly justified
- sampling changes that only act as hidden prompt-length inflation
- sampling rules that break the paper's conservation story

### 10.4 Layer 4 menu: token compression / consumption

Priority attempts:

- modest graph-token count adjustments
- separate local evidence tokens from one global summary token
- move toward consumption-friendly packaging without changing the frozen-LLM assumption
- narrow prompt-structure changes around graph-token placement

Deferred attempts:

- large prompt-template rewrites
- highly task-specific tokenization interfaces
- complex hierarchical token compressors before simpler slot splits are tested

Prohibited or currently low-value attempts:

- free-form prompt wording search
- LLM-side changes that violate the frozen-LLM protocol
- token-count inflation without an evidence-preservation hypothesis

## 11. Experiment Operation Norms

This section defines the concrete operating rules for Stage 3 experiments.

### 11.1 Candidate registration

Every new bundle must have:

- one candidate JSON under `stage3_notes/candidates/`
- a unique bundle id and three run ids
- a declared source type:
  - `paper`
  - `ablation`
  - `paper+ablation`
- literature queries
- one to three papers or explicit prior-ablation evidence
- a short causal hypothesis
- explicit common overrides
- optional task-specific overrides

No bundle should be launched directly from chat text.

Every candidate should also include:

- one explicit reason it may improve the representative-task bundle instead of only one task
- one explicit reason it may fail
- a short rollback note if the bundle requires temporary code edits

### 11.2 Change scope

One bundle should test one main idea.

Allowed:

- one main mechanism plus small supporting implementation needed to make it runnable

Not allowed:

- mixing several unrelated mechanisms in one bundle
- changing protocol and model behavior at the same time unless the protocol change itself is
  the bundle

### 11.3 Protocol discipline

Until explicitly changed, the screening protocol is:

- `main.py`
- finetune-only
- representative-task bundle of three runs
- refreshed strict baselines in [baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)
- `eval_steps=512`
- `periodic_test_steps=512`
- `test_steps=512`
- `model_selection_source=test_subset`

Do not compare results across mixed protocols.

Additional interpretation rule:

- when a candidate lands close to the strict baseline, do not assume a single-run delta is a real
  gain or a real regression until it is checked against the current noise-floor protocol
- fixed-hyperparameter screening is intentionally cheaper than the Optuna-selected baseline, so a
  clean architectural candidate that is near baseline and inside the replay-aware effective delta
  bands may be preserved as `promising / under-tuned` for later Optuna retune consideration
- a clean architectural candidate can also be preserved as `fixed-screen failed,
  retune-plausible mechanism signal` even when it is outside an effective delta band, if the
  evidence says it is not a collapse:
  - it substantially improves a blocker task over a known failed plateau or weak family
  - it keeps a scalable, Stage4-compatible information path
  - practical throughput is acceptable in effective items/sec
  - the gap to the Optuna-selected baseline is plausibly a fixed-hyperparameter gap rather than an
    information-path failure
- this does not promote the candidate or allow manual value scans; it only prevents retiring the
  direction prematurely. A representative-task primary metric clearly worse outside the effective
  delta band is still a fixed-screen failure unless the run was abnormal or implementation
  blocked

### 11.4 Remote launch discipline

Default launch rules:

- use generated wrappers and launcher files
- launch through the local Stage 3 programs first, not through ad-hoc manual SSH commands
- prefer `stage3_research.py` and `stage3_orchestrator.py` as the normal interface
- prefer any currently idle GPU that satisfies the run's resource needs
- when multiple configured servers have idle GPUs, allow the local Stage 3 launcher to use
  all of them rather than artificially capping routine work to one host or a fixed three-GPU
  subset
- when a candidate explicitly requests a multi-GPU fallback for a resource-blocked task, allow the
  local Stage 3 launcher to reserve multiple currently idle GPUs on the same target for that one
  task instead of forcing the rerun to stay on one GPU
- do not keep stale hard-coded GPU exclusions once the underlying cluster issue is resolved
- keep long jobs in remote `tmux`
- keep persistent logs under `/tmp/stage3-expNNN.log`

Manual SSH is allowed only for:

- diagnosis when the local Stage 3 programs are insufficient
- targeted remote inspection
- minimal recovery actions

Manual SSH should not become the default experiment-launch path.

### 11.5 Remote patch and rollback discipline

Stage 3 often needs temporary remote code edits for finetune-only candidates.

Rules:

- temporary candidate code may be synced remotely through generated files and `scp`
- remote code changes should stay minimal and tied to the current candidate
- if a candidate fails, remote tracked code should be restored to the latest repository version
  before the next unrelated candidate
- use Git only for restoring tracked repository files to the latest checked-out version
- do not reset, clean, or delete unrelated remote files
- do not touch remote model weights, virtual environments, caches, or other non-repo artifacts
- do not treat untracked remote runtime files as cleanup targets

The rollback target is:

- latest tracked repository state for the specific edited code files
- not the removal of remote runtime assets

### 11.6 Automation and thread-binding discipline

Heartbeat automation is required for continuous Stage 3 work and must be treated as
thread-bound state.

Rules:

- when Stage 3 remains active beyond the current foreground turn, create or recreate a heartbeat
  automation from the currently live thread instead of relying on manual follow-up
- create or recreate heartbeat automation from the currently live thread
- prefer a fresh automation over trying to revive a stale one bound to an old session path
- if Codex reports a thread-resume path mismatch, treat the existing automation as broken
- delete the broken automation and create a new one attached to the current thread
- do not rely on hand-editing old automation files to repair a path-mismatch heartbeat
- do not delete an otherwise healthy automation only because one monitored bundle finished if the
  broader Stage 3 task is still active
- when a bundle ends but the research program should continue, retarget the automation to the next
  candidate or to the active architecture-review step instead of stopping it
- do not leave a launched Stage 3 bundle or an active architecture-review step without an attached
  heartbeat automation that will resume monitoring and next-step execution
- a heartbeat that sees its monitored bundle complete must continue the Stage 3 loop before
  returning: finalize/judge the bundle, write the conclusion to the candidate JSON and
  `experiment_log.md`, choose the next paper- or ablation-backed action, and either launch it or
  retarget/create a heartbeat for the non-launch next step
- post-completion analysis must be acted on immediately inside the same heartbeat/foreground
  invocation. Do not finalize a bundle, write the interpretation, and then wait idly for the next
  automation tick. Use the interpretation to choose or draft the next candidate, implement and
  validate the next patch, launch if gates pass, or retarget the heartbeat to a concrete blocker.
  Every new or retargeted heartbeat prompt must include this as a short explicit rule.
- deleting the current bundle heartbeat is allowed only after a replacement heartbeat has been
  created for the next active Stage 3 step, or when Stage 3 is explicitly paused / complete
- every newly created or retargeted heartbeat automation must include a throughput guardrail in
  its prompt: each received heartbeat should judge running experiments by window state, metrics,
  and practical throughput, including latest eval/progress, per-GPU train items/sec when
  derivable from step/sec and batch shape, aggregate train items/sec when derivable, GPU
  placement, projected wall-clock tail, and whether a long-tail task is now hurting tuning
  efficiency
- do not rely on long Stage 3 documents being reread during every heartbeat. Any rule that should
  govern heartbeat-time behavior must be written directly into the automation prompt as a short,
  explicit instruction. Keep these inline prompt rules concise so the heartbeat can act on them.

Operational note:

- a mismatch between `C:\...` and `\\?\C:\...` should be treated as a Codex path
  canonicalization issue, not as an experiment failure

### 11.7 Monitoring discipline

Monitoring rules:

- use local programmatic checks first
- prefer `stage3_orchestrator.py status ... --sync-logs`
- allow training-level early stop to finish naturally
- do not use bundle-level manual kill as a routine policy
- keep machine-readable reports and cached logs
- every monitoring pass must evaluate throughput, not just metric deltas; this is especially
  important for autocomplete multiclass tasks where a theoretically scalable scorer can still be
  too slow in practice
- use per-GPU items/sec as the primary throughput number; keep aggregate items/sec and wall-clock
  as secondary operational context. Raw step/sec is only supporting evidence unless the compared
  runs have the same per-rank batch shape
- when a bundle completes or is early-killed, record whether a below-baseline result is a true
  mechanism exhaustion signal or a retune-plausible fixed-screen gap; do not retire a direction
  solely because a fixed-hyperparameter run trails the Optuna-selected baseline if it materially
  improves over known collapsed bands and throughput is acceptable
- exception: if the only remaining active task is a wall-clock outlier and the bundle is already
  failed or that task is still clearly below its strict baseline at the latest completed
  evaluation, early kill is allowed
- when that exception is used, record the throughput reason and the latest evaluation evidence
  explicitly rather than treating the task as unobserved
- for serial bundles, if an earlier representative task is already clearly outside the noise band
  on the negative side, later queued tasks may be skipped because the bundle-level verdict is
  already failure

### 11.8 Literature-backed modification discipline

Stage 3 should not rely on intuition-only edits when a stronger evidence path is available.

Rules:

- do not modify major behavior purely by feel if a relevant paper or prior ablation can be
  checked first
- prefer tricks or mechanisms that have already shown gains in nearby graph-LLM, graph prompt,
  graph alignment, or structured reasoning settings
- write down why the cited trick should transfer to this codebase
- if a trick is borrowed from literature, keep the implementation close to the paper's core
  mechanism before inventing additional variations

The goal is not to ban intuition.
The goal is to force intuition to pass through a literature or prior-evidence filter before it
becomes an experiment bundle.

### 11.9 Judgment discipline

After completion:

- sync logs
- judge against the refreshed strict baseline
- write a machine-readable report
- update candidate status
- update `experiment_log.md`
- record one interpretation that changes the future search space

Every failed bundle must still produce a reusable conclusion.

### 11.10 Promotion discipline

Promotable candidates must satisfy all of:

- no representative-task regression on the primary metric
- at least one representative-task gain
- the gain is consistent with the bundle hypothesis
- the candidate does not depend on invalid protocol drift

### 11.11 Additional safety and quality rules

Before launch, prefer to record:

- affected local files
- affected remote tracked files
- whether the candidate is config-only or code-changing
- expected failure signature

Do not:

- combine protocol changes with model changes in the same bundle
- leave a temporary remote patch in place across unrelated bundles without documenting it
- overwrite unexplained remote edits in tracked files without first confirming they belong to
  the current Stage 3 candidate flow

### 11.12 Documentation discipline

After each bundle:

- update the candidate JSON status
- update `experiment_log.md`
- if the bundle changes the search boundary, update this file
- if a direction is repeatedly disproven, mark it as deprioritized or retired here

## 12. Candidate Queue

This queue translates the current menu into likely next bundles.
It is intentionally short so that Stage 3 does not fan out uncontrollably.

### 12.1 Immediate queue

1. token-only reconstruction follow-up
   - status: completed and failed globally
   - rationale: the follow-up preserved the `user-churn` gain but still regressed
     `user-ltv` and `item-incoterms`, so this branch should not receive another near-duplicate
     rerun
   - layer: constraint-aware loss

2. route-aware conservative loss
   - status: temporarily saturated
   - rationale: replace global graph reconstruction pressure with a more local route/readability
     conservation target
   - layer: constraint-aware loss

3. bridge- and route-aware sampling
   - status: completed and failed globally
   - rationale: the first `bridge_route` retention bundle regressed all three representative
     tasks, so this heuristic sampling branch should not receive a near-duplicate rerun next
   - layer: constraint-aware sampling

4. calibrated basis gating
   - status: completed and failed globally
   - rationale: keep the current basis mechanism but make token- and graph-side injection more
     selective when basis posterior confidence is weak, but the first confidence-gated bundle
     still regressed on all three representative tasks
   - layer: basis construction / injection

5. split local/global graph-token packaging
   - status: completed and failed globally
   - rationale: improve LLM consumption by separating one explicit global summary token from the
     local evidence block while keeping the retrieved/aligned content fixed; the first bundle
     improved `user-churn` and `user-ltv` but still regressed `item-incoterms`
   - layer: token compression / consumption

6. candidate-aware prompt evidence routing
   - status: temporarily saturated after `EXP180/181/182`, `EXP183/184/185`, and
     `EXP186/187/188`
   - rationale: EXP180 produced a fixed-screen failed but retune-plausible salt-side signal, but
     both sample-conditioned routing and group-preserving routing repeated the known
     `item-incoterms` collapse band
   - layer: token compression / LLM consumption / candidate-aware evidence interface
   - consequence: do not launch another prompt-order, candidate-set routing, sample-conditioned
     routing, group-routing, top-k, or token-budget variant without a new mechanism-level
     justification

### 12.2 Deferred queue

- task-type-specific prompt templates
- prediction-head redesign
- large GNN architecture changes
- standalone GNN training phases

These are not forbidden forever, but they are intentionally deferred until the four core layers
show clearer saturation.

### 12.3 Explicitly paused directions

- pretrain and any `--pretrain` path
- uniform neighbor-budget pruning as a primary candidate family
- broad graph-global reconstruction pressure as the default next step
- free-form prompt wording search

## 13. Failure Escalation Protocol

This section defines what to do when a queue, a layer, or the current lightweight search
space appears exhausted.

### 13.1 Failure categories

Not every failed run means the same thing.

Distinguish:

- implementation failure
  - code bug
  - remote sync issue
  - protocol drift
  - broken logging or judge path
- scientific failure
  - the candidate ran correctly and still regressed
- layer-level failure
  - several candidates from the same layer fail in a consistent pattern

Only scientific failure and repeated layer-level failure should prune the search space.

### 13.2 Stable failure-pattern extraction

When a bundle fails, record the most stable pattern, for example:

- `user-churn` improves while `user-ltv` regresses
- `item-incoterms` is consistently the most fragile task
- graph-global pressure helps classification but hurts regression
- sampling changes mostly act as noise amplification

If a failure cannot be summarized into a stable pattern, the next candidate should usually stay
small and diagnostic.

### 13.3 Tiered response

Tier 1: bundle failure, layer still active

- conditions:
  - the candidate failed
  - the layer still has one or two strong unexplored hypotheses
- action:
  - continue in the same layer
  - limit continuation to one or two more clearly differentiated bundles

Tier 2: layer saturation

- conditions:
  - two or more candidates from the same layer fail with a similar pattern
  - no new strong paper-backed hypothesis remains in that layer's low-cost menu
- action:
  - mark the layer as temporarily saturated
  - pause that layer's main family
  - switch to the next layer in the priority order

Tier 3: lightweight search-space exhaustion

- conditions:
  - the priority candidates across the four layers have been tried
  - failures are consistent enough that more local tweaks look repetitive
- current trigger:
  - as of `EXP186/187/188`, the low-cost queues across basis injection, conservative loss,
    sampling, token packaging, scorer families, and candidate-aware prompt evidence routing have
    all produced stable failures or retune-plausible-but-not-promotable partial signals
- action:
  - stop adding small tricks
  - run an architecture-consistency review
  - estimate the current noise floor if near-boundary partial wins are appearing
  - decide whether a medium-scale redesign is needed

### 13.4 Architecture-consistency review

When Tier 3 is reached, check:

- whether the paper's claimed innovations are fully implemented as trainable signals
- whether the basis truly functions as a database-conditioned coordinate system
- whether conservative alignment is explicit enough in the actual loss and data path
- whether token compression destroys the invariants the earlier layers tried to preserve
- whether the current representative-task protocol is revealing a real bottleneck or only
  screening noise

The output of this review should be one of:

- resume Stage 3 with a clearly redesigned candidate family
- freeze Stage 3 and move to a more structural implementation phase
- redefine the paper-story-to-code mapping before launching more bundles

### 13.5 Anti-stuck rule

Do not continue launching near-duplicate bundles just to keep the experiment count growing.

If the search feels stuck, the default action is:

- summarize the failure pattern
- decide whether the current layer is saturated
- either escalate to the next layer or pause for redesign

Experiment count by itself is not evidence of progress.

## 14. Route-Aware Conservative Loss Design Set

This section refines the next loss-family branch into concrete bundleable mechanisms.

The shared design goal is:

- replace brittle graph-global reconstruction pressure with more local conservative alignment
  signals
- preserve evidence-path readability without entering pretrain or redesigning the full encoder

### 14.1 Candidate A: route-consistency loss

Core idea:

- encourage nodes whose sampled evidence routes map to similar schema-path signatures to produce
  similar route-conditioned alignment states

Mechanism sketch:

- derive a compact route signature from the sampled path or relation sequence already exposed to
  the finetune path
- form pairs within a batch that share the same or nearby route signature
- penalize divergence between their route-conditioned basis-query states or route summaries

Why it may help:

- it directly targets route readability rather than global graph reconstruction
- it should preserve cross-example consistency for structurally similar evidence paths

Why it may fail:

- if route signatures are too coarse, the loss may collapse useful task-specific distinctions
- regression may still be sensitive if route consistency ignores magnitude-related information

Implementation cost:

- medium

Recommended bundle shape:

- one bundle with a single new route-consistency term and conservative weight

### 14.2 Candidate B: FK-direction conservative loss

Core idea:

- preserve directional asymmetry implied by foreign-key flow so that alignment does not wash out
  source-vs-target role information

Mechanism sketch:

- identify directional relation instances already present in the sampled subgraph
- add a penalty when aligned states for reversed-direction roles become too similar
- optionally anchor this at the graph-query or token-query level instead of the final output

Why it may help:

- it targets a paper-central invariant with lower risk than graph-global reconstruction
- it is especially relevant for join reasoning and role-sensitive classification signals

Why it may fail:

- if the current graph encoder already encodes direction strongly, the added loss may be mostly
  redundant
- it may help classification more than regression unless paired with scale-sensitive safeguards

Implementation cost:

- low to medium

Recommended bundle shape:

- one isolated bundle with only the FK-direction term added

### 14.3 Candidate C: bridge-sensitivity conservative loss

Core idea:

- preserve the distinct alignment footprint of bridge-mediated evidence so that many-to-many
  routing does not get compressed into generic neighborhood mass

Mechanism sketch:

- detect whether a sampled evidence path crosses bridge-like structures
- encourage bridge-mediated routes to maintain separable intermediate summaries or basis states
- penalize over-smoothing between direct routes and bridge-mediated routes

Why it may help:

- it attacks a likely blind spot of current compression and sampling
- it aligns with the paper's claim that route effects are part of the conserved structure

Why it may fail:

- bridge detection may be noisy or dataset-dependent
- if bridge effects matter mainly through sampling, a loss-only fix may be insufficient

Implementation cost:

- medium

Recommended bundle shape:

- launch only after one simpler route-aware loss is tried first

### 14.4 Recommended order inside this family

Run in this order unless new evidence overrides it:

1. FK-direction conservative loss
2. route-consistency loss
3. bridge-sensitivity conservative loss

Reason:

- FK-direction is the cleanest and most local invariant
- route consistency is broader but still controlled
- bridge sensitivity is promising but slightly more implementation-sensitive

## 15. Current Frontier

This section intentionally overrides older queue text above when the two conflict.

Current working frontier as of 2026-05-09:

- Stage 3 remains finetune-only; do not enter pretrain and do not use `--pretrain`
- current manual tuning work is architecture / implementation tuning, not hyperparameter search
- new architecture parameters may be exposed as knobs, but their value search belongs to Optuna
  after a candidate family earns retuning; do not manually launch value scans
- the representative bundle is still judged jointly across:
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- route-aware conservative loss, bridge/route sampling, calibrated basis gating, and split
  local/global graph-token packaging are not the active frontier
- the exhaustive `candidate_pairwise_ar_hybrid` family remains valuable evidence that
  candidate-token discrimination matters, but it is retired as a final path because it scales like
  `O(L*C)`
- do not return to target-database-only fixed multiclass heads or fixed pooled-label tweaks as the
  mainline endpoint
- active candidates should stay scalable, candidate-aware, non-autoregressive, and compatible
  with the Stage 4 proxy candidate-ranking story
- screening launch should now prefer `packed_batch_equivalent` over the older coarse
  `serial_batch_equivalent` shape when enough safe GPUs are idle, because the packed scheduler can
  run multiple representative tasks in one wave while preserving effective global batch size

Current active candidate family:

- the learned token-stat MLP line has now failed cleanly as `EXP156/157/158`
- do not launch another near-duplicate fixed pooled-label, static token-stat, or small
  token-stat weighting variant
- the active follow-up is now `EXP159/160/161`, a baseline-anchored candidate-aware residual
  scorer
- this candidate preserves the raw-label baseline multiclass scorer as the main decision surface
  and adds only a zero-initialized, bounded pairwise residual
- the mechanism-level question is whether candidate-aware interaction can add a useful correction
  without repeating the salt-side score-surface collapse seen in the replacement-scorer family

Current required next step:

- monitor `EXP159/160/161` with the local orchestrator and judge jointly with replay-aware
  effective deltas
- if it fails, do not immediately increase the residual scale; first decide whether pairwise
  residuals are another exhausted subpart and move to a different Stage 4-compatible mechanism
