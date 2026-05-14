# Stage 3 Workflow

This file defines the Stage 3 optimization workflow for the current Stage 2 model.

Programmatic companion files:

- [STAGE3_PROGRAM.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_PROGRAM.md)
- [STAGE3_TUNING_SPACE.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_TUNING_SPACE.md)
- [STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md)
- [baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)
- [pipeline_config.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/pipeline_config.json)
- [stage3_research.py](/G:/RelLLM-2/Rel-LLM/stage3_research.py)
- [stage3_orchestrator.py](/G:/RelLLM-2/Rel-LLM/stage3_orchestrator.py)

Precompute / graph-cache operational rule:

- for large-table text embedding materialization, GPU may be used for embedding forward passes, but
  aggregation / concatenation / tensor-frame materialization must return to CPU rather than keeping
  the full table embedding tensor on GPU
- for single-table wall-clock bottlenecks, prefer the unified precompute path with
  `stage3_precompute.py launch-cache --parallelism-mode embed_multi_gpu ... --force-cpu-aggregate`
  so embedding can fan out across multiple GPUs while CPU owns the final aggregation step

## 1. Baseline

Stage 3 baseline:

- Branch/model family: current Stage 2 `impl-b`
- Purpose: all Stage 3 adjustments are compared against this baseline
- Note: public comparisons may report full-test results, but Stage 3 iteration does **not**
  run full test by default
- Note: the stored `screening_baseline_metrics` and `full_test_reference_metrics` serve different
  purposes and must not be mixed:
  - `screening_baseline_metrics` are the routine fast-gate reference under the fixed
    `test_subset` protocol
  - `full_test_reference_metrics` are the final reference when deciding whether a direction
    actually closes the gap to external baselines / SOTA claims
  - final continuation decisions must still be task-aware rather than purely mechanical:
    `user-ltv` is known to have a stable subset/full-test scale mismatch, while
    `item-incoterms` subset baselines can be materially optimistic

Baseline tasks:

- `rel-amazon / user-churn` for binary classification
- `rel-amazon / user-ltv` for regression
- `rel-salt / item-incoterms` for multiclass / MRR

## 2. Representative Task Set

These three tasks are the only mandatory tasks for routine Stage 3 iteration:

1. `rel-amazon / user-churn`
2. `rel-amazon / user-ltv`
3. `rel-salt / item-incoterms`

Interpretation:

- these three tasks are treated as a proxy set for the broader task pool
- stage3 does **not** optimize each representative task independently
- a candidate that is good for only one representative task is considered failed

Any adjustment that is not at least neutral on this set should be treated as failed.

## 3. Optimization Strategy

Stage 3 default loop:

1. Register a candidate bundle and record its evidence source.
2. Search related papers or explicitly cite the prior failed ablations that motivate it.
3. Keep hyperparameters fixed.
4. Modify one component of the pipeline.
5. Run `main.py` directly instead of `tune_hyperparameters.py`.
6. Prefer a single-GPU screening run unless a change specifically targets DDP behavior or a
   single-GPU attempt is resource-blocked (for example by OOM).
6a. If one representative task is structurally much slower than the others, screening may also
    switch to a serial multi-GPU shape, but only when the launch keeps the optimization setting
    comparable rather than silently changing the effective global batch.
7. Evaluate on validation and a test subset.
8. Compare against the current baseline.
9. Only after a change shows a real gain do we consider committing it and later retuning
   hyperparameters.

Additional long-term compatibility rule:

- even though Stage 3 itself stays finetune-only, active autocomplete multiclass families should
  be preferred when they preserve a plausible Stage 4 leave-one-database pretrain story
- specifically, prefer shared candidate-scoring or ranking interfaces that could later be trained
  through proxy candidate-selection tasks across source databases rather than only through
  target-database supervised multiclass labels

Retune follow-up:

- once a candidate family clears the fixed-hyperparameter screening gate, Stage 3 may enter a
  task-specific hyperparameter retune stage
- this retune stage does not need one shared hyperparameter setting across `user-churn`,
  `user-ltv`, and `item-incoterms`
- each representative task may run its own `tune_hyperparameters.py` search after the structural
  family itself has been justified by the shared screening gate
- final Stage 3 judgment still remains bundle-level across the three representative tasks
- for autocomplete multiclass work, the promoted implementation path must avoid exhaustive
  `O(L*C)` scoring over input length `L` and class count `C`; `O(L + C)`-style interfaces are
  acceptable, but `O(L*C)` is not an acceptable final path for large-class tasks

Reason:

- A full Optuna sweep is too expensive for every idea.
- `main.py` with fixed hyperparameters is the fast screening path.
- `tune_hyperparameters.py` is reserved for confirmed candidates.
- Stage 3 architecture work should change implementation, information flow, scorer form, or
  model architecture. Do not manually try hyperparameter values as the main experimental action.
- New tunable knobs may be added when the architecture needs them, but their values should be
  exposed to `tune_hyperparameters.py` / Optuna and left fixed during initial architecture
  screening unless a prior candidate explicitly justifies a retune.
- single-GPU screening is the default because it is cheaper and avoids DDP teardown /
  startup instability during rapid iteration
- this is a default, not a hard cap; if a representative-task screening run is otherwise
  justified but resource-blocked on one GPU, Stage 3 may escalate that task to the currently idle
  multi-GPU set needed to make the run viable
- if Stage 3 uses multi-GPU during screening for throughput rather than for pure OOM rescue, it
  should preserve approximate batch equivalence; do not casually multiply the effective global
  batch just because extra GPUs are idle

## 4. Stage 3 Optimization Targets

Stage 3 work is organized around four linked layers:

1. `representation fidelity`
2. `basis construction / injection`
3. `constraint-driven conservative alignment`
4. `token compression / LLM consumption`

The detailed search boundary and per-layer rules live in
[STAGE3_TUNING_SPACE.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_TUNING_SPACE.md).

The long-term Stage 4 compatibility guardrail lives in
[STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md).

## 5. Allowed Adjustment Directions

Current Stage 3 directions:

1. Adjust Stage 2 basis construction and injection.
2. Adjust finetune-time conservative alignment objectives.
3. Adjust constraint-aware sampling and token consumption.
4. Use prompt or output-interface changes only as secondary engineering knobs.

## 6. Evaluation Protocol

### 6.1 During adjustment

Use `main.py` with fixed hyperparameters.

Rules:

- do not run full test by default
- use a test subset for fast feedback
- use the same subset protocol when comparing candidate changes
- prefer a single-GPU command for routine screening
- reserve multi-GPU / DDP runs for confirmation, for experiments that explicitly change
  distributed behavior, or for otherwise justified screening reruns that are blocked by
  single-GPU memory limits
- when one representative task is known to dominate wall clock, serial screening is allowed:
  launch the three representative tasks one after another and let the active task consume a
  batch-equivalent multi-GPU slice of the currently idle pool
- the preferred serial screening shape is batch-equivalent DDP, for example by using up to the
  largest divisor of the original `batch_size` that fits in the idle GPU pool and reducing the
  per-rank `batch_size` accordingly
- do not wait mechanically for natural early stop when the only remaining task is a long-tail
  throughput outlier and the bundle is already failed or that task still remains clearly below its
  strict baseline at the latest completed evaluation
- in that case, early kill is allowed, but the kill reason and latest evaluation evidence must be
  recorded explicitly in the candidate notes and experiment log
- every automated or manual status pass for a running bundle must inspect practical throughput in
  addition to metric/window state: active task, latest eval, GPU placement, and projected
  wall-clock tail
- throughput should be compared primarily as per-GPU items/sec, not raw steps/sec:
  - train primary metric:
    `per_gpu_items/sec = visible step/sec * per_rank_batch_size`
  - train secondary aggregate metric:
    `global_items/sec = visible step/sec * per_rank_batch_size * world_size`
  - eval/test primary metric, when the visible eval batch is per-rank and known:
    `per_gpu_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size`
  - eval/test secondary aggregate metric, when the visible eval batch is per-rank and known:
    `global_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size * world_size`
  - raw step/sec should still be recorded as supporting evidence, but it is not comparable across
    different GPU counts or per-rank batch shapes by itself
- use per-GPU items/sec, aggregate items/sec, and progress cadence to decide whether the remaining
  wall-clock tail is
  now a tuning-efficiency bottleneck
- if a formally scalable autocomplete scorer still creates a long-tail throughput bottleneck on a
  representative task, treat that as candidate evidence, not merely as an operations nuisance

Recommended:

- `eval_steps` fixed to a moderate and stable screening value
- `periodic_test_steps` should match the screening protocol used for checkpoint selection
- `test_steps` fixed to the same screening value during routine Stage 3 comparison
- `test_steps = -1` only for final confirmation

Promotion rule:

- subset screening is the default gate
- only candidates that are non-regressive across all three representative tasks may
  advance toward full-test confirmation
- do not treat a large gap versus a `test_subset` screening number by itself as proof that a
  direction is scientifically bad on the real task; `test_subset` is a fast operational gate and
  may be materially more optimistic than full test
- for final scientific judgment, compare the promoted candidate against the task's
  `full_test_reference_metrics`, not against the more optimistic `screening_baseline_metrics`
- because Stage 3 architecture screening uses fixed hyperparameters while the stored baseline was
  selected by Optuna, a near-baseline fixed-hyperparameter candidate can be marked as
  `promising / under-tuned` rather than discarded immediately when:
  - its regressions stay inside the replay-aware effective delta bands
  - it changes the intended mechanism cleanly
  - it has a plausible retune path through `tune_hyperparameters.py`
- task-aware continuation policy:
  - `user-churn`: may remain `retune-plausible` when screening already beats the fixed screening
    baseline and the full-test gap stays small
  - `user-ltv`: may remain `retune-plausible` when screening is non-regressive, because this task
    is known to keep subset MAE in the 70-scale even when full-test MAE is much lower
  - `item-incoterms`: use the full-test reference band near `0.70` for scientific continuation;
    do not let the optimistic subset baseline near `0.81` force a premature retirement
- a candidate can also be recorded as `fixed-screen failed, retune-plausible mechanism signal`
  when a primary metric is still worse outside the effective delta band but the run shows clear
  mechanism value, for example:
  - it materially improves the blocker task over a known collapsed plateau or retired weak band
  - it preserves a Stage4-compatible information path that prior ablations say is important
  - practical throughput in per-GPU items/sec is acceptable, and aggregate items/sec / wall-clock
    remain operationally reasonable
  - the remaining gap to the Optuna-selected baseline is plausibly tunable rather than a score
    surface collapse
- this distinction protects promising directions from being retired too early, but it does not
  promote the candidate: a clearly worse representative-task primary metric outside the effective
  delta band still fails the fixed-screen bundle unless there is an explicitly recorded
  abnormal-run or implementation-blocker reason
- do not turn `retune-plausible` into manual value scans; retuning is only through the documented
  Optuna path after deciding that the mechanism is worth the extra budget
- do not run a single-task full test for a candidate that has already regressed on
  another representative task
- apply the same noise-floor logic to losses as to gains:
  - a small near-boundary drop should remain `neutral / inconclusive`
  - only a drop that clearly exceeds the relevant screening delta or rerun-control spread should
    be treated as `worse`

### 6.2 For final confirmation

Only after a candidate is clearly better:

1. run `tune_hyperparameters.py`
2. run the best setting on the full test split if needed

Final-comparison rule:

- use subset / `test_subset` metrics to decide whether a candidate deserves more budget
- use full-test metrics to decide whether the candidate actually improves the task in the sense
  that matters for external comparison and eventual SOTA-facing claims
- when the subset metric is much more optimistic than the full-test metric, record that explicitly
  and avoid interpreting the subset number as the true task level

Retune protocol:

- `tune_hyperparameters.py` may be run separately for each representative task
- the search protocol should stay aligned with Stage 3 screening:
  - use the same `periodic_test_steps`
  - keep `model_selection_source=test_subset`
  - do not run full test for each trial
- for Optuna, cap one task to at most two GPUs so batch-size tuning keeps useful granularity
- if more GPUs are idle, prefer running multiple Optuna tasks in parallel over making one tuning
  task wider than two GPUs
- only the final selected setting for each task should advance to a full-test confirmation run
- split Optuna and final full test into two scheduling phases:
  - phase 1: tuning-only run that writes best params and stops
  - phase 2: separate final full-test confirmation run using the selected best setting
- routine candidate tries and fixed-hyperparameter subset screening should stay on one GPU per
  representative task by default
- if a task is blocked by single-GPU memory limits, it may be relaunched on multiple currently
  idle GPUs without waiting for the full-validation phase
- if screening is instead bottlenecked by one much slower representative task, the bundle may be
  launched serially so the active task uses the same GPU budget that parallel screening would have
  left idle after the short tasks finish
- under serial screening, once an earlier representative task is already clearly `worse` under the
  symmetric noise-floor rule, later queued representative tasks may be skipped because the bundle
  is already failed
- after subset evidence is clearly effective, per-task Optuna retune is allowed, but Stage 3
  should run the three representative tasks serially for long validation phases rather than
  manually splitting GPU counts across tasks
- for long retune or full-test confirmation, the active representative task should consume all
  currently idle GPUs that the local Stage 3 launcher can safely claim
- the default confirmation shape is therefore:
- single-GPU per-task attempts during screening by default
- multi-GPU per-task fallback during screening when one GPU is not enough to make the intended
  candidate runnable
- serial batch-equivalent multi-GPU screening when one representative task would otherwise leave
  most of the bundle GPU budget idle
- subset-backed per-task Optuna retune
- serial full validation where each task uses the full currently idle GPU pool

Repository-result rule:

- if a subset-screened candidate is going to be treated as a real Stage 3 result and kept as a
  repository-facing outcome rather than a disposable exploration, run a full-test confirmation
  before committing or pushing it as a claimed gain

## 7. Experiment Rules

Each experiment should change only one narrow factor or one coherent bundle.

Examples:

- basis bucket definition only
- token alignment loss only
- prompt token ordering only
- GNN pretraining only

Avoid mixing unrelated changes in one run.

New default:

- use `python stage3_research.py new-candidate ...` to create a machine-readable bundle
- use `python stage3_research.py render <candidate.json>` to generate wrappers and a
  launcher
- exception: for `launch_mode=serial_batch_equivalent`, the current local tool must first
  resolve a fresh launch plan from the current idle GPU pool; use
  `python stage3_research.py launch <candidate.json> --dry-run` to inspect the resolved placement
  or `python stage3_research.py launch <candidate.json>` to resolve, render, sync, and start the
  serial bundle in one local-program step
- preferred high-utilization screening mode: use `launch_mode=packed_batch_equivalent` when
  multiple safe idle GPUs are available; it packs representative tasks into batch-equivalent
  waves, keeps each task's effective global batch unchanged by choosing a divisor of the original
  `batch_size`, and can run combinations such as a `batch_size=4` task plus a `batch_size=2` task
  concurrently instead of forcing the whole bundle to be serial
- in `packed_batch_equivalent`, do not shrink a task below its largest currently safe
  batch-equivalent world size merely to squeeze in extra concurrent tasks. Resolve each task to
  its largest legal world size from the current safe idle GPU pool first, then pack only those
  full-footprint launches that jointly fit
- for both `serial_batch_equivalent` and `packed_batch_equivalent`, do not reuse an old
  recorded `task_launches` snapshot as the future launch decision. Re-resolve placement from the
  current safe idle GPU pool each time a candidate is relaunched, then write the new resolved
  placement back as a runtime record
- use `python stage3_research.py judge <candidate.json> --log-dir <dir>` to compare
  parsed metrics against strict screening baselines

## 8. Experiment Recording

Stage 3 progress must be written to files, not only kept in chat context.

For each candidate, record at least:

- date
- branch
- commit or temporary remote patch note
- target component
- hypothesis
- fixed hyperparameters
- command used
- validation metrics
- test-subset metrics
- result summary: `better / neutral / worse`
- decision: `keep / drop / retune later`

Candidate bundles should now live under:

- [stage3_notes/candidates](</G:/RelLLM-2/Rel-LLM/stage3_notes/candidates>)

Parsed verdict reports should live under:

- [stage3_notes/reports](</G:/RelLLM-2/Rel-LLM/stage3_notes/reports>)

Recommended storage:

- add run notes under `stage3_notes/`
- or maintain a single structured markdown log

## 8a. Continuous Progression After Completion

Stage 3 is a continuous research program, not a one-bundle monitoring task.

When a running bundle finishes, the active agent must not stop after only reporting the result.
It must immediately do the next program step unless the user explicitly says to pause:

1. sync logs and run the judge/report path
2. update the candidate JSON and `experiment_log.md`
3. decide the next action from the documented search boundary
4. if a next candidate is justified, create or update its machine-readable candidate JSON
5. run static validation / sanity checks as appropriate
6. launch the next justified bundle when it clears the documented gates

## 8b. Debug Versus Long-Run Monitoring

Do not use the same pacing rule for all work:

- for immediate-feedback debugging, continue within the same active session until the concrete bug
  is fixed or the next blocker is precisely identified
- do not wait for a 30-minute automation heartbeat between quick debug iterations
- use automation heartbeat primarily for long-running remote jobs such as precompute, Optuna, and
  full-test confirmation
- if a run becomes a long job after a local debug fix lands, switch back to the documented remote
  monitoring cadence and keep the automation attached to the new concrete blocker
7. keep or create a heartbeat automation attached to the current thread for the active next step

If a bundle finishes and no next launch is justified yet, the heartbeat should be retargeted to
the architecture-review / candidate-selection step rather than deleted. Delete a heartbeat only
when Stage 3 is truly paused, finished, or explicitly handed back to the user.

## 9. Temporary Modification Policy

Default policy during Stage 3 exploration:

- use temporary remote modifications via `scp` for fast testing
- do **not** push every tentative change to GitHub
- only commit and push a change after it has shown a real accuracy gain

This policy applies to model changes, prompt changes, and evaluation logic changes.

## 10. Suggested Practical Loop

For each new Stage 3 idea:

1. write the hypothesis into the Stage 3 note file
2. apply the change temporarily
3. run `main.py` on the three representative tasks
4. compare against the current baseline
5. if the gain is weak or inconsistent, discard it
6. if the gain is clear, keep the patch
7. only then consider:
   - local commit
   - remote push
   - later full tuning with `tune_hyperparameters.py`

## 11. Current Operating Assumptions

Current agreed assumptions:

- baseline is the current Stage 2 merged model
- representative tasks are fixed to the three tasks above
- hyperparameter sweeps are expensive and should be postponed
- subset testing is the default adjustment protocol
- full test is only for final confirmation
- persistent workflow documentation is mandatory
- before each remote run, check GPU occupancy and avoid cards already used by others or
  cards with stale blocked processes
- do not cap Stage 3 launch usage to one server or to a fixed three-card subset when more
  configured remote targets and idle GPUs are available
- do not globally exclude GPU 0 by default. It may be excluded in a candidate spec or pipeline
  config only when a current health check or recent run evidence shows that using it would likely
  corrupt the bundle, and that reason must be recorded.
- when Stage 3 remains active after the current interactive turn, attach or refresh a heartbeat
  automation on the current thread so monitoring and follow-up execution continue automatically
- do not use a long-interval heartbeat as a substitute for foreground debugging. If the current
  blocker is an implementation bug, argument mismatch, sync failure, or other issue that can be
  reproduced and checked immediately, keep working in the same foreground turn until the bug is
  fixed or the blocker changes materially
- a 30-minute heartbeat cadence is appropriate for long-running jobs whose outcome is not
  immediately observable, such as training, graph-cache materialization, or large embedding runs
- for fast-feedback debugging, use one of two paths instead of waiting for the next long heartbeat:
  - continue patch / run / inspect cycles in the current turn until the issue is resolved
  - or, if background automation is still useful, shorten the automation interval to match the
    expected debug feedback loop
- after a heartbeat or status check reveals an immediate startup failure, do not wait for the next
  scheduled automation tick before acting. Treat the failure as a live foreground debugging task
  and continue immediately
- every new or retargeted heartbeat automation must explicitly instruct the monitor to evaluate
  throughput on each status pass, not only completion and metrics; this includes checking
  `--recommend-kill` output, computing per-GPU items/sec from step/sec and per-rank batch,
  computing aggregate items/sec from step/sec, per-rank batch, and world size when possible,
  comparing the active task's progress cadence with the rest of the bundle, and recording
  throughput/ETA evidence before any early kill
- every new or retargeted heartbeat automation must also remind the monitor to separate
  `fixed-screen failed` from `mechanism exhausted`: if a candidate is below the Optuna-selected
  baseline but materially improves over known collapsed bands and has acceptable per-GPU
  throughput, record it as retune-plausible instead of retiring the direction
- every new or retargeted heartbeat automation must include the essential heartbeat-time rules
  directly in its prompt as short explicit instructions. Do not assume the heartbeat handler will
  reread long Stage 3 documents before acting.
- do not keep an `O(L*C)` autocomplete multiclass head on the final Stage 3 mainline; treat such
  implementations as research evidence only, then redesign them into a scalable `O(L + C)`-style
  interface before repo-facing promotion
- after a heartbeat observes that a bundle has completed, it must continue the Stage 3 loop:
  finalize/report the bundle, record the search-space consequence, choose the next justified
  paper- or ablation-backed action, and retarget/create a heartbeat for that next action instead
  of simply stopping because the monitored bundle ended
- after the completion analysis is written, do not wait for the next heartbeat before acting on
  that analysis. In the same invocation, immediately proceed to the next justified step: create or
  update the next candidate JSON, implement the minimal coherent patch, run static/sanity gates,
  launch if the gates pass, or retarget the heartbeat to the concrete non-launch blocker. A
  heartbeat prompt must state this rule explicitly in short form.
- the current implementation priority is coarse-grained end-to-end pipeline completion, not
  premature performance polish
- once a major part is implemented well enough to enter real validation, treat it as
  `implemented, under validation` rather than reopening it for micro-optimization by default
- only return to an earlier part during this phase if validation exposes a concrete correctness,
  interface, stability, or launchability blocker
- performance-intensive refinement belongs after all major parts are implemented and connected
