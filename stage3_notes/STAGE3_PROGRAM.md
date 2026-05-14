# Stage 3 Program

This file turns Stage 3 from a chat-driven sequence of experiments into a small research
program.

It is inspired by the `autoresearch` style of work:

- define a narrow research objective
- register each candidate before code or launch
- keep each change small
- run the same evaluation gate every time
- use machine-readable artifacts so negative results prune later work

## 1. Objective

Stage 3 is not allowed to optimize one task at a time.

The optimization target is:

- find one structural/model change that is non-regressive across:
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- and produces at least one credible gain within that representative set

Anything else is a failed candidate.

Additional long-term objective:

- avoid converging to a multiclass interface that only works when target-database supervised
  multiclass labels are already available
- prefer candidate-aware decision mechanisms that Stage 4 could later pretrain through
  source-database proxy ranking or value-selection tasks

See:

- [STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md)

## 2. Candidate Sources

Every new candidate must declare one of these source types:

1. `paper`
2. `ablation`
3. `paper+ablation`

Rules:

- `paper` and `paper+ablation` candidates must include related papers
- pure `ablation` candidates must cite the prior experiments they are reacting to
- a candidate with neither papers nor prior ablation evidence should not be launched

## 3. Literature Gate

Before implementing a new idea, write down:

1. search queries
2. one to three relevant papers or an explicit note that the search found nothing useful
3. the concrete mechanism the paper suggests
4. why that mechanism maps to this codebase

This is not bureaucracy. It prevents random prompt tinkering and forces each candidate to
carry an explicit causal story.

For autocomplete multiclass candidates, the causal story should now answer two questions:

1. why it should help `item-incoterms` inside the current Stage 3 bundle
2. why the same decision interface still has a believable Stage 4 leave-one-database training
   story even if source databases do not expose true supervised multiclass benchmark tasks

## 4. Bundle Shape

One optimization idea is one bundle.

A bundle contains:

- one candidate spec JSON under `stage3_notes/candidates/`
- three run ids, one per representative task
- one shared override set
- optional task-specific overrides
- three generated wrappers
- one generated remote launcher
- one parsed verdict report

## 5. Automatic Gate

The screening gate is fixed:

- single-GPU `main.py` by default
- fixed baseline hyperparameters
- `periodic_test_steps=512`
- `model_selection_source=test_subset`
- when screening needs multi-GPU for wall-clock reasons rather than only OOM rescue, keep the
  optimization setting batch-equivalent instead of silently scaling global batch with world size
- `test_subset` screening metrics are an operational fast gate, not the final scientific metric;
  final comparisons against task-level references or SOTA-facing claims must use full-test metrics

Automatic verdict rule:

- keep both screening and full-test-reference comparisons in the report
- the scientific verdict should compare against `full_test_reference_metrics`, not against an
  optimistic subset baseline
- candidate continuation must stay task-aware rather than purely mechanical:
  - `user-churn` can stay `retune_plausible` when screening is already better and the full-test
    gap remains small
  - `user-ltv` can stay `retune_plausible` when screening is non-regressive, because this task is
    known to show much worse absolute subset MAE than final full-test MAE
  - `item-incoterms` should be judged against the full-test reference band near `0.70`, not the
    subset-only `0.81` regime
- if a task is not yet scientifically better but still matches the documented continuation policy,
  record it as `retune_plausible` rather than dropping it automatically
- `better` and `worse` should both be interpreted symmetrically through the current noise-floor
  rule rather than by raw sign alone
- near-boundary gains and near-boundary drops should both remain `neutral / inconclusive` unless
  they are clearly outside the relevant screening delta or rerun-control spread

The primary metric is fixed per task:

- `user-churn`: `roc_auc`
- `user-ltv`: `mae`
- `item-incoterms`: `mrr`

Post-screening retune rule:

- Stage 3 interactive architecture work should focus on implementation path, scorer form,
  information flow, and model architecture; do not manually scan hyperparameter values as the
  main experiment.
- If a new architecture introduces a necessary tunable scalar or choice, wire it into
  `main.py` and, when appropriate, into `tune_hyperparameters.py`, but leave value selection to
  Optuna after the fixed-hyperparameter screening gate.
- if a bundle is strong enough to justify hyperparameter search, `tune_hyperparameters.py` may be
  run separately for each representative task
- those retune searches should keep the same Stage 3 subset-selection protocol
- do not run full test for each Optuna trial
- only the final selected configuration for each task should move on to full-test confirmation
- even after per-task retune, the final keep / drop decision is still made on the three-task
  bundle outcome
- when subset and full-test metrics differ materially, do not reject a mechanism purely because it
  trails the optimistic subset baseline by a large margin; instead use subset for budget
  allocation and full test for the final scientific comparison
- routine tries and subset screening stay on one GPU per representative task by default
- if a representative-task screening run is otherwise justified but blocked by single-GPU memory,
  Stage 3 may escalate that task to a multi-GPU launch on the currently idle GPUs needed to make
  the run viable
- if one representative task is structurally much slower than the others, Stage 3 may also use a
  serial screening shape so the active task consumes a batch-equivalent multi-GPU slice of the
  currently idle pool instead of leaving the bundle tail mostly idle
- once a structural family is justified and long validation starts, do not hand-balance GPU counts
  across representative tasks
- instead, run the representative tasks serially and let the active task consume all currently idle
  GPUs that the local Stage 3 launcher can safely claim
- for autocomplete multiclass heads, treat `O(L*C)` exhaustive candidate scoring as a research-only
  implementation class; final promoted paths must move back to `O(L + C)`-style scaling

## 6. Monitoring Rule

The default Stage 3 policy is:

- allow training-level early stop to finish naturally
- do not use bundle-level manual kill as a routine policy
- use local programmatic status checks and final judge reports to make bundle decisions
- when work must continue beyond the current foreground turn, create a heartbeat automation on the
  current thread so monitoring and next-step execution resume automatically
- do not let a long heartbeat interval block immediate debugging progress. If the current failure
  can be reproduced and checked quickly, continue foreground patch / rerun cycles in the same turn
  until the bug is fixed or the blocker changes
- use long heartbeat intervals such as 30 minutes for jobs that genuinely need waiting time, such
  as training, large cache builds, or long embedding runs
- if automation is still desired during a fast-feedback debug loop, shorten the automation interval
  instead of waiting for a long default interval
- the heartbeat prompt must say that every monitoring pass checks practical throughput as a
  first-class signal: latest eval/progress, train step rate when visible, GPU placement, projected
  remaining wall clock, and whether a long-tail task is now slowing candidate turnover
- the throughput summary in heartbeat prompts and monitoring notes should use:
  - primary: `per_gpu_items/sec = visible step/sec * per_rank_batch_size`
  - secondary: `global_items/sec = visible step/sec * per_rank_batch_size * world_size`
  - operational context: projected wall clock and visible eval/test cadence
- the heartbeat prompt must include essential runtime rules directly, in short form; do not rely
  on the heartbeat handler rereading long Stage 3 documents before acting
- bundle completion is not a stopping condition for the Stage 3 agent
- after a bundle completes, the agent must finalize the report, update the candidate JSON and
  experiment log, extract the search-space consequence, and immediately continue to the next
  justified Stage 3 action unless the user explicitly pauses the program
- the post-completion analysis is an action input, not a waiting point. The same heartbeat or
  foreground turn that records the analysis must continue from it: draft the next candidate,
  implement and validate it, launch it if gates pass, or retarget the heartbeat to the specific
  remaining blocker. Do not stop merely because the next scheduled heartbeat will arrive later.
- if the next action is not yet a launch, retarget the heartbeat to candidate selection,
  architecture review, static validation, or remote sanity checking rather than deleting it

Allowed exception:

- a remaining long-tail representative-task run may be killed early when:
  - completed representative tasks already make the bundle globally failed, or the active
    remaining task is still clearly below its strict baseline at the latest completed evaluation
  - the active remaining task has structurally abnormal throughput relative to the rest of the
    bundle
  - the expected information gain from waiting for natural early stop is low relative to the
    wall-clock cost
- when using this exception, record the kill reason explicitly in the candidate notes and
  experiment log, and use the latest completed evaluation as the task-side evidence rather than
  treating the run as missing data
- for serial bundles specifically, if an earlier representative task has already crossed the
  `worse` boundary under the symmetric noise-floor rule, later queued representative tasks do not
  need to be launched just to confirm the same global failure

Reason:

- the current representative-task runs are short enough that final completion is cheap
- bundle-level manual kill can hide useful cross-task evidence

## 6.1 Precompute Infrastructure Note

For long graph-cache / embedding precompute jobs:

- treat large-table text embedding throughput as a first-class iteration bottleneck
- embedding forward passes may use one GPU or multiple GPUs, but the post-embed aggregation and
  torch-frame materialization step must run from CPU-backed tensors rather than holding the full
  table embedding matrix on GPU
- when one table dominates wall clock, prefer the unified single-table multi-GPU path
  `stage3_precompute.py launch-cache --parallelism-mode embed_multi_gpu --table-names ... --force-cpu-aggregate`
  over repeatedly retrying a slow single-GPU materialization path

## 7. Files

- `stage3_notes/baseline_registry.json`
  - machine-readable strict screening baselines
- `stage3_notes/STAGE3_TUNING_SPACE.md`
  - strategy document for the Stage 3 search space and experiment norms
- `stage3_notes/pipeline_config.json`
  - remote and directory config
- `stage3_notes/candidates/*.json`
  - candidate bundle specs
- `stage3_notes/reports/*.json`
  - parser/verdict output
- `stage3_notes/paper_shortlist_2026-05-05.md`
  - the current working paper shortlist for next-stage candidate generation
- `stage3_research.py`
  - local helper CLI
- `stage3_orchestrator.py`
  - local SSH monitor and judge loop for remote stage3 bundles

## 8. Workflow

### 8.1 Register a candidate

```bash
python stage3_research.py new-candidate ^
  --slug basis_token_only ^
  --title "Disable graph residual, keep token residual" ^
  --family basis_injection ^
  --source-type ablation ^
  --start-exp 42
```

Then fill the generated JSON:

- `literature.queries`
- `literature.papers`
- `evidence.prior_experiments`
- `common_overrides`
- `task_overrides`

### 8.2 Render wrappers

```bash
python stage3_research.py render stage3_notes/candidates/exp042_basis_token_only.json
```

This generates:

- three wrappers in `.codex_remote/stage3/`
- one launcher script per remote target in `.codex_remote/stage3/`

Operational caveat:

- for `launch_mode=serial_batch_equivalent` and `launch_mode=packed_batch_equivalent`,
  standalone `render` should not be treated as permission to reuse an old recorded
  `task_launches` snapshot as the future placement decision
- in those modes, use `python stage3_research.py launch <candidate.json> --dry-run` to inspect a
  freshly resolved placement from the current idle GPU pool, or use
  `python stage3_research.py launch <candidate.json>` directly to resolve placement, render
  wrappers, sync code, and start the controller
- this is the preferred local Stage 3 path; do not work around it with hand-written SSH launch
  scripts unless the local tool itself is insufficient for diagnosis

### 8.2a Packed batch-equivalent scheduling

For routine three-task screening, prefer `launch_mode=packed_batch_equivalent` when more than one
safe idle GPU is available.

This mode:

- chooses a per-task DDP world size that divides that task's original `batch_size`
- rewrites per-rank `batch_size` so the effective global batch stays comparable to the baseline
- groups representative tasks into waves that run concurrently when GPU capacity permits
- should use the largest safe batch-equivalent world size allowed by the currently idle GPU pool
  for each task, rather than shrinking a task just to pack more concurrent launches or blindly
  reusing a stale previously recorded placement
- excludes GPUs listed in `gpu_exclude` in `pipeline_config.json` or in the candidate spec
- `gpu_exclude` should be empty by default; use it only for a documented temporary health or
  abnormal-run reason, not as a permanent hard-coded ban on GPU 0
- keeps using generated wrappers, `tmux`, and the local Stage 3 launcher

Practical examples:

- with six usable GPUs, a `batch_size=4` task and a `batch_size=2` task can run in the same wave
  while preserving batch equivalence
- with three usable GPUs, the scheduler may run two or three tasks in the same wave depending on
  batch divisibility and optional `task_cost_weights`
- if a task is known to dominate wall clock, set `task_cost_weights` in the candidate spec so the
  packer can allocate more GPUs to that task without changing effective batch size

### 8.3 Launch through the local program

```bash
python stage3_research.py launch stage3_notes/candidates/exp042_basis_token_only.json
```

The launcher now:

- probes every configured remote target in `stage3_notes/pipeline_config.json`
- finds currently idle GPUs using the configured memory / utilization thresholds
- assigns each representative-task run to any idle GPU that satisfies the run's needs
- may assign multiple GPUs to one representative-task run when the candidate explicitly requests
  a multi-GPU fallback for OOM or other single-GPU memory blocking
- may also launch a candidate in `serial_batch_equivalent` mode, where representative tasks run
  one after another and each task gets the largest batch-equivalent DDP world size that fits in
  the currently idle GPU pool
- syncs the active local code files plus generated wrappers to the selected target(s)
- starts the remote jobs in `tmux`
- records `task_launches` back into the candidate JSON so later status checks know the true
  host and GPU placement for that launched run; this recorded snapshot is for monitoring/history,
  not a standing instruction to reuse the same placement on future relaunches
- should be followed by creating or refreshing a heartbeat automation on the current thread if the
  bundle is expected to outlive the current interactive turn

### 8.4 Parse and judge

```bash
python stage3_research.py judge stage3_notes/candidates/exp042_basis_token_only.json ^
  --log-dir downloaded_logs
```

or point it at copied `/tmp/stage3-exp*.log` files.

The tool compares against strict screening baselines and writes a JSON report.

### 8.5 Monitor a remote bundle locally

```bash
python stage3_orchestrator.py status stage3_notes/candidates/exp039_basis_token_only.json ^
  --sync-logs ^
  --recommend-kill
```

```bash
python stage3_orchestrator.py monitor stage3_notes/candidates/exp039_basis_token_only.json ^
  --interval-sec 60 ^
  --write-report
```

The orchestrator:

- checks remote `tmux` windows on every target used by the bundle
- checks remote GPU occupancy across the configured target set
- syncs `/tmp/stage3-exp*.log` files locally
- prints a compact local status view
- recommends early bundle kill when the first representative task is already worse
- automatically judges a bundle once its remote windows disappear
- must be read for both metric verdicts and practical throughput evidence; a status pass is
  incomplete if it ignores a remaining task whose step rate or projected tail would materially
  slow the Stage 3 tuning loop

## 9. What This Changes

Before:

- ideas were mostly generated ad hoc
- wrappers accumulated manually
- negative results lived mainly in markdown prose

After:

- every idea must have an evidence source
- every bundle is machine-readable
- wrappers are generated, not hand-written
- verdicts are comparable and scriptable
- paper search becomes a standard input, not a nice-to-have
