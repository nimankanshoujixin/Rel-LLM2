# Stage 3 Experiment Log

## Baseline

- Status: full-test baseline fixed from prior stage2 artifacts; subset baseline pending live rerun
- Branch: `main`
- Model line: current Stage 2 merged `impl-b`
- Representative tasks:
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- Operating rule:
  - these three tasks are a proxy for the broader task pool
  - a candidate must be non-regressive across the full representative set to remain
    alive
  - a single-task gain is not enough to justify full-test confirmation
- Baseline artifact sources:
  - `optuna_runs/amazon_user_churn_llama1b_ddp_implb2/best_trial.json`
  - `optuna_runs/amazon_user_ltv_llama1b_ddp_implb2/best_trial.json`
  - `optuna_runs/salt_item-incoterms_llama1b_ddp_implb2/best_trial.json`
- Full-test reference metrics:
  - `user-churn`: `roc_auc=0.6918065868366201`, `average_precision=0.7420686063230703`
  - `user-ltv`: `r2=0.10848423938250884`, `mae=16.671180345060154`, `rmse=52.348502253874656`
  - `item-incoterms`: `mrr=0.7043105782857789`, `accuracy=0.580488289249941`

## EXP193 Optuna completion

- Date: 2026-05-14
- Study: `exp193_user_ltv_optuna_20260514t141722`
- Status: completed on `lab25211`
- Protocol:
  - `periodic_test_steps=512`
  - `model_selection_source=test_subset`
  - `max_gpus_per_task=2`
  - no bundled final full test
- Result:
  - `best_trial=6`
  - `best_value(mae)=70.44936827350408`
  - artifact-compatible best params kept:
    - `channels=256`
    - `num_layers=2`
    - `aggr=mean`
    - `batch_size=2`
- Synced artifacts:
  - `optuna_runs/exp193_user_ltv_optuna_20260514t141722/best_trial.json`
  - `optuna_runs/exp193_user_ltv_optuna_20260514t141722/stage3_optuna_exp193_user_ltv_20260514t141722.db`
  - `stage3_notes/reports/log_cache/stage3-exp193.log`
- Interpretation:
  - Phase 1 Optuna is done
  - any full-test confirmation must be launched separately with `--final-test-only`
  - the separate final-test-only confirmation has now completed for `user-ltv`
  - full-test metrics from the selected best setting:
    - `r2=0.18005665780277025`
    - `mae=15.633182804023788`
    - `rmse=50.20323115209195`
  - relative to the stored full-test reference
    (`r2=0.10848423938250884`, `mae=16.671180345060154`, `rmse=52.348502253874656`),
    this is a real improvement on `user-ltv`

## EXP192 / EXP194 Phase 2 monitoring refresh

- Date: 2026-05-14
- Context:
  - this continuation pass revalidated Stage 3 from the clean worktree
    `G:\RelLLM-2\Rel-LLM-clean-p13` on branch `codex/stage3-clean-p13`
  - the older heartbeat thread could no longer be trusted after the restart / bluescreen risk, so
    the thread heartbeat automation `monitor-exp192-optuna-continuation` was refreshed onto the
    current thread with a `30` minute cadence and explicit continuation rules
- Verified remote state on `lab25211`:
  - tmux session `lymtmux` still contains:
    - `0:bash`
    - `1:stage3-exp194-final`
    - `2:stage3-exp192-final`
  - live process chains still exist in remote repo `/fs/fast/u2021201693/lym/Rel-LLM`
  - `exp192_user_churn_optuna_20260513t2307` final-test-only remains alive on GPUs `0,1`
    with two DDP ranks
  - `exp194_item_incoterms_optuna_20260513t2307` final-test-only remains alive on GPUs `2,3`
    with two DDP ranks
- Sampled throughput and progress:
  - `exp192` / `user-churn`:
    - sampled around `63266/175943` at `36:27`
    - about `28.6` aggregate test examples/sec
    - about `14.3` per-GPU test examples/sec
  - `exp194` / `item-incoterms`:
    - sampled around `156531/201418` at `2:20:01`
    - about `19.2` aggregate test examples/sec
    - about `9.6` per-GPU test examples/sec
- Artifact status:
  - both remote study directories already contain `best_trial.json`
  - both remote study directories also have actively growing `best_trial_test.log`
  - `exp193` remote final-test artifacts remain complete and consistent with the locally synced copy
- Operational decision:
  - do not relaunch or interrupt either Phase 2 run while both are making real forward progress
  - the next required action is still completion handling:
    sync `best_trial.json`, `best_trial_test.log`, and `/tmp/stage3-exp19x-final.log` into the
    clean worktree as soon as either run finishes, then update the report and scientific judgment

## EXP194 final-test-only completion

- Date: 2026-05-15
- Study: `exp194_item_incoterms_optuna_20260513t2307`
- Status: Phase 2 `--final-test-only` completed on `lab25211`
- Runtime outcome:
  - the dedicated tmux window `stage3-exp194-final` is gone while `stage3-exp192-final` remains
    alive
  - GPUs `2,3` returned to idle after completion
  - the final test log ended at `201418/201418` after about `2:59:40`
  - sampled completion throughput was about `18.7-19.0` aggregate test examples/sec
  - throughput interpretation:
    - about `9.3-9.5` test examples/sec/GPU
- Synced artifacts:
  - `optuna_runs/exp194_item_incoterms_optuna_20260513t2307/best_trial.json`
  - `optuna_runs/exp194_item_incoterms_optuna_20260513t2307/best_trial_test.log`
  - `stage3_notes/reports/log_cache/stage3-exp194-final.log`
  - `stage3_notes/reports/exp194_item_incoterms_optuna_20260513t2307.report.json`
- Final metrics from the selected best setting:
  - `mrr=0.7254209687717201`
  - `accuracy=0.6087479985602046`
  - `macro_f1=0.09577581222004289`
  - `micro_f1=0.6087479985602046`
- Scientific interpretation:
  - this beats the stored item-incoterms full-test reference
    (`mrr=0.7043105782857789`, `accuracy=0.580488289249941`)
  - this confirms the earlier task-specific caution was handled correctly:
    the subset metric around `0.7026` was only a continuation gate, while the decisive scientific
    comparison remained the full test, which came out better than reference
  - therefore `EXP194` is now a confirmed positive Phase 2 outcome rather than merely a
    retune-plausible continuation
- Next active action:
  - keep monitoring `exp192_user_churn_optuna_20260513t2307` until its separate final-test-only
    pass finishes, then sync and judge it before making the overall next-step decision

## EXP192 final-test-only completion

- Date: 2026-05-15
- Study: `exp192_user_churn_optuna_20260513t2307`
- Status: Phase 2 `--final-test-only` completed its full test on `lab25211`, but the remote
  wrapper stalled during post-test cleanup before printing the final save/exit lines
- Runtime outcome:
  - `best_trial_test.log` and `/tmp/stage3-exp192-final.log` both reached full test completion at
    `175943/175943` after about `1:41:33`
  - sampled completion throughput was about `28.9-29.3` aggregate test examples/sec
  - throughput interpretation:
    - about `14.4-14.7` test examples/sec/GPU
  - the final metrics line was present in the remote log, so the scientific result itself was not
    lost
  - after confirming the test output was complete and the remaining remote process chain was only
    stuck in cleanup, the stale final-test-only process chain and tmux window were cleared to free
    GPUs `0,1`
- Synced artifacts:
  - `optuna_runs/exp192_user_churn_optuna_20260513t2307/best_trial.json`
  - `optuna_runs/exp192_user_churn_optuna_20260513t2307/best_trial_test.log`
  - `stage3_notes/reports/log_cache/stage3-exp192-final.log`
  - `stage3_notes/reports/exp192_user_churn_optuna_20260513t2307.report.json`
- Final metrics from the selected best setting:
  - `roc_auc=0.6681287838379838`
  - `average_precision=0.7236622468421053`
  - `accuracy=0.6535089588928201`
  - `f1=0.7253997103662313`
- Scientific interpretation:
  - this does not beat the stored user-churn full-test reference
    (`roc_auc=0.6918065868366201`, `average_precision=0.7420686063230703`)
  - that is still consistent with the earlier task-specific framing: `user-churn` screening was a
    valid continuation gate, but the decisive scientific comparator remains the full test
  - therefore `EXP192` remains a mixed Phase 2 result rather than a full cross-task confirmation

## EXP192 Phase 2 summary

- Date: 2026-05-15
- Final Phase 2 outcomes for the EXP192 family:
  - `exp193` / `user-ltv`: confirmed positive on full test
  - `exp194` / `item-incoterms`: confirmed positive on full test
  - `exp192` / `user-churn`: full test below stored reference
- Overall judgment:
  - the `gnn_repr_alignment_invariants` direction is no longer just a `retune_plausible` placeholder
    with no final evidence
  - it now has two real Phase 2 wins on the harder downstream scientific comparison, including the
    key `item-incoterms` task that motivated the continuation
  - however, it is not a clean representative-set sweep because `user-churn` did not confirm on
    full test
  - the right interpretation is therefore: promising confirmed mechanism family with mixed
    representative-task confirmation, not yet a terminal winner and not a direction to drop
- Next action after persistence:
  - persist the completed Phase 2 record and use it as the basis for the next Stage 3 candidate or
    follow-up decision, rather than rerunning this same full-test-only wave again immediately

## EXP192 Part 1 persistence threshold reached

- Date: 2026-05-15
- Revalidated remote cleanup state:
  - tmux session `lymtmux` is back to only `0:bash`
  - no live `tune_hyperparameters.py`, `torch.distributed.run`, or `main.py` process chain remains
    from the EXP192-family Phase 2 runs
  - remote study artifacts remain present for the completed `exp192`, `exp193`, and `exp194`
    final-test-only confirmations
- Program consequence:
  - the current EXP192-family implementation should now be treated as `part1 implemented,
    validation-backed`
  - this does not mean the mechanism is already a full representative-set winner
  - it does mean the repo now has enough real Phase 2 evidence to use the current clean-worktree
    state as the committed base before further Part 3 follow-up work
- Immediate follow-up rule:
  - do not relaunch the same Optuna or final-test-only wave again
  - retarget heartbeat automation from Phase 2 monitoring to the next concrete follow-up blocker,
    such as Part 3 candidate selection, static validation, or launch preparation

## EXP192 concrete Part 3 blocker narrowed

- Date: 2026-05-15
- Narrowed blocker:
  - the next EXP192-family step should no longer be described only as generic Part 3 continuation
  - the concrete blocker should stay aligned with the planned Part 3 direction:
    implement the constraint-conservation layer on top of the validated Part 1 base
- Why this blocker is preferred:
  - the Stage 3 tuning-space docs already define the pipeline as
    `图编码表示 -> 对齐到数据库条件化语义基底 -> 受约束守恒迁移 -> 压缩成 graph tokens -> 注入冻结 LLM`
  - Phase 2 is sufficient to treat Part 1 as implemented and validation-backed, so the right next
    move is to advance into the documented constraint-conservation layer rather than inserting a
    new attribution detour
- Preparation rule:
  - first persist the Part 3 constraint-conservation candidate story and static gate
  - only after that should the next launch decision be made

## Screening Protocol Update

- Date: 2026-05-05
- Change:
  - raise Stage 3 screening caps from `128` to `4096` for:
    - `eval_steps`
    - `periodic_test_steps`
    - `test_steps`
- Reason:
  - `128`-step subset evaluation is too noisy for candidate comparison
  - since checkpoint selection uses `model_selection_source=test_subset`, raising only
    `eval_steps` would not fix the main source of screening variance
- Effect:
  - future candidate renders from `baseline_registry.json` will inherit the wider
    screening protocol
  - old reports remain historically valid, but they were produced under the noisier
    `128`-step screening regime

## Screening Protocol Revision

- Date: 2026-05-05
- Change:
  - revise Stage 3 screening caps from `4096` down to `512` for:
    - `eval_steps`
    - `periodic_test_steps`
    - `test_steps`
- Reason:
  - `4096` reduced evaluation noise, but it also slowed candidate turnover too much for
    routine Stage 3 exploration
  - `512` is the new compromise between variance control and tuning throughput
- Effect:
  - the in-flight `EXP048/049/050` 4096-step baseline refresh was cancelled and must not
    be treated as the new strict baseline
  - baseline refresh must be rerun under the new `512`-step protocol before judging
    further candidates

## Automation Throughput Guardrail

- Date: 2026-05-10
- Change:
  - every new or retargeted Stage 3 heartbeat automation must explicitly ask the monitor to judge
    practical throughput on each status pass, not only window state and metrics
  - required throughput evidence includes latest eval/progress, effective train items/sec when
    derivable from step/sec and batch shape, GPU placement, projected wall-clock tail, and whether
    a remaining long-tail task is slowing candidate turnover enough to trigger the documented
    early-kill exception
  - raw step/sec should be treated as supporting evidence only; primary comparisons should use
    `items/sec = step/sec * per_rank_batch_size * world_size` for training and eval items/sec when
    the eval batch size is known
- Reason:
  - `EXP167` showed that a scorer can be formally non-`O(L*C)` while still being too slow for
    practical Stage 3 screening on `item-incoterms`
  - ignoring throughput lets long-tail tasks consume tuning time after the bundle is already failed
    or after the active task remains clearly below its strict baseline
- Effect:
  - heartbeat prompts and manual monitoring summaries should always include throughput judgment
    alongside replay-aware metric verdicts
  - throughput failures are candidate evidence and should be recorded in candidate notes and the
    experiment log rather than treated as incidental operations noise

## Throughput Metric Revision

- Date: 2026-05-12
- Change:
  - unify Stage 3 throughput reporting so the primary comparison is single-GPU productivity, not
    aggregate throughput inflated or deflated by temporary GPU availability
  - primary training throughput:
    `per_gpu_items/sec = visible step/sec * per_rank_batch_size`
  - secondary aggregate training throughput:
    `global_items/sec = visible step/sec * per_rank_batch_size * world_size`
  - primary eval/test throughput, when per-rank eval batch is known:
    `per_gpu_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size`
  - secondary aggregate eval/test throughput, when per-rank eval batch is known:
    `global_eval_items/sec = visible eval it/sec * per_rank_eval_batch_size * world_size`
- Reason:
  - aggregate throughput alone can mis-rank candidates when one run happens to receive more GPUs
    than another
  - Stage 3 candidate judgment should compare method efficiency first, then use aggregate
    throughput and wall-clock only as operational context
- Effect:
  - existing historical entries that say `effective items/sec` should be read as the old
    aggregate-throughput convention
  - new entries and new heartbeat prompts should report per-GPU items/sec first, and aggregate
    items/sec only as a secondary number

## Automation Continuation Guardrail

- Date: 2026-05-11
- Change:
  - every new or retargeted Stage 3 heartbeat automation must explicitly say that completing a
    bundle, judging it, and writing the analysis is not a waiting point
  - the same heartbeat / foreground turn must use the analysis to continue the program immediately:
    choose or draft the next paper/prior-backed candidate, implement the minimal coherent patch,
    run static and sanity gates, launch if gates pass, or retarget the heartbeat to a concrete
    non-launch blocker
  - do not stop merely because another scheduled heartbeat will arrive later
- Reason:
  - after `EXP180/181/182`, the automation correctly finalized the failed bundle and recorded the
    retune-plausible routing signal, but it should have continued directly from that analysis into
    the next justified Stage 3 step rather than waiting for the next automation tick
- Effect:
  - future heartbeat prompts must include a short explicit continuation rule in addition to the
    finetune-only, throughput, replay-aware delta, and retired-scan rules

## Fixed-Hyperparameter Interpretation Rule

- Date: 2026-05-10
- Change:
  - Stage 3 architecture screening is interpreted as a fixed-hyperparameter mechanism test, not a
    final tuned comparison against the Optuna-selected baseline
  - a clean candidate that is near the strict baseline and stays inside replay-aware effective
    delta bands may be preserved as `promising / under-tuned` for later Optuna retune
    consideration
- Boundary:
  - this does not override the representative-task bundle gate: a primary metric clearly worse
    outside the effective delta band still fails fixed screening unless the run is abnormal or
    implementation-blocked
  - however, fixed-screen failure is not the same as mechanism exhaustion
  - if a candidate materially improves a blocker task over a known collapsed plateau or retired
    weak band, keeps a scalable Stage4-compatible information path, and has acceptable effective
    items/sec, record it as `fixed-screen failed, retune-plausible mechanism signal` rather than
    retiring the direction
  - example interpretation: an `item-incoterms` MRR around `0.755` against the Optuna-selected
    strict baseline around `0.815` is still fixed-screen worse, but it is meaningfully different
    from the repeated `0.6785764663938492` collapse and may be a retune-plausible signal
  - no manual value scans are allowed; any hyperparameter value search still belongs to Optuna
    after the structural family earns it

### EXP-180 / EXP-181 / EXP-182

- Date: 2026-05-11
- Branch: `main` (temporary remote patch only, not committed)
- Target component: candidate-set guided prompt evidence routing
- Candidate spec:
  - `stage3_notes/candidates/exp180_candidate_set_guided_prompt_routing.json`
- Hypothesis:
  - recent Stage4-compatible scorer variants kept changing the final autocomplete decision surface
    but repeatedly failed to preserve the salt-side `item-incoterms` ranking signal
  - moving candidate information earlier into prompt evidence routing may expose more
    discriminative graph evidence to the frozen LLM without changing scorer form, prompt length, or
    the finetune-only protocol
- Evidence basis:
  - G-Retriever / GraphRAG motivate task-relevant graph evidence retrieval and graph-guided
    retrieval before generation
  - `EXP099/108` showed candidate-token evidence matters
  - `EXP156/159/168/171/174/177` showed pooled-label, token-stat, residual, late-interaction,
    shared-prefix, and Poly-encoder near-lines were not sufficient as the next active frontier
- Change summary:
  - added `--prompt_candidate_guided_routing`
  - for autocomplete multiclass tasks only, build a compact candidate-set query from class-id and
    raw-label candidate text embeddings
  - reorder existing graph prompt evidence groups by candidate relevance before LLM encoding
  - do not add prompt tokens, do not change the scorer, do not use `--pretrain`, and do not scan
    routing top-k or token budgets
- Validation before launch:
  - `python -m py_compile main.py model.py tune_hyperparameters.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp180_candidate_set_guided_prompt_routing.json`
- Remote sanity result:
  - ran a temporary 64-step `item-incoterms` finetune-only check on `lab25211` GPU `7`
  - command used `--prompt_candidate_guided_routing` and did not use `--pretrain`
  - run completed cleanly with no shape/runtime error
  - training stabilized around `6.1` step/s with `batch_size=4`, about `24.4` effective train
    items/sec
  - val/test ran around `12.3` it/s with `val_size=1`, about `12.3` eval/test items/sec
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp180_candidate_set_guided_prompt_routing.json`
  - resolved packed batch-equivalent placement:
    - `EXP180` / `user-churn`: target `lab25211`, GPUs `4,5`, per-rank batch `2`, wave `0`
    - `EXP181` / `user-ltv`: target `lab25211`, GPUs `6,7`, per-rank batch `1`, wave `0`
    - `EXP182` / `item-incoterms`: target `lab25211`, GPUs `4,5,6,7`, per-rank batch `1`, wave
      `1`
  - first post-launch status check:
    - `EXP180` and `EXP181` windows are up in `rel-amazon` DB-load startup
    - `EXP182` is correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Automation:
  - created current-thread heartbeat automation `monitor-stage-3-exp180-bundle`
- Decision:
  - completed and finalized on 2026-05-11
  - global verdict `failed`
- Final subset test metrics:
  - `EXP180` / `user-churn`:
    - `average_precision=0.6941850723020724`
    - `accuracy=0.64453125`
    - `f1=0.7589403973509934`
    - `roc_auc=0.6846461354497603`
  - `EXP181` / `user-ltv`:
    - `r2=-0.13529166202790743`
    - `mae=79.0516936504784`
    - `rmse=150.42208637599003`
  - `EXP182` / `item-incoterms`:
    - `accuracy=0.6484375`
    - `macro_f1=0.13262150137242457`
    - `micro_f1=0.6484375`
    - `mrr=0.7432317302922772`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.030625232211524356` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `0.22087904855246165` is far inside the
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.07155635705440389` against the
    effective threshold `0.013037038915945098`
- Throughput evidence:
  - `EXP182` ran on four ranks with per-rank batch `1`
  - visible training stabilized around `6.8` train steps/sec, about `27.2` effective train
    items/sec
  - final test progress was around `12.16` it/sec
  - this is not a practical-throughput failure
- Deeper interpretation:
  - candidate-set guided prompt routing did not recover the strict salt-side baseline, so the
    fixed-screen bundle fails and must not be promoted
  - the salt-side result is nevertheless meaningfully above the repeated `0.6785764663938492`
    collapse band, the `EXP159` `0.6990761110634157` baseline-residual result, and the `EXP177`
    `0.7044836795691288` late-interaction-residual result
  - because the mechanism is Stage4-compatible, keeps acceptable effective items/sec, and changes
    the information path before frozen-LLM consumption rather than another final-scorer near
    variant, record it as `fixed-screen failed, retune-plausible mechanism signal`
- Search-space consequence:
  - do not promote EXP180 and do not manually scan routing top-k, prompt token budgets, residual
    scales, batching/chunk sizes, or rank auxiliary weights
  - preserve candidate-set guided evidence routing as a plausible ingredient for later Optuna or a
    more structural candidate, but only after choosing a new paper- or ablation-backed action that
    is not a near-duplicate of the retired scorer/residual/token-stat/shared-prefix lines

### EXP-183 / EXP-184 / EXP-185

- Date: 2026-05-11
- Branch: `main` (temporary remote patch only, not committed)
- Target component: sample-conditioned candidate-set prompt evidence routing
- Candidate spec:
  - `stage3_notes/candidates/exp183_sample_candidate_guided_prompt_routing.json`
- Hypothesis:
  - EXP180 showed that candidate-set-only evidence routing was a fixed-screen failure but still
    moved `item-incoterms` above repeated scorer-collapse bands
  - G-Retriever / GraphRAG motivate query-focused graph evidence selection, so combining a compact
    candidate-set query with the current sample/seed representation might route evidence toward
    both the candidate set and the current row
  - the change should preserve prompt length, scorer form, finetune-only execution, and the Stage4
    proxy candidate-ranking story
- Change summary:
  - added `--prompt_sample_candidate_guided_routing`
  - reuse the class-id/raw-label candidate-set query from EXP180
  - add the seed/sample prompt representation to that query before scoring existing graph prompt
    tokens for relevance
  - reorder existing prompt evidence only; do not add tokens, change the scorer, or use
    `--pretrain`
- Validation before launch:
  - `python -m py_compile main.py model.py tune_hyperparameters.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp183_sample_candidate_guided_prompt_routing.json`
- Remote sanity result:
  - ran a temporary 64-step `item-incoterms` finetune-only check on `lab25211` GPU `6`
  - command used `--prompt_sample_candidate_guided_routing` and did not use `--pretrain`
  - run completed cleanly with no shape/runtime error
  - training reached roughly `6.2` step/s with `batch_size=4`, about `24.8` effective train
    items/sec
  - eval/test ran around `12.1` it/sec with `val_size=1`
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp183_sample_candidate_guided_prompt_routing.json`
  - resolved packed batch-equivalent placement:
    - `EXP183` / `user-churn`: target `lab25211`, GPUs `4,6`, per-rank batch `2`, wave `0`
    - `EXP184` / `user-ltv`: target `lab25211`, GPUs `4,6`, per-rank batch `1`, wave `1`
    - `EXP185` / `item-incoterms`: target `lab25211`, GPUs `4,6`, per-rank batch `2`, wave
      `2`
- Decision:
  - completed and finalized on 2026-05-11
  - global verdict `failed`
- Final subset test metrics:
  - `EXP183` / `user-churn`:
    - `average_precision=0.6961670548868191`
    - `accuracy=0.66015625`
    - `f1=0.7563025210084033`
    - `roc_auc=0.6877176117513276`
  - `EXP184` / `user-ltv`:
    - `r2=-0.1286743092048419`
    - `mae=79.40659794325009`
    - `rmse=149.98305789663073`
  - `EXP185` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6782075427827381`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.0336967085130917` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE drop `-0.1340252442192309` is far inside the
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.13658054456394297` against the
    effective threshold `0.013037038915945098`
- Throughput evidence:
  - `EXP185` ran on two ranks with per-rank batch `2`
  - visible training was around `6.5` train steps/sec, about `26` effective train items/sec
  - final test progress was around `12.17` it/sec with `val_size=1`
  - this is not a practical-throughput failure
- Deeper interpretation:
  - sample/seed conditioning did not refine EXP180's candidate-set routing signal; it degraded the
    salt-side autocomplete task back to the known weak `0.678` collapse band
  - the Amazon-side gain is useful as a control that the implementation is active, but the
    representative bundle fails because `item-incoterms` is the blocker task and is far outside the
    replay-aware MRR band
  - EXP180 remains the stronger evidence-routing signal: candidate-set-only routing was still
    worse than the strict baseline but materially above this collapse band
- Search-space consequence:
  - do not pursue sample-conditioned prompt routing as the next manual branch
  - if the evidence-routing line continues, keep the candidate-set-only query and reduce prompt
    order destruction by routing metadata groups/blocks while preserving the original order inside
    each block
  - do not scan routing top-k, prompt token budgets, residual scales, batching/chunk sizes, scorer
    near-neighbors, or route sampling

### EXP-186 / EXP-187 / EXP-188

- Date: 2026-05-11
- Branch: `main` (temporary remote patch only, not committed)
- Target component: candidate-set guided prompt group routing
- Candidate spec:
  - `stage3_notes/candidates/exp186_candidate_group_guided_prompt_routing.json`
- Hypothesis:
  - EXP180 suggests candidate-set-only prompt routing carries a useful salt-side signal, because it
    reached `item-incoterms mrr=0.7432317302922772` instead of the repeated weak collapse bands
  - EXP183 shows sample/seed conditioning is harmful for the same blocker task, returning
    `item-incoterms` to `mrr=0.6782075427827381`
  - the next coherent test is therefore to keep candidate-set-only guidance but preserve more graph
    route structure by sorting metadata groups/blocks and not reordering tokens inside each group
- Evidence basis:
  - G-Retriever and GraphRAG motivate query-relevant graph evidence selection before generation
  - Lost-in-the-Middle-style order sensitivity motivates a less destructive ordering mechanism
  - the immediate ablation evidence is EXP180 versus EXP183: candidate-only routing is the stronger
    signal, while sample-conditioned fine-grained routing is a collapse trigger
- Change summary:
  - added `--prompt_candidate_group_routing`
  - reuse the EXP180 candidate-set query from class-id and raw-label candidate text embeddings
  - compute relevance for existing prompt tokens, aggregate relevance by `(table, route_signature)`
    metadata group, sort groups by mean relevance, and preserve the original token order inside
    each group
  - keep prompt length fixed, leave the scorer unchanged, keep Stage 3 finetune-only, and avoid
    `O(L*C)` LLM rollout
- Validation before launch:
  - `python -m py_compile main.py model.py tune_hyperparameters.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp186_candidate_group_guided_prompt_routing.json`
- Remote sanity result:
  - ran a temporary 64-step `item-incoterms` finetune-only check on `lab25211` GPU `6`
  - command used `--prompt_candidate_group_routing` and did not use `--pretrain`
  - run completed cleanly with no shape/runtime error
  - training stabilized around `6.0` step/s with `batch_size=4`, about `24` effective train
    items/sec
  - eval/test ran around `12.15` it/sec with `val_size=1`
- Launch plan:
  - use `packed_batch_equivalent` scheduling
  - judge `user-churn`, `user-ltv`, and `item-incoterms` jointly with replay-aware effective deltas
  - throughput will be judged by effective items/sec, not raw step/sec
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp186_candidate_group_guided_prompt_routing.json`
  - resolved packed batch-equivalent placement:
    - `EXP186` / `user-churn`: target `lab25211`, GPUs `4,6`, per-rank batch `2`, wave `0`
    - `EXP187` / `user-ltv`: target `lab25211`, GPUs `4,6`, per-rank batch `1`, wave `1`
    - `EXP188` / `item-incoterms`: target `lab25211`, GPUs `4,6`, per-rank batch `2`, wave
      `2`
  - first post-launch status check:
    - `EXP186` window is up in `rel-amazon` DB-load startup
    - `EXP187/188` are correctly waiting for later waves
    - orchestrator recommendation: `keep_watching`
- Automation:
  - retargeted current-thread heartbeat automation `monitor-stage-3-exp180-bundle` to monitor
    `EXP186/187/188`
- Decision:
  - completed and finalized on 2026-05-11
  - global verdict `failed`
- Final subset test metrics:
  - `EXP186` / `user-churn`:
    - `average_precision=0.689626057197749`
    - `accuracy=0.666015625`
    - `f1=0.7550143266475645`
    - `roc_auc=0.6841889309621224`
  - `EXP187` / `user-ltv`:
    - `r2=-0.08403738161497887`
    - `mae=77.78193103098369`
    - `rmse=146.9873679621845`
  - `EXP188` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6782075427827381`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.030168027723886492` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `1.490641668047175` is inside the
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.13658054456394297` against the
    effective threshold `0.013037038915945098`
- Throughput evidence:
  - `EXP188` ran on two ranks with per-rank batch `2`
  - visible training was around `6.5` train steps/sec, about `26` effective train items/sec
  - final test progress was around `12.07` it/sec with `val_size=1`
  - this is not a practical-throughput failure
- Deeper interpretation:
  - candidate-set group routing did not preserve EXP180's partial salt-side signal
  - it exactly repeated the EXP183 salt-side collapse value, despite a different ordering
    granularity and acceptable throughput
  - the stable pattern now is: `user-churn` improves, `user-ltv` is neutral, and any routing
    variant after EXP180 collapses `item-incoterms` back to the known weak MRR band
  - EXP180 should remain recorded as fixed-screen failed but retune-plausible; however, EXP183 and
    EXP186 show that manual prompt evidence routing variants are not the right next active branch
- Search-space consequence:
  - pause candidate-aware prompt evidence routing as an active manual Stage 3 branch
  - do not launch another candidate-set routing, sample-conditioned routing, group-routing,
    prompt-order, top-k, or token-budget variant without a new mechanism-level justification
  - because the low-cost queues across basis injection, conservative loss, sampling, token
    packaging, scorer families, and prompt evidence routing have all produced stable failures,
    trigger a new architecture-consistency review before the next launch

### EXP-189 / EXP-190 / EXP-191

- Date: 2026-05-11
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prefix-distilled candidate pairwise scorer
- Candidate spec:
  - `stage3_notes/candidates/exp189_candidate_pairwise_prefix_distilled.json`
- Architecture-review basis:
  - `stage3_notes/STAGE3_ARCHITECTURE_REVIEW_2026-05-11.md`
  - the review concluded that another routing / scorer small variant is not justified
  - the missing medium-scale mechanism is a way to transfer the candidate-token score surface from
    EXP108 into a cheap Stage4-compatible candidate scorer
- Hypothesis:
  - EXP099 showed candidate-token scoring can be subset-promotable, but the exhaustive hybrid path
    is not an acceptable final endpoint
  - EXP108 preserved `item-incoterms` near the strict baseline with class-id shared-prefix
    candidate-token scoring, but using shared-prefix directly as the final scorer failed the bundle
    and has weak practical throughput on salt-side eval/test
  - cheap pairwise scorers and prompt-routing follow-ups repeatedly collapsed around the same
    `item-incoterms` MRR band
  - therefore train the cheap pairwise scorer against a detached class-id shared-prefix teacher
    distribution during finetune, but keep eval/test prediction on the cheap scorer only
- Paper / prior basis:
  - knowledge distillation: soft teacher distributions can train a compact student more richly than
    hard labels alone
  - listwise ranking: candidate distributions are a better fit for MRR-style candidate ordering
    than another scalar residual or rank-weight scan
  - prior ablation evidence: `EXP099`, `EXP108`, `EXP156`, `EXP180`, `EXP183`, and `EXP186`
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_prefix_distilled`
  - add `autocomplete_prefix_distill_weight`
  - add `autocomplete_prefix_distill_temperature`
  - during training, compute pairwise student logits and a detached class-id shared-prefix teacher
    candidate distribution
  - add KL soft-target distillation to the normal cross-entropy task loss
  - at eval/test, return only the pairwise student logits; do not use shared-prefix as the final
    decision path
  - keep Stage 3 finetune-only; no `--pretrain`
- Validation before launch:
  - `python -m py_compile main.py model.py tune_hyperparameters.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp189_candidate_pairwise_prefix_distilled.json`
- Remote sanity result:
  - ran a temporary 64-step `item-incoterms` finetune-only check on `lab25211` GPU `6`
  - command used `--autocomplete_decision_interface=candidate_pairwise_prefix_distilled`,
    `--autocomplete_prefix_distill_weight=1.0`, and did not use `--pretrain`
  - run completed cleanly with no shape/runtime error
  - training stabilized around `2.45` step/s with `batch_size=4`, about `9.8` effective train
    items/sec
  - eval/test ran around `14.5` it/sec with `val_size=1`
  - this is slower than prompt-routing candidates but healthier than the old EXP110 shared-prefix
    long tail on item-incoterms, which was around `1.28` train step/s and `3.23` test it/sec
- Launch plan:
  - use `packed_batch_equivalent` scheduling
  - judge `user-churn`, `user-ltv`, and `item-incoterms` jointly with replay-aware effective deltas
  - throughput will be judged by effective items/sec, not raw step/sec
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp189_candidate_pairwise_prefix_distilled.json`
  - resolved packed batch-equivalent placement:
    - `EXP190` / `user-ltv`: target `lab25211`, GPUs `4,6`, per-rank batch `1`, wave `0`
    - `EXP189` / `user-churn`: target `lab25211`, GPU `4`, per-rank batch `4`, wave `1`
    - `EXP191` / `item-incoterms`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave
      `1`
  - first post-launch status check:
    - `EXP190` window is up in `rel-amazon` DB-load startup
    - `EXP189/191` are correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Final subset test metrics:
  - `EXP189` / `user-churn`:
    - `average_precision=0.6772381239893106`
    - `accuracy=0.623046875`
    - `f1=0.7541401273885351`
    - `roc_auc=0.6306934031178256`
  - `EXP190` / `user-ltv`:
    - `r2=-0.04303941722681537`
    - `mae=77.16605039620961`
    - `rmse=144.18107010220234`
  - `EXP191` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6793995690724206`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `worse`, with ROC-AUC delta `-0.023327500120410294`
    against the effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `2.106522302821247` is inside
    the effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.13538851827426046` against the
    effective threshold `0.013037038915945098`
- Throughput:
  - pre-launch `item-incoterms` sanity established roughly `9.8` effective train items/sec
  - final `item-incoterms` test ran around `14.6` it/sec
  - this is slower than prompt-routing candidates but healthier than the old shared-prefix long
    tail, so the bundle is a scientific failure rather than a practical-throughput failure
- Decision:
  - completed and finalized on 2026-05-12
  - global verdict `failed`
- Deeper interpretation:
  - the pairwise student did not absorb the class-id shared-prefix teacher's salt-side ranking
    signal under the fixed screen
  - the final `item-incoterms` result remains in the same collapse band as other cheap scorer and
    prompt-routing attempts, far below the strict baseline `mrr=0.8147880873466811`
  - `EXP180` remains retune-plausible as a fixed-screen mechanism signal, but this distillation
    variant is not promotable and does not justify another manual scorer/routing neighbor
- Search-space consequence:
  - stop the small candidate-aware scorer / routing / distillation branch as an active Stage 3 line
  - do not manually scan distillation weight, temperature, token budgets, batching, routing,
    shared-prefix, residual scale, or pairwise scorer variants
  - the next justified work should move up a level to larger pipeline blocks: independent GNN
    representation training and explicit relational invariant preservation during cross-space
    alignment
- Automation:
  - retargeted current-thread heartbeat automation `monitor-stage-3-exp180-bundle` to monitor
    `EXP189/190/191`

### EXP-051 / EXP-052 / EXP-053

- Date: 2026-05-05
- Branch: `main` (temporary remote patch only, not committed)
- Target component: baseline refresh under the 512-step screening protocol
- Hypothesis:
  - no model change
  - rerunning the current Stage 2 merged line under the new `512`-step screening
    protocol should provide the strict baselines for the next candidate round
- Change summary:
  - no model or prompt changes
  - protocol only:
    - `eval_steps=512`
    - `periodic_test_steps=512`
    - `test_steps=512`
    - `model_selection_source=test_subset`
- Final subset test metrics:
  - `EXP-051` / `user-churn`:
    - `average_precision=0.691080591213041`
    - `accuracy=0.646484375`
    - `f1=0.7723270440251573`
    - `roc_auc=0.6540209032382359`
  - `EXP-052` / `user-ltv`:
    - `r2=-0.168956371616511`
    - `mae=79.27257269903086`
    - `rmse=136.96325750843027`
  - `EXP-053` / `item-incoterms`:
    - `accuracy=0.7421875`
    - `macro_f1=0.16367651470184644`
    - `micro_f1=0.7421875`
    - `mrr=0.8147880873466811`
- Decision:
  - refresh `baseline_registry.json` from `EXP-051/052/053`
  - use these values as the new strict screening baselines for all subsequent
    Stage 3 candidate judgment under the `512`-step protocol

### EXP-054 / EXP-055 / EXP-056

- Date: 2026-05-05
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - token-only basis residual remains the strongest basis-side partial win seen so far
  - under the more stable `512`-step screening protocol, the earlier `user-churn`
    regression may shrink enough to tell us whether this family should remain alive
    or be retired completely
- Change summary:
  - keep the strict `512`-step screening protocol from `EXP-051/052/053`
  - set `basis_graph_alpha=0.0`
  - keep task-specific `basis_residual_alpha` at the refreshed baseline values
- Launch status:
  - launched in remote `tmux` windows:
    - `stage3-exp054`
    - `stage3-exp055`
    - `stage3-exp056`
  - persistent logs:
    - `/tmp/stage3-exp054.log`
    - `/tmp/stage3-exp055.log`
    - `/tmp/stage3-exp056.log`
  - current operating rule:
    - allow training-level early stop to finish naturally
    - do not apply bundle-level manual kill
- Decision:
  - running

### EXP-177 / EXP-178 / EXP-179

- Date: 2026-05-11
- Branch: `main` (temporary remote patch only, not committed)
- Target component: baseline-anchored late-interaction residual scorer
- Candidate spec:
  - `stage3_notes/candidates/exp177_candidate_baseline_late_interaction_residual.json`
- Hypothesis:
  - keep the raw-label baseline score surface as the main autocomplete multiclass decision surface
  - add only a centered, `tanh`-bounded late-interaction residual over selected prompt tokens and
    class-id/raw-label candidate tokens
  - test whether token-level sample-candidate evidence can help without repeating the scorer
    replacement collapse or turning into a residual-scale scan
- Evidence basis:
  - `EXP159` showed baseline anchoring softened the collapse but stayed far below baseline
  - `EXP168` showed late interaction was healthy-throughput and above the repeated
    `0.6785764663938492` plateau but still failed `item-incoterms`
  - `EXP174` showed Poly-encoder remained scalable but scientifically failed
  - ColBERTv2 / PLAID motivate token-level late interaction as a practical candidate-aware scoring
    path
- Change summary:
  - set `autocomplete_decision_interface=candidate_baseline_late_interaction_residual`
  - keep `autocomplete_residual_scale=0.1` fixed; do not hand-scan it
  - keep Stage 3 finetune-only; no `--pretrain`
  - no rank auxiliary, poly-code scan, token-budget scan, sampling change, or prompt/basis change
- Validation:
  - `python -m py_compile main.py model.py tune_hyperparameters.py stage3_research.py stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp177_candidate_baseline_late_interaction_residual.json`
  - remote 64-step `item-incoterms` sanity check completed cleanly
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp177_candidate_baseline_late_interaction_residual.json`
  - resolved packed batch-equivalent placement:
    - `EXP177` / `user-churn`: target `lab25211`, GPUs `3,4`, per-rank batch `2`, wave `0`
    - `EXP178` / `user-ltv`: target `lab25211`, GPUs `5,7`, per-rank batch `1`, wave `0`
    - `EXP179` / `item-incoterms`: target `lab25211`, GPUs `3,4,5,7`, per-rank batch `1`, wave `1`
- Final subset test metrics:
  - `EXP177` / `user-churn`:
    - `average_precision=0.6911024837976345`
    - `accuracy=0.646484375`
    - `f1=0.7570469798657719`
    - `roc_auc=0.684794629214976`
  - `EXP178` / `user-ltv`:
    - `r2=-0.11360384003675517`
    - `mae=78.94613023620798`
    - `rmse=148.97837879429125`
  - `EXP179` / `item-incoterms`:
    - `accuracy=0.5771484375`
    - `macro_f1=0.0731888544891641`
    - `micro_f1=0.5771484375`
    - `mrr=0.7044836795691288`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.030773725976740107` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `0.32644246282288236` is far inside the
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.11030440777755224` against the
    effective threshold `0.013037038915945098`
- Throughput evidence:
  - `EXP179` ran on four ranks with per-rank batch `1`
  - visible training stabilized around `7.4` train steps/sec, about `29.6` effective train
    items/sec
  - final test progress was around `16.38` it/sec
  - this is not a practical-throughput failure
- Decision:
  - completed and finalized on 2026-05-11
  - global verdict `failed`
- Deeper interpretation:
  - baseline-anchored late interaction did not preserve the salt-side score surface
  - the result stayed above the repeated `0.6785764663938492` collapse band but below the naive
    late-interaction `EXP170` MRR and far below the strict baseline `0.8147880873466811`
  - this is a fixed-screen scientific failure rather than a retune-plausible mechanism signal
- Search-space consequence:
  - do not hand-scan `autocomplete_residual_scale`, late-interaction token budget,
    `autocomplete_poly_codes`, ranking margin, or auxiliary weight
  - move to a different paper- or ablation-backed Stage4-compatible candidate-aware information
    path
- Final / abnormal outcomes:
  - `EXP162` / `user-churn`:
    - abnormal pre-evaluation termination
    - launch placement: target `lab25211`, GPUs `0,5`, per-rank batch `2`, wave `0`
    - failure signature:
      - `local_rank: 0`
      - `exitcode=-15`
      - `Signal 15 (SIGTERM) received by PID 41012`
    - interpretation:
      - not scientific evidence for or against the hard-negative auxiliary, because
        `user-churn` does not execute the autocomplete rank auxiliary branch
      - likely an external / placement-side startup abnormality involving GPU0 as rank0 for this
        task shape
  - `EXP163` / `user-ltv`:
    - `r2=-0.07930135265570515`
    - `mae=76.62298232233618`
    - `rmse=146.66593143117672`
  - `EXP164` / `item-incoterms`:
    - `accuracy=0.6494140625`
    - `macro_f1=0.13224845986908887`
    - `micro_f1=0.6494140625`
    - `mrr=0.7555898718252234`
- Result under replay-aware effective deltas:
  - `user-churn` is `abnormal / missing`, with no evaluation line
  - `user-ltv` is `neutral`, because the MAE improvement over baseline is inside the
    `6.511244718243262` effective threshold
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.0591982155214577` against the
    effective threshold `0.013037038915945098`
- Decision:
  - completed and manually finalized on 2026-05-09
  - global verdict `failed`
  - do not rerun `EXP162` solely to recover the missing churn control, because the decisive
    autocomplete task is already clearly worse
- Deeper interpretation:
  - hard-negative supervision improved salt-side ranking materially versus the repeated
    `0.6785764663938492` and `EXP159` `0.6990761110634157` collapsed bands
  - it still remains far below the strict baseline `mrr=0.8147880873466811`, so objective-side
    hard-negative pressure is insufficient by itself
  - this is useful signal: direct ranking pressure can move the default scorer, but the default
    scorer still lacks the candidate-token mechanism needed for baseline-level `item-incoterms`
    performance
- Search-space consequence:
  - do not hand-scan rank margins or auxiliary weights next
  - keep this as evidence that ranking supervision is directionally useful but not enough
  - the next Stage 4-compatible candidate should combine the useful low-cost ranking pressure with
    a genuinely different scalable information path, not another pooled-label, token-stat, or
    residual-strength tweak

### EXP-165 / EXP-166 / EXP-167

- Date: 2026-05-09
- Branch: `main`
- Target component: scalable shared-prefix scorer with hard-negative ranking auxiliary
- Hypothesis:
  - `EXP162` showed that direct hard-negative ranking pressure can move salt-side MRR upward, but
    the default scorer still lacks enough candidate-token information to approach the strict
    baseline
  - `EXP108` showed that the scalable shared-prefix scorer preserves much more of the salt-side
    candidate-token signal than the later replacement scorers
  - combining `candidate_pairwise_shared_prefix_hybrid` with the margin-free hard-negative
    auxiliary should test whether the useful candidate-token information path and MRR-aligned
    supervision are complementary
- Evidence basis:
  - `EXP108` preserved `item-incoterms` near the strict baseline (`mrr=0.815460689484127`) but did
    not become a promotable bundle
  - `EXP126` showed that autoregressive shortlist reranking is too expensive relative to its
    payoff
  - `EXP159` showed that baseline residuals over the current pairwise feature path remain clearly
    worse on `item-incoterms`
  - `EXP162` showed ranking supervision is directionally useful but insufficient on the default
    scorer
- Change summary:
  - set `autocomplete_decision_interface=candidate_pairwise_shared_prefix_hybrid`
  - keep `autocomplete_rank_auxiliary=true`
  - keep Stage 3 finetune-only; no `--pretrain`
  - do not add a new margin, auxiliary weight, residual scale, token-stat feature, or pooled-label
    variant
- Launch plan:
  - candidate spec:
    `stage3_notes/candidates/exp165_shared_prefix_rank_auxiliary.json`
  - use `packed_batch_equivalent`
  - exclude GPU0 for this candidate only because `EXP162` had a current pre-eval SIGTERM when
    `user-churn` used GPU0 as local_rank 0
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp165_shared_prefix_rank_auxiliary.json`
  - resolved packed batch-equivalent placement at launch time:
    - `user-churn` / `EXP165`: target `lab25211`, GPUs `5,6`, per-rank batch `2`, wave `0`
    - `user-ltv` / `EXP166`: target `lab25211`, GPU `5`, per-rank batch `2`, wave `1`
    - `item-incoterms` / `EXP167`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave `1`
  - first post-launch status check:
    - `EXP165` window is up in startup / DB-load
    - `EXP166/167` are correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Decision:
  - running
- Final subset test metrics:
  - `EXP165` / `user-churn`:
    - `average_precision=0.7010961272748281`
    - `accuracy=0.64453125`
    - `f1=0.7576564580559254`
    - `roc_auc=0.6901716666080507`
  - `EXP166` / `user-ltv`:
    - `r2=-0.257243272396539`
    - `mae=82.67677471245727`
    - `rmse=142.0412827092452`
  - `EXP167` / `item-incoterms`:
    - latest completed evaluation before manual early kill: `4096`
    - `accuracy=0.6015625`
    - `macro_f1=0.1323765149333822`
    - `micro_f1=0.6015625`
    - `mrr=0.6993458581349206`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.03615076336981482` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE regression `3.4042020134264135` is inside the wide
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.1154422292117605` against the
    effective threshold `0.013037038915945098`
- Throughput / early-kill note:
  - `EXP167` was manually killed on 2026-05-10 after the `4096-step` evaluation
  - rationale:
    - it had become the only remaining active task
    - throughput was about `1.26` train steps/sec on GPUs `6,7`, with the progress bar still
      projecting more than six hours to the full training cap
    - latest/best test-subset MRR remained far below the strict baseline
    - waiting for natural early stop had low expected information value relative to wall-clock
      cost
- Decision:
  - completed and finalized on 2026-05-10
  - global verdict `failed`
- Deeper interpretation:
  - the shared-prefix scorer remains formally more scalable than exhaustive `O(L*C)` rollout, but
    the current implementation does not satisfy the practical Stage 3 throughput requirement for
    routine screening
  - combining hard-negative ranking pressure with shared-prefix candidate-token scoring did not
    preserve the near-baseline salt-side behavior seen in `EXP108`
  - this weakens the case for more shared-prefix reruns unless a new implementation removes the
    per-step prefix-scoring bottleneck
- Search-space consequence:
  - do not continue shared-prefix + ranking variants as the next active frontier
  - do not hand-scan ranking margins or auxiliary weights
  - the next Stage 4-compatible candidate should avoid both exhaustive autoregressive rollout and
    the current slow shared-prefix path

### EXP-168 / EXP-169 / EXP-170

- Date: 2026-05-10
- Branch: `main`
- Target component: autocomplete late-interaction candidate scorer
- Status:
  - completed and finalized on 2026-05-10
  - global verdict: `failed`
  - candidate spec:
    `stage3_notes/candidates/exp168_autocomplete_late_interaction_scorer.json`
- Hypothesis:
  - the remaining useful `item-incoterms` signal appears to be fine-grained candidate-token
    evidence rather than another pooled-label, scalar token-stat, or residual-strength tweak
  - ColBERT-style late interaction offers a paper-backed way to keep token-level sample-candidate
    matching through batched non-autoregressive MaxSim-like scoring
  - this should preserve more of the `EXP099` / `EXP108` candidate-token signal without returning
    to exhaustive per-candidate LLM rollout or the slow shared-prefix recursion path
- Evidence basis:
  - `EXP099` showed that candidate-token evidence can make the representative bundle
    non-regressive, but its exhaustive autoregressive implementation is not a scalable endpoint
  - `EXP108` preserved `item-incoterms` near the strict baseline, but shared-prefix throughput was
    later shown to be too slow for routine Stage 3 screening
  - `EXP156` / `EXP159` exhausted the current single-vector token-stat and residual paths
  - ColBERTv2 and PLAID motivate late-interaction token matching as a scalable retrieval/scoring
    interface rather than a per-candidate decoding loop
- Launch gate:
  - implement `candidate_pairwise_late_interaction`
  - keep Stage 3 finetune-only; no `--pretrain`
  - run `py_compile` and `git diff --check`
  - before launch, perform a throughput sanity check showing that `item-incoterms` avoids the
    `EXP110` / `EXP167` roughly `1.25` train steps/sec shared-prefix tail
  - do not launch if the implementation collapses back into a pooled-label/token-stat near
    variant or a practical throughput outlier
- Implementation update:
  - added `candidate_pairwise_late_interaction` to `model.py`, `main.py`, and
    `tune_hyperparameters.py`
  - the scorer uses one prompt LLM forward, selects up to 32 valid prompt tokens by relevance to
    the sample representation, and scores class-id plus raw-label candidate tokens with batched
    MaxSim-style late interaction
  - this intentionally avoids shared-prefix recursive `past_key_values` decoding and does not add
    a new manually tuned scalar weight
- Validation:
  - `python -m py_compile tune_hyperparameters.py main.py model.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp168_autocomplete_late_interaction_scorer.json`
- Remaining gate:
  - local tensor smoke testing could not run because the local Windows environment lacks `torch`
  - before launch, run the shape/throughput sanity check in the remote training environment and
    confirm `item-incoterms` does not show the shared-prefix roughly `1.25` train steps/sec
    long-tail signature
- Remote sanity result:
  - ran a temporary 64-step `item-incoterms` finetune-only check on `lab25211` GPU `7`
  - command used `autocomplete_decision_interface=candidate_pairwise_late_interaction` and did not
    use `--pretrain`
  - the run completed cleanly, with no shape/runtime error
  - training reached roughly `5.5` train steps/sec after warmup; validation ran around `12.8` it/s
  - this clears the throughput gate relative to the shared-prefix `EXP110` / `EXP167` roughly
    `1.25` train steps/sec long-tail signature
- Launch decision:
  - proceed to a normal three-task `EXP168/169/170` Stage 3 screening bundle
  - use `packed_batch_equivalent`
  - exclude GPU `0` for this candidate only because recent `EXP162` evidence showed pre-eval
    `user-churn` SIGTERM when GPU0 was local rank 0
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp168_autocomplete_late_interaction_scorer.json`
  - resolved packed batch-equivalent placement at launch time:
    - `user-churn` / `EXP168`: target `lab25211`, GPUs `5,6`, per-rank batch `2`, wave `0`
    - `user-ltv` / `EXP169`: target `lab25211`, GPU `5`, per-rank batch `2`, wave `1`
    - `item-incoterms` / `EXP170`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave `1`
  - first post-launch status check:
    - `EXP168` window is up in `rel-amazon` DB-load startup
    - `EXP169/170` are correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Final subset test metrics:
  - `EXP168` / `user-churn`:
    - `average_precision=0.6946536671904824`
    - `accuracy=0.6416015625`
    - `f1=0.7554963357761493`
    - `roc_auc=0.6888469459130999`
  - `EXP169` / `user-ltv`:
    - `r2=-0.14079958117948088`
    - `mae=78.29212675422431`
    - `rmse=135.30367787569554`
  - `EXP170` / `item-incoterms`:
    - `accuracy=0.6025390625`
    - `macro_f1=0.1168646489921827`
    - `micro_f1=0.6025390625`
    - `mrr=0.7227430555555556`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.034826042674864` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `0.9804459448065472` is inside the wide
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.09204503179112544` against the
    effective threshold `0.013037038915945098`
- Throughput conclusion:
  - `EXP170` did not reproduce the shared-prefix long-tail bottleneck
  - the main training loop ran around `6.1` to `6.3` train steps/sec on GPUs `6,7`
  - final test evaluation ran around `13.05` it/s
  - this is a scientific failure, not a practical-throughput failure
- Deeper interpretation:
  - the late-interaction scorer improves over the repeated `0.6785764663938492` plateau and the
    `EXP159` / `EXP165` roughly `0.699` band
  - it still remains far below the strict baseline `0.8147880873466811` and below the stronger
    hard-negative auxiliary result from `EXP164` (`mrr=0.7555898718252234`)
  - therefore batched hidden-token MaxSim preserves some candidate-token information, but not the
    useful conditional candidate-token signal that made `EXP099` / `EXP108` strong on
    `item-incoterms`
- Search-space consequence:
  - do not hand-scan the prompt-token budget, late-interaction scalar weights, or residual scales
    as the next move
  - treat the naive hidden-token late-interaction path as insufficient unless a new paper- or
    ablation-backed mechanism changes the information path
  - the next candidate, if any, must remain Stage 4-compatible and pass the same throughput gate

### EXP-174 / EXP-175 / EXP-176

- Date: 2026-05-10
- Branch: `main`
- Target component: candidate pairwise Poly-encoder scorer
- Candidate spec:
  - `stage3_notes/candidates/exp174_candidate_pairwise_poly_encoder.json`
- Hypothesis:
  - the current fast single-vector and token-stat replacement scorers lose the candidate-token
    discrimination that made `EXP099` and `EXP108` strong on `item-incoterms`
  - a Poly-encoder-style scorer can extract a fixed small set of learned prompt context vectors
    and let each candidate attend over those vectors, giving a richer candidate-aware interaction
    than pooled labels without returning to exhaustive per-candidate LLM rollout
- Evidence basis:
  - Poly-encoders motivate learned context codes for efficient candidate-aware multi-sentence
    scoring
  - `EXP093` established candidate-conditioned pairwise interaction as a real signal
  - `EXP099` / `EXP108` showed candidate-token evidence is the strongest salt-side path, but
    exhaustive and recursive token scoring are not acceptable final throughput/scaling endpoints
  - `EXP156` / `EXP159` exhausted token-stat and residual near-neighbors
  - `EXP168` showed naive hidden-token MaxSim late interaction is scientifically insufficient
  - the `EXP171` audit showed the batched shared-prefix rewrite is not score-exact, so the next
    move should not be a batching/chunk-size scan
- Implementation update:
  - added `autocomplete_decision_interface=candidate_pairwise_poly_encoder`
  - added fixed learned Poly-encoder context codes over prompt hidden-token states
  - each candidate attends over those fixed context vectors and is scored through both class-id
    and raw-label candidate views on top of the existing pairwise scorer
  - added `autocomplete_poly_codes` as a future tuning knob but kept the screening value fixed at
    `16`; do not hand-scan it
  - kept Stage 3 finetune-only; no `--pretrain`
- Launch gate:
  - run `py_compile`, `git diff --check`, candidate JSON validation, and a remote
    `item-incoterms` shape/throughput sanity check before any full bundle launch
  - do not launch if throughput shows the old shared-prefix roughly `1.25` train steps/sec
    long-tail signature
- Validation before launch:
  - `python -m py_compile tune_hyperparameters.py main.py model.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp174_candidate_pairwise_poly_encoder.json`
- Remote throughput sanity:
  - ran a temporary 64-step `item-incoterms` check on `lab25211` GPU `7`
  - command used `autocomplete_decision_interface=candidate_pairwise_poly_encoder` with
    `autocomplete_poly_codes=16` and did not use `--pretrain`
  - the run completed cleanly, with no shape/runtime error
  - training reached roughly `3.5` to `3.7` train steps/sec after warmup with `batch_size=4`,
    i.e. about `14.0` to `14.8` train items/sec
  - validation used `val_size=1` and ran around `6.7` to `6.8` eval items/sec
  - this clears the launch gate relative to the old shared-prefix roughly `1.25` train steps/sec
    long-tail signature, but throughput should still be watched because it is slower than late
    interaction
- Launch status:
  - launched once enough GPUs were idle under `packed_batch_equivalent`
  - resolved placement:
    - `EXP175` / `user-ltv`: target `lab25211`, GPUs `5,6`, per-rank batch `1`, wave `0`
    - `EXP174` / `user-churn`: target `lab25211`, GPU `5`, per-rank batch `4`, wave `1`
    - `EXP176` / `item-incoterms`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave `1`
  - GPU `0` was excluded for this candidate only because recent `EXP162` evidence showed a
    user-churn pre-eval SIGTERM when GPU0 was local rank 0; global `pipeline_config.json` keeps
    `gpu_exclude=[]`
- Final subset test metrics:
  - `EXP174` / `user-churn`:
    - `average_precision=0.6955263636934258`
    - `accuracy=0.65234375`
    - `f1=0.7775`
    - `roc_auc=0.6593831778701815`
  - `EXP175` / `user-ltv`:
    - `r2=-0.19356479904194224`
    - `mae=82.46907187100966`
    - `rmse=154.23427096138067`
  - `EXP176` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6886574769631411`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.005362274631945607` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE regression `3.1964991719788003` is inside the wide
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.12613061038354` against the effective
    threshold `0.013037038915945098`
- Throughput:
  - the completed `EXP176` item-incoterms run was acceptable on effective throughput: about
    `5.67` train steps/sec on two ranks with per-rank batch `2`, i.e. about `22.7` effective train
    items/sec
  - final `EXP176` test progress was around `6.86` it/sec with val/test batch effectively `1`
  - `EXP174` final test progress was around `12.81` it/sec
  - therefore this is not a practical-throughput failure
- Decision:
  - completed and finalized on 2026-05-10
  - global verdict `failed`
- Deeper interpretation:
  - the Poly-encoder scorer kept the desired scalable and Stage4-compatible sample-candidate
    ranking interface, but it did not recover the candidate-token discrimination needed by
    `item-incoterms`
  - the salt-side result `mrr=0.6886574769631411` is only slightly above the repeated
    `0.6785764663938492` collapse band and far below the strict baseline
    `mrr=0.8147880873466811`
  - unlike a hypothetical fixed-screen `0.75` to `0.76` MRR result, this is not strong enough to
    record as a retune-plausible mechanism signal; it looks like another score-surface collapse
    rather than an under-tuned near miss
  - the Amazon-side better result should again be treated mostly as protocol-control / rerun drift
    evidence because this candidate changes only the autocomplete multiclass scorer
- Search-space consequence:
  - do not hand-scan `autocomplete_poly_codes`
  - treat this Poly-encoder implementation path as failed unless a new paper- or ablation-backed
    mechanism changes the information path
  - the next candidate should continue the architecture-review path and should not be a pooled
    label, token-stat, pairwise residual-scale, naive late-interaction, or non-exact shared-prefix
    near variant

### EXP-171 / EXP-172 / EXP-173

- Date: 2026-05-10
- Branch: `main`
- Target component: batched shared-prefix hybrid scorer
- Candidate spec:
  - `stage3_notes/candidates/exp171_batched_shared_prefix_hybrid_scorer.json`
- Hypothesis:
  - `EXP108` showed that class-id shared-prefix candidate-token scoring can preserve
    `item-incoterms` near the strict baseline, while `EXP165` showed the current recursive
    shared-prefix implementation is too slow for practical Stage 3 screening
  - the new implementation keeps the same class-id shared-prefix score surface but batches
    same-depth trie expansions in bounded chunks, so it tests the implementation path rather than
    a new scorer weight, rank margin, residual scale, prompt-token budget, or token-stat variant
- Evidence basis:
  - `EXP099` established that explicit candidate-token scoring is the strongest successful signal
    so far, but its exhaustive autoregressive implementation is not an acceptable scalable
    endpoint
  - `EXP108` reached `item-incoterms mrr=0.815460689484127`, essentially preserving the strict
    salt-side baseline, and its LTV drop is inside the current wide replay-aware neutral band
  - `EXP126` showed shortlist autoregressive reranking was too costly, and `EXP168` showed fast
    hidden-token late interaction was scientifically insufficient
- Implementation update:
  - added `autocomplete_decision_interface=candidate_pairwise_batched_shared_prefix_hybrid`
  - added a batched same-depth prefix scorer that concatenates KV-cache states for bounded groups
    of trie expansions instead of running one LLM forward per prefix node
  - kept Stage 3 finetune-only; no `--pretrain`
- Validation before launch:
  - `python -m py_compile tune_hyperparameters.py main.py model.py stage3_research.py
    stage3_orchestrator.py`
  - `git diff --check`
  - `python -m json.tool stage3_notes/candidates/exp171_batched_shared_prefix_hybrid_scorer.json`
- Remote throughput sanity:
  - ran a temporary 64-step `item-incoterms` check on `lab25211` GPU `7`
  - command used `autocomplete_decision_interface=candidate_pairwise_batched_shared_prefix_hybrid`
    and did not use `--pretrain`
  - the run completed cleanly, with no shape/runtime error
  - training reached roughly `3.4` to `3.6` train steps/sec after warmup; validation ran around
    `9.7` it/s
  - this clears the launch gate relative to `EXP165`'s roughly `1.26` train steps/sec
    shared-prefix long-tail signature, though throughput must still be watched during the full
    bundle
- Launch decision:
  - proceed to a normal three-task `EXP171/172/173` Stage 3 screening bundle
  - use `packed_batch_equivalent`
  - exclude GPU `0` for this candidate only because recent `EXP162` evidence showed pre-eval
    `user-churn` SIGTERM when GPU0 was local rank 0
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp171_batched_shared_prefix_hybrid_scorer.json`
  - resolved packed batch-equivalent placement at launch time:
    - `user-churn` / `EXP171`: target `lab25211`, GPUs `5,6`, per-rank batch `2`, wave `0`
    - `user-ltv` / `EXP172`: target `lab25211`, GPU `5`, per-rank batch `2`, wave `1`
    - `item-incoterms` / `EXP173`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave `1`
  - first post-launch status check:
    - `EXP171` window is up in `rel-amazon` DB-load startup
    - `EXP172/173` are correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Automation:
  - retargeted current-thread heartbeat automation `continue-stage-3-finetune-work` to monitor
    `EXP171/172/173`
  - heartbeat prompt explicitly carries the practical-throughput guardrail for every status pass
- Monitoring requirement:
  - every status pass must include throughput evidence: latest eval/progress, visible train step
    cadence, GPU placement, projected wall-clock tail, and early-kill relevance if a long-tail
    task emerges
- Final subset test metrics:
  - `EXP171` / `user-churn`:
    - `average_precision=0.6957868859789716`
    - `accuracy=0.6494140625`
    - `f1=0.7585743106926698`
    - `roc_auc=0.6906093324423707`
  - `EXP172` / `user-ltv`:
    - `r2=-0.21918498204541437`
    - `mae=80.68303950697185`
    - `rmse=139.8748801994279`
  - `EXP173` / `item-incoterms`:
    - `accuracy=0.5703125`
    - `macro_f1=0.10059490084985837`
    - `micro_f1=0.5703125`
    - `mrr=0.7078466021825397`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.03658842920413474` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE drop `-1.4104668079409919` is inside the wide
    effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.10694148516414137` against the
    effective threshold `0.013037038915945098`
- Throughput conclusion:
  - `EXP173` did not reproduce the `EXP165` shared-prefix long-tail bottleneck
  - the main training loop ran around `4.6` train steps/sec on GPUs `6,7`
  - final test evaluation ran around `9.44` it/s
  - this is not a practical-throughput failure
- Decision:
  - completed and finalized on 2026-05-10
  - global verdict `failed`
- Deeper interpretation:
  - batching same-depth prefix expansions solved the practical screening bottleneck but did not
    preserve the `EXP108` salt-side behavior
  - `EXP173` improved over the repeated `0.6785764663938492` plateau and the `EXP159` /
    `EXP165` roughly `0.699` band, but stayed below `EXP168` late interaction
    (`mrr=0.7227430555555556`) and far below `EXP108` (`mrr=0.815460689484127`)
  - because this candidate was intended as an implementation-path rewrite of the same
    class-id shared-prefix signal, the negative result should not be followed by batching chunk
    scans or another shared-prefix rerun
- Search-space consequence:
  - do not tune batching/chunk sizes, rank margins, residual scales, prompt-token budgets, or
    auxiliary weights next
  - before any further shared-prefix implementation candidate, run a score-equivalence or
    information-path audit against the original recursive shared-prefix scorer
  - otherwise continue the architecture-review path only with a genuinely different,
    paper- or prior-ablation-backed, scalable, Stage4-compatible candidate-aware scorer
- Post-failure equivalence audit:
  - ran a no-training `item-incoterms` audit on `lab25211` GPU `7`; the audit did not use
    `--pretrain`
  - artifact:
    `stage3_notes/reports/exp171_shared_prefix_equivalence_audit_2026-05-10.json`
  - first pass with `val_size=1` found recursive-vs-batched prefix/final score
    `max_abs=0.049210548400878906`, `mean_abs=0.013607429340481758`, with `0` top1 and `0`
    top5 order mismatches
  - second pass with `val_size=8` found `max_abs=0.06859016418457031`,
    `mean_abs=0.0110812122002244`, again with `0` top1 and `0` top5 order mismatches
  - conclusion: EXP171 is not a score-exact implementation-path rewrite of the recursive
    `EXP108` shared-prefix scorer; it remains a failed candidate, but its salt-side regression
    should be treated as evidence against this non-exact batched score path rather than as a clean
    falsification of the original recursive shared-prefix information path
  - next shared-prefix work, if any, must first prove score equivalence or explicitly justify the
    changed score surface while retaining the practical-throughput guardrail

### EXP-054 / EXP-055 / EXP-056 Final Bundle Verdict

- Date: 2026-05-05
- Bundle: token-only basis residual rerun under 512-step screening (`BUNDLE-054`)
- Final subset test metrics:
  - `EXP-054` / `user-churn`:
    - `average_precision=0.6823087211656008`
    - `accuracy=0.658203125`
    - `f1=0.7606019151846786`
    - `roc_auc=0.6384638849198067`
  - `EXP-055` / `user-ltv`:
    - `r2=-0.14314323456713485`
    - `mae=77.75073617188261`
    - `rmse=135.4425901923732`
  - `EXP-056` / `item-incoterms`:
    - `accuracy=0.71484375`
    - `macro_f1=0.12715021559023196`
    - `micro_f1=0.71484375`
    - `mrr=0.7995673759833916`
- Result:
  - `user-churn` regressed versus refreshed strict baseline `EXP-051`
  - `user-ltv` improved versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - retire token-only basis residual as a global candidate family
- Interpretation:
  - the refreshed 512-step protocol preserves the same pattern seen before:
    token-only basis helps regression but is not globally safe across the representative
    set

### EXP-057 / EXP-058 / EXP-059

- Date: 2026-05-05
- Branch: `main` (temporary remote patch only, not committed)
- Target component: graph-aware pretraining warmup
- Hypothesis:
  - prompt-side and basis-side coefficient scans have failed repeatedly, so the next
    higher-value slot is upstream representation/alignment warmup
  - a short pretrain warmup before task finetuning may improve the graph-text interface
    more globally than further basis residual tuning
- Paper basis:
  - `GALM` motivates graph-aware pretraining for cross-task transfer
  - `RGLM` motivates stronger alignment supervision before downstream task learning
  - `GraphPrompter` motivates improving alignment quality rather than prompt decoration
- Change summary:
  - keep the refreshed `512`-step strict screening protocol
  - enable:
    - `--pretrain`
    - `--pretrain_epochs=5`
  - keep basis, prompt, optimizer, and evaluation settings otherwise unchanged
- Launch status:
  - launched in remote `tmux` windows:
    - `stage3-exp057`
    - `stage3-exp058`
    - `stage3-exp059`
  - persistent logs:
    - `/tmp/stage3-exp057.log`
    - `/tmp/stage3-exp058.log`
    - `/tmp/stage3-exp059.log`
- Decision:
  - running

### EXP-057 / EXP-058 / EXP-059 Final Bundle Verdict

- Date: 2026-05-05
- Bundle: paper-guided graph-aware pretrain warmup (`BUNDLE-057`)
- Result:
  - all three runs crashed before the first evaluation point
  - shared error:
    - `AttributeError: 'NodeStorage' object has no attribute 'df'`
  - failure point:
    - current pretrain path in [model.py](/G:/RelLLM-2/Rel-LLM/model.py) accesses
      `batch[select_table].df` inside `pretrain(...)`
    - that attribute is not present on the remote runtime's `NodeStorage` objects
- Judge status:
  - `stage3_research.py judge` could not emit a verdict report because the logs contain no
    evaluation lines
  - operationally treat this as a failed / implementation-blocked bundle
- Decision:
  - stop this no-code pretrain-warmup direction
  - next direction should be a stronger explicit alignment/reconstruction-loss change,
    not another immediate rerun of the current pretrain path

### EXP-060 / EXP-061 / EXP-062

- Date: 2026-05-06
- Branch: `main` (temporary remote patch only, not committed)
- Target component: finetune-time alignment supervision
- Hypothesis:
  - existing basis BCE terms are too indirect to stabilize the graph-text interface
    across the representative tasks
  - adding explicit latent reconstruction/alignment losses for token-level and
    graph-level basis queries should be a better finetune-only test than more basis
    coefficient ablations
- Paper basis:
  - `RGLM` motivates stronger reconstructive alignment supervision
  - `GALM` motivates upstream graph-aware representation objectives
  - `GraphPrompter` motivates treating alignment quality as the core bottleneck
- Change summary:
  - add new finetune-only auxiliary losses:
    - `basis_lambda_tok_recon`
    - `basis_lambda_g_recon`
  - reconstruct target basis embeddings from the existing token / graph basis targets
  - penalize cosine mismatch between basis queries and target basis embeddings
  - keep residual mixing, prompt construction, and screening protocol otherwise
    unchanged
- Launch status:
  - ready to render and launch

### EXP-060 / EXP-061 / EXP-062 Final Bundle Verdict

- Date: 2026-05-06
- Bundle: stronger alignment and reconstruction loss (`BUNDLE-060`)
- Final subset test metrics:
  - `EXP-060` / `user-churn`:
    - `average_precision=0.7207803313113492`
    - `accuracy=0.681640625`
    - `f1=0.7739251040221914`
    - `roc_auc=0.679724501099748`
  - `EXP-061` / `user-ltv`:
    - `r2=-0.24710226136298363`
    - `mae=82.834228047973`
    - `rmse=141.4672654535929`
  - `EXP-062` / `item-incoterms`:
    - `accuracy=0.708984375`
    - `macro_f1=0.13746639818946532`
    - `micro_f1=0.708984375`
    - `mrr=0.7946269333964646`
- Result:
  - `user-churn` improved versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
- Interpretation:
  - stronger reconstruction-style supervision helps the classification representative
    task
  - graph-level reconstruction pressure appears too costly for regression and salt-side
    ranking, so the next candidate should keep local token reconstruction while removing
    extra graph reconstruction pressure

### EXP-063 / EXP-064 / EXP-065 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: token-level reconstruction only (`BUNDLE-063`)
- Final subset test metrics:
  - `EXP-063` / `user-churn`:
    - `average_precision=0.7204805045000372`
    - `accuracy=0.658203125`
    - `f1=0.732824427480916`
    - `roc_auc=0.6789217653764028`
  - `EXP-064` / `user-ltv`:
    - `r2=-0.4212395238274347`
    - `mae=86.06939905427396`
    - `rmse=151.0214249186481`
  - `EXP-065` / `item-incoterms`:
    - `accuracy=0.716796875`
    - `macro_f1=0.12965118785840557`
    - `micro_f1=0.716796875`
    - `mrr=0.802530536954365`
- Result:
  - `user-churn` improved versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not extend token-only reconstruction as a promotable bundle family
- Interpretation:
  - removing graph reconstruction was not enough to recover the cross-task regressions of
    the stronger alignment branch
  - the next loss-family branch should stay finetune-only but become more local and more
    conservative, centered on route / foreign-key direction invariants rather than any
    graph-global reconstruction target

### EXP-066 / EXP-067 / EXP-068

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: route-aware conservative loss
- Hypothesis:
  - current basis BCE and reconstruction losses can still blur source-vs-target FK roles
    because both sides of a relation are optimized mostly toward shared pair-level
    semantics
  - a local conservative penalty against FK-direction collapse should preserve join-role
    readability without reintroducing graph-global reconstruction pressure
- Paper basis:
  - `GraphPrompter` motivates enforcing structural alignment quality at the graph-to-prompt
    interface
  - `RGLM` motivates explicit graph-side supervision, but also implies the next safer step
    after failed global reconstruction is a more local invariant
  - `GALM` motivates structure-preserving auxiliary supervision as a transfer-oriented
    finetune signal
- Prior ablation basis:
  - `EXP-060` showed stronger reconstruction helps `user-churn` but hurts `user-ltv` and
    `item-incoterms`
  - `EXP-063` showed removing graph reconstruction alone still does not recover cross-task
    stability
- Change summary:
  - code path:
    - add `basis_lambda_fk_dir`
    - add `basis_fk_dir_margin`
    - add a token-query FK-direction conservative loss that penalizes source-role /
      target-role collapse for observed directed FK pairs
  - keep the refreshed `512`-step screening protocol unchanged
  - do not change prompt structure, basis residual coefficients, or any pretrain path
- Launch status:
  - synced current `main.py` and `model.py` to the remote repo as the active candidate
    patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp066`
    - `stage3-exp067`
    - `stage3-exp068`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach remote startup /
    evaluation stages

### EXP-066 / EXP-067 / EXP-068 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: FK-direction conservative loss (`BUNDLE-066`)
- Final subset test metrics:
  - `EXP-066` / `user-churn`:
    - `average_precision=0.7187501746332561`
    - `accuracy=0.6484375`
    - `f1=0.7738693467336684`
    - `roc_auc=0.6819079422672467`
  - `EXP-067` / `user-ltv`:
    - `r2=-0.35878666291198225`
    - `mae=85.81952953252011`
    - `rmse=147.666017780751`
  - `EXP-068` / `item-incoterms`:
    - `accuracy=0.7265625`
    - `macro_f1=0.15046037394791842`
    - `micro_f1=0.7265625`
    - `mrr=0.8074187748015873`
- Result:
  - `user-churn` improved versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not relaunch FK-direction conservative loss as a near-duplicate bundle
- Interpretation:
  - a more local direction-only invariant is still not enough to recover the persistent
    regression-side and salt-side damage
  - the next strongest paper-backed branch inside route-aware conservative loss is
    route-signature consistency, not another narrower direction penalty tweak

### EXP-069 / EXP-070 / EXP-071

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: route-aware conservative loss
- Hypothesis:
  - the current alignment path still allows tokens from the same sampled schema-route
    pattern to drift apart across examples
  - a batch-level route-signature consistency penalty should preserve repeated evidence
    routes more broadly than FK-direction alone, without falling back to graph-global
    reconstruction
- Paper basis:
  - `GraphPrompter` motivates enforcing structure at the graph-to-prompt alignment interface
  - `RGLM` motivates explicit graph-aware supervision, but the supervision should match the
    useful local invariant
  - `GraphRAG Survey` motivates route-aware graph evidence organization before moving to a
    sampling-layer redesign
- Prior ablation basis:
  - `EXP-060` improved `user-churn` but regressed `user-ltv` and `item-incoterms`
  - `EXP-063` showed removing graph reconstruction alone was insufficient
  - `EXP-066` showed FK-direction alone was still too narrow to recover cross-task stability
- Change summary:
  - code path:
    - add `basis_lambda_route_consistency`
    - carry route-signature metadata through prompt-token construction
    - add a batch-level route-signature consistency loss over token-query states
  - keep the refreshed `512`-step screening protocol unchanged
  - do not change basis residual coefficients, prompt structure, or any pretrain path
- Launch status:
  - synced current `main.py` and `model.py` to the remote repo as the active candidate patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp069`
    - `stage3-exp070`
    - `stage3-exp071`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach remote startup /
    evaluation stages

### EXP-069 / EXP-070 / EXP-071 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: route-consistency conservative loss (`BUNDLE-069`)
- Final subset test metrics:
  - `EXP-069` / `user-churn`:
    - `average_precision=0.6955445693133318`
    - `accuracy=0.650390625`
    - `f1=0.7759699624530664`
    - `roc_auc=0.6531218392280894`
  - `EXP-070` / `user-ltv`:
    - `r2=-0.23576294128481146`
    - `mae=80.9891262448393`
    - `rmse=140.82264882661647`
  - `EXP-071` / `item-incoterms`:
    - `accuracy=0.708984375`
    - `macro_f1=0.13746639818946532`
    - `micro_f1=0.708984375`
    - `mrr=0.7984630766369047`
- Result:
  - `user-churn` was neutral versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - treat route-aware conservative loss as saturated for now
  - do not continue to bridge-sensitivity loss as the default next bundle
- Interpretation:
  - broadening the conservative loss from FK-direction to route-consistency reduced the
    classification gain without recovering regression or salt-side stability
  - the bottleneck now looks more like evidence exposure than another missing alignment
    penalty, so the next branch should switch layers to bridge- and route-aware sampling

### EXP-072 / EXP-073 / EXP-074

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: constraint-aware sampling
- Hypothesis:
  - current prompt-side evidence exposure is still too noisy because `recursive_sample`
    serializes whatever first-hop nodes survive the loader without deterministic bridge- or
    route-aware retention
  - preserving route continuations and bridge-like fanout inside the already loaded batch
    subgraph should improve cross-task stability more safely than another alignment loss
- Paper basis:
  - `G-Retriever` motivates evidence-selection quality over more prompt-side decoration
  - `GraphRAG Survey` motivates route-aware evidence organization as a first-class design axis
  - `GraphPrompter` motivates preserving useful structure before graph tokens are projected
    into prompt space
- Prior ablation basis:
  - `EXP-045` showed that uniform neighbor pruning is too destructive
  - `EXP-066` and `EXP-069` showed that repeated route-aware loss tweaks are not recovering
    `user-ltv` and `item-incoterms`
- Change summary:
  - code path:
    - add prompt-side sampling knobs:
      - `prompt_sampling_strategy`
      - `prompt_sampling_hops`
      - `prompt_sampling_topk_per_type`
    - add deterministic `bridge_route` retention inside `recursive_sample(...)`
    - rank candidates by route continuation and bridge-like fanout inside the existing batch
      subgraph
  - candidate config:
    - `prompt_sampling_strategy=bridge_route`
    - `prompt_sampling_hops=2`
    - `prompt_sampling_topk_per_type=2`
  - keep the refreshed `512`-step screening protocol unchanged
- Launch status:
  - synced current `main.py` and `model.py` to the remote repo as the active candidate patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp072`
    - `stage3-exp073`
    - `stage3-exp074`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach remote startup /
    evaluation stages

### EXP-072 / EXP-073 / EXP-074 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: bridge- and route-aware sampling (`BUNDLE-072`)
- Final subset test metrics:
  - `EXP-072` / `user-churn`:
    - `average_precision=0.621444439359319`
    - `accuracy=0.611328125`
    - `f1=0.7587878787878788`
    - `roc_auc=0.5316839790004335`
  - `EXP-073` / `user-ltv`:
    - `r2=-0.6047400848155526`
    - `mae=98.5119142881874`
    - `rmse=160.47495991708772`
  - `EXP-074` / `item-incoterms`:
    - `accuracy=0.71875`
    - `macro_f1=0.14975581642248312`
    - `micro_f1=0.71875`
    - `mrr=0.8027554422573954`
- Result:
  - `user-churn` regressed versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not relaunch this first `bridge_route` sampling bundle or nearby sampling retries that
    only vary the same heuristic strength
- Interpretation:
  - unlike the recent loss-family failures, this sampling bundle did not produce a useful
    tradeoff; it degraded all three representative tasks
  - the failure pattern suggests this deterministic bridge/route retention rule amplified noisy
    prompt-side evidence exposure rather than stabilizing it, so the next branch should move to
    calibrated basis injection instead of another near-duplicate sampling rerun

### EXP-075 / EXP-076 / EXP-077

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: basis construction / injection
- Hypothesis:
  - the current basis path is likely over-injecting low-confidence residuals into prompt tokens,
    which preserves some classification-side signal but destabilizes regression and salt-side
    ranking
  - scaling token and graph residual mixing by the sharpness of the inferred basis posterior
    should keep confident basis transfers while damping noisy graph-side perturbations
- Paper basis:
  - `GraphPrompter` motivates treating prompt-side projection quality as a first-class alignment
    bottleneck
  - `G-Retriever` motivates more selective prompt-side graph evidence exposure instead of uniform
    injection
  - `GraphRAG Survey` motivates structure-preserving organization of graph evidence at the LLM
    interface
- Prior ablation basis:
  - `EXP-039` and `EXP-054` showed that removing graph residual injection helps `user-ltv` but is
    not globally safe
  - `EXP-042` showed that simple graph-side rebalance still regresses `user-churn`
  - `EXP-072` showed that moving to sampling heuristics alone can make all three representative
    tasks worse
- Change summary:
  - code path:
    - add basis gating knobs:
      - `basis_gate_strategy`
      - `basis_gate_token_floor`
      - `basis_gate_graph_floor`
    - scale token- and graph-side basis residual mixing by basis posterior confidence inside
      `align_token_prompts(...)`
    - log `align_token_gate` and `align_graph_gate` for training-time inspection
  - candidate config:
    - `basis_gate_strategy=confidence`
    - `basis_gate_token_floor=0.35`
    - `basis_gate_graph_floor=0.0`
  - keep the refreshed `512`-step screening protocol unchanged
- Launch status:
  - synced current `main.py` and `model.py` to the remote repo as the active candidate patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp075`
    - `stage3-exp076`
    - `stage3-exp077`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
  - first orchestrator status check:
    - all three windows `up`
    - logs reached remote DB-load stage
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach the first evaluation
    point

### EXP-078 / EXP-079 / EXP-080 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: split local/global graph-token packaging (`BUNDLE-078`)
- Final subset test metrics:
  - `EXP-078` / `user-churn`:
    - `average_precision=0.7002034190887916`
    - `accuracy=0.650390625`
    - `f1=0.7765293383270911`
    - `roc_auc=0.6575208309920207`
  - `EXP-079` / `user-ltv`:
    - `r2=-0.16332260308390079`
    - `mae=78.98161479765551`
    - `rmse=136.63281265258124`
  - `EXP-080` / `item-incoterms`:
    - `accuracy=0.71875`
    - `macro_f1=0.13942109782058973`
    - `micro_f1=0.71875`
    - `mrr=0.8017376612103174`
- Result:
  - `user-churn` improved versus refreshed strict baseline `EXP-051`
  - `user-ltv` improved versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not promote, commit, or push this bundle
  - do not launch a near-neighbor packaging tweak immediately
- Interpretation:
  - this is the first recent Stage 3 bundle to produce a credible two-task gain on the
    representative set, but the persistent salt-side regression still invalidates it as a global
    candidate
  - combined with the prior failures in basis, loss, sampling, and gating branches, this moves
    Stage 3 into Tier-3 lightweight-search exhaustion rather than just another local bundle miss

### EXP-081 / EXP-082 / EXP-083

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: label interface
- Hypothesis:
  - the persistent `item-incoterms` regression is partly a multiclass label-interface problem, not
    only an evidence-retrieval or alignment problem
  - autocomplete-style multiclass direct supervision currently uses integer class-id targets
    against label embeddings built from a separate raw-label verbalizer, so unifying the scorer
    around the same class-id interface should reduce salt-side ranking mismatch
- Paper basis:
  - `GraphPrompter` motivates treating the graph/output interface as a first-class bottleneck
  - `G-Retriever` motivates making downstream graph evidence consumption internally consistent
  - `GraphRAG Survey` motivates aligning evidence organization with the decision interface
- Prior ablation basis:
  - `EXP-078` improved both `user-churn` and `user-ltv` but still failed on `item-incoterms`
  - post-`EXP-078` code inspection identified the active-path mismatch in the autocomplete
    multiclass label scorer
- Change summary:
  - code path:
    - add `autocomplete_label_interface`
    - for autocomplete-style multiclass tasks, allow the label scorer to build class prototypes
      from class-id texts instead of `target_col: raw_label` verbalizers
    - keep retrieval, basis alignment, loss, and screening protocol otherwise unchanged
  - candidate config:
    - `autocomplete_label_interface=class_id_unified`
  - the new path is intended to activate only for autocomplete-style multiclass tasks; the two
    Amazon representatives act as no-op controls but remain part of the same global verdict gate
- Launch status:
  - synced current `main.py`, `model.py`, and `utils.py` to the remote repo as the active
    candidate patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp081`
    - `stage3-exp082`
    - `stage3-exp083`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
  - first orchestrator status check:
    - all three windows `up`
    - logs reached remote DB-load stage
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach the first evaluation
    point

### EXP-081 / EXP-082 / EXP-083 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: autocomplete label-interface unification (`BUNDLE-081`)
- Final subset test metrics:
  - `EXP-081` / `user-churn`:
    - `average_precision=0.689861793482772`
    - `accuracy=0.6484375`
    - `f1=0.7744360902255639`
    - `roc_auc=0.6549520766773161`
  - `EXP-082` / `user-ltv`:
    - `r2=-0.20358555405560197`
    - `mae=81.47815621679416`
    - `rmse=138.977152325377`
  - `EXP-083` / `item-incoterms`:
    - `accuracy=0.734375`
    - `macro_f1=0.15535397516529592`
    - `micro_f1=0.734375`
    - `mrr=0.811239769345238`
- Result:
  - `user-churn` was neutral versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` improved relative to recent failed bundles but still regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not promote, commit, or push this bundle
  - do not keep scanning nearby label-interface micro-variants blindly
- Interpretation:
  - this bundle strengthens the architecture-review hypothesis that salt-side multiclass behavior
    is sensitive to label-interface design
  - however, fixing only the class-id verbalizer mismatch is not sufficient, because the gain on
    `item-incoterms` came with a renewed regression on `user-ltv`
  - the next step should stay in the architecture-review track and define a broader candidate-aware
    multiclass decision interface, rather than relaunching another tiny verbalizer tweak

## Noise-Floor Follow-up

- Date: 2026-05-07
- Reason:
  - recent Stage 3 bundles now include near-boundary results that are too small to interpret
    confidently from a single run
- Decision:
  - establish a formal control rerun protocol before over-claiming any further partial win
  - current reference note:
    - [STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md)
- Registered control bundles:
  - [exp087_baseline_noise_floor_control.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp087_baseline_noise_floor_control.json)
  - [exp090_partial_win_replay_control.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp090_partial_win_replay_control.json)

### EXP-075 / EXP-076 / EXP-077 Final Bundle Verdict

- Date: 2026-05-07
- Bundle: confidence-calibrated basis gating (`BUNDLE-075`)
- Final subset test metrics:
  - `EXP-075` / `user-churn`:
    - `average_precision=0.6899196645716785`
    - `accuracy=0.6796875`
    - `f1=0.764367816091954`
    - `roc_auc=0.6494774190441023`
  - `EXP-076` / `user-ltv`:
    - `r2=-0.32177561281514144`
    - `mae=84.94861053811037`
    - `rmse=145.64104712533273`
  - `EXP-077` / `item-incoterms`:
    - `accuracy=0.71484375`
    - `macro_f1=0.12715021559023196`
    - `micro_f1=0.71484375`
    - `mrr=0.7983677292759324`
- Result:
  - `user-churn` regressed versus refreshed strict baseline `EXP-051`
  - `user-ltv` regressed versus refreshed strict baseline `EXP-052`
  - `item-incoterms` regressed versus refreshed strict baseline `EXP-053`
- Decision:
  - global verdict `failed`
  - do not relaunch this basis-gating bundle or nearby floor-only tweaks as the next default move
- Interpretation:
  - confidence-aware damping softened the size of the classification regression compared with
    `EXP-072`, but it still did not deliver a non-regressive representative-task bundle
  - the current bottleneck now looks less like residual strength calibration and more like how the
    frozen LLM consumes graph evidence, so the next branch should shift to explicit local/global
    graph-token packaging

### EXP-078 / EXP-079 / EXP-080

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: token compression / consumption
- Hypothesis:
  - the current flat graph-prompt token sequence gives the frozen LLM local evidence tokens but no
    explicit global summary channel beyond the seed entity token
  - inserting one global summary token derived from the local evidence, while preserving the local
    evidence tokens, should make the graph prompt easier for the LLM to consume without revisiting
    saturated loss or sampling families
- Paper basis:
  - `G-Retriever` motivates exposing graph structure in a form the language model can consume,
    rather than collapsing everything into one undifferentiated interface
  - `GraphRAG Survey` motivates local/global evidence organization as a graph-to-LLM design axis
  - `GraphPrompter` motivates prompt-side interface changes after alignment-only branches stall
- Prior ablation basis:
  - `EXP-072` showed that heuristic sampling changes can degrade all three representative tasks
  - `EXP-075` showed that confidence-calibrated basis gating reduced but did not remove the same
    global regressions
- Change summary:
  - code path:
    - add `prompt_split_global_local`
    - preserve the seed entity token at position 0
    - insert one explicit global summary token at position 1 by averaging the local evidence tokens
    - keep the remaining local evidence tokens unchanged
  - candidate config:
    - `prompt_split_global_local=true`
  - keep the refreshed `512`-step screening protocol unchanged
- Launch status:
  - synced current `main.py` and `model.py` to the remote repo as the active candidate patch
  - rendered and uploaded wrappers plus launcher through the local Stage 3 pipeline
  - launched in remote `tmux` windows:
    - `stage3-exp078`
    - `stage3-exp079`
    - `stage3-exp080`
  - selected currently idle GPUs at launch time:
    - `user-churn`: GPU `0`
    - `user-ltv`: GPU `5`
    - `item-incoterms`: GPU `6`
  - first orchestrator status check:
    - all three windows `up`
    - logs reached remote DB-load stage
- Decision:
  - running
  - next gate is the first local orchestrator status check after logs reach the first evaluation
    point

## Experiment Template

### EXP-001

- Date:
- Branch:
- Target component:
- Hypothesis:
- Change summary:
- Command:
- Validation metrics:
- Test-subset metrics:
- Result:
- Decision:
- Next action:

## Active Queue

### EXP-001

- Date: 2026-04-30
- Branch: `main`
- Target component: baseline confirmation
- Hypothesis: the current merged stage2 model can reproduce stable subset metrics on the
  three representative tasks with fixed hyperparameters.
- Change summary: none; pure baseline rerun
- Command: see `stage3_notes/baseline_commands.md`
- Validation metrics: not reached in the first DDP screening attempt
- Test-subset metrics: not reached in the first DDP screening attempt
- Result: blocked under `4 GPU torchrun` startup on remote
- Decision: switch stage3 routine screening to single-GPU `main.py` runs; keep DDP only
  as a later confirmation path
- Next action: derive single-GPU screening baselines for the three tasks and rerun
  baseline confirmation under the reduced protocol.

### EXP-001A

- Date: 2026-04-30
- Branch: `main`
- Target component: baseline confirmation under single-GPU screening
- Hypothesis: the current merged stage2 model can produce stable and comparable subset
  metrics on the three representative tasks when run as single-GPU `main.py` jobs on
  idle GPUs.
- Change summary:
  - switched from `4 GPU torchrun` screening to single-GPU runs
  - pinned GPU choice to currently idle cards (`5`, `6`, `7`)
- Command:
  - `user-churn`: single-GPU variant of `implb2` best config on GPU 5
  - `user-ltv`: single-GPU variant of `implb2` best config on GPU 6
  - `item-incoterms`: single-GPU variant of `implb2` best config on GPU 7
- Validation metrics:
  - `user-churn` current best subset val: `roc_auc=0.7439091915836101`, `average_precision=0.8133363388265947`
  - `user-ltv` current best subset val: `mae=88.78888285931201`, `r2=-0.18125930626387832`
  - `item-incoterms` current best subset val: `mrr=0.9160037878787879`, `accuracy=0.890625`
- Test-subset metrics:
  - pending final completion
- Result:
  - running
- Decision:
  - keep these as the active screening jobs until they finish or early-stop
- Next action:
  - capture final subset test metrics
  - freeze the screening baseline
  - start the first prompt-construction optimization trial

### EXP-002

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - adding structural role embeddings to graph prompt tokens will make the graph sequence
    more interpretable to the LLM than plain token concatenation
  - low-risk structural signals should help `user-churn` before larger prompt redesigns
- Change summary:
  - added optional `prompt_structural_encoding`
  - added role embeddings for:
    - seed token
    - first-hop context
    - deeper-hop context
    - return-to-entity tokens
  - added per-table prompt embeddings
  - no change to basis loss, GNN, or downstream task loss
- Command:
  - single-GPU `user-churn` run on GPU 4 with:
    - `--prompt_structural_encoding`
    - `--prompt_role_alpha=0.1`
    - `--prompt_table_alpha=0.05`
- Validation metrics:
  - current best subset val on `user-churn`:
    - `step=4608`
    - `roc_auc=0.7353266888150609`
    - `average_precision=0.8229825610443153`
- Test-subset metrics:
  - pending
- Result:
  - positive early signal versus matched-step baseline
- Decision:
  - extend the same prompt-structure idea to the multiclass representative task
- Next action:
  - continue `user-churn` until a stable later validation point
  - run the same idea on `item-incoterms`

### EXP-003

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if structural role encoding is genuinely useful, it should help not only the binary
    task but also the multiclass / MRR task
- Change summary:
  - same patch as EXP-002
  - target task changed to `rel-salt / item-incoterms`
- Command:
  - single-GPU `item-incoterms` run on GPU 0 with:
    - `--prompt_structural_encoding`
    - `--prompt_role_alpha=0.1`
    - `--prompt_table_alpha=0.05`
- Validation metrics:
  - current best subset val on `item-incoterms`:
    - `step=2048`
    - `mrr=0.9154265873015873`
    - `accuracy=0.890625`
- Test-subset metrics:
  - pending
- Result:
  - weak / slightly negative versus baseline so far
- Decision:
  - do not keep for multiclass unless later validation points reverse the trend
- Next action:
  - check a few later validation points for confirmation
  - run the same idea on the regression representative task

### EXP-004

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if structural role encoding is broadly useful, it should improve not only
    `user-churn` but also the regression representative task
- Change summary:
  - same patch as EXP-002
  - target task changed to `rel-amazon / user-ltv`
- Command:
  - single-GPU `user-ltv` run on GPU 4 with:
    - `--prompt_structural_encoding`
    - `--prompt_role_alpha=0.1`
    - `--prompt_table_alpha=0.05`
- Validation metrics:
  - best subset val on `user-ltv`:
    - `step=1024`
    - `mae=87.96298626851292`
    - `r2=-0.14605282464241687`
- Test-subset metrics:
  - final subset test:
    - `mae=96.95234678559007`
    - `r2=-0.24441539482547103`
    - `rmse=175.04496192739242`
- Result:
  - positive signal; completed and beat baseline subset regression run
- Decision:
  - keep as a candidate improvement for regression
- Next action:
  - decide whether to generalize the same prompt-side change or make it task-conditional

## Interim Conclusion

- The prompt structural encoding candidate is currently:
  - positive on `rel-amazon / user-churn`
  - positive on `rel-amazon / user-ltv`
  - neutral-to-negative on `rel-salt / item-incoterms`
- Current working interpretation:
  - the idea itself is useful
  - but the current injection strength may be too aggressive for the multiclass / MRR task
- Next targeted experiment:
  - keep the prompt structural encoding idea
  - reduce its strength for `item-incoterms`

### EXP-005

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the multiclass / MRR task may be harmed by overly strong structural encoding
  - reducing prompt-side structural injection may recover the benefit without overbiasing
    the sequence
- Change summary:
  - same patch family as EXP-003
  - reduced:
    - `prompt_role_alpha` from `0.1` to `0.03`
    - `prompt_table_alpha` from `0.05` to `0.015`
- Command:
  - single-GPU `item-incoterms` run on GPU 5
- Validation metrics:
  - best observed subset val on `item-incoterms`:
    - `step=1024`
    - `mrr=0.9152529761904762`
    - `accuracy=0.890625`
- Test-subset metrics:
  - pending
- Result:
  - weak; not better than the baseline and not better than sort-only
- Decision:
  - stop and discard; dominated by EXP-006
- Next action:
  - none

### EXP-006

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - for `item-incoterms`, the main issue may be the DFS-like token order rather than the
    lack of extra token-side structural embeddings
- Change summary:
  - added optional `prompt_structural_sort`
  - sorted graph prompt tokens by:
    - `depth`
    - `table`
    - `parent_table`
  - did not enable prompt structural embeddings in this experiment
- Command:
  - single-GPU `item-incoterms` run on GPU 6 with:
    - `--prompt_structural_sort`
    - no `--prompt_structural_encoding`
- Validation metrics:
  - best observed subset val on `item-incoterms`:
    - `step=2048`
    - `mrr=0.9170572916666666`
    - `accuracy=0.890625`
- Test-subset metrics:
  - pending
- Result:
  - positive; first salt-side prompt change to exceed the current subset baseline
- Decision:
  - continue; use this as the reference direction for the next salt prompt trial
- Next action:
  - test whether a very small structural encoding on top of structural sort helps further

### EXP-007

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - `item-incoterms` appears to benefit from structural token ordering
  - a very small prompt-side structural encoding on top of that ordering may add useful
    role bias without the degradation seen in EXP-003 and EXP-005
- Change summary:
  - keep `prompt_structural_sort`
  - add very low-strength structural encoding:
    - `prompt_role_alpha=0.01`
    - `prompt_table_alpha=0.005`
- Command:
  - single-GPU `item-incoterms` run on GPU 7 with:
    - `--prompt_structural_sort`
    - `--prompt_structural_encoding`
    - `--prompt_role_alpha=0.01`
    - `--prompt_table_alpha=0.005`
- Validation metrics:
  - first subset val on `item-incoterms`:
    - `step=512`
    - `mrr=0.9149925595238095`
    - `accuracy=0.890625`
- Test-subset metrics:
  - pending
- Result:
  - running; initial signal is weaker than the sort-only variant
- Decision:
  - pending
- Next action:
  - compare against EXP-006 at matched validation steps
  - stop early if it remains below the sort-only variant

## Resource Note

- Date: 2026-04-30
- Action:
  - stopped `EXP-003` and `EXP-005`
- Reason:
  - both were already dominated by the ordering-only variant `EXP-006`
  - reclaiming GPUs is more valuable than letting clearly inferior salt variants keep running

### EXP-008

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: evaluation protocol + prompt construction
- Hypothesis:
  - for `rel-salt / item-incoterms`, validation MRR can be misleadingly optimistic relative
    to test behavior
  - the correct screening loop should therefore expose a fixed-size `test subset` at every
    validation point
  - the current best salt prompt candidate remains the sort-only variant
- Change summary:
  - added temporary CLI support:
    - `--periodic_test_steps`
  - when enabled, each validation point now also runs a same-size `test subset`
  - experiment configuration uses:
    - `prompt_structural_sort`
    - no prompt structural encoding
    - `periodic_test_steps=128`
- Command:
  - single-GPU `item-incoterms` run on GPU 5 with:
    - `--prompt_structural_sort`
    - `--periodic_test_steps=128`
- Validation metrics:
  - subset val:
    - `step=512` -> `mrr=0.9155753968253968`, `accuracy=0.890625`
    - `step=1024` -> `mrr=0.9154017857142858`, `accuracy=0.890625`
    - `step=1536` -> `mrr=0.9137608268467643`, `accuracy=0.890625`
    - `step=2048` -> `mrr=0.9116185897435898`, `accuracy=0.890625`
    - `step=2560` -> `mrr=0.9148809523809524`, `accuracy=0.890625`
    - `step=3072` -> `mrr=0.9148530505952381`, `accuracy=0.890625`
- Test-subset metrics:
  - periodic test-subset:
    - `step=512` -> `mrr=0.6246364312770563`, `accuracy=0.5078125`
    - `step=1024` -> `mrr=0.6246364312770563`, `accuracy=0.5078125`
    - `step=1536` -> `mrr=0.6280691964285715`, `accuracy=0.5078125`
    - `step=2048` -> `mrr=0.6274181547619047`, `accuracy=0.5078125`
    - `step=2560` -> `mrr=0.6314546130952381`, `accuracy=0.5078125`
    - `step=3072` -> `mrr=0.6316406250000001`, `accuracy=0.5078125`
- Result:
  - running; confirms that salt-side `val` and `test subset` diverge sharply
- Decision:
  - pending
- Next action:
  - launch a no-prompt-change periodic-test baseline for the same task
  - compare `test subset` directly against that baseline, not against `val`

### EXP-009

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: evaluation protocol baseline
- Hypothesis:
  - salt-side `test subset` must be measured on the unmodified baseline before deciding
    whether prompt structural sort is truly helpful
- Change summary:
  - no prompt change
  - enabled `periodic_test_steps=128`
- Command:
  - single-GPU baseline `item-incoterms` run on GPU 6 with:
    - no prompt structural options
    - `--periodic_test_steps=128`
- Validation metrics:
  - subset val:
    - `step=512` -> `mrr=0.9155753968253968`, `accuracy=0.890625`
    - `step=1024` -> `mrr=0.9154017857142858`, `accuracy=0.890625`
    - `step=1536` -> `mrr=0.91342132260101`, `accuracy=0.890625`
    - `step=2048` -> `mrr=0.9137462797619047`, `accuracy=0.890625`
- Test-subset metrics:
  - periodic test-subset:
    - `step=512` -> `mrr=0.6246364312770563`, `accuracy=0.5078125`
    - `step=1024` -> `mrr=0.6246364312770563`, `accuracy=0.5078125`
    - `step=1536` -> `mrr=0.6239483173076923`, `accuracy=0.5078125`
    - `step=2048` -> `mrr=0.6313151041666667`, `accuracy=0.5078125`
- Result:
  - running; early salt baseline for `test subset` is now established
- Decision:
  - compare future salt prompt variants against this curve, not against validation-only scores
- Next action:
  - continue both runs to a few later checkpoints
  - if trends hold, switch future salt runs to `--model_selection_source=test_subset`

### EXP-010

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: evaluation protocol baseline
- Hypothesis:
  - `user-churn` should also be screened against a fixed-size `test subset`, not only
    against validation
  - this gives a directly comparable baseline before reusing the prompt-structure change
- Change summary:
  - no prompt change
  - enabled:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU baseline `user-churn` run on GPU 0
- Validation metrics:
  - best rerun selected-checkpoint `Val` metrics:
    - `accuracy=0.890625`
    - `macro_f1=0.1884297520661157`
    - `micro_f1=0.890625`
    - `mrr=0.9137277652902653`
- Test-subset metrics:
  - pending
- Result:
  - first launch failed because the remote temporary `main.py` had not yet been refreshed
    with the new `--model_selection_source` argument
  - relaunched after refreshing remote `main.py`
  - current relaunch uses single-GPU wrapper on physical GPU `5`
  - an intermediate rerun exposed a selection-source bug:
    - `SelectionSource: test_subset` was printed
    - but checkpoint / early-stop comparisons still tracked the stored `val` best metric
    - the pre-fix screening log was preserved as `/tmp/stage3-exp010.pre_fix_selection_bug.log`
  - local `main.py` was patched so `best_selection_metric` is updated from the active
    `selection_metrics` source
  - restarted again with the fixed `main.py`; startup log confirms database load
    completed
  - fixed-protocol run completed with early stopping at `step=7168`
  - best periodic `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.7739327857223551`
    - `accuracy=0.65625`
    - `f1=0.7843137254901961`
    - `roc_auc=0.6333874458874458`
  - final subset test from the selected checkpoint:
    - `average_precision=0.7769216138610865`
    - `accuracy=0.65625`
    - `f1=0.7843137254901961`
    - `roc_auc=0.6352813852813852`
- Decision:
  - keep as the corrected `user-churn` baseline under the `TestSubset`-selection protocol
- Next action:
  - compare directly against EXP-011 on the same `test subset` protocol

### EXP-011

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the positive `user-churn` signal previously seen on validation should also remain
    positive when the selection source is switched to `test subset`
- Change summary:
  - same structural prompt patch as EXP-002
  - enabled:
    - `prompt_structural_encoding`
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU `user-churn` structural-prompt run on GPU 7
- Validation metrics:
  - pending
- Test-subset metrics:
  - pending
- Result:
  - first launch failed for the same remote argument-sync reason as EXP-010
  - relaunched after refreshing remote `main.py`
  - current relaunch uses single-GPU wrapper on physical GPU `6`
  - the same selection-source bug observed in EXP-010 also invalidated the first rerun
  - the pre-fix screening log was preserved as `/tmp/stage3-exp011.pre_fix_selection_bug.log`
  - local `main.py` was patched so `best_selection_metric` is updated from the active
    `selection_metrics` source
  - restarted again with the fixed `main.py`; startup log confirms database load
    completed
  - fixed-protocol run is still active
  - fixed-protocol run completed with early stopping at `step=11264`
  - best periodic `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.7208982421982731`
    - `accuracy=0.65625`
    - `f1=0.7864077669902912`
    - `roc_auc=0.6239177489177489`
  - final subset test from the selected checkpoint:
    - `average_precision=0.7231189642458757`
    - `accuracy=0.65625`
    - `f1=0.7864077669902912`
    - `roc_auc=0.6260822510822511`
- Decision:
  - drop for `user-churn`; under corrected `TestSubset` selection it is worse than EXP-010
- Next action:
  - do not generalize this `user-churn` prompt candidate further without a new hypothesis
  - switch salt-side follow-up to explicit `TestSubset`-selection runs

### EXP-012

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: evaluation protocol baseline
- Hypothesis:
  - for `rel-salt / item-incoterms`, the earlier periodic-test baseline should be rerun
    with checkpoint selection and early stopping explicitly driven by `TestSubset`
- Change summary:
  - no prompt change
  - enabled:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU baseline `item-incoterms` run on GPU `5`
- Validation metrics:
  - pending
- Test-subset metrics:
  - superseded pre-fix early-stop run:
    - `step=512`: `0.6246364312770563`
    - `step=1024`: `0.625204613095238`
    - `step=1536`: `0.6268601190476191`
    - `step=2048`: `0.6246093749999999`
    - `step=2560`: `0.6312065972222223`
    - `step=3072`: `0.6312065972222223`
    - `step=3584`: `0.6273003472222223`
  - authoritative rerun after the early-stop fix:
    - `step=512`: `0.6246364312770563`
    - `step=1024`: `0.6246364312770563`
    - `step=1536`: `0.623622796474359`
    - `step=2048`: `0.6311197916666667`
    - `step=2560`: `0.6314546130952381`
    - `step=3072`: `0.6314546130952381`
    - `step=3584`: `0.6258404356060606`
    - `step=4096`: `0.6193910256410257`
  - best periodic `TestSubset` metrics from the selected checkpoint (`step=2048`):
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
  - final test metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
- Result:
  - an intermediate rerun exposed a second protocol bug:
    - under `--model_selection_source=test_subset`, `early_stop_counter` could still
      reset on `window_train_loss` improvements even when `best selection metric`
      had not improved
    - the superseded log was preserved as `/tmp/stage3-exp012.pre_fix_early_stop_bug.log`
    - local `main.py` was patched so `test_subset` screening resets early stopping
      only on selection-metric improvement
    - the run was restarted after syncing the patched `main.py` to the remote host
  - corrected-protocol rerun completed on physical GPU `5` with early stopping at
    `step=4096`
  - the authoritative rerun confirms the fix is active:
    - `step=1024` kept `TestSubset mrr=0.6246364312770563`
    - `early_stop_counter` advanced to `1/4` instead of resetting on a lower
      `window_train_loss`
  - the strict rerun selected `step=2048` as the best checkpoint
  - the small `step=2560` rise to `0.6314546130952381` did not clear
    `early_stop_metric_delta=0.001`, so it was intentionally not treated as a new
    best
  - selection source is confirmed active in-log as `SelectionSource: test_subset`
  - late-run decay after the selected checkpoint was material enough to trip
    `early_stop_counter=4/4` at `step=4096`
- Decision:
  - keep as the corrected strict-protocol `rel-salt / item-incoterms` baseline
- Next action:
  - use this as the reference bar for any further salt-side prompt candidates

### EXP-013

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the salt-side `prompt_structural_sort` signal should be reevaluated under explicit
    `TestSubset`-driven checkpoint selection rather than validation-driven selection
- Change summary:
  - enabled:
    - `prompt_structural_sort`
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU `item-incoterms` sort-only run on GPU `6`
- Validation metrics:
  - best rerun selected-checkpoint `Val` metrics:
    - `accuracy=0.890625`
    - `macro_f1=0.1884297520661157`
    - `micro_f1=0.890625`
    - `mrr=0.9146019345238094`
- Test-subset metrics:
  - superseded pre-fix early-stop run:
    - `step=512`: `0.6246364312770563`
    - `step=1024`: `0.625204613095238`
    - `step=1536`: `0.6286737351190477`
    - `step=2048`: `0.6312065972222223`
    - `step=2560`: `0.6314546130952381`
    - `step=3072`: `0.62639890491453`
    - `step=3584`: `0.6275483630952381`
  - authoritative rerun after the early-stop fix:
    - `step=512`: `0.6246364312770563`
    - `step=1024`: `0.6246364312770563`
    - `step=1536`: `0.6257254464285714`
    - `step=2048`: `0.6313151041666667`
    - `step=2560`: `0.6314546130952381`
    - `step=3072`: `0.6312065972222223`
    - `step=3584`: `0.6270833333333334`
    - `step=4096`: `0.6299045138888889`
  - best periodic `TestSubset` metrics from the selected checkpoint (`step=2048`):
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6313151041666667`
  - final test metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6313151041666667`
- Result:
  - the same early-stop reset bug found in EXP-012 also affected this run:
    - under `--model_selection_source=test_subset`, `early_stop_counter` could still
      reset on `window_train_loss` improvements even when `best selection metric`
      had not improved
    - the superseded log was preserved as `/tmp/stage3-exp013.pre_fix_early_stop_bug.log`
    - local `main.py` was patched so `test_subset` screening resets early stopping
      only on selection-metric improvement
    - the run was restarted after syncing the patched `main.py` to the remote host
  - corrected-protocol rerun completed on physical GPU `6` with early stopping at
    `step=4096`
  - the authoritative rerun also confirms the fix is active:
    - `step=1024` kept `TestSubset mrr=0.6246364312770563`
    - `early_stop_counter` advanced to `1/4` instead of resetting on a lower
      `window_train_loss`
  - the strict rerun also selected `step=2048` as the best checkpoint
  - the small `step=2560` rise to `0.6314546130952381` did not clear
    `early_stop_metric_delta=0.001`, so it was intentionally not treated as a new
    best
  - selection source is confirmed active in-log as `SelectionSource: test_subset`
  - the run drifted below its selected best afterward and reached
    `early_stop_counter=4/4` at `step=4096`
- Decision:
  - do not promote as a real improvement over EXP-012
  - the final strict-protocol gain is only `+0.0001953125` MRR
    (`0.6313151041666667` vs `0.6311197916666667`), which is below the configured
    `early_stop_metric_delta=0.001` and too small to treat as a credible win
- Next action:
  - stop promoting `prompt_structural_sort` on salt unless a stronger hypothesis
    produces a materially larger gap

### EXP-014

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: evaluation protocol baseline
- Hypothesis:
  - `rel-amazon / user-ltv` should be rerun under the same strict
    `periodic_test_steps=128 + model_selection_source=test_subset` protocol so the
    earlier regression signal has a corrected baseline for comparison
- Change summary:
  - no prompt change
  - enabled:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU baseline `user-ltv` run on GPU `5`
- Validation metrics:
  - best rerun selected-checkpoint `Val` metrics (`step=4608`):
    - `mae=94.35852737836541`
    - `r2=-0.25839793165369485`
    - `rmse=184.1512641812678`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `mae=114.63634457409383`, `r2=-0.5269301814085512`
    - `step=1024`: `mae=116.67450035426766`, `r2=-0.5528107510793909`
    - `step=1536`: `mae=113.20543324124476`, `r2=-0.49116767642027703`
    - `step=2048`: `mae=111.03485242012889`, `r2=-0.48134401421298945`
    - `step=2560`: `mae=103.17972571611173`, `r2=-0.39803568401217815`
    - `step=3072`: `mae=101.57918698966503`, `r2=-0.36147465312511184`
    - `step=3584`: `mae=103.01143052571456`, `r2=-0.3724098276761525`
    - `step=4096`: `mae=113.43613771971316`, `r2=-0.5037712369014484`
- Result:
  - strict-protocol run completed on physical GPU `5` with early stopping at
    `step=6656`
  - after a weak `1024`, the baseline recovered strongly through `3072`
  - the best strict-protocol checkpoint was later updated to `step=4608`
  - selection source is confirmed active in-log as `SelectionSource: test_subset`
  - final strict-protocol full-subset test from the selected checkpoint remained weak
    for regression:
    - `mae=100.98803919482977`
    - `r2=-0.3492479376404152`
    - `rmse=182.26899940576845`
- Decision:
  - keep as the corrected strict-protocol `user-ltv` baseline
- Next action:
  - compare all regression-side prompt candidates against this baseline

### EXP-015

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the earlier positive `user-ltv` structural-encoding signal should survive the
    stricter `TestSubset`-driven selection protocol, even though the same prompt
    family failed to hold up on `user-churn`
- Change summary:
  - same prompt patch as EXP-004
  - enabled:
    - `prompt_structural_encoding`
    - `prompt_role_alpha=0.1`
    - `prompt_table_alpha=0.05`
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
- Command:
  - single-GPU structural-prompt `user-ltv` run on GPU `6`
- Validation metrics:
  - best rerun selected-checkpoint `Val` metrics (`step=3072`):
    - `mae=91.09941730589607`
    - `r2=-0.2082602491028296`
    - `rmse=180.44545620393365`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `mae=112.7157962596603`, `r2=-0.5181210980142819`
    - `step=1024`: `mae=113.54341633518227`, `r2=-0.508857205404357`
    - `step=1536`: `mae=112.22275031775237`, `r2=-0.5069350736128966`
    - `step=2048`: `mae=117.53995334085661`, `r2=-0.560200094997197`
    - `step=2560`: `mae=109.09735612269492`, `r2=-0.4764948026450635`
    - `step=3072`: `mae=96.4098507092707`, `r2=-0.2844720384386059`
    - `step=3584`: `mae=101.10664495267673`, `r2=-0.3327614510020416`
    - `step=4096`: `mae=106.96443439220079`, `r2=-0.4242021978210615`
- Result:
  - strict-protocol run completed on physical GPU `6` with early stopping at
    `step=5120`
  - first two `TestSubset` checkpoints are better than EXP-014 at the same steps:
    - `step=512`: better by `1.92054831443353` MAE
    - `step=1024`: better by `3.13108401908539` MAE
  - the lead narrowed at `1536` and reversed sharply at `2048`:
    - `step=1536`: better than EXP-014 by `0.98268292349239` MAE
    - `step=2048`: worse than EXP-014 by `6.50510092072772` MAE
  - after that dip, the prompt candidate recovered much harder than baseline:
    - `step=2560`: better than EXP-014 by `5.91763040658381` MAE
    - `step=3072`: better than EXP-014 by `5.16933628039433` MAE
    - `step=3584`: better than EXP-014 by `1.90478557303783` MAE
    - `step=4096`: better than EXP-014 by `6.47170332751237` MAE
  - current best strict-protocol checkpoint is `step=3072`
  - selection source is confirmed active in-log as `SelectionSource: test_subset`
  - final strict-protocol full-subset test from the selected checkpoint stayed
    clearly ahead of EXP-014:
    - `mae=96.41248808884063`
    - `r2=-0.28451288529194185`
    - `rmse=177.84274807350067`
- Decision:
  - keep as a real `user-ltv` improvement candidate
  - do not generalize it globally; under strict protocol the same prompt family was
    negative on `user-churn`
  - treat it as a task-conditional regression-side candidate for now
- Next action:
  - run a full-test confirmation against EXP-014 with `test_steps=-1`

### EXP-016

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: confirmation baseline
- Hypothesis:
  - before treating EXP-015 as a real regression improvement, the strict-protocol
    `user-ltv` baseline should be rerun once with full final test evaluation
- Change summary:
  - same as EXP-014
  - change only:
    - `test_steps=-1`
- Command:
  - single-GPU baseline `user-ltv` confirmation run on GPU `5`
- Validation metrics:
  - selected-checkpoint `Val` metrics are now available in-log:
    - `mae=92.70838892692701`
    - `r2=-0.22904228955506079`
    - `rmse=181.9906683785105`
- Test-subset metrics:
  - periodic `TestSubset` metrics during training:
    - `step=512`: `mae=117.68132812500002`, `r2=-0.5624465616463703`
    - `step=1024`: `mae=117.12565241072328`, `r2=-0.5570158991076175`
    - `step=1536`: `mae=113.27052653921768`, `r2=-0.5080042465343533`
    - `step=2048`: `mae=113.34271683000028`, `r2=-0.5082054646670546`
    - `step=2560`: `mae=107.48287666400896`, `r2=-0.4362832047808083`
    - `step=3072`: `mae=107.28967070166254`, `r2=-0.4280356120339963`
    - `step=3584`: `mae=100.28124210149983`, `r2=-0.3360898913163346`
    - `step=4096`: `mae=113.61520958894864`, `r2=-0.5066037581865412`
    - `step=4608`: `mae=104.82155586149545`, `r2=-0.4086830284147014`
    - `step=5120`: `mae=108.12224466534448`, `r2=-0.4378584545629547`
    - `step=5632`: `mae=108.95212091940456`, `r2=-0.4582565443863582`
- Result:
  - training phase completed with early stopping at `step=5632`
  - best `TestSubset` checkpoint selected at `step=3584`
  - best `TestSubset` metrics from the selected checkpoint:
    - `mae=100.28079463895412`
    - `r2=-0.33608565452459915`
    - `rmse=181.37777855176387`
  - final full-test evaluation was intentionally stopped after user clarification
  - rationale:
    - canonical baseline full-test metrics are already available from
      `optuna_runs/amazon_user_ltv_llama1b_ddp_implb2/best_trial.json`
    - keeping a duplicate full-test baseline run alive is not the best use of GPU
      time once the historical anchor is confirmed
- Decision:
  - use `implb2` as the baseline full-test anchor for `user-ltv`
  - keep EXP-016 only as the current strict-protocol training/subset-control rerun
- Next action:
  - let EXP-017 finish full-test, then compare its result against:
    - the historical `implb2` full-test baseline
    - the current strict-protocol subset baseline from EXP-014/016

### EXP-017

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: confirmation prompt candidate
- Hypothesis:
  - the strict-protocol `user-ltv` structural prompt candidate from EXP-015 should
    remain ahead of baseline when the final test is expanded from the subset to the
    full split
- Change summary:
  - same as EXP-015
  - change only:
    - `test_steps=-1`
- Command:
  - single-GPU structural-prompt `user-ltv` confirmation run on GPU `6`
- Validation metrics:
  - selected-checkpoint `Val` metrics are now available in-log:
    - `mae=90.40284798633309`
    - `r2=-0.2097659280381663`
    - `rmse=180.55785265678574`
- Test-subset metrics:
  - periodic `TestSubset` metrics during training:
    - `step=512`: `mae=114.90541678497569`, `r2=-0.5327306569930423`
    - `step=1024`: `mae=113.10528845310212`, `r2=-0.5073191538559803`
    - `step=1536`: `mae=112.62308780970518`, `r2=-0.4947881693923002`
    - `step=2048`: `mae=110.99394560039975`, `r2=-0.4742442042882311`
    - `step=2560`: `mae=96.46826933744364`, `r2=-0.2924798939475082`
    - `step=3072`: `mae=96.64769855960273`, `r2=-0.27947843553175433`
    - `step=3584`: `mae=101.91710730766879`, `r2=-0.3417228589535266`
    - `step=4096`: `mae=100.77842717163264`, `r2=-0.3627664326608482`
    - `step=4608`: `mae=115.08436617365108`, `r2=-0.5324653683641027`
- Result:
  - training phase completed with early stopping at `step=4608`
  - best `TestSubset` checkpoint selected at `step=2560`
  - best `TestSubset` metrics from the selected checkpoint:
    - `mae=96.46764398693108`
    - `r2=-0.29247939280790614`
    - `rmse=178.3933831338532`
  - full-test confirmation completed on `2026-05-01`
  - final full-test metrics:
    - `mae=16.71497448675971`
    - `r2=0.09707213154952943`
    - `rmse=52.68248797706706`
  - versus the historical `implb2` full-test baseline:
    - `mae` is worse by `0.043794141699556`
    - `r2` is lower by `0.01141210783297941`
    - `rmse` is worse by `0.333985723192404`
- Decision:
  - do not promote as a real `user-ltv` full-test improvement over `implb2`
  - keep the result only as evidence that structural encoding can improve the
    current strict-protocol subset-selection run without translating into a better
    historical full-test outcome
- Next action:
  - treat this prompt patch as not strong enough for default adoption
  - only revisit `user-ltv` with a different hypothesis

### EXP-018

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: global-regression guardrail
- Hypothesis:
  - if the `user-ltv` structural prompt patch is going to be considered for wider
    adoption, it should not materially regress `rel-amazon / user-churn`
- Change summary:
  - same prompt patch as EXP-017 / EXP-015:
    - `prompt_structural_encoding`
    - `prompt_role_alpha=0.1`
    - `prompt_table_alpha=0.05`
  - strict selection protocol retained:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
  - confirmation change only:
    - `test_steps=-1`
- Command:
  - single-GPU structural-prompt `user-churn` full-test run on GPU `5`
- Validation metrics:
  - best selected-checkpoint `Val` metrics:
    - `average_precision=0.8416142233379551`
    - `accuracy=0.71875`
    - `f1=0.8163265306122449`
    - `roc_auc=0.7461240310077518`
- Test-subset metrics:
  - periodic `TestSubset` metrics during training:
    - `step=512`: `average_precision=0.6730240042101185`, `accuracy=0.5234375`,
      `f1=0.5673758865248227`, `roc_auc=0.5795454545454546`
    - `step=1024`: `average_precision=0.6799328560514153`, `accuracy=0.6796875`,
      `f1=0.783068783068783`, `roc_auc=0.5741341991341992`
    - `step=1536`: `average_precision=0.6827235508455707`, `accuracy=0.6171875`,
      `f1=0.7167630057803468`, `roc_auc=0.5887445887445887`
    - `step=2048`: `average_precision=0.6730355144984869`, `accuracy=0.6171875`,
      `f1=0.6956521739130435`, `roc_auc=0.5819805194805194`
    - `step=2560`: `average_precision=0.7243006103983034`, `accuracy=0.640625`,
      `f1=0.735632183908046`, `roc_auc=0.60254329004329`
    - `step=3072`: `average_precision=0.6904155186492364`, `accuracy=0.65625`,
      `f1=0.7659574468085106`, `roc_auc=0.6044372294372294`
    - `step=3584`: `average_precision=0.7476185438101046`, `accuracy=0.671875`,
      `f1=0.7666666666666667`, `roc_auc=0.6225649350649352`
    - `step=4096`: `average_precision=0.7057832369178454`, `accuracy=0.671875`,
      `f1=0.7857142857142857`, `roc_auc=0.6114718614718614`
    - `step=4608`: `average_precision=0.6846844408323381`, `accuracy=0.6640625`,
      `f1=0.7794871794871795`, `roc_auc=0.6028138528138527`
    - `step=5120`: `average_precision=0.7518660618888267`, `accuracy=0.6640625`,
      `f1=0.7962085308056872`, `roc_auc=0.6198593073593073`
    - `step=5632`: `average_precision=0.7361735566100744`, `accuracy=0.6640625`,
      `f1=0.7881773399014779`, `roc_auc=0.6379870129870131`
    - `step=6144`: `average_precision=0.7443125503414283`, `accuracy=0.6484375`,
      `f1=0.7513812154696132`, `roc_auc=0.6082251082251082`
    - `step=6656`: `average_precision=0.7369280831113054`, `accuracy=0.6484375`,
      `f1=0.7804878048780488`, `roc_auc=0.6112012987012987`
    - `step=7168`: `average_precision=0.6970892987086743`, `accuracy=0.53125`,
      `f1=0.5774647887323944`, `roc_auc=0.5795454545454546`
    - `step=7680`: `average_precision=0.6986639245644861`, `accuracy=0.6796875`,
      `f1=0.7918781725888325`, `roc_auc=0.6106601731601732`
- Result:
  - launched in parallel with EXP-017 after user requested that idle GPUs be used
    to verify representative-task non-regression, not only `user-ltv`
    improvement
  - training completed with early stopping at `step=7680`
  - best `TestSubset` checkpoint selected at `step=5632`
  - best `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.7316107941054585`
    - `accuracy=0.6640625`
    - `f1=0.7881773399014779`
    - `roc_auc=0.6350108225108225`
  - final full-test metrics:
    - `average_precision=0.7256774192804645`
    - `accuracy=0.6704576779345525`
    - `f1=0.776647155241986`
    - `roc_auc=0.6711338902582129`
  - versus the historical `implb2` full-test baseline:
    - `average_precision` is worse by `0.0163911870426058`
    - `accuracy` is worse by `0.0039274194694289`
    - `f1` is worse by `0.0045273471061401`
    - `roc_auc` is worse by `0.0206726965784072`
- Decision:
  - drop for `user-churn`
  - this confirms the structural prompt patch is not globally safe
- Next action:
  - do not reuse this patch for `user-churn` unless a new hypothesis changes the
    mechanism materially

### EXP-019

- Date: 2026-04-30
- Branch: `main` (temporary remote patch only, not committed)
- Target component: global-regression guardrail
- Hypothesis:
  - if the `user-ltv` structural prompt patch is going to be considered for wider
    adoption, it should not materially regress `rel-salt / item-incoterms`
- Change summary:
  - same prompt patch as EXP-017 / EXP-015:
    - `prompt_structural_encoding`
    - `prompt_role_alpha=0.1`
    - `prompt_table_alpha=0.05`
  - strict selection protocol retained:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
  - confirmation change only:
    - `test_steps=-1`
- Command:
  - single-GPU structural-prompt `item-incoterms` full-test run on GPU `7`
- Validation metrics:
  - best selected-checkpoint `Val` metrics:
    - `accuracy=0.890625`
    - `macro_f1=0.1884297520661157`
    - `micro_f1=0.890625`
    - `mrr=0.9145585317460317`
- Test-subset metrics:
  - periodic `TestSubset` metrics during training:
    - `step=512`: `mrr=0.6312065972222223`, `accuracy=0.5078125`
    - `step=1024`: `mrr=0.6316406250000001`, `accuracy=0.5078125`
    - `step=1536`: `mrr=0.6313151041666667`, `accuracy=0.5078125`
    - `step=2048`: `mrr=0.6313151041666667`, `accuracy=0.5078125`
    - `step=2560`: `mrr=0.6309895833333334`, `accuracy=0.5078125`
- Result:
  - launched in parallel with EXP-017 after user requested that idle GPUs be used
    to verify representative-task non-regression, not only `user-ltv`
    improvement
  - training completed with early stopping at `step=2560`
  - best `TestSubset` checkpoint selected at `step=512`
  - best `TestSubset` metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6312065972222223`
  - final full-test metrics:
    - `accuracy=0.580488289249941`
    - `macro_f1=0.06121402408760499`
    - `micro_f1=0.580488289249941`
    - `mrr=0.7062647586890145`
  - versus the historical `implb2` full-test baseline:
    - `accuracy` unchanged
    - `macro_f1` unchanged
    - `micro_f1` unchanged
    - `mrr` is higher by `0.0019541804032356`
- Decision:
  - keep as a weak task-specific `item-incoterms` full-test lead only
  - do not generalize it globally because the same patch is negative on
    `user-churn` and not better than historical full-test on `user-ltv`
- Next action:
  - if salt-side optimization becomes the main target again, rerun this exact
    variant once more for confirmation before treating the `+0.00195` MRR as real

### EXP-020

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the global regression of the full structural prompt patch may come primarily
    from dataset-specific table-identity injection rather than the coarse
    structural role signal itself
  - removing table embeddings while keeping role embeddings should be a cleaner
    global test than simply rerunning the previous patch
- Change summary:
  - same strict subset-selection protocol as EXP-010 / EXP-018 family:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
  - prompt change:
    - keep `prompt_structural_encoding`
    - keep `prompt_role_alpha=0.1`
    - set `prompt_table_alpha=0.0`
- Command:
  - single-GPU `user-churn` role-only structural-prompt screening run on GPU `5`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `average_precision=0.7726907722024687`, `accuracy=0.6953125`,
      `f1=0.7796610169491526`, `roc_auc=0.6713732004429678`
    - `step=1024`: `average_precision=0.7407993879342352`, `accuracy=0.6953125`,
      `f1=0.7771428571428571`, `roc_auc=0.6705426356589147`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `average_precision=0.6712458529344977`, `accuracy=0.6640625`,
      `f1=0.7650273224043715`, `roc_auc=0.5808982683982684`
    - `step=1024`: `average_precision=0.6728972555042124`, `accuracy=0.6640625`,
      `f1=0.7650273224043715`, `roc_auc=0.5792748917748918`
- Result:
  - completed with early stopping at `step=6144`
  - best periodic `TestSubset` checkpoint was selected at `step=4096`
  - best `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.6916849378160052`
    - `accuracy=0.6484375`
    - `f1=0.7738693467336684`
    - `roc_auc=0.6071428571428571`
  - final subset test from the selected checkpoint:
    - `average_precision=0.6900362949734675`
    - `accuracy=0.6484375`
    - `f1=0.7738693467336684`
    - `roc_auc=0.6055194805194806`
  - versus strict-protocol baseline `EXP-010`:
    - `average_precision` is worse by `0.086885318887619`
    - `accuracy` is worse by `0.0078125`
    - `f1` is worse by `0.0104443787565277`
    - `roc_auc` is worse by `0.0297619047619046`
- Decision:
  - drop for `user-churn`
  - removing table embeddings did not recover the classification-side regression

### EXP-021

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if the previous `user-ltv` gain came from traversal-role bias rather than
    table-identity bias, a role-only structural prompt should hold up better than
    EXP-015/017 when screened under the strict subset protocol
- Change summary:
  - same strict subset-selection protocol as EXP-014 / EXP-015:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
  - prompt change:
    - keep `prompt_structural_encoding`
    - keep `prompt_role_alpha=0.1`
    - set `prompt_table_alpha=0.0`
- Command:
  - single-GPU `user-ltv` role-only structural-prompt screening run on GPU `6`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `mae=101.02596457714216`, `r2=-0.37706767946614983`,
      `rmse=192.63861595891302`
    - `step=1024`: `mae=105.16232851485141`, `r2=-0.3945880982565868`,
      `rmse=193.8602122759254`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `mae=109.6057071030233`, `r2=-0.4875726758891783`,
      `rmse=191.38417495052488`
    - `step=1024`: `mae=113.59627964282875`, `r2=-0.5102536918936471`,
      `rmse=192.83767250416503`
- Result:
  - completed with early stopping at `step=5120`
  - best periodic `TestSubset` checkpoint was selected at `step=3072`
  - best `TestSubset` metrics from the selected checkpoint:
    - `mae=94.62971329735593`
    - `r2=-0.25068378511296197`
    - `rmse=175.48527785455707`
  - final subset test from the selected checkpoint:
    - `mae=94.63119348464535`
    - `r2=-0.25070950518901447`
    - `rmse=175.48708225609394`
  - versus strict-protocol baseline `EXP-014`:
    - `mae` is better by `6.35684571018442`
    - `r2` is higher by `0.09853843245140072`
    - `rmse` is better by `6.78191714967451`
  - versus full structural prompt `EXP-015`:
    - `mae` is better by `1.78129460419528`
    - `r2` is higher by `0.03380338010292738`
    - `rmse` is better by `2.35566581740673`
- Decision:
  - drop as a global candidate
  - although this is the strongest strict-protocol `user-ltv` prompt result so far,
    the same candidate already failed on `user-churn`, so it does not satisfy the
    representative-set objective

### EXP-022

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if table-identity embeddings were the main source of overbias on the salt
    task, a role-only structural prompt may behave better than the earlier
    full-strength structural patch while remaining directly comparable to EXP-012
    and EXP-019
- Change summary:
  - same strict subset-selection protocol as EXP-012 / EXP-019:
    - `periodic_test_steps=128`
    - `model_selection_source=test_subset`
  - prompt change:
    - keep `prompt_structural_encoding`
    - keep `prompt_role_alpha=0.1`
    - set `prompt_table_alpha=0.0`
- Command:
  - single-GPU `item-incoterms` role-only structural-prompt screening run on GPU `7`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9139322916666668`
    - `step=1024`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9152529761904762`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6311197916666667`
    - `step=1024`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6316406250000001`
- Result:
  - completed with early stopping at `step=2560`
  - best periodic `TestSubset` checkpoint remained the first one at `step=512`
  - best `TestSubset` metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
  - final subset test from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
  - versus strict-protocol baseline `EXP-012`:
    - exactly tied on the selected subset metrics
  - versus sort-only `EXP-013`:
    - worse by `0.0001953125` MRR
- Decision:
  - drop as a salt-side follow-up
  - role-only structural encoding does not improve on the existing salt references

### EXP-023

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: confirmation prompt candidate
- Hypothesis:
  - superseded by objective clarification
- Change summary:
  - same as EXP-021
  - attempted confirmation change:
    - `test_steps=-1`
- Command:
  - single-GPU role-only structural-prompt `user-ltv` full-test confirmation run on GPU `6`
- Validation metrics:
  - not authoritative; run was canceled before completion
- Test-subset metrics:
  - not authoritative; run was canceled before completion
- Result:
  - launched, then canceled after objective clarification from the user
  - cancellation reason:
    - the project goal is to find one candidate that is non-regressive across the
      representative task set, not a task-conditional improvement on `user-ltv`
    - `EXP-020` had already shown that the same role-only prompt candidate is worse
      on `user-churn`
    - therefore this candidate failed the global objective before full-test
      confirmation on `user-ltv` was worth the GPU time
- Decision:
  - cancel and drop as a global candidate

### EXP-024

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - pure structural token ordering is milder than prompt-side structural embeddings
  - if any prompt-side adjustment has a chance to be globally non-regressive, this
    ordering-only variant is more plausible than the earlier embedding-based changes
- Change summary:
  - same strict subset-selection protocol as `EXP-010`
  - prompt change only:
    - enable `prompt_structural_sort`
    - do not enable `prompt_structural_encoding`
- Command:
  - single-GPU `user-churn` sort-only screening run on GPU `5`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `average_precision=0.7992415667973793`, `accuracy=0.6953125`,
      `f1=0.7771428571428571`, `roc_auc=0.680232558139535`
    - `step=1024`: `average_precision=0.7546849275884382`, `accuracy=0.6875`,
      `f1=0.7590361445783133`, `roc_auc=0.673034330011074`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `average_precision=0.7089342170517587`, `accuracy=0.65625`,
      `f1=0.7582417582417582`, `roc_auc=0.5900974025974026`
    - `step=1024`: `average_precision=0.6844899222835262`, `accuracy=0.6171875`,
      `f1=0.7065868263473054`, `roc_auc=0.5865800865800866`
- Result:
  - completed with early stopping at `step=5120`
  - best periodic `TestSubset` checkpoint was selected at `step=3584`
  - best `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.6924642320397933`
    - `accuracy=0.65625`
    - `f1=0.7821782178217822`
    - `roc_auc=0.6109307359307359`
  - final subset test from the selected checkpoint:
    - `average_precision=0.6943457725989044`
    - `accuracy=0.65625`
    - `f1=0.7821782178217822`
    - `roc_auc=0.612012987012987`
  - versus strict-protocol baseline `EXP-010`:
    - `average_precision` is worse by `0.0825758412621821`
    - `accuracy` is unchanged
    - `f1` is worse by `0.0021355076684139`
    - `roc_auc` is worse by `0.0232683982683982`
- Decision:
  - drop as a global candidate
  - full structural sorting still hurts the classification representative task

### EXP-025

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if structural ordering is a genuinely better serialization strategy rather than a
    salt-only artifact, it should remain at least non-regressive on the regression
    representative task
- Change summary:
  - same strict subset-selection protocol as `EXP-014`
  - prompt change only:
    - enable `prompt_structural_sort`
    - do not enable `prompt_structural_encoding`
- Command:
  - single-GPU `user-ltv` sort-only screening run on GPU `6`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `mae=108.61166877674872`, `r2=-0.4376364968202464`,
      `rmse=196.8295281946853`
    - `step=1024`: `mae=107.79372943498196`, `r2=-0.4309320166364281`,
      `rmse=196.370030323858`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `mae=117.30525377409533`, `r2=-0.5587738471858232`,
      `rmse=195.91084748982976`
    - `step=1024`: `mae=116.48351503878833`, `r2=-0.5508801796158791`,
      `rmse=195.41416931535034`
- Result:
  - completed with early stopping at `step=5120`
  - best periodic `TestSubset` checkpoint was selected at `step=3072`
  - best `TestSubset` metrics from the selected checkpoint:
    - `mae=97.97278903910892`
    - `r2=-0.30201945990753276`
    - `rmse=179.0505525625174`
  - final subset test from the selected checkpoint:
    - `mae=97.97196466356517`
    - `r2=-0.3020216828335658`
    - `rmse=179.0507054081456`
  - versus strict-protocol baseline `EXP-014`:
    - `mae` is better by `3.0160745312646`
    - `r2` is higher by `0.0472262548068494`
    - `rmse` is better by `3.21829399762285`
- Decision:
  - drop as a global candidate
  - although it helps the regression representative task, the same candidate already
    failed on `user-churn`

### EXP-026

- Date: 2026-05-01
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - salt-side results already suggest sort-only is the least harmful prompt-side
    structural change; this rerun is the salt leg of the global-candidate check
- Change summary:
  - same strict subset-selection protocol as `EXP-012`
  - prompt change only:
    - enable `prompt_structural_sort`
    - do not enable `prompt_structural_encoding`
- Command:
  - single-GPU `item-incoterms` sort-only screening run on GPU `7`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9155753968253968`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6246364312770563`
- Result:
  - completed with early stopping at `step=3584`
  - best periodic `TestSubset` checkpoint was selected at `step=3072`
  - best `TestSubset` metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6313151041666667`
  - final subset test from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6313151041666667`
  - versus strict-protocol baseline `EXP-012`:
    - `mrr` is higher by `0.0001953125`
  - versus `EXP-013`:
    - exactly tied
- Decision:
  - drop as a global candidate
  - salt-side gain remains below the screening delta and does not rescue the churn-side regression

### EXP-027

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - the previous full structural sort may be too aggressive because it rewrites both
    cross-depth and within-depth order
  - a depth-only stable sort may preserve local context for Amazon tasks while still
    reducing the most DFS-skewed ordering artifact on salt
- Change summary:
  - added `--prompt_structural_sort_mode {full,depth_only}`
  - keep `prompt_structural_sort`
  - set `prompt_structural_sort_mode=depth_only`
  - stable-sort graph prompt tokens by `depth` only, preserving original order within
    each depth bucket
  - same strict subset-selection protocol as `EXP-010`
- Command:
  - single-GPU `user-churn` depth-only sort screening run on GPU `5`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `average_precision=0.7496825426388667`, `accuracy=0.6953125`,
      `f1=0.7845303867403315`, `roc_auc=0.6658361018826136`
    - `step=1024`: `average_precision=0.732849631671097`, `accuracy=0.4453125`,
      `f1=0.3826086956521739`, `roc_auc=0.6392580287929125`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `average_precision=0.6781577194663408`, `accuracy=0.671875`,
      `f1=0.776595744680851`, `roc_auc=0.5836038961038961`
    - `step=1024`: `average_precision=0.6704748167787429`, `accuracy=0.390625`,
      `f1=0.25`, `roc_auc=0.5533008658008658`
- Result:
  - launched and reached the first two evaluation points
- Decision:
  - pending

### EXP-028

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if the main benefit comes from shallower global serialization rather than from
    table-level reordering, depth-only stable sorting should remain at least
    non-regressive on the regression representative task
- Change summary:
  - same code/path as `EXP-027`
  - same strict subset-selection protocol as `EXP-014`
- Command:
  - single-GPU `user-ltv` depth-only sort screening run on GPU `6`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `mae=94.19747272788548`, `r2=-0.26043761239169494`,
      `rmse=184.30044501586184`
    - `step=1024`: `mae=104.6937942442298`, `r2=-0.40487370195239336`,
      `rmse=194.5737943897573`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `mae=101.42350772424604`, `r2=-0.36562057879270915`,
      `rmse=183.37154928821846`
    - `step=1024`: `mae=113.39856939852238`, `r2=-0.5204729601733611`,
      `rmse=193.48899932741656`
- Result:
  - launched and reached the first two evaluation points
- Decision:
  - pending

### EXP-029

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt construction
- Hypothesis:
  - if the salt-side prompt issue is mostly about depth skew rather than table/name
    ordering, depth-only stable sorting should retain some of the salt improvement
    without importing the churn-side regression of full sorting
- Change summary:
  - same code/path as `EXP-027`
  - same strict subset-selection protocol as `EXP-012`
- Command:
  - single-GPU `item-incoterms` depth-only sort screening run on GPU `7`
- Validation metrics:
  - interim selected-checkpoint `Val` metrics:
    - `step=512`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9155753968253968`
    - `step=1024`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9152597402597402`
    - `step=1536`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9138693337912087`
- Test-subset metrics:
  - interim periodic `TestSubset` metrics:
    - `step=512`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6246364312770563`
    - `step=1024`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.625204613095238`
    - `step=1536`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.625204613095238`
- Result:
  - launched and reached the first three evaluation points
- Decision:
  - pending

### EXP-027 (continued)

- Date: 2026-05-02
- Additional `TestSubset` checkpoints observed after launch:
  - `step=1536`: `average_precision=0.6904485167331356`, `accuracy=0.59375`,
    `f1=0.6623376623376623`, `roc_auc=0.5876623376623377`
  - `step=2048`: `average_precision=0.6777878893442091`, `accuracy=0.6640625`,
    `f1=0.7624309392265194`, `roc_auc=0.589556277056277`
  - `step=2560`: `average_precision=0.7018799347495639`, `accuracy=0.640625`,
    `f1=0.7604166666666666`, `roc_auc=0.610930735930736`
  - `step=3072`: `average_precision=0.6960313997063579`, `accuracy=0.65625`,
    `f1=0.7708333333333334`, `roc_auc=0.6133658008658008`
  - `step=3584`: `average_precision=0.7090115911695941`, `accuracy=0.671875`,
    `f1=0.7878787878787878`, `roc_auc=0.6103896103896104`
- Decision:
  - manually stopped before completion
  - despite some recovery versus its own early checkpoints, the churn leg stayed
    materially below strict baseline `EXP-010` on the primary screening metrics
    (`average_precision` and `roc_auc`)
  - therefore the depth-only sort candidate failed the representative-task
    non-regression requirement and was dropped as a global candidate

### EXP-028 (continued)

- Date: 2026-05-02
- Additional `TestSubset` checkpoints observed after launch:
  - `step=1536`: `mae=109.34423093549674`, `r2=-0.4749642405934147`,
    `rmse=190.5713777069541`
  - `step=2048`: `mae=110.7379285119276`, `r2=-0.47992860537307136`,
    `rmse=190.8918163624961`
  - `step=2560`: `mae=98.40088823222557`, `r2=-0.32343819376905825`,
    `rmse=180.51727126189212`
  - `step=3072`: `mae=96.95573861462996`, `r2=-0.2767010173803943`,
    `rmse=177.30114090828195`
- Decision:
  - manually stopped together with `EXP-027/029`
  - the regression leg looked directionally positive, but under the clarified
    global-candidate rule it could not rescue a candidate that had already failed
    on `user-churn`

### EXP-029 (continued)

- Date: 2026-05-02
- Additional `TestSubset` checkpoints observed after launch:
  - `step=2048`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
    `micro_f1=0.5078125`, `mrr=0.6250828598484849`
  - `step=2560`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
    `micro_f1=0.5078125`, `mrr=0.6313151041666667`
  - `step=3072`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
    `micro_f1=0.5078125`, `mrr=0.6316406250000001`
- Decision:
  - manually stopped together with `EXP-027/028`
  - salt-side signal remained only mildly positive, while the same candidate was
    already globally disqualified by the churn-side regression

### EXP-030

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt projector output
- Hypothesis:
  - prompt-side structural changes may be too task-sensitive because they change
    token identity or order directly
  - a lighter structural change is to normalize prompt projector outputs with a
    shared LayerNorm, reducing cross-task scale drift without changing prompt
    serialization
- Change summary:
  - add `--prompt_projector_layernorm`
  - apply a LayerNorm immediately after projector output in:
    - pretraining masked-node prompt path
    - demo prompt path
    - main graph-prompt path
  - keep the strict subset-selection protocol from `EXP-010`
  - do not enable prompt sorting or structural embeddings
- Command:
  - single-GPU `user-churn` LayerNorm screening run on GPU `5`
- Launch status:
  - local `main.py` and `model.py` re-synced to remote before launch
  - launched in remote `tmux` window `stage3-exp030`
  - persistent log path: `/tmp/stage3-exp030.log`
  - initial log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Result:
  - completed with early stopping at `step=4096`
  - best periodic `TestSubset` checkpoint was selected during training
  - best `TestSubset` metrics from the selected checkpoint:
    - `average_precision=0.7434278016969904`
    - `accuracy=0.6484375`
    - `f1=0.7692307692307693`
    - `roc_auc=0.6287878787878787`
  - final subset test from the selected checkpoint:
    - `average_precision=0.7469008579114519`
    - `accuracy=0.6484375`
    - `f1=0.7692307692307693`
    - `roc_auc=0.6301406926406926`
  - versus strict-protocol baseline `EXP-010`:
    - `average_precision` is worse by `0.0300207559496346`
    - `accuracy` is worse by `0.0078125`
    - `f1` is worse by `0.0150829562594268`
    - `roc_auc` is worse by `0.0051406926406926`
- Decision:
  - drop as a global candidate
  - LayerNorm on projector output is directly harmful on the classification
    representative task

### EXP-031

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt projector output
- Hypothesis:
  - same global candidate as `EXP-030`, evaluated on the regression representative
    task under the strict subset-selection protocol from `EXP-014`
- Change summary:
  - same code/path as `EXP-030`
  - no prompt sorting or structural embeddings
- Command:
  - single-GPU `user-ltv` LayerNorm screening run on GPU `6`
- Launch status:
  - launched in remote `tmux` window `stage3-exp031`
  - persistent log path: `/tmp/stage3-exp031.log`
  - initial log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Result:
  - completed with early stopping at `step=6144`
  - best periodic `TestSubset` checkpoint was selected during training
  - best `TestSubset` metrics from the selected checkpoint:
    - `r2=-0.3521705769159218`
    - `mae=101.64759715467693`
    - `rmse=182.46630130867982`
  - final subset test from the selected checkpoint:
    - `r2=-0.35214502556470917`
    - `mae=101.64360212728383`
    - `rmse=182.46457730926207`
  - versus strict-protocol baseline `EXP-014`:
    - `r2` is lower by `0.00289708792429397`
    - `mae` is worse by `0.65556293245406`
    - `rmse` is worse by `0.19557790349362`
- Decision:
  - drop as a global candidate
  - the regression representative task also moved slightly in the wrong direction,
    so this candidate does not even have a compensating Amazon-side win

### EXP-032

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: prompt projector output
- Hypothesis:
  - same global candidate as `EXP-030`, evaluated on the salt representative task
    under the strict subset-selection protocol from `EXP-012`
- Change summary:
  - same code/path as `EXP-030`
  - no prompt sorting or structural embeddings
- Command:
  - single-GPU `item-incoterms` LayerNorm screening run on GPU `7`
- Launch status:
  - launched in remote `tmux` window `stage3-exp032`
  - persistent log path: `/tmp/stage3-exp032.log`
  - initial log shows repeated fast DB-load calls on `rel-salt`, consistent with the
    earlier salt startup pattern
- Early metrics:
  - `step=512`
    - `Val`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9155753968253968`
    - `TestSubset`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6246364312770563`
  - interpretation:
    - this exactly matches the familiar baseline-like salt startup point at
      `step=512`
    - so the LayerNorm candidate has not shown an early salt regression, but it has
      not shown an early salt gain either
- Result:
  - completed with early stopping at `step=4608`
  - best periodic `TestSubset` checkpoint was selected during training
  - best `TestSubset` metrics from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
  - final subset test from the selected checkpoint:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6311197916666667`
  - versus strict-protocol baseline `EXP-012`:
    - exactly tied on all tracked subset metrics
- Decision:
  - drop as a global candidate
  - salt is only neutral, while both Amazon representatives already regressed

### EXP-033

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - repeated prompt-side structural changes have failed to produce a global win
  - the more likely shared failure mode is now the Stage 2 basis injection itself
  - in the current code path, `--disable_basis_token_head` also bypasses basis
    artifact loading, so this run effectively measures a no-basis ablation against
    the strict subset baseline
- Change summary:
  - keep the strict subset-selection protocol from `EXP-010`
  - add `--disable_basis_token_head`
  - note: under current `main.py`, this disables basis artifact loading entirely
- Command:
  - single-GPU `user-churn` no-basis screening run on GPU `5`
- Launch status:
  - launched in remote `tmux` window `stage3-exp033`
  - persistent log path: `/tmp/stage3-exp033.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Early metrics:
  - `step=1536`
    - `Val`: `average_precision=0.7705274777400603`, `accuracy=0.3984375`,
      `f1=0.26666666666666666`, `roc_auc=0.6780177187153931`
    - `TestSubset`: `average_precision=0.6764262895024205`, `accuracy=0.34375`,
      `f1=0.125`, `roc_auc=0.5814393939393939`
  - versus strict-protocol baseline `EXP-010`:
    - `average_precision` is already worse by `0.1004953243586659`
    - `accuracy` is worse by `0.3125`
    - `f1` is worse by `0.6593137254901961`
    - `roc_auc` is worse by `0.0538419913419913`
- Decision:
  - manually stopped and dropped as a global candidate
  - the no-basis ablation collapses the classification representative task far too
    early to justify further compute

### EXP-034

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same no-basis global candidate as `EXP-033`, evaluated on the regression
    representative task under the strict subset-selection protocol from `EXP-014`
- Change summary:
  - same code/path as `EXP-033`
- Command:
  - single-GPU `user-ltv` no-basis screening run on GPU `6`
- Launch status:
  - launched in remote `tmux` window `stage3-exp034`
  - persistent log path: `/tmp/stage3-exp034.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Early metrics:
  - `step=512`
    - `Val`: `r2=-0.4407532469997626`, `mae=108.98492153821165`,
      `rmse=197.04277275740804`
    - `TestSubset`: `r2=-0.5624334566756817`, `mae=117.6805861119868`,
      `rmse=196.14068739968272`
  - `step=1024`
    - `Val`: `r2=-0.36890346082714887`, `mae=101.74033723063766`,
      `rmse=192.06671894210984`
    - `TestSubset`: `r2=-0.4815399182140443`, `mae=109.9580365779251`,
      `rmse=190.9957074407823`
  - versus strict-protocol baseline `EXP-014`:
    - `step=512` `TestSubset mae` is worse by `1.00603740389147`
    - `step=1024` `TestSubset mae` is worse by `2.55718115240794`
- Decision:
  - manually stopped and dropped as a global candidate
  - the regression representative task also moved in the wrong direction from the
    start, so there is no compensating signal here

### EXP-035

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same no-basis global candidate as `EXP-033`, evaluated on the salt
    representative task under the strict subset-selection protocol from `EXP-012`
- Change summary:
  - same code/path as `EXP-033`
- Command:
  - single-GPU `item-incoterms` no-basis screening run on GPU `7`
- Launch status:
  - launched in remote `tmux` window `stage3-exp035`
  - persistent log path: `/tmp/stage3-exp035.log`
  - startup log shows repeated fast DB-load calls on `rel-salt`, consistent with the
    earlier salt startup pattern
- Early metrics:
  - `step=512`
    - `Val`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.916189236111111`
    - `TestSubset`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6250828598484849`
  - `step=1536`
    - `Val`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.913671875`
    - `TestSubset`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6313151041666667`
  - versus strict-protocol baseline `EXP-012`:
    - `step=512` `mrr` is higher by `0.0004464285714286`
    - `step=1536` `mrr` is higher by `0.0001953125`
- Decision:
  - manually stopped and dropped as a global candidate
  - the salt-side gain is still only noise-scale, and the same candidate is already
    invalidated by both Amazon representatives

### EXP-036

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - the failure may be in the direct basis-driven prompt rewrite, not in basis
    supervision itself
  - set `basis_residual_alpha=0` and `basis_graph_alpha=0` so basis alignment still
    provides supervision through its losses, but prompt tokens are no longer mixed
    with basis residuals during inference
- Change summary:
  - keep the strict subset-selection protocol from `EXP-010`
  - keep basis artifacts and basis losses enabled
  - set:
    - `basis_residual_alpha=0.0`
    - `basis_graph_alpha=0.0`
- Command:
  - single-GPU `user-churn` basis-loss-only screening run on GPU `5`
- Launch status:
  - launched in remote `tmux` window `stage3-exp036`
  - persistent log path: `/tmp/stage3-exp036.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Early metrics:
  - `step=1024`
    - `Val`: `average_precision=0.7629371132575058`, `accuracy=0.7109375`,
      `f1=0.8042328042328042`, `roc_auc=0.6746954595791805`
    - `TestSubset`: `average_precision=0.6859759907422167`, `accuracy=0.640625`,
      `f1=0.7653061224489796`, `roc_auc=0.5922619047619048`
  - versus strict-protocol baseline `EXP-010`:
    - `average_precision` is worse by `0.0909456231188698`
    - `accuracy` is worse by `0.015625`
    - `f1` is worse by `0.0190076030412165`
    - `roc_auc` is worse by `0.0430194805194804`
- Decision:
  - manually stopped and dropped as a global candidate
  - removing all basis-driven prompt mixing while keeping only basis losses still
    hurts the classification representative task too much

### EXP-037

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same basis-loss-only global candidate as `EXP-036`, evaluated on the regression
    representative task under the strict subset-selection protocol from `EXP-014`
- Change summary:
  - same code/path as `EXP-036`
- Command:
  - single-GPU `user-ltv` basis-loss-only screening run on GPU `6`
- Launch status:
  - launched in remote `tmux` window `stage3-exp037`
  - persistent log path: `/tmp/stage3-exp037.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Early metrics:
  - `step=512`
    - `Val`: `r2=-0.43815411955659167`, `mae=108.70943902645261`,
      `rmse=196.8649593589748`
    - `TestSubset`: `r2=-0.559451218175943`, `mae=117.39576658111068`,
      `rmse=195.95340976123325`
  - versus strict-protocol baseline `EXP-014`:
    - `mae` is worse by `0.72121787301535`
    - `r2` is lower by `0.2102032805355278`
    - `rmse` is worse by `13.6844103554648`
- Decision:
  - manually stopped and dropped as a global candidate
  - the regression representative task is immediately worse than the strict baseline

### EXP-038

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same basis-loss-only global candidate as `EXP-036`, evaluated on the salt
    representative task under the strict subset-selection protocol from `EXP-012`
- Change summary:
  - same code/path as `EXP-036`
- Command:
  - single-GPU `item-incoterms` basis-loss-only screening run on GPU `7`
- Launch status:
  - launched in remote `tmux` window `stage3-exp038`
  - persistent log path: `/tmp/stage3-exp038.log`
  - startup log shows repeated fast DB-load calls on `rel-salt`, consistent with the
    earlier salt startup pattern
- Early metrics:
  - `step=512`
    - `Val`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9155753968253968`
    - `TestSubset`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6246364312770563`
  - `step=1024`
    - `Val`: `accuracy=0.890625`, `macro_f1=0.1884297520661157`,
      `micro_f1=0.890625`, `mrr=0.9154017857142858`
    - `TestSubset`: `accuracy=0.5078125`, `macro_f1=0.13471502590673573`,
      `micro_f1=0.5078125`, `mrr=0.6237623054029304`
  - versus strict-protocol baseline `EXP-012`:
    - `step=512` is exactly tied
    - `step=1024` `mrr` is worse by `0.0008741258741259`
- Decision:
  - manually stopped and dropped as a global candidate
  - salt does not compensate for the Amazon regressions, and even its own early
    signal drifts slightly negative by `step=1024`

### EXP-039

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - the harmful part may be only the graph-level basis residual `r_graph`
  - keep the original token residual path, but set `basis_graph_alpha=0.0` so the
    candidate becomes token-only basis injection
- Change summary:
  - keep the strict subset-selection protocol from `EXP-010`
  - keep `basis_residual_alpha` at the task baseline value
  - set `basis_graph_alpha=0.0`
- Command:
  - single-GPU `user-churn` token-only basis screening run on GPU `5`
- Launch status:
  - launched in remote `tmux` window `stage3-exp039`
  - persistent log path: `/tmp/stage3-exp039.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Decision:
  - running

### EXP-042 / EXP-043 / EXP-044 Final Bundle Verdict

- Date: 2026-05-05
- Bundle: paper-guided graph-alignment rebalance (`BUNDLE-042`)
- Final subset test metrics:
  - `EXP-042` / `user-churn`:
    - `average_precision=0.7476724172800624`
    - `accuracy=0.6171875`
    - `f1=0.7065868263473054`
    - `roc_auc=0.6168831168831168`
  - `EXP-043` / `user-ltv`:
    - `r2=-0.26793235440061625`
    - `mae=96.76873391260393`
    - `rmse=176.691220249056`
  - `EXP-044` / `item-incoterms`:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6313151041666667`
- Result:
  - `user-churn` regressed versus strict baseline `EXP-010`
  - `user-ltv` improved versus strict baseline `EXP-014`
  - `item-incoterms` was neutral versus strict baseline `EXP-012`
- Decision:
  - global verdict `failed`
  - orchestrator early-killed the bundle once `user-churn` was clearly worse on the
    primary metric
  - do not extend this direction as a promotable global candidate
- Interpretation:
  - reducing graph residual strength while strengthening graph-side supervision again
    helps regression but does not protect the classification representative task
  - this weakens the hypothesis that a simple graph-residual rebalance is enough to
    recover cross-task stability

### EXP-045 / EXP-046 / EXP-047

- Date: 2026-05-05
- Branch: `main` (temporary remote patch only, not committed)
- Target component: graph retrieval / pruning
- Hypothesis:
  - current Stage 3 failures may be caused by overloading the prompt channel with too
    much graph context rather than by basis coefficients alone
  - halving the sampled neighbor budget should produce a cleaner graph prompt and may
    recover `user-churn` without losing recent `user-ltv` gains
- Paper basis:
  - `G-Retriever` motivated subgraph selection over heavier prompt-side graph mixing
  - `GraphRAG Survey` motivated allocating a Stage 3 slot to graph-guided retrieval
  - `GraphPrompter` motivated cleaning the graph-text interface instead of another
    prompt-decoration change
- Change summary:
  - keep basis, prompt, optimizer, and evaluation settings unchanged
  - halve `num_neighbors` from the strict baseline for each representative task:
    - `user-churn`: `16 -> 8`
    - `user-ltv`: `128 -> 64`
    - `item-incoterms`: `64 -> 32`
- Launch status:
  - launched in remote `tmux` windows:
    - `stage3-exp045`
    - `stage3-exp046`
    - `stage3-exp047`
  - monitored locally via `stage3_orchestrator.py monitor --kill-failed`
- Final subset test metrics:
  - `EXP-045` / `user-churn`:
    - `average_precision=0.6714347364407169`
    - `accuracy=0.65625`
    - `f1=0.7924528301886793`
    - `roc_auc=0.5608766233766234`
  - `EXP-046` / `user-ltv`:
    - `r2=-0.5293588180812869`
    - `mae=114.52524603873492`
    - `rmse=194.053564147847`
  - `EXP-047` / `item-incoterms`:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6244969223484849`
- Decision:
  - global verdict `failed`
  - early-killed once `user-churn` dropped below baseline at the first checkpoint
  - final judge confirms all three representative tasks were worse
- Interpretation:
  - simple neighbor-budget pruning is too destructive in the current pipeline
  - graph retrieval quality likely needs a more selective mechanism than uniform
    budget reduction

### EXP-040

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same token-only basis global candidate as `EXP-039`, evaluated on the regression
    representative task under the strict subset-selection protocol from `EXP-014`
- Change summary:
  - same code/path as `EXP-039`
- Command:
  - single-GPU `user-ltv` token-only basis screening run on GPU `6`
- Launch status:
  - launched in remote `tmux` window `stage3-exp040`
  - persistent log path: `/tmp/stage3-exp040.log`
  - startup log confirmed remote DB load started:
    - `Loading Database object from /home/u2021201693/.cache/relbench/rel-amazon/db...`
- Decision:
  - running

### EXP-041

- Date: 2026-05-02
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection
- Hypothesis:
  - same token-only basis global candidate as `EXP-039`, evaluated on the salt
    representative task under the strict subset-selection protocol from `EXP-012`
- Change summary:
  - same code/path as `EXP-039`
- Command:
  - single-GPU `item-incoterms` token-only basis screening run on GPU `7`
- Launch status:
  - launched in remote `tmux` window `stage3-exp041`
  - persistent log path: `/tmp/stage3-exp041.log`
  - startup log shows repeated fast DB-load calls on `rel-salt`, consistent with the
    earlier salt startup pattern
- Decision:
  - running

### EXP-039 / EXP-040 / EXP-041 Final Bundle Verdict

- Date: 2026-05-05
- Bundle: token-only basis residual (`BUNDLE-039`)
- Final subset test metrics:
  - `EXP-039` / `user-churn`:
    - `average_precision=0.6984435100447597`
    - `accuracy=0.6640625`
    - `f1=0.7881773399014779`
    - `roc_auc=0.6114718614718615`
  - `EXP-040` / `user-ltv`:
    - `r2=-0.31381819829603286`
    - `mae=99.00623853391038`
    - `rmse=179.8599898607692`
  - `EXP-041` / `item-incoterms`:
    - `accuracy=0.5078125`
    - `macro_f1=0.13471502590673573`
    - `micro_f1=0.5078125`
    - `mrr=0.6312065972222223`
- Result:
  - `user-ltv` improved versus strict baseline `EXP-014`
  - `item-incoterms` was neutral versus strict baseline `EXP-012`
  - `user-churn` regressed below strict baseline `EXP-010`
- Decision:
  - global verdict `failed`
  - do not relaunch or extend this bundle
  - keep it only as evidence that graph residual removal helps regression but is not globally safe

### EXP-042 / EXP-043 / EXP-044

- Date: 2026-05-05
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 2 basis injection / graph-alignment rebalance
- Hypothesis:
  - recent ablations imply the basis channel is necessary, but direct graph residual mixing is
    probably too aggressive relative to graph-query alignment quality
  - a paper-backed compromise is to keep token basis residuals unchanged, weaken graph residual
    injection, and strengthen graph-side alignment supervision
- Paper basis:
  - `GraphPrompter` supports focusing on alignment quality rather than prompt decoration
  - `G-Retriever` supports reducing over-compressed graph prompt injection
  - `RGLM` supports stronger graph-side supervision for graph/token alignment
- Change summary:
  - keep `basis_residual_alpha` unchanged from each task baseline
  - set `basis_graph_alpha` to `0.5x` its strict-baseline value for each representative task
  - set `basis_lambda_g` to `1.5x` its strict-baseline value for each representative task
  - set `basis_tau` to `0.85x` its strict-baseline value for each representative task
  - no prompt-order, projector-LayerNorm, or no-basis changes
- Launch status:
  - candidate bundle spec filled in [exp042_paper_guided_basis_scan.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp042_paper_guided_basis_scan.json)
  - first generated launcher failed before training because `stage3_research.py` emitted CRLF shell
    scripts and kebab-case CLI flags
  - fixed local research pipeline:
    - wrapper and launcher files now render with LF newlines
    - wrapper flags now preserve underscore-style argument names expected by [main.py](/G:/RelLLM-2/Rel-LLM/main.py)
  - rerendered bundle and relaunched successfully in remote `tmux` windows:
    - `stage3-exp042`
    - `stage3-exp043`
    - `stage3-exp044`
  - persistent logs:
    - `/tmp/stage3-exp042.log`
    - `/tmp/stage3-exp043.log`
    - `/tmp/stage3-exp044.log`
  - latest orchestrator check after relaunch:
    - all three windows `up`
    - logs reached remote DB-load stage
    - GPUs still constrained to `5/6/7`
- Decision:
  - running
  - next gate is the first periodic `TestSubset` checkpoint; kill the whole bundle immediately if any
    representative task is already below its strict primary-metric baseline

### Stage 3 Launcher Upgrade

- Date: 2026-05-07
- Scope:
  - upgrade the local Stage 3 workflow so launch no longer depends on a manually chosen fixed
    GPU triple
- Program changes:
  - [stage3_research.py](/G:/RelLLM-2/Rel-LLM/stage3_research.py) now supports:
    - multi-target `remote_targets` config
    - idle-GPU probing with memory / utilization thresholds
    - local `launch` command that auto-assigns representative-task runs to currently idle GPUs
      across all configured targets
    - automatic code + wrapper sync before remote `tmux` launch
    - writing resolved `task_launches` back into the candidate JSON
  - [stage3_orchestrator.py](/G:/RelLLM-2/Rel-LLM/stage3_orchestrator.py) now tracks bundle state
    by per-task target placement instead of assuming one fixed remote host
  - [pipeline_config.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/pipeline_config.json) now carries:
    - `remote_targets`
    - `gpu_idle_max_memory_mb`
    - `gpu_idle_max_utilization`
    - `launch_sync_paths`
- Operating rule update:
  - do not artificially cap routine Stage 3 work to a single server or to a fixed `3`-GPU subset
    when more configured remote targets and idle GPUs are available
- Current practical limit:
  - this workstation currently has only one experiment server configured in local SSH /
    `pipeline_config.json`, so immediate benefit is "use all idle GPUs on that host"
  - once more servers are added to `remote_targets`, the same launcher path can consume them
    without further code changes

### EXP-087 / EXP-088 / EXP-089

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 3 control protocol / baseline noise floor
- Hypothesis:
  - rerunning the unchanged refreshed `512`-step strict baseline bundle will quantify the current
    representative-task noise floor before any additional near-boundary candidate is interpreted
- Evidence basis:
  - `EXP-078` and `EXP-081` produced informative but non-promotable near-boundary outcomes
  - [STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md)
    formalized baseline rerun control as the next required step
- Change summary:
  - no model change
  - no prompt change
  - no loss change
  - pure baseline control rerun under the current local Stage 3 launcher
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp087_baseline_noise_floor_control.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP087`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP088`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP089`: target `lab25211`, GPU `6`
  - first orchestrator status check:
    - all three windows `up`
    - all three logs reached DB-load startup
    - no evaluation checkpoint yet
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `failed`
- Final subset test metrics:
  - `EXP087` / `user-churn`:
    - `average_precision=0.7118759640215568`
    - `accuracy=0.6484375`
    - `f1=0.7398843930635838`
    - `roc_auc=0.6565736028384734`
  - `EXP088` / `user-ltv`:
    - `r2=-0.39784040866607495`
    - `mae=85.78381741727412`
    - `rmse=149.7730661748275`
  - `EXP089` / `item-incoterms`:
    - `accuracy=0.71484375`
    - `macro_f1=0.125535961520957`
    - `micro_f1=0.71484375`
    - `mrr=0.801751048430736`
- Result:
  - `user-churn` improved slightly versus the stored strict baseline
  - `user-ltv` regressed heavily on a pure baseline rerun
  - `item-incoterms` regressed by almost exactly the same MRR margin seen earlier in `EXP080`
- Interpretation:
  - baseline rerun spread is already large enough that single near-boundary outcomes cannot be
    trusted in isolation
  - the earlier `EXP078` salt-side failure is now partially confounded, because the unchanged
    baseline reproduced nearly the same `item-incoterms` drop
  - the earlier `EXP078` `user-ltv` gain remains interesting because the baseline control moved in
    the opposite direction
- Next action:
  - launch the partial-win replay control bundle `EXP090/091/092` as the next required control
    step

### EXP-090 / EXP-091 / EXP-092

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: Stage 3 control protocol / partial-win replay
- Hypothesis:
  - replaying the `EXP078` split local/global graph-token packaging bundle under the same protocol
    will show whether its apparent Amazon-side gains and salt-side regression are stable relative
    to the baseline rerun drift revealed by `EXP087`
- Evidence basis:
  - `EXP078` was the strongest recent partial win
  - `EXP087` showed that an unchanged baseline rerun can reproduce nearly the same
    `item-incoterms` regression seen in `EXP078`
- Change summary:
  - exact replay of the `EXP078` modeling change only:
    - `prompt_split_global_local=true`
  - no new loss, basis, sampling, or label-interface changes
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp090_partial_win_replay_control.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP090`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP091`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP092`: target `lab25211`, GPU `6`
  - first orchestrator status check:
    - all three windows `up`
    - `user-churn` and `user-ltv` have completed DB load
    - `item-incoterms` has already entered train-step logging
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `failed`
- Final subset test metrics:
  - `EXP090` / `user-churn`:
    - `average_precision=0.7043522288128657`
    - `accuracy=0.6484375`
    - `f1=0.7738693467336684`
    - `roc_auc=0.6523351582192111`
  - `EXP091` / `user-ltv`:
    - `r2=-0.18637433425522332`
    - `mae=80.3553999617463`
    - `rmse=137.97989074559283`
  - `EXP092` / `item-incoterms`:
    - `accuracy=0.70703125`
    - `macro_f1=0.13234504751981307`
    - `micro_f1=0.70703125`
    - `mrr=0.7972931040313853`
- Result:
  - the earlier `EXP078` `user-churn` gain did not survive replay
  - the earlier `EXP078` `user-ltv` gain also collapsed below the stored strict baseline, though it
    remained much better than the pure-baseline rerun `EXP088`
  - the salt-side `item-incoterms` regression became even worse than both `EXP080` and the baseline
    rerun `EXP089`
- Interpretation:
  - the split local/global packaging family does not survive replay as a promotable Stage 3 branch
  - `item-incoterms` remains the decisive global blocker, and the replay result now points away
    from additional packaging-nearby reruns
  - the remaining justified frontier is a broader candidate-aware multiclass decision interface,
    not more token-packaging scans
- Next action:
  - return to the architecture-review path centered on `EXP084/085/086`

### EXP-084 / EXP-085 / EXP-086

- Date: 2026-05-07
- Branch: `main` (temporary remote patch only, not committed)
- Target component: candidate-aware multiclass decision interface
- Hypothesis:
  - for autocomplete multiclass tasks, keeping both class-id text and raw-label semantic text
    per candidate and fusing those candidate views at scoring time will preserve salt-side ranking
    discrimination better than either verbalizer choice alone
- Evidence basis:
  - `EXP081` showed that class-id label-interface unification helps `item-incoterms` directionally
    but is too weak by itself
  - `EXP087` and `EXP090` ruled out treating packaging-nearby effects as the active frontier
- Change summary:
  - add `autocomplete_decision_interface`
  - first implementation is `candidate_dual_view_fusion`
  - for autocomplete multiclass tasks only:
    - build one candidate representation from class-id text
    - build one candidate representation from raw-label semantic text
    - use a prompt-conditioned fusion gate to combine the two candidate-view score matrices
  - do not mix in new loss, sampling, or packaging changes
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp084_candidate_aware_multiclass_decision_interface.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP084`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP085`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP086`: target `lab25211`, GPU `6`
  - first stable orchestrator status check:
    - all three windows `up`
    - `user-churn` and `user-ltv` completed DB load
    - `item-incoterms` remained in repeated salt DB-load startup when checked
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `failed`
- Final subset test metrics:
  - `EXP084` / `user-churn`:
    - `average_precision=0.7396361496298718`
    - `accuracy=0.66015625`
    - `f1=0.7535410764872521`
    - `roc_auc=0.6844606418674843`
  - `EXP085` / `user-ltv`:
    - `r2=-0.31436299532490475`
    - `mae=85.05232373203734`
    - `rmse=145.2320899561588`
  - `EXP086` / `item-incoterms`:
    - `accuracy=0.716796875`
    - `macro_f1=0.13086055768982596`
    - `micro_f1=0.716796875`
    - `mrr=0.80168347819715`
- Result:
  - `user-churn` improved strongly and clearly above the stored strict baseline
  - `user-ltv` regressed heavily again
  - `item-incoterms` stayed near the same regressive band seen in `EXP078`, `EXP089`, and
    `EXP092`, with no meaningful recovery
- Interpretation:
  - the first dual-view fusion implementation did not produce a real cross-task win
  - salt-side multiclass scoring still looks bottlenecked, but simple global fusion between static
    class-id and raw-label candidate views is too weak
  - the next justified architecture step is to move from static dual-view fusion to a more explicit
    candidate-conditioned pairwise scorer, rather than scanning more packaging or sampling variants
- Next action:
  - draft the next post-control architecture candidate around candidate-conditioned pairwise
    scoring

### EXP-093 / EXP-094 / EXP-095

- Date: 2026-05-07
- Branch: `main`
- Target component: candidate-conditioned pairwise multiclass scorer
- Hypothesis:
  - the salt-side task may need candidate-specific interaction terms between the prompt
    representation and each candidate view, instead of a single shared fusion gate over static
    candidate embeddings
- Evidence basis:
  - `EXP081` showed that class-id label-interface unification is directionally helpful but too weak
  - `EXP084` showed that static dual-view candidate fusion still fails to move `item-incoterms`
    above its regressive band
  - `EXP087` and `EXP090` reduced the value of packaging-nearby reruns
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp093_candidate_conditioned_pairwise_multiclass_scorer.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP093`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP094`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP095`: target `lab25211`, GPU `6`
  - first stable orchestrator status check:
    - all three windows `up`
    - `user-churn` and `user-ltv` completed DB load
    - `item-incoterms` remained in early salt startup when checked
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `failed`
- Final subset test metrics:
  - `EXP093` / `user-churn`:
    - `average_precision=0.6929896697865823`
    - `accuracy=0.64453125`
    - `f1=0.7736318407960199`
    - `roc_auc=0.6482251513156839`
  - `EXP094` / `user-ltv`:
    - `r2=-0.12999408699130965`
    - `mae=78.75786603566958`
    - `rmse=134.66136451832674`
  - `EXP095` / `item-incoterms`:
    - `accuracy=0.69921875`
    - `macro_f1=0.09144316730523627`
    - `micro_f1=0.69921875`
    - `mrr=0.8042052999084248`
- Result:
  - `user-ltv` finally moved above the stored strict baseline
  - `item-incoterms` improved versus the earlier candidate-aware runs, but still remained below
    the strict MRR baseline
  - `user-churn` regressed below the strict baseline
- Interpretation:
  - candidate-conditioned pairwise interaction carries a real positive signal for `user-ltv`
    and a smaller positive signal for salt, but it still is not enough as a standalone scorer
  - the remaining missing piece is likely the explicit autoregressive candidate-token signal that
    already exists elsewhere in the codebase
- Next action:
  - launch the pairwise + autoregressive hybrid scorer as `EXP096/097/098`

### EXP-096 / EXP-097 / EXP-098

- Date: 2026-05-07
- Branch: `main`
- Target component: pairwise autoregressive hybrid scorer
- Hypothesis:
  - combining the new candidate-conditioned pairwise interaction score with the existing
    autoregressive candidate-token sequence score will preserve more salt-side ranking signal than
    either path alone
- Evidence basis:
  - `EXP093` improved `user-ltv` and modestly improved salt relative to recent candidate-aware
    runs, but still left `item-incoterms` below baseline and flipped `user-churn` negative
  - the codebase already contains an existing autoregressive multiclass candidate scorer, so this
    hybrid step is a reuse of an available path rather than a new loss family
- Change summary:
  - `autocomplete_decision_interface=candidate_pairwise_ar_hybrid`
  - for autocomplete multiclass tasks only:
    - keep the pairwise interaction scorer from `EXP093`
    - add the existing autoregressive candidate-token score to the final candidate score
  - do not mix in new loss, sampling, or packaging changes
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp096_pairwise_autoregressive_hybrid_scorer.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP096`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP097`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP098`: target `lab25211`, GPU `6`
  - first stable orchestrator status check:
    - all three windows `up`
    - `user-churn` and `user-ltv` completed DB load
    - `item-incoterms` progressed past repeated DB-load output and reached normal startup logging
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `failed`
- Final subset test metrics:
  - `EXP096` / `user-churn`:
    - `average_precision=0.6797240595370794`
    - `accuracy=0.66015625`
    - `f1=0.7507163323782235`
    - `roc_auc=0.6404546695137028`
  - `EXP097` / `user-ltv`:
    - `r2=-0.117544729874274`
    - `mae=78.02303016898222`
    - `rmse=133.91751528840277`
  - `EXP098` / `item-incoterms`:
    - `accuracy=0.74609375`
    - `macro_f1=0.18464375065825897`
    - `micro_f1=0.74609375`
    - `mrr=0.8337611607142856`
- Result:
  - `user-ltv` improved clearly above the stored strict baseline
  - `item-incoterms` improved clearly above the stored strict baseline
  - `user-churn` regressed clearly below the stored strict baseline
- Interpretation:
  - this is the strongest post-control salt-side result so far
  - the hybrid scorer is the first architecture family to push both `user-ltv` and
    `item-incoterms` above their strict baselines at the same time
  - because the implemented scorer change only targets autocomplete multiclass tasks, the
    `user-churn` regression is high-value replay-control territory rather than an immediate reason
    to discard the family
- Next action:
  - launch a direct replay control of the hybrid scorer as `EXP099/100/101`

### EXP-099 / EXP-100 / EXP-101

- Date: 2026-05-07
- Branch: `main`
- Target component: hybrid scorer replay control
- Hypothesis:
  - replaying the pairwise + autoregressive hybrid scorer will show whether the `user-ltv` and
    `item-incoterms` gains survive rerun variance and whether the `user-churn` regression was a
    no-op drift rather than a stable global failure
- Evidence basis:
  - `EXP096` is the first post-control architecture result with simultaneous strict-baseline gains
    on `user-ltv` and `item-incoterms`
  - the implemented scorer change only affects autocomplete multiclass tasks, so the `user-churn`
    regression is especially important to replay before discarding the family
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp099_hybrid_scorer_replay_control.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP099`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP100`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP101`: target `lab25211`, GPU `6`
  - first stable orchestrator status check:
    - all three windows `up`
    - `user-churn` and `user-ltv` are in Amazon DB-load startup
    - `item-incoterms` is in early salt startup
- Decision:
  - completed and finalized on 2026-05-07
  - global verdict `promotable`
- Final subset test metrics:
  - `EXP099` / `user-churn`:
    - `average_precision=0.7057405160186948`
    - `accuracy=0.646484375`
    - `f1=0.7723270440251573`
    - `roc_auc=0.6586286062902371`
  - `EXP100` / `user-ltv`:
    - `r2=-0.1071982314749147`
    - `mae=77.25856611595606`
    - `rmse=133.29615345170015`
  - `EXP101` / `item-incoterms`:
    - `accuracy=0.73828125`
    - `macro_f1=0.1719483030313045`
    - `micro_f1=0.73828125`
    - `mrr=0.8193460131448412`
- Result:
  - `user-churn` replayed above the stored strict baseline and removed the earlier blocker
  - `user-ltv` replayed above the stored strict baseline
  - `item-incoterms` replayed above the stored strict baseline
- Interpretation:
  - the pairwise + autoregressive hybrid scorer is the first post-control family to survive replay
    as a non-regressive representative-task bundle
  - the earlier `EXP096` `user-churn` drop now looks more like rerun drift than a stable causal
    regression from the autocomplete-only scorer change
  - this is strong enough to stop lightweight search and move into repository-result confirmation
- Next action:
  - launch full-test confirmation as `EXP102/103/104` with `test_steps=-1` and no additional
    modeling changes

### EXP-102 / EXP-103 / EXP-104

- Date: 2026-05-07
- Branch: `main`
- Target component: hybrid scorer full-test confirmation
- Hypothesis:
  - if the replay-confirmed hybrid scorer is a real cross-task gain rather than a subset-only
    effect, it should remain non-regressive across the representative bundle when final test
    evaluation is expanded to the full split
- Evidence basis:
  - `EXP096` first moved `user-ltv` and `item-incoterms` above their strict baselines
  - `EXP099` replay then moved all three representative tasks above their stored strict subset
    baselines
  - Stage 3 workflow now requires full-test confirmation before any repo-facing claim or later
    commit / push decision
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_ar_hybrid`
  - change only:
    - `test_steps=-1`
  - do not mix any new loss, sampling, packaging, or interface changes into this confirmation
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp102_hybrid_scorer_full_test_confirmation.json`
  - stopped by user decision on `2026-05-08` before bundle completion
  - reason:
    - fixed-hyperparameter full-test confirmation exposed severe wall-clock imbalance
    - `user-churn` and `user-ltv` finished quickly while single-GPU `item-incoterms` stretched
      toward a ~38 hour pass
  - keep this only as a control point:
    - `user-churn` full-test finished below the historical full-test baseline
    - `user-ltv` full-test finished above the historical full-test baseline
    - `item-incoterms` was intentionally not allowed to finish under this inefficient allocation

### EXP-105 / EXP-106 / EXP-107

- Date: 2026-05-08
- Branch: `main`
- Target component: hybrid scorer per-task Optuna retune
- Hypothesis:
  - after the hybrid scorer family survived subset replay, the next fair comparison is to retune
    each representative task separately under the same `512-step test_subset` Stage 3 protocol
    rather than to reuse the old baseline hyperparameters unchanged
- Evidence basis:
  - `EXP099` was promotable on subset replay
  - `EXP102` showed that fixed baseline hyperparameters are not a reliable final verdict for this
    modified architecture
- Change summary:
  - switch from fixed-hyperparameter confirmation to per-task `tune_hyperparameters.py`
  - keep Stage 3 subset protocol inside Optuna:
    - `periodic_test_steps=512`
    - `test_steps=512`
    - `model_selection_source=test_subset`
  - keep architecture override:
    - `autocomplete_decision_interface=candidate_pairwise_ar_hybrid`
- Launch status:
  - synced updated `tune_hyperparameters.py`, `main.py`, `model.py`, and `utils.py` to remote
  - launched in remote tmux windows:
    - `EXP105` / `user-churn`: `stage3-retune105`, GPU `0`
    - `EXP106` / `user-ltv`: `stage3-retune106`, GPU `5`
    - `EXP107` / `item-incoterms`: `stage3-retune107`, GPUs `6,7`
  - first launch attempt failed because multiple studies shared one SQLite storage file during
    initial table creation
  - relaunched successfully with per-task SQLite storage files:
    - `stage3_optuna_hybrid_churn.db`
    - `stage3_optuna_hybrid_ltv.db`
    - `stage3_optuna_hybrid_incoterms.db`
  - stopped by user decision after the scalability review
- Result:
  - keep the hybrid scorer as a valuable Stage 3 attempt and evidence source
  - retire the current exhaustive autocomplete head from the final implementation path because it
    scales as `O(L*C)` rather than `O(L + C)`
  - drop the earlier task-by-task GPU-balancing plan from Stage 3 policy
- Next action:
  - redesign the next multiclass scorer family around the same candidate-aware signal with
    scalable `O(L + C)`-style behavior

### EXP-108 / EXP-109 / EXP-110

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable shared-prefix hybrid multiclass scorer
- Hypothesis:
  - keep the replay-confirmed candidate-aware hybrid signal from `EXP099`, but replace
    exhaustive per-candidate autoregressive rollout with a shared-prefix candidate lattice
    over class-id token sequences
  - pairwise candidate interaction should preserve the positive `user-ltv` / salt-side signal,
    while prefix sharing should move the implementation back toward the required scalable path
- Evidence basis:
  - `EXP093` showed that candidate-conditioned pairwise interaction is a real positive signal
  - `EXP096` and `EXP099` showed that adding candidate-token autoregressive signal produces the
    first replay-confirmed non-regressive representative-task bundle
  - `EXP105` and the architecture review retired the current exhaustive hybrid head because it
    is not acceptable as the long-term `O(L + C)`-style path
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_shared_prefix_hybrid`
  - keep the pairwise candidate scorer from the hybrid line
  - replace exhaustive per-candidate autoregressive scoring with shared-prefix candidate-lattice
    scoring that reuses one prompt cache across common class-id token prefixes
  - do not mix new loss, sampling, packaging, or pretrain changes into this bundle
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp108_scalable_shared_prefix_hybrid_scorer.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP108`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP109`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP110`: target `lab25211`, GPU `6`
  - first orchestrator status check with `--sync-logs`:
    - all three windows `up`
    - all three runs remained in startup / DB-load logging
- Decision:
  - completed and finalized on 2026-05-08
  - global verdict `failed`
- Final subset test metrics:
  - `EXP108` / `user-churn`:
    - `average_precision=0.7004797528906123`
    - `accuracy=0.65234375`
    - `f1=0.7775`
    - `roc_auc=0.6688072952622537`
  - `EXP109` / `user-ltv`:
    - `r2=-0.3481232598266888`
    - `mae=82.32764533515206`
    - `rmse=147.08545432637374`
  - `EXP110` / `item-incoterms`:
    - `accuracy=0.740234375`
    - `macro_f1=0.16464802841271287`
    - `micro_f1=0.740234375`
    - `mrr=0.815460689484127`
- Result:
  - `user-churn` improved clearly above the stored strict baseline
  - `user-ltv` regressed below the stored strict baseline
  - `item-incoterms` landed only in the neutral band and did not clear the strict MRR delta
- Interpretation:
  - replacing exhaustive class-id autoregressive rollout with shared-prefix class-id scoring did
    preserve most of the salt-side signal, but it did not recreate a promotable bundle
  - the next scalable family should keep shared-prefix scoring but restore a second semantic
    candidate-token view rather than relying on class-id token sequences alone
- Next action:
  - move to a scalable dual-prefix hybrid scorer that combines pairwise interaction with
    shared-prefix scoring over both class-id and raw-label token views

### EXP-111 / EXP-112 / EXP-113

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable dual-prefix hybrid multiclass scorer
- Hypothesis:
  - keep the scalable shared-prefix rewrite from `EXP108`, but score both class-id token prefixes
    and raw-label semantic token prefixes, then fuse those two candidate-token views on top of the
    pairwise scorer
  - this should preserve more of the replay-confirmed hybrid signal than class-id-only prefix
    scoring while remaining scalable and avoiding exhaustive `O(L*C)` rollout
- Evidence basis:
  - `EXP084` showed that class-id and raw-label views both matter but static dual-view fusion is
    too weak
  - `EXP093` and `EXP099` established that candidate-aware interaction plus candidate-token signal
    is the right architecture family
  - `EXP108` showed that class-id shared-prefix scoring alone preserves some salt-side signal but
    is still too weak as a promotable bundle
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_dual_shared_prefix_hybrid`
  - keep the pairwise candidate scorer
  - add a second shared-prefix scorer over raw-label semantic token sequences
  - fuse class-id and raw-label shared-prefix scores with a lightweight gate instead of returning
    to exhaustive per-candidate rollout
  - do not mix new loss, sampling, packaging, or pretrain changes into this bundle
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp111_scalable_dual_prefix_hybrid_scorer.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP111`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP112`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP113`: target `lab25211`, GPU `6`
  - first orchestrator status check with `--sync-logs`:
    - all three windows `up`
    - all three runs remained in startup / DB-load logging
- Decision:
  - completed and finalized on 2026-05-08
  - implementation-blocked
- Partial outcomes before failure:
  - `EXP111` / `user-churn`:
    - `average_precision=0.7185721401901151`
    - `accuracy=0.654296875`
    - `f1=0.7643142476697736`
    - `roc_auc=0.6598327098752548`
  - `EXP112` / `user-ltv`:
    - `r2=-0.06351214324271748`
    - `mae=75.49950079527102`
    - `rmse=130.6399940399006`
- Failure signature:
  - `EXP113` / `item-incoterms` crashed before the first evaluation
  - root cause:
    - `candidate_pairwise_dual_shared_prefix_hybrid` instantiated the fusion gate but not
      `candidate_pairwise_scorer`
    - runtime error:
      - `TypeError: 'NoneType' object is not callable`
- Interpretation:
  - do not use `EXP111` as scientific evidence for or against the dual-prefix family
  - this was an implementation failure caused by an initialization bug in the new scorer path
- Next action:
  - fix the dual-prefix scorer initialization and relaunch the same architecture family as a clean
    rerun

### EXP-114 / EXP-115 / EXP-116

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable dual-prefix hybrid multiclass scorer rerun
- Hypothesis:
  - with the dual-prefix initialization bug fixed, a clean rerun should finally test the intended
    architecture: pairwise candidate-aware interaction plus shared-prefix scoring over both
    class-id and raw-label token views, without returning to exhaustive `O(L*C)` rollout
- Evidence basis:
  - `EXP111` was implementation-blocked rather than scientifically failed
  - `EXP084`, `EXP093`, `EXP099`, and `EXP108` still jointly support this rerun as the most
    justified scalable frontier
- Change summary:
  - no architecture change relative to `EXP111`
  - fix only the scorer initialization bug so the dual-prefix path instantiates both:
    - `candidate_fusion_gate`
    - `candidate_pairwise_scorer`
  - keep `autocomplete_decision_interface=candidate_pairwise_dual_shared_prefix_hybrid`
- Decision:
  - completed and finalized on 2026-05-08
  - resource-blocked
- Partial outcomes before failure:
  - `EXP114` / `user-churn`:
    - `average_precision=0.7252808668155011`
    - `accuracy=0.671875`
    - `f1=0.7777777777777778`
    - `roc_auc=0.6575529404209548`
  - `EXP115` / `user-ltv`:
    - `r2=-0.3050123183747435`
    - `mae=84.53018488912377`
    - `rmse=144.71456081712864`
- Failure signature:
  - `EXP116` / `item-incoterms` ran out of memory before the first evaluation
  - root cause:
    - the dual-prefix rerun kept the same architecture but the batched raw-label shared-prefix
      recursion exhausted GPU memory on the salt-side task
    - runtime error:
      - `torch.OutOfMemoryError`
- Interpretation:
  - do not use `EXP114` as scientific evidence against the dual-prefix family
  - this is a memory-scaling failure in the current implementation path
- Next action:
  - keep the same dual-prefix architecture and rerun it with memory-safe per-sample shared-prefix
    scoring rather than switching away from the family prematurely

### EXP-117 / EXP-118 / EXP-119

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable dual-prefix hybrid memory-safe rerun
- Hypothesis:
  - computing the dual-prefix shared-prefix scorer per sample should preserve the intended
    architecture while reducing peak KV-cache memory enough for item-incoterms to complete
- Evidence basis:
  - `EXP111` was implementation-blocked
  - `EXP114` showed the corrected dual-prefix family is still resource-blocked on the batched
    raw-label shared-prefix recursion
- Change summary:
  - no scientific architecture change relative to `EXP114`
  - convert the dual-prefix shared-prefix scorer to a memory-safe per-sample execution path
  - keep `autocomplete_decision_interface=candidate_pairwise_dual_shared_prefix_hybrid`
- Decision:
  - completed and finalized on 2026-05-08
  - resource-blocked
- Partial outcomes before failure:
  - `EXP117` / `user-churn`:
    - `average_precision=0.7023842662272168`
    - `accuracy=0.650390625`
    - `f1=0.7754077791718946`
    - `roc_auc=0.6645849053574582`
  - `EXP118` / `user-ltv`:
    - `r2=-0.1199883019418817`
    - `mae=78.96212843418004`
    - `rmse=134.0638442924545`
- Failure signature:
  - `EXP119` / `item-incoterms` still ran out of memory before the first evaluation
  - root cause:
    - even the per-sample dual-prefix shared-prefix path remained too large for single-GPU salt-side
      screening
    - runtime error:
      - `torch.OutOfMemoryError`
- Interpretation:
  - do not treat `EXP117` as scientific evidence against the dual-prefix family
  - the remaining blocker is no longer an obvious implementation bug; it is the single-GPU
    resource ceiling on the intended architecture
- Next action:
  - use the newly enabled Stage 3 multi-GPU screening fallback and rerun the same architecture with
    item-incoterms escalated to multiple idle GPUs

### EXP-120 / EXP-121 / EXP-122

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable dual-prefix hybrid multi-GPU rerun
- Hypothesis:
  - keeping the same dual-prefix architecture but escalating only the OOM-blocked item-incoterms
    run to multiple currently idle GPUs should make the bundle runnable without changing its
    scientific content
- Evidence basis:
  - `EXP111` was implementation-blocked
  - `EXP114` and `EXP117` were resource-blocked on item-incoterms under single-GPU screening
  - Stage 3 policy now explicitly allows multi-GPU fallback for justified OOM-blocked screening
    reruns
- Change summary:
  - no scientific architecture change relative to `EXP117`
  - keep `autocomplete_decision_interface=candidate_pairwise_dual_shared_prefix_hybrid`
  - escalate only `item-incoterms` to a multi-GPU launch
- Decision:
  - completed and finalized on 2026-05-08
  - resource-blocked
- Outcomes before failure:
  - `EXP120` / `user-churn`:
    - `average_precision=0.686994085110161`
    - `accuracy=0.6640625`
    - `f1=0.75`
    - `roc_auc=0.6418514296723232`
  - `EXP121` / `user-ltv`:
    - `r2=-0.110016824855802`
    - `mae=78.72752015134785`
    - `rmse=133.46571154859893`
- Failure signature:
  - `EXP122` / `item-incoterms` still ran out of memory before the first evaluation even under a
    2-GPU DDP launch
  - root cause:
    - the full-candidate dual-prefix raw-label autoregressive recursion remains too memory-heavy
      even after single-GPU fallback was escalated to multi-GPU
    - runtime error:
      - `torch.OutOfMemoryError`
- Interpretation:
  - do not treat `EXP120` as scientific evidence against hybrid candidate-aware scoring
  - do treat it as strong evidence that the full-candidate raw-label token recursion path is not a
    viable scalable implementation direction
- Next action:
  - keep the hybrid shortlist-and-rerank signal
  - stop spending more reruns on full-candidate dual-prefix token recursion
  - move to a fixed top-k candidate-aware reranker that preserves token-level signal only on the
    pairwise shortlist

### EXP-123 / EXP-124 / EXP-125

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable top-k dual hybrid reranker
- Hypothesis:
  - use the pairwise candidate-aware scorer as an `O(C)` shortlist stage, then apply a fixed top-k
    dual-view autoregressive reranker only to the shortlisted candidates so the bundle can keep
    the useful token-level signal from `EXP099` without returning to exhaustive `O(L*C)` scoring
- Evidence basis:
  - `EXP093` established that candidate-conditioned pairwise interaction carries real signal
  - `EXP099` established that token-level candidate scoring adds useful information beyond static
    pairwise scoring
  - `EXP108` showed that a scalable class-id-only shared-prefix rewrite leaves too much signal
    behind
  - `EXP120` showed that full-candidate raw-label token recursion is still not a practically
    scalable path even under multi-GPU fallback
- Change summary:
  - add `candidate_pairwise_topk_dual_hybrid`
  - keep pairwise candidate-aware scoring over all candidates
  - shortlist the top-k candidates from the pairwise scores
  - run dual-view autoregressive reranking only on that shortlist
  - set `autocomplete_candidate_topk=4` for the first screening bundle
- Decision:
  - completed and finalized on 2026-05-08
  - resource-blocked
- Outcomes before failure:
  - `EXP123` / `user-churn`:
    - `average_precision=0.6845097661925856`
    - `accuracy=0.662109375`
    - `f1=0.7481804949053857`
    - `roc_auc=0.6383675566330052`
  - `EXP124` / `user-ltv`:
    - `r2=-0.16289887070683284`
    - `mae=79.17673224401194`
    - `rmse=136.60792659804616`
- Failure signature:
  - `EXP125` / `item-incoterms` still ran out of memory before the first evaluation
  - root cause:
    - even after moving to a fixed top-k shortlist, the training-time autoregressive rerank branch
      still retained too much activation state through the frozen LLM
    - runtime error:
      - `torch.OutOfMemoryError`
- Interpretation:
  - do not reject the shortlist architecture itself from `EXP123`
  - do treat this as strong evidence that the remaining blocker sits in the training-time gradient
    path of the autoregressive reranker
- Next action:
  - keep the same top-k shortlist architecture
  - detach the autoregressive rerank branch during training and rerun as a clean bundle

### EXP-126 / EXP-127 / EXP-128

- Date: 2026-05-08
- Branch: `main`
- Target component: scalable top-k dual hybrid detached-train rerun
- Hypothesis:
  - keep the same shortlist-based scorer, but detach the autoregressive rerank branch during
    training so the model preserves candidate-token signal at scoring time without backpropagating
    the full rerank activation graph through the frozen LLM
- Evidence basis:
  - `EXP093` established the pairwise shortlist path is trainable and useful
  - `EXP099` established candidate-token reranking adds real signal
  - `EXP123` localized the remaining memory failure to the training-time autoregressive rerank
    branch rather than to the shortlist architecture itself
- Change summary:
  - no scientific interface change relative to `EXP123`
  - keep `candidate_pairwise_topk_dual_hybrid`
  - detach the top-k autoregressive rerank branch during training
  - preserve the same inference-time top-k dual reranker
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp126_scalable_topk_dual_hybrid_detached_train_rerun.json`
  - resolved idle-GPU placement at launch time:
    - `user-churn` / `EXP126`: target `lab25211`, GPU `0`
    - `user-ltv` / `EXP127`: target `lab25211`, GPU `5`
    - `item-incoterms` / `EXP128`: target `lab25211`, GPU `6`
  - first stable orchestrator status check with `--sync-logs`:
    - all three windows `up`
    - all three logs have advanced into DB-load startup
- Final subset test metrics:
  - `EXP126` / `user-churn`:
    - `average_precision=0.7117074117720339`
    - `accuracy=0.6796875`
    - `f1=0.7734806629834254`
    - `roc_auc=0.6751970716200812`
  - `EXP127` / `user-ltv`:
    - `r2=-0.18033543835773203`
    - `mae=80.86395973237347`
    - `rmse=137.6282693268028`
  - `EXP128` / `item-incoterms`:
    - `accuracy=0.69921875`
    - `macro_f1=0.09144316730523627`
    - `micro_f1=0.69921875`
    - `mrr=0.8037867731227106`
- Result:
  - `user-churn` improved clearly above the stored strict baseline
  - `user-ltv` regressed clearly below the stored strict baseline
  - `item-incoterms` remained clearly below the stored strict baseline even after the detached
    training fix removed the earlier OOM path
- Interpretation:
  - the detached training rerun solved the memory failure but not the practical throughput problem
    or the multiclass accuracy problem
  - this bundle is the clearest evidence so far that the current autoregressive rerank family is
    too expensive to justify continued routine screening
  - the remaining high-value next step is to keep candidate-conditioned interaction while dropping
    the autoregressive rerank path entirely
- Early-kill note:
  - `EXP128` was killed on 2026-05-09 after the `3072-step` evaluation
  - rationale:
    - it had become the only remaining active task
    - its throughput was structurally abnormal relative to the rest of the bundle
    - its latest completed evaluation still left `mrr` below the strict baseline
    - waiting for natural early stop had low expected information value relative to the remaining
      wall-clock cost
- Next action:
  - return to the cheaper pairwise-only candidate-conditioned scorer and replay it under the new
    scalability and throughput constraints

### EXP-129 / EXP-130 / EXP-131

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise interaction scalable replay
- Hypothesis:
  - if the robust part of the autocomplete gain comes from candidate-conditioned pairwise
    interaction rather than from expensive autoregressive candidate-token reranking, then replaying
    the pairwise-only scorer should preserve a meaningful fraction of the salt-side gain while
    restoring practical throughput and a cleaner Stage 4-compatible candidate-scoring story
- Evidence basis:
  - `EXP093` first established that candidate-conditioned pairwise interaction carries real signal
  - `EXP099` showed that stronger candidate-token reranking can help, but only through an
    autoregressive path that later proved too expensive
  - `EXP123` and `EXP126` showed that even the scalable shortlist reranker family is still too
    costly relative to the observed multiclass payoff
- Change summary:
  - keep only `autocomplete_decision_interface=candidate_pairwise_interaction`
  - remove the autoregressive rerank path entirely
  - launch with `serial_batch_equivalent` scheduling so the active task can absorb the idle GPU
    pool without silently changing effective global batch
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp129_candidate_pairwise_interaction_scalable_replay.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP129`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP130`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP131`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
  - first orchestrator status check with `--sync-logs`:
    - the serial controller is active
    - the first representative-task window is up
    - later representative tasks are still queued behind the serial controller, as expected for
      `launch_mode=serial_batch_equivalent`
- Final subset test metrics:
  - `EXP129` / `user-churn`:
    - `average_precision=0.6959108457603717`
    - `accuracy=0.6376953125`
    - `f1=0.7557603686635944`
    - `roc_auc=0.6853104496625675`
  - `EXP130` / `user-ltv`:
    - `r2=-0.04421898262452095`
    - `mae=75.82281380551868`
    - `rmse=144.26257370726995`
  - `EXP131` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6933005877585956`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is only `neutral` because its apparent gain stays within the baseline replay spread
  - `item-incoterms` is clearly `worse`, far outside the multiclass replay spread
- Interpretation:
  - the serial batch-equivalent schedule worked as intended and restored practical throughput
  - the pure pairwise candidate-conditioned scorer is too weak on salt-side ranking once the
    expensive autoregressive path is removed
  - the next justified scalable family should keep pairwise candidate interaction but add a cheap
    token-order-aware label representation rather than returning to autoregressive reranking

### EXP-132 / EXP-133 / EXP-134

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise positional interaction replay
- Hypothesis:
  - the pairwise-only scorer likely loses too much signal because its label representations are
    just mean-pooled token embeddings; adding a cheap position-aware token pooling module on the
    label side may recover part of the token-order signal seen in the stronger hybrid line without
    reintroducing autoregressive candidate rollout
- Evidence basis:
  - `EXP099` showed that candidate-token signal matters
  - `EXP129` showed that pure pairwise interaction is fast enough but too weak on
    `item-incoterms`
  - `EXP126` showed that the detached autoregressive reranker remains too expensive to keep as the
    active scalable path
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_positional_interaction`
  - keep the same pairwise candidate-conditioned scorer
  - replace mean-pooled label-token representations with a cheap position-aware attention pooling
    module over label tokens
- Decision:
  - completed and finalized on 2026-05-09
  - implementation-blocked
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp132_candidate_pairwise_positional_interaction_replay.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP132`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP133`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP134`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Outcomes before failure:
  - `EXP132` / `user-churn`:
    - `average_precision=0.6927990993424956`
    - `accuracy=0.6513671875`
    - `f1=0.7532826537664132`
    - `roc_auc=0.6845640731058252`
  - `EXP133` / `user-ltv`:
    - `r2=-0.12158674226713195`
    - `mae=79.41398763551376`
    - `rmse=149.51140321822558`
- Failure signature:
  - `EXP134` / `item-incoterms` failed before the first evaluation
  - root cause:
    - the new positional label pooler referenced `Tanh()` without importing it
    - runtime error:
      - `NameError: name 'Tanh' is not defined`
- Interpretation:
  - do not use `EXP132` as scientific evidence against the positional-interaction family
  - the correct next step is a clean rerun with the missing import fixed

### EXP-135 / EXP-136 / EXP-137

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise positional interaction rerun
- Hypothesis:
  - no scientific change relative to `EXP132`; after fixing the missing `Tanh` import, the
    positional-interaction family can be evaluated cleanly on `item-incoterms`
- Evidence basis:
  - `EXP129` showed that the pairwise-only scorer is fast enough but too weak on salt-side ranking
  - `EXP126` showed that the autoregressive reranker line remains too expensive
  - `EXP132` was implementation-blocked rather than scientifically resolved
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_positional_interaction`
  - fix only the missing `Tanh` import
- Decision:
  - completed and finalized on 2026-05-09
  - implementation-blocked
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp135_candidate_pairwise_positional_interaction_rerun.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP135`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP136`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP137`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Outcomes before failure:
  - `EXP135` / `user-churn`:
    - `average_precision=0.7360844554081769`
    - `accuracy=0.6513671875`
    - `f1=0.7353595255744997`
    - `roc_auc=0.6853280344505536`
  - `EXP136` / `user-ltv`:
    - `r2=-0.09231944512271895`
    - `mae=78.22817166646419`
    - `rmse=147.5477925456318`
- Failure signature:
  - `EXP137` / `item-incoterms` failed before the first evaluation
  - root cause:
    - after the missing `Tanh` import was fixed, `candidate_pairwise_positional_interaction` was
      still incorrectly routed through the `candidate_pairwise_topk_dual_hybrid` forward branch
    - runtime error:
      - `TypeError: 'NoneType' object is not callable`
      - thrown at `candidate_fusion_gate(prompt_repr)` inside the top-k dual hybrid scorer
- Interpretation:
  - do not use `EXP135` as scientific evidence against the positional-interaction family
  - the family still needs one clean rerun after the forward-routing fix

### EXP-138 / EXP-139 / EXP-140

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise positional interaction forward-fix rerun
- Hypothesis:
  - no scientific change relative to `EXP132` or `EXP135`; after fixing the forward-routing bug,
    the cheap position-aware label pooler can now be judged cleanly on `item-incoterms`
- Evidence basis:
  - `EXP129` showed that pure pairwise interaction is fast enough but too weak on salt-side
    ranking
  - `EXP132` and `EXP135` were implementation-blocked rather than scientifically resolved
  - the remaining open question was whether static position-aware label pooling could recover
    enough candidate-token signal without returning to autoregressive scoring
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_positional_interaction`
  - fix only the forward routing so this family reaches the intended pairwise scorer path on
    `item-incoterms`
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp138_candidate_pairwise_positional_interaction_forwardfix_rerun.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP138`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP139`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP140`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Final subset test metrics:
  - `EXP138` / `user-churn`:
    - `average_precision=0.693653147581452`
    - `accuracy=0.6494140625`
    - `f1=0.7608261159227182`
    - `roc_auc=0.6857617925542099`
  - `EXP139` / `user-ltv`:
    - `r2=-0.12173902423158967`
    - `mae=76.88132114700508`
    - `rmse=149.5215527307712`
  - `EXP140` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6785764663938492`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is still only `neutral`
  - `item-incoterms` is clearly `worse`, far outside the multiclass replay spread
- Interpretation:
  - the implementation is now cleanly resolved, so this is real scientific evidence
  - static position-aware label pooling is not enough to recover the missing salt-side
    candidate-token signal
  - the next justified scalable family should make label-token aggregation sample-conditioned,
    rather than continuing to tune static label encoders

### EXP-141 / EXP-142 / EXP-143

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned interaction
- Hypothesis:
  - if the missing scalable signal is prompt-conditioned token selection rather than token order
    alone, then replacing static label pooling with sample-conditioned token attention inside the
    pairwise scorer should recover part of the salt-side candidate-token benefit without
    autoregressive rollout
- Evidence basis:
  - `EXP099` remained the strongest evidence that candidate-token signal matters
  - `EXP129` showed that pure pairwise interaction restores throughput but loses too much salt-side
    ranking signal
  - `EXP138` showed that static position-aware label pooling is still too weak
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_query_conditioned_interaction`
  - keep the fast pairwise scorer
  - replace static candidate-view pooling with sample-conditioned token attention over class-id and
    raw-label candidate text
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp141_candidate_pairwise_query_conditioned_interaction.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP141`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP142`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP143`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Final subset test metrics:
  - `EXP141` / `user-churn`:
    - `average_precision=0.7041466419405796`
    - `accuracy=0.6494140625`
    - `f1=0.7572684246112238`
    - `roc_auc=0.6912892775778322`
  - `EXP142` / `user-ltv`:
    - `r2=-0.12130674121794516`
    - `mae=79.37475268162555`
    - `rmse=149.49273949776486`
  - `EXP143` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6785764663938492`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is `neutral`
  - `item-incoterms` is clearly `worse`
- Interpretation:
  - the query-conditioned implementation is cleanly resolved, so this is real scientific evidence
  - moving from static pooling to sample-conditioned single-vector pooling still does not recover
    the missing salt-side candidate-token discrimination
  - the next justified scalable family should keep token-level similarity statistics in the final
    scorer, rather than collapsing each candidate view to one vector too early

### EXP-144 / EXP-145 / EXP-146

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned token stats
- Hypothesis:
  - if the missing scalable signal is carried by token-level similarity peaks and spread rather
    than by a single pooled label vector, then augmenting the fast pairwise scorer with
    query-conditioned token-statistics residuals should recover part of the salt-side
    candidate-token benefit without autoregressive rollout
- Evidence basis:
  - `EXP099` remained the strongest evidence that candidate-token discrimination matters
  - `EXP138` and `EXP141` showed that both static and sample-conditioned single-vector pooling are
    too weak
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_query_conditioned_token_stats`
  - keep the fast pairwise scorer and query-conditioned token attention
  - preserve mean and max token-similarity statistics as an explicit residual in the candidate
    score
- Decision:
  - completed and finalized on 2026-05-09
  - implementation-blocked / abnormal-run
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp144_candidate_pairwise_query_conditioned_token_stats.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP144`: target `lab25211`, GPUs `0,5,6,7`, per-rank batch `1`
    - `user-ltv` / `EXP145`: target `lab25211`, GPUs `0,5`, per-rank batch `1`
    - `item-incoterms` / `EXP146`: target `lab25211`, GPUs `0,5,6,7`, per-rank batch `1`
- Outcomes before failure:
  - `EXP145` / `user-ltv`:
    - `r2=0.012684041259022671`
    - `mae=74.3539266570937`
    - `rmse=140.27683625804093`
- Failure signature:
  - `EXP144` / `user-churn` terminated before the first evaluation with external `SIGTERM`, so the
    first representative task never produced usable scientific evidence
  - `EXP146` / `item-incoterms` failed before the first evaluation
  - root cause:
    - the new token-stats path did not initialize its query/key projection modules
    - runtime error:
      - `ValueError: Query-conditioned token stats require query/key projections.`
- Interpretation:
  - do not use `EXP144` as scientific evidence against the token-stats family
  - the correct next step is a clean rerun after fixing the projection initialization bug

### EXP-147 / EXP-148 / EXP-149

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned token stats rerun
- Hypothesis:
  - no scientific change relative to `EXP144`; after fixing the projection initialization bug,
    the token-stats scorer can now be evaluated cleanly across the three representative tasks
- Evidence basis:
  - `EXP099` remained the strongest evidence that candidate-token discrimination matters
  - `EXP138` and `EXP141` showed that pooled-label families are too weak
  - `EXP144` was implementation-blocked / abnormal rather than scientifically resolved
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_query_conditioned_token_stats`
  - fix only the missing query/key projection initialization
- Decision:
  - completed and finalized on 2026-05-09
  - abnormal-run / implementation-blocked as a bundle
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp147_candidate_pairwise_query_conditioned_token_stats_rerun.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP147`: target `lab25211`, GPUs `0,5,6,7`, per-rank batch `1`
    - `user-ltv` / `EXP148`: target `lab25211`, GPUs `0,5`, per-rank batch `1`
    - `item-incoterms` / `EXP149`: target `lab25211`, GPUs `0,5,6,7`, per-rank batch `1`
- Outcomes before bundle invalidation:
  - `EXP148` / `user-ltv`:
    - `r2=-0.035474764288790483`
    - `mae=75.82734610630433`
    - `rmse=143.65728141191846`
  - `EXP149` / `item-incoterms`:
    - `accuracy=0.5771484375`
    - `macro_f1=0.0731888544891641`
    - `micro_f1=0.5771484375`
    - `mrr=0.7090690491691468`
- Failure signature:
  - `EXP147` / `user-churn` was externally terminated by `SIGTERM` before the first evaluation
  - the same external-termination pattern had already appeared on `EXP144` under the 4-GPU serial
    placement that included `GPU 0`
- Interpretation:
  - this rerun is not a clean scientific bundle because the first representative task never
    produced a valid result
  - the next correct action is a clean rerun with an explicit safer serial GPU placement that
    avoids the repeated `GPU 0` abnormal-termination pattern

### EXP-150 / EXP-151 / EXP-152

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned token stats safe-GPU rerun
- Hypothesis:
  - if the token-stats scorer itself is viable and the recent invalid bundles were mainly polluted
    by the unstable GPU0-inclusive serial placement, then rerunning the same scorer on the safer
    GPU 5,6 pair should yield a clean scientific verdict
- Evidence basis:
  - `EXP144` and `EXP147` were both invalid as bundle-level scientific evidence
  - `EXP147` nevertheless showed that the implementation bug was fixed and produced usable
    user-ltv / item-incoterms results
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_query_conditioned_token_stats`
  - change only the serial GPU placement to explicit `GPU 5,6` for all three representative tasks
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp150_candidate_pairwise_query_conditioned_token_stats_safegpu_rerun.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP150`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP151`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP152`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Final subset test metrics:
  - `EXP150` / `user-churn`:
    - `average_precision=0.6936289516988936`
    - `accuracy=0.6455078125`
    - `f1=0.7555555555555555`
    - `roc_auc=0.6825633150060766`
  - `EXP151` / `user-ltv`:
    - `r2=-0.059709514497534366`
    - `mae=76.46346467481314`
    - `rmse=145.32867058315142`
  - `EXP152` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6785764663938492`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is `neutral`
  - `item-incoterms` is clearly `worse`
- Interpretation:
  - the safe GPU placement removed the earlier abnormal-run pollution, so this is clean
    scientific evidence against the fixed-weight token-stats variant
  - preserving token-level mean/max statistics helps less than needed when they are only injected
    through a hand-designed residual
  - the next justified scalable family should keep the token-level statistics but let the model
    learn how to weight them, rather than hard-coding the residual combination

### EXP-153 / EXP-154 / EXP-155

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned token stats MLP
- Hypothesis:
  - if fixed token-statistics residuals are too rigid, then a learned MLP over the same
    query-conditioned token statistics may recover more salt-side candidate discrimination while
    keeping the scorer non-autoregressive and Stage 4-compatible
- Evidence basis:
  - `EXP150/151/152` gave a clean scientific failure for the fixed-weight token-stats residual
  - the observed pattern was `user-churn` better, `user-ltv` neutral, and `item-incoterms` clearly
    worse, so the justified next step was to learn the token-stat weights rather than hand-code
    them
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_query_conditioned_token_stats_mlp`
  - keep the same query-conditioned token statistics as the fixed residual family
  - replace the fixed residual combination with a small learned token-stats scorer
- Decision:
  - completed on 2026-05-09 as `implementation-blocked`
  - not usable as scientific evidence against the learned token-stats MLP family
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp153_candidate_pairwise_query_conditioned_token_stats_mlp.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP153`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP154`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP155`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
- Completed task metrics:
  - `EXP153` / `user-churn`:
    - `average_precision=0.7014241143974325`
    - `accuracy=0.6552734375`
    - `f1=0.7619689817936615`
    - `roc_auc=0.6882353860642509`
  - `EXP154` / `user-ltv`:
    - `r2=-0.10342013304848963`
    - `mae=79.65295739190421`
    - `rmse=148.29562404154896`
- Partial interpretation under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is `neutral` because the MAE delta is far inside the current wide LTV noise band
- Failure signature:
  - `EXP155` / `item-incoterms` failed before the first evaluation
  - runtime error:
    - `ValueError: Query-conditioned token stats require query/key projections.`
  - root cause:
    - the MLP interface was included in the inner query/key projection creation branch but omitted
      from the outer initialization guard
- Interpretation:
  - the bundle cannot be judged globally because the decisive salt-side task produced no
    evaluation line
  - the correct next action is a clean projection-fix rerun, not a new architecture branch

### EXP-156 / EXP-157 / EXP-158

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise query-conditioned token stats MLP projection-fix rerun
- Hypothesis:
  - no scientific change relative to `EXP153/154/155`; after fixing the missing projection
    initialization guard, the learned token-stats MLP scorer should finally produce a clean
    three-task verdict
- Evidence basis:
  - `EXP153` was clearly better on `user-churn`
  - `EXP154` was neutral under replay-aware effective delta on `user-ltv`
  - `EXP155` was implementation-blocked before any salt-side evaluation
- Code fix:
  - added `candidate_pairwise_query_conditioned_token_stats_mlp` to the label-token projection
    initialization guard in `model.py`
  - verified locally with `python -m py_compile model.py main.py stage3_research.py stage3_orchestrator.py`
- Change summary:
  - keep `autocomplete_decision_interface=candidate_pairwise_query_conditioned_token_stats_mlp`
  - keep Stage 3 finetune-only
  - keep serial batch-equivalent scheduling and explicit safe GPU `5,6`
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp156_candidate_pairwise_query_conditioned_token_stats_mlp_projectionfix_rerun.json`
  - resolved serial batch-equivalent placement at launch time:
    - `user-churn` / `EXP156`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
    - `user-ltv` / `EXP157`: target `lab25211`, GPUs `5,6`, per-rank batch `1`
    - `item-incoterms` / `EXP158`: target `lab25211`, GPUs `5,6`, per-rank batch `2`
  - first post-launch status check:
    - `EXP156` window is up and loading `rel-amazon`
    - `EXP157/158` are correctly queued behind the serial controller
    - orchestrator recommendation: `keep_watching`
- Automation:
  - created a fresh current-thread heartbeat automation named `Monitor Stage 3 EXP156 bundle`
    with id `monitor-stage-3-exp156-bundle`
- Final subset test metrics:
  - `EXP156` / `user-churn`:
    - `average_precision=0.6905745143107572`
    - `accuracy=0.658203125`
    - `f1=0.7596153846153846`
    - `roc_auc=0.6879442601298149`
  - `EXP157` / `user-ltv`:
    - `r2=-0.07244480884880922`
    - `mae=77.38431440794841`
    - `rmse=146.199322363753`
  - `EXP158` / `item-incoterms`:
    - `accuracy=0.560546875`
    - `macro_f1=0.0718397997496871`
    - `micro_f1=0.560546875`
    - `mrr=0.6785764663938492`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`
  - `user-ltv` is `neutral`, because the MAE improvement is smaller than the current
    `6.511244718243262` noise-floor-aware threshold
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.1362116209528319` against an
    effective threshold of `0.013037038915945098`
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Deeper interpretation:
  - this scorer only changes autocomplete multiclass behavior, so the Amazon-side changes should
    be treated mainly as no-op rerun drift / protocol-control evidence rather than as causal
    support for the scorer
  - the decisive salt-side result exactly repeats the failed MRR plateau from several scalable
    pairwise descendants: `EXP140`, `EXP143`, `EXP152`, and now `EXP158` all land at
    `mrr=0.6785764663938492`
  - this repeated equality is stronger than a generic regression: it suggests the current scalable
    pairwise/token-stat scorer family is collapsing to the same weak candidate-ranking behavior,
    not gradually approaching the hybrid scorer's useful candidate-token signal
  - therefore the learned token-stat MLP is not just under-tuned; the current feature interface is
    missing a qualitatively different signal
- Search-space consequence:
  - do not launch another near-duplicate fixed pooled-label, static token-stat, or small
    token-stat weighting variant
  - preserve the baseline design for this already-tested subpart and shift the next candidate to a
    different mechanism that is still scalable, candidate-aware, non-`O(L*C)`, and Stage
    4-compatible
  - before launching the next bundle, improve the GPU scheduler so representative tasks can run in
    packed batch-equivalent waves instead of the current coarse all-serial shape

### EXP-159 / EXP-160 / EXP-161

- Date: 2026-05-09
- Branch: `main`
- Target component: candidate pairwise baseline-anchored residual scorer
- Hypothesis:
  - the repeated salt-side failure may be caused by replacing a useful baseline autocomplete score
    surface with a newly learned candidate-aware scorer
  - keep the raw-label baseline scorer as the primary multiclass decision surface, then add only a
    zero-initialized, `tanh`-bounded candidate-aware residual
  - if the residual signal is useful, it can learn a small correction; if it is noisy, the model
    should remain much closer to the already strong baseline ordering
- Evidence basis:
  - `EXP108` preserved the `item-incoterms` salt-side signal near strict baseline
    (`mrr=0.815460689484127`) while later scorer-replacement variants collapsed
  - `EXP129`, `EXP141`, `EXP150`, and `EXP156` all repeated the pattern:
    `user-churn` better, `user-ltv` neutral, `item-incoterms` worse
  - the exact repeated `item-incoterms` plateau at `mrr=0.6785764663938492` across several
    descendants suggests a mechanism-level score-surface collapse rather than simple under-tuning
- Change summary:
  - add `autocomplete_decision_interface=candidate_pairwise_baseline_residual`
  - add `autocomplete_residual_scale=0.1`
  - initialize the residual scorer's final affine layer to zero for this interface
  - return `baseline_raw_label_scores + tanh(residual) * residual_scale`
  - keep Stage 3 finetune-only and non-autoregressive; no `--pretrain`
- GPU scheduling update:
  - implemented `packed_batch_equivalent` tie-breaking so equal-time plans prefer fewer waves and
    better GPU-time utilization
  - verified pure scheduler examples:
    - six usable GPUs, `batch_size=4` plus `batch_size=2` -> one wave with `4+2`
    - three usable GPUs, `batch_size=4` plus `batch_size=2` -> one wave with `2+1`
    - three usable GPUs for the Stage 3 task shape -> `user-churn` first, then
      `user-ltv + item-incoterms` in one packed wave
- Launch status:
  - launched with `python stage3_research.py launch stage3_notes/candidates/exp159_candidate_pairwise_baseline_residual.json`
  - resolved packed batch-equivalent placement at launch time:
    - `user-churn` / `EXP159`: target `lab25211`, GPUs `5,6`, per-rank batch `2`, wave `0`
    - `user-ltv` / `EXP160`: target `lab25211`, GPU `5`, per-rank batch `2`, wave `1`
    - `item-incoterms` / `EXP161`: target `lab25211`, GPUs `6,7`, per-rank batch `2`, wave `1`
  - first post-launch status check:
    - packed controller is active
    - `EXP159` window is up and loading `rel-amazon`
    - `EXP160/161` are correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Automation:
  - refreshed the current-thread heartbeat automation `monitor-stage-3-exp156-bundle` so it now
    monitors `EXP159/160/161`
- Interpretation to apply after completion:
  - this bundle should be judged by salt-side preservation first: `item-incoterms` must not repeat
    the `0.6785764663938492` collapse
  - Amazon-side movement remains useful as a protocol-control check, but causal weight should stay
    on the autocomplete task because this interface changes only autocomplete multiclass scoring
  - if the bundle still fails on `item-incoterms`, preserve the baseline scorer and move the next
    attempt away from pairwise scorer residuals rather than increasing residual scale immediately
- Final subset test metrics:
  - `EXP159` / `user-churn`:
    - `average_precision=0.6947261791048163`
    - `accuracy=0.6552734375`
    - `f1=0.758714969241285`
    - `roc_auc=0.6856582376916254`
  - `EXP160` / `user-ltv`:
    - `r2=-0.15993473908143652`
    - `mae=79.12969741201726`
    - `rmse=136.4337144441213`
  - `EXP161` / `item-incoterms`:
    - `accuracy=0.6015625`
    - `macro_f1=0.11599718111346018`
    - `micro_f1=0.6015625`
    - `mrr=0.6990761110634157`
- Result under replay-aware effective deltas:
  - `user-churn` is clearly `better`, with ROC-AUC delta `+0.031637334453389454` against the
    effective threshold `0.002552699600237518`
  - `user-ltv` is `neutral`, because the MAE improvement `0.1428752870135952` is far inside the
    wide effective threshold `6.511244718243262`
  - `item-incoterms` is clearly `worse`, with MRR delta `-0.11571197628326535` against the
    effective threshold `0.013037038915945098`
- Decision:
  - completed and finalized on 2026-05-09
  - global verdict `failed`
- Deeper interpretation:
  - this bundle partially avoided the exact repeated salt-side plateau: `EXP161` reached
    `mrr=0.6990761110634157` instead of the earlier `0.6785764663938492`
  - that improvement over the plateau is real-looking in size, but it is still much smaller than
    the gap to the strict baseline `mrr=0.8147880873466811`
  - therefore baseline anchoring softened the scorer-replacement collapse but did not preserve the
    useful raw-label score surface enough for `item-incoterms`
  - since this interface only changes autocomplete multiclass scoring, the Amazon-side gain should
    again be treated mostly as no-op rerun drift / protocol-control evidence
- Search-space consequence:
  - do not manually tune `autocomplete_residual_scale` or hand-scan residual strength
  - treat pairwise residuals over the current feature path as exhausted unless a new paper- or
    ablation-backed mechanism changes the information path
  - the next candidate should keep the Stage 4-compatible constraints but move away from small
    pairwise scorer residuals and away from fixed pooled-label / token-stat near-neighbors

### EXP-162 / EXP-163 / EXP-164

- Date: 2026-05-09
- Branch: `main`
- Target component: autocomplete hard-negative ranking auxiliary
- Hypothesis:
  - the repeated salt-side failure may be less about needing another candidate representation and
    more about the supervised objective not directly optimizing the hard-negative ordering that
    drives MRR
  - keep the existing cheap candidate-score surface and add a margin-free hard-negative ranking
    auxiliary for autocomplete multiclass tasks only
  - this should pressure the true label score above the strongest wrong candidate without returning
    to autoregressive candidate rollout, token-stat near-neighbors, or residual-scale tuning
- Evidence basis:
  - `EXP108` preserved the `item-incoterms` signal near the strict baseline with a scalable
    candidate-token / raw-label-related surface
  - `EXP126` showed that shortlist autoregressive reranking remained too expensive relative to
    its salt-side payoff
  - `EXP129`, `EXP156`, and `EXP159` showed that scorer replacement, token-stat weighting, and
    baseline-anchored residuals still leave `item-incoterms` clearly worse
- Change summary:
  - add `--autocomplete_rank_auxiliary`
  - when enabled for autocomplete multiclass tasks, add
    `softplus(hardest_negative_score - true_label_score)` to the normal cross-entropy task loss
  - keep Stage 3 finetune-only; no `--pretrain`
  - do not change `autocomplete_decision_interface`, prompt packaging, sampling, basis losses, or
    residual scale
  - wire the flag through `tune_hyperparameters.py` for later retune compatibility, but do not run
    an Optuna search during this screening step
- Launch plan:
  - candidate spec:
    `stage3_notes/candidates/exp162_autocomplete_hard_negative_rank_auxiliary.json`
  - use `packed_batch_equivalent` scheduling
  - judge jointly with replay-aware effective deltas
- Launch status:
  - launched with
    `python stage3_research.py launch stage3_notes/candidates/exp162_autocomplete_hard_negative_rank_auxiliary.json`
  - resolved packed batch-equivalent placement at launch time:
    - `user-churn` / `EXP162`: target `lab25211`, GPUs `0,5`, per-rank batch `2`, wave `0`
    - `user-ltv` / `EXP163`: target `lab25211`, GPUs `6,7`, per-rank batch `1`, wave `0`
    - `item-incoterms` / `EXP164`: target `lab25211`, GPUs `0,5,6,7`, per-rank batch `1`, wave `1`
  - first post-launch status check:
    - `EXP162` and `EXP163` windows are up in startup / DB-load
    - `EXP164` is correctly waiting for wave `1`
    - orchestrator recommendation: `keep_watching`
- Decision:
  - running
