# EXP192 Optuna Continuation

Date: 2026-05-13

Scope:

- clean worktree only: `G:\RelLLM-2\Rel-LLM-clean-p13`
- branch: `codex/stage3-clean-p13`
- candidate family: `exp192_gnn_repr_alignment_invariants`
- Stage 3 remains finetune-only
- no `--pretrain`

Current interpretation:

- part1 should now be treated as implemented and under validation
- the active goal is coarse-grained pipeline completion and representative-task verification
- do not over-rotate into performance polish before the remaining major parts are implemented
- only reopen part1 early if the current validation exposes a concrete correctness, interface,
  stability, or launchability blocker

## Why this continuation exists

The fixed-hyperparameter representative bundle is not a final scientific win on full-test
reference comparison, but it is explicitly retained as `retune_plausible`.

That means:

- do not retire it by mechanically comparing to optimistic subset numbers
- do not full-test every speculative adjustment
- do run a bounded Optuna phase under the same subset-selection protocol
- only send the selected best setting per task to final full-test confirmation

## Required Optuna protocol

Every EXP192-family retune run must keep the Stage 3 screening protocol aligned with the current
judging rules:

- `periodic_test_steps=512`
- `model_selection_source=test_subset`
- subset metrics are the trial-selection gate
- full test runs only once for the selected best trial
- Optuna search and final full test are separate launches, not one bundled run
- cap each Optuna task at at most 2 GPUs so batch-size tuning remains fine-grained
- if 4 GPUs are idle, prefer two parallel 2-GPU Optuna tasks over one 4-GPU Optuna task

For artifact-backed continuation, the tuning launcher must also preserve the mechanism path:

- rel-amazon tasks use `artifacts/gnn_repr/rel-amazon/gnn_repr.pt`
- rel-salt task uses `artifacts/gnn_repr/rel-salt/gnn_repr.pt`
- preserve alignment-invariant fixed settings:
  - `basis_lambda_postalign_tok=0.1`
  - `basis_lambda_entity_identity=0.05`
  - `basis_entity_identity_temperature=0.1`
  - `basis_lambda_branch_orth=0.02`
  - `basis_assignment_topk=4`

## Priority order

1. `user-churn`
2. `item-incoterms`
3. `user-ltv`

## Transition rule for part3

Starting part3 means the current part1 state is accepted for this program phase.

That requires:

- part1 is already implemented in the repo
- representative-task validation is running on the real pipeline rather than only toy checks
- no active blocker currently says part1 is structurally unlaunchable or interface-incomplete

If that threshold is met and the next justified move is part3, first do two things:

- update the persistent docs to record part1 as implemented and validation-backed
- commit the clean-worktree part1 state before using it as the base for part3 work

## Initial bounded budgets

- `user-churn`: `20-30` trials
- `item-incoterms`: `20-30` trials
- `user-ltv`: `10-15` trials

## Operational rule

- for immediate debug feedback, continue within the same working session instead of waiting for a
  half-hour heartbeat
- use automation heartbeat only for long-running remote jobs such as Optuna, precompute, and full
  test confirmation
- during Optuna, do not enforce batch-equivalent semantics: `batch_size` is a normal hyperparameter
  and may scale up if the search finds it useful
- still cap each Optuna task at at most 2 GPUs and preserve artifact compatibility constraints when
  a run depends on fixed Amazon GNN representation artifacts
- do not reuse stale `task_launches`; resolve placement fresh each time

## Phase 2 closure on 2026-05-15

Separate final-test-only confirmation is now complete for all three representative tasks:

- `exp193_user_ltv_optuna_20260514t141722`
  - full-test result beat the stored reference:
    - `mae=15.633182804023788`
    - `rmse=50.20323115209195`
- `exp194_item_incoterms_optuna_20260513t2307`
  - full-test result beat the stored reference:
    - `mrr=0.7254209687717201`
    - `accuracy=0.6087479985602046`
- `exp192_user_churn_optuna_20260513t2307`
  - full-test result remained below the stored reference:
    - `roc_auc=0.6681287838379838`
    - `average_precision=0.7236622468421053`

Program reading:

- this is a mixed but scientifically meaningful Phase 2 outcome
- the direction should not be dropped
- the same final-test-only wave should not be relaunched immediately
- the transition threshold for using the current Part 1 state as a base for later work is now met:
  - part1 is implemented in the repo
  - representative-task validation reached real Phase 2 full-test evidence
  - no remaining blocker says part1 is structurally unlaunchable or interface-incomplete

Required next-step handling:

- persist the docs and candidate record as the clean-worktree Part 1 baseline
- commit that baseline before starting any Part 3-specific follow-up work
- retarget heartbeat automation from final-test monitoring to the next concrete Part 3 blocker,
  such as candidate selection, static validation, or launch preparation

## Concrete Part 3 blocker after Phase 2

The next blocker should now be treated more narrowly than a generic "continue Part 3" instruction.

Correct concrete blocker:

- continue with the documented Part 3 direction: implement the constraint-conservation layer
  rather than retargeting Part 3 into a separate Goal B attribution exercise
- use the current validated Part 1 state as the base for that work, exactly as the transition rule
  intended

Why this is the right next blocker:

- the documented Stage 3 pipeline view already places Part 3 at the
  `受约束守恒迁移 / constraint-conservation transfer` layer
- Phase 2 gave enough evidence that Part 1 is implemented and validation-backed, so the next step
  is to move forward into the planned constraint-conservation implementation path
- this still preserves the mixed Phase 2 reading: the family is promising enough to continue, but
  not yet a representative-set winner

Required handling for this blocker:

- keep the current `gnn_repr_artifact` path and validated Part 1 state as the base
- write or refine the Part 3 candidate/spec around constraint conservation, not around a new
  user-churn-specific attribution detour
- keep Optuna and final-test confirmation as separate later phases after the Part 3 implementation
  step is ready

## Part 3 implementation status on 2026-05-15

The documented constraint-conservation layer has now started moving from placeholder knobs to
actual code in the clean worktree.

Current persisted implementation status:

- `model.py` now contains a real constraint-conservation transfer path on top of the validated
  Part 1 base:
  - schema/value branch-split transfer from `basis_types`
  - sparse top-k assignment enforcement
  - optional confidence gating
  - post-alignment token-target retention
  - entity-identity contrastive preservation
  - branch orthogonality regularization
  - directed-FK-aware target construction
- `main.py` now logs the new Part 3 alignment components so later screening runs can diagnose
  whether the new mechanism is actually active

Immediate program consequence:

- the next required action is no longer more documentation-only correction
- it is to run the first fair comparison wave using the current Part 1 Phase 2 best settings as
  fixed hyperparameters
- only after that fixed-hyperparameter Part 3 screening wave should the program decide whether to
  invest in a separate Optuna phase

## Part 3 screening result on 2026-05-15

That first fair fixed-hyperparameter Part 3 screening wave is now complete under candidate
`exp195_constraint_conservation_transfer`.

Result:

- official bundle verdict:
  - `failed`
- bundle continuation decision:
  - do **not** run Optuna for this exact integrated Part 3 bundle

Task-aware reading:

- `user-churn`
  - regressed below the screening baseline and remained far below the full-test reference
- `user-ltv`
  - regressed below the screening baseline beyond the current noise-aware threshold
- `item-incoterms`
  - remained below the optimistic screening baseline
  - but still beat the stored full-test reference band comfortably, which means the salt-side
    mechanism signal is real enough to preserve as search-space evidence

Search-space consequence:

- treat the current coarse constraint-conservation integration as too aggressive for the Amazon
  tasks
- preserve the finding that the salt-side task can benefit materially from this family
- next action is not bundle-level Optuna, but a narrower follow-up Part 3 design that keeps the
  useful `item-incoterms` signal while softening or gating the transfer path for Amazon-side tasks

## Narrowed Part 3 follow-up on 2026-05-15

The next concrete follow-up is now registered as
`exp198_constraint_conservation_gated_amazon_transfer`.

Its design choice is intentionally narrow:

- keep the same validated Part 1 base and the same task-specific Phase 2 best settings
- keep the salt-side `item-incoterms` transfer close to the first Part 3 pass
- do not promote the failed `EXP195` bundle into Optuna
- instead, soften only the Amazon-side transfer path before spending more search budget

Mechanism-level change for this follow-up:

- `user-churn` and `user-ltv` now use explicit confidence gating
- Amazon-side `basis_residual_alpha` and `basis_graph_alpha` are reduced
- Amazon-side post-alignment retention and extra orthogonality pressure are reduced or removed
- `item-incoterms` keeps the stronger ungated conservation path so the useful salt-side signal is
  not accidentally erased before re-screening

Program consequence:

- this remains a screening-only bundle
- if the narrowed bundle still fails, the next move should likely be a deeper mechanism split
  rather than immediate Optuna
- if the narrowed bundle becomes non-regressive enough under the normal task-aware gate, only then
  should it advance to separate Optuna and later separate full test
