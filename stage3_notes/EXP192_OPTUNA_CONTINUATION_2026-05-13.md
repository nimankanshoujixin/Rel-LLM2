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
