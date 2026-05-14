# EXP192 Follow-up Plan

Date: 2026-05-13

Scope:

- candidate: `stage3_notes/candidates/exp192_gnn_repr_alignment_invariants.json`
- clean worktree only
- finetune-only Stage 3 continuation
- no `--pretrain`

## Corrected interpretation

This bundle should not be retired by mechanically comparing to optimistic subset screening
baselines.

Task-aware reading:

- `user-churn`
  - screening is already stronger than the fixed screening baseline
  - full-test reference gap is small enough to justify task-specific Optuna retune
- `user-ltv`
  - the task is known to keep subset MAE in the `70+` range even when full-test MAE lands much
    lower
  - non-regression on the subset gate remains worth a bounded Optuna + full-test confirmation path
- `item-incoterms`
  - the meaningful scientific comparator is the full-test reference band near `mrr ~= 0.704`
  - the current `0.676` result is worse, but still close enough to justify one bounded retune pass
  - do not use the optimistic subset baseline near `0.81` as the retirement trigger

## Advancement decision

Record this candidate as:

- candidate status: `retune_plausible`
- global scientific verdict on the fixed-hyperparameter bundle: `failed`
- practical continuation decision: `worth Optuna plus final full-test confirmation`

## Optuna priority

Priority order:

1. `user-churn`
2. `item-incoterms`
3. `user-ltv`

Reasoning:

- `user-churn` already has positive screening evidence and only a small full-test-reference gap
- `item-incoterms` remains near the real full-test reference band and is the key blocker task
- `user-ltv` is still plausible, but should receive a smaller initial tuning budget because its
  subset/full-test mismatch makes interpretation noisier

## Recommended bounded continuation

### Phase 1: per-task Optuna on subset protocol

Use `tune_hyperparameters.py` with the same Stage 3 subset-selection protocol:

- `periodic_test_steps=512`
- `model_selection_source=test_subset`
- no per-trial full test
- keep the GNN representation artifact path and alignment-invariant mechanism fixed

Suggested initial budgets:

- `user-churn`: `20-30` trials
- `item-incoterms`: `20-30` trials
- `user-ltv`: `10-15` trials

### Phase 2: final full-test confirmation only for selected best settings

Run full test only for the final selected best setting per task.

Do not:

- run full test for every Optuna trial
- hand-scan values outside the Optuna path
- retire the mechanism before the bounded retune stage finishes

## Operational notes

- use the clean worktree scheduler policy:
  - during Optuna, treat `batch_size` as a normal search dimension rather than preserving
    batch-equivalent semantics
  - keep each Optuna task at at most 2 GPUs and prefer two parallel 2-GPU Optuna jobs over one
    4-GPU Optuna when 4 GPUs are idle
  - preserve artifact-backed compatibility constraints for Amazon GNN representation runs:
    `channels=256`, `num_layers=2`, `aggr=mean`
- do not reuse stale `task_launches`
- if GPU supply is tight, run the active retune task on the largest safe legal slice rather than
  spreading GPUs thinly across low-value concurrent jobs
