# Part 3 Constraint Conservation Implementation

Date: 2026-05-15

Scope:

- clean worktree only: `G:\RelLLM-2\Rel-LLM-clean-p13`
- branch: `codex/stage3-clean-p13`
- base state: EXP192-family Part 1 after Phase 2 persistence

## What is implemented now

This pass moves Part 3 from "documented intent plus exposed knobs" to a real code path inside
`model.py`.

The implemented constraint-conservation layer now does all of the following during
`align_token_prompts(...)`:

- split basis transfer into schema branch vs value/stat branch using `basis_types`
- keep sparse basis assignment and honor `basis_assignment_topk`
- support confidence-gated conservative transfer through:
  - `basis_gate_strategy`
  - `basis_gate_token_floor`
  - `basis_gate_graph_floor`
- apply post-alignment target retention when `basis_lambda_postalign_tok > 0`
- apply minibatch entity-identity contrastive preservation when
  `basis_lambda_entity_identity > 0`
- apply schema/value branch orthogonality regularization when
  `basis_lambda_branch_orth > 0`
- preserve FK direction asymmetry in target construction by preferring directed FK basis indices
  over undirected table-pair fallback when available

`main.py` now also logs the new alignment-side components:

- `train/align_postalign_token_bce`
- `train/align_entity_identity`
- `train/align_branch_orth`
- `train/align_token_gate`
- `train/align_graph_gate`

## Important boundary

This is intentionally a coarse-grained Part 3 implementation pass, not a full scientific win
claim.

Still not implemented in this pass:

- a new standalone bridge-sensitivity loss
- a revived route-consistency or FK-direction-only bundle
- Part 3 Optuna or final full test

That is deliberate. The current goal is to make the documented Part 3 layer real and launchable
before launching a fresh comparison wave.

## Fair-comparison rule for the next run

The first Part 3 comparison run should keep the same task-specific hyperparameters as the current
validated Part 1 Phase 2 best settings, so the mechanism comparison is apples-to-apples.

Use these fixed best-setting sources as the first comparison baseline:

- `optuna_runs/exp192_user_churn_optuna_20260513t2307/best_trial.json`
- `optuna_runs/exp193_user_ltv_optuna_20260514t141722/best_trial.json`
- `optuna_runs/exp194_item_incoterms_optuna_20260513t2307/best_trial.json`

Interpretation rule remains unchanged:

- screening / `test_subset` is only the continuation gate
- final science still uses full-test references
- user-ltv subset/full-test scale mismatch still applies
- item-incoterms subset remains too optimistic for final judgment

## Next concrete action

1. render or register the Part 3 screening candidate using the same task-specific best settings
2. run static sanity plus a short launchability/smoke gate
3. run the first fixed-hyperparameter Part 3 screening wave
4. only if that wave is promising, move on to separate Optuna
5. only after Optuna selection, run separate final-test-only confirmation

## First screening outcome

The first fair fixed-hyperparameter screening wave has now completed as:

- `EXP195` / `user-churn`
- `EXP196` / `user-ltv`
- `EXP197` / `item-incoterms`

Outcome summary:

- official report:
  - `stage3_notes/reports/exp195_constraint_conservation_transfer.report.json`
- bundle verdict:
  - `failed`
- task reading:
  - `user-churn`
    - final judged test metric `roc_auc=0.6280925393741872`
    - worse than the screening baseline and worse than the full-test reference
  - `user-ltv`
    - final judged test metric `mae=86.30371976418479`
    - worse than the screening baseline and worse than the full-test reference
  - `item-incoterms`
    - final judged test metric `mrr=0.7937972780257936`
    - still below the optimistic screening baseline `0.8147880873466811`
    - but clearly above the stored full-test reference `0.7043105782857789`

Program consequence:

- this first coarse integrated Part 3 pass should not advance directly to Optuna
- the mechanism is not retired as pure noise, because the salt-side signal remains meaningful
- but the current integrated transfer is too harmful on the Amazon tasks to promote as-is
- the next justified action is a narrower Part 3 follow-up that keeps the useful
  `item-incoterms` signal while reducing Amazon-side over-transfer

## Next narrowed follow-up

The next follow-up should stay screening-only and should not jump to Optuna yet.

Current working hypothesis from `EXP195/196/197`:

- the useful signal is not "constraint conservation everywhere at the same strength"
- instead, the current integrated transfer is likely too permissive on the Amazon tasks
- the salt task can still benefit from the stronger transfer path

So the next bundle should narrow the mechanism rather than replace it:

- keep the validated Part 1 base unchanged
- keep the salt-side transfer close to the first Part 3 pass
- soften the Amazon-side transfer through confidence gating and lower transfer amplitudes
- reduce Amazon-side post-alignment retention pressure before spending any Optuna budget

That follow-up is the right next test because it still probes the same Part 3 causal story,
but avoids turning the search into manual broad hyperparameter scanning.

## Second narrowed follow-up outcome

The second screening-only narrowed follow-up has now completed as:

- `EXP198` / `user-churn`
- `EXP199` / `user-ltv`
- `EXP200` / `item-incoterms`

Outcome summary:

- official report:
  - `stage3_notes/reports/exp198_constraint_conservation_gated_amazon_transfer.report.json`
- bundle verdict:
  - `failed`
- candidate status:
  - `retune_plausible`
- task reading:
  - `user-churn`
    - final judged test metric `roc_auc=0.6333745404337985`
    - still worse than the screening baseline and worse than the full-test reference
    - this improved only slightly over `EXP195`, so Amazon-side harm was reduced only marginally
  - `user-ltv`
    - final judged test metric `mae=82.44562131792307`
    - versus the screening baseline it moved from `worse` in `EXP196` to `neutral`
    - this is evidence that narrower Amazon transfer was directionally helpful, but still not
      enough to treat the bundle as promotable
  - `item-incoterms`
    - final judged test metric `mrr=0.7937578209550865`
    - still below the optimistic screening baseline `0.8147880873466811`
    - but again clearly above the stored full-test reference `0.7043105782857789`

Program consequence:

- `EXP198` should not advance as a full bundle to Optuna or final full test
- the Part 3 family is still not retired, because the salt-side continuation signal remains strong
  and the Amazon-side narrowing did improve `user-ltv`
- the most plausible remaining issue is now that the Amazon tasks are still over-constrained by the
  new conservation losses themselves, not only by residual transfer amplitude

## Third narrower follow-up

The next justified screening step should therefore isolate conservative transfer from the newer
Amazon-side conservation penalties:

- keep the validated Part 1 base unchanged
- keep the stronger salt-side transfer path unchanged
- keep Amazon-side confidence gating, but raise the gate floors further
- reduce Amazon-side residual/graph injection further
- set Amazon-side `basis_lambda_postalign_tok=0`
- set Amazon-side `basis_lambda_entity_identity=0`
- keep Amazon-side `basis_lambda_branch_orth=0`

That next follow-up is the cleanest causal split now available because it asks whether the useful
signal is "gated transfer only" rather than "gated transfer plus added conservation losses" on the
Amazon tasks.
