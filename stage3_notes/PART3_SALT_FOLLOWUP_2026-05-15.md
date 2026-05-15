# Part 3 Salt Follow-up Boundary Correction

Date: 2026-05-15

Scope:

- clean worktree only: `G:\RelLLM-2\Rel-LLM-clean-p13`
- branch: `codex/stage3-clean-p13`
- active blocker after `exp210`, `exp211`, and the stopped `exp212`: define a truly differentiated
  salt-side Part 3 candidate before any next Optuna launch

## Boundary correction

The practical salt-side comparison target is **not** "any command line that contains the same
Part 3 flag names as `exp194`".

Important nuance:

- before the Part 3 implementation pass, several constraint-conservation flags were already
  exposed through CLI / launcher plumbing
- the implementation checkpoint on 2026-05-15 is what made those knobs become an active code path
  inside `model.py`
- therefore command-line overlap with the earlier Phase 2 best-setting records is not enough by
  itself to define a genuinely new salt-side Part 3 mechanism

The more reliable active comparison anchors are the repeated post-implementation screens:

- `EXP206` / `item-incoterms`
- `EXP209` / `item-incoterms`

Those two screens kept the same active salt-side configuration and produced the same judged metric:

- `mrr=0.7937755766369047`

So the real reason `exp212` should stay stopped is narrower and more scientific:

- it would have retuned the unchanged active salt-side Part 3 path already screened in
  `EXP206` / `EXP209`
- it would not have isolated which conservation component is actually carrying the surviving
  `item-incoterms` signal

## Next candidate

The next coherent salt-side move is
`stage3_notes/candidates/exp213_constraint_conservation_salt_postalign_only.json`.

Design:

- keep `user-churn` and `user-ltv` on Part1-like controls under the current code
- keep the validated `item-incoterms` transfer amplitudes from the Phase 2 best setting
- keep sparse assignment on salt-side with `basis_assignment_topk=4`
- keep salt-side `basis_lambda_postalign_tok=0.1`
- remove the coupled salt-side penalties:
  - `basis_lambda_entity_identity=0.0`
  - `basis_lambda_branch_orth=0.0`

Hypothesis:

- if the useful salt-side continuation signal mainly comes from post-alignment target retention,
  then this narrower postalign-only path should remain reference-positive on `item-incoterms`
- if it collapses, the previous salt-side continuation signal was not being carried by
  post-alignment retention alone, and later salt-side Optuna should not be spent on this narrower
  direction

## Operational consequence

- this candidate remains screening-only
- do **not** jump directly to Optuna or final full test from registration alone
- first run normal static validation and rendered-launch validation
- only if the candidate is clean and launchable should it become the next remote screening wave

## Current execution state

Static validation and local launch validation both passed from the clean worktree:

- `python -m json.tool stage3_notes/candidates/exp213_constraint_conservation_salt_postalign_only.json`
- `python -m py_compile model.py main.py tune_hyperparameters.py stage3_research.py stage3_orchestrator.py`
- `git diff --check`
- `python stage3_research.py render ... --allow-no-papers`
- `python stage3_research.py launch ... --allow-no-papers --dry-run`

The candidate was then launched on `lab25211` through the safe remote root with:

- `EXP213` / `user-churn` on GPU `0`
- `EXP214` / `user-ltv` on GPU `1`
- `EXP215` / `item-incoterms` on GPU `2`

First live recheck:

- all three tmux windows are present
- `EXP215` is already training normally
- `EXP213` and `EXP214` are still progressing through Amazon DB preload

Latest live outcome:

- `EXP215` / `item-incoterms` has already completed cleanly while `EXP213` and `EXP214` remain
  active
- the synced log shows normal early stopping at train step `2560`, not a runtime failure
- finishing metrics:
  - best test-subset `mrr=0.7937578209550865`
  - best test `mrr=0.7937578209550865`
- interpretation:
  - this is effectively unchanged from the repeated active salt-side anchor
    `mrr=0.7937755766369047`
  - so removing salt-side entity-identity and branch-orth while keeping post-alignment retention
    did not destroy the surviving `item-incoterms` continuation signal
  - the next decision still needs the completed `EXP213` / `EXP214` Amazon-side results before any
    separate follow-up Optuna is justified

Final bundle outcome:

- official report:
  `stage3_notes/reports/exp213_constraint_conservation_salt_postalign_only.report.json`
- bundle-level judgment:
  - global verdict `failed`
  - candidate status `retune_plausible`
- per-task reading:
  - `EXP213` / `user-churn`
    - best test `roc_auc=0.6569589159856792`
    - screening is slightly better than the strict baseline `0.6540209032382359`
    - still not strong enough to justify a separate user-churn Part 3 Optuna continuation
  - `EXP214` / `user-ltv`
    - best test `mae=70.9812234421447`
    - this is a strong improvement over the screening baseline `79.27257269903086`
    - under the task rule this remains `retune_plausible`, not a direct bundle winner
  - `EXP215` / `item-incoterms`
    - best test `mrr=0.7937578209550865`
    - still clearly above the stored full-test reference `0.7043105782857789`
    - so the narrower salt-side postalign-only path preserved the useful reference-positive signal

Next action:

- do **not** rerun the same three-task screening bundle
- do **not** spend more Amazon-side bundle budget just to keep the control tasks attached
- the justified next move is a separate salt-side single-task Optuna on `item-incoterms` for this
  genuinely differentiated postalign-only mechanism
- keep the normal Stage 3 separation rule:
  - Optuna first with `periodic_test_steps=512` and `model_selection_source=test_subset`
  - if the selected best setting is still scientifically worthwhile, run final full test later as
    a separate `--final-test-only` launch

Live continuation:

- the next justified launch has now started as a separate single-task Optuna:
  - study:
    `exp216_item_incoterms_part3_salt_postalign_optuna_20260515t130652`
  - target: `lab25211`
  - safe remote root only:
    `/fs/fast/u2021201693/lym/Rel-LLM-codex-stage3-clean-p13`
  - GPUs: `0,1`
  - world size: `2`
  - master port: `29684`
  - protocol:
    - Optuna-only
    - `periodic_test_steps=512`
    - `model_selection_source=test_subset`
    - `max_gpus_per_task=2`
    - no bundled final full test
- launch boundary:
  - this is the differentiated salt-side postalign-only mechanism, not a resume of the earlier
    `exp212` equivalence class
  - `basis_lambda_postalign_tok=0.1`
  - `basis_lambda_entity_identity=0.0`
  - `basis_lambda_branch_orth=0.0`
  - `basis_gate_strategy=none`
  - `basis_assignment_topk=4`
- first live verification:
  - remote tmux window `stage3-exp216-optuna` is alive
  - sqlite study DB and `trial_0000.log` were created in the safe remote root
  - `trial 0` entered real training and passed 250+ train steps
