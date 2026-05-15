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
