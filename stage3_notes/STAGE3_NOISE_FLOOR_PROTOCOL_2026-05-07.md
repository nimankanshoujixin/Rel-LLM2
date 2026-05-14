# Stage 3 Noise-Floor Protocol

Date: `2026-05-07`

## Why This Exists

Recent Stage 3 bundles have reached a point where some results are:

- clearly worse
- clearly not promotable
- but close enough to the strict screening baselines that single-run interpretation is no longer
  sufficient

Examples:

- `EXP078/079/080` improved `user-churn` and `user-ltv` but regressed `item-incoterms`
- `EXP081/082/083` improved `item-incoterms` versus recent failed bundles, but not enough to cross
  the strict baseline and it regressed `user-ltv`

At this point, Stage 3 needs an explicit noise-floor protocol before more structural candidate
families are judged from one run alone.

## Core Principle

Fixed seed is not enough to guarantee identical outcomes in the current training stack.

The current pipeline still contains likely nondeterminism and variance sources:

- GPU execution and floating-point reduction order
- mixed precision / autocast
- PyG `NeighborLoader` training shuffle
- checkpoint selection on periodic test-subset metrics
- early stopping and scheduler decisions that amplify small metric differences

Therefore:

- one run is enough to reject obviously bad bundles
- one run is not enough to confirm small gains or small regressions near the decision boundary

## Control Bundles

The current control protocol starts with two bundle types:

1. baseline noise-floor control
   - rerun the unchanged strict-baseline Stage 3 config multiple times
   - purpose: estimate representative-task metric variance under the current protocol

2. partial-win replay control
   - rerun the strongest recent partial-win bundle family multiple times
   - purpose: estimate whether its observed gains or regressions are outside the baseline
     noise floor

Current machine-readable specs:

- [exp087_baseline_noise_floor_control.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp087_baseline_noise_floor_control.json)
- [exp090_partial_win_replay_control.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp090_partial_win_replay_control.json)

## Judgment Rule

Until the noise floor is estimated, use this conservative rule:

- treat large regressions as real failures
- treat small near-boundary wins or losses as inconclusive
- do not promote a candidate whose gain is not clearly larger than the observed rerun spread

## Immediate Next Step

Baseline control is now complete:

- `EXP087/088/089` confirmed that one unchanged-baseline rerun can still produce:
  - a mild positive `user-churn / roc_auc` drift
  - a large negative `user-ltv / mae` drift
  - an `item-incoterms / mrr` drop almost identical to the earlier `EXP078` salt-side regression

Implication:

- `EXP078`'s `item-incoterms` regression can no longer be treated as clean evidence against the
  packaging family from one run alone
- `EXP078`'s `user-ltv` gain becomes more interesting because the pure-baseline rerun moved
  sharply in the opposite direction

Therefore the next control step is no longer optional.

Before launching another structural candidate beyond the current architecture-review queue:

1. summarize the baseline rerun control bundle
2. replay the strongest recent partial-win bundle
3. compare:
   - `EXP078` vs `EXP090` for packaging-family stability
   - both against `EXP087` baseline-rerun drift

Replay control is now also complete:

- `EXP090/091/092` showed that the apparent `EXP078` packaging-family gains do not survive replay
- `user-churn` flipped from a clear positive delta to a negative one
- `user-ltv` stayed much better than the pure-baseline rerun `EXP088`, but still fell below the
  stored strict baseline
- `item-incoterms` regressed even more than both `EXP080` and `EXP089`

Operational conclusion:

- the control protocol has done its job for the packaging family
- further packaging-nearby reruns are low value
- Stage 3 should return to the broader candidate-aware multiclass decision-interface branch

## New Replay Trigger

`EXP096/097/098` created a new replay-worthy state:

- the pairwise + autoregressive hybrid scorer pushed both `user-ltv` and `item-incoterms` above
  their strict baselines
- the only regressing task was `user-churn`
- the implemented code change only targets autocomplete multiclass tasks, so the `user-churn`
  regression is especially likely to be a no-op rerun drift rather than a direct causal effect

Operational implication:

- unlike the earlier packaging family, this bundle is valuable enough to replay before rejection
- the next control step should be a direct replay of the hybrid scorer itself

## Replay Resolution: EXP099

`EXP099/100/101` completed the replay-control check for the hybrid scorer.

Observed result:

- `user-churn`: replay moved above the stored strict baseline
- `user-ltv`: replay remained above the stored strict baseline
- `item-incoterms`: replay remained above the stored strict baseline

Operational conclusion:

- the replay-confirmed hybrid scorer now exceeds the current rerun-control bar
- the earlier `EXP096` `user-churn` drop should not be treated as a stable blocker anymore
- the control protocol has done its job for this family, so the next step is no longer another
  replay but full-test confirmation under the same modeling configuration

## Current Resolution Update: 2026-05-09

The `EXP099/100/101` result remains valuable evidence, but it is no longer the active final
promotion target.

What changed after the replay resolution:

- the exhaustive `candidate_pairwise_ar_hybrid` line established that candidate-token evidence
  matters
- later architecture review rejected that exact head as a final endpoint because it scales like
  `O(L*C)` over input length and class count
- Stage 3 should therefore keep the hybrid line as evidence, not as the mainline implementation
- the active multiclass route is now the scalable, non-autoregressive, candidate-aware scorer
  family that preserves token-level candidate evidence without exhaustive candidate rollout

Current judgment rule:

- use replay-aware effective delta for every representative task
- `effective_delta = max(screening_metric_delta, noise_floor_primary_metric_delta)`
- current effective primary-metric deltas from `baseline_registry.json` are:
  - `user-churn / roc_auc`: `0.002552699600237518`
  - `user-ltv / mae`: `6.511244718243262`
  - `item-incoterms / mrr`: `0.013037038915945098`
- especially for `user-ltv`, small MAE movements inside the neutral band must not be called
  scientific wins or regressions

## Non-Goal

This protocol is not an excuse to stall forever.

Its purpose is:

- to avoid over-claiming tiny movements as real gains
- to avoid discarding promising directions for changes that may be within training variance
