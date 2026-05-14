# Stage 3 Architecture Review

Date: `2026-05-07`

## Trigger

This review is triggered by Tier-3 lightweight-search exhaustion after the latest token-consumption
bundle:

- [exp078_split_local_global_graph_token_packaging.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp078_split_local_global_graph_token_packaging.json)
- [exp078_split_local_global_graph_token_packaging.report.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/reports/exp078_split_local_global_graph_token_packaging.report.json)

Current conclusion:

- do not launch another near-neighbor Stage 3 trick bundle immediately
- pause the lightweight finetune-only search loop
- use this review to redefine the next candidate family at a more structural level

## Stable Failure Pattern

Recent stable pattern across the representative tasks:

- `user-churn` is the easiest task to improve
- `user-ltv` can sometimes be recovered by weakening graph residual aggression or improving prompt
  consumption
- `item-incoterms` is the persistent blocker

The strongest recent partial win is `EXP078/079/080`:

- `user-churn`: improved
- `user-ltv`: improved
- `item-incoterms`: regressed

This means the current pipeline is not purely failing at global alignment.
It is failing to preserve the salt-side multiclass ranking signal while making the Amazon tasks
better.

## Layer Review

### Layer 1: basis construction / injection

Observed pattern:

- removing or weakening graph residual pressure can help `user-ltv`
- this does not protect `user-churn` and `item-incoterms` at the same time
- confidence gating reduced damage size but did not change the global verdict

Interpretation:

- the issue is not only residual strength
- the basis channel may still be semantically mismatched for the salt-side candidate-ranking task

### Layer 2: constraint-aware loss

Observed pattern:

- stronger reconstruction-style and route-aware conservative losses can help `user-churn`
- they repeatedly hurt `user-ltv` and `item-incoterms`

Interpretation:

- loss-side supervision is not the main missing ingredient anymore
- further local loss tweaks are likely to repeat the same tradeoff pattern

### Layer 3: constraint-aware sampling

Observed pattern:

- uniform pruning was destructive
- bridge/route-aware deterministic retention made all three tasks worse

Interpretation:

- simple heuristic evidence selection is not enough
- the problem is not solved by changing what is shown alone, at least in the current prompt path

### Layer 4: token compression / consumption

Observed pattern:

- separating an explicit global token from local evidence improved both Amazon representatives
- the salt-side ranking task still regressed

Interpretation:

- LLM consumption is part of the bottleneck
- but the current single shared prompt-token interface is still not preserving the right semantics
  for multiclass candidate ranking

## Architecture Questions

The next candidate family should answer these questions explicitly:

1. Is `item-incoterms` failing because the graph prompt does not preserve candidate-set
   discrimination, even when it improves global and regression signals?
2. Is the current unified graph-token interface forcing one representation to serve both scalar
   regression and multiclass ranking in a way that washes out salt-side label separation?
3. Is the seed/global/local token packaging still too shallow because there is no explicit
   candidate-aware or label-set-aware interface for multiclass tasks?

## New Code-Level Finding

Inspection on `2026-05-07` identified a likely structural mismatch in the current multiclass
autocomplete path:

- the active direct-supervision path uses `sample_head(hidden)` against
  `label_head(get_label_texts(...))`
- `get_label_texts(...)` for multiclass tasks can use semantic raw labels such as
  `target_col: raw_label`
- the training targets for that same path are integer class ids via cross-entropy on
  `batch[entity_table].y.long()`
- the codebase also contains a separate multiclass candidate-scoring path over integer class-id
  token sequences, which reinforces that the label interface is fragmented even if that path is
  not the active one in the current direct-supervision setup

This means the current `item-incoterms` path is at least training with mismatched target ids vs
label verbalizers, and the broader codebase still carries multiple multiclass label interfaces.

Implication:

- the next candidate family should prioritize train/infer label-interface unification for
  autocomplete-style MRR tasks before adding more lightweight retrieval, basis, or loss tweaks

## Follow-up Result: EXP081

`EXP081/082/083` tested the smallest viable form of label-interface unification by aligning the
autocomplete multiclass label scorer with class-id texts.

Observed result:

- `user-churn`: neutral
- `user-ltv`: regressed
- `item-incoterms`: improved versus recent failed bundles, but still below the strict MRR baseline

Updated implication:

- the label-interface hypothesis is real enough to move the salt-side metric in the right
  direction
- but a verbalizer-only unification is still too small
- the next structural family should be broader than a label-text tweak and should probably target
  candidate-aware multiclass decision structure end-to-end

## Variance Note

After `EXP081`, another issue became operationally important:

- some recent bundle deltas are small enough that they may be comparable to current training /
  screening variance

This does not invalidate the architecture findings.
It changes how aggressively we should interpret small improvements or regressions.

Required follow-up:

- establish a control rerun protocol before claiming that the next near-boundary delta is a real
  tuning gain
- see [STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md)

## Follow-up Result: EXP087

`EXP087/088/089` reran the unchanged refreshed strict baseline as a noise-floor control.

Observed result:

- `user-churn`: mild improvement versus the stored strict baseline
- `user-ltv`: strong regression versus the stored strict baseline
- `item-incoterms`: MRR regression almost numerically identical to the earlier `EXP078` result

Updated implication:

- a single near-boundary `item-incoterms` drop is not enough to reject the packaging family,
  because the baseline rerun itself reproduced nearly the same salt-side fall
- the `EXP078` `user-ltv` gain becomes more meaningful, not less, because the pure baseline rerun
  moved sharply in the opposite direction
- `user-churn` still looks easier to improve than the other two tasks, but the size of its
  current positive drift is not yet enough to call the packaging family confirmed

Next control decision:

- replay `EXP078` directly as `EXP090/091/092` before launching a new architecture family

## Follow-up Result: EXP090

`EXP090/091/092` replayed the `EXP078` split local/global packaging family directly.

Observed result:

- `user-churn`: replay flipped below the stored strict baseline
- `user-ltv`: replay stayed much stronger than the pure-baseline rerun `EXP088`, but still landed
  below the stored strict baseline
- `item-incoterms`: replay regressed even more than both the original `EXP080` run and the
  baseline-rerun `EXP089`

Updated implication:

- the split local/global packaging family is not stable enough to remain on the active frontier
- replay control reduces the chance that `EXP078` was a hidden global win waiting for confirmation
- the multiclass bottleneck is still real, but it now points away from more packaging-nearby
  variants and toward a more explicit candidate-aware decision interface

Current front-runner:

- `EXP084/085/086`: candidate-aware multiclass decision interface

Current concrete proposal for that family:

- autocomplete multiclass tasks only
- keep two candidate views at the label-scorer level:
  - class-id text
  - raw-label semantic text
- use the prompt-conditioned sample representation to fuse those two per-candidate views at
  scoring time instead of forcing one verbalizer choice globally

## Follow-up Result: EXP084

`EXP084/085/086` implemented the first dual-view fusion version of that idea.

Observed result:

- `user-churn`: strong improvement
- `user-ltv`: strong regression
- `item-incoterms`: remained in the same regressive MRR band as the earlier packaging and control
  bundles

Updated implication:

- a single global fusion gate over static class-id and raw-label candidate embeddings is too weak
- the salt-side task likely needs candidate-conditioned interaction terms, not just a better
  global mixture of two label views
- the next architecture step should move from `candidate_dual_view_fusion` toward a
  candidate-conditioned pairwise scorer

Next front-runner:

- `EXP093/094/095`: candidate-conditioned pairwise multiclass scorer

## Follow-up Result: EXP093

`EXP093/094/095` implemented the first candidate-conditioned pairwise scorer.

Observed result:

- `user-churn`: regressed below the stored strict baseline
- `user-ltv`: improved above the stored strict baseline
- `item-incoterms`: improved versus recent candidate-aware runs, but still remained below the
  strict MRR baseline

Updated implication:

- candidate-conditioned interaction is a real positive signal, not a dead end
- but it is still incomplete as a standalone scorer
- the most plausible missing signal is the explicit autoregressive candidate-token path already
  present elsewhere in the codebase

Next front-runner:

- `EXP096/097/098`: pairwise autoregressive hybrid scorer

## Follow-up Result: EXP096

`EXP096/097/098` implemented the pairwise + autoregressive hybrid scorer.

Observed result:

- `user-churn`: regressed below the stored strict baseline
- `user-ltv`: improved clearly above the stored strict baseline
- `item-incoterms`: improved clearly above the stored strict baseline

Updated implication:

- this is the strongest salt-side result reached in the post-control phase
- the hybrid scorer is the first family to move both `user-ltv` and `item-incoterms` above their
  strict baselines at the same time
- because the implemented change only targets autocomplete multiclass tasks, the `user-churn`
  regression is high-value replay territory rather than enough by itself to discard the family

Next front-runner:

- `EXP099/100/101`: hybrid scorer replay control

## Follow-up Result: EXP099

`EXP099/100/101` replayed the pairwise + autoregressive hybrid scorer directly.

Observed result:

- `user-churn`: replay moved above the stored strict baseline
- `user-ltv`: replay stayed above the stored strict baseline
- `item-incoterms`: replay stayed above the stored strict baseline

Updated implication:

- the hybrid scorer is no longer just a high-value replay candidate; it is now the first
  post-control architecture family to survive replay as a non-regressive representative-task
  bundle
- the earlier `EXP096` `user-churn` regression is more plausibly rerun drift than a stable causal
  side effect from the autocomplete-only change
- Stage 3 should stop local architecture branching here and switch to repository-result
  confirmation

Next front-runner:

- `EXP102/103/104`: hybrid scorer full-test confirmation

## Follow-up Conclusion: Hybrid scorer implementation status

The `candidate_pairwise_ar_hybrid` line is still a valuable Stage 3 attempt.

What it established:

- candidate-conditioned interaction is a real positive signal
- semantic raw-label and class-id candidate views both carry useful information
- adding explicit candidate-token scoring helped reveal a stronger salt-side recovery than earlier
  dual-view-only interfaces

What prevents it from staying on the final mainline:

- the current exhaustive autocomplete multiclass implementation scales like `O(L*C)` rather than
  `O(L + C)`
- that complexity is not acceptable as the final path for future large-class tasks with hundreds
  of labels

Updated implication:

- keep the hybrid scorer as a valuable evidence source in the architecture review
- retire the exhaustive `candidate_pairwise_ar_hybrid` head as the final scalable implementation
- the next architecture family should preserve the candidate-aware signal while redesigning the
  multiclass scorer toward `O(L + C)`-style behavior

## Recommended Next Step

Do not continue the exhaustive hybrid head as the final promotion target.

Instead:

1. keep the hybrid scorer result as valuable evidence
2. stop further final-validation spending on the `O(L*C)` head
3. design the next multiclass scorer family around the same candidate-aware signal with
   `O(L + C)`-style scaling
4. only return to repo-facing confirmation after that scalable family clears subset and replay

## Non-Recommendations

Do not do these next:

- another route-aware conservative loss tweak
- another bridge-route sampling strength tweak
- another basis gate floor scan
- another tiny packaging reorder tweak
- pretrain or any `--pretrain` path

## Addendum: 2026-05-09

The post-review candidate-aware scorer branch has now produced a sharper constraint.

Updated interpretation:

- `candidate_pairwise_interaction`, positional pooling, and query-conditioned single-vector
  pooling all repeated the pattern: `user-churn` better, `user-ltv` neutral, `item-incoterms`
  worse
- the fixed-weight token-stats residual also failed cleanly under safe GPU placement
- the current justified follow-up is learned token-level evidence weighting, not another fixed
  pooled-label or hand-weighted token-stat tweak
- the active interface remains `candidate_pairwise_query_conditioned_token_stats_mlp`
- this direction is still Stage 4-compatible because it preserves a shared sample-candidate
  scoring form rather than a target-database-only fixed-class head

Operational note:

- `EXP153/154/155` is implementation-blocked rather than scientifically resolved because
  `EXP155` failed before evaluation with missing query/key projections
- the clean projection-fix rerun is `EXP156/157/158`

Follow-up after `EXP156/157/158`:

- the learned token-stat MLP line also failed cleanly
- the decisive salt-side metric repeated the same collapsed plateau as earlier scalable
  replacement scorers: `item-incoterms mrr=0.6785764663938492`
- the current interpretation is no longer "make the token-stat scorer stronger"; it is "stop
  replacing the useful baseline score surface with this scorer family"
- the next active candidate is `EXP159/160/161`, which keeps the raw-label baseline scorer and
  adds only a zero-initialized, bounded candidate-aware residual
- this is still Stage 4-compatible because it remains a shared, non-autoregressive, O(C)
  candidate scorer after one prompt encoding
