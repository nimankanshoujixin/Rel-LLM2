# Stage 3 Architecture Review

Date: `2026-05-11`

## Trigger

This review is triggered by Tier-3 lightweight-search exhaustion after:

- `EXP180/181/182`: candidate-set guided prompt routing
- `EXP183/184/185`: sample-conditioned candidate-set prompt routing
- `EXP186/187/188`: candidate-set guided prompt group routing

All three bundles stayed finetune-only and passed static / sanity gates before launch.

## Latest Stable Pattern

The recent candidate-aware evidence-routing branch has a clear signature:

- `user-churn` reliably improves
- `user-ltv` remains neutral under the wide replay-aware effective delta
- `item-incoterms` remains the decisive blocker

The salt-side details matter:

- EXP180 reached `item-incoterms mrr=0.7432317302922772`
- EXP183 reached `item-incoterms mrr=0.6782075427827381`
- EXP186 reached `item-incoterms mrr=0.6782075427827381`
- strict screening baseline remains `mrr=0.8147880873466811`
- replay-aware effective delta is `0.013037038915945098`

Interpretation:

- EXP180 remains a fixed-screen failed but retune-plausible mechanism signal
- EXP183 and EXP186 show that manual prompt evidence routing variants are not a stable route to
  recovering `item-incoterms`
- continuing with top-k, token-budget, prompt-order, sample-conditioning, or group-routing variants
  would be near-duplicate search

## Layer Saturation Update

Layer 1, basis construction / injection:

- earlier basis residual, token-only, confidence-gating, and reconstruction variants produced
  partial wins but failed the representative bundle
- local basis injection changes have not protected `item-incoterms` while preserving `user-ltv`

Layer 2, conservative alignment:

- FK-direction and route-consistency conservative losses were paper-backed and already tested
- they did not become cross-task stable
- route-aware conservative loss is temporarily saturated

Layer 3, sampling:

- bridge/route sampling failed globally
- further route sampling is explicitly retired for now

Layer 4, token consumption:

- split local/global packaging helped some Amazon-side behavior but did not preserve salt-side MRR
- candidate-aware prompt routing gave one retune-plausible signal, then repeated the collapse band
  on two follow-ups

Candidate-aware scorer/interface family:

- token-stat, pooled-label, residual, late-interaction, Poly-encoder, shared-prefix, and routing
  near-lines have all failed fixed screening or proved impractical
- old `O(L*C)` reruns and manual shortlist/token-budget scans remain off limits

## Current Blocker

Do not launch another candidate until the next Stage 3 action answers this design question:

> What medium-scale mechanism changes the multiclass information path enough to preserve
> `item-incoterms`, while remaining Stage4-compatible and non-`O(L*C)`?

The answer cannot be:

- another prompt evidence routing order variant
- another residual scale or rank auxiliary weight
- another token-stat / pooled-label scorer neighbor
- another shared-prefix or old exhaustive autoregressive rerun
- another batching, chunk-size, or token-budget scan

## Review Work To Finish Next

The next heartbeat should not launch immediately.

It should finish a concrete architecture-consistency review:

1. Inspect the current autocomplete multiclass data path in `model.py`, `main.py`, and `utils.py`.
2. Identify whether `item-incoterms` is failing because candidate ranking depends on a brittle
   target-database score surface rather than a transferable sample-candidate mechanism.
3. Compare the best known salt-side signals:
   - EXP108 near-baseline candidate-token/raw-label behavior
   - EXP180 retune-plausible candidate-set routing
   - scorer-collapse bands around `0.678`
4. Draft exactly one next candidate family, or explicitly pause Stage 3 launch work if the review
   says a larger redesign is needed.

Any new candidate must be paper- or prior-ablation-backed, finetune-only, Stage4-compatible,
candidate-aware, and non-`O(L*C)` in LLM rollout.

## Review Result

The current autocomplete multiclass path has two separable facts:

1. The strongest salt-side evidence still comes from candidate-token scoring:
   - EXP099 was subset-promotable but used an exhaustive hybrid path that is not an acceptable
     long-term endpoint.
   - EXP108 kept most of that salt-side signal through class-id shared-prefix scoring and reached
     `item-incoterms mrr=0.815460689484127`, but failed the bundle because `user-ltv` regressed.
2. The later cheap student scorers repeatedly collapsed:
   - token-stat, pooled-label, residual, late-interaction, Poly-encoder, and prompt-routing
     variants mostly failed to reproduce the candidate-token score surface.
   - after EXP180, two routing follow-ups exactly repeated the `0.6782075427827381` salt-side
     collapse band.

This suggests the missing mechanism is not another prompt ordering rule. The bottleneck is that
the cheap sample-candidate scorer is being asked to discover the candidate-token score surface only
from hard labels and local pair features. That is a brittle target-database score surface, and it
does not provide a transferable Stage4 training story by itself.

## Next Candidate Family

Draft exactly one next family:

- `candidate_pairwise_prefix_distilled`

Core idea:

- keep the cheap pairwise sample-candidate scorer as the evaluation and inference interface
- during finetune only, compute the EXP108-style class-id shared-prefix candidate-token scores as a
  detached teacher
- add a distillation loss from the teacher candidate distribution to the cheap scorer distribution
- do not use shared-prefix scores as the final prediction at eval/test

Why this is not a retired shared-prefix rerun:

- shared-prefix is used only as a training signal, not as the promoted decision interface
- inference remains a shared sample-candidate scorer with `O(L+C)`-style behavior
- there is no manual prefix budget, top-k, residual-scale, or routing-order scan

Evidence basis:

- EXP099 proves candidate-token scoring can produce a non-regressive subset bundle.
- EXP108 proves class-id shared-prefix candidate-token scoring preserves near-baseline
  `item-incoterms` ranking.
- EXP156/158, EXP183/186, and related collapse-band repeats show the cheap scorer needs a stronger
  teacher signal, not another near-neighbor feature.

Stage4 compatibility:

- Stage4 can synthesize proxy categorical-value ranking episodes and distill candidate-token or
  text-conditioned value-selection teachers into the same cheap sample-candidate scorer.
- The final interface remains candidate-aware and does not require source databases to expose true
  supervised multiclass benchmark tasks.

Launch condition:

- implement the smallest coherent patch
- pass `py_compile`, `git diff --check`, and candidate JSON validation
- run an `item-incoterms` shape/throughput sanity gate
- launch only if sanity shows no abnormal effective items/sec regression
