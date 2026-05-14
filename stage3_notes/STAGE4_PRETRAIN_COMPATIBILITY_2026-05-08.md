# Stage 4 Pretrain Compatibility Memo (2026-05-08)

This note does **not** change the current execution stage.

Stage 3 remains finetune-only.

The purpose of this memo is different:

- prevent Stage 3 from converging to a multiclass interface that becomes unusable once Stage 4
  enters leave-one-database pretrain
- make the long-term target explicit now, while active Stage 3 architecture choices are still
  being made

## 1. Long-Term Objective

The final program objective is not only:

- strong Stage 3 finetuned performance on the representative-task bundle

It is jointly:

1. strong Stage 4 pretrain-time zero-shot transfer across databases
2. strong downstream finetune performance after target-database adaptation

That means Stage 3 should avoid winning through an interface that depends on target-database
supervision in a way that Stage 4 cannot reuse.

## 2. Why This Matters

The current representative-task bundle contains:

- `user-churn` for binary classification
- `user-ltv` for regression
- `item-incoterms` for autocomplete multiclass ranking

RelBench does **not** provide the same task-form coverage uniformly across databases.

In particular:

- multiclass autocomplete supervision is structurally rarer than binary / regression supervision
- a leave-one-database Stage 4 pretrain program may easily end up with no true source-database
  supervised multiclass task that matches the held-out target-database multiclass task

Therefore a Stage 3 solution that requires:

- a database-specific multiclass softmax head
- target-database-only label supervision
- a decision rule that cannot be trained through transferable proxy tasks

is not a safe long-term final architecture.

## 3. Stage 4-Compatible Design Preference

For future Stage 4 compatibility, prefer multiclass interfaces that can be interpreted as:

- shared sample representation
- shared candidate representation
- shared sample-candidate scoring or ranking function

Prefer:

- candidate-aware scorers
- candidate text / candidate token interfaces
- shortlist + rerank structures
- ranking-compatible losses or decision rules
- interfaces that can be trained through proxy candidate-selection tasks

Deprioritize as final endpoints:

- database-specific fixed-class softmax heads
- heads whose semantics exist only when the exact target-database label set is known during
  training
- implementations that require true supervised multiclass tasks in source databases in order to
  learn any useful decision mechanism

## 4. What Stage 4 Should Learn

Stage 4 should aim to pretrain a **task-form** rather than only a target label set.

The reusable capability is:

- given a sample and a candidate set, score or rank candidates correctly

This is stronger and more reusable than:

- given one database-specific class inventory, fit a single multiclass classifier head

Concretely, Stage 4 should prefer transferable supervision forms such as:

1. candidate ranking
2. candidate retrieval
3. text-conditioned value selection
4. categorical-column pseudo-multiclass prediction
5. entity / attribute candidate discrimination

These can be synthesized even on databases that do not expose a benchmark multiclass task.

## 5. Stage 3 Implication

Stage 3 is still allowed to optimize `item-incoterms` directly, because Stage 3 is a finetune
program and the representative-task bundle requires it.

But Stage 3 should now interpret multiclass-family results with one extra question:

- does this family reveal a decision mechanism that Stage 4 could later train from shared
  candidate-ranking or value-selection proxy tasks?

If the answer is no, the family may still be useful as evidence, but it should not become the
preferred long-term endpoint.

## 6. Current Preferred Long-Term Form

The currently safest long-term direction is:

- shared candidate-aware scorer
- non-exhaustive shortlist stage over all candidates
- optional richer reranking on a small subset
- no dependence on exhaustive `O(L*C)` rollout
- no dependence on a database-exclusive fixed-class head

This does **not** mean the exact Stage 3 detached-train top-k reranker is already the final
Stage 4 design.

It means the **interface family** is more future-compatible than:

- a target-database-only multiclass softmax head
- a scorer that only works when full supervised multiclass training is already available

## 7. Stage 4 Planning Guardrail

When Stage 4 starts, do not ask:

- how to pretrain the exact final multiclass head directly

Ask instead:

1. what transferable sample-candidate scoring tasks can be built across source databases
2. what proxy candidate sets can be constructed from categorical values, schema values, entities,
   or column values
3. how to train the scorer backbone and candidate interface so the held-out database can adapt
   quickly during finetune

## 8. Practical Rule Going Forward

From this point on, active Stage 3 multiclass architecture choices should satisfy both:

1. strong or at least promotable Stage 3 bundle behavior
2. a plausible Stage 4 training story that does not depend on source-database true supervised
   multiclass tasks

If a Stage 3 candidate cannot tell a believable Stage 4 training story, treat it as:

- local finetune evidence only
- not a safe final mainline endpoint
