# Stage 3 Architecture Note (2026-05-12)

## Scope

This note defines the next Stage 3 implementation step after `EXP189/190/191` finalized failed.

It follows the current hard constraints:

- Stage 3 representative experiments remain finetune-only.
- Do not use the existing `--pretrain` path.
- Do not launch another small scorer / routing / distillation neighbor.
- Keep the mechanism Stage4-compatible and cross-database plausible.

## Reuse Boundary

Reusable current components:

- the current `HeteroEncoder` + `HeteroGraphSAGE` backbone
- the existing basis artifact and basis-index maps
- the existing prompt-token alignment path in `model.py`
- the Stage 3 candidate / launch / judge tooling

Not reusable for Goal A:

- the current `pretrain()` path in `model.py`
- any path gated by `--pretrain`
- any LLM-coupled pretraining logic that reads `batch[table].df` and trains through
  yes/no prompt reconstruction

The new Goal A path must therefore be a separate GNN-only artifact pipeline.

## Goal A: GNN Independent Representation Training

Minimal implementation:

1. add a standalone `python -m gnn_repr` artifact builder
2. train only the graph-side `HeteroEncoder` + `HeteroGraphSAGE`
3. save a reusable artifact containing graph-side weights and metadata
4. load that artifact into Stage 3 finetune runs through explicit new args, without invoking
   `--pretrain`

Paper / prior basis:

- `GraphMAE: Self-Supervised Masked Graph Autoencoders`:
  masked feature reconstruction is a strong graph-SSL signal and avoids tying the objective to
  downstream labels.
- `Heterogeneous Graph Masked Autoencoders`:
  heterogeneous graphs benefit from attribute masking plus structure-aware edge reconstruction.
- `Strategies for Pre-training Graph Neural Networks`:
  transferable GNN pretraining should use self-supervised node-level objectives rather than only
  downstream labels.

Chosen objectives:

- masked node-attribute reconstruction at the graph-embedding level
  - practical reason: the current graph tensors already expose stable encoded node features, while
    raw-column generative reconstruction would drag the old LLM-coupled path back in
  - causal story: force the GNN to recover masked node information from neighbors, schema context,
    and graph structure
- directed FK / relation edge prediction
  - score observed directed edge types against corrupted negatives
  - this makes relation existence and direction explicit in the artifact objective

Stage4 compatibility:

- the artifact is dataset-level and label-free
- it can be trained on source databases without target-task labels
- it learns a reusable graph representation module rather than a target-head shortcut

## Goal B: Relational Invariant Preservation During Alignment

Minimal implementation:

1. keep the current basis-alignment path
2. split basis residuals into:
   - schema branch
   - value/statistics branch
   - entity residual branch
3. add explicit post-alignment invariant losses

Chosen invariants to protect:

- entity identity
- relation / FK direction evidence
- table / schema / value distinction
- sparse basis assignment instead of dense mixing

Concrete patch:

- entity identity contrastive loss
  - align each sample's pre-alignment seed-entity state with its post-alignment state
  - push away mismatched entities within the minibatch
- post-alignment basis-target retention loss
  - after alignment, token states must still predict their table / FK / join / stat basis targets
  - this is the minimal auxiliary mechanism for preserving relation/table/edge-path cues after
    cross-space transfer
- branch-preserving alignment
  - schema bases and stat/value bases are projected separately
  - a learned gate fuses schema/value/entity branches
  - optional orthogonality regularization prevents schema and value branches from collapsing into
    one dense mixture
- sparse basis assignment
  - keep sparsemax and add a fixed top-k mask option instead of a manual scan

Paper / prior basis:

- `GraphCL: Contrastive Self-Supervised Learning of Graph Representations`:
  contrastive agreement is a reasonable way to preserve node identity across transformed views.
- prior Stage 3 ablations:
  - `EXP060/063`: reconstruction-only alignment pressure was too global
  - `EXP066/069`: local direction / route penalties alone were too weak
  - `EXP189/190/191`: downstream cheap scorer work is exhausted, so the missing signal must move
    upstream into representation + alignment

## Why This Stays Inside The Search Boundary

This patch does not:

- use `--pretrain`
- re-enter the old pretrain warmup path
- launch another scorer / routing / residual / token-budget / batching near-neighbor
- introduce a target-database-only fixed multiclass head

This patch does:

- move up one level in the pipeline
- add one coherent representation artifact module
- add one coherent alignment-invariant module
- preserve the Stage 4 story: reusable graph representations plus reusable alignment constraints
