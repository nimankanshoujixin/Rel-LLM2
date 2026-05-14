# Paper Shortlist (2026-05-05)

This shortlist is not a literature survey. It is a working set of papers that can directly
inform the next Stage 3 candidates.

## 1. GraphPrompter

- Title: `Can we Soft Prompt LLMs for Graph Learning Tasks?`
- Link: [arXiv:2402.10359](https://arxiv.org/abs/2402.10359)
- Why it matters here:
  - it is very close to our current regime: GNN encoder + soft prompt into an LLM
  - it treats the main problem as a graph/text modality mismatch problem
- Practical implication:
  - our failures may be less about prompt token order and more about alignment quality
  - future candidates should focus more on alignment objectives and projection quality than on
    prompt serialization tricks

## 2. G-Retriever

- Title: `G-Retriever: Retrieval-Augmented Generation for Graph Question Answering`
- Link: [arXiv:2402.07630](https://arxiv.org/abs/2402.07630)
- Why it matters here:
  - it explicitly reports that a single graph embedding or soft prompt can fail to preserve the
    graph faithfully
  - it argues that direct retrieval of relevant subgraphs can reduce hallucination and information
    loss
- Practical implication:
  - our current prompt-side basis injection may be trying to compress too much relational
    structure into too small a prompt channel
  - a stronger direction may be task-conditioned subgraph selection or better graph pruning
    instead of more prompt-side decoration

## 3. GALM

- Title: `Graph-Aware Language Model Pre-Training on a Large Graph Corpus Can Help Multiple Graph Applications`
- Link: [arXiv:2306.02592](https://arxiv.org/abs/2306.02592)
- Why it matters here:
  - it studies joint graph + language pre-training on text-rich heterogeneous graphs
  - it is one of the more relevant papers for cross-task transfer rather than single-task prompt
    tricks
- Practical implication:
  - if we want one change that helps the representative task set broadly, representation learning
    and pre-training may be more promising than local prompt rearrangements
  - this supports shifting some Stage 3 budget toward GNN / alignment pretraining experiments

## 4. RGLM

- Title: `Toward Graph-Tokenizing Large Language Models with Reconstructive Graph Instruction Tuning`
- Link: [arXiv:2603.01385](https://arxiv.org/abs/2603.01385)
- Why it matters here:
  - it argues that text supervision alone causes text-dominant bias in graph-token LLM alignment
  - it adds explicit graph reconstruction supervision to improve alignment
- Practical implication:
  - our current basis-loss experiments may still be too weak or too indirect
  - a more promising candidate family is to add reconstruction-style graph supervision or stronger
    latent-space alignment instead of hand-tuning basis residual mixing coefficients

## 5. GraphRAG Survey

- Title: `Graph Retrieval-Augmented Generation: A Survey`
- Link: [arXiv:2408.08921](https://arxiv.org/abs/2408.08921)
- Why it matters here:
  - it organizes the design space into graph indexing, graph-guided retrieval, and graph-enhanced
    generation
- Practical implication:
  - our recent 40-experiment search has over-weighted the graph-enhanced generation corner
  - the next pipeline phase should deliberately allocate candidate slots to:
    - graph indexing / basis construction
    - graph-guided retrieval / pruning
    - graph-text alignment losses
  - and spend fewer slots on raw prompt token rearrangement

## Immediate Research Consequences

Based on these papers, the next better-than-random candidate families are:

1. stronger alignment supervision rather than prompt order changes
2. subgraph selection / pruning / retrieval quality changes
3. graph-aware pretraining or auxiliary reconstruction losses
4. projector / alignment architecture changes only when tied to an explicit alignment hypothesis
