# Rel-LLM

Rel-LLM is a relational learning project built on top of RelBench. The current training entrypoint is `main.py`.

The runtime pipeline is:

1. Load a temporal multi-table database from RelBench.
2. Materialize table columns into `TensorFrame`.
3. Build a heterogeneous graph from primary key / foreign key relations.
4. Encode nodes with tabular encoders, temporal encoders, and `GraphSAGE`.
5. Run one of two prediction backends:
   - `GNN` mode: pure graph model output
   - `LLM` mode: project graph embeddings into the LLM embedding space for classification or regression

This README is rewritten for the current code in this repository. The example commands below only use arguments that are actually supported by `main.py`.

---

## Key Files

- `main.py`
  - Training, validation, and test entrypoint
- `model.py`
  - Model definition for both GNN and LLM paths
- `text_embedder.py`
  - Text column embedding wrapper, supports `glove` and `mpnet`
- `train_script.txt`
  - Historical experiment command reference
- `tune_hyperparameters.py`
  - Optuna hyperparameter tuning wrapper for `main.py`
- `HYPERPARAMETER_TUNING_GUIDE.md`
  - Hyperparameter inventory and tuning strategy
- `TUNE_SCRIPT_README.md`
  - Short usage guide for the tuning script
- `MAIN_PIPELINE_EXPLANATION.md`
  - Detailed walkthrough of the `main.py` pipeline

---

## Installation

Recommended setup:

```bash
conda env create -f environment.yml
conda activate llm
python -m pip install -U "relbench[full]==2.1.1"
```

The environment file already includes the main dependencies used by this project:

- `torch`
- `torch-geometric`
- `transformers`
- `peft`
- `sentence-transformers`
- `relbench`
- `optuna`
- `wandb`

If this is a fresh machine, make sure the PyG-related packages match your CUDA and PyTorch versions.

Note:

- The repository keeps the old vendored RelBench snapshot under `relbench_v1_vendor/` only as a backup reference.
- The repository keeps the old vendored Torch Frame snapshot under `torch_frame_v1_vendor/` only as a backup reference.
- Runtime imports now resolve to the installed `relbench` package from the active environment.
- Runtime imports now resolve to the installed `torch_frame` package from the active environment.
- You can verify this with:

```bash
python -c "import relbench, importlib.metadata; print(relbench.__file__); print(importlib.metadata.version('relbench'))"
python -c "import torch_frame; print(torch_frame.__file__); print(torch_frame.__version__)"
```

---

## Data And Model Downloads

### RelBench data

`main.py` calls:

```python
get_dataset(..., download=True)
get_task(..., download=True)
```

So the first run will download the required RelBench dataset and task files automatically.

### HuggingFace models

LLM mode can load models such as:

- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

If a model requires gated access, log in to HuggingFace first.

`--model_type` can also be a local model directory, for example:

```bash
--model_type=./Llama-3.2-1B
```

### Text embedding models

Raw text columns are embedded through `text_embedder.py`. Current options:

- `glove`
- `mpnet`

Mapping:

- `glove` -> `average_word_embeddings_glove.6B.300d`
- `mpnet` -> `all-mpnet-base-v2`

---

## Main Supported Arguments

### Data and cache

- `--dataset`
- `--task`
- `--cache_dir`
- `--debug`

### Graph model parameters

- `--channels`
- `--aggr`
- `--num_layers`
- `--num_neighbors`
- `--temporal_strategy`
- `--text_embedder`
- `--text_embedder_path`

### LLM parameters

- `--model_type`
- `--llm_frozen`
- `--output_mlp`
- `--dropout`
- `--num_demo`
- `--max_new_tokens`
- `--loss_class_weight`

### Training parameters

- `--train_steps`
- `--pretrain`
- `--pretrain_epochs`
- `--val_steps`
- `--eval_steps`
- `--test_steps`
- `--early_stop_patience`
- `--early_stop_metric_delta`
- `--early_stop_loss_delta`
- `--batch_size`
- `--val_size`
- `--num_workers`
- `--lr`
- `--wd`
- `--seed`

Important:

- The current `main.py` does not support older arguments such as `--context` or `--context_table`.
- Treat the `argparse` definitions in `main.py` as the source of truth.
- `--train_steps` is the finetuning budget and replaces the old epoch-based budget.
- `--val_steps` controls how often validation runs during training.
- `--eval_steps` caps how many loader batches are consumed per intermediate validation pass.
- `--test_steps` caps how many loader batches are consumed by the final test pass.
- early stopping is optional and uses both validation metric plateau and windowed training-loss plateau.

---

## Common Training Commands

## LLM mode with frozen 1B model

This is the most common setup in the current repo.

### Amazon `user-churn`

```bash
python main.py \
  --dataset=rel-amazon \
  --task=user-churn \
  --train_steps=32768 \
  --batch_size=1 \
  --val_size=1 \
  --lr=0.001 \
  --wd=0.0015 \
  --dropout=0.4 \
  --val_steps=1000 \
  --eval_steps=1024 \
  --temporal_strategy=last \
  --text_embedder=mpnet \
  --llm_frozen \
  --loss_class_weight 0.6 0.4
```

To disable `wandb`, add:

```bash
--debug
```

## DDP multi-GPU launch

The current repo now supports single-node DDP training through `torchrun`.

If your server has NCCL P2P issues, launch with:

```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 main.py \
  --dataset=rel-amazon \
  --task=user-churn \
  --train_steps=32768 \
  --batch_size=1 \
  --val_size=1 \
  --lr=0.001 \
  --wd=0.0015 \
  --dropout=0.4 \
  --val_steps=1000 \
  --eval_steps=1024 \
  --temporal_strategy=last \
  --text_embedder=mpnet \
  --llm_frozen \
  --loss_class_weight 0.6 0.4 \
  --debug
```

In DDP mode:

- `batch_size` means per-GPU batch size
- global batch size is `batch_size * number_of_gpus`
- rank 0 handles `wandb` logging and console summaries
- validation and test predictions are gathered across ranks before metric computation

### Stack `user-engagement`

```bash
python main.py \
  --dataset=rel-stack \
  --task=user-engagement \
  --train_steps=8192 \
  --batch_size=256 \
  --val_size=256 \
  --lr=0.005 \
  --wd=0.0015 \
  --dropout=0.4 \
  --val_steps=200 \
  --eval_steps=1024 \
  --temporal_strategy=last \
  --text_embedder=mpnet \
  --loss_class_weight 0.2 0.8
```

## Pure GNN mode

If you do not want to load an LLM, set `model_type=gnn`:

```bash
python main.py \
  --dataset=rel-stack \
  --task=user-engagement \
  --model_type=gnn \
  --batch_size=512 \
  --lr=0.005 \
  --wd=0.15 \
  --dropout=0.45 \
  --val_steps=200
```

## LLM with `output_mlp`

If you want to use the final hidden state plus an MLP head instead of text generation style output, add:

```bash
--output_mlp
```

Example:

```bash
python main.py \
  --dataset=rel-trial \
  --task=study-adverse \
  --train_steps=32768 \
  --batch_size=256 \
  --val_size=256 \
  --lr=0.0001 \
  --wd=0.0015 \
  --dropout=0.15 \
  --val_steps=1000 \
  --eval_steps=1024 \
  --temporal_strategy=last \
  --llm_frozen \
  --text_embedder=mpnet \
  --output_mlp \
  --max_new_tokens=1
```

---

## Logs And Cache

### `wandb`

If `--debug` is not set, training will try to initialize `wandb`:

- Pretraining project: `rel-LLM-zero`
- Finetuning project: `rel-LLM`

If you only want local runs without online tracking, use:

```bash
--debug
```

### Cache directories

Default cache directory:

```bash
~/.cache/relbench_examples
```

It stores:

- task tables
- graph construction intermediates
- `stypes.json`
- materialized tensor data

Default text embedder cache directory:

```bash
./cache
```

You can override them with:

```bash
--cache_dir
--text_embedder_path
```

---

## Hyperparameter Tuning

The repo includes an Optuna wrapper:

```bash
python tune_hyperparameters.py --gpu-id 0
```

A typical run:

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --gpu-id 6 \
  --n-trials 30 \
  --train-steps 4096 \
  --eval-steps 1024 \
  --study-name amazon_user_churn_llama1b
```

If you want to restart the same study name from scratch instead of resuming it, add:

```bash
--reset-study
```

The tuning script will:

- sample hyperparameters
- launch `main.py`
- parse validation metrics
- skip the test split during trial search
- rerun the best trial once and record its final test metrics
- store trial logs under `optuna_runs/<study_name>/`
- write the best result to `best_trial.json`

See:

- [HYPERPARAMETER_TUNING_GUIDE.md](./HYPERPARAMETER_TUNING_GUIDE.md)
- [TUNE_SCRIPT_README.md](./TUNE_SCRIPT_README.md)

---

## GPU Notes

The current code supports:

- single-GPU execution through `python main.py ...`
- single-node multi-GPU execution through `torchrun`

Recommended DDP launch pattern:

```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<NUM_GPUS> main.py ...
```

So:

- `CUDA_VISIBLE_DEVICES=6 python main.py ...`
  means only physical GPU 6 is visible to the process
- inside the process it is still treated as one GPU, `cuda:0`
- `NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 main.py ...`
  launches 4 DDP worker processes, one per visible GPU

Text embedding memory usage is mainly affected by:

- `mpnet` itself
- materializing large text columns

This repo already includes a fix for one text embedding materialization OOM issue: embedding chunks are no longer accumulated on GPU before `torch.cat`.

---

## FAQ

### Why do some commands from the old README fail?

Because the old README included arguments that are no longer present in the current codebase, especially:

- `--context`
- `--context_table`

These are not defined in the current `main.py`.

### How do I confirm the currently supported CLI arguments?

Run:

```bash
python main.py --help
```

### How do I do a quick smoke test?

Use:

- `--debug`
- fewer `train_steps`
- small `batch_size` and `val_size`

Example:

```bash
python main.py \
  --dataset=rel-amazon \
  --task=user-churn \
  --train_steps=128 \
  --batch_size=1 \
  --val_size=1 \
  --lr=0.001 \
  --wd=0.0015 \
  --dropout=0.4 \
  --val_steps=1000 \
  --eval_steps=64 \
  --temporal_strategy=last \
  --text_embedder=mpnet \
  --llm_frozen \
  --loss_class_weight 0.6 0.4 \
  --debug
```

---

## Additional Docs

If you want to understand the codebase before modifying it, start with:

- [MAIN_PIPELINE_EXPLANATION.md](./MAIN_PIPELINE_EXPLANATION.md)
- [HYPERPARAMETER_TUNING_GUIDE.md](./HYPERPARAMETER_TUNING_GUIDE.md)
- [TUNE_SCRIPT_README.md](./TUNE_SCRIPT_README.md)
