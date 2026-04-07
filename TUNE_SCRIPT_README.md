# `tune_hyperparameters.py` Usage

This document only describes the tuning wrapper [tune_hyperparameters.py](/G:/RelLLM-2/Rel-LLM/tune_hyperparameters.py).

---

## Fixed defaults

By default, the script fixes these training-mode choices:

- `model_type=meta-llama/Llama-3.2-1B`
- `llm_frozen=True`
- `output_mlp=False`
- `pretrain=False`
- `debug=True`

This means the default tuning scenario is:

- LLM mode
- default 1B model
- frozen LLM
- no `output_mlp`
- no pretraining
- no `wandb` logging during tuning

These are also fixed unless you override them manually:

- `epochs`
- `val_steps`
- `val_size`
- `seed`
- `dataset`
- `task`
- `cache_dir`
- `text_embedder_path`

Code-internal hyperparameters are not tuned by this script, including:

- `TextEmbedderConfig(batch_size=256)`
- LoRA settings
- `gamma=2.0`
- `mask_ratio=0.5`
- projector hidden dimension
- scheduler settings
- optimizer betas

---

## Arguments you should decide before running

Most commonly:

- `--dataset`
- `--task`
- `--gpu-id`
- `--n-trials`
- `--epochs`

For DDP tuning:

- `--nproc-per-node`
- `--master-port`
- `--nccl-p2p-disable`

Other useful arguments:

- `--study-name`
- `--storage`
- `--timeout`
- `--val-steps`
- `--val-size`
- `--cache-dir`
- `--text-embedder-path`
- `--python-executable`
- `--output-dir`

You can also override the fixed training mode if needed:

- `--model-type`
- `--text-embedder`
- `--llm-frozen / --no-llm-frozen`
- `--output-mlp / --no-output-mlp`
- `--pretrain / --no-pretrain`
- `--debug / --no-debug`

---

## Hyperparameters that are automatically tuned

### Optimization

- `lr`
  - range: `1e-5 ~ 3e-3`
  - sampling: log-uniform

- `wd`
  - range: `1e-6 ~ 1e-2`
  - sampling: log-uniform

- `dropout`
  - range: `0.0 ~ 0.5`

### Graph/model structure

- `channels`
  - default choices: `{64, 128, 256}`

- `num_layers`
  - default choices: `{1, 2, 3}`

- `num_neighbors`
  - default choices: `{16, 32, 64, 128}`

- `aggr`
  - default choices: `{sum, mean}`

- `temporal_strategy`
  - default choices: `{uniform, last}`

### Training batch size

- `batch_size`
  - default choices: `{1, 2, 4}`

Note:

- this is the per-process batch size
- in DDP, global batch size is `batch_size * nproc_per_node`

### Class imbalance

The script currently tunes only the positive-class weight:

- `w_pos`
  - range: `0.5 ~ 3.0`

The negative-class weight is fixed to:

- `w_neg = 1.0`

So the final command uses:

```bash
--loss_class_weight 1.0 <w_pos>
```

---

## Default search space summary

```text
lr:                1e-5 ~ 3e-3      (log)
wd:                1e-6 ~ 1e-2      (log)
dropout:           0.0 ~ 0.5
channels:          {64, 128, 256}
num_layers:        {1, 2, 3}
num_neighbors:     {16, 32, 64, 128}
aggr:              {sum, mean}
temporal_strategy: {uniform, last}
batch_size:        {1, 2, 4}
w_pos:             0.5 ~ 3.0
```

---

## Single-GPU example

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --gpu-id 6 \
  --n-trials 30 \
  --epochs 5 \
  --study-name amazon_user_churn_llama1b
```

## DDP example

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --gpu-id 4,5,6,7 \
  --nproc-per-node 4 \
  --master-port 29501 \
  --nccl-p2p-disable 1 \
  --n-trials 30 \
  --epochs 5 \
  --study-name amazon_user_churn_llama1b_ddp
```

When `--nproc-per-node > 1`, the script launches:

```bash
python -m torch.distributed.run --nproc_per_node=<N> --master_port=<PORT> main.py ...
```

and exports:

```bash
NCCL_P2P_DISABLE=1
```

for every trial by default.

---

## Outputs

The script writes:

- per-trial logs:
  - `optuna_runs/<study_name>/trial_XXXX.log`

- best trial summary:
  - `optuna_runs/<study_name>/best_trial.json`

- Optuna study database:
  - by default: `sqlite:///optuna_rel_llm.db`

