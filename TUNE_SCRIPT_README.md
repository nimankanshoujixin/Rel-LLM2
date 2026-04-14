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
- `basis_root=artifacts/basis`
- `disable_basis_cls_head=False`
- `basis_tau=0.07`
- `basis_lambda_bce=1.0`

This means the default tuning scenario is:

- LLM mode
- default 1B model
- frozen LLM
- no `output_mlp`
- no pretraining
- no `wandb` logging during tuning
- impl-a basis alignment enabled

These are also fixed unless you override them manually:

- `train_steps`
- `val_steps`
- `eval_steps`
- `test_steps`
- `early_stop_patience`
- `early_stop_metric_delta`
- `early_stop_loss_delta`
- `val_size`
- `seed`
- `dataset`
- `task`
- `cache_dir`
- `text_embedder_path`
- `basis_root`
- `basis_artifact`
- `disable_basis_cls_head`
- `basis_tau`
- `basis_tau_res` when basis tuning is disabled
- `basis_topk` when basis tuning is disabled
- `basis_residual_alpha` when basis tuning is disabled
- `basis_lambda_bce`
- `basis_lambda_ctr` when basis tuning is disabled
- `basis_lambda_mgn` when basis tuning is disabled
- `basis_margin` when basis tuning is disabled

Code-internal hyperparameters are not tuned by this script, including:

- `TextEmbedderConfig(batch_size=256)`
- LoRA settings
- `gamma=2.0`
- `mask_ratio=0.5`
- projector hidden dimension
- scheduler settings
- optimizer betas

Semantics:

- `train_steps` is the total finetuning budget passed to `main.py`
- `val_steps` controls how often intermediate evaluation is triggered
- `eval_steps` caps how many validation loader batches each intermediate evaluation consumes
- `test_steps` caps how many test loader batches the final best-trial test consumes
- early stopping triggers only when validation metric and windowed training loss both stop improving

During tuning:

- intermediate trial runs do not execute the test split
- the test split is run only once, after the best trial is selected
- `--reset-study` removes both the existing Optuna study record and `optuna_runs/<study_name>/` before starting
- impl-a runs require a prebuilt basis artifact at `artifacts/basis/<dataset>/basis.pt` unless you pass `--basis-artifact`

---

## Arguments you should decide before running

Most commonly:

- `--dataset`
- `--task`
- `--gpu-id`
- `--n-trials`
- `--train-steps`
- `--reset-study` when you want a fresh run with the same study name

For DDP tuning:

- `--nproc-per-node`
- `--master-port`
- `--nccl-p2p-disable`

Other useful arguments:

- `--study-name`
- `--reset-study`
- `--storage`
- `--timeout`
- `--val-steps`
- `--eval-steps`
- `--test-steps`
- `--early-stop-patience`
- `--early-stop-metric-delta`
- `--early-stop-loss-delta`
- `--val-size`
- `--cache-dir`
- `--text-embedder-path`
- `--basis-root`
- `--basis-artifact`
- `--disable-basis-cls-head / --no-disable-basis-cls-head`
- `--basis-tau`
- `--basis-tau-res`
- `--basis-topk`
- `--basis-residual-alpha`
- `--basis-lambda-bce`
- `--basis-lambda-ctr`
- `--basis-lambda-mgn`
- `--basis-margin`
- `--tune-basis-hparams / --no-tune-basis-hparams`
- `--basis-topk-choices`
- `--basis-tau-res-choices`
- `--basis-margin-choices`
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

### Impl-a basis alignment

When `--disable-basis-cls-head` is not set, the script also tunes a small subset of impl-a basis hyperparameters by default:

- `basis_tau_res`
  - default choices: `{0.03, 0.07, 0.15}`

- `basis_topk`
  - default choices: `{4, 8, 16}`

- `basis_residual_alpha`
  - range: `0.02 ~ 0.5`
  - sampling: log-uniform

- `basis_lambda_ctr`
  - range: `1e-3 ~ 1.0`
  - sampling: log-uniform

- `basis_lambda_mgn`
  - range: `1e-3 ~ 1.0`
  - sampling: log-uniform

- `basis_margin`
  - default choices: `{0.1, 0.2, 0.4}`

These remain fixed unless you override them manually:

- `basis_tau`
- `basis_lambda_bce`

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
basis_tau_res:     {0.03, 0.07, 0.15}
basis_topk:        {4, 8, 16}
basis_residual_alpha: 0.02 ~ 0.5   (log)
basis_lambda_ctr:  1e-3 ~ 1.0      (log)
basis_lambda_mgn:  1e-3 ~ 1.0      (log)
basis_margin:      {0.1, 0.2, 0.4}
```

---

## Single-GPU example

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --basis-root artifacts/basis \
  --gpu-id 6 \
  --n-trials 30 \
  --train-steps 4096 \
  --val-steps 512 \
  --eval-steps 128 \
  --test-steps 1024 \
  --study-name amazon_user_churn_llama1b
```

## DDP example

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --basis-root artifacts/basis \
  --gpu-id 4,5,6,7 \
  --nproc-per-node 4 \
  --master-port 29501 \
  --nccl-p2p-disable 1 \
  --n-trials 30 \
  --train-steps 4096 \
  --val-steps 512 \
  --eval-steps 128 \
  --test-steps 1024 \
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

- final best-trial test log:
  - `optuna_runs/<study_name>/best_trial_test.log`

- best trial summary:
  - `optuna_runs/<study_name>/best_trial.json`

- Optuna study database:
  - by default: `sqlite:///optuna_rel_llm.db`
