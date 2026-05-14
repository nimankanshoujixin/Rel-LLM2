# Stage 3 Baseline Commands

These are the canonical Stage 3 baseline commands for quick screening on the
current `main` model line.

## Screening Protocol

- Use fixed hyperparameters.
- Run `main.py` directly.
- Prefer single-GPU screening commands for routine stage3 iteration.
- Keep `eval_steps=128`.
- Use a test subset during routine iteration:
  - default `test_steps=128`
- Reserve `test_steps=-1` for final confirmation only.

## Representative Tasks

1. `rel-amazon / user-churn`
2. `rel-amazon / user-ltv`
3. `rel-salt / item-incoterms`

## Baseline A: `rel-amazon / user-churn`

Reference source:
- `optuna_runs/amazon_user_churn_llama1b_ddp_implb2/best_trial.json`

Full-test reference:
- `roc_auc = 0.6918065868366201`
- `average_precision = 0.7420686063230703`
- `accuracy = 0.6743850974039814`
- `f1 = 0.7811745023481261`

Reference full-capacity command:

```bash
/fs/fast/u2021201693/lym/Rel-LLM/conda/envs/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 main.py \
  --dataset=rel-amazon \
  --task=user-churn \
  --model_type=./Llama-3.2-1B \
  --text_embedder=mpnet \
  --train_steps=32768 \
  --val_steps=512 \
  --eval_steps=128 \
  --test_steps=128 \
  --early_stop_patience=4 \
  --early_stop_metric_delta=0.001 \
  --early_stop_loss_delta=0.0005 \
  --val_size=1 \
  --channels=256 \
  --num_layers=3 \
  --num_neighbors=16 \
  --aggr=sum \
  --temporal_strategy=last \
  --dropout=0.12714124285290368 \
  --batch_size=4 \
  --lr=2.1736088071412522e-05 \
  --wd=7.914020107853111e-05 \
  --seed=42 \
  --basis_root=artifacts/basis \
  --basis_tau=0.0618424925257609 \
  --basis_residual_alpha=0.32065307158713396 \
  --basis_graph_alpha=0.16855552643022662 \
  --basis_lambda_tok=0.9064393078452517 \
  --basis_lambda_g=2.8949808533759014 \
  --basis_lambda_sharp=0.004251108866117479 \
  --loss_class_weight 1.0 1.0231539079677257 \
  --llm_frozen \
  --debug
```

## Baseline B: `rel-amazon / user-ltv`

Reference source:
- `optuna_runs/amazon_user_ltv_llama1b_ddp_implb2/best_trial.json`

Full-test reference:
- `r2 = 0.10848423938250884`
- `mae = 16.671180345060154`
- `rmse = 52.348502253874656`

Reference full-capacity command:

```bash
/fs/fast/u2021201693/lym/Rel-LLM/conda/envs/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 main.py \
  --dataset=rel-amazon \
  --task=user-ltv \
  --model_type=./Llama-3.2-1B \
  --text_embedder=mpnet \
  --train_steps=32768 \
  --val_steps=512 \
  --eval_steps=128 \
  --test_steps=128 \
  --early_stop_patience=4 \
  --early_stop_metric_delta=0.001 \
  --early_stop_loss_delta=0.0005 \
  --val_size=1 \
  --channels=256 \
  --num_layers=3 \
  --num_neighbors=128 \
  --aggr=mean \
  --temporal_strategy=last \
  --dropout=0.2990078235121656 \
  --batch_size=2 \
  --lr=0.0008413563569086949 \
  --wd=0.0019708370843829455 \
  --seed=42 \
  --basis_root=artifacts/basis \
  --basis_tau=0.08576768686631123 \
  --basis_residual_alpha=0.05504297486907467 \
  --basis_graph_alpha=0.08138871681852786 \
  --basis_lambda_tok=2.0279409466586937 \
  --basis_lambda_g=0.211313078616332 \
  --basis_lambda_sharp=0.0874326087705442 \
  --loss_class_weight 1.0 2.3432256610727644 \
  --debug
```

## Baseline C: `rel-salt / item-incoterms`

Reference source:
- `optuna_runs/salt_item-incoterms_llama1b_ddp_implb2/best_trial.json`

Full-test reference:
- `mrr = 0.7043105782857789`
- `accuracy = 0.580488289249941`
- `macro_f1 = 0.06121402408760499`
- `micro_f1 = 0.580488289249941`

Reference full-capacity command:

```bash
/fs/fast/u2021201693/lym/Rel-LLM/conda/envs/bin/python main.py \
  --dataset=rel-salt \
  --task=item-incoterms \
  --model_type=./Llama-3.2-1B \
  --text_embedder=mpnet \
  --train_steps=32768 \
  --val_steps=512 \
  --eval_steps=128 \
  --test_steps=128 \
  --early_stop_patience=4 \
  --early_stop_metric_delta=0.001 \
  --early_stop_loss_delta=0.0005 \
  --val_size=1 \
  --channels=256 \
  --num_layers=2 \
  --num_neighbors=64 \
  --aggr=mean \
  --temporal_strategy=last \
  --dropout=0.09186134874406687 \
  --batch_size=4 \
  --lr=0.00015309578868476833 \
  --wd=0.0003299095764385189 \
  --seed=42 \
  --basis_root=artifacts/basis \
  --basis_tau=0.04039653092727967 \
  --basis_residual_alpha=0.40679327531075476 \
  --basis_graph_alpha=0.20440872218323222 \
  --basis_lambda_tok=0.8182114653708158 \
  --basis_lambda_g=0.58449427417046 \
  --basis_lambda_sharp=0.0995956756696025 \
  --loss_class_weight 1.0 1.6590937102322016 \
  --llm_frozen \
  --debug
```

## Operational Note

On 2026-04-30, a direct reuse of the `4 GPU torchrun` baseline command for
`rel-amazon / user-churn` in stage3 screening caused a startup hang with empty logs and
`D`-state worker processes on the remote server.

Current stage3 default:

- keep the above commands as the full-capacity reference
- derive single-GPU screening variants from them for routine optimization
- only return to DDP after a candidate change is already validated
