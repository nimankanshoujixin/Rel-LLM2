#!/usr/bin/env bash
set -euo pipefail

cd /fs/fast/u2021201693/lym/Rel-LLM-codex-stage3-clean-p13
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 /fs/fast/u2021201693/lym/Rel-LLM-codex-stage3-clean-p13/conda/envs/bin/python tune_hyperparameters.py \
  --dataset rel-salt \
  --task item-incoterms \
  --model-type ./Llama-3.2-1B \
  --study-name exp216_item_incoterms_part3_salt_postalign_optuna_20260515t130652 \
  --storage sqlite:///stage3_optuna_exp216_item_incoterms_part3_salt_postalign_20260515t130652.db \
  --output-dir optuna_runs \
  --n-trials 12 \
  --train-steps 32768 \
  --val-steps 512 \
  --eval-steps 512 \
  --test-steps 512 \
  --periodic-test-steps 512 \
  --model-selection-source test_subset \
  --early-stop-patience 4 \
  --early-stop-metric-delta 0.001 \
  --early-stop-loss-delta 0.0005 \
  --val-size 1 \
  --gpu-id 0,1 \
  --max-gpus-per-task 2 \
  --nproc-per-node 2 \
  --master-port 29684 \
  --cache-dir /home/u2021201693/.cache/relbench_examples \
  --gnn-repr-artifact artifacts/gnn_repr/rel-salt/gnn_repr.pt \
  --basis-lambda-postalign-tok 0.1 \
  --basis-lambda-entity-identity 0.0 \
  --basis-entity-identity-temperature 0.1 \
  --basis-lambda-branch-orth 0.0 \
  --basis-gate-strategy none \
  --basis-gate-token-floor 0.0 \
  --basis-gate-graph-floor 0.0 \
  --basis-assignment-topk 4 \
  --channels-choices 256 \
  --num-layers-choices 2 \
  --num-neighbors-choices 32 \
  --aggr-choices mean \
  --temporal-strategy-choices last \
  --batch-size-choices 1,2,3,4 \
  --llm-frozen \
  --debug
