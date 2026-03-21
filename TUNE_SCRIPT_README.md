# `tune_hyperparameters.py` 使用说明

这份文档只介绍调参脚本 [tune_hyperparameters.py](/G:/RelLLM-2/Rel-LLM/tune_hyperparameters.py) 的使用方式。

---

## 1. 固定参数

脚本默认固定了以下训练范式参数，不会进入自动搜索空间。

### 1.1 默认固定的训练范式

- `model_type=meta-llama/Llama-3.2-1B`
- `llm_frozen=True`
- `output_mlp=False`
- `pretrain=False`
- `debug=True`

这意味着默认调参场景是：

- 使用 `LLM 模式`
- 使用默认 `1B` 模型
- 冻结 LLM 参数
- 不启用 `output_mlp`
- 不启用预训练
- 调参时关闭 wandb

### 1.2 训练中固定、不做搜索的参数

下面这些参数会直接传给 `main.py`，但默认不参与搜索：

- `epochs`
- `val_steps`
- `val_size`
- `seed`
- `dataset`
- `task`
- `cache_dir`
- `text_embedder_path`

其中：

- `epochs` 默认是 `5`
- `val_steps` 默认是 `500`
- `val_size` 默认是 `1`
- `seed` 默认是 `42`

### 1.3 代码内固定但当前不调的内部参数

脚本不会修改这些代码内超参数：

- `TextEmbedderConfig(batch_size=256)`
- LoRA 参数 `r=8, alpha=16, dropout=0.05`
- `gamma=2.0`
- `mask_ratio=0.5`
- `projector hidden dim = 1024`
- `scheduler factor=0.8, patience=100`
- `optimizer betas=(0.9, 0.95)`

---

## 2. 需要从命令行提前指定的参数

这些参数不是自动搜索得到的，而是你在启动调参脚本时就要决定的。

### 2.1 最重要的运行参数

- `--dataset`
- `--task`
- `--gpu-id`
- `--n-trials`
- `--epochs`

其中最常用的是：

- `--dataset`
  - 要调哪个数据集
- `--task`
  - 要调哪个任务
- `--gpu-id`
  - 用哪张 GPU 跑 trial
- `--n-trials`
  - 总共跑多少组参数
- `--epochs`
  - 每个 trial 跑多少个 epoch

### 2.2 常用可选参数

- `--study-name`
  - Optuna study 名称
- `--storage`
  - Optuna 数据库存储地址
- `--timeout`
  - 调参总时长限制
- `--val-steps`
  - 多少 step 做一次验证
- `--val-size`
  - 验证 batch size
- `--cache-dir`
  - 数据缓存目录
- `--text-embedder-path`
  - 文本嵌入模型缓存目录
- `--python-executable`
  - 用哪个 Python 解释器启动 `main.py`
- `--output-dir`
  - trial 日志和 best trial 输出目录

### 2.3 可手动改动的训练范式参数

虽然这些不是自动搜索参数，但你可以在命令行显式指定是否启用：

- `--model-type`
- `--text-embedder`
- `--llm-frozen / --no-llm-frozen`
- `--output-mlp / --no-output-mlp`
- `--pretrain / --no-pretrain`
- `--debug / --no-debug`

默认推荐保持不动。

---

## 3. 会被自动调优的参数

脚本当前会自动搜索下面这些参数。

### 3.1 学习率与正则

- `lr`
  - 搜索范围：`1e-5 ~ 3e-3`
  - 方式：`log-uniform`

- `wd`
  - 搜索范围：`1e-6 ~ 1e-2`
  - 方式：`log-uniform`

- `dropout`
  - 搜索范围：`0.0 ~ 0.5`
  - 方式：连续均匀采样

### 3.2 图模型结构参数

- `channels`
  - 默认候选：`{64, 128, 256}`

- `num_layers`
  - 默认候选：`{1, 2, 3}`

- `num_neighbors`
  - 默认候选：`{16, 32, 64, 128}`

- `aggr`
  - 默认候选：`{sum, mean}`

- `temporal_strategy`
  - 默认候选：`{uniform, last}`

### 3.3 训练批大小

- `batch_size`
  - 默认候选：`{1, 2, 4}`

注意：

- 这里是训练 batch size
- 验证 batch size 不调，固定由 `--val-size` 指定

### 3.4 类别不平衡参数

脚本当前只搜索正类权重：

- `w_pos`
  - 搜索范围：`0.5 ~ 3.0`

然后固定：

- `w_neg = 1.0`

最终传给 `main.py` 的形式是：

```bash
--loss_class_weight 1.0 <w_pos>
```

---

## 4. 当前默认搜索空间汇总

当前脚本默认搜索的是：

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

## 5. 一个最常用的启动方式

如果你要调 `rel-amazon / user-churn`，最常见的启动方式是：

```bash
python tune_hyperparameters.py \
  --dataset rel-amazon \
  --task user-churn \
  --gpu-id 6 \
  --n-trials 30 \
  --epochs 5 \
  --study-name amazon_user_churn_llama1b
```

---

## 6. 输出内容

脚本运行后会产出：

- 每个 trial 的日志：
  - `optuna_runs/<study_name>/trial_XXXX.log`

- 最优 trial 摘要：
  - `optuna_runs/<study_name>/best_trial.json`

- Optuna study 数据：
  - 默认写到 `sqlite:///optuna_rel_llm.db`

