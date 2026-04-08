# Rel-LLM 超参数清单与自动调优方案

## 说明范围

这份文档只覆盖 `main.py` 作为训练入口时会实际使用到的超参数。

- 命令行参数：全部列出，并区分是否建议自动调优
- 代码内固定超参数：全部列出，但仅记录，不纳入自动调优
- 自动调优方案：默认针对 `LLM 模式 + 1B 模型 + 冻结 LLM` 这一类实验

---

## 1. 超参数总览

### 1.1 命令行可配置超参数

这些参数来自 [main.py](/G:/RelLLM-2/Rel-LLM/main.py)。

| 参数 | 默认值 | 类型 | 作用 | 是否建议自动调优 |
| --- | --- | --- | --- | --- |
| `dataset` | `rel-stack` | str | 数据集名称 | 否，实验上下文固定 |
| `task` | `user-engagement` | str | 任务名称 | 否，实验上下文固定 |
| `cache_dir` | `~/.cache/relbench_examples` | str | 数据/图缓存目录 | 否 |
| `debug` | `False` | flag | 调试模式，通常关闭日志 | 否 |
| `channels` | `128` | int | GNN/表格编码统一隐藏维度 | 是 |
| `aggr` | `sum` | str | GraphSAGE 邻居聚合方式 | 是 |
| `num_layers` | `2` | int | GNN 层数 | 是 |
| `num_neighbors` | `128` | int | 采样邻居数基值，实际会按层衰减 | 是 |
| `temporal_strategy` | `uniform` | str | 时间邻居采样策略 | 是 |
| `text_embedder` | `glove` | str | 表中原始文本列嵌入器 | 是，但建议粗粒度调 |
| `text_embedder_path` | `./cache` | str | 文本嵌入器缓存目录 | 否 |
| `model_type` | `meta-llama/Llama-3.2-1B` | str | LLM 后端模型 | 是，但建议分阶段调 |
| `llm_frozen` | `False` | flag | 是否冻结 LLM 参数 | 是，但建议实验级切换 |
| `output_mlp` | `False` | flag | 是否用最后隐状态 + MLP，而不是文本生成 | 是，但建议实验级切换 |
| `dropout` | `0.1` | float | 输出头 / projector dropout | 是 |
| `num_demo` | `0` | int | few-shot demo 数量 | 是 |
| `max_new_tokens` | `1` | int | 推理时最多生成 token 数 | 否，分类任务一般固定 |
| `loss_class_weight` | `None` | list[float] | 类别权重，二分类时影响 focal/损失加权 | 是 |
| `train_steps` | `32768` | int | 微调总训练 step 数 | 否，通常作为预算固定 |
| `pretrain` | `False` | flag | 是否先做预训练 | 是，但建议实验级切换 |
| `pretrain_epochs` | `200` | int | 预训练轮数 | 否，通常作为预算固定 |
| `val_steps` | `1000` | int | 每多少 step 做一次验证 | 是，但一般只小范围调 |
| `eval_steps` | `1024` | int | 每次验证最多跑多少个 batch step | 否，通常作为预算固定 |
| `test_steps` | `4096` | int | 最终测试最多跑多少个 batch step | 否，通常作为预算固定 |
| `batch_size` | `256` | int | 训练 batch size | 是 |
| `val_size` | `None` | int/None | 验证/测试 batch size | 否，一般按显存上限设置 |
| `num_workers` | `0` | int | DataLoader worker 数 | 否，偏工程参数 |
| `lr` | `1e-4` | float | 学习率 | 是，核心参数 |
| `wd` | `0.05` | float | weight decay | 是，核心参数 |
| `seed` | `42` | int | 随机种子 | 否，调优时应固定 |

### 1.2 由命令行间接决定的派生超参数

这些参数不是直接从命令行读取，但会由命令行参数推导出来。

| 参数 | 来源 | 当前逻辑 | 是否建议自动调优 |
| --- | --- | --- | --- |
| `num_neighbors per layer` | `num_neighbors + num_layers` | `[num_neighbors / 2^i for i in range(num_layers)]` | 否，已被 `num_neighbors`/`num_layers` 间接覆盖 |
| `optimizer type` | `wd` | `wd != 0` 用 `AdamW`，否则 `Adam` | 否 |
| `scheduler mode` | `task_info(task)` | 分类任务通常 `max`，回归任务通常 `min` | 否 |
| `out_channels` | `task_info(task)` | 二分类/回归一般为 `1` | 否 |
| `loss_fn` | `task_info(task)` | 分类 BCE，回归 L1，多标签 BCE | 否 |
| `tune_metric` | `task_info(task)` | 例如二分类常用 `roc_auc` | 否 |

---

## 2. 代码内固定超参数

这些参数没有暴露到命令行。当前阶段仅列出，不建议纳入自动调优。

### 2.1 `main.py` 中固定的内部参数

| 参数 | 固定值 | 位置 | 说明 |
| --- | --- | --- | --- |
| 文本物化 batch size | `256` | [main.py:130](/G:/RelLLM-2/Rel-LLM/main.py:130) | `TextEmbedderConfig(batch_size=256)` |
| `pin_memory` | `True` | [main.py:141](/G:/RelLLM-2/Rel-LLM/main.py:141) | `NeighborLoader` 固定开启 |
| `persistent_workers` | `num_workers > 0` | [main.py:141](/G:/RelLLM-2/Rel-LLM/main.py:141) | worker 持久化策略 |
| AdamW `betas` | `(0.9, 0.95)` | [main.py:155](/G:/RelLLM-2/Rel-LLM/main.py:155) | 优化器动量参数 |
| Scheduler `factor` | `0.8` | [main.py:159](/G:/RelLLM-2/Rel-LLM/main.py:159) | 学习率衰减倍率 |
| Scheduler `patience` | `100` | [main.py:159](/G:/RelLLM-2/Rel-LLM/main.py:159) | 学习率衰减耐心值 |
| 设备选择 | `cuda:0` | [main.py:97](/G:/RelLLM-2/Rel-LLM/main.py:97) | 当前为单卡逻辑 |
| CUDA 线程数 | `1` | [main.py:99](/G:/RelLLM-2/Rel-LLM/main.py:99) | CUDA 场景下固定设置 |

### 2.2 `model.py` 中固定的内部参数

| 参数 | 固定值 | 位置 | 说明 |
| --- | --- | --- | --- |
| `norm` | `batch_norm` | [model.py:35](/G:/RelLLM-2/Rel-LLM/model.py:35) | 输出 MLP/头部归一化类型 |
| `output_probs` | `True` | [model.py:36](/G:/RelLLM-2/Rel-LLM/model.py:36) | 二分类推理时输出概率 |
| `gamma` | `2.0` | [model.py:37](/G:/RelLLM-2/Rel-LLM/model.py:37) | focal loss gamma |
| `alpha` 默认值 | `[1.0, 1.0]` | [model.py:37](/G:/RelLLM-2/Rel-LLM/model.py:37) | 类别权重默认值 |
| `mask_ratio` | `0.5` | [model.py:37](/G:/RelLLM-2/Rel-LLM/model.py:37) | 预训练 mask 比例 |
| `pretrain_random_table` | `False` | [model.py:37](/G:/RelLLM-2/Rel-LLM/model.py:37) | 预训练是否随机表 |
| `pretrain_mask_cell` | `True` | [model.py:37](/G:/RelLLM-2/Rel-LLM/model.py:37) | 预训练是否 mask 单元格 |
| tokenizer `use_fast` | `False` | [model.py:68](/G:/RelLLM-2/Rel-LLM/model.py:68) | HuggingFace tokenizer 设置 |
| tokenizer `padding_side` | `left` | [model.py:68](/G:/RelLLM-2/Rel-LLM/model.py:68) | 左 padding |
| LLM `torch_dtype` | `float16` | [model.py:72](/G:/RelLLM-2/Rel-LLM/model.py:72) | LLM 权重精度 |
| LLM `device_map` | `{"": 0}` | [model.py:72](/G:/RelLLM-2/Rel-LLM/model.py:72) | 强制单卡 |
| LoRA `r` | `8` | [model.py:82](/G:/RelLLM-2/Rel-LLM/model.py:82) | 仅 `llm_frozen=False` 时生效 |
| LoRA `alpha` | `16` | [model.py:83](/G:/RelLLM-2/Rel-LLM/model.py:83) | 同上 |
| LoRA `dropout` | `0.05` | [model.py:84](/G:/RelLLM-2/Rel-LLM/model.py:84) | 同上 |
| LoRA `target_modules` | `["q_proj", "v_proj"]` | [model.py:85](/G:/RelLLM-2/Rel-LLM/model.py:85) | 同上 |
| Qwen `out_dim` | `3584` | [model.py:92](/G:/RelLLM-2/Rel-LLM/model.py:92) | projector 输出维度 |
| Llama-3.2-1B `out_dim` | `2048` | [model.py:94](/G:/RelLLM-2/Rel-LLM/model.py:94) | projector 输出维度 |
| projector hidden dim | `1024` | [model.py:95](/G:/RelLLM-2/Rel-LLM/model.py:95) | 两层 projector 中间维度 |
| autocast dtype | `bfloat16` | [model.py:138](/G:/RelLLM-2/Rel-LLM/model.py:138) | GPU autocast 精度 |
| 邻居上下文 hop 数 | `1` | [model.py:348](/G:/RelLLM-2/Rel-LLM/model.py:348) | `forward()` 里固定传入 |

### 2.3 `relbench/modeling/nn.py` 中固定的内部参数

| 参数 | 固定值 | 位置 | 说明 |
| --- | --- | --- | --- |
| 表格编码器类型 | `ResNet` | [relbench/modeling/nn.py:35](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:35) | 每个表默认都用这个编码器 |
| 表格编码器 channels | `128` | [relbench/modeling/nn.py:35](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:35) | `torch_frame_model_kwargs` 内固定 |
| 表格编码器 num_layers | `4` | [relbench/modeling/nn.py:35](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:35) | `torch_frame_model_kwargs` 内固定 |
| HeteroGraphSAGE 默认 `aggr` | `mean` | [relbench/modeling/nn.py:84](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:84) | 若外部未覆盖则生效 |
| HeteroGraphSAGE 默认层数 | `2` | [relbench/modeling/nn.py:84](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:84) | 若外部未覆盖则生效 |
| HeteroConv 聚合 | `sum` | [relbench/modeling/nn.py:88](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:88) | 多边类型消息汇总方式 |
| 节点归一化 | `LayerNorm(mode="node")` | [relbench/modeling/nn.py:95](/G:/RelLLM-2/Rel-LLM/relbench/modeling/nn.py:95) | 每层 GNN 后固定使用 |

### 2.4 `torch_frame` / 文本嵌入中的固定内部参数

| 参数 | 固定值 | 位置 | 说明 |
| --- | --- | --- | --- |
| `mpnet` 实际模型名 | `all-mpnet-base-v2` | [text_embedder.py:13](/G:/RelLLM-2/Rel-LLM/text_embedder.py:13) | 文本嵌入器别名映射 |
| `glove` 实际模型名 | `average_word_embeddings_glove.6B.300d` | [text_embedder.py:15](/G:/RelLLM-2/Rel-LLM/text_embedder.py:15) | 文本嵌入器别名映射 |
| `ResNet` normalization | `layer_norm` | [torch_frame/nn/models/resnet.py:116](/G:/RelLLM-2/Rel-LLM/torch_frame/nn/models/resnet.py:116) | 表格编码器内部默认归一化 |
| `ResNet` dropout_prob | `0.2` | [torch_frame/nn/models/resnet.py:116](/G:/RelLLM-2/Rel-LLM/torch_frame/nn/models/resnet.py:116) | 表格编码器内部默认 dropout |

### 2.5 任务定义中的固定任务级参数

以你当前关心的 `rel-amazon` 为例，任务本身还包含固定定义：

| 参数 | 固定值 | 位置 | 说明 |
| --- | --- | --- | --- |
| `UserChurnTask.timedelta` | `365 // 4 days` | [relbench/tasks/amazon.py:28](/G:/RelLLM-2/Rel-LLM/relbench/tasks/amazon.py:28) | 预测时间窗口 |
| `UserChurnTask.metrics` | `average_precision, accuracy, f1, roc_auc` | [relbench/tasks/amazon.py:29](/G:/RelLLM-2/Rel-LLM/relbench/tasks/amazon.py:29) | 验证/测试指标 |

这些参数是任务定义的一部分，不建议和训练超参数混在一起调。

---

## 3. 建议纳入自动调优的参数集合

如果你要做自动化调优，不建议一开始把所有命令行参数都扔进搜索空间。

### 3.1 第一优先级：强烈建议调

| 参数 | 原因 |
| --- | --- |
| `lr` | 对收敛速度和最终指标影响最大 |
| `wd` | 决定正则强度，LLM/GNN 混合模型很敏感 |
| `dropout` | 影响过拟合，尤其是小 batch 场景 |
| `num_neighbors` | 直接影响图上下文量和显存 |
| `num_layers` | 决定图信息传播深度 |
| `channels` | 决定图表征容量 |
| `loss_class_weight` | 类别不平衡任务下影响非常大 |

### 3.2 第二优先级：可调，但建议后置

| 参数 | 原因 |
| --- | --- |
| `aggr` | `sum/mean` 可能影响稳定性 |
| `temporal_strategy` | 时序任务上可能有收益 |
| `batch_size` | 受显存约束，收益不如 `lr/wd` 稳定 |
| `num_demo` | few-shot 可能有效，但成本高 |
| `text_embedder` | `glove/mpnet` 差异大，但训练成本也差很多 |
| `output_mlp` | 这是模型范式切换，不建议和常规调参混在一个搜索里 |
| `llm_frozen` | 这是训练范式切换，不建议和常规调参混在一个搜索里 |
| `pretrain` | 这是训练流程切换，不建议和常规调参混在一个搜索里 |

### 3.3 不建议放入自动调优

| 参数 | 原因 |
| --- | --- |
| `dataset`, `task` | 实验定义，不是超参数 |
| `cache_dir`, `text_embedder_path` | 工程路径参数 |
| `debug` | 调试开关 |
| `train_steps`, `pretrain_epochs`, `eval_steps`, `test_steps` | 更像预算，不是模型行为参数 |
| `val_size`, `num_workers` | 工程性能参数 |
| `seed` | 需要固定，避免试验噪声 |
| `max_new_tokens` | 二分类一般固定为 `1` |

---

## 4. 自动化调优方法

## 4.1 推荐方法

推荐用 **Optuna + TPE + Pruner**，不要从网格搜索开始。

原因：

- 这个项目单次实验成本高
- 搜索空间里同时有离散参数和连续参数
- 图模型和 LLM 混合训练噪声较大
- 需要尽早停掉明显不好的 trial

推荐组合：

- Sampler: `TPESampler`
- Pruner: `MedianPruner` 或 `SuccessiveHalvingPruner`

---

## 4.2 调优目标

目标函数直接用当前任务的验证集主指标。

对于 `rel-amazon / user-churn`：

- 主指标：`roc_auc`
- 方向：最大化

这和当前代码的 `task_info(task)` 保持一致。

---

## 4.3 调优原则

### 原则 1：分阶段调，不要一次全开

建议分 3 阶段：

#### 阶段 A：固定训练范式

先固定以下设置：

- `model_type=meta-llama/Llama-3.2-1B`
- `llm_frozen=True`
- `output_mlp=False`
- `pretrain=False`
- `text_embedder=mpnet` 或 `glove` 先选一个固定

只调常规训练参数和图参数。

#### 阶段 B：缩小范围后精调

在阶段 A 的 top-k 配置附近继续细化：

- `lr`
- `wd`
- `dropout`
- `num_neighbors`
- `channels`
- `loss_class_weight`

#### 阶段 C：再切换训练范式做对比实验

等常规参数稳定后，再单独做几组“范式切换”实验：

- `output_mlp=True/False`
- `llm_frozen=True/False`
- `text_embedder=glove/mpnet`
- `pretrain=True/False`

这类不是普通超参数，更像模型设计选择，不建议和第一阶段混搜。

---

## 4.4 推荐搜索空间

下面是针对 **`1B LLM + llm_frozen=True + 二分类任务`** 的建议搜索空间。

### 核心搜索空间

| 参数 | 建议搜索空间 | 说明 |
| --- | --- | --- |
| `lr` | `1e-5 ~ 3e-3`，log-uniform | 核心参数 |
| `wd` | `1e-6 ~ 1e-2`，log-uniform | 核心参数 |
| `dropout` | `0.0 ~ 0.5` | 推荐连续搜索 |
| `channels` | `{64, 128, 256}` | 容量参数 |
| `num_layers` | `{1, 2, 3}` | 图传播深度 |
| `num_neighbors` | `{16, 32, 64, 128}` | 同时影响信息量和显存 |
| `aggr` | `{sum, mean}` | 图聚合方式 |
| `temporal_strategy` | `{uniform, last}` | 时序采样策略 |
| `batch_size` | `{1, 2, 4}` | 受显存限制 |

### 类别不平衡相关参数

`loss_class_weight` 可以按二分类做成两个参数：

- `w_neg`
- `w_pos`

建议先限制到：

- `w_neg`: `0.2 ~ 1.5`
- `w_pos`: `0.2 ~ 1.5`

如果你想减少搜索空间，也可以只搜索一个 `w_pos`，然后固定：

- `w_neg = 1.0`
- `w_pos ∈ [0.5, 3.0]`

这通常更稳。

### 可选后置参数

| 参数 | 建议搜索空间 | 说明 |
| --- | --- | --- |
| `num_demo` | `{0, 2, 4}` | few-shot 成本高，建议后置 |
| `text_embedder` | `{glove, mpnet}` | 先固定再比较 |
| `val_steps` | `{200, 500, 1000}` | 主要影响调度和早停频率 |

---

## 4.5 推荐自动化调优流程

### 阶段 1：粗搜索

目的：

- 快速找到大致有效的区域

配置建议：

- `train_steps=2048~4096`
- `batch_size` 按显存上限取小
- `n_trials=20~50`
- 用 `MedianPruner`
- 只保存验证集最好结果

### 阶段 2：精搜索

目的：

- 围绕 top-5 配置缩小范围

配置建议：

- `train_steps=8192~32768`
- 缩小 `lr/wd/dropout` 范围
- `n_trials=10~20`

### 阶段 3：最终确认

目的：

- 用最佳配置跑完整训练预算

配置建议：

- 固定 `seed`
- 跑 `3` 个随机种子重复实验
- 最终比较平均验证/测试性能

---

## 4.6 实现方式建议

推荐不要直接在 `main.py` 里硬改调参逻辑，而是写一个外层调度脚本：

1. Optuna 采样一组参数
2. 用子进程调用 `python main.py ...`
3. 从 stdout / 日志文件中解析验证集最佳指标
4. 把该指标回传给 Optuna
5. 若 trial 表现差，则尽早 prune

这样做的优点是：

- 不侵入原训练代码
- 可以直接复用已有训练命令
- 出问题时也更容易回放单次实验

---

## 4.7 推荐的命令行模板

对于当前项目，外层调参脚本最终应当生成类似这样的命令：

```bash
python main.py \
  --dataset=rel-amazon \
  --task=user-churn \
  --model_type=meta-llama/Llama-3.2-1B \
  --llm_frozen \
  --channels=128 \
  --num_layers=2 \
  --num_neighbors=64 \
  --aggr=sum \
  --temporal_strategy=last \
  --text_embedder=mpnet \
  --dropout=0.2 \
  --batch_size=1 \
  --val_size=1 \
  --lr=0.0003 \
  --wd=0.001 \
  --val_steps=500 \
  --eval_steps=1024 \
  --loss_class_weight 1.0 1.5 \
  --train_steps=4096
```

---

## 5. 一份可直接执行的调优策略

如果你现在就要开始调，我建议先用这套最实用的方案。

### 固定不动

- `dataset=rel-amazon`
- `task=user-churn`
- `model_type=meta-llama/Llama-3.2-1B`
- `llm_frozen=True`
- `output_mlp=False`
- `pretrain=False`
- `text_embedder=mpnet`
- `batch_size=1`
- `val_size=1`
- `seed=42`

### 第一轮自动调参只搜这些

- `lr`
- `wd`
- `dropout`
- `channels`
- `num_layers`
- `num_neighbors`
- `aggr`
- `temporal_strategy`
- `loss_class_weight`

### 第一轮搜索空间

```text
lr:               1e-5 ~ 3e-3      (log)
wd:               1e-6 ~ 1e-2      (log)
dropout:          0.0 ~ 0.5
channels:         {64, 128, 256}
num_layers:       {1, 2, 3}
num_neighbors:    {16, 32, 64, 128}
aggr:             {sum, mean}
temporal_strategy:{uniform, last}
loss_class_weight:
  方案A:          w_neg=1.0, w_pos in [0.5, 3.0]
  方案B:          w_neg in [0.2, 1.5], w_pos in [0.2, 1.5]
```

### 第一轮预算

- `train_steps=4096`
- `n_trials=30`

### 第二轮

取第一轮前 5 名附近继续精调：

- `train_steps=8192~32768`
- `n_trials=10~15`

---

## 6. 当前阶段不纳入自动调优的内部参数

以下参数虽然是超参数，但当前只记录，不建议自动化搜索：

- `TextEmbedderConfig.batch_size=256`
- LoRA 参数：`r=8`, `alpha=16`, `dropout=0.05`, `target_modules=["q_proj","v_proj"]`
- `gamma=2.0`
- `mask_ratio=0.5`
- `pretrain_random_table=False`
- `pretrain_mask_cell=True`
- `autocast dtype=bfloat16`
- projector hidden dim `1024`
- `torch_frame` 表格编码器 `channels=128`, `num_layers=4`
- scheduler `factor=0.8`, `patience=100`
- optimizer `betas=(0.9, 0.95)`

---

## 7. 总结

这份项目里真正适合自动调优的，不是“所有参数”，而是：

- 学习率和正则：`lr`, `wd`, `dropout`
- 图建模参数：`channels`, `num_layers`, `num_neighbors`, `aggr`
- 时序采样参数：`temporal_strategy`
- 类别不平衡参数：`loss_class_weight`

推荐做法是：

1. 固定训练范式
2. 用 Optuna 先粗搜
3. 再围绕 top-k 结果精搜
4. 最后再做 `output_mlp` / `llm_frozen` / `text_embedder` / `pretrain` 这类范式级比较
