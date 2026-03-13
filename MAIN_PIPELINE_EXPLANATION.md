# Rel-LLM `main.py` 运行逻辑说明

## 说明范围

这份文档只分析把 `main.py` 作为入口时，实际会走到的代码路径。

- 重点覆盖 `main.py` 直接调用到的本地文件
- 默认参数路径是 `--dataset rel-stack --task user-engagement`
- 同时注明哪些地方会因为命令行参数不同而切换到别的文件或分支
- 不分析 `examples/`、`main_wrapper.py`、`train.sh` 这类不会被 `main.py` 直接调用的文件

这个项目的核心思想是：

1. 用 `relbench` 把关系型数据库数据集整理成多表时序数据
2. 用 `torch_frame` 把每张表的列转成张量特征
3. 用 `torch_geometric` 把多表主外键关系变成异构图
4. 先做表内编码，再做 GNN 消息传递
5. 最后走两条预测路线之一
   - 纯 GNN 路线：GNN 表征 -> MLP 输出
   - GNN + LLM 路线：GNN 表征 -> 投影到 LLM embedding 空间 -> 让因果语言模型完成分类/回归

---

## 一、`main.py` 入口实际会使用到的文件及其功能

下面按“本地文件”维度解释各自职责。

### 1. 根目录下的核心文件

#### `main.py`

主训练入口。

- 解析命令行参数
- 固定随机种子、选择设备
- 加载数据集和任务
- 推断每张表每一列的语义类型 `stype`
- 构建异构时序图
- 构建 `NeighborLoader`
- 初始化 `Model`
- 执行可选预训练
- 执行监督微调
- 在验证集和测试集上评估

它本身不负责定义模型细节，也不负责定义数据集结构，而是负责把这些模块串起来。

#### `model.py`

项目最核心的模型定义文件，`Model` 类在这里。

它把 3 层能力拼在一起：

1. 表格编码：`HeteroEncoder`
2. 时序编码：`HeteroTemporalEncoder`
3. 图结构聚合：`HeteroGraphSAGE`

之后再分成两种输出模式：

- `model_type == 'gnn'`
  - 只用 GNN 输出
  - 直接用 `self.head` 做预测
- 其他 `model_type`
  - 加载 HuggingFace 因果语言模型
  - 把图节点 embedding 投影到 LLM token embedding 空间
  - 用文本任务描述 + 图表示联合做训练或推理

同时它还实现了：

- `pretrain()`: 自监督/弱监督预训练
- `forward()`: 微调训练和推理的主入口
- `get_demo_info()`: few-shot demo 的图表征准备
- `recursive_sample()` / `get_neighbor_embedding()`: 取邻居上下文作为图提示

#### `text_embedder.py`

文本列嵌入器封装。

- 如果参数是 `glove`，实际加载 `average_word_embeddings_glove.6B.300d`
- 如果参数是 `mpnet`，实际加载 `all-mpnet-base-v2`
- 如果本地缓存存在，就从本地加载
- 否则从 HuggingFace 下载并缓存

这个类会传给 `torch_frame`，用于把表中的文本列先编码成向量。

#### `utils.py`

放的是任务级别的辅助信息，不是通用工具库。

主要内容：

- `description_dict`: 每个数据集/任务的自然语言任务描述
- `question_dict`: 每个数据集/任务对应的自然语言问题模板
- `task_info(task)`: 根据任务类型返回
  - 输出维度 `out_channels`
  - 损失函数 `loss_fn`
  - 调参指标 `tune_metric`
  - 指标方向 `higher_is_better`
  - 回归任务的裁剪范围 `clamp_min/clamp_max`
- `initialize_weights()`: 用于初始化模型中的线性层

这里的 `description_dict` 和 `question_dict` 是 LLM 分支非常关键的提示词来源。

---

### 2. `relbench` 中入口会直接触发的文件

#### `relbench/datasets/__init__.py`

数据集注册表。

- `get_dataset(name, download=False)` 根据名字返回数据集对象
- 默认 `main.py` 调 `get_dataset("rel-stack", download=True)`
- 这里决定会实例化哪个数据集类

#### `relbench/tasks/__init__.py`

任务注册表。

- `get_task(dataset_name, task_name, download=False)` 根据数据集名和任务名返回任务对象
- 默认 `main.py` 调 `get_task("rel-stack", "user-engagement", download=True)`
- 这里决定最终落到哪个任务类

#### `relbench/datasets/stack.py`

默认数据集 `rel-stack` 的定义。

主要负责：

- 下载原始论坛数据
- 读取 `Users/Comments/Posts/Votes/...` 等 CSV
- 删除会造成时间泄漏的列
- 清洗时间列
- 组装成多张 `Table`
- 最终返回 `Database`

默认运行 `main.py` 时，数据集就是从这里构建出来的。

#### `relbench/tasks/stack.py`

默认任务 `user-engagement` 的定义文件。

其中默认会命中：

- `UserEngagementTask`

这个任务的标签语义是：

- 对每个用户，预测其在未来一个时间窗口内是否会继续产生互动
- 互动来源包括 `posts`、`votes`、`comments`

它通过 DuckDB SQL 从数据库表中生成 train/val/test 任务表。

#### `relbench/base/dataset.py`

`Dataset` 基类。

负责：

- 生成或读取缓存后的 `Database`
- 在 `test_timestamp` 处截断数据库，避免测试泄漏
- 调用 `validate_and_correct_db()` 修正越界外键

#### `relbench/base/database.py`

`Database` 基类。

负责多张表的管理：

- `save/load`
- `upto(timestamp)` 时间截断
- `reindex_pkeys_and_fkeys()` 把原始主键/外键映射成连续整数索引

这一步非常关键，因为后面的异构图节点索引必须是连续的整数。

#### `relbench/base/table.py`

`Table` 基类。

负责：

- 包装单张 `pandas.DataFrame`
- 保存主键列、外键列、时间列信息
- 支持 `save/load`
- 支持按时间切片 `upto()` / `from_()`

#### `relbench/base/task_base.py`

任务基类 `BaseTask`。

负责：

- 统一 train/val/test 的时间切分逻辑
- 构建各 split 的任务表
- 缓存任务表
- 对 test split 自动隐藏标签列，避免泄漏

#### `relbench/base/task_entity.py`

实体预测任务基类 `EntityTask`。

默认任务 `UserEngagementTask` 继承自这里。

负责：

- 过滤悬空实体
- `evaluate()`：把预测值与任务表标签做指标计算

#### `relbench/modeling/utils.py`

图构建前的辅助逻辑。

主要包括：

- `get_stype_proposal(db)`: 自动推断每张表每列的语义类型
- `remove_pkey_fkey()`: 从输入特征里去掉主键/外键列
- `to_unix_time()`: 时间列转 UNIX 秒

#### `relbench/modeling/graph.py`

把关系型数据库变成图数据的核心文件。

主要函数：

- `make_pkey_fkey_graph()`
  - 把数据库每张表转成 `TensorFrame`
  - 把主外键关系转成异构图边
  - 给带时间列的表附加 `time`
- `get_node_train_table_input()`
  - 把任务表变成 `NeighborLoader` 的输入节点、时间和标签附着变换

#### `relbench/modeling/nn.py`

定义图编码组件。

主要类：

- `HeteroEncoder`
  - 每种节点类型一套表格编码器
- `HeteroTemporalEncoder`
  - 用相对时间差给节点加时间编码
- `HeteroGraphSAGE`
  - 对异构图做多层 GraphSAGE 消息传递

#### `relbench/metrics.py`

指标实现。

默认任务 `user-engagement` 是二分类，所以主要会用到：

- `average_precision`
- `accuracy`
- `f1`
- `roc_auc`

#### `relbench/utils.py`

默认 `rel-stack` 数据集构建时会用到。

主要作用：

- `clean_datetime()`: 清洗时间列
- `unzip_processor()`: 处理下载的 zip 数据

---

### 3. `torch_frame` 中被实际触发的文件

#### `torch_frame/config/text_embedder.py`

定义 `TextEmbedderConfig`。

它只是一个配置对象，把：

- 文本嵌入函数
- 批大小

交给 `torch_frame` 在物化表时使用。

#### `torch_frame/utils/infer_stype.py`

自动识别列类型。

它会把列推断成例如：

- `numerical`
- `categorical`
- `timestamp`
- `multicategorical`
- `text_embedded`

这个项目里它的结果直接决定：

- 某列是数值编码还是类别编码
- 某列是否会被送去文本嵌入器

#### `torch_frame/data/dataset.py`

把 `DataFrame` 物化成 `TensorFrame` 的核心实现。

主要做 4 件事：

1. 统计每列的统计量 `col_stats`
2. 依据 `stype` 选择对应 mapper
3. 把 DataFrame 各列转成张量
4. 产出 `tensor_frame`

其中如果列是 `text_embedded`，它会调用你在 `main.py` 里传入的 `TextEmbedding`。

#### `torch_frame/data/tensor_frame.py`

`TensorFrame` 数据结构。

可以理解成“按语义类型组织的表格张量容器”。

它保存：

- `feat_dict`: 各种类型列的张量
- `col_names_dict`: 每种类型下有哪些原始列名
- `y`: 标签

之后 `HeteroEncoder` 就是直接吃这个对象。

#### `torch_frame/nn/models/resnet.py`

表格编码器默认实现。

在这个项目里，`HeteroEncoder` 默认给每种节点表都用这个 `ResNet`。

其结构是：

- 先按列类型编码
- 再把所有列 embedding 展平
- 再经过若干残差全连接块
- 输出每个节点一条固定维度向量

#### `torch_frame/nn/encoder/stype_encoder.py`

按列语义类型做编码的底层实现。

在本项目实际会用到的几类编码器包括：

- `EmbeddingEncoder`: 编码类别列
- `LinearEncoder`: 编码数值列
- `MultiCategoricalEmbeddingEncoder`: 编码多值类别列
- `TimestampEncoder`: 编码时间列
- `LinearEmbeddingEncoder`: 编码预先算好的文本 embedding 列

也就是说，表格列不是直接喂给 GNN，而是先在这里变成统一维度的列向量，再汇总成节点向量。

---

## 二、从 `main.py` 开始的完整运行流水线

下面按“程序真实执行顺序”来讲。

---

### 阶段 0：启动与参数解析

入口是：

- `main.py`
- `if __name__ == '__main__':`

这里先执行 `argparse`，得到关键参数。

主要参数可分为四组：

#### A. 数据与缓存

- `--dataset`
- `--task`
- `--cache_dir`
- `--debug`

#### B. 图模型参数

- `--channels`
- `--aggr`
- `--num_layers`
- `--num_neighbors`
- `--temporal_strategy`
- `--text_embedder`
- `--text_embedder_path`

#### C. LLM 参数

- `--model_type`
- `--llm_frozen`
- `--output_mlp`
- `--dropout`
- `--num_demo`
- `--max_new_tokens`
- `--loss_class_weight`

#### D. 训练参数

- `--epochs`
- `--pretrain`
- `--pretrain_epochs`
- `--val_steps`
- `--batch_size`
- `--val_size`
- `--num_workers`
- `--lr`
- `--wd`
- `--seed`

然后执行：

1. `seed_everything(args.seed)`
2. 选择 `device`
3. 如果是 CUDA，限制线程数

---

### 阶段 1：加载数据集对象

代码入口：

- `dataset = get_dataset(args.dataset, download=True)`

真实调用链：

1. `main.py -> relbench.datasets.get_dataset()`
2. `get_dataset()` 根据名字去 `dataset_registry` 查
3. 默认 `rel-stack` 会命中 `relbench/datasets/stack.py` 里的 `StackDataset`
4. 返回 `StackDataset(cache_dir=...)`

然后：

- `db = dataset.get_db()`

继续调用链：

1. `Dataset.get_db()`
2. 如果缓存存在，直接从 parquet 加载 `Database`
3. 如果缓存不存在，调用 `StackDataset.make_db()`
4. `StackDataset.make_db()` 下载原始论坛 CSV
5. 清洗时间列、删除泄漏列、构造多张 `Table`
6. `Database.reindex_pkeys_and_fkeys()`
   - 把每张表主键重排成连续整数
   - 把所有外键映射到这些整数索引
7. 按 `test_timestamp` 截断，避免未来数据泄漏

这一阶段的输出是：

- `dataset`: 数据集对象
- `db`: 一个关系数据库对象，里面有多张表

对默认 `rel-stack`，`db.table_dict` 里主要有：

- `users`
- `posts`
- `comments`
- `votes`
- `badges`
- `postLinks`
- `postHistory`

---

### 阶段 2：加载任务对象

代码入口：

- `task = get_task(args.dataset, args.task, download=True)`

真实调用链：

1. `main.py -> relbench.tasks.get_task()`
2. 先根据 `dataset_name + task_name` 在 `task_registry` 里查
3. 默认会命中 `relbench/tasks/stack.py` 里的 `UserEngagementTask`
4. `task = UserEngagementTask(dataset, cache_dir=...)`

此时 `task` 只是一份“任务定义”，还没有真正生成 train/val/test 表。

---

### 阶段 3：推断每张表每列的语义类型 `stype`

代码入口：

- 读取或生成 `stypes.json`

调用链：

1. `main.py` 先尝试读取缓存的 `stypes.json`
2. 如果没有缓存，执行 `get_stype_proposal(db)`
3. `get_stype_proposal()` 内部对每张表采样
4. 调 `torch_frame.utils.infer_df_stype()`
5. `infer_df_stype()` 逐列判断是：
   - 数值
   - 类别
   - 时间
   - 多类别
   - 文本嵌入列

这一阶段的输出是：

- `col_to_stype_dict`

它是后续“如何把列变成张量”的总配置。

---

### 阶段 4：构建文本嵌入器

代码入口：

- `text_embedder = TextEmbedding(args.text_embedder, args.text_embedder_path, device=device)`

逻辑：

1. 若参数是 `glove`
   - 实际模型名改为 `average_word_embeddings_glove.6B.300d`
2. 若参数是 `mpnet`
   - 实际模型名改为 `all-mpnet-base-v2`
3. 若本地目录存在缓存
   - 直接加载
4. 否则下载并保存到本地

这一阶段的输出是：

- 一个可调用对象 `text_embedder(sentences) -> Tensor`

它会在下一步表格物化时被自动调用。

---

### 阶段 5：把数据库转成异构图

代码入口：

- `make_pkey_fkey_graph(...)`

这是整个数据流水线里最关键的一步。

真实调用链：

1. `main.py -> relbench.modeling.graph.make_pkey_fkey_graph(db, col_to_stype_dict, text_embedder_cfg, cache_dir)`
2. 对 `db.table_dict` 中每一张表循环执行

对单张表，内部逻辑是：

1. 取出原始 `df`
2. 检查主键是否已是连续整数
3. 复制该表的 `col_to_stype`
4. `remove_pkey_fkey()`
   - 从输入特征中删掉主键/外键列
   - 因为主外键只用于建图，不作为表特征输入
5. 构造 `torch_frame.data.Dataset(df, col_to_stype, col_to_text_embedder_cfg=...)`
6. 调 `materialize(path=...)`
   - 统计列统计量
   - 调不同 mapper 把列映射成张量
   - 对文本列调用 `TextEmbedding`
   - 最终得到 `TensorFrame`
7. 把结果写入 `data[table_name].tf`
8. 如果表有时间列，转成 UNIX 时间写入 `data[table_name].time`
9. 遍历每个外键关系，建立两条边
   - 正向边：`fkey -> pkey`
   - 反向边：`pkey -> fkey`

这一阶段的两个核心输出：

- `data`: `torch_geometric.data.HeteroData`
- `col_stats_dict`: 每张表每列的统计信息

其中 `data` 里每个节点类型包含：

- `tf`: 该表的张量化特征
- `df`: 该表 DataFrame，后续给 LLM 构造文本上下文时会再次使用
- `time`: 节点时间

---

### 阶段 6：为 train/val/test 构造采样器输入

代码入口：

- `for split in ["train", "val", "test"]`

每个 split 的调用链如下：

1. `table = task.get_table(split)`
2. `task.get_table(split)` 内部
   - 若缓存存在，直接加载任务表
   - 否则调用 `BaseTask._get_table(split)`
3. 对默认 `UserEngagementTask`
   - `BaseTask._get_table()` 会根据 split 生成时间点
   - 调 `UserEngagementTask.make_table(db, timestamps)`
4. `UserEngagementTask.make_table()` 用 DuckDB SQL 生成一张任务表
   - 每一行表示“某个用户在某个时间点的预测样本”
   - 标签列是 `contribution`

然后主程序继续：

5. `table_input = get_node_train_table_input(table, task)`
6. 该函数把任务表转成：
   - `nodes`: 要预测的实体节点 ID
   - `time`: 每个样本的 seed time
   - `target`: 标签张量
   - `transform`: 一个把标签挂到 batch 上的变换
7. 用这些信息创建 `NeighborLoader`

`NeighborLoader` 的作用是：

- 不是一次把整张大图送进模型
- 而是对每个目标节点按时间约束做邻居采样
- 返回一个适合当前 batch 的异构子图

这一阶段的输出是：

- `loader_dict["train"]`
- `loader_dict["val"]`
- `loader_dict["test"]`

---

### 阶段 7：确定损失函数与评估方向

代码入口：

- `out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = task_info(task)`

默认 `user-engagement` 是二分类，所以得到：

- `out_channels = 1`
- `loss_fn = BCEWithLogitsLoss()`
- `tune_metric = "roc_auc"`
- `higher_is_better = True`

这组输出决定：

- 模型最后输出几维
- 训练如何算 loss
- 验证时用哪个指标挑最优模型

---

### 阶段 8：初始化模型

代码入口：

- `model = Model(...)`

`Model.__init__()` 内部真实逻辑如下。

#### 8.1 表格编码器初始化

- `self.encoder = HeteroEncoder(...)`

其作用：

- 对每个节点类型单独建一个 `torch_frame` 表格编码器
- 默认每张表都用 `torch_frame.nn.models.ResNet`

也就是：

- `users` 表一套编码器
- `posts` 表一套编码器
- `comments` 表一套编码器
- 彼此参数不共享

#### 8.2 时间编码器初始化

- `self.temporal_encoder = HeteroTemporalEncoder(...)`

其作用：

- 计算每个节点记录时间相对当前预测时间 `seed_time` 的差值
- 将相对时间转成位置编码
- 再加到节点特征上

#### 8.3 异构图 GNN 初始化

- `self.gnn = HeteroGraphSAGE(...)`

其作用：

- 对每种边类型建立 `SAGEConv`
- 多层迭代传播
- 每层后做 `LayerNorm + ReLU`

#### 8.4 输出头初始化

- `self.head = MLP(channels, out_channels=..., num_layers=1, ...)`

这个只在纯 GNN 分支中直接输出。

#### 8.5 根据 `model_type` 选择后端

##### 分支 A：`model_type == 'gnn'`

- 不加载 LLM
- `self.model = None`
- 训练时：
  - `encode -> gnn -> head`

##### 分支 B：`model_type != 'gnn'`

- 加载 HuggingFace 因果语言模型
- 加载 tokenizer
- 增加 `<MASK>` token
- 根据 `llm_frozen` 决定：
  - 全冻结
  - 或用 LoRA 微调
- 建立 `self.projector`
  - 把 GNN 输出的 `channels` 维向量映射到 LLM 的 token embedding 维度
- 如果 `output_mlp=True`
  - 再建立 `lm_head`
  - 用 LLM 最后隐状态做分类/回归，不走文本生成
- 如果 `output_mlp=False`
  - 直接走语言模型生成/teacher forcing 训练

---

### 阶段 9：优化器、学习率调度器和参数统计

主程序里接着做：

1. 统计 `requires_grad=True` 的参数
2. 创建优化器
   - 有权重衰减时用 `AdamW`
   - 且不给 `bias` / `LayerNorm` 加衰减
3. 创建 `ReduceLROnPlateau`
   - 根据验证集指标自动降学习率
4. 打印可训练参数量

---

### 阶段 10：可选预训练 `model.pretrain()`

只有在 `--pretrain` 时才会执行。

单个 batch 的调用链是：

1. `batch = batch.to(device)`
2. `loss = model.pretrain(batch, task.entity_table)`

`model.pretrain()` 的核心逻辑如下：

#### 10.1 先决定怎么做 mask

有两种预训练思路，但当前默认更偏向“掩码某个单元格”：

- `pretrain_mask_cell=True`
  - 随机选一列
  - 把 batch 里一部分样本该列清零
- 否则
  - 直接把某些节点 embedding 替换成 mask embedding

#### 10.2 先走一次图编码

- `x_dict, _ = self.encode(batch, entity_table)`
- `x_dict = self.gnn(x_dict, batch.edge_index_dict)`
- 得到被 mask 节点的图表示
- 再通过 `self.projector` 映射到 LLM embedding 空间

#### 10.3 从原始表里取被 mask 节点对应的真实字段值

这里会访问：

- `batch[select_table].df`

并基于真实列值构造一个真假判断任务，例如：

- `"column_x is value_y."`
- 再问一句：
  - `Is the statement correct? Give Yes or No as answer.`

#### 10.4 用图 embedding 作为软提示，喂给 LLM

模型输入实际上是：

- `BOS`
- 图节点 embedding
- 问题 embedding
- 文本 statement embedding

然后让 LLM 预测标签：

- `Yes`
- `No`

#### 10.5 返回预训练 loss

- `outputs = self.model(..., labels=...)`
- `return outputs.loss`

主循环拿到 loss 后：

1. `loss.backward()`
2. `optimizer.step()`
3. 每 `val_steps` 做一次 `test()`
4. 若验证指标更优，就保存 `state_dict`

预训练本质上是在教 LLM 分支把“图表示”和“表格事实”对齐。

---

### 阶段 11：监督微调 `model.forward()`

这是 `main.py` 的主训练阶段。

单个 batch 的逻辑分两条大分支。

#### 分支 A：纯 GNN 或 `output_mlp=True`

主程序调用：

- `output_pred = model(batch, task.entity_table)`

此时会进入 `Model.forward(batch, entity_table, context=True, demo_info=None, inference=False)`

内部流水线如下：

1. `x_dict, batch_size = self.encode(batch, entity_table)`

`self.encode()` 又做三件事：

- `self.encoder(batch.tf_dict)`
  - 对每种节点类型的 `TensorFrame` 做表格编码
- `self.temporal_encoder(...)`
  - 计算相对时间 embedding
- 如果配置了浅层 ID embedding，再叠加 ID embedding

输出：

- `x_dict`: 每种节点类型的初始节点向量
- `batch_size`: 当前目标节点数

2. `x_dict = self.gnn(x_dict, batch.edge_index_dict)`
   - 做异构图消息传递

3. `node_embed = x_dict[entity_table][:batch_size]`
   - 只取当前 seed 节点对应的前 `batch_size` 行

4. 若 `self.model is None`
   - 纯 GNN 直接 `self.head(node_embed)`

5. 若 `output_mlp=True`
   - 先 `self.projector(node_embed)`
   - 构造 prompt embedding
   - 跑 LLM 前向，取最后一层隐藏状态
   - `self.lm_head(hidden)` 得到预测值

6. 主程序拿到 `output_pred`
   - 二分类/多标签时会与真实标签计算 `loss_fn`

#### 分支 B：LLM 文本监督训练，`output_mlp=False`

主程序调用还是：

- `loss = model(batch, task.entity_table)`

但此时 `Model.forward()` 不直接返回预测，而是直接返回训练 loss。

内部流程如下。

##### 11.1 图编码阶段

与前面一样：

1. `self.encode()`
2. `self.gnn()`
3. `node_embed = self.projector(node_embed)`

##### 11.2 构造任务文本

从 `utils.py` 中取：

- `task_desc = description_dict[self.dataset][self.task.name]`
- `question = question_dict[self.dataset][self.task.name]`

默认任务下，问题本质上类似：

- “基于活动记录，这个用户在未来 3 个月是否会继续互动？请回答 Yes 或 No。”

##### 11.3 可选 few-shot demo

如果 `num_demo > 0`，会先通过：

- `get_demo_info()`

得到一批 demo 节点的图 embedding 和标签 token。

训练/推理时再把这些 demo 作为 in-context examples 插入 prompt。

##### 11.4 可选邻居上下文

如果 `context=True`，会执行：

- `recursive_sample()`
- `get_neighbor_embedding()`

也就是：

- 递归收集当前目标节点的一跳邻居
- 取这些邻居的图 embedding
- 拼到当前节点 embedding 后面

等价于把“目标节点 + 邻居上下文”一起作为软提示送给 LLM。

##### 11.5 构造 LLM 的 `inputs_embeds`

每个样本会拼成这样：

1. `BOS`
2. 图 prompt embedding
3. 任务描述文本 embedding
4. 任务问题文本 embedding
5. 若训练阶段且非推理
   - 再拼上标签 token

注意这里不是把图信息转成文字，而是把图 embedding 直接拼进 token embedding 序列。

##### 11.6 训练目标

如果不是二分类：

- 直接让 HuggingFace 模型按 `labels=...` 计算标准因果语言模型 loss

如果是二分类：

- 不直接用标准 CE
- 而是手工取目标 token 的概率
- 用 focal loss 形式计算
- 再可选乘 `alpha` 类别权重

返回值：

- 一个标量 `loss`

主循环拿到后执行：

1. `loss.backward()`
2. `optimizer.step()`
3. 每 `val_steps` 做验证和测试
4. 如果验证更优，保存当前 `state_dict`

---

### 阶段 12：验证与测试 `test()`

评估入口在 `main.py` 顶部定义的：

- `test(loader, demo_info=None)`

这个函数依赖外部全局变量：

- `model`
- `device`
- `task`
- `args`
- `clamp_min / clamp_max`

执行流程如下：

1. `model.eval()`
2. 遍历 `loader`
3. `pred = model(test_batch, task.entity_table, demo_info, inference=True)`
4. 若是回归任务
   - 用训练集分位数裁剪输出
5. 若是 GNN 分支或 `output_mlp=True`，且任务是二分类/多标签
   - 对输出做 `sigmoid`
6. 整理维度
7. 拼接所有 batch 的预测
8. 返回 `numpy.ndarray`

然后主程序会调用：

- `task.evaluate(val_pred, task.get_table("val"))`
- `task.evaluate(test_pred)`

对于默认二分类任务，会在 `EntityTask.evaluate()` 中调用：

- `average_precision`
- `accuracy`
- `f1`
- `roc_auc`

---

### 阶段 13：训练结束后的最终评估

主程序最后会：

1. `model.load_state_dict(state_dict)`
   - 载入验证集最优参数
2. 再跑一次 `val_pred = test(loader_dict["val"])`
3. 再跑一次 `test_pred = test(loader_dict["test"])`
4. 打印最佳验证和测试指标
5. 若开启 wandb，再把最终测试指标写入日志

这就是完整闭环。

---

## 三、把默认配置代入后的“可读版调用链”

下面用默认参数举一个更直观的版本。

### 默认情况下，程序大致会这样跑

1. 进入 `main.py`
2. 解析参数，默认得到：
   - `dataset = rel-stack`
   - `task = user-engagement`
   - `model_type = meta-llama/Llama-3.2-1B`
3. 调 `get_dataset("rel-stack")`
4. 进入 `StackDataset`
5. `dataset.get_db()` 读取或构建论坛数据库
6. 调 `get_task("rel-stack", "user-engagement")`
7. 进入 `UserEngagementTask`
8. 推断每张表每一列的 `stype`
9. 构建 `TextEmbedding`
10. 调 `make_pkey_fkey_graph()`
11. 每张表被物化成 `TensorFrame`
12. 主外键关系被转成异构图边
13. 调 `task_info(task)`，得出这是二分类任务，用 `BCEWithLogitsLoss`
14. 对 train/val/test 分别调用 `task.get_table(split)`
15. `UserEngagementTask.make_table()` 用 SQL 生成用户未来是否活跃的标签表
16. `get_node_train_table_input()` 把任务表转成图采样输入
17. `NeighborLoader` 开始能按时间采样局部子图
18. 初始化 `Model`
19. `Model` 内部初始化：
   - 表格编码器
   - 时间编码器
   - 异构 GraphSAGE
   - Llama tokenizer 和 LLM
   - projector
20. 开始训练
21. 每个 batch 进入 `Model.forward()`
22. `encode()` 先把子图里的每种节点表格编码成向量
23. `temporal_encoder` 给节点加相对时间信息
24. `gnn()` 让不同表之间通过主外键边传播信息
25. 取目标用户节点 embedding
26. 用 `projector` 把用户节点 embedding 投影到 LLM token embedding 维度
27. 从 `utils.py` 取出任务描述和问题模板
28. 把“图 embedding + 任务文字”拼成 LLM 输入
29. 若训练阶段：
   - 用真实标签 token 计算 loss
30. 若验证/测试阶段：
   - 用 `generate()` 生成答案
   - 对 `Yes/No` 的概率做解析，得到预测值
31. `task.evaluate()` 用标签表算出 `roc_auc/f1/...`
32. 保存验证集最优模型
33. 训练结束后载入最佳参数，重新评估并打印最终指标

---

## 四、代码中的关键分支

理解这个项目时，最重要的是先抓住下面几个分支。

### 1. 任务分支

由这两个参数决定：

- `--dataset`
- `--task`

它们会决定：

- 加载哪个 `relbench/datasets/*.py`
- 加载哪个 `relbench/tasks/*.py`
- 任务的标签定义是什么
- 用什么评估指标

如果你改成别的任务，整体骨架不变，但：

- 任务表 SQL 变了
- 标签语义变了
- 描述词和问题模板也变了

### 2. 模型后端分支

由 `--model_type` 决定。

#### 当 `model_type == 'gnn'`

是标准图学习流水线：

- 表格编码
- 时间编码
- GNN
- MLP

#### 当 `model_type` 是 Llama/Qwen/DeepSeek

是图表示驱动的 LLM 流水线：

- 表格编码
- 时间编码
- GNN
- projector
- LLM

### 3. LLM 输出分支

由 `--output_mlp` 决定。

#### `output_mlp=False`

- 走文本生成/teacher forcing
- 更像 prompt-based learning

#### `output_mlp=True`

- 不直接生成文本
- 只取 LLM 的最后隐藏状态，再过 `lm_head`
- 更像“LLM 作为特征提取器”

### 4. 预训练分支

由 `--pretrain` 决定。

#### `pretrain=False`

- 直接进入监督微调

#### `pretrain=True`

- 先做一轮自监督/弱监督预训练
- 再加载最优预训练参数进入微调

---

## 五、你阅读这套代码时建议优先看的顺序

如果你接下来要改代码，建议按下面顺序读。

1. `main.py`
   - 先搞清楚训练主循环和参数入口
2. `model.py`
   - 再搞清楚图表示如何进入 LLM
3. `relbench/modeling/graph.py`
   - 看数据库如何变成异构图
4. `relbench/tasks/stack.py`
   - 看默认任务标签是怎么生成的
5. `relbench/modeling/nn.py`
   - 看表格编码、时间编码、GraphSAGE 的拼接方式
6. `text_embedder.py`
   - 看文本列是怎么变 embedding 的
7. `utils.py`
   - 看 prompt 模板与任务配置

如果你后面主要想改“模型”，重点读：

- `model.py`
- `relbench/modeling/nn.py`

如果你主要想改“数据/样本定义”，重点读：

- `relbench/datasets/stack.py`
- `relbench/tasks/stack.py`
- `relbench/modeling/graph.py`

如果你主要想改“LLM 提示或输出方式”，重点读：

- `utils.py`
- `model.py`

---

## 六、一句话总结

这份代码的主线可以概括为：

`main.py` 先把关系型数据库整理成“带时间的异构图”，再用 `torch_frame` 编码表格列、用 `GraphSAGE` 聚合跨表关系，最后把目标节点的图表示交给纯 GNN 头或 LLM 头完成预测。

