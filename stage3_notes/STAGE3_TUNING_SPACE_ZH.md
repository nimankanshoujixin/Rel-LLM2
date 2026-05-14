# Stage 3 调优空间总纲（中文）

这份文档是给人读的中文版本，用来说明当前 Stage 3 的整体目标、调优边界、实验规范和失败后决策。

英文主文档见：

- [STAGE3_TUNING_SPACE.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_TUNING_SPACE.md)

相关配套文档：

- [STAGE3_WORKFLOW.md](/G:/RelLLM-2/Rel-LLM/STAGE3_WORKFLOW.md)
- [STAGE3_PROGRAM.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_PROGRAM.md)
- [experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)
- [baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)

## Repo result confirmation rule / Repo 缁撴灉纭瑙勫垯

- subset screening only means fast evidence, not a repo-facing result
- if a candidate will be kept as a real Stage 3 outcome or used to justify commit / push,
  it must first pass full-test confirmation
- 鍗曟 test-subset 浼樺娍涓嶈兘鐩存帴褰撲綔 repo-level claimed gain

## 1. 当前目标

Stage 3 当前是一个 **只做 finetune 的研究程序**。

当前明确不做：

- 不进入 pretrain
- 不使用 `--pretrain`
- 不切换到新的训练阶段
- 不只为单个代表任务做专项优化
- 在没有真实全局提升前，不提交、不 push 探索性改动

当前优化目标是：

- 在三个代表任务上都不退化：
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- 并且至少有一个代表任务出现可信提升

如果只提升一个任务、但另外任务退化，这仍然算失败候选。

## 2. 架构重构后的四层视图

当前 Stage 3 不再把调优空间看成很多平铺的小模块，而是收束成四层：

1. 表示保真层
2. 语义基底构造与注入层
3. 守恒约束对齐层
4. token 压缩与 LLM 消费层

对应的数据流是：

`图编码表示 -> 对齐到数据库条件化语义基底 -> 受约束守恒迁移 -> 压缩成 graph tokens -> 注入冻结 LLM`

其中真正优先优化的是论文创新点相关部分：

1. `basis construction / injection`
2. `constraint-aware loss`
3. `constraint-aware sampling`
4. `token compression / consumption`

像 prompt 结构、输出格式、预测头这些工程项也允许做，但在优先级上应放在后面。

## 3. 总体指导思想

当前最重要的问题不是“模型还能不能更大、更复杂”，而是：

- 图表示能不能被稳定迁移到数据库条件化语义坐标系里
- 迁移后还能不能保住结构、时间和尺度等关键不变量
- 压缩后的 graph tokens 能不能被冻结 LLM 稳定消费

所以每一个 candidate 在设计时都应回答下面至少一个问题：

- 它是否让 basis 更像数据库特定语义坐标系
- 它是否让 alignment 更像受约束的坐标变换
- 它是否让真正有用的结构证据更稳定地暴露给 LLM
- 它是否让 graph tokens 更容易被冻结 LLM 使用

如果一个 candidate 无法清楚回答这些问题，通常就离论文主线太远，应降优先级。

## 4. 四层的操作菜单

这一部分回答“每层优先试什么、暂缓什么、明确不要试什么”。

### 4.1 第一层：basis 构造 / 注入

优先尝试：

- 在不改外层 prompt 接口的前提下，先细调 token basis 与 graph basis 的注入平衡
- 加入基于 schema 元数据的 route-aware / join-aware basis 语义
- 尝试基于置信度或归一化的 basis gating，而不是固定残差强度
- 降低 graph-side basis 注入噪声

暂缓尝试：

- 需要大规模重做预处理的 basis 重构
- 按任务类型各自设计一套 basis 系统
- 用大规模 encoder 改动去补 basis 路径的不足

当前不优先：

- 把 prompt wording 改动伪装成 basis 改进
- 只能优化单一任务族的 basis 技巧

### 4.2 第二层：constraint-aware loss

优先尝试：

- 局部 token-level reconstruction / consistency loss
- route-aware conservative penalty
- foreign-key 方向、角色、基数等信息保持目标
- 针对 regression 的尺度稳定损失

暂缓尝试：

- 强 global reconstruction bottleneck
- 一次性塞入很多 auxiliary loss
- 在 loss 家族还没验证前就引入复杂调度

当前不优先：

- pretrain 风格目标
- 看不出在保护什么不变量的黑盒 loss

### 4.3 第三层：constraint-aware sampling

优先尝试：

- 在保持 token budget 不变时降低无意义随机性
- route-aware 邻居优先级
- bridge-table-aware 保留策略
- 与预测时刻一致的 temporal-safe 采样

暂缓尝试：

- 单纯扩大邻居规模
- 先上 learned sampler
- 缺乏跨数据库解释力的数据集特化采样

当前不优先：

- 统一砍半 neighbor budget 这种粗暴 pruning
- 本质上只是变相增加 prompt 长度的采样方案

### 4.4 第四层：token 压缩 / LLM 消费

优先尝试：

- 小范围调整 graph token 数量
- 区分 local evidence token 和 global summary token
- 围绕 graph token 插入位置做结构级优化
- 改善 token packaging，而不是先大改 prompt wording

暂缓尝试：

- 大幅重写 prompt 模板
- 强任务特化的 token 接口
- 过早上复杂层级压缩器

当前不优先：

- 自由 prompt wording 搜索
- 破坏冻结 LLM 假设的做法

## 5. 当前候选队列

目前的短队列是：

1. token-only reconstruction follow-up
   - 当前 active
   - 目标：保留对 `user-churn` 有益的部分，去掉可能伤害 `user-ltv` 和
     `item-incoterms` 的 graph-global pressure

2. route-aware conservative loss
   - 如果 token-only reconstruction 仍然跨任务回归，则优先转这一条

3. bridge / route-aware sampling
   - 已完成且全局失败
   - 第一版 `bridge_route` 规则让三个代表任务一起回撤，不应继续做近似重跑

4. calibrated basis gating
   - 已完成且全局失败
   - basis posterior 置信度缩放减轻了 churn 侧回撤幅度，但仍然没有带来跨三任务不退化

5. split local/global graph-token packaging
   - 已完成且全局失败
   - 该 bundle 同时改善了 `user-churn` 和 `user-ltv`，但 `item-incoterms` 仍然回撤，因此不能 promote

当前明确暂停：

- pretrain 及任何 `--pretrain` 路径
- uniform neighbor pruning 作为主方向
- broad graph-global reconstruction pressure 作为默认下一步
- 自由 prompt wording 搜索

## 6. 实验规范

### 6.1 候选注册规范

每个 bundle 必须先有：

- 一个 `stage3_notes/candidates/*.json`
- 三个代表任务 run id
- 明确的 source type：
  - `paper`
  - `ablation`
  - `paper+ablation`
- literature queries
- 论文或 prior ablation 证据
- 清楚的 causal hypothesis
- common overrides
- 必要时的 task-specific overrides

不能只在对话里口头说一个想法就直接起实验。

每个 candidate 还应写清楚：

- 为什么它可能带来全局提升
- 为什么它也可能失败
- 如果改了代码，失败后怎么回退

### 6.2 实验启动规范

默认规则：

- 优先使用本地 Stage 3 程序启动
- 默认走 `stage3_research.py` 和 `stage3_orchestrator.py`
- 不把手写 SSH 当作主要启动路径
- 优先使用当前空闲、且满足资源需求的 GPU
- 如果多个已配置服务器上都有空闲 GPU，就让本地 Stage 3 launcher 一并使用，不要
  人为把常规推进限制在单机或固定 3 张卡上
- 如果集群侧旧限制已经修好，不再保留过时的硬编码 GPU 禁用规则

手动 SSH 只允许用于：

- 诊断
- 远端检查
- 最小恢复操作

### 6.3 自动化与线程绑定规范

允许使用 heartbeat 自动化持续推进 Stage 3，但必须把它视为“绑定当前线程”的状态。

规则：

- 自动化应从当前正常打开的活跃线程创建
- 如果自动化恢复线程时报 path mismatch，应把旧自动化视为已损坏
- 遇到这类错误时，删除旧自动化，并在当前线程重新创建新的自动化
- 不依赖手改旧自动化文件去修复 path mismatch
- 如果一个 monitored bundle 结束了，但更大的 Stage 3 任务还在继续，不要因为 bundle 结束就把健康的自动化删除
- 此时应把自动化重定向到下一个 candidate，或重定向到当前的 architecture-review / 下一步执行步骤

说明：

- `C:\...` 和 `\\?\C:\...` 这种差异，本质上是 Codex 的路径规范化问题
- 这不应被当成实验失败或 repo 配置错误

### 6.4 临时远程补丁与回退规范

如果某个 candidate 需要临时代码改动：

- 可以用 `scp` 同步少量远程补丁
- 远程补丁应尽量小，并且只服务当前 candidate
- candidate 失败后，应把远程被修改过的 tracked repo 文件恢复到当前最新版本
- 可以用 Git 来恢复 tracked repo 文件
- 但不要去 reset / clean / 删除无关内容

明确不要动：

- 远程模型权重
- 虚拟环境
- cache
- 其他非 repo 运行产物

### 6.5 监控与判定规范

当前默认策略：

- 允许训练程序自己的 early stop 自然结束
- 不把 bundle-level manual kill 当成常规策略
- 先用本地程序化方式检查状态
- 优先用 `stage3_orchestrator.py status ... --sync-logs`
- 完成后统一 judge

完成后必须做：

- sync logs
- judge against refreshed strict baseline
- 更新 candidate status
- 更新 `experiment_log.md`
- 写出这次失败或成功如何改变后续搜索空间

### 6.6 文献驱动规范

当前不鼓励“纯凭感觉改”。

更合理的做法是：

- 先找 graph-LLM / graph prompt / graph alignment / structured reasoning 相关文献
- 看已有 trick 是否在相近问题上被证明有效
- 再解释为什么它可能迁移到当前代码库

不是说不能用直觉，而是直觉最好先经过 paper 或 prior ablation 过滤，再变成 bundle。

## 7. 失败后决策协议

这一部分非常重要，专门用来避免“实验做了很多，但一直被困住”。

### 7.1 先区分失败类型

不是所有失败都一样。

要先判断是：

- 实现失败
  - 代码 bug
  - 远程同步问题
  - 协议漂移
  - 日志或 judge 链路坏了
- 科学失败
  - 实验正确跑完，但指标确实不行
- 层级失败
  - 同一层连续多个 candidate 都以相似模式失败

只有科学失败和重复层级失败，才真正用于收缩搜索空间。

### 7.2 抽取稳定失败模式

每个失败 bundle 都要尽量总结出“稳定失败模式”，例如：

- `user-churn` 总是升，但 `user-ltv` 总是掉
- `item-incoterms` 总是最脆弱
- graph-global pressure 对分类有益，但伤 regression
- sampling 改动更多是在放大噪声

如果还看不出稳定模式，下一步 candidate 应保持小而诊断性强。

### 7.3 分层升级处理

Tier 1：单个 bundle 失败，但该层还没打透

- 条件：
  - 当前 bundle 失败
  - 这一层仍有 1 到 2 个强理由候选没试
- 动作：
  - 继续该层
  - 但最多再试 1 到 2 个明显不同的 bundle

Tier 2：该层局部饱和

- 条件：
  - 同层两个及以上 candidate 以相似模式失败
  - 该层 low-cost menu 里已经没有强 paper-backed 假设
- 动作：
  - 标记该层暂时饱和
  - 暂停这一层主方向
  - 切到下一优先层

Tier 3：轻量搜索空间基本打完

- 条件：
  - 四层 priority candidate 基本都试过
  - 再往下试看起来只是重复小修小补
- 动作：
  - 停止继续加小 trick
  - 做一次架构一致性审查
  - 判断是否进入中等规模 redesign

### 7.4 架构一致性审查

当进入 Tier 3 时，要集中检查：

- 论文的两个创新点是否真的被实现成了可训练信号
- basis 是否真的在充当数据库条件化语义坐标系
- conservative alignment 是否在真实 loss/data path 中足够显式
- token compression 是否破坏了前面层想保住的不变量
- 当前 screening protocol 反映的是实质瓶颈还是筛选噪声

这个审查的输出只能是以下几类之一：

- 带着新的中等规模 redesign 重启 Stage 3
- 暂停 Stage 3，进入更结构性的实现阶段
- 先重定义“论文叙事和代码实现的映射”，再继续实验

### 7.5 防卡死规则

不要为了让实验数继续增长，就不断发射近似重复的 bundle。

如果感觉“又卡住了”，默认动作应是：

- 总结失败模式
- 判断当前层是否已经饱和
- 选择切层，或暂停做 redesign

实验次数本身不等于进展。

## 8. route-aware conservative loss 细化方案

这一部分把下一条 loss 家族候选进一步拆成可以直接起 bundle 的具体机制。

整体目标是：

- 不再继续使用脆弱的 graph-global reconstruction pressure
- 改成更局部、更保守的守恒对齐信号
- 优先保护“证据路径可读性”而不是逼迫所有任务经过同一个全局重构瓶颈

### 8.1 方案 A：route-consistency loss

核心思想：

- 如果两个样本暴露出的 schema-path / relation-path 签名相近，那么它们的
  route-conditioned alignment 状态也应更一致

机制草图：

- 从当前 finetune 路径里已经可见的 sampled route 或 relation sequence 中提取紧凑
  route signature
- 在 batch 内形成“同类 route”样本对
- 约束它们的 route-conditioned basis-query state 或 route summary 不要偏得太开

可能收益：

- 它直接打“路径可读性”，而不是全局图重建
- 有机会提升结构相似样本之间的对齐稳定性

可能风险：

- 如果 route signature 太粗，会把本来应区分的任务差异抹平
- 对 regression 来说，如果它不保留尺度信息，可能还是会伤 `user-ltv`

实现成本：

- 中等

### 8.2 方案 B：FK-direction conservative loss

核心思想：

- 显式保护 foreign-key 的方向性，不让 alignment 把 source / target 角色洗平

机制草图：

- 识别 sampled subgraph 中已有的方向关系
- 当正向和反向角色的 aligned state 过于接近时施加惩罚
- 可以落在 graph-query 或 token-query 层，而不必等到最终输出头

可能收益：

- 它保护的是论文里非常核心的不变量
- 相比 graph-global reconstruction，风险更可控

可能风险：

- 如果当前 encoder 已经把方向信息编码得很强，这个 loss 可能增益有限
- 也可能更偏向帮助分类，而不一定直接帮助回归

实现成本：

- 低到中等

### 8.3 方案 C：bridge-sensitivity conservative loss

核心思想：

- 显式保护 many-to-many bridge route 的独特对齐痕迹，避免它被压成普通邻域噪声

机制草图：

- 检测 sampled evidence path 是否经过 bridge-like 结构
- 要求 bridge-mediated route 保持可分辨的中间 summary 或 basis state
- 惩罚 direct route 与 bridge route 过度平滑

可能收益：

- 它直接针对 join / bridge 路由效应
- 和论文中的“结构守恒”叙事高度一致

可能风险：

- bridge detection 可能带来数据集依赖
- 如果 bridge 问题本质在 sampling 而非 loss，单靠 loss 可能不够

实现成本：

- 中等

### 8.4 当前建议顺序

当前建议按下面顺序尝试这一家族：

1. FK-direction conservative loss
2. route-consistency loss
3. bridge-sensitivity conservative loss

原因是：

- FK-direction 最局部，也最容易和论文中的守恒叙事对齐
- route-consistency 更广，但仍然比较可控
- bridge-sensitivity 很有价值，但实现和诊断都更敏感

## 9. 这份中文文档的用途

这份文档主要是给你快速看全局、做策略讨论、判断下一步是否合理。

英文文档仍然保留，因为：

- 便于 agent 后续持续沿统一格式推进
- 便于和机器可读配置、bundle 规范保持一致

后续如果英文总纲发生重要变化，这份中文文档也应同步更新。
# Encoding / authority note

This file currently appears mojibake-corrupted in this workspace. Do not treat it as the primary
source of truth until it is regenerated as valid UTF-8 Chinese.

Use these files as authoritative for current Stage 3 decisions:

- `STAGE3_WORKFLOW.md`
- `stage3_notes/STAGE3_PROGRAM.md`
- `stage3_notes/STAGE3_TUNING_SPACE.md`
- `stage3_notes/STAGE3_ARCHITECTURE_REVIEW_2026-05-07.md`
- `stage3_notes/STAGE3_NOISE_FLOOR_PROTOCOL_2026-05-07.md`
- `stage3_notes/STAGE4_PRETRAIN_COMPATIBILITY_2026-05-08.md`
- `stage3_notes/experiment_log.md`
