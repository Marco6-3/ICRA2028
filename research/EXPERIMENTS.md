# 实验路线：从 Physics Bottleneck 到 Temporal Memory

状态：方案，未运行。更新：2026-10-05。

## 总原则

当前路线不先比较一堆时序网络，而先回答两个问题：

1. 低维 physics contact state 是否是一个有价值的 tactile bottleneck？
2. 如果是，长历史 temporal memory 是否能在固定延迟预算下进一步提高闭环控制？

所有阶段都必须把**表示、时序、控制**分开验证，避免同时改多个轴后只报告最终成功率。

---

## Stage 0 — Zero-Network Physics Sanity Check

候选硬件优先使用 FlexiTac 一类二维法向触觉阵列。固定真实硬件、固件、采样配置与记录格式后，采集重复序列：

no-contact → contact → load → redistribution / roll / tilt → release

第一阶段不训练神经网络。

### 必测量

- raw sampling rate 与 frame timestamp；
- transport latency；
- baseline update / threshold / filtering 设置；
- descriptor compute latency；
- P50 / P95 event-to-descriptor delay；
- repeated-trial noise、drift 和 repeatability；
- sustained load 是否被 baseline tracking 错误抹掉；
- descriptor 是否在 contact / release / redistribution 等事件处出现可重复变化。

### Phase-plane visualization

可视化包括但不限于：

- $(P,\Delta P)$；
- $(c_x,\Delta c_x)$、$(c_y,\Delta c_y)$；
- $(A,\sigma_1/\sigma_2)$；
- orientation-confidence trajectory。

这些图只用于形成 hypothesis。结构化轨迹不能替代后续定量比较。

### 继续条件

只有当 descriptor 在重复实验中：

- 数值稳定；
- 不依赖测试期未来信息；
- 事件响应具有可重复性；
- 不主要由 threshold / filter artifact 产生；

才进入 Stage 1。

---

## Stage 1 — Representation Bottleneck

比较三类表示：

1. **Raw**：原始 tactile map；
2. **Physics**：约 11D physically interpretable contact state；
3. **Learned**：从 Raw 学得、维度与 Physics 匹配的 compact latent。

推荐的最小读出：

| 输入 | 读出 / 模型 | 目的 |
| --- | --- | --- |
| Physics 11D | Linear / tiny MLP | 检查物理 bottleneck 的低样本可分性 |
| Raw | Linear | 检查高维 raw 的直接可分性 |
| Raw | Tiny CNN | learned spatial baseline |
| Learned 11D | Tiny encoder + linear head | 与 Physics 同维度比较 inductive bias |

不要预设 Physics 必须击败 Raw。Physics 的潜在价值是 **compactness、data efficiency、OOD robustness、latency 与 interpretability**，不是信息量更大。

### 数据效率

至少报告：

1%、5%、10%、25%、50%、100% training data

下的主要任务指标与方差。

### OOD

按完整 trial / object / acquisition batch 分割，优先测试：

- 未见物体；
- 未见接触位置；
- 未见刚度 / 表面条件；
- 必要时新的 indenter geometry。

不能随机打散帧后声称 OOD。

### 对抗性信息丢失测试

#### A. Moment-matched but locally different

不要人工强行同时匹配 $P,CoP,A,\Sigma$。从真实数据中跨类别搜索：

$$
(i,j)^*=\arg\min_{y_i\neq y_j}\|s_i-s_j\|_2
$$

同时要求 raw tactile map 差异较大。

目标：找到 physics descriptor 近似相同、局部 pressure pattern 不同的真实样本，量化 many-to-one compression 的信息损失。

#### B. Different contact dynamics

至少覆盖两类：

- click / bistable event：$\Delta P$ 大，CoP 变化相对小；
- rolling / tilting / redistribution：总压力变化较缓，CoP / shape 持续变化。

目标不是让 Physics “碾压” Raw，而是判断 compact physical state 是否以更低成本显式暴露控制相关动力学。

### Stage 1 主要指标

- classification / regression task metric；
- sample efficiency curve；
- IID → OOD performance drop；
- representation dimensionality；
- preprocessing / encoder P50、P95 latency；
- CPU / GPU / memory footprint；
- failure examples。

### 继续条件

Physics bottleneck 至少需要在以下一个或多个维度体现清晰价值，且没有不可接受的信息损失：

- data efficiency；
- OOD robustness；
- latency / compute；
- interpretability / diagnostic value。

若 Physics 与 learned compact latent 相比没有明显价值，或丢失的信息正是目标闭环任务所必需，则应停止、修改 descriptor 或转向 learned representation，而不是直接进入 Mamba。

---

## Stage 2 — Temporal Memory

只有 Stage 0 / 1 通过后，才研究：

$$
s_{1:t}\rightarrow h_t.
$$

强候选包括 TacMamba / Mamba 式递推 SSM，但不预设骨干。

建议比较：

| Representation | Temporal model / interface | 所回答的问题 |
| --- | --- | --- |
| scalar total force | streaming memory | TacMamba 风格低维基线 |
| Physics state | streaming memory | 物理结构是否是更好的长历史接口 |
| Learned compact latent | matched temporal model | 收益是否来自 physics inductive bias |
| Raw / rich tactile | temporal encoder | 信息更丰富但成本更高的参照 |

需要固定：

- 历史可见范围；
- hidden-state size / parameter budget；
- sensor update rate；
- training data；
- decision frequency；
- end-to-end latency budget。

不能只比较模型 forward time。

---

## Stage 3 — Closed-Loop Control

Representation 与 memory 都只是假设链条中的中间变量。最终若要声称“改善 tactile-reactive manipulation”，必须进入真实闭环任务。

优先任务应由 Stage 1/2 暴露出的 contact-state需求决定，而不是提前冻结。候选包括：

- sustained contact with disturbance；
- rolling / contact redistribution；
- insertion / snap / click-like transitions；
- 对局部 pattern 有要求的 negative-control task。

至少报告：

- task success / continuous quality；
- physical event → action effect 的端到端 P50 / P95 latency；
- dangerous / excessive-contact events；
- unseen-object / unseen-contact-condition performance；
- failure taxonomy。

---

## 因果性与数据纪律

- 每个传感器记录 sampling timestamp 与 arrival timestamp；
- 动作记录 command timestamp 与实际执行时间；
- 所有在线输入必须在决策时已经到达；
- episode 开始重置 memory state；
- baseline、normalization、threshold 与 calibration 不得使用测试 episode 的未来数据；
- no-contact baseline 若在线更新，接触时必须按协议冻结；
- 训练 / 验证 / 测试按完整轨迹、对象和采集批次划分；
- 仿真 privileged friction / object state 默认只用于锁定模型后的只读诊断。

## 投稿前证据门槛

- Physics representation 的收益和失败边界都有定量证据；
- 同维 learned latent 是认真调优的 baseline；
- temporal model 的收益不能只是更多参数、更多历史或不同刷新率；
- 报告完整端到端 latency，不只报 descriptor 或网络 forward；
- 若主张真实 contact-rich manipulation，必须有真实机器人闭环证据；
- 负结果与停止条件保留，不能追溯性重写假设。

详细 descriptor 与底层预处理定义见 [PHYSICS_BOTTLENECK.md](PHYSICS_BOTTLENECK.md)。
