# 研究主线：从接触状态压缩到长时触觉记忆

状态：研究路线，未形成已验证贡献。更新：2026-10-06。

## 长期问题

机器人如何把高维、快速变化的触觉观测压缩成足够有用的内部接触状态，并在长时间执行过程中利用这些状态及时修正动作？

当前最高优先级候选问题：

> **What is the minimal tactile state that preserves control-relevant spatial contact information over long horizons?**

第一篇不从“设计一个更复杂网络”开始，而先拆成两个可独立否证的问题：

1. **Representation**：低维、物理可解释的 contact state 是否保留了闭环控制需要的主要空间接触信息？
2. **Memory**：长历史是否提供当前 tactile state 无法唯一确定的动态信息，以及如何在固定延迟预算下保存这种历史？

详细的第一阶段定义见 [PHYSICS_BOTTLENECK.md](PHYSICS_BOTTLENECK.md)。

## 当前候选方法接口

候选 testbed 是 FlexiTac 一类二维法向触觉阵列。先把 raw tactile field 通过无参或少量标定的解析统计压缩为约 11D contact descriptor，再与同维 learned latent、raw tactile representation 比较。

当前候选 contact state 包括：

- total contact intensity / force proxy；
- contact centroid；
- normalized active area；
- 二阶矩主尺度；
- 带各向异性置信度的主方向表示；
- total load 与 centroid 的因果动态变化。

这些变量称为 **physically interpretable contact descriptors**。不声称它们包含完整空间信息，也不把 CoP 漂移直接解释为 slip。

## 与核心论文的关系

| 论文 | 已有能力 | 对当前候选问题的作用 |
| --- | --- | --- |
| TacMamba | 单点 1D force 的高频流式长历史压缩，递推更新与低单步延迟 | 提供“长历史 + streaming memory”母体；其空间接触信息不足构成当前表示问题的动机之一 |
| T-Rex | 将 temporal force dynamics 与 spatial deformation 分开编码，并做快慢触觉修正 | 说明空间与时间触觉可以分工；也是 rich tactile representation 的强参照 |
| TacForcing | execution-time tactile feedback 与动作块对齐 | 提醒最终贡献必须进入真实执行闭环，而不能停在离线 representation accuracy |
| FlexiTac / VTAP | 柔性高密度法向触觉阵列及指尖部署 | 提供研究“spatial tactile field → compact contact state”的硬件基础 |

这不是模块拼装路线。当前仍不把“FlexiTac + Mamba”视为既定系统；但允许一个低成本 matched-memory probe 检查表示×记忆交互，避免因为单帧表示结果模糊而误杀真正只在时序条件下出现的价值。

## 当前 baseline 定位

正式实验不使用“找很多论文逐一击败”的叙事，而用三篇代表工作锚定三个维度：

| 维度 | 核心 baseline | 角色 |
| --- | --- | --- |
| Long tactile memory | **TacMamba** | 1D force + streaming SSM memory；用于隔离 spatial contact state 的增量价值 |
| Dense tactile → policy/VLA | **LeFlexiTac** | FlexiTac dense map / tactile tokens 接入 LeRobot policies 与 π0.5；用于回答为什么不直接 token 化高维触觉 |
| Fast tactile control | **RDP** | slow-fast tactile-reactive control；用于比较即时 action reaction 与 persistent state memory |

π0 / π0.5 主要作为 backbone / visual-only system ablation；T-Rex、TacForcing、VTAP、ImplicitRDP 等继续作为 supporting / nearest work，而不是从论文地图中删除。

详细定位、禁止的过强表述与诊断任务见 [BASELINES.md](BASELINES.md)。

## 当前待检验假设

### H1 — Compact physical state

在接触开始、持续载荷、载荷重分布、滚动 / 倾覆和释放等事件中，一个约 10–12 维的 physically structured contact state 可以以远低于 raw tactile map 的表示维度，保留足够的 control-relevant information。

H1 的优势若存在，应主要表现为：

- 更好的低样本 inductive bias；
- 更稳定的 OOD generalization；
- 更小的表示与推理成本；
- 可解释的失败边界。

H1 **不要求** Physics 在信息量上优于 Raw；Raw 包含计算这些 descriptor 所需的原始信息。

### H2 — Coupled spatiotemporal value

候选核心不是“11D 单帧一定更好”，而是 structured spatial state 与 temporal memory 是否存在可测的交互：当总载荷历史不足以区分不同空间接触过程，spatial state 应提供额外可观测性；当当前 spatial state 本身存在歧义时，历史轨迹应提供额外信息。

低成本 Stage-2P 可以在 Stage 1 尚未正式通过时检验这一点，使用 Scalar、Scalar+centroid、Physics11D、Learned11D 接入同一 streaming temporal backbone。Mamba / SSM 是强候选母体，但不能把“用了 Mamba”本身当贡献，也不能预设 11D 必须优于更小的 spatial state。

### H3 — Hybrid extension（后续）

若 H1 在局部纹理、partial slip 或细粒度 pressure pattern 上存在明确失败，可进一步测试：

- fast compact physics state；
- low-rate / event-triggered rich tactile token。

这属于后续扩展，不是第一阶段默认系统。

## 必须排除的替代解释

任何收益都需要排除：

- 仅来自归一化或 threshold 调参；
- Physics baseline 使用了测试期未来统计；
- learned baseline 训练不足；
- 更多参数、更多数据或更长历史；
- hardware sampling rate 不一致；
- descriptor computation 快，但 sensor / transport / actuator 仍主导端到端延迟；
- 离线事件分类更好，却没有闭环控制价值。

## 可能的第一篇贡献形式

只有证据成立后，才可能形成以下贡献：

1. 一个可复现的、sensor-aware 的 compact tactile contact-state representation；
2. 对它保留 / 丢失哪些触觉信息的定量边界；
3. 在长历史与实时闭环条件下，相对 scalar force、learned compact latent 或 rich tactile encoder 的效率 / 泛化收益；
4. 必要的真实机器人闭环任务与失败案例。

如果 Stage 0 / Stage 1 不能支持前两项，且 Stage-2P 也没有显示 representation×memory 交互，则停止或修改路线，而不是继续堆 temporal model 或接入 VLA。若 Stage-2P 只支持更小的 spatial state，则缩减 descriptor，而不是维护“完整 11D 必要”的故事。

## 研究纪律

- 论文事实、作者报告、候选假设和本项目结果分开记录。
- 不把漂亮 phase plot 当作贡献成立。
- 不把 FlexiTac、Mamba、physics descriptor 或神经科学类比本身当 novelty。
- 仿生快慢系统只可作为设计启发，不能替代机器人系统的定量证据。
- 第一篇不同时承诺世界模型、跨本体、通用 VLA、事件触发通信和完整多传感器融合。
